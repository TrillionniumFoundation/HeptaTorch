"""Seeded biased convolution composed on the pinned native HIA wire."""
from pathlib import Path
import importlib.util
import math
import struct
import tempfile

_pool = Path(__file__).resolve().parents[1] / "opt_in_adaptive_avg_pool2d" / "adaptive_pool.py"
_spec = importlib.util.spec_from_file_location("heptatorch_wire", _pool)
wire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wire)
ELF = "5591987579976a9f0df1b5381eccc295b92dab324bbd6c33e2a2071b9b6714b1"
MAX_WORDS = 4096
MAX_SPATIAL = 64


def _integers(values, rank, lo, hi):
    return (type(values) in (tuple,list) and len(values)==rank
            and all(type(v) is int and lo <= v <= hi for v in values))


def geometry(x, weight, stride, padding, dilation, groups):
    """Check fixed-scope extents and derive the expected NCHW shape (no FP work)."""
    if not (_integers(x,4,1,MAX_SPATIAL) and _integers(weight,4,1,MAX_SPATIAL)
            and _integers(stride,2,1,MAX_SPATIAL) and _integers(padding,2,0,MAX_SPATIAL)
            and _integers(dilation,2,1,MAX_SPATIAL) and type(groups) is int
            and 1 <= groups <= MAX_SPATIAL):
        raise ValueError("expected bounded positive NCHW/OIHW convolution geometry")
    if x[1] != weight[1]*groups or weight[0] % groups:
        raise ValueError("incompatible group channels")
    h,w = [(i+2*p-d*(k-1)-1)//s+1 for i,p,d,k,s in
           zip(x[2:],padding,dilation,weight[2:],stride)]
    out=(x[0],weight[0],h,w)
    if min(h,w)<1 or max(map(math.prod,(x,weight,out)))>MAX_WORDS:
        raise ValueError("kernel/output exceeds bounded scope")
    if math.prod(out) * math.prod(weight[1:]) > (1 << 24):
        raise ValueError("bounded native work exceeded")
    return out


def _check_words(data, shape):
    if type(data) is not bytes or len(data) != 4*math.prod(shape):
        raise ValueError("F32 buffer size mismatch")
    # Deliberately restrict the API to the qualified small dyadic grid.
    if any(not (-16 <= v*16 <= 16 and (v*16).is_integer())
           for (v,) in struct.iter_unpack("<f",data)):
        raise ValueError("F32 input must be finite and in the dyadic grid [-1,1] / 16")


def pair(values):
    return values[0]<<32 | values[1]


def factory(s,shape,data,req):
    dims=(*shape,*((0,)*(4-len(shape))))
    config=struct.pack("<8Q",len(shape),int(req),*dims,160,len(data))
    return s.exchange(wire.request((97,0,96),config+data),97)[0]


def observe(s,handle,shape,req):
    rank,n,g,dims,strides,storage,version_id,version,offset,size = wire.unpack_meta(s.call(13,handle)[1])
    count=math.prod(shape)
    wire.need((rank,n,g,dims,strides)==(len(shape),count,int(req),list(shape),wire.contiguous_strides(shape)),
              "unexpected F32 tensor metadata")
    wire.need(storage and version_id and (version,offset,size)==(0,0,4*count),
              "unexpected F32 tensor physical version")
    num,data=s.call(98,handle,0,count)
    wire.need(num==count and len(data)==size,"unexpected F32 payload extent")
    return ((handle,storage,version_id),data)


def conv2d_bias_seeded(binary, x_shape, w_shape, x_data, w_data, bias_data, seed_data,
                        *, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1,
                        requires_grad=(True,True,True), accumulate_seed=None):
    """Return (output_shape, output bytes, three optional pullback bytes, frame count)."""
    y_shape=geometry(x_shape,w_shape,stride,padding,dilation,groups)
    b_shape=(w_shape[0],)
    if (type(requires_grad) not in (list,tuple) or len(requires_grad)!=3
        or any(type(x) is not bool for x in requires_grad) or not any(requires_grad)):
        raise ValueError("require at least one of the three native gradient paths")
    shapes=(x_shape,w_shape,b_shape)
    data=(x_data,w_data,bias_data)
    for raw,shape in zip((*data,seed_data),(*shapes,y_shape)):
        _check_words(raw,shape)
    if accumulate_seed is not None:
        _check_words(accumulate_seed,y_shape)
    with tempfile.TemporaryDirectory(prefix="hia-conv-bias-ad-") as tmp:
        native=wire.stage_executable(Path(tmp),wire.pinned_bytes(binary,ELF,wire.ELF_CAP))
        with wire.Session(native) as s:
            hx,hw,hb=handles=[factory(s,shape,raw,g) for shape,raw,g in zip(shapes,data,requires_grad)]
            before=[observe(s,h,shape,g) for h,shape,g in zip(handles,shapes,requires_grad)]
            wire.need(tuple(x[1] for x in before)==data,"native input load changed bytes")
            raw_conv,_=s.call(22,hx,hw,pair(stride),pair(padding),pair(dilation),groups)
            bias_view,_=s.call(17,hb,4,1,b_shape[0],1,1)
            y,_=s.call(2,1,raw_conv,bias_view)
            observed=observe(s,y,y_shape,True)
            wire.need(all(len(set(v))==4 for v in zip(*(x[0] for x in (*before,observed)))),
                      "result aliases an input owner/version")
            # Drop both public intermediate owners before seeded backward. This
            # exercises the native graph's real private retention semantics.
            s.call(41,raw_conv)
            s.call(41,bias_view)
            for seed_bytes in (seed_data,) if accumulate_seed is None else (seed_data,accumulate_seed):
                seed=factory(s,y_shape,seed_bytes,False)
                seed_before=observe(s,seed,y_shape,False)
                wire.need(seed_before[1]==seed_bytes and
                          all(len(set(v))==5 for v in zip(*(x[0] for x in (*before,observed,seed_before)))),
                          "seed custody or owner isolation mismatch")
                s.call(29,y,seed)
                wire.need(observe(s,seed,y_shape,False)==seed_before,"backward mutated seed")
                s.call(41,seed)
            gradients=[]
            for h,shape,want in zip(handles,shapes,requires_grad):
                if want:
                    count,raw=s.call(11,h)
                    wire.need(count==math.prod(shape) and len(raw)==4*count,"gradient extent mismatch")
                    gradients.append(raw)
                else:
                    gradients.append(None)
            for h,shape,g,old in zip(handles,shapes,requires_grad,before):
                wire.need(observe(s,h,shape,g)==old,"operation mutated an input")
                s.call(41,h)
            wire.need(observe(s,y,y_shape,True)==observed,"output changed after releasing inputs")
            s.call(41,y)
            wire.need(s.call(53)[0]==1,"native owner invariant failed")
            return y_shape,observed[1],tuple(gradients),s.frames
