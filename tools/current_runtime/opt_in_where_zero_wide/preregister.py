"""Preregister bit predictions for true rank-0 value tensors; no floating evaluation."""
import itertools,json,math

W={0:4,1:1,2:8,4:8,5:2,6:2}
F64=[0,1<<63,1,(1<<63)|1,0x7ff0000000000000,0xfff0000000000000,
     0x7ff8123456789abc,0xfff8123456789abc,0x7ff0123456789abc,0xfff0123456789abc,
     0x3ff0000000000000,0xbff0000000000000]
P={
 0:[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f800000,0xff800000,
    0x7fc12345,0xffc54321,0x7f812345,0xff812345,0x7f7fffff,0xff7fffff],
 5:[0,0x8000,1,0x8001,0x03ff,0x0400,0x7c00,0xfc00,0x7e53,0xfe59,0x7c35,0xfc39,0x7bff,0xfbff],
 6:[0,0x8000,1,0x8001,0x007f,0x0080,0x7f80,0xff80,0x7fca,0xffc9,0x7f85,0xff89,0x7f7f,0xff7f],
 2:[0,1,-1,-2**63,-2**63+1,2**63-1]+
   [sg*((1<<e)+v) for e,v in ((53,-1),(53,0),(53,1),(53,2),(53,3),(53,4),(62,511),(62,512),(62,513),(62,1535),(62,1536),(62,1537)) for sg in (-1,1)],
}

def bits_f64(t,w):
    if t==4:return w
    if t==2:
        assert 0<=w<2**64
        n=w if w<2**63 else w-2**64
        if not n:return 0
        a=abs(n)
        spacing=1<<max(0,a.bit_length()-53)
        q,r=divmod(a,spacing)
        if min(((r,q&1,q),(spacing-r,(q+1)&1,q+1)))[2]!=q:q+=1
        chosen=q*spacing
        exp=chosen.bit_length()-1
        sig=(chosen<<52)>>exp
        return (int(n<0)<<63)|((exp+1023)<<52)|(sig&((1<<52)-1))
    eb,mb={0:(8,23),5:(5,10),6:(8,7)}[t]
    sign=(w>>(eb+mb))<<63
    exp=(w>>mb)&((1<<eb)-1)
    frac=w&((1<<mb)-1)
    if exp==(1<<eb)-1:
        return sign|(0x7ff<<52)|((frac<<(52-mb))|(1<<51) if frac else 0)
    if exp==0 and frac==0:return sign
    bias=(1<<(eb-1))-1
    if exp==0:
        e=1-bias-mb+frac.bit_length()-1
        sig=frac<<(53-frac.bit_length())
    else:
        e=exp-bias
        sig=(frac+(1<<mb))<<(52-mb)
    return sign|((e+1023)<<52)|(sig&((1<<52)-1))

def array(t,shape,vals):
    assert len(vals)==math.prod(shape)
    sz=W[t]
    vs=[a&((1<<(sz*8))-1) for a in vals]
    return dict(dtype=t,shape=shape,words=vs,byte_hex=b''.join(x.to_bytes(sz,'little') for x in vs).hex())

def shape_join(*shapes):
    rank=max(map(len,shapes))
    g=[set(c)-{1} for c in zip(*[[1]*(rank-len(s))+s for s in shapes])]
    assert all(len(c)<2 for c in g)
    return [next(iter(c),1) for c in g]

def tensor_index(a,c):
    shape=a['shape']; ind=0
    for size,col in zip(shape,c[len(c)-len(shape):]):
        ind=ind*size+(0 if size==1 else col)
    return a['words'][ind]

def fixture(label,m,x,y,order='inputs-first'):
    shapes=[a['shape'] for a in (m,x,y)]
    shape=shape_join(*shapes)
    vals=[]
    for coord in itertools.product(*(range(d) for d in shape)):
        src=x if tensor_index(m,coord) else y
        vals.append(bits_f64(src['dtype'],tensor_index(src,coord)))
    return dict(label=label,mask=m,x=x,y=y,result=array(4,shape,vals),lifetime_order=order)

def make():
    assert bits_f64(5,0x7c35)==0x7ff8d40000000000
    assert bits_f64(5,0xfc39)==0xfff8e40000000000
    assert bits_f64(6,0x7f85)==0x7ff8a00000000000
    assert bits_f64(6,0xff89)==0xfff9200000000000
    assert bits_f64(2,2**53+1)==0x4340000000000000
    assert bits_f64(2,2**53+3)==0x4340000000000002
    rows=[]; guards=[]
    for tag,vals in P.items():
        for rev in (False,True):
            floating=array(4,[1,len(F64)],F64)
            for j,v in enumerate(vals):
                s=array(tag,[],[v]);x,y=(floating,s) if rev else (s,floating)
                m=array(1,[2,1,1],[not rev, rev])
                rows.append(fixture(f'mixed-mask-t{tag}-r{int(rev)}-word{j}',m,x,y,
                    'result-first' if rev else 'inputs-first'))
            s=array(tag,[],[vals[-1]])
            x,y=(floating,s) if rev else (s,floating)
            for take_scalar in (False,True):
                m=array(1,[],[take_scalar != rev])
                rows.append(fixture(f'scalar-mask-t{tag}-r{int(rev)}-scalar{int(take_scalar)}',m,x,y,
                    'result-first' if rev else 'inputs-first'))
            for sh,ms in (([2,0],[1,1]),([1,4],[0,1,1])):
                m=array(1,ms,[0]*math.prod(ms)); f=array(4,sh,F64[:math.prod(sh)])
                x,y=(f,s) if rev else (s,f)
                rows.append(fixture(f'empty-t{tag}-r{int(rev)}-sh{sh}-mask{ms}',m,x,y))
            for negative,nx,ny in [('reverse-role',array(tag,[1],[vals[-1]]),array(4,[],[F64[8]])),
                                    ('both-rank0',array(tag,[],[vals[-1]]),array(4,[],[F64[8]]))]:
                if rev:nx,ny=ny,nx
                guards.append(dict(label=f'{negative}-t{tag}-r{int(rev)}',mask=array(1,[],[1]),x=nx,y=ny,policy_expected_status=7))
    return dict(scope='predictions only, no official PyTorch or HIA execution',
                positive=rows,policy_negative=guards,
                half_nan_scope='direct 16-to-F64 expected signed payload alignment and signaling quiet; do not route via PyTorch float32 FBGEMM tail',
                scope_exclusions=['views','requires_grad','private dtype3','Python scalar overload','zero+zero mixed','dimensional non-F64 with scalar F64'])
if __name__=='__main__':
    print(json.dumps(make(),separators=(',',':'),sort_keys=True))
