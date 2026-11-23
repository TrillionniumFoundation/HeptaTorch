"""Actual native candidate/parent, no Torch imports."""
import hashlib,json,struct,sys,zlib
from pathlib import Path
H,WIRE,E,F,O,MODE=map(Path,sys.argv[1:]);mode=MODE.name
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
assert sha((H/'test_component.py').read_bytes())=='97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c'
pin={'baseline':'77e2ee440e0cb2f568d350a19e2bedd3138ffb407767f4d224f4567ea7fd393b',
     'candidate':'322b419c867210439f49313ee8d43ff1c220c4f15ae4d4ee673c4df78434683b'}[mode]
assert sha(E.read_bytes())==pin and sha(F.read_bytes())=='10648ec9e86ee7b5cd7763b1ee088f6299dae2a796ebc30b991e39fee3955c51'
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
W=load_wire(WIRE);s=W.Session(E,directory=O);(O/'owners').mkdir()
assert 'torch' not in sys.modules
suite=json.loads(F.read_bytes());ledger=[];witness=set()
def run(*cs):
    st,res=W.parse(s.exchange(W.packet(cs)))
    assert st==0 and [v['opcode'] for v in res]==[v[0] for v in cs],(cs,st)
    return res
def make(t):
    st,res=W.parse(s.exchange(b''.join(factory(t['dtype'],t['shape'],t['words']))))
    assert st==0 and len(res)==1 and res[0]['opcode']==97,(t['dtype'],t['shape'],st)
    return res[0]['value']
def snap():
    a=s.owner();body=a['bytes'];dig=sha(body);p=O/'owners'/(dig+'.zlib')
    if not p.exists():p.write_bytes(zlib.compress(body,6))
    else:assert zlib.decompress(p.read_bytes())==body
    return dict(sha256=dig,length=len(body),ptr=a['pointer'])
def deny(status,*cs):
    b=snap();req=W.packet(cs);actual,data=W.parse(s.exchange(req));f=s.frames;a=snap()
    assert actual==status and data==[] and a==b,(cs,status,actual,a,b)
    return dict(pre=b,post=a,frame=f,status=status,request_sha256=sha(req))
def read(h,n,payload):
    a,b=run((13,h),(98,h,0,n));assert b['value']==n and b['payload'].hex()==payload
    return s.frames,W.meta(a)
def drop(*hs):
    for h in hs:assert run((41,h))[0]['value']==1
def shapeop(op,h,shape):
    inp=W.Payload(struct.pack('<5Q',len(shape),*shape,*[0]*(4-len(shape))),8)
    a,=run((op,h,inp,40));return a['value'],s.frames
def strides(shape):
    import math
    return [math.prod(max(1,x) for x in shape[i+1:]) for i in range(len(shape))]
try:
 for i,t in enumerate(suite):
    attrs=t['operands'];h=[make(j) for j in attrs];w=(129,*h); classification=t['classification']
    if mode=='baseline' or classification!='canonical':
        ledger.append(dict(label=t['label'],kind='refusal',handles=h,**deny(7,w)));drop(*h);continue
    pre=snap(); out=run(w)[0]['value'];op=s.frames;post=snap()
    ro,meta=read(out,0,'')
    rank=len(t['shape'])
    assert list(meta['shape'][:rank])==t['shape'] and list(meta['strides'][:rank])==t['strides']
    assert meta['rank']==rank and meta['version']==meta['numel']==meta['requires_grad']==meta['storage_bytes']==meta['storage_offset']==0
    wk=run((42,out))[0]['value'];weak=s.frames
    extra={}
    if rank not in witness:
        witness.add(rank); same=t['shape']; changed=[0,1] if rank==1 else [0]
        def shape_pkt(op,he,shape):
            return (op,he,W.Payload(struct.pack('<5Q',len(shape),*shape,*[0]*(4-len(shape))),8),40)
        unsupported_112=deny(7,shape_pkt(112,out,same)); unsupported_113=deny(7,shape_pkt(113,out,changed))
        inp=W.Payload(struct.pack('<10Q',rank,*same,*[0]*(4-rank),*t['strides'],*[0]*(4-rank),0),8)
        a=run((111,out,inp,80))[0]['value'];ar=s.frames
        ra,am=read(a,0,'')
        assert (meta['storage_id'],meta['version_id'])==(am['storage_id'],am['version_id'])
        assert list(am['strides'][:rank])==t['strides']
        repeat=run(w)[0]['value'];rpt=s.frames
        rr,rmeta=read(repeat,0,'')
        assert rmeta['storage_id'] not in (meta['storage_id'],am['storage_id'])
        drop(repeat)
        oob_read=deny(2,(98,out,0,1));oob_write=deny(7,(12,out,0,0x3f800000))
        compound=deny(4,w,(65500,))
        drop(out); survivor,_=read(a,0,'');drop(a)
        extra=dict(alias=a,alias_call=ar,alias_read=ra,alias_survives_root=survivor,repeat=repeat,repeat_call=rpt,repeat_read=rr,
                   checks=[oob_read,oob_write,compound,unsupported_112,unsupported_113])
    elif i%2==0:
        drop(*h);surv,_=read(out,0,'');extra=dict(result_survives_inputs=surv);drop(out);hdrop=False
    else:
        drop(out);z=attrs[1];surv,_=read(h[1],len(z['words']),z['byte_hex']);extra=dict(inputs_survive_result=surv)
    if 'result_survives_inputs' not in extra:drop(*h)
    exp=deny(6,(43,wk));run((44,wk))
    ledger.append(dict(label=t['label'],kind='positive',handles=h,out=out,pre=pre,post=post,op=op,read=ro,
                       weak_frame=weak,weak=wk,expired=exp,**extra))
except BaseException as e:
 (O/'error.json').write_text(json.dumps(dict(error=repr(e))));raise
finally:
 assert 'torch' not in sys.modules
 (O/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,no_torch=True,frames=s.frames,rows=ledger),sort_keys=True,indent=2)+'\n')
 s.finish()
