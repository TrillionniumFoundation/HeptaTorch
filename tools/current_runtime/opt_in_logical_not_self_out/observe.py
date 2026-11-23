import sys, hashlib, struct, json, zlib
from pathlib import Path
import argparse, tempfile, importlib
pins={
    'test_runtime_candidate.py':'40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226',
    'qualify_runtime_torch.py':'b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd',
    'test_component.py':'97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c',
}
images={
    'baseline':'568946d257cd81d00f4fb8b5a6fdd966abf1bb8de6333b59b098be08a1afc62d',
    'candidate':'6c68e0469a2d96b071300d1bcb25db408e3177e83e35617566bb0d76b18d6c08',
}
p=argparse.ArgumentParser(description='Verify exact opt-in native images; use locked CPU Torch for candidate')
p.add_argument('mode',choices=images)
for arg in ('binary','output','protocol','helper-dir','oracle-lock','install-report'):
    p.add_argument('--'+arg,type=Path,required=arg not in ('oracle-lock','install-report'))
a=p.parse_args()
assert __debug__ and not sys.flags.optimize
mode,image,root=a.mode,a.binary.resolve(strict=True),a.output
pin=images[mode]
with tempfile.TemporaryDirectory(prefix='logical-out-helpers-') as tmp:
    temp=Path(tmp)
    for name,expected in pins.items():
        src=a.helper_dir/name
        data=src.read_bytes()
        assert hashlib.sha256(data).hexdigest()==expected,name
        (temp/name).write_bytes(data)
    sys.path.insert(0,str(temp))
    try:
        from test_runtime_candidate import factory, load_wire, digest
        from qualify_runtime_torch import load_oracle
        from test_component import WIDTHS, known_words
    finally:
        sys.path.remove(str(temp))
assert digest(image)==pin
wire=load_wire(a.protocol)
if mode=='candidate':
    assert a.oracle_lock is not None and a.install_report is not None
    torch,oracle=load_oracle(a.oracle_lock,a.install_report)
s=wire.Session(image,directory=root)
rows=[]
def raw(body):
    a=s.exchange(body)
    return wire.parse(a)
def run(*cmds):
    status,out=raw(wire.packet(cmds))
    assert status==0 and [r['opcode'] for r in out]==[c[0] for c in cmds],(status,cmds)
    return out if len(out)!=1 else out[0]
def create(dt,shape,v):
    st,r=raw(b''.join(factory(dt,shape,v)))
    assert st==0 and [q['opcode'] for q in r]==[97],(st,dt,shape)
    return r[0]['value']
def inspect(x):
    m,d=run((13,x),(96,x,0)); meta=wire.meta(m); n=meta['numel']
    dat=run((98,x,0,n))
    assert dat['value']==n
    return meta,d['payload'],dat['payload']
def refused(label,*cmds,exp=None):
    pre=s.owner()
    st,r=raw(wire.packet(cmds))
    post=s.owner()
    assert st and not r and (exp is None or st==exp),(label,st,r)
    assert pre==post,(label,hashlib.sha256(pre['bytes']).hexdigest(),hashlib.sha256(post['bytes']).hexdigest())
    for suffix,witness in [('pre',pre),('post',post)]:
        (root/f'{s.frames:03d}-{suffix}.owner.zlib').write_bytes(zlib.compress(witness['bytes'],6))
    rows.append(dict(case=label,frame=s.frames,status=st,length=pre['length'],sha=hashlib.sha256(pre['bytes']).hexdigest()))
def positive(label,dt,shape,words):
    n=len(words); buf=bytearray().join(x.to_bytes(WIDTHS[dt],'little') for x in words)
    td={0:torch.float32,1:torch.bool,2:torch.int64,3:torch.float64,4:torch.float64,5:torch.float16,6:torch.bfloat16}
    inp=torch.frombuffer(buf,dtype=td[dt]).reshape(shape) if n else torch.empty(shape,dtype=td[dt])
    expected=torch.empty(shape,dtype=torch.bool); expected.fill_(False)
    v=expected._version; data_ptr=expected.data_ptr(); layout=expected.stride()
    assert torch.logical_not(inp,out=expected) is expected
    assert expected._version==v+1 and expected.data_ptr()==data_ptr and expected.stride()==layout
    vals=bytes(expected.reshape(-1).tolist())
    assert torch.logical_not(inp,out=expected) is expected and expected._version==v+2
    xx=create(dt,shape,words); yy=create(1,shape,[0]*n)
    im=inspect(xx); om=inspect(yy)
    assert im[0]['storage_id']!=om[0]['storage_id'] and im[0]['version_id']!=om[0]['version_id']
    results=run((128,xx,yy),(128,xx,wire.reference(1)),(98,wire.reference(2),0,n))
    write_frame=s.frames
    assert results[0]['value']==results[1]['value']==yy and results[0]['payload']==results[1]['payload']==b''
    assert results[2]['payload']==vals
    after=inspect(yy)
    wanted=dict(om[0]); wanted['version']+=2
    assert after==(wanted,om[1],vals) and inspect(xx)==im,(label,after,wanted)
    rows.append(dict(case=label,write_frame=write_frame,destination=yy,numel=n,output_sha=hashlib.sha256(vals).hexdigest(),version_delta=2))
def self_positive(label,shape,values):
    t=torch.tensor(values,dtype=torch.bool).reshape(shape)
    ptr=t.data_ptr(); before=t._version; first=torch.logical_not(t,out=t)
    assert first is t and t._version==before+1 and t.data_ptr()==ptr
    inverted=bytes(t.reshape(-1).tolist())
    second=torch.logical_not(t,out=t)
    assert second is t and t._version==before+2 and t.data_ptr()==ptr
    assert bytes(t.reshape(-1).tolist())==bytes(values)
    h=create(1,shape,values)
    original=inspect(h); n=len(values)
    refused(label+'-rollback-after-in-place-write',(128,h,h),(128,0,h), exp=1)
    if n:
        results=run((128,h,h),(98,wire.reference(1),0,n),(128,wire.reference(1),wire.reference(1)),(98,wire.reference(3),0,n))
        assert results[0]['value']==results[2]['value']==h and not results[0]['payload'] and not results[2]['payload']
        assert results[1]['payload']==inverted and results[3]['payload']==bytes(values)
        version_pair=s.frames
    else:
        assert run((128,h,h))['value']==h
        meta=inspect(h); want=dict(original[0]);want['version']+=1
        assert meta==(want,original[1],b'')
        assert run((128,h,h))['value']==h
        version_pair=s.frames
    final=inspect(h); want=dict(original[0]);want['version']+=2
    assert final==(want,original[1],bytes(values))
    rows.append(dict(case=label,self_alias=True,destination=h,paired_frame=version_pair,shape=shape,numel=n,expected_inversion=inverted.hex(),expected_restoration=bytes(values).hex(),version_before=original[0]['version'],version_after=final[0]['version']))
try:
    x=create(0,(3,),[0,0x80000000,0x3f800000]); y=create(1,(3,),[0,0,1])
    for j in range(2,8):
        c=[127,x]+[0]*6;c[j]=y
        refused('old-reserved-'+str(j),c,exp=7)
    good=run((127,x))['value']
    assert good not in (x,y) and inspect(good)[2]==bytes([1,1,0])
    if mode=='baseline':
        refused('old-no-exact-bool-self', (128,y,y),exp=7)
        assert run((128,x,y))['value']==y
    else:
        for j in range(3,8):
            c=[128,x,y]+[0]*5;c[j]=1
            refused('new-reserved-'+str(j),c,exp=7)
        refused('same-node-non-bool',(128,x,x),exp=7)
        refused('wrong-output-dtype',(128,y,x),exp=7)
        z=create(1,(1,3),[0,0,1]); refused('wrong-shape-equal-count',(128,x,z),exp=7)
        refused('bad-source',(128,0,y),exp=1)
        refused('bad-destination',(128,x,0),exp=1)
        refused('forward-reference',(128,x,wire.reference(2)),(13,y),exp=1)
        refused('valid-write-then-invalid',(128,x,y),(128,0,y),exp=1)
        assert run((128,x,y))['value']==y and inspect(y)[2]==bytes([1,1,0])
        for dt in sorted(set(WIDTHS)-{3}):
            vals=known_words(dt)
            positive('typed-'+str(dt),dt,(len(vals),),vals)
        for shape in [(),(1,),(2,2),(1,2,2),(1,2,1,2),(0,),(2,0,3)]:
            import math
            n=math.prod(shape)
            positive('shape-'+str(shape),0,shape,([0,0x80000001,0x7f800000,0x7fc00001]*max(1,n))[:n])
        for shape,values in [((),[1]),((4,),[1,0,0,1]),((2,2),[0,1,0,1]),((1,2,1,2),[0,0,1,1]),((0,),[]),((2,0,3),[])]:
            self_positive('exact-self-'+str(shape),shape,values)
    summary=dict(mode=mode,image=pin,rows=rows,frames=s.frames)
    if mode!='baseline': summary['oracle']=oracle
    (root/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
finally:
    s.finish()
