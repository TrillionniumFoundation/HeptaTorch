import sys, hashlib, struct, json, zlib, os
from pathlib import Path
import argparse, tempfile, importlib
pins={
    'test_runtime_candidate.py':'40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226',
    'qualify_runtime_torch.py':'b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd',
    'test_component.py':'97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c',
}
images={
    'baseline':'2e9aad8249051b380fab53fd7449d46913cb5304c01f74da2506fdf965336d7a',
    'candidate':'b8558c915c0c1b475bfcbd46e654dda64930a30e18e835d3504e8b3c95851bef',
}
p=argparse.ArgumentParser(description='Verify exact opt-in native images; use locked CPU Torch for candidate')
p.add_argument('mode',choices=[*images,'fixture'])
for arg in ('binary','output','protocol','helper-dir','oracle-lock','install-report'):
    p.add_argument('--'+arg,type=Path,required=arg not in ('oracle-lock','install-report','binary'))
p.add_argument('--fixture',type=Path)
p.add_argument('--fixture-sha256')
a=p.parse_args()
assert __debug__ and not sys.flags.optimize
mode,root=a.mode,a.output
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
def case_identity(name,parameters):
    return hashlib.sha256(json.dumps([name,parameters],separators=(',',':')).encode()).hexdigest()
def lookup(label,name,params):
    assert label not in used
    item=fixture['rows'][label]
    assert item['input_sha256']==case_identity(name,params)
    used.add(label)
    return tuple(bytes.fromhex(x) for x in item['payload'])
def fixture_positive(label,dt,shape,words):
    n=len(words); buf=bytearray().join(x.to_bytes(WIDTHS[dt],'little') for x in words)
    td={0:torch.float32,1:torch.bool,2:torch.int64,3:torch.float64,4:torch.float64,5:torch.float16,6:torch.bfloat16}
    inp=torch.frombuffer(buf,dtype=td[dt]).reshape(shape) if n else torch.empty(shape,dtype=td[dt])
    expected=torch.empty(shape,dtype=torch.bool); expected.fill_(False)
    v=expected._version; data_ptr=expected.data_ptr(); layout=expected.stride()
    assert torch.logical_not(inp,out=expected) is expected
    assert expected._version==v+1 and expected.data_ptr()==data_ptr and expected.stride()==layout
    vals=bytes(expected.reshape(-1).tolist())
    assert torch.logical_not(inp,out=expected) is expected and expected._version==v+2
    assert label not in results
    results[label]=dict(input_sha256=case_identity('positive',(dt,shape,words,)),payload=[item.hex() for item in (vals,)])
def fixture_self_positive(label,shape,values):
    t=torch.tensor(values,dtype=torch.bool).reshape(shape)
    ptr=t.data_ptr(); before=t._version; first=torch.logical_not(t,out=t)
    assert first is t and t._version==before+1 and t.data_ptr()==ptr
    inverted=bytes(t.reshape(-1).tolist())
    second=torch.logical_not(t,out=t)
    assert second is t and t._version==before+2 and t.data_ptr()==ptr
    assert bytes(t.reshape(-1).tolist())==bytes(values)
    assert label not in results
    results[label]=dict(input_sha256=case_identity('self_positive',(shape,values,)),payload=[item.hex() for item in (inverted,)])
def fixture_numeric_positive(label,dt,ot,shape,values):
    tags={0:torch.float32,1:torch.bool,2:torch.int64,4:torch.float64,5:torch.float16,6:torch.bfloat16}
    n=len(values)
    buf=bytearray().join(v.to_bytes(WIDTHS[dt],'little') for v in values)
    ti=torch.frombuffer(buf,dtype=tags[dt]).reshape(shape) if n else torch.empty(shape,dtype=tags[dt])
    o=torch.full(shape,-2,dtype=tags[ot]); orig_ptr=o.data_ptr(); ver=o._version
    for step in (1,2):
        assert torch.logical_not(ti,out=o) is o and o._version==ver+step and o.data_ptr()==orig_ptr
    want=bytes(o.reshape(-1).view(torch.uint8).tolist())
    assert label not in results
    results[label]=dict(input_sha256=case_identity('numeric_positive',(dt,ot,shape,values,)),payload=[item.hex() for item in (want,)])
def fixture_numeric_self(label,dt,shape,values):
    tags={0:torch.float32,2:torch.int64,4:torch.float64,5:torch.float16,6:torch.bfloat16}
    w=WIDTHS[dt]; n=len(values)
    init=b''.join(v.to_bytes(w,'little') for v in values)
    y=torch.frombuffer(bytearray(init),dtype=tags[dt]).reshape(shape) if n else torch.empty(shape,dtype=tags[dt])
    dp=y.data_ptr();v=y._version
    assert torch.logical_not(y,out=y) is y and y._version==v+1 and y.data_ptr()==dp
    want1=bytes(y.reshape(-1).view(torch.uint8).tolist())
    assert torch.logical_not(y,out=y) is y and y._version==v+2 and y.data_ptr()==dp
    want2=bytes(y.reshape(-1).view(torch.uint8).tolist())
    assert label not in results
    results[label]=dict(input_sha256=case_identity('numeric_self',(dt,shape,values,)),payload=[item.hex() for item in (want1,want2)])
def matrix(positive,self_positive,numeric_positive,numeric_self):
    for dt in sorted(set(WIDTHS)-{3}):
        vals=known_words(dt)
        positive('typed-'+str(dt),dt,(len(vals),),vals)
    for shape in [(),(1,),(2,2),(1,2,2),(1,2,1,2),(0,),(2,0,3)]:
        import math
        n=math.prod(shape)
        positive('shape-'+str(shape),0,shape,([0,0x80000001,0x7f800000,0x7fc00001]*max(1,n))[:n])
    for shape,values in [((),[1]),((4,),[1,0,0,1]),((2,2),[0,1,0,1]),((1,2,1,2),[0,0,1,1]),((0,),[]),((2,0,3),[])]:
        self_positive('exact-self-'+str(shape),shape,values)
    for src in [0,1,2,4,5,6]:
        values = ([0,1,0,1] if src==1 else ([0,0x8000,1,0x7c00,0x7e00,0xffff] if src==5 else [0,0x8000,1,0x7f80,0x7fc0,0xffff] if src==6 else known_words(src)[:24]))
        for dst in [0,2,4,5,6]:
            numeric_positive(f'cast-{src}-{dst}',src,dst,(len(values),),values)
    for sh, words in [((),[0]),((2,0,3),[]),((1,2,1,2),[0,0x80000000,1,0x7fc00000])]:
        for dst in [0,2,4,5,6]:
            numeric_positive(f'layout-{sh}-{dst}',0,dst,sh,words)
    raw={0:[0,0x80000000,1,0x7f800000,0x7fc00001,0xc0000000],2:[0,1,2,(1<<63),(1<<63)-1,(1<<64)-2],4:[0,1<<63,1,0x7ff0000000000000,0x7ff8000000000001,0xc000000000000000],5:[0,0x8000,1,0x7c00,0x7e01,0xc000],6:[0,0x8000,1,0x7f80,0x7fc1,0xc000]}
    for dt,words in raw.items():
        numeric_self(f'numeric-self-{dt}-vector',dt,(len(words),),words)
        numeric_self(f'numeric-self-{dt}-scalar',dt,(),[0])
        numeric_self(f'numeric-self-{dt}-empty',dt,(2,0,3),[])

if mode=='fixture':
    assert a.oracle_lock is not None and a.install_report is not None
    torch,oracle=load_oracle(a.oracle_lock,a.install_report)
    results={}
    matrix(fixture_positive,fixture_self_positive,fixture_numeric_positive,fixture_numeric_self)
    assert len(results)==79
    root.write_text(json.dumps(dict(oracle=oracle,rows=results),sort_keys=True,separators=(',',':'))+'\n')
    sys.exit(0)
assert a.binary is not None
image=a.binary.resolve(strict=True)
pin=images[mode]
assert digest(image)==pin
wire=load_wire(a.protocol)
if mode=='candidate':
    assert a.fixture is not None and a.fixture_sha256 is not None
    raw_fixture=a.fixture.read_bytes()
    assert hashlib.sha256(raw_fixture).hexdigest()==a.fixture_sha256
    fixture=json.loads(raw_fixture)
    assert len(fixture['rows'])==79
    oracle=fixture['oracle']; used=set()
    assert 'torch' not in sys.modules
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
    pre_path=root/f'{s.frames:03d}-pre.owner.zlib'
    post_path=root/f'{s.frames:03d}-post.owner.zlib'
    import stat
    raw_digest=hashlib.sha256(pre['bytes']).hexdigest()
    cas_dir=root/'unique-owner';cas_dir.mkdir(exist_ok=True)
    canonical=cas_dir/(raw_digest+'.owner.zlib')
    if canonical.exists():
        assert stat.S_ISREG(canonical.lstat().st_mode)
        assert zlib.decompress(canonical.read_bytes())==pre['bytes']
    else:
        with canonical.open('xb') as stream:stream.write(zlib.compress(pre['bytes'],6))
    os.link(canonical,pre_path)
    os.link(canonical,post_path)
    rows.append(dict(case=label,frame=s.frames,status=st,length=pre['length'],sha=hashlib.sha256(pre['bytes']).hexdigest()))
def positive(label,dt,shape,words):
    n=len(words)
    vals,=lookup(label,'positive',(dt,shape,words,))
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
    inverted,=lookup(label,'self_positive',(shape,values,))
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
def numeric_positive(label,dt,ot,shape,values):
    n=len(values)
    want,=lookup(label,'numeric_positive',(dt,ot,shape,values,))
    raw_one={0:0x3f800000,2:1,4:0x3ff0000000000000,5:0x3c00,6:0x3f80}
    sentinels={0:0xc0000000,2:(1<<64)-2,4:0xc000000000000000,5:0xc000,6:0xc000}
    assert len(want)==n*WIDTHS[ot]
    xx=create(dt,shape,values); yy=create(ot,shape,[sentinels[ot]]*n)
    im=inspect(xx); om=inspect(yy)
    assert im[0]['storage_id']!=om[0]['storage_id'] and im[0]['version_id']!=om[0]['version_id']
    refused(label+'-later-invalid',(128,xx,yy),(128,0,yy),exp=1)
    results=run((128,xx,yy),(128,xx,wire.reference(1)),(98,wire.reference(2),0,n))
    write_frame=s.frames
    assert results[0]['value']==results[1]['value']==yy and results[0]['payload']==results[1]['payload']==b''
    assert results[2]['payload']==want
    after=inspect(yy); new=dict(om[0]); new['version']+=2
    assert after==(new,om[1],want) and inspect(xx)==im
    assert all(int.from_bytes(want[i:i+WIDTHS[ot]],'little') in (0,raw_one[ot]) for i in range(0,len(want),WIDTHS[ot]))
    rows.append(dict(case=label,numeric_out=True,write_frame=write_frame,destination=yy,numel=n,output_sha=hashlib.sha256(want).hexdigest(),output_tag=ot,output_width=WIDTHS[ot],raw_one=raw_one[ot],version_delta=2))
def numeric_self(label,dt,shape,values):
    w=WIDTHS[dt]; n=len(values)
    init=b''.join(v.to_bytes(w,'little') for v in values)
    want1,want2=lookup(label,'numeric_self',(dt,shape,values,))
    h=create(dt,shape,values)
    old=inspect(h); assert old[2]==init
    refused(label+'-after-valid-self-write',(128,h,h),(128,0,h),exp=1)
    if n:
        arr=run((128,h,h),(98,wire.reference(1),0,n),(128,wire.reference(1),wire.reference(1)),(98,wire.reference(3),0,n))
        assert arr[0]['value']==arr[2]['value']==h and arr[0]['payload']==arr[2]['payload']==b''
        assert arr[1]['payload']==want1 and arr[3]['payload']==want2
    else:
        assert run((128,h,h))['value']==h
        mid=inspect(h); want=dict(old[0]); want['version']+=1
        assert mid==(want,old[1],b'')
        assert run((128,h,h))['value']==h
    frame=s.frames
    actual=inspect(h); want=dict(old[0]);want['version']+=2
    assert actual==(want,old[1],want2)
    rows.append(dict(case=label,numeric_self=True,dtype=dt,width=w,destination=h,paired_frame=frame,shape=shape,numel=n,raw_initial=init.hex(),raw_first=want1.hex(),raw_second=want2.hex(),version_before=old[0]['version'],version_after=actual[0]['version']))
try:
    x=create(0,(3,),[0,0x80000000,0x3f800000]); y=create(1,(3,),[0,0,1])
    for j in range(2,8):
        c=[127,x]+[0]*6;c[j]=y
        refused('old-reserved-'+str(j),c,exp=7)
    good=run((127,x))['value']
    assert good not in (x,y) and inspect(good)[2]==bytes([1,1,0])
    if mode=='baseline':
        for dt in [0,2,4,5,6]:
            token=create(dt,(3,),[0,1,2])
            refused('old-numeric-self-'+str(dt),(128,token,token),exp=7)
        assert run((128,y,x))['value']==x
        assert run((128,y,y))['value']==y
        assert run((128,x,y))['value']==y
    else:
        for j in range(3,8):
            c=[128,x,y]+[0]*5;c[j]=1
            refused('new-reserved-'+str(j),c,exp=7)
        z=create(1,(1,3),[0,0,1]); refused('wrong-shape-equal-count',(128,x,z),exp=7)
        refused('bad-source',(128,0,y),exp=1)
        refused('bad-destination',(128,x,0),exp=1)
        refused('forward-reference',(128,x,wire.reference(2)),(13,y),exp=1)
        refused('valid-write-then-invalid',(128,x,y),(128,0,y),exp=1)
        assert run((128,x,y))['value']==y and inspect(y)[2]==bytes([1,1,0])
        matrix(positive,self_positive,numeric_positive,numeric_self)
    summary=dict(mode=mode,image=pin,rows=rows,frames=s.frames)
    if mode!='baseline':
        assert used==set(fixture['rows']) and 'torch' not in sys.modules
        summary.update(oracle=oracle,fixture_sha256=a.fixture_sha256,separate_no_torch_native_process=True)
    (root/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
finally:
    s.finish()
