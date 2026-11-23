"""Run only reviewed ordinary v3 fixture against three exact coexisting images."""
import hashlib,json,math,sys,zlib
from pathlib import Path
H,protocol,F,L,combined,where,add,out=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
pins=dict(combined='5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d',
 where='2ee8c061f44e0496321141992881ca59557a22ccb7782e5b7cf1a1a313fce8f6',
 add='31a267d82b274ae81a956a03f8da1c9eb5a28bbcc903f78e56744333d1a88d4d')
assert sha(F.read_bytes())=='08ab226b27f729c544adb42365ed4629414561575f40ac6cba0bbc4a4397cc4b'
assert sha(L.read_bytes())=='cddbbd855bd10a26d681a1b956625d6ca3fb0e01d3d8765a702f1149f8603cc3'
for name,d in dict(test_runtime_candidate='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226',test_component='97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c').items():
 assert sha((H/(name+'.py')).read_bytes())==d
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert 'torch' not in sys.modules
W=load_wire(protocol)
DT={'Bool':(1,1),'I64':(2,8),'F32':(0,4),'F64':(4,8),'F16':(5,2),'BF16':(6,2)}
F=json.loads(F.read_bytes()); oracle=json.loads(L.read_bytes());assert oracle['status']=='PASS' and not oracle['mismatches']
lab={s['id']:s['report'] for s in oracle['rows'] if s['mode']=='composed'}
out.mkdir(exist_ok=False)
for lane,img in dict(combined=combined,where=where,add=add).items():
 assert sha(img.read_bytes())==pins[lane]
 d=out/lane
 S=W.Session(img,directory=d);(d/'owners').mkdir()
 cases=[]
 def exchange(q,op):
  st,rr=W.parse(S.exchange(q))
  assert st==0 and len(rr)==1 and rr[0]['opcode']==op,(lane,S.frames,op,st)
  return rr[0]
 def one(*c):
  return exchange(W.packet((c,)),c[0])
 def make(s):
  tag,w=DT[s['dtype']]; b=bytes.fromhex(s['raw_le'])
  arr=[int.from_bytes(b[i:i+w],'little') for i in range(0,len(b),w)]
  y=exchange(b''.join(factory(tag,s['shape'],arr)),97)
  return y['value'],S.frames
 def observe(h,s):
  tag,w=DT[s['dtype']]; n=math.prod(s['shape']); x=one(13,h);mf=S.frames
  m=W.meta(x);y=one(98,h,0,n)
  assert y['value']==n and y['payload']==bytes.fromhex(s['raw_le']),(lane,S.frames,'raw',s['dtype'])
  rank=len(s['shape'])
  assert m['rank']==rank and m['numel']==n and m['shape'][:rank]==s['shape'] and m['strides'][:rank]==s['strides'],(lane,S.frames,'meta',m,s)
  assert m['version']==m['storage_offset']==m['requires_grad']==0 and m['storage_bytes']==n*w
  return dict(handle=h,storage=m['storage_id'],version=m['version_id'],meta_frame=mf,read_frame=S.frames)
 def snapshot():
  x=S.owner();b=x['bytes'];k=sha(b);p=d/'owners'/(k+'.zlib')
  if not p.exists():p.write_bytes(zlib.compress(b,4))
  return dict(sha256=k,bytes=len(b))
 try:
  for c in F['cases']:
   chosen=[row for row in c['composed_reference'] if lane=='combined' or (row[1]==129)==(lane=='where')]
   if not chosen:continue
   case=dict(id=c['id'],factories={},outputs=[])
   spec=dict(c['inputs'])
   used={p for row in chosen for p in row[2:]}
   if lane=='add' and 'where' in used:spec['where']=c['expect']['where']
   live={}
   for name,s in spec.items():
    if name not in used:continue
    h,setup=make(s); current=observe(h,s);live[name]=(h,current,s)
    case['factories'][name]=dict(setup=setup,observation=current)
   for name,op,*args in chosen:
    before=snapshot()
    rr=one(op,*[live[k][0] for k in args]); f=S.frames
    aft=snapshot()
    h=rr['value']; s=c['expect'][name]; z=observe(h,s)
    assert all(h!=q[0] and z['storage']!=q[1]['storage'] and z['version']!=q[1]['version'] for q in live.values())
    assert lab[c['id']][name]['logical_sha256']==s['sha256']
    live[name]=(h,z,s)
    case['outputs'].append(dict(name=name,operator=op,args=args,frame=f,pre=before,post=aft,observation=z))
   case['drops']=[]
   for k,(h,_,s) in reversed(live.items()):
    one(41,h);case['drops'].append(dict(name=k,frame=S.frames))
   cases.append(case)
 finally:
  (d/'report.json').write_text(json.dumps(dict(lane=lane,image=pins[lane],frames=S.frames,cases=cases,no_torch=True),sort_keys=True,indent=2)+'\n')
  S.finish()
 assert len(cases)==(14 if lane=='where' else 50)
 print(lane,len(cases),S.frames,flush=True)
