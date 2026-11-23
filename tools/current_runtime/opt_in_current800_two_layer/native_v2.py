"""Positive-only numerical one-session application driver; no host compute backend."""
import hashlib,json,struct,sys,zlib
from pathlib import Path
h,w,image,fixture,out=map(Path,sys.argv[1:]);out.mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
pin='5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d'
assert sha(image.read_bytes())==pin
assert sha(fixture.read_bytes())=='899f9aaecbd6621625bf2b702ab445922cb4d3d10ba03e4c5fc0a8b8eaa05065'
assert sha((h/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(h))
from test_runtime_candidate import load_wire
W=load_wire(w);assert 'torch' not in sys.modules
p=json.loads(fixture.read_bytes());done=[]
for i,admit in enumerate(p['admitted']):
 c=admit['original_frozen'];name=c['label'];evidence=out/name;S=W.Session(image,directory=evidence)
 (evidence/'owners').mkdir(); handles={};expected={};lines=[]
 def shot():
  b=S.owner()['bytes']; q=sha(b)
  d=evidence/'owners'/(q+'.zlib')
  if not d.exists():d.write_bytes(zlib.compress(b,4))
  else:assert zlib.decompress(d.read_bytes())==b
  return q
 def issue(q,op):
  st,r=W.parse(S.exchange(q));assert st==0 and len(r)==1 and r[0]['opcode']==op,(name,S.frames,op,st)
  return r[0]
 def one(*args):return issue(W.packet((args,)),args[0])
 def verify(key):
  m=expected[key];id=handles[key];n=m['storage_bytes']//4
  z=W.meta(one(13,id));mf=S.frames
  r=one(98,id,0,n);rf=S.frames
  rank=len(m['shape'])
  assert z['rank']==rank and z['numel']==n and z['shape'][:rank]==m['shape'] and z['strides'][:rank]==m['strides']
  assert [z[key] for key in ('version','storage_offset','requires_grad','storage_bytes')]==[m['version'],m['offset'],0,m['storage_bytes']]
  assert r['value']==n and r['payload'].hex()==m['raw_f32le_hex'],(key,r['payload'].hex(),m['raw_f32le_hex'])
  return dict(handle=id,storage=z['storage_id'],version=z['version_id'],meta=mf,raw=rf)
 try:
  for k,e in c['inputs'].items():
   shape=e['shape'];raw=bytes.fromhex(e['raw_f32le_hex'])
   q=b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)+struct.pack('<8Q',97,0,96,0,0,0,0,0)
   q+=struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),160,len(raw))+raw
   handles[k]=issue(q,97)['value'];expected[k]=e
  base=shot(); snap={k:verify(k) for k in handles};lines.append(dict(kind='input',owner=base,observations=snap))
  assert len(set(v['storage'] for v in snap.values()))==len(set(v['version'] for v in snap.values()))==5
  for s in c['stages']:
   args=tuple(handles[v] if isinstance(v,str) else v for v in s['command']);k=s['name'];previous=dict(handles)
   val=one(*args)['value'];opframe=S.frames
   assert val not in previous.values()
   handles[k]=val;expected[k]=s['expected'];owner=shot();live={z:verify(z) for z in handles}
   ids=[live[z] for z in handles];n=len(ids)
   assert len(set(id['storage'] for id in ids))==len(set(id['version'] for id in ids))==n
   assert all(live[z]['storage']==snap[z]['storage'] and live[z]['version']==snap[z]['version'] for z in previous)
   snap=live;lines.append(dict(kind='stage',name=k,command=args,frame=opframe,owner=owner,observations=live))
  order=list(handles)
  order=(list(reversed(order)) if i==1 else order) # leases are all still legal and live
  for n in order:
   one(41,handles[n]);f=S.frames;del handles[n]
   if handles:
    k=next(reversed(handles)) if i!=1 else next(iter(handles)); ref=verify(k)
    assert ref['storage']==snap[k]['storage'] and ref['version']==snap[k]['version']
   else:k=None;ref=None
   lines.append(dict(kind='drop',name=n,frame=f,survivor=k,survival=ref))
  done.append(dict(case=name,frames=S.frames,release_order=('result-first' if i==1 else 'inputs-first')))
 except BaseException as e:
  (evidence/'error.txt').write_text(repr(e)+'\n');raise
 finally:
  (evidence/'ledger.json').write_text(json.dumps(lines,sort_keys=True,indent=2)+'\n')
  S.finish()
assert len(done)==3
(out/'report.json').write_text(json.dumps(dict(elf=pin,one_owned_session_per_case=True,literal_roots=15,stages=15,
  external_torch_backend=False,cases=done),sort_keys=True,indent=2)+'\n')
print(done,flush=True)
