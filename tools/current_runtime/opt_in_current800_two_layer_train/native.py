"""Exact normal numeric one-step app, no Torch or host numerical intermediate."""
import hashlib,json,struct,sys,zlib
from pathlib import Path
h,w,image,fixture,measured,out=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
pin='5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d'
assert sha(image.read_bytes())==pin
assert sha(fixture.read_bytes())=='e9719317873eb160adfc5260a1515bda9cf1da8295802418c0618234f96f553b'
assert sha((measured/'receipt.json').read_bytes())=='b6ad4841e8b14e91703d85fee3850b5acbf6ccd3c3e96fab7d0dace76177599d'
oracle=(measured/'all-f32le.bin').read_bytes()
assert sha(oracle)=='add68ff7d41e9e1aaefec9fe73188e42b47f0ef8ef50284c7f15d4db11addc08'
rr=json.loads((measured/'receipt.json').read_bytes()); assert rr['status']=='MATCH_LOCKED_TWO_LAYER_STEP_EXACT'
R={r['label']:r for r in rr['rows']}
assert sha((h/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(h))
from test_runtime_candidate import load_wire
W=load_wire(w);assert 'torch' not in sys.modules
p=json.loads(fixture.read_bytes());S=W.Session(image,directory=out);(out/'owners').mkdir()
live={};expected={};log=[];ids={}
def shot():
 b=S.owner()['bytes'];q=sha(b);d=out/'owners'/(q+'.zlib')
 if not d.exists():d.write_bytes(zlib.compress(b,4))
 else:assert zlib.decompress(d.read_bytes())==b
 return q
def issue(q,op):
 st,r=W.parse(S.exchange(q));assert st==0 and len(r)==1 and r[0]['opcode']==op,(S.frames,op,st)
 return r[0]
def one(*args):return issue(W.packet((args,)),args[0])
def obs_bytes(label,fixture):
 row=R[label];start=row['offset'];raw=oracle[start:start+row['length']]
 assert raw.hex()==fixture['raw_f32le_hex']
 return raw
def verify(name):
 label,e=expected[name];raw=obs_bytes(label,e);h=live[name];n=len(raw)//4;rnk=len(e['shape'])
 z=W.meta(one(13,h));mf=S.frames;r=one(98,h,0,n);rf=S.frames
 assert (z['rank'],z['numel'],z['shape'][:rnk],z['strides'][:rnk])==(rnk,n,e['shape'],e['strides']),(name,z,e)
 assert [z[k] for k in ('version','storage_offset','requires_grad','storage_bytes')]==[e['version'],0,int(e['requires_grad']),len(raw)],(name,z,e)
 assert r['value']==n and r['payload']==raw,(name,r['payload'].hex(),raw.hex())
 value=(h,z['storage_id'],z['version_id'])
 if name in ids:assert value==ids[name]
 ids[name]=value
 return dict(handle=h,storage=value[1],version=value[2],meta=mf,raw=rf)
def record(kind,frame=None,extra=None):
 owner=shot();ob={name:verify(name) for name in live}
 assert all(len({t[k] for t in ob.values()})==len(ob) for k in ('handle','storage','version'))
 log.append(dict(kind=kind,frame=frame,owner=owner,observations=ob)|({} if extra is None else extra))
def graph(ps,phase):
 for s in ps:
  a=[live[v] if isinstance(v,str) else v for v in s['command']];n=s['name'];row=one(*a);f=S.frames
  assert row['value'] not in live.values()
  live[n]=row['value'];expected[n]=(n,s['expected']);record(phase,f,dict(stage=n,command=a))
def gradient(label):
 return {name: grad_one(label,name) for name in ('w1','w2')}
def grad_one(label,name):
 r=one(11,live[name]);fr=S.frames
 want=obs_bytes('gradient_'+name,p['backward']['gradient_'+name])
 assert r['value']==len(want)//4 and r['payload']==want,(label,name,r)
 return fr
def drop_graph(seq,label):
 for s in reversed(seq):
  n=s['name'];one(41,live.pop(n));log.append(dict(kind=label,frame=S.frames,stage=n))
try:
 for n,e in p['inputs'].items():
  raw=obs_bytes(n,e);s=e['shape'];q=b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)+struct.pack('<8Q',97,0,96,0,0,0,0,0)
  q+=struct.pack('<8Q',len(s),int(e['requires_grad']),*s,*([0]*(4-len(s))),160,len(raw))+raw
  live[n]=issue(q,97)['value'];expected[n]=(n,e)
 record('input')
 graph(p['before_update'],'before')
 one(7,live['loss']);bfr=S.frames
 gfr=gradient('after-ordinary-backward');record('backward',bfr,dict(gradient_frames=gfr))
 drop_graph(p['before_update'],'release-before')
 for u in p['updates']:
  a=u['command']; name=a[1];assert a==[9,name,0x3d000000] and name in ('w1','w2')
  one(a[0],live[name],a[2]);ufr=S.frames
  expected[name]=('one_sgd_update_'+name,u['expected'])
  gafter=gradient('after-update-'+name)
  record('sgd',ufr,dict(stage=name,gradient_frames=gafter))
 graph(p['after_update'],'after')
 drop_graph(p['after_update'],'release-after')
 for n in ('w2','w1','x','target'):
  one(41,live.pop(n));log.append(dict(kind='release-input',frame=S.frames,stage=n))
 assert not live
except BaseException as e:
 (out/'error.txt').write_text(repr(e)+'\n');raise
finally:
 (out/'ledger.json').write_text(json.dumps(log,sort_keys=True,indent=2)+'\n')
 S.finish()
(out/'receipt.json').write_text(json.dumps(dict(elf=pin,one_owned_session=True,frames=S.frames,
 ordinary_backward_calls=1,ordinary_sgd_calls=2,external_roots=4,graph_numeric_stages=12,external_torch_backend=False),sort_keys=True,indent=2)+'\n')
print('PASS ordinary two-layer training step native',S.frames,flush=True)
