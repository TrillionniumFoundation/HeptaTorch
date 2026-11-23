"""Guarded no-Torch physical lane for exact predecessor and standalone candidate."""
from pathlib import Path
import hashlib,math,json,struct,sys,zlib
H,wire,image,D,out=sys.argv[1:6]; mode=sys.argv[6]
H,wire,image,D,out=map(Path,(H,wire,image,D,out))
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
assert sha((H/'test_component.py').read_bytes())=='97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c'
sys.path.insert(0,str(H));from test_runtime_candidate import factory,load_wire
W=load_wire(wire)
p=(D/'preregister.json').read_bytes(); assert sha(p)=='20d3c5f5383e2ca80cfab1f26884e6ae76031b18c804ed15b86594d5321e735c'
o=(D/'oracle.json').read_bytes(); assert sha(o)=='c88ded5784f1fbd0e8db86bb48582bb5b5eef2d723217d0cfa0cd7a6f7f1561d'
elf=sha(image.read_bytes())
pins=json.loads((D/'physical-pins.json').read_bytes()); assert elf==pins[mode]
ref={r['label']:r for r in json.loads(o)['rows']}; fixture=json.loads(p)
F={0:4,1:1,2:8,4:8,5:2,6:2}
OP=131; S=W.Session(image,directory=out); (out/'owners').mkdir()
ledger=[]
def ex(*ops):return W.parse(S.exchange(W.packet(ops)))
def run(*ops):
 st,r=ex(*ops);assert not st and [x['opcode'] for x in r]==[x[0] for x in ops],(st,ops,r)
 return r
def make(d,sh,words):
 st,r=W.parse(S.exchange(b''.join(factory(d,sh,words))))
 assert not st and len(r)==1 and r[0]['opcode']==97,(st,d,sh)
 return r[0]['value']
def build(z):return make(z['dtype'],z['shape'],z['words'])
def drop(*ids):
 for a in dict.fromkeys(ids):run((41,a))
def read(h,n,want):
 a,b=run((13,h),(98,h,0,n));m=W.meta(a)
 assert b['value']==n and b['payload'].hex()==want,(h,n,want,b)
 return S.frames,m
def snap():
 q=S.owner(); raw=q['bytes']; h=sha(raw);f=out/'owners'/(h+'.zlib')
 if f.exists():assert zlib.decompress(f.read_bytes())==raw
 else:f.write_bytes(zlib.compress(raw,6))
 return dict(sha256=h,length=len(raw),ptr=q['pointer'])
def refuse(label,status,*cmd):
 pre=snap();q=W.packet(cmd);st,r=W.parse(S.exchange(q));j=S.frames;post=snap()
 ledger.append(dict(kind='refusal',label=label,status=st,expected=status,pre=pre,post=post,frame=j,request_sha256=sha(q)))
 assert st==status and not r and pre==post,(label,st,status,r,pre,post)
 return j
def mkview(n,sh,strides,off):
 pad=[0]*(4-len(sh))
 return run((111,n,W.Payload(struct.pack('<10Q',len(sh),*sh,*pad,*strides,*pad,off)),80))[0]['value']
try:
 for i,z in enumerate(fixture['rows']):
  l=z['left']; r=z['right']; expected=z['expected'];t=ref[z['label']];n=len(expected['words'])
  assert t['raw_hex']==expected['raw_hex'] and t['shape']==expected['shape'] and t['strides']==expected['strides']
  x=build(l);y=x if z['owner_relationship']=='same_handle' else build(r)
  roads=[read(u,n,a['raw_hex'])[0] for u,a in ((x,l),(y,r))]
  if mode=='baseline':
   refuse('original-'+z['label'],4,(OP,x,y));drop(x,y);continue
  old=snap();out_id=run((OP,x,y))[0]['value'];j=S.frames; after=snap()
  f,m=read(out_id,n,expected['raw_hex'])
  assert m['rank']==len(expected['shape']) and m['numel']==n and m['version']==m['storage_offset']==m['requires_grad']==0
  assert m['storage_bytes']==n and tuple(m['shape'][:m['rank']])==tuple(expected['shape'])
  assert tuple(m['strides'][:m['rank']])==tuple(expected['strides'])
  after_reads=[read(u,n,a['raw_hex'])[0] for u,a in ((x,l),(y,r))]
  wk=run((42,out_id))[0]['value'];wkf=S.frames
  if i%2==0:
   drop(x,y); surv=[read(out_id,n,expected['raw_hex'])[0]];drop(out_id)
  else:
   drop(out_id);surv=[read(x,n,l['raw_hex'])[0],read(y,n,r['raw_hex'])[0]];drop(x,y)
  expired=refuse('weak-expire-'+z['label'],6,(43,wk));run((44,wk))
  ledger.append(dict(kind='positive',label=z['label'],input_reads=roads,after_reads=after_reads,frame=j,out_read=f,inputs=[x,y],output=out_id,
                     pre=old,post=after,weak=wk,weak_frame=wkf,expired=expired,survivor=surv))
 if mode=='candidate':
  for z in fixture['scope_controls']:
   x,y=[make(d,sh,[0]*math.prod(sh)) for d,sh in zip(z['tags'],(z['left_shape'],z['right_shape']),strict=True)]
   refuse(z['label'],7,(OP,x,y));drop(x,y)
  base=make(0,[9],list(range(9)))
  for lab,sh,sd,off in [('transpose',[3,2],[1,3],0),('offset',[2,3],[3,1],1),('singleton-view',[2,1,2],[2,9,1],0)]:
   v=mkview(base,sh,sd,off);x=make(0,sh,[0]*math.prod(sh))
   a,=run((13,v)); md=W.meta(a);assert tuple(md['strides'][:len(sh)])==tuple(sd) and md['storage_offset']==4*off
   refuse(lab+'-left',7,(OP,v,x));refuse(lab+'-right',7,(OP,x,v));drop(v,x)
  drop(base)
  x=make(0,[1],[0x3f800000]); y=make(0,[1],[1]);run((90,y,1));mr,=run((13,y));assert W.meta(mr)['requires_grad']==1
  refuse('grad-left',7,(OP,y,x));refuse('grad-right',7,(OP,x,y));drop(y)
  dead=make(0,[1],[0]);drop(dead)
  refuse('stale-left',1,(OP,dead,x));refuse('stale-right',1,(OP,x,dead))
  for k in range(3,8):
   row=[OP,x,x,0,0,0,0,0];row[k]=1
   refuse('padding-'+str(k),7,tuple(row))
  cap_a=make(0,[513,1],[0x3f800000]*513);cap_b=make(0,[1,512],[0x00000000]*512)
  big=run((2,1,cap_a,cap_b))[0]['value'];md,=run((13,big));assert W.meta(md)['numel']==262656
  refuse('capacity-over-limit',3,(OP,big,big));drop(big,cap_a,cap_b)
  for a in (130,132):refuse('neighbor-'+str(a),4,(a,x,x))
  refuse('later-unknown-rollback',4,(OP,x,x),(65500,))
  a,b,c=run((OP,x,x),(13,W.reference(1)),(98,W.reference(1),0,1))
  assert b['value']==a['value'] and W.meta(b)['version']==0 and c['payload']==b'\x01'
  ledger.append(dict(kind='same-frame',frame=S.frames,input=x,output=a['value']));drop(x,a['value'])
finally:
 assert 'torch' not in sys.modules
 (out/'report.json').write_text(json.dumps(dict(no_torch=True,mode=mode,elf_sha256=elf,fixture_sha256=sha(p),oracle_sha256=sha(o),frames=S.frames,rows=ledger),indent=2,sort_keys=True)+'\n')
 S.finish()
