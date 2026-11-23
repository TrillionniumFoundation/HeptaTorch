"""Physical production-protocol observer; no Torch."""
from pathlib import Path
import hashlib,json,struct,sys,zlib
H,protocol,image,fixture,out,mode=map(Path,sys.argv[1:]); mode=mode.name
def sha(b):return hashlib.sha256(b).hexdigest()
for n,digest in [('test_runtime_candidate.py','40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'),
                 ('test_component.py','97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c')]:
 assert sha((H/n).read_bytes())==digest
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert 'torch' not in sys.modules
pin={'predecessor':'6c8e5d338f63f7cf2cd4437b1f49294e635d6f07e9927b4b1f0ccdd11d8a1095','candidate':'87e7d880848f363374d020536d16b5a67ed17a60b22189b23bf0eb9e782a2c11'}[mode]
assert sha(image.read_bytes())==pin
raw_fixture=fixture.read_bytes(); rows=json.loads(raw_fixture)['rows']
W=load_wire(protocol); S=W.Session(image,directory=out);(out/'owners').mkdir()
evidence=[]
def packet(*cmd):return W.parse(S.exchange(W.packet(cmd)))
def run(*cmd):
 status,rr=packet(*cmd); assert status==0 and [r['opcode'] for r in rr]==[x[0] for x in cmd],(status,cmd,rr)
 return rr
def make(shape,words,dtype=5):
 status,rr=W.parse(S.exchange(b''.join(factory(dtype,shape,words))))
 assert status==0 and len(rr)==1 and rr[0]['opcode']==97,(status,rr,shape,dtype)
 return rr[0]['value']
def snap():
 r=S.owner();b=r['bytes'];h=sha(b);p=out/'owners'/(h+'.zlib')
 if p.exists(): assert zlib.decompress(p.read_bytes())==b
 else: p.write_bytes(zlib.compress(b,6))
 return dict(sha256=h,length=len(b),ptr=r['pointer'])
def config(shape,strides,offset=0):
 p=[0]*(4-len(shape))
 return W.Payload(struct.pack('<10Q',len(shape),*shape,*p,*strides,*p,offset))
def view(h,shape,strides,offset=0):return (111,h,config(shape,strides,offset),80)
def refuse(name,status,*cmd):
 pre=snap();st,reply=packet(*cmd);f=S.frames;post=snap()
 evidence.append(dict(kind='refusal',case=name,frame=f,status=st,expected=status,pre=pre,post=post))
 assert st==status and not reply and pre==post,(name,st,reply,pre,post)
 return f
def read(h,n,hexraw):
 m,r=run((13,h),(98,h,0,n));f=S.frames
 assert r['payload'].hex()==hexraw,(r['payload'].hex(),hexraw)
 return f,W.meta(m)
try:
 for row in rows:
  n=len(row['root_u16']);r=make(row['root_shape'],row['root_u16'])
  create_frame=S.frames
  root_read,old=read(r,n,row['raw_root'])
  a=view(r,row['shape'],row['strides'])
  if mode=='predecessor':
   refuse('original-real-'+row['label'],7,a)
   run((41,r))
   continue
  before=snap()
  v=run(a)[0]['value'];fv=S.frames
  sib=run(a)[0]['value'];fs=S.frames
  assert len({r,v,sib})==3
  after=snap()
  fvr,mv=read(v,n,row['raw_view']);fsr,ms=read(sib,n,row['raw_view']);frr,mr=read(r,n,row['raw_root'])
  assert tuple(mv['shape'][:mv['rank']])==tuple(row['shape']) and tuple(mv['strides'][:mv['rank']])==tuple(row['strides'])
  for m in (mv,ms,mr):
   assert (m['version_id'],m['storage_id'],m['version'],m['storage_offset'],m['storage_bytes'],m['requires_grad'])==(old['version_id'],old['storage_id'],old['version'],0,2*n,0)
  refuse('nested-'+row['label'],7,view(v,row['shape'],row['strides']))
  refuse('valid-view-followed-by-stale-view-'+row['label'],1,a,view((1<<32)|(1<<24)|16372,row['shape'],row['strides']))
  source=make(row['shape'],[0]*n)
  if row['label']=='rank2-row-major':
   refuse('new-f16-view-is-still-not-out-covered-'+row['label'],7,(128,source,v))
  else:
   refuse('accepted-permuted-out-followed-by-invalid-'+row['label'],1,(128,source,v),(128,0,v))
  run((41,source))
  weak=run((42,r))[0]['value'];fweak=S.frames
  # Both release orders: rank 3 child first; other rows root first.
  if row['label']=='rank3-permutation':
   run((41,v));drop1=S.frames;survivor=r;expect=row['raw_root'];drop2_handle=r
  else:
   run((41,r));drop1=S.frames;survivor=v;expect=row['raw_view'];drop2_handle=v
  middle=snap()
  mid1,_=read(survivor,n,expect);mid2,_=read(sib,n,row['raw_view'])
  run((41,drop2_handle));drop2=S.frames
  open_hold=run((43,weak))[0]['value'];hold_frame=S.frames;run((45,open_hold))
  after_two=snap()
  mid3,_=read(sib,n,row['raw_view'])
  run((41,sib));drop3=S.frames
  expired=refuse('last-view-leaves-weak-expired-'+row['label'],6,(43,weak))
  run((44,weak))
  evidence.append(dict(kind='positive',fixture=row['label'],create=create_frame,root_read=root_read,constructor=fv,sibling_constructor=fs,root=r,view=v,sibling=sib,view_read=fvr,sib_read=fsr,root_after_read=frr,before=before,after=after,weak=weak,weak_frame=fweak,hold=open_hold,hold_frame=hold_frame,drop_frames=[drop1,drop2,drop3],mid_reads=[mid1,mid2,mid3],middle=middle,after_two=after_two,survivor=survivor,expired=expired))
 if mode=='candidate':
  root=make([6],[0,1,2,3,4,5])
  for label,shape,strides,offset in [('nonwhole-offset0',[2,2],[2,1],0),('nonzero-offset',[2,2],[2,1],1),('rank1',[6],[1],0),('rank4',[1,1,2,3],[6,6,3,1],0),('positive-overlap',[2,3],[1,1],0),('zero-stride',[2,3],[0,1],0),('empty-child',[2,0],[1,1],0)]:
   refuse(label,7,view(root,shape,strides,offset))
  gapped=make([10],list(range(10)))
  refuse('gapped-inbounds-nonwhole',7,view(gapped,[2,3],[1,3]))
  r_st,r_reply=packet((90,root,1))
  if r_st:
   evidence.append(dict(kind='setup_refusal',case='f16-root-grad-toggle',frame=S.frames,status=r_st))
  else:
   m,_=run((13,root),(96,root,0)); assert W.meta(m)['requires_grad']==1
   refuse('grad-root',5,view(root,[2,3],[1,2]))
 report=dict(mode=mode,elf=pin,fixture_sha256=sha(raw_fixture),rows=evidence,native_frames=S.frames,no_torch='torch' not in sys.modules)
except BaseException as e:
 report=dict(mode=mode,elf=pin,rows=evidence,native_frames=S.frames,error=repr(e))
 raise
finally:
 (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 S.finish()
