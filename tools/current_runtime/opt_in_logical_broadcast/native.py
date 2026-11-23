"""Guarded native observer. PyTorch is never imported."""
from pathlib import Path
import hashlib,json,math,os,resource,struct,sys,zlib
H,wire,ELF,D,OUT=map(Path,sys.argv[1:6]);mode=sys.argv[6]
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(H));from test_runtime_candidate import factory,load_wire
W=load_wire(wire)
s=(D/'suite.json').read_bytes();pins=json.loads((D/'physical-pins.json').read_bytes())
assert sha(s)==pins['suite'] and sha(ELF.read_bytes())==pins[mode]
P=json.loads(s);S=W.Session(ELF,directory=OUT);(OUT/'owners').mkdir()
ledger=[]
def exchange(*args):return W.parse(S.exchange(W.packet(args)))
def run(*args):
 st,r=exchange(*args);assert st==0 and [x['opcode'] for x in r]==[g[0] for g in args],(st,args,r)
 return r
def make(t,sh,ws):
 st,r=W.parse(S.exchange(b''.join(factory(t,sh,ws))));assert st==0 and len(r)==1 and r[0]['opcode']==97
 return r[0]['value']
def load(x):return make(x['dtype'],x['shape'],x['words'])
def drop(*h):
 for v in dict.fromkeys(h):run((41,v))
def read(h,n,expected=None):
 m,r=run((13,h),(98,h,0,n));v=r['payload'];assert r['value']==n
 if expected is not None:assert v.hex()==expected,(h,n,sha(v),expected[:40])
 return S.frames,W.meta(m),v
def snap():
 z=S.owner();raw=z['bytes'];h=sha(raw);p=OUT/'owners'/(h+'.zlib')
 if p.exists():assert zlib.decompress(p.read_bytes())==raw
 else:p.write_bytes(zlib.compress(raw,5))
 return dict(hash=h,length=len(raw),ptr=z['pointer'],scratch=list(struct.unpack_from('<4Q',raw,128)))
def refusal(label,status,*cmd):
 pre=snap();q=W.packet(cmd);st,v=W.parse(S.exchange(q));i=S.frames;post=snap()
 ledger.append(dict(kind='refusal',label=label,status=st,expected=status,before=pre,after=post,frame=i,request_sha256=sha(q)))
 assert st==status and not v and pre==post,(label,st,status)
 return i
def desc(md,t):
 r=len(t['shape']);n=math.prod(t['shape'])
 assert md['rank']==r and md['numel']==n and tuple(md['shape'][:r])==tuple(t['shape']) and tuple(md['strides'][:r])==tuple(t['strides'])
 assert md['version']==md['storage_offset']==md['requires_grad']==0 and md['storage_bytes']==n
def schedule_case(z,i):
 op=z['op']; a,b=z['left'],z['right'];n=z['output']['numel'];x,y=load(a),load(b)
 oldreads=[read(h,len(t['words']),t['raw_hex'])[0] for h,t in ((x,a),(y,b))]
 if mode=='baseline':
  refusal('parent-'+z['label'],7,(op,x,y));drop(x,y);return
 before=snap();o,=run((op,x,y));f=S.frames;after=snap()
 assert before['scratch']==after['scratch'],(z['label'],before['scratch'],after['scratch'])
 tr,md,_=read(o['value'],n,z['output']['raw_hex']); desc(md,z['output'])
 newreads=[read(h,len(t['words']),t['raw_hex'])[0] for h,t in ((x,a),(y,b))]
 w,=run((42,o['value']));wf=S.frames
 if i%2:
  drop(o['value']); surv=[read(h,len(t['words']),t['raw_hex'])[0] for h,t in ((x,a),(y,b))];drop(x,y)
 else:
  drop(x,y); surv=[read(o['value'],n,z['output']['raw_hex'])[0]];drop(o['value'])
 wno=refusal('weak-'+z['label'],6,(43,w['value']));run((44,w['value']))
 ledger.append(dict(kind='positive',label=z['label'],frame=f,out=h_out(o),inputs=[x,y],before=before,after=after,
  read=tr,old_reads=oldreads,new_reads=newreads,weak=w['value'],weak_frame=wf,weak_refusal=wno,surv=surv))
def h_out(r):return r['value']
def capacity_digest(op,md,raw,n):
 z=next(x for x in P['locked_caps'] if int(x['op'])==op and x['n']==n)
 assert sha(raw)==z['sha']==z['physical_sha'] and sum(raw)==z['ones'] and raw[:8].hex()==z['first'] and raw[-8:].hex()==z['last']
 desc(md,z)
def seed():
 a=make(0,[2,3,4],[0]*24); b=make(0,[2,3,4],[0]*24)
 result=run((2,1,a,b))[0]['value'];drop(a,b,result)
 return snap()['scratch']
try:
 for i,z in enumerate(P['rows']):schedule_case(z,i)
 # Proof that caller-owned scratch contains live, nondefault words before actual staging.
 if mode=='candidate':
  seeded=seed();assert seeded[:3]==[2,3,4],seeded
 for c in P['caps']:
  x,y=load(c['left']),load(c['right']);n=c['output_count']
  for op in (131,133,134):
   if mode=='baseline' or c['predicted_status']:
    refusal(f'cap-{op}-{n}',7 if mode=='baseline' else 3,(op,x,y));continue
   before=snap();a,=run((op,x,y));f=S.frames;after=snap()
   assert before['scratch']==after['scratch'] and before['scratch'][:3]==[2,3,4]
   rr,m,data=read(a['value'],n);capacity_digest(op,m,data,n)
   ledger.append(dict(kind='cap',op=op,frame=f,read=rr,output=a['value'],inputs=[x,y],before=before,after=after))
   drop(a['value'])
  drop(x,y)
 if mode=='candidate':
  for c in P['controls']:
   x,y=(make(1,z,[1]*math.prod(z)) for z in (c['left'],c['right']))
   for op in (131,133,134):refusal('geometry-'+c['label']+'-'+str(op),7,(op,x,y))
   drop(x,y)
  # All three same-frame result references exercise broadcast output cursor; late error rolls back allocation and scratch.
  for op in (131,133,134):
   z=next(c for c in P['rows'] if c['op']==op and 'dtype1-crossed-rank3' in c['label'])
   x,y=load(z['left']),load(z['right']);n=z['output']['numel']
   refusal(f'late-broadcast-{op}',4,(op,x,y),(65500,))
   a,b,c=run((op,x,y),(13,W.reference(1)),(98,W.reference(1),0,n))
   assert a['value']==b['value'] and c['payload'].hex()==z['output']['raw_hex'];desc(W.meta(b),z['output'])
   ledger.append(dict(kind='same-frame',label=z['label'],frame=S.frames,op=op,input=[x,y],output=a['value']))
   drop(x,y,a['value'])
  # Actual cap pressure: build valid roots first; probe the exact same allocation before and after lifting a process-local AS cap.
  assert seed()[:3]==[2,3,4]
  c=P['caps'][0];x,y=load(c['left']),load(c['right']);op=133;n=c['output_count'];q=W.packet([(op,x,y)])
  owner=S.owner();page=os.sysconf('SC_PAGE_SIZE');prior=resource.prlimit(S.pid,resource.RLIMIT_AS)
  vm=int(Path(f'/proc/{S.pid}/statm').read_text().split()[0])*page
  ceiling=vm+((owner['length']+16+page-1)//page)*page+65536
  kept=[]; failure=None; lead=[]
  try:
   resource.prlimit(S.pid,resource.RLIMIT_AS,(ceiling,prior[1]));assert resource.prlimit(S.pid,resource.RLIMIT_AS)[0]==ceiling
   for attempt in range(1,9):
    before=snap();st,r=W.parse(S.exchange(q));f=S.frames;after=snap()
    assert before['scratch']==after['scratch'] and before['scratch'][:3]==[2,3,4]
    if st:
     assert st==3 and not r and before==after,(st,len(r),before,after)
     failure=dict(kind='pressure',frame=f,lead=lead,before=before,after=after,request_sha256=sha(q),status=st,expected=3,
        limit_bytes=ceiling,pre_limit=list(prior),attempt=attempt,inputs=[x,y],op=op)
     break
    assert len(r)==1 and r[0]['opcode']==op
    kept.append(r[0]['value']);lead.append(f)
  finally:resource.prlimit(S.pid,resource.RLIMIT_AS,prior)
  assert failure is not None and resource.prlimit(S.pid,resource.RLIMIT_AS)==prior
  b,a= snap(),run((op,x,y))[0];f=S.frames;p=snap()
  assert b['scratch']==p['scratch'] and p['scratch'][:3]==[2,3,4]
  rr,md,raw=read(a['value'],n);capacity_digest(op,md,raw,n)
  failure.update(recover_frame=f,recover_read=rr,recovered_output=a['value'],recovery_before=b,recovery_after=p)
  ledger.append(failure);drop(x,y,*kept,a['value'])
finally:
 assert 'torch' not in sys.modules
 (OUT/'report.json').write_text(json.dumps(dict(stage='native-no-Torch',mode=mode,elf=pins[mode],fixture=sha(s),frames=S.frames,rows=ledger),sort_keys=True,indent=2)+'\n')
 S.finish()
