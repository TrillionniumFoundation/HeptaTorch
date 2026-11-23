"""Independent readback of exact raw request, reply, owner and pressure transcripts."""
from pathlib import Path
import hashlib,itertools,json,math,sys,zlib
D=Path(__file__).parent;sys.path.insert(0,str(D))
from read_old import Owner,frames,req,resp,q,sha
suite=(D/'suite.json').read_bytes()
pins=json.loads((D/'physical-pins.json').read_bytes());assert sha(suite)==pins['suite']
V=json.loads(suite);rows={z['label']:z for z in V['rows']}
WIDTH={0:4,1:1,2:8,4:8,5:2,6:2}
def verify_shape(sh,s):
 return tuple(math.prod(max(1,i) for i in sh[k+1:]) for k in range(len(sh)))==tuple(s)
def expected(a,b,op):
 sh=[a['shape'],b['shape']];rank=max(map(len,sh));dims=[]
 for x,y in zip(*([1]*(rank-len(s))+s for s in sh),strict=True):
  assert x==y or x==1 or y==1
  dims.append(max(x,y))
 tag=a['dtype'];assert b['dtype']==tag;w=WIDTH[tag]
 data=[bytes.fromhex(h['raw_hex']) for h in (a,b)]
 masks=(1<<(8*w-(tag in (0,4,5,6))))-1
 def projected(arg,i):
  pos=0
  s=sh[arg]
  for j,n in zip(i[rank-len(s):],s,strict=True):pos=pos*n+(j if n>1 else 0)
  bs=data[arg];return min(1,int.from_bytes(bs[pos*w:(pos+1)*w],'little')&masks)
 vals=[]
 for i in itertools.product(*(range(d) for d in dims)):
  x,y=(projected(k,i) for k in (0,1))
  vals.append(min(x,y) if op==131 else max(x,y) if op==133 else (x+y)%2)
 return dims,bytes(vals)
def analyze(p):
 r=json.loads((p/'report.json').read_bytes());assert r['elf']==pins[r['mode']] and r['fixture']==pins['suite']
 transport=json.loads((p/'transport.json').read_bytes());assert transport['executable_sha256']==r['elf'] and transport['exit_code']==0 and transport['responses_before_stdin_eof']
 bq=list(frames(p/'requests.frames.bin',b'HTOWNR01'));br=list(frames(p/'responses.frames.bin',b'HTOWNS01'))
 Q=list(map(req,bq));R=list(map(resp,br));assert len(Q)==len(R)==r['frames']
 def before(meta):
  raw=zlib.decompress((p/'owners'/(meta['hash']+'.zlib')).read_bytes())
  assert len(raw)==meta['length'] and sha(raw)==meta['hash'] and list(q(raw,4,128))==meta['scratch'];return raw
 def fetch(frame,h,n,data):
  assert Q[frame-1]==[(13,h,0,0,0,0,0,0),(98,h,0,n,0,0,0,0)];s,records=R[frame-1]
  assert s==0 and records[0][:2]==(13,h) and records[1]==(98,n,data);return q(records[0][2],16)
 def inspect(own,t,tag,shape,strides,data):
  node=own.node(t);rs=len(shape);row=own.row(1,t)[1]
  assert node[:2]==(rs,math.prod(shape)) and node[12:12+rs]==tuple(shape) and node[16:16+rs]==tuple(strides)
  assert verify_shape(shape,strides) and node[29]==node[83]==node[4]==node[5]==0 and node[72]==tag
  st,ver=node[27:29];strow=own.row(2,st)[1];vrow=own.row(3,ver)[1]
  assert row[2:4]==(st,ver) and row[4:6]==(0,len(data)) and own.storage(st)[1:]==(len(data),data)
  assert node[2]==(1<<63)+strow[2];return node,row,st,ver,strow,vrow
 def inputs(own,handles,z):
  return [inspect(own,h,x['dtype'],x['shape'],x['strides'],bytes.fromhex(x['raw_hex'])) for h,x in zip(handles,(z['left'],z['right']),strict=True)]
 def prove_result(e,z,handles,output,frame,rd,b,a):
  op=z['op'];shape,data=expected(z['left'],z['right'],op);n=len(data)
  if 'output' in z:assert tuple(shape)==tuple(z['output']['shape']) and data.hex()==z['output']['raw_hex']
  before_owners=inputs(Owner(b),handles,z);after_owners=inputs(Owner(a),handles,z); assert before_owners==after_owners
  assert q(b,4,128)==q(a,4,128)
  assert Q[frame-1]==[(op,*handles,0,0,0,0,0)] and R[frame-1]==(0,[(op,output,b'')])
  rnk=len(shape); ss=[math.prod(shape[i+1:]) for i in range(rnk)]
  result=inspect(Owner(a),output,1,shape,ss,data); _,_,st,vr,_,tick=result
  assert output not in handles and st not in {x[2] for x in before_owners} and vr not in {x[3] for x in before_owners}
  assert tick[2]==0
  mm=fetch(rd,output,n,data)
  assert mm[:3]==(rnk,n,0) and mm[3:3+rnk]==tuple(shape) and mm[7:7+rnk]==tuple(ss) and mm[11:16]==(st,vr,0,0,n)
  return n,data
 total=dict(positive=0,positive_words=0,archival_positive=0,prior_status7=0,caps=0,overcaps=0,refusals=0,sameframe=0,pressure=0,raw_frames=len(Q),successful_seeded_scratch=0)
 known={};owned_cap=V['caps'][0]
 for e in r['rows']:
  kind=e['kind'];i=e['frame']-1
  if kind=='refusal':
   b,a=map(before,(e['before'],e['after']));assert b==a and e['before']['ptr']==e['after']['ptr']
   assert R[i]==(e['expected'],[]) and e['status']==e['expected'] and sha(bq[i])==e['request_sha256'];total['refusals']+=1
   label=e['label']
   if label.startswith('parent-'):
    z=rows[label[7:]];op,x,y,*padded=Q[i][0];assert op==z['op'] and padded==[0]*5
    inputs(Owner(b),(x,y),z);total['prior_status7']+=1
   if label.startswith('cap-') and e['status']==3:
    c=V['caps'][1];op,x,y,*padded=Q[i][0];assert c['output_count']==262656 and padded==[0]*5
    inputs(Owner(b),(x,y),c);total['overcaps']+=1
   if label.startswith('late-broadcast-'):assert len(Q[i])==2 and Q[i][1][0]==65500
   continue
  if kind=='same-frame':
   z=rows[e['label']];n,data=expected(z['left'],z['right'],e['op']);n=math.prod(n)
   assert Q[i][0]==(e['op'],*e['input'],0,0,0,0,0) and Q[i][1][0:2]==(13,0xfe000001) and Q[i][2]==(98,0xfe000001,0,n,0,0,0,0)
   assert R[i][0]==0 and R[i][1][0]==(e['op'],e['output'],b'') and R[i][1][2]==(98,n,data);total['sameframe']+=1;continue
  if kind in ('cap','pressure'):
   z=dict(**owned_cap,op=e['op'])
   if kind=='cap':
    b,a=map(before,(e['before'],e['after']))
    n,d=prove_result(e,z,e['inputs'],e['output'],e['frame'],e['read'],b,a);total['caps']+=1
    assert q(b,3,128)==(2,3,4);total['successful_seeded_scratch']+=1
   else:
    b,a=map(before,(e['before'],e['after']));assert b==a and q(b,3,128)==(2,3,4)
    assert R[i]==(3,[]) and sha(bq[i])==e['request_sha256'] and e['limit_bytes']>0
    for f in e['lead']:assert bq[f-1]==bq[i] and R[f-1][0]==0
    bb,aa=map(before,(e['recovery_before'],e['recovery_after']))
    n,d=prove_result(e,z,e['inputs'],e['recovered_output'],e['recover_frame'],e['recover_read'],bb,aa)
    assert bq[e['recover_frame']-1]==bq[i] and q(bb,3,128)==(2,3,4);total['pressure']+=1
    known['pressure_receipt']=dict(request_sha256=e['request_sha256'],as_limit=e['limit_bytes'],failed_frame=e['frame'],lead=e['lead'],recovered_frame=e['recover_frame'],attempt=e['attempt'])
   k=z['result_digests'][str(z['op'])];assert n==262144 and sha(d)==k['raw_sha256'] and sum(d)==k['ones']
   continue
  assert kind=='positive';z=rows[e['label']];b,a=map(before,(e['before'],e['after']))
  n,data=prove_result(e,z,e['inputs'],e['out'],e['frame'],e['read'],b,a)
  x,y=e['inputs']; assert x!=y
  for j,(h,t) in enumerate(zip((x,y),(z['left'],z['right']),strict=True)):
   nr=len(t['words']);vb=bytes.fromhex(t['raw_hex'])
   assert fetch(e['old_reads'][j],h,nr,vb)==fetch(e['new_reads'][j],h,nr,vb)
   if len(e['surv'])==2:fetch(e['surv'][j],h,nr,vb)
  if len(e['surv'])==1:fetch(e['surv'][0],e['out'],n,data)
  w=e['weak_frame']-1;assert Q[w][0][:2]==(42,e['out']) and R[w]==(0,[(42,e['weak'],b'')])
  w=e['weak_refusal']-1;assert Q[w][0][:2]==(43,e['weak']) and R[w]==(6,[])
  total['positive']+=1;total['positive_words']+=n;total['archival_positive']+='archive_frame' in z
 if r['mode']=='candidate':
  assert tuple(total[k] for k in ('positive','positive_words','archival_positive','caps','overcaps','sameframe','pressure','successful_seeded_scratch'))==(405,8310,9,3,3,3,1,3)
 else:assert total['prior_status7']==405
 return dict(**total,**known,request_stream=sha((p/'requests.frames.bin').read_bytes()),reply_stream=sha((p/'responses.frames.bin').read_bytes()))
if __name__=='__main__':print(json.dumps({Path(x).name:analyze(Path(x)) for x in sys.argv[1:]},sort_keys=True,indent=2))
