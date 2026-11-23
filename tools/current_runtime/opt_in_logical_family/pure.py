"""Independent raw request/reply/owner verifier; no Torch, producer or protocol imports."""
from pathlib import Path
import json,math,sys,zlib
D=Path(__file__).parent;sys.path.insert(0,str(D))
from read_old import Owner,frames,req,resp,q,sha
p=json.loads((D/'suite.json').read_bytes());o=json.loads((D/'oracle.json').read_bytes())
fixtures={r['label']:r for r in p['rows']}
locked={name+':'+r['label']:r for name,rows in o['rows'].items() for r in rows}
W={0:4,1:1,2:8,4:8,5:2,6:2}
dtype_widths={0:32,4:64,5:16,6:16}
def pure_expect(f):
 a,b=f['left'],f['right']; tag=a['dtype'];assert b['dtype']==tag
 x,y=[bytes.fromhex(h['raw_hex']) for h in (a,b)]
 width=W[tag];assert len(x)==len(y)==math.prod(a['shape'])*width
 mask=(1<<(dtype_widths[tag]-1))-1 if tag in dtype_widths else (1<<(width*8))-1
 u=lambda blob,i:int.from_bytes(blob[i:i+width],'little') & mask
 def gate(a,b):
  if f['op']==133:return max(a,b)
  assert f['op']==134
  return (a+b)%2
 return bytes(gate(min(1,u(x,i)),min(1,u(y,i))) for i in range(0,len(x),width))
def inspect(path):
 path=Path(path);p=json.loads((path/'report.json').read_bytes()); rawreq=list(frames(path/'requests.frames.bin',b'HTOWNR01'))
 rawrep=list(frames(path/'responses.frames.bin',b'HTOWNS01'));requests=list(map(req,rawreq));replies=list(map(resp,rawrep))
 mode=p['mode'];pins=json.loads((D/'physical-pins.json').read_bytes())
 trans=json.loads((path/'transport.json').read_bytes());assert trans['exit_code']==0 and trans['responses_before_stdin_eof']
 assert trans['executable_sha256']==p['elf_sha256']==pins[mode]
 assert p['fixture_sha256']==pins['suite'] and p['no_torch'] and len(requests)==len(replies)==p['frames']
 totals=dict(positive=0,exact_refusals=0,main_old_status4=0,words=0,true_scalars=0,input_owner_aliased=0,
             drop_inputs_first=0,drop_result_first=0,same_frame=0,controls=[])
 def snapshot(a):
  raw=zlib.decompress((path/'owners'/(a['sha256']+'.zlib')).read_bytes())
  assert sha(raw)==a['sha256'] and len(raw)==a['length'];return raw
 def read(f,handle,n,expected):
  assert requests[f-1]==[(13,handle,0,0,0,0,0,0),(98,handle,0,n,0,0,0,0)]
  st,r=replies[f-1];assert st==0 and r[0][:2]==(13,handle) and r[1]==(98,n,expected);return q(r[0][2],16)
 def inspect_node(a,h,tag,shape,strides,raw):
  d=a.node(h);rank=len(shape);k=a.row(1,h)[1]
  assert d[:2]==(rank,math.prod(shape)) and d[4:6]==(0,0) and d[29]==d[83]==0 and d[72]==tag
  assert d[12:12+rank]==tuple(shape) and d[16:16+rank]==tuple(strides)
  sk,vk=d[27:29];assert k[2:4]==(sk,vk) and k[4:6]==(0,len(raw))
  sr=a.row(2,sk)[1];vr=a.row(3,vk)[1];assert d[2]==(1<<63)+sr[2]
  assert a.storage(sk)[1:]==(len(raw),raw)
  return d,k,sk,vk,sr,vr
 seen=set()
 for z in p['rows']:
  j=z['frame']-1;kind=z['kind'];rq=requests[j];rr=replies[j]
  if kind=='refusal':
   b,c=map(snapshot,(z['pre'],z['post']))
   label=z['label'];code=z['expected'];assert z['status']==code and rr==(code,[]) and sha(rawreq[j])==z['request_sha256']
   assert b==c and z['pre']['ptr']==z['post']['ptr'];totals['exact_refusals']+=1
   if label.startswith('original-'):
    label=label[9:];f=fixtures[label]; assert label not in seen;seen.add(label)
    assert mode=='baseline' and code==4 and len(rq)==1 and rq[0][0]==f['op'] and rq[0][3:]==(0,)*5
    owner=Owner(b)
    for h,arg in zip(rq[0][1:3],(f['left'],f['right']),strict=True):
     inspect_node(owner,h,arg['dtype'],arg['shape'],arg['strides'],bytes.fromhex(arg['raw_hex']))
    assert (rq[0][1]==rq[0][2])==(f['owner_relationship']=='same_handle')
    totals['main_old_status4']+=1
   elif not label.startswith('weak-expire-'):
    totals['controls'].append(label)
    pre,label=label.split('-',1); op=int(pre[2:]);assert op in (133,134)
    if label.startswith(('transpose-','offset-','singleton-view-')):
     side=1 if label.endswith('-left') else 2
     node=Owner(b).node(rq[0][side]);assert node[83]==1
     if label.startswith('offset-'):assert node[29]==4
     if label.startswith('singleton-view-'):assert node[16:19]==(2,9,1)
    if label.startswith('grad-'):
     side=1 if label.endswith('-left') else 2
     assert Owner(b).node(rq[0][side])[4]==1
    if label=='capacity-over-limit':
     node=Owner(b).node(rq[0][1]);assert rq[0][1]==rq[0][2] and node[:2]==(2,262656) and node[12:14]==(513,512) and code==3
    if label=='later-unknown-rollback':assert len(rq)==2 and [x[0] for x in rq]==[op,65500]
   continue
  if kind=='same-frame':
   code=z['op'];result=bytes((int(code==133),));assert code in (133,134)
   assert rq==[(code,z['input'],z['input'],0,0,0,0,0),(13,0xfe000001,0,0,0,0,0,0),(98,0xfe000001,0,1,0,0,0,0)]
   assert rr[0]==0 and rr[1][0]==(code,z['output'],b'') and rr[1][1][:2]==(13,z['output']) and rr[1][2]==(98,1,result)
   totals['same_frame']+=1
   continue
  assert mode=='candidate' and kind=='positive'; label=z['label']; f=fixtures[label]; t=f['expected']
  assert label not in seen;seen.add(label)
  x,y=z['inputs'];out=z['output'];assert rq==[(f['op'],x,y,0,0,0,0,0)] and rr==(0,[(f['op'],out,b'')]) and out not in (x,y)
  a,b=map(Owner,map(snapshot,(z['pre'],z['post'])))
  prev=[]
  for i,(h,s) in enumerate(zip((x,y),(f['left'],f['right']))):
   rb=bytes.fromhex(s['raw_hex'])
   args=(h,s['dtype'],s['shape'],s['strides'],rb)
   h0=inspect_node(a,*args);h1=inspect_node(b,*args);assert h0==h1
   x0=read(z['input_reads'][i],h,len(s['words']),rb); x1=read(z['after_reads'][i],h,len(s['words']),rb);assert x0==x1
   prev.append(h0)
  same=(f['owner_relationship']=='same_handle'); assert (x==y)==same and (prev[0][2:4]==prev[1][2:4])==same
  expected=pure_expect(f);n=len(expected)
  assert expected.hex()==t['raw_hex']==locked[label]['raw_hex'] and set(expected)<={0,1}
  if same and f['op']==134:assert expected==bytes(n)
  if same and f['op']==133:assert expected.hex()==f['and_immutable_bytes']
  result=inspect_node(b,out,1,t['shape'],t['strides'],expected);_,_,sk,vk,_,vr=result
  assert sk not in {x[2] for x in prev} and vk not in {x[3] for x in prev} and vr[2]==0
  output_meta=read(z['out_read'],out,n,expected);rank=len(t['shape'])
  assert output_meta[:3]==(rank,n,0) and output_meta[3:3+rank]==tuple(t['shape']) and output_meta[7:7+rank]==tuple(t['strides'])
  assert output_meta[11:16]==(sk,vk,0,0,n)
  wk=z['weak'];assert requests[z['weak_frame']-1][0][:2]==(42,out) and replies[z['weak_frame']-1]==(0,[(42,wk,b'')])
  assert requests[z['expired']-1][0][:2]==(43,wk) and replies[z['expired']-1]==(6,[])
  if len(z['survivor'])==1:
   assert read(z['survivor'][0],out,n,expected)==output_meta;totals['drop_inputs_first']+=1
  else:
   for h,j,arg in zip((x,y),z['survivor'],(f['left'],f['right']),strict=True):read(j,h,n,bytes.fromhex(arg['raw_hex']))
   totals['drop_result_first']+=1
  totals['positive']+=1; totals['words']+=n;totals['true_scalars']+=rank==0;totals['input_owner_aliased']+=same
 assert seen==set(fixtures)
 if mode=='candidate':
  assert (totals['positive'],totals['words'],totals['true_scalars'],totals['input_owner_aliased'],totals['same_frame'])==(312,2202,48,12,2)
 else: assert totals['main_old_status4']==312
 return dict(**totals,frames=len(requests),request_stream=sha((path/'requests.frames.bin').read_bytes()),response_stream=sha((path/'responses.frames.bin').read_bytes()))
if __name__=='__main__':print(json.dumps({Path(x).name:inspect(x) for x in sys.argv[1:]},indent=2,sort_keys=True))
