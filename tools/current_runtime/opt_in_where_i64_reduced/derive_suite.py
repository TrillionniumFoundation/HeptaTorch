"""Join archived status7 requests and full rollback owners to exact new fixtures, no native."""
import hashlib,json,math,struct,sys,zlib
from pathlib import Path
D,O,H,A,dest=map(Path,sys.argv[1:])
sys.path.insert(0,str(Path(__file__).parent))
from pure import Owner,frames,req,resp,sha
K=json.loads((O/'retain-final/replay.json').read_bytes())
def source(p,n):
 if n=='i64_f32':return O/'candidate',O/'candidate'
 if n=='half_f64':return O/'retain-final'/n,H/'candidate'
 if n=='i64_f64':return O/'retain-final'/n,A/'candidate'
 return O/'retain-final'/n,A/'retain'/n
def wires(p,request):
 return list(frames(p/('requests.frames.bin' if request else 'responses.frames.bin'),b'HTOWNR01' if request else b'HTOWNS01'))
def load(p):return json.loads(p.read_bytes())
def pack(dt,words):
 fmt={1:'B',2:'q',5:'H',6:'H'}[dt]
 return struct.pack(f'<{len(words)}{fmt}',*words)
assert sha((D/'prereg.json').read_bytes())=='57b06e065d8ac10bf92f8d4133d1f525be91cd2f864bbc4d068bd8260a044502'
assert sha((D/'locked-profile.json').read_bytes())=='79cc71ff0200b2f24567c8823e703516cb581e740980509091924668115c4b36'
lock={r['label']:r for r in load(D/'locked-profile.json')['rows']}
def norm(r):
 m,x,y=r['operands']; result=dict(label=r['label'],dtype=r.get('dtype',next(a['tag'] for a in (x,y) if a['tag']!=2)),width=2,shape=r['shape'],result_hex=r['expected_hex'])
 stride=1; rev=[]
 for d in reversed(r['shape']):rev.append(stride);stride*=max(1,d)
 result['strides']=rev[::-1]
 for k,a in zip(('mask','left','right'),(m,x,y),strict=True):
  result[k+'_dtype']=a['tag']; result[k+'_shape']=a['shape']; result[k]=[n if a['tag']!=2 else n%2**64 for n in a['words']]
  result[k+'_hex']=pack(a['tag'],a['words']).hex()
 return result
P=load(D/'prereg.json')['rows']; rows=[norm(r) for r in P]
assert len(rows)==len(lock) and all(sha(bytes.fromhex(r['result_hex']))==lock[r['label']]['where_sha256'] for r in rows)
# Independently use integer grid arithmetic; only needed for new literal archived inputs.
def roundn(a,p):
 k=1<<max(0,a.bit_length()-p); q,r=divmod(a,k)
 return (q+int(2*r>k or 2*r==k and q%2))*k
def red(v,tag):
 p,b,end=(11,15,0x7c00) if tag==5 else (8,127,0x7f80)
 a=roundn(roundn(abs(v),24),p); sign=0x8000 if v<0 else 0
 if a==0:return 0
 e=a.bit_length()-1; base=(e+b)<<(p-1)
 return sign | (end if base>=end else base | (((a<<(p-1))>>e)&((1<<(p-1))-1)))
from itertools import product
def historical(ops,label):
 rank=max(map(lambda o:len(o['shape']),ops));result=[]
 for z in zip(*[[1]*(rank-len(o['shape']))+o['shape'] for o in ops],strict=True):
  q=set(z)-{1};assert len(q)<=1
  result.append(next(iter(q),1))
 tag=next(o['tag'] for o in ops[1:] if o['tag']!=2)
 def get(o,coord):
  idx=0
  for size,i in zip(o['shape'],coord[rank-len(o['shape']):],strict=True):idx=idx*size+(0 if size==1 else i)
  return o['words'][idx]
 words=[]
 for coord in product(*(range(n) for n in result)):
  selected=ops[1 if get(ops[0],coord) else 2]
  v=get(selected,coord);words.append(red(v,tag) if selected['tag']==2 else v)
 return norm(dict(label=label,shape=result,dtype=tag,operands=ops,expected_hex=pack(tag,words).hex()))
hist={};fixtures={}
for run in ('i64_f32',*K):
 p,old=source(D,run)
 assert p.exists() and old.exists()
 if run in K:
  assert sha((p/'requests.frames.bin').read_bytes())==K[run]['request_sha256'] and sha((p/'responses.frames.bin').read_bytes())==K[run]['response_sha256']
 q,r=wires(p,True),wires(p,False)
 oq,orr=wires(old,True),wires(old,False)
 for en in load(old/'report.json')['rows']:
  if en.get('kind')!='refusal' or en.get('status')!=7:continue
  label=en.get('label','')
  if label not in ('mixed-value-dtype','f16-i64-forward','f16-i64-reverse','bf16-i64-forward','bf16-i64-reverse'):continue
  i=en['frame']-1; v,b=oq[i],orr[i]
  assert resp(b)==(7,[])
  pre=zlib.decompress((old/'owners'/(en['pre']['sha256']+'.zlib')).read_bytes())
  post=zlib.decompress((old/'owners'/(en['post']['sha256']+'.zlib')).read_bytes())
  assert pre==post and sha(pre)==en['pre']['sha256']
  o=Owner(pre);cmd,=req(v);assert cmd[0]==129 and not any(cmd[4:])
  ops=[]
  for x in cmd[1:4]:
   nd=o.node(x);tag=nd[72]
   pattern={0:'i',1:'B',2:'q',4:'q',5:'H',6:'H'}[tag]
   ops.append(dict(tag=tag,shape=list(nd[12:12+nd[0]]),words=[y[0] for y in struct.iter_unpack('<'+pattern,o.storage(nd[27])[2])]))
  ts={a['tag'] for a in ops[1:]}
  if ts not in ({2,5},{2,6}):continue
  assert ops[0]['tag']==1 and all(a['shape'] for a in ops[1:])
  frames_at=[i for i,b1 in enumerate(q,1) if b1==v and r[i-1]==b]
  assert len(frames_at)==1,(run,label,frames_at)
  key=sha(json.dumps(ops,sort_keys=True,separators=(',',':')).encode())
  if key not in fixtures:fixtures[key]=historical(ops,'historical-'+key[:12])
  hist.setdefault(run,{})[frames_at[0]]=dict(label=fixtures[key]['label'],previous_label=label,owner_sha256=sha(pre),request_sha256=sha(v),response_sha256=sha(b))
rows+=list(fixtures.values())
output=dict(rows=rows,retired=hist,prior_frozen_sha256=sha((D/'prereg.json').read_bytes()),locked_measured_sha256=sha((D/'locked-profile.json').read_bytes()))
dest.write_text(json.dumps(output,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps(dict(retired_by_suite={k:len(v) for k,v in hist.items()},retired=sum(map(len,hist.values())),unique_witnesses=len(fixtures),rows=len(rows),words=sum(math.prod(r['shape']) for r in rows))))
