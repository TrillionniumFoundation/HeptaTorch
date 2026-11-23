"""Read archived contexts only. Request bytes can alias across independent sessions."""
from pathlib import Path
import hashlib,sys,json,struct
d,stage,p=map(Path,sys.argv[1:]);f=json.loads((d/'prereg.json').read_bytes())
ids={x['recipe']['request_sha256']:(x['label'],x['recipe']['response_sha256']) for x in f['cases'] if x['label'].startswith('historical')}
sha=lambda b:hashlib.sha256(b).hexdigest()
ss=[('retained/'+z.name,z) for z in sorted(stage.iterdir()) if z.is_dir()]+[('four/'+i,p/i) for i in ('new','supplement','supplement-tail')]
def frames(x,magic):
 i=0
 while i<len(x):
  key,n=struct.unpack_from('<8sQ',x,i); assert key==magic and 32<=n<=1<<20 and i+16+n<=len(x)
  i+=16;yield x[i:i+n];i+=n
 assert i==len(x)
wanted={'retained/i64':[657,667,672],'retained/new':[577,587,592],'retained/witness-new':[657,667,672],
 'retained/witness-old':[657,667,672],'four/new':[1945,1955]}
items=[];same_request_different_context=[]
for stream,path in ss:
 q=(path/'requests.frames.bin').read_bytes();r=(path/'responses.frames.bin').read_bytes();seen=[]
 for at,(u,v) in enumerate(zip(frames(q,b'HTOWNR01'),frames(r,b'HTOWNS01'),strict=True),1):
  h=sha(u)
  if h not in ids:continue
  label,pin=ids[h]; rs=struct.unpack_from('<Q',v,24)[0]; record=dict(label=label,frame=at,request_sha256=h,response_sha256=sha(v),old_status=rs)
  if at in wanted.get(stream,()):
   assert rs==7 and sha(v)==pin;seen.append(record)
  else:
   assert stream=='four/new' and at in (679,728,784) and rs==0 and sha(v)!=pin
   same_request_different_context.append(dict(stream=stream,**record))
 assert [i['frame'] for i in seen]==wanted.get(stream,[])
 if seen:items.append(dict(stream=stream,full_q_sha=sha(q),full_r_sha=sha(r),matches=seen))
assert {z['request_sha256'] for s in items for z in s['matches']}==set(ids)
result=dict(matched_physical_occurrences=sum(len(x['matches']) for x in items),unique_requests=len(ids),stream_count=len(ss),
 matching_streams=items,do_not_retire_positive_context_aliases=same_request_different_context)
assert result['matched_physical_occurrences']==14 and len(same_request_different_context)==3
raw=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
(d/'historical-occurrences.json').write_bytes(raw)
print(sha(raw),result['matched_physical_occurrences'],[(r['stream'],len(r['matches'])) for r in items])
