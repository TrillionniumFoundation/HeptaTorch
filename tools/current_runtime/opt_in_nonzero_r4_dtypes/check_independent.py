"""Independent stdlib readback for the numeric rank4 nonzero prediction."""
import hashlib, json, struct, sys
from pathlib import Path
import freeze

h=lambda b:hashlib.sha256(b).hexdigest()
fmt={'bool':'B','int64':'q','float16':'H','bfloat16':'H','float32':'I','float64':'Q'}
zeros={'float16':(0,0x8000),'bfloat16':(0,0x8000),'float32':(0,0x80000000),
       'float64':(0,0x8000000000000000),'int64':(0,),'bool':(0,)}
data=Path(sys.argv[1]).read_bytes()
f=json.loads(data); count=0
for c in f['cases']:
 dt=c['input_dtype'];a,b,cc,d=c['input_shape']
 raw=freeze.source(dt,c['input_shape'],c['recipe'])
 assert h(raw)==c['input_raw_sha256']
 cols=[b'',b'',b'',b'']; ls=b''; selected=0
 for i,(word,) in enumerate(struct.iter_unpack('<'+fmt[dt],raw)):
  if word in zeros[dt]:continue
  assert min(a,b,cc,d)>0
  vals=(i//(b*cc*d),i//(cc*d)%b,i//d%cc,i%d)
  ls+=struct.pack('<4q',*vals)
  for j,item in enumerate(vals):
   cols[j]+=struct.pack('<q',item)
  selected+=1
 m=c['expected_torch']
 assert selected==c['k'] and m['shape']==[selected,4] and m['strides']==[1,selected]
 assert h(ls)==m['logical_sha256'] and h(b''.join(cols))==m['physical_sha256']
 assert [h(col) for col in cols]==m['column_sha256']
 assert len(ls)==m['storage_bytes']==32*selected
 count+=4*selected
assert count==f['summary']['words_each_order']
print(h(data),len(f['cases']),count)
