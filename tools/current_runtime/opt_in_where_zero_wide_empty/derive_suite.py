import hashlib,json,math,sys
from pathlib import Path
D=Path(__file__).parent; P=Path('/tmp/hepta-where-zero-wide-prereg')
sha=lambda b:hashlib.sha256(b).hexdigest()
sys.path.insert(0,str(P));from preregister import array
r=json.loads((D/'preregister.json').read_bytes());o=json.loads((D/'oracle-receipt.json').read_bytes())
m={s['label']:s for s in o['results']+o['old_witnesses']}
old={s['label']:s for s in json.loads((P/'expected.json').read_bytes())['positive']}
rows=[]
for q in r['empty']+r['known']+r['shape_errors']:
 k=q['label']
 if k in old:
  operands=[old[k][a] for a in ('mask','x','y')]
 else:
  operands=[array(1,q['mask_shape'],q['mask_words'])]
  for n in ('left','right'):
   t=q[n+'_dtype'];operands.append(array(t,q[n+'_shape'],q['f64_words'] if t==4 else [q['scalar_word']]))
 case=dict(label=k,operands=operands,classification='incompatible')
 if k in m:
  rec=m[k]; p=q['prediction'];assert p['shape']==rec['shape'] and p['strides']==rec['strides']
  case.update(classification='canonical' if rec['contiguous_branch'] else 'explicit',shape=rec['shape'],strides=rec['strides'],inherited=k in old)
 rows.append(case)
assert len(rows)==192
(D/'suite.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
