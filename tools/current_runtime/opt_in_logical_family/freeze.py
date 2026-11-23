"""Detach two new logical proofs, with immutable independently qualified AND witnesses."""
from pathlib import Path
import hashlib,json,struct,math,itertools
D=Path(__file__).parent
main=Path('/tmp/hepta-logical-and-prereg')
sha=lambda b:hashlib.sha256(b).hexdigest()
a=(main/'preregister.json').read_bytes();o=(main/'oracle.json').read_bytes()
assert sha(a)=='20d3c5f5383e2ca80cfab1f26884e6ae76031b18c804ed15b86594d5321e735c'
assert sha(o)=='c88ded5784f1fbd0e8db86bb48582bb5b5eef2d723217d0cfa0cd7a6f7f1561d'
a,o=map(json.loads,(a,o))
and_read={r['label']:r for r in o['rows']}
assert len(a['rows'])==len(and_read)==156
width={0:4,1:1,2:8,4:8,5:2,6:2}
ieee={0:(8,23),4:(11,52),5:(5,10),6:(8,7)}
def classify(t,x):
 if t not in ieee:return bool(x)
 e,m=ieee[t]; fraction=x%2**m;exponent=(x//2**m)%2**e
 return bool(fraction or exponent)
def raw_check(f):
 t=f['left']['dtype'];w=width[t]; mask=(1<<(8*w-(t in ieee)))-1
 fmt={1:'<B',2:'<H',4:'<I',8:'<Q'}[w]
 x,y=[tuple(z[0] for z in struct.iter_unpack(fmt,bytes.fromhex(side['raw_hex']))) for side in (f['left'],f['right'])]
 assert len(x)==len(y)==math.prod(f['left']['shape']) and list(x)==f['left']['words'] and list(y)==f['right']['words']
 coords=itertools.product(*(range(n) for n in f['left']['shape']))
 key=lambda z:bool(z&mask)
 return [(key(v),key(u)) for _,v,u in zip(coords,x,y,strict=True)]
rows=[]; same=0;nan_same=0
for f in a['rows']:
 x,y=f['left'],f['right'];t=x['dtype'];shape=x['shape'];stride=x['strides']
 assert len(shape)<=4 and x['shape']==y['shape'] and len(x['words'])==len(y['words'])==math.prod(shape)>0 and x['dtype']==y['dtype']
 assert stride==y['strides']
 pairs=[(classify(t,v),classify(t,u)) for v,u in zip(x['words'],y['words'],strict=True)]
 ref=raw_check(f); assert pairs==ref
 oracle=f['expected']['raw_hex']; old=and_read[f['label']]
 assert oracle==old['raw_hex'] and shape==old['shape'] and stride==old['strides']
 both=bytes(u and v for u,v in pairs)
 either=bytes(u or v for u,v in pairs); unequal=bytes(u!=v for u,v in pairs)
 assert both.hex()==oracle
 assert either==bytes(u|v for u,v in ref) and unequal==bytes((u+v)%2 for u,v in ref)
 assert all(c==(a|b) and not(a&b) for a,b,c in zip(both,unequal,either,strict=True))
 is_same=f['owner_relationship']=='same_handle'
 if is_same:
  same+=1; assert x==y and unequal==bytes(len(unequal)) and either==both
  if any('nan' in z for z in x['raw_classes']): nan_same+=1;assert any(either)
 rows.append(dict(label=f['label'],left=x,right=y,output_shape=shape,output_strides=stride,output_dtype=1,output_width=1,
                  outputs=dict(logical_and=both.hex(),logical_or=either.hex(),logical_xor=unequal.hex()),relation=f['owner_relationship']))
assert same==6 and nan_same==4
b=dict(stage='frozen-stdlib-no-measurement',reference_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 source_basis_combined_sha256='72ae75a5e6c10b06c6716cb607b453c6bd2b1214b6ccc6e9a7fe1a3a666c023a',target_wire_opcodes=None,
 ops=['aten.logical_or.default','aten.logical_xor.default'],scope=a['native_scope'],strict_rows=rows,
 scope_controls=a['scope_controls'],physical_contrasts=a['physical_contrasts'],
 ownership='Shared inputs allowed; output owner/storage/version independent, version0, exact Bool bytes. Even signed q/sNaN compared with same raw input gives XOR false.',
 historical_and='Keep exact AND wire/native/owner evidence. Re-run complete existing 131 stream, including original rollback, except separately qualify any exact old unused-wire probes that become newly assigned. No default changes.',
 same_input_cases_per_operator=same,same_input_nan_cases_per_operator=nan_same)
raw=(json.dumps(b,sort_keys=True,separators=(',',':'))+'\n').encode();assert len(raw)<450*1024
(D/'preregister.json').write_bytes(raw)
print(len(rows),sum(len(r['outputs']['logical_or'])//2 for r in rows),same,nan_same,len(raw),sha(raw))
