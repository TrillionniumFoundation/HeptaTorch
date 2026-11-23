from pathlib import Path
import hashlib,json
D=Path(__file__).parent; sha=lambda b:hashlib.sha256(b).hexdigest()
f=(D/'preregister.json').read_bytes();o=(D/'oracle.json').read_bytes()
assert sha(f)=='299e9a2fc982c4a50d0b023bd0378f7dd6b1c7aaf24a9ea7bd60627173b7614e'
assert sha(o)=='8f40b657800b6074a499175d7c6b00dcddc840b09de1bdbe7bdd174261442e8d'
a,c=map(json.loads,(f,o));assert c['discrepancies']==[] and c['truths_against_historical_and']==156
rows=[]; os={133:'logical_or',134:'logical_xor'}
for code,name in os.items():
 ref={r['label']:r for r in c['rows'][name]}
 for r in a['strict_rows']:
  x=ref[r['label']];h=r['outputs'][name];s=r['output_shape'];stride=r['output_strides']
  words=list(bytes.fromhex(h));assert h==x['raw_hex'] and s==x['shape'] and stride==x['strides'] and len(words)==x['numel']
  rows.append(dict(label=name+':'+r['label'],operand_label=r['label'],owner_relationship=r['relation'],left=r['left'],right=r['right'],op=code,
      expected=dict(dtype=1,shape=s,strides=stride,words=words,raw_hex=h),and_immutable_bytes=r['outputs']['logical_and']))
assert len(rows)==312 and sum(len(x['expected']['words']) for x in rows)==2202
(D/'suite.json').write_text(json.dumps(dict(rows=rows,scope_controls=a['scope_controls'],physical_contrasts=a['physical_contrasts'],
   preregister_sha256=sha(f),oracle_sha256=sha(o)),sort_keys=True,separators=(',',':'))+'\n')
