import hashlib,json,math
from pathlib import Path
D=Path(__file__).parent
p=json.loads((D/'expected.json').read_bytes());h=json.loads((D/'historical.json').read_bytes());r=json.loads((D/'oracle-receipt.json').read_bytes())
oracle={a['label']:a for a in r['where']}
def flat(f):
    result=f['result']; o=oracle[f['label']]
    assert hashlib.sha256(bytes.fromhex(result['byte_hex'])).hexdigest()==o['result_sha256']
    out=dict(label=f['label'],dtype=4,width=8,shape=result['shape'],result_hex=result['byte_hex'],torch_strides=o['stride'],strides=[math.prod(max(1,v) for v in result['shape'][i+1:]) for i in range(len(result['shape']))])
    for x,k in (('mask','mask'),('left','x'),('right','y')):
        a=f[k];out[x]=a['words'];out[x+'_hex']=a['byte_hex'];out[x+'_shape']=a['shape'];out[x+'_dtype']=a['dtype']
    return out
rows=list(map(flat,p['positive']+[x['expected_if_admitted'] for x in h['retiring_candidates']]))
controls=p['policy_negative']+h['retained_policy_candidates']
assert len(rows)==184 and len(controls)==32
(D/'suite.json').write_text(json.dumps(dict(rows=rows,controls=controls,historical=h),sort_keys=True,separators=(',',':'))+'\n')
