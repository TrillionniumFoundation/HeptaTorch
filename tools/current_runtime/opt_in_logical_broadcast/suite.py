"""Stdlib schedule construction from exact frozen preflight and archived readbacks."""
from pathlib import Path
import hashlib,json
D=Path(__file__).parent;sha=lambda b:hashlib.sha256(b).hexdigest()
f=(D/'preregister.json').read_bytes();o=(D/'oracle.json').read_bytes()
assert sha(f)=='c343761909fb44e3fd568fa8d192b8950b3e323e756e612bf543e9b69da2e81b'
assert sha(o)=='bb0e0afab0ca9ff0b4308fb542804709c98b52edccfc0dda761112d3203a35f1'
a,b=map(json.loads,(f,o));assert b['metadata_or_order_discrepancies']==[]
ref={(x['label'],x['op']):x for x in b['rows']};archref={x['frame']:x for x in b['archive']}
rows=[]
def case(label,op,x,y,e,r):
 n=e['numel']; raw=e['raw_hex']
 assert r['raw_hex']==r['physical_hex']==raw and r['shape']==e['shape'] and r['strides']==e['strides'] and n==r['n']
 return dict(label=label,op=int(op),left=x,right=y,output=e,owner_relationship='distinct')
for x in a['rows']:
 for k,v in x['expect'].items():rows.append(case(x['label']+':'+k,k,x['left'],x['right'],v,ref[x['label'],k]))
for x in a['exact_archived_refusals']:
 r=archref[x['frame']]
 assert r['source_owner_sha256']==x['owner_sha256'] and r['source_request_sha256']==x['request_sha256']
 if x['after_status']==0:
  rows.append(dict(**case('exact-old-'+str(x['frame']),x['op'],*x['operands'],x['prediction'],r),
    archive_frame=x['frame'],archive_owner_sha256=x['owner_sha256'],archive_q=x['request_sha256']))
 else:assert r['outcome']=='upstream-shape-error'
assert len(ref)==396 and len(rows)==405 and sum(x['output']['numel'] for x in rows)==8310
out=dict(rows=rows,controls=a['geometry_refusals'],caps=a['capacity'],old=a['exact_archived_refusals'],locked_caps=b['capacity'],prereg=sha(f),oracle=sha(o))
z=(json.dumps(out,sort_keys=True,separators=(',',':'))+'\n').encode();(D/'suite.json').write_bytes(z)
print(len(rows),8310,sha(z))
