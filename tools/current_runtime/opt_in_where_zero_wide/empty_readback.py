"""Strictly report observed empty metadata differences, with raw on-disk frame readback."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from pure import frames,q,resp,sha
d=Path(__file__).parent
p=d/'candidate'; run=json.loads((p/'report.json').read_bytes())
f={x['label']:x for x in json.loads((d/'suite.json').read_bytes())['rows']}
rs=list(map(resp,frames(p/'responses.frames.bin',b'HTOWNS01')))
rows=[]
for src in run['rows']:
 if src['kind']!='positive':continue
 x=f[src['label']]
 if 0 not in x['shape']:continue
 st,((op,h,metadata),(rd,n,blob))=rs[src['read_result']-1]
 fields=q(metadata,16); rank,count,grad=fields[:3]
 shape=list(fields[3:3+rank]);strides=list(fields[7:7+rank])
 assert st==grad==count==n==0 and op==13 and rd==98 and h==src['handles'][3] and not blob
 assert shape==x['shape'] and fields[14:16]==(0,0)
 rows.append(dict(label=src['label'],mask=x['mask_shape'],left=x['left_shape'],right=x['right_shape'],shape=shape,
                  native_strides=strides,torch_strides=x['torch_strides'],differs=strides!=x['torch_strides'],raw_payload_bytes=0,
                  native_frame=src['read_result']))
assert len(rows)==16 and sum(x['differs'] for x in rows)==8
proof=dict(qualifies_for_full_raw_metadata=False,empty_cases=rows,
 native_stream_sha256=sha((p/'responses.frames.bin').read_bytes()))
print(json.dumps(proof,indent=2,sort_keys=True))
