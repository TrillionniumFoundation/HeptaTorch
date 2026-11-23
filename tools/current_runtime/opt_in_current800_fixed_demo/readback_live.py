"""Read back the single successful guarded demo record, without running a target."""
import ast, hashlib, json, re, struct
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda d:hashlib.sha256(d).hexdigest()
program=(P/'run_demo.py').read_bytes()
assert sha(program)=='075c79075e1d6c0229160e3fade252ee20e4cd00a19a166de690a13d3977949f'
tree=ast.parse(program)
im=set()
for n in ast.walk(tree):
 if isinstance(n,ast.Import):im.update(s.name for s in n.names)
 if isinstance(n,ast.ImportFrom):im.add(n.module)
assert im=={'argparse','hashlib','json','os','pathlib','select','stat','struct','subprocess','tempfile','time'}
assert not any(bad in program for bad in (b'/proc',b'import torch',b'owner_client',b'shell=True'))
text=(P/'live'/'guard-and-output.log').read_text()
m=re.fullmatch(r'before \((\d+), (\d+), (\d+)\)\n(\{.*\})\nafter \((\d+), (\d+), (\d+)\) min_sampled_mem (\d+)\n',text,re.S)
assert m
start,tmp,disk,end,etmp,edisk,minimum=map(int,m.groups()[:3]+m.groups()[4:])
assert start>=6500000000 and minimum>=5700000000 and min(tmp,etmp)>=10**9 and min(disk,edisk)>=950*1024**2
observed=json.loads(m[4]);assert observed['binary_sha256']=='5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d'
inf=json.loads((P/'inference.json').read_bytes());s=json.loads((P/'sgd.json').read_bytes())
assert inf['prereg_sha256']=='d54fa50ff0a9aaf839d38a1c69946cb6582cec48e388fa130fc2dec72ee34f73'
def words(e):
 b=bytes.fromhex(e['raw_f32le_hex'])
 return list(struct.unpack('<'+'f'*(len(b)//4),b))
for row,record in zip(inf['admitted'],observed['inference'],strict=True):
 f=row['original_frozen'];t=f['stages'][-1]['expected']
 assert record==dict(case=f['label'],frames=128,prediction=words(t),prediction_shape=t['shape'],verified_stages=5)
assert observed['sgd']==dict(frames=128,gradient=words(s['backward']['expected_gradient']),
 loss_before=words(s['before_update'][-1]['expected'])[0],loss_after=words(s['after_update'][-1]['expected'])[0],
 weight=words(s['update']['weight_after']),weight_version=[0,1])
raw=(json.dumps(observed,sort_keys=True,indent=2)+'\n').encode()
assert raw==m[4].encode()+b'\n'
(P/'live'/'observed.json').write_bytes(raw)
r=dict(status='PASS_SINGLE_GUARDED_STD_LIB_BOTH',binary_sha256=observed['binary_sha256'],
 script_sha256=sha(program),log_sha256=sha(text.encode()),deterministic_output_sha256=sha(raw),
 frames=512,sessions=4,inference_stages=15,sgd_steps=1,memory_start_bytes=start,memory_min_bytes=minimum,
 ordinary_only=True,no_owner_inspection=True)
(P/'live'/'receipt.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
print(r)
