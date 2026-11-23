"""Reversible explicit dimensioned F16/F32 and F16/BF16 FBGEMM profile pairs for one existing where opcode."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
src,checker=map(Path,sys.argv[1:3]);src=src.resolve(strict=True)
local=Path(__file__).parent
sha=lambda v:hashlib.sha256(v).hexdigest()
if not __debug__:raise ValueError('source checks require assertions')
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',checker);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(src/'manifest.json').read_bytes()
assert sha(mraw)=='c4c9ffa3c89ea7d38a5d6ce61018d3ba62077dd461e5275e1b3b4c64c3d7f703'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
r,units,_=b.check_sources(m,src,b.read_file)
assert sha(r)=='930932bc9dce73a45ce5c8e7e728e8f95071b1e2457de95048ca701e3dafd9c9'
u=next(row for row in m['units'] if row['name']=='runtime800_12');up=u['path']
rows={row['path']:raw for row,raw,_ in units}
assert sha(rows[up])=='cf8765d76cfedbb8ac73487930ad0fee8b3713f08babe9fa88e4d8e88a803809'
from_body=lambda name:re.search(rb'^fn '+name+rb'\([^\n]+\n.*?^end\n',r,re.M|re.S)[0]
def replace(raw,old,new):
 assert raw.count(old)==1
 return raw.replace(old,new,1)
def region(body,begin,end):
 p=body.index(b'block '+begin+b'(');q=body.index(b'block '+end+b'(',p)
 return body[p:q]
a=from_body(b'where_self_tensor')
k=from_body(b'where_full_where_core')
prior_a=b'bf32) bad(bps)'
new_a=b'bf32) f16_pair_tags(bps,bpm,bpx,bpy,bxt,byt,bmask_ok)'
a_new=replace(a,prior_a,new_a)
a_new=replace(a_new,b'block wide_ranks(', (local/'f16_tags.hepta.inc').read_bytes()+b'block wide_ranks(')
subs=[
(b'rsource,word,r_mix) bad(rs)',b'rsource,word,r_mix,rp) bad(rs)'),
(b'nword:u64,n_mix:bool)',b'nword:u64,n_mix:bool,np:u64)'),
(b'nsource,nword,n_mix) store',b'nsource,nword,n_mix,np) store'),
(b'pword:u64,check_mix:bool)',b'pword:u64,check_mix:bool,pp:u64)'),
(b'check_mix,is_bool,stype,otype)',b'check_mix,is_bool,stype,otype,psrc,pp)'),
(b'wsrc:u64,wtgt:u64)',b'wsrc:u64,wtgt:u64,wselected:u64,wsp:u64)'),
(b'wmode,wbool,wsrc,wtgt)',b'wmode,wbool,wsrc,wtgt,wselected,wsp)'),
(b'bcsrc:u64,bctgt:u64)',b'bcsrc:u64,bctgt:u64,bcselected:u64,bcsp:u64)'),
(b') convert_bool(bcs,bcm,bcx,bcy,bco,bcw,bcn,bci,bcword,bcmode,bcbool,bctgt)',b') f16_pair_convert(bcs,bcm,bcx,bcy,bco,bcw,bcn,bci,bcword,bcmode,bcbool,bcsrc,bctgt,bcselected,bcsp)'),
(b'block convert_bool(', (local/'f16_convert.hepta.inc').read_bytes()+b'block convert_bool('),
]
k_new=k
for old,nxt in subs:k_new=replace(k_new,old,nxt)
rev=k_new
for old,nxt in reversed(subs):rev=replace(rev,nxt,old)
assert rev==k
fbg=(local/'fbgemm.hepta.inc').read_bytes()
assert region(a_new,b'wide_ranks',b'mixed_roots')==region(a,b'wide_ranks',b'mixed_roots')
assert region(k_new,b'convert_bool',b'store')==region(k,b'convert_bool',b'store')
assert region(k_new,b'f32_to_f64',b'bf_pair_convert')==region(k,b'f32_to_f64',b'bf_pair_convert')
assert region(k_new,b'bf16_to_f32',b'f16_pair_convert')==region(k,b'bf16_to_f32',b'convert_bool')
new=replace(replace(r,a,a_new),k,fbg+k_new)
assert replace(replace(new,a_new,a),fbg+k_new,k)==r
split=replace(replace(rows[up],a,a_new),k,fbg+k_new)
head=b'fn half_widen_bf16('
extern=b'fn widen_f16(x:u64) -> (u64) effects(read,write,trap) external()\n'
split=replace(split,head,extern+head)
e0=b'exports(where_bool_pair_true_bits,'
e1=b'exports(where_f16_fbgemm_profile_word,where_bool_pair_true_bits,'
split=replace(split,e0,e1)
assert replace(replace(replace(replace(split,e1,e0),extern+head,head),a_new,a),fbg+k_new,k)==rows[up]
before=b.parse_source(r);after=b.parse_source(new)
changed={f for f in before['definitions'] if before['definitions'][f]!=after['definitions'][f]}
new_names=set(after['definitions'])-set(before['definitions'])
assert changed=={'where_self_tensor','where_full_where_core'} and new_names=={'where_f16_fbgemm_profile_word'}
assert all(before['signatures'][f]==after['signatures'][f] for f in before['signatures'])
for f in changed|new_names:
 body=after['definitions'][f]
 bindings=[]
 for line in body.splitlines()[1:]:
  if line.startswith(b'block '):bindings+=re.findall(rb'\b(\w+):\w+',line)
  else:bindings+=re.findall(rb'^  (\w+):\w+ =',line)
 assert len(bindings)==len(set(bindings)), (f,[k for k in set(bindings) if bindings.count(k)>1])
 args=lambda v:v.count(b',')+1 if v else 0
 targets={hit[1]:args(hit[2]) for hit in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
 for terminator in re.findall(rb'^  (?:branch|jump) ([^\n]+)',body,re.M):
  for hit in re.finditer(rb'(\w+)\(([^)]*)\)',terminator):
   assert hit[1] in targets and targets[hit[1]]==args(hit[2]),(f,hit[1])
rows[up]=split;rows[m['runtime']['path']]=new
m.pop('expected_native_sha256')
m['name']='opt_in_joint_where_f16_fbgemm_locked_avx512_profile_cpu'
m['runtime'].update(sha256=sha(new),**{k:after[k] for k in ('functions','blocks','instructions')})
u.pop('expected_object_sha256');u['sha256']=sha(split)
def mem(path,expected):
 data=rows[path.relative_to(src).as_posix()]
 assert sha(data)==expected
 return data
b.check_sources(m,src,mem)
if len(sys.argv)==4:
 dest=Path(sys.argv[3]);dest.mkdir();(dest/'units').mkdir()
 for k,v in rows.items():(dest/k).write_bytes(v)
 (dest/'manifest.json').write_bytes(b.json_bytes(m))
 b.check_sources(m,dest,b.read_file)
print(json.dumps(dict(parent_head='2ff996a1b2ee5168e15c82f5e708aace741ec731',runtime_sha256=sha(new),unit12_sha256=sha(split),
 manifest_sha256=sha(b.json_bytes(m)),check_sources='PASS',byte_exact_inverse=True,
 changed=sorted(changed),added=sorted(new_names),native_torch_compile_executed=False),indent=2))
