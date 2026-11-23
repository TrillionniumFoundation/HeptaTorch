"""Reversible exact Bool/I64 value-pair where route, source-only by default."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
P=Path(__file__).parent
h=lambda b:hashlib.sha256(b).hexdigest()
if not __debug__:raise ValueError('use default Python assertions')
assert h(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('builder',B);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(S/'manifest.json').read_bytes()
assert h(mraw)=='02be94c40e2da1e46248264dbb71f9afd55c39687dffaadc3f208904962631d9'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
runtime,units,_=b.check_sources(m,S,b.read_file)
assert h(runtime)=='a99ffaa8bcfee978016cf6a379a04b1e4ddf9540f4d84f79769d33c520340bf5'
u=next(row for row in m['units'] if row['name']=='runtime800_12');up=u['path']
rows={row['path']:raw for row,raw,_ in units}
assert h(rows[up])=='feeac6ad423af873add28dbdf6da5eb221f0794c0a7a55bbd20fffb73d22b635'
from_body=lambda name:re.search(rb'^fn '+name+rb'\([^\n]+\n.*?^end\n',runtime,re.M|re.S)[0]
def sub(raw,old,new):
 assert raw.count(old)==1
 return raw.replace(old,new,1)
admit=from_body(b'where_self_tensor')
new_admit=sub(admit,b'branch valid_tags roots(s,m,x,y,xt,bool_tag) bad(s)',
 b'branch valid_tags roots(s,m,x,y,xt,bool_tag) mixed_tags(s,m,x,y,xt,yt,mask_bool)')
new_admit=sub(new_admit,b'block roots(', (P/'mixed.hepta.inc').read_bytes()+b'block roots(')
old_full=from_body(b'where_full_where')
hdr=old_full.split(b'\n',1)[0]
wrapper=b'''fn where_full_where(s:mutbuf,m:u64,x:u64,y:u64,t:u64,w:u64) -> (u64) effects(read,write,trap) entry(start)
block start(s:mutbuf,m:u64,x:u64,y:u64,t:u64,w:u64)
  zero:u64 = const() {"value":0}
  not_mixed:bool = compare.ne(zero,zero)
  result:u64 = call(s,m,x,y,t,w,not_mixed) {"function":"where_full_where_core"}
  ret(result)
end
'''
core=sub(old_full,hdr,hdr.replace(b'where_full_where(',b'where_full_where_core(').replace(b'w:u64)',b'w:u64,plan_mix:bool)'))
# Existing blocks/variable prefixes in order, including the loop backedge.
modes={
 b'plan':b'plan_mix', b'prove':b'p_mix', b'capacity':b'c_mix', b'allocate':b'a_mix',
 b'loop':b'l_mix',b'maskpos':b'm_mix',b'maskword':b'w_mix',b'sourcepos':b's_mix',
 b'value':b'r_mix',b'store':b't_mix',
}
current=None
lines=[]
for line in core.splitlines(keepends=True):
 if line.startswith(b'block '):
  current=line[6:line.index(b'(')]
  if current in modes:
   assert line.endswith(b')\n')
   line=line[:-2]+b','+modes[current]+b':bool)\n'
 elif line.startswith((b'  branch ',b'  jump ')) and current in modes:
  line=re.sub(rb'\b([a-z]+)\(([^()]*)\)',lambda z:(z[1]+b'('+z[2]+b','+modes[current]+b')') if z[1] in modes else z[0],line)
 lines.append(line)
core=b''.join(lines)
core=sub(core,b'  branch read_ok store(rs,rm,rx,ry,ro,rw,rn,ri,word,r_mix) bad(rs)\n',(P/'value.hepta.inc').read_bytes())
new_full=wrapper+core
agg=sub(sub(runtime,admit,new_admit),old_full,new_full)
assert sub(sub(agg,new_admit,admit),new_full,old_full)==runtime
split=sub(sub(rows[up],admit,new_admit),old_full,new_full)
e0=b'exports(where_full_aligned,'
e1=b'exports(where_full_where_core,where_full_aligned,'
split=sub(split,e0,e1)
assert sub(sub(sub(split,e1,e0),new_admit,admit),new_full,old_full)==rows[up]
before=b.parse_source(runtime);after=b.parse_source(agg)
assert all(before['signatures'][k]==after['signatures'][k] for k in before['signatures'])
assert set(after['definitions'])-set(before['definitions'])=={'where_full_where_core'}
changed={k for k in before['definitions'] if before['definitions'][k]!=after['definitions'][k]}
assert changed=={'where_self_tensor','where_full_where'}
for name in changed|{'where_full_where_core'}:
 body=after['definitions'][name]
 bindings=[]
 for line in body.splitlines()[1:]:
  if line.startswith(b'block '):bindings+=re.findall(rb'\b(\w+):\w+',line)
  else:bindings+=re.findall(rb'^  (\w+):\w+ =',line)
 assert len(bindings)==len(set(bindings)), (name,[k for k in set(bindings) if bindings.count(k)>1])
 targets={hit[1]:hit[2].count(b',')+1 if hit[2] else 0
      for hit in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
 for terminator in re.findall(rb'^  (?:branch|jump) ([^\n]+)',body,re.M):
  for hit in re.finditer(rb'(\w+)\(([^)]*)\)',terminator):
   assert hit[1] in targets and targets[hit[1]]==(hit[2].count(b',')+1 if hit[2] else 0),(name,hit[1])
rows[up]=split;rows[m['runtime']['path']]=agg
m.pop('expected_native_sha256')
m['name']='opt_in_joint_where_explicit_bool_i64_pair_cpu'
m['runtime'].update(sha256=h(agg),**{k:after[k] for k in ('functions','blocks','instructions')})
u.pop('expected_object_sha256');u['sha256']=h(split)
def mem(path,expected):
 raw=rows[path.relative_to(S).as_posix()]
 assert h(raw)==expected
 return raw
b.check_sources(m,S,mem)
if len(sys.argv)==4:
 D=Path(sys.argv[3]);D.mkdir();(D/'units').mkdir()
 for k,v in rows.items():(D/k).write_bytes(v)
 (D/'manifest.json').write_bytes(b.json_bytes(m))
 b.check_sources(m,D,b.read_file)
print(json.dumps(dict(parent_head='95347ab4031551a386408dadb171d1d6a5f2955e',runtime_sha256=h(agg),unit12_sha256=h(split),
 manifest_sha256=h(b.json_bytes(m)),check_sources='PASS',byte_exact_inverse=True,
 changed=sorted(changed),added=['where_full_where_core'],native_torch_compile_executed=False),indent=2))
