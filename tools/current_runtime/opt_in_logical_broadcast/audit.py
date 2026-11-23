"""Independent syntax, typed calls/edges and exhaustive Boolean selector audit."""
from pathlib import Path
import hashlib,re,json,sys
D=Path(__file__).parent;sha=lambda b:hashlib.sha256(b).hexdigest()
old=Path(sys.argv[1]).read_bytes();raw=(D/'source/runtime.hepta').read_bytes()
lexer=Path(sys.argv[2]).read_bytes();assert sha(lexer)=='27ca2a8a6da366214cd4d611f0c1aeffff993f325db87b158ca67f2916f96cfa'
assert sha(old)=='e37dd3c347cb6143a53d283c39c9e2ff637640d568c8347095117c43b9ab6144'
assert sha(raw)=='a9a5d93a97709069d57109aa073775c6748f53a26aa3b7eb51b92b55f02658bc'
s=lexer.decode();section=lambda n:s.split('def '+n+'(',1)[1].split('\ndef ',1)[0]
words=set();n=0
for l in section('keyword').splitlines():
 if q:=re.search(r'if size == (\d+)',l):n=int(q[1])
 if q:=re.search(r'choose\((.*?), (\d+), result\)',l):
  z={v:int(x) for v,x in re.findall(r'([abcd]) == (\d+)',q[1])}
  words.add(b''.join(z.get(k,0).to_bytes(8,'little') for k in 'abcd')[:n])
rules=set()
for n in ('reserved_function','export_loader_hook'):
 for q in re.finditer(r'match\(s, begin, size, (\d+), (\d+), (\d+), (\d+), (\d+)\)',section(n)):
  size,*v=map(int,q.groups());rules.add(b''.join(x.to_bytes(8,'little') for x in v)[:size])
assert len(words)==99
def funcs(x):
 return {m[1]:m for m in re.finditer(rb'^fn (\w+)\(([^\n]*)\) -> \((\w+)\) effects\([^\n]*?\) entry\((\w+)\)\n(.*?)^end$',x,re.M|re.S)}
before=funcs(old);after=funcs(raw)
args=lambda x:x.split(b',') if x else []
def form(x):return [tuple(z.split(b':')) for z in args(x)]
def blocks(m):
 return {b[1]:(form(b[2]),b[3]) for b in re.finditer(rb'^block (\w+)\(([^\n]*)\)\n(.*?)(?=^block |\Z)',m[5],re.M|re.S)}
def terms(body):return [m for l in body.splitlines() if l.startswith((b'  branch ',b'  jump ')) for m in re.finditer(rb'(\w+)\(([^)]*)\)',l)]
counts=dict(blocks=0,direct_calls=0,typed_edges=0)
for f in ([b'la_tensor']+[k for k in after if k.startswith(b'la_bc_')]):
 m=after[f];bs=blocks(m);assert form(m[2])==bs[m[4]][0]
 env={}
 for _,(f0,b0) in bs.items():
  for k,t in f0+[(m[1],m[2]) for m in re.finditer(rb'^  (\w+):(\w+) =',b0,re.M)]:
   assert k not in env and k not in words and len(k)<=128 and not k.startswith(b'__hia_');env[k]=t
 for bn,(f0,b0) in bs.items():
  assert bn not in words,(f,bn);counts['blocks']+=1
  for q in terms(b0):
   assert [env[v] for v in args(q[2])]==[t for _,t in bs[q[1]][0]]
   counts['typed_edges']+=1
  for l in b0.splitlines():
   if m0:=re.fullmatch(rb'  (\w+):(\w+) = call\(([^)]*)\) \{"function":"(\w+)"\}',l):
    a=after[m0[4]];assert m0[2]==a[3] and [env[k] for k in args(m0[3])]==[t for _,t in form(a[2])],l
    counts['direct_calls']+=1
  assert b'_ = call(' not in b0
 assert f==b'entry' or f not in rules

assert before[b'entry'][0]==after[b'entry'][0] and before[b'la_command'][0]==after[b'la_command'][0]
t0=blocks(before[b'la_tensor']);t1=blocks(after[b'la_tensor'])
for n in (b'la_admit',b'la_capacity',b'la_allocate',b'la_output',b'la_bad',b'la_too_big'):
 assert t0[n]==t1[n]
assert b'la_bc_index(' in t1[b'la_read'][1] and b'la_values(lrd_s,lrd_x,lrd_y,lrd_t,lrd_out,lrd_n,lrd_i,lrd_i,lrd_i,' in t1[b'la_read'][1]
assert t1[b'la_bc_index'][1].count(b'{"function":"bindex"}')==2
core=blocks(after[b'la_bc_plan'])
plan_entry=core[b'lbf_bl_plan'][1]
assert plan_entry.count(b'{"function":"la_bc_extent"}')==4
assert b'branch bl_ok lbf_bl_cap(' in plan_entry and b') lbf_bl_bad(bl_s)' in plan_entry
assert core[b'lbf_bl_cap'][1].count(b'{"function":"la_bc_product"}')==3
assert b'{"value":262144}' in core[b'lbf_bl_cap'][1]
assert b'branch bc_ok lbf_bl_make(' in core[b'lbf_bl_cap'][1] and b') lbf_bl_too_large(bc_s)' in core[b'lbf_bl_cap'][1]
assert after[b'la_bc_plan'][5].count(b'{"function":"putw"}')==0
scratch=blocks(after[b'la_bc_alloc'])[b'lbf_stage'][1]
p,q=scratch.split(b'{"function":"typed_descriptor"}',1)
for i in range(4):
 z=str(i).encode()
 assert b'call(s,p'+z+b') {"function":"getw"}' in p
 assert b'call(s,p'+z+b',d'+z+b') {"function":"putw"}' in p
 assert b'call(s,p'+z+b',old'+z+b') {"function":"putw"}' in q
assert b'branch made lbf_verify(' in q and q.index(b'restore3:u64')<q.index(b'branch made lbf_verify')
assert b'{"function":"la_canonical"}' in blocks(after[b'la_bc_alloc'])[b'lbf_canonical_check'][1]
assert all(before[n][0]==after[n][0] for n in (b'bindex',b'la_ordinary',b'la_owned_layout',b'la_canonical',b'typed_descriptor'))
# Test pure shape bound algebra with representative limits including an incompatible axis later.
S=262145
for x in (1,2,255,512,513,262144):
 for y in (1,2,255,512,513,262144):
  for z in (1,9,513):
   for w in (1,7,513):
    assert min(S,min(S,x*y)*min(S,z*w))==min(S,x*y*z*w)
print(json.dumps(dict(**counts,source_sha256=sha(raw),stage='source-only',native_keyword_count=len(words),lexer_sha256=sha(lexer),
 discarded_scalar_calls=0,unchanged_entry_and_cursor=True,both_scratch_paths_restore_all_four=True,
 numeric_cap_product_algebra='PASS',native_or_compiler_executed=False),sort_keys=True,indent=2))
