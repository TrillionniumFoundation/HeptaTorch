"""Independent syntax, typed calls/edges and exhaustive Boolean selector audit."""
from pathlib import Path
import hashlib,re,json,sys
D=Path(__file__).parent;sha=lambda b:hashlib.sha256(b).hexdigest()
old=Path(sys.argv[1]).read_bytes();raw=(D/'source/runtime.hepta').read_bytes()
lexer=Path(sys.argv[2]).read_bytes();assert sha(lexer)=='27ca2a8a6da366214cd4d611f0c1aeffff993f325db87b158ca67f2916f96cfa'
assert sha(old)=='72ae75a5e6c10b06c6716cb607b453c6bd2b1214b6ccc6e9a7fe1a3a666c023a'
assert sha(raw)=='e37dd3c347cb6143a53d283c39c9e2ff637640d568c8347095117c43b9ab6144'
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
for f in (b'entry',b'la_tensor',b'la_command'):
 m=after[f];bs=blocks(m);bs0=blocks(before[f]);assert form(m[2])==bs[m[4]][0]
 assert not (set(bs)-set(bs0))
 env={}
 for _,(f0,b0) in bs.items():
  for k,t in f0+[(m[1],m[2]) for m in re.finditer(rb'^  (\w+):(\w+) =',b0,re.M)]:
   assert k not in env and k not in words and len(k)<=128 and not k.startswith(b'__hia_');env[k]=t
 for bn,(f0,b0) in bs.items():
  if f==b'entry' and bn!=b'b12' and not bn.startswith(b'la_'):continue
  assert bn not in words;counts['blocks']+=1
  for q in terms(b0):
   assert [env[v] for v in args(q[2])]==[t for _,t in bs[q[1]][0]]
   counts['typed_edges']+=1
  for l in b0.splitlines():
   if m0:=re.fullmatch(rb'  (\w+):(\w+) = call\(([^)]*)\) \{"function":"(\w+)"\}',l):
    a=after[m0[4]];assert m0[2]==a[3] and [env[k] for k in args(m0[3])]==[t for _,t in form(a[2])],l
    counts['direct_calls']+=1
  assert b'_ = call(' not in b0
 assert f==b'entry' or f not in rules
# Old scope and ABI function behavior preserved at byte level, except opcode value and forwarding.
assert before[b'entry'][5].split(b'block la_route',1)[1]==after[b'entry'][5].split(b'block la_route',1)[1]
assert before[b'la_command'][5].replace(b'call(es,ex,em) ',b'call(es,ex,em,eop) ')==after[b'la_command'][5]
store=blocks(after[b'la_tensor'])[b'la_store'][1]
assert b'call(lst_s,lst_at,lst_one,lst_bits) {"function":"typed_store"}' in store
def parse_codes(txt,pat):
 return {int(m[1]) for m in re.finditer(pat,txt,re.M)}
router=blocks(after[b'entry'])[b'b12'][1]
white=parse_codes(router,rb'^  la_(?:opcode|or_opcode|xor_opcode):u64 = const\(\) \{"value":(\d+)\}',)
assert white=={131,133,134} and 132 not in white
assert b'lst_bits:u64 = select(lst_is_xor,lst_xor_bits,lst_and_or)' in store and b'lst_and_or:u64 = select(lst_is_or,lst_or_bits,lst_and_bits)' in store
table={}
for op in sorted(white):
 got=[((x^y) if op==134 else ((x|y) if op==133 else (x&y))) for x in (0,1) for y in (0,1)]
 assert got=={131:[0,0,0,1],133:[0,1,1,1],134:[0,1,1,0]}[op];table[op]=got
print(json.dumps(dict(**counts,source_sha256=sha(raw),stage='source-only',native_keyword_count=len(words),
 lexer_sha256=sha(lexer),discarded_scalar_calls=0,unchanged_cursor_body=True,concrete_truth_table=table,opcode132_in_route=False,
 native_or_compiler_executed=False),indent=2,sort_keys=True))
