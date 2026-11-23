"""Immutable exact main-only opt-in masked_select source projection, no execution."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
P=Path(__file__).parent
src,checker=map(Path,sys.argv[1:3]); src=src.resolve(strict=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert __debug__
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',checker); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
raw=(src/'manifest.json').read_bytes()
assert sha(raw)=='348ed8e949256f985cdca0d9c612dc646ae738beee6013c27ab405d2df3e67c8'
m=json.loads(raw,object_pairs_hook=b.unique_object)
r,units,_=b.check_sources(m,src,b.read_file)
assert sha(r)=='0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43'
typedef=next(u for u in m['units'] if u['name']=='runtime775_11'); up=typedef['path']
rows={u['path']:raw for u,raw,_ in units}
assert sha(rows[up])=='8095595a1859fb2973f02bb567d56b4bc960c5b4666c8914f8a53079adfe7db9'
names=b'nz_error nz_tensor nz_output nz_command nz_owned_layout nz_ordinary'.split()
assert not re.findall(rb'"value":\s*135\b',r)
def replace(v,old,new):
 assert v.count(old)==1
 return v.replace(old,new,1)
KEEP=b'  branch v144 b13(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139) b14(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139)\n'
ROUTE=b'''  nz_opcode:u64 = const() {"value":135}
  nz_chosen:bool = compare.eq(v139,nz_opcode)
  branch nz_chosen nz_route(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139,v138) nz_keep(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139,v144)
block nz_keep(mkr:readbuf,mko:mutbuf,mks:mutbuf,mkw:mutbuf,mkn:u64,mkout:u64,mki:u64,mkp:u64,mkf:u64,mkop:u64,mka:bool)
  branch mka b13(mkr,mko,mks,mkw,mkn,mkout,mki,mkp,mkf,mkop) b14(mkr,mko,mks,mkw,mkn,mkout,mki,mkp,mkf,mkop)
block nz_route(mrr:readbuf,mro:mutbuf,mrs:mutbuf,mrw:mutbuf,mrn:u64,mrout:u64,mri:u64,mrp:u64,mrf:u64,mrop:u64,mrat:u64)
  nz_written:u64 = call(mrr,mro,mrs,mrat,mrop,mrp) {"function":"nz_command"}
  jump b45(mrr,mro,mrs,mrw,mrn,mrout,mri,mrp,mrf,mrop,nz_written)
'''
inc=b''.join((P/f'{which}.hepta.inc').read_bytes() for which in ('core','wire','owner'))
anchor=b'fn predicate_command('
def newly_import(name):
 q=re.search(rb'^fn '+name+rb'\([^)]+\) -> \(u64\) effects\(read,write,trap\)',r,re.M)[0]
 assert q not in rows[up]
 return q+b' external()\n'
additional=b''.join(map(newly_import,(b'so_version_live',)))
orig_header=b'exports(half_cast_admit,'
new_header=b'exports('+b','.join(names)+b',half_cast_admit,'
oldext=b'fn absf(v0:u64) -> (u64) effects(read,write,trap) external()\n'
def proj(v,split=False,inverse=False):
 pairs=[(KEEP,ROUTE),(anchor,inc+anchor)]
 if split:pairs.extend([(orig_header,new_header),(oldext,additional+oldext)])
 for a,c in reversed(pairs) if inverse else pairs:
  v=replace(v,c,a) if inverse else replace(v,a,c)
 return v
new=proj(r); split=proj(rows[up],True)
assert proj(new,inverse=True)==r and proj(split,True,True)==rows[up]
R=b.parse_source(r); N=b.parse_source(new); U=b.parse_source(split,unit=True)
assert (set(N['definitions'])-set(R['definitions']))=={n.decode() for n in names}
assert {n for n in R['definitions'] if R['definitions'][n]!=N['definitions'][n]}=={'entry'}
assert all(N['signatures'][n]==R['signatures'][n] for n in R['signatures'])
for f in (b'entry',*names):
 body=N['definitions'][f.decode()];bindings=[]
 for line in body.splitlines()[1:]:
  if line.startswith(b'block '):bindings+=re.findall(rb'\b(\w+):\w+',line)
  else:bindings+=re.findall(rb'^  (\w+):\w+ =',line)
 assert len(bindings)==len(set(bindings)),(f,[k for k in set(bindings) if bindings.count(k)>1])
 args=lambda v:v.count(b',')+1 if v else 0
 targets={q[1]:args(q[2]) for q in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
 for t in re.findall(rb'^  (?:branch|jump) ([^\n]+)',body,re.M):
  for q in re.finditer(rb'(\w+)\(([^)]*)\)',t):
   assert q[1] in targets and targets[q[1]]==args(q[2]),(f,q[1])
C=N['definitions']['nz_tensor']; O=N['definitions']['nz_output']
assert b'call(as,a2,p0,a0,a1,a2) {"function":"typed_descriptor"}' in C
assert b'call(ws,loc,w8,wi) {"function":"typed_store"}' in C
assert b'call(ps,pp,pk) {"function":"putw"}' in O
assert b'call(mrr,mro,mrs,mrat,mrop,mrp)' in N['definitions']['entry']
assert b'branch no_data done(to1) loop(ts1,tx1,tn1,tk1,to1,tz1,tz1)' in C and b'branch more bit(cs0,cx0,cn0,ci,ck) alloc_result(cs0,cx0,cn0,ck)' in C
assert C.count(b'"function":"typed_descriptor"')==1 and C.count(b'"function":"typed_store"')==1
assert b'_ = call(' not in inc
m.pop('expected_native_sha256');m['name']='opt_in_exact_main_rank1_bool_nonzero'
typedef.pop('expected_object_sha256');typedef['sha256']=sha(split)
rp=m['runtime']['path'];m['runtime'].update(sha256=sha(new),**{k:N[k] for k in ('functions','blocks','instructions')})
rows.update({up:split,rp:new})
def reader(path,want):
 data=rows[path.relative_to(src).as_posix()];assert sha(data)==want;return data
b.check_sources(m,src,reader)
dest=Path(sys.argv[3]);dest.mkdir();(dest/'units').mkdir()
for path,data in rows.items():(dest/path).write_bytes(data)
(dest/'manifest.json').write_bytes(b.json_bytes(m));b.check_sources(m,dest,b.read_file)
print(json.dumps(dict(base_git='e9444902df9878060ab389a70ee957c1b7f4a887',parent=sha(r),combined=sha(new),unit11=sha(split),
 manifest=sha(b.json_bytes(m)),inverse=True,split=True,source_only=True,opcode_proposed=135),sort_keys=True))
