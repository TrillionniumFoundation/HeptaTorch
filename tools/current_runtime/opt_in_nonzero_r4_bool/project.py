"""Reversible exact e11598e rank4 Bool nonzero source-only projection."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
src,checker,out=map(Path,sys.argv[1:]);src=src.resolve(strict=True)
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('source_checker',checker)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(src/'manifest.json').read_bytes()
assert sha(mraw)=='7a581fb5ab4b1b3128488386eeed062b832b386d668e7500c3a74175d96df97a'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
a,units,_=b.check_sources(m,src,b.read_file)
assert sha(a)=='b6d42642301c7b35da2632c9ef64db61715f2f21dcb24547b9b34aaba98dba52'
unit=next(x for x in m['units'] if x['name']=='runtime775_11'); p=unit['path']
rows={row['path']:raw for row,raw,_ in units}
assert sha(rows[p])=='a5aba8401333acd7916873d05c58e3f89ecd24ccc64b7924f5fa40c786c9e907'
A=b.parse_source(a)
names=['nz_rank_route','nz_rank2'];edited={}
def change_once(v,x,y):
 assert v.count(x)==1,x
 return v.replace(x,y,1)
v=A['definitions']['nz_rank_route'].decode()
v=change_once(v,'''  shared:bool = bool.or(second,third_rank)
  branch shared r2(s,x,d) r1(s,x)''','''  rfour:u64 = const() {"value":4}
  fourth_rank:bool = compare.eq(r,rfour)
  low_shared:bool = bool.or(second,third_rank)
  shared:bool = bool.or(low_shared,fourth_rank)
  branch shared r2(s,x,d) r1(s,x)''')
edited['nz_rank_route']=v.encode()
v=A['definitions']['nz_rank2'].decode()
v=change_once(v,'  branch ir3 g3(is0,ix0,id0) g2(is0,ix0,id0)','''  branch ir3 g3(is0,ix0,id0) rk24(is0,ix0,id0,rk0)
block rk24(uks:mutbuf,ukx:u64,ukd:u64,ukr:u64)
  ik4:u64 = const() {"value":4}
  ir4:bool = compare.eq(ukr,ik4)
  branch ir4 g4(uks,ukx,ukd) g2(uks,ukx,ukd)''')
v=change_once(v,'block limit(',Path(__file__).with_name('geom4.inc').read_text()+'block limit(')
v=change_once(v,'  maxk:u64 = select(isthree,max3,max2)','''  max_old:u64 = select(isthree,max3,max2)
  ac4:u64 = const() {"value":4}
  max4:u64 = const() {"value":65536}
  isfour:bool = compare.eq(alcols,ac4)
  maxk:u64 = select(isfour,max4,max_old)''')
v=change_once(v,'''  sth:u64 = const() {"value":3}
  wide:bool = compare.eq(scols,sth)
  branch wide coords3(ss,sx,sn,sc,sk,so,si,sj,si2,stag,scols,splane) coords2(ss,sx,sn,sc,sk,so,si,sj,si2,stag,scols,splane)
''','''  stwo:u64 = const() {"value":2}
  twoonly:bool = compare.eq(scols,stwo)
  branch twoonly coords2(ss,sx,sn,sc,sk,so,si,sj,si2,stag,scols,splane) pick34(ss,sx,sn,sc,sk,so,si,sj,si2,stag,scols,splane)
'''+Path(__file__).with_name('copy4.inc').read_text())
edited['nz_rank2']=v.encode()
before=A['definitions']['nz_rank2']
assert before.split(b'block g2(',1)[1].split(b'block limit(',1)[0]==edited['nz_rank2'].split(b'block g2(',1)[1].split(b'block g4(',1)[0]
assert before.split(b'block coords2(',1)[1]==edited['nz_rank2'].split(b'block coords2(',1)[1]
assert edited['nz_rank2'].count(b'{"function":"typed_store"}')==4
assert edited['nz_rank2'].count(b'{"function":"nz_truth"}')==2
def change(v,inverse=False):
 for n in names:
  x,y=A['definitions'][n],edited[n]
  if inverse:x,y=y,x
  assert v.count(x)==1,n
  v=v.replace(x,y,1)
 return v
c=change(a);u=change(rows[p]);assert change(c,True)==a and change(u,True)==rows[p]
C=b.parse_source(c);U=b.parse_source(u,unit=True)
assert C['definitions'].keys()==A['definitions'].keys()
assert {f for f in A['definitions'] if A['definitions'][f]!=C['definitions'][f]}==set(names)
assert C['signatures']==A['signatures']
def decl(v):
 return tuple(tuple(x.split(b':',1)) for x in v.split(b',') if x)
def args(v):
 return tuple(x for x in v.split(b',') if x)
for f in [n.encode() for n in names]:
 body=C['definitions'][f.decode()]
 targets={x[1]:decl(x[2]) for x in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
 # Pinned native parser requires identical value-name IDs in the entry.
 head=body.splitlines()[0]
 sig=re.fullmatch(rb'fn (\w+)\(([^)]*)\) -> \([^)]*\) effects\([^)]*\) entry\((\w+)\)',head)
 assert sig and sig[1]==f and decl(sig[2])==targets[sig[3]],(f,head,targets)
 assert max(len(v) for v in targets.values())<=16
 global_keys=[]
 for nm,pv in targets.items():global_keys.extend(k for k,_ in pv)
 global_keys+=re.findall(rb'^  (\w+):\w+ =',body,re.M)
 assert len(global_keys)==len(set(global_keys)),(f,[k for k in set(global_keys) if global_keys.count(k)>1])
 for block in re.split(rb'(?m)(?=^block )',body)[1:]:
  label=block.split(b'(',1)[0][6:];loc=dict(targets[label])
  for line in block.splitlines()[1:]:
   if x:=re.fullmatch(rb'  (\w+):(\w+) = call\(([^)]*)\) \{"function":"(\w+)"\}',line):
    parameters,returns,_=C['signatures'][x[4].decode()]
    assert tuple(loc[k] for k in args(x[3]))==parameters and (x[2],)==returns,(f,line)
   if line.startswith((b'  branch ',b'  jump ')):
    if line.startswith(b'  branch '):assert loc[line.split()[1]]==b'bool'
    for z in re.finditer(rb'(\w+)\(([^)]*)\)',line):
     assert tuple(loc[k] for k in args(z[2]))==tuple(t for _,t in targets[z[1]]),(f,line)
   if x:=re.fullmatch(rb'  (\w+):(\w+) = .*',line):loc[x[1]]=x[2]

m['name']='opt_in_exact_nonzero_rank4_bool'
m.pop('expected_native_sha256',None);unit.pop('expected_object_sha256',None)
unit['sha256']=sha(u)
rp=m['runtime']['path'];m['runtime'].update(sha256=sha(c),**{k:C[k] for k in ('functions','blocks','instructions')})
rows.update({p:u,rp:c})
def reader(path,wanted):
 data=rows[path.relative_to(src).as_posix()]
 assert sha(data)==wanted
 return data
b.check_sources(m,src,reader)
out.mkdir();(out/'units').mkdir()
for path,raw in rows.items():(out/path).write_bytes(raw)
(out/'manifest.json').write_bytes(b.json_bytes(m));b.check_sources(m,out,b.read_file)
print(json.dumps(dict(base_git='e11598e93472bbed0e4d1c8e72ae63e800db19df',base_tree='42b71a446e9ac6751b48de238316a82f25b81e1a',
 base=sha(a),combined=sha(c),unit11=sha(u),manifest=sha(b.json_bytes(m)),unit11_counts={k:U[k] for k in ('functions','blocks','instructions')},
 exact_inverse=True,changed_bodies=names,added_bodies=[],other_14_units_exact=True,wire_unchanged=135,
 proven_ssa_call_and_branch_types=True),sort_keys=True))
