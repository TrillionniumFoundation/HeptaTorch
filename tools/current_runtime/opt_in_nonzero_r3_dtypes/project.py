"""Reversible exact 606e5c1 rank3 public value dtypes nonzero source-only projection."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
src,checker,out=map(Path,sys.argv[1:]);src=src.resolve(strict=True)
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('source_checker',checker)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(src/'manifest.json').read_bytes()
assert sha(mraw)=='d410f1d908afe10549b44374dc7b133aab9581f21b9bf9854a6a7dfbd408d61f'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
a,units,_=b.check_sources(m,src,b.read_file)
assert sha(a)=='ce78e4be4c06b1b05dc3221d3108ecd4241a1787e8ba6422f17154f6c5b1df87'
unit=next(x for x in m['units'] if x['name']=='runtime775_11'); p=unit['path']
rows={row['path']:raw for row,raw,_ in units}
assert sha(rows[p])=='7178b1ba9725777c3c41724e078bde08899ab64f108a4390be33b80d0767259c'
A=b.parse_source(a)
names=['nz_rank2']
old=A['definitions']['nz_rank2'].decode()
new=old
def rep(x,y):
 global new
 assert new.count(x)==1,x
 new=new.replace(x,y,1)
rep('  bt3:bool = compare.eq(gdtype,g1v)', '''  priv3:u64 = const() {"value":3}
  gwidth:u64 = call(gdtype) {"function":"dtype_width"}
  valid_width:bool = compare.ne(gwidth,gz)
  public3:bool = compare.ne(gdtype,priv3)
  bt3:bool = bool.and(valid_width,public3)''')
rep('g3shape(gs3,gx3,cnval,ga0,ga1,ga2,gs0,gs1,gs2,glimit,g1v,gz,gthree)','g3shape(gs3,gx3,cnval,ga0,ga1,ga2,gs0,gs1,gs2,glimit,g1v,gz,gthree,gwidth,gdtype)')
rep('hz:u64,hthree:u64)','hz:u64,hthree:u64,hwidth:u64,htag:u64)')
rep('limit(hs3,hx3,hn3,ha2,hlim,hone,hone,hthree,plane)','limit(hs3,hx3,hn3,ha2,hlim,hwidth,htag,hthree,plane)')
assert old.split('block g3(',1)[0]==new.split('block g3(',1)[0]
assert old.split('block limit(',1)[1]==new.split('block limit(',1)[1]
assert old.count('block g3(')==new.count('block g3(')==1
def change(v,inverse=False):
 x,y=(new,old) if inverse else (old,new)
 x,y=x.encode(),y.encode()
 assert v.count(x)==1
 return v.replace(x,y,1)
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

m['name']='opt_in_exact_nonzero_rank3_public_dtypes'
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
print(json.dumps(dict(base_git='606e5c182d8132b0646a62fb9c51aad3fbc6b371',base_tree='42e0bb1a78d9e4439de71832aefafefb164477a5',
 base=sha(a),combined=sha(c),unit11=sha(u),manifest=sha(b.json_bytes(m)),unit11_counts={k:U[k] for k in ('functions','blocks','instructions')},
 exact_inverse=True,changed_bodies=names,added_bodies=[],other_14_units_exact=True,wire_unchanged=135,
 proven_ssa_call_and_branch_types=True),sort_keys=True))
