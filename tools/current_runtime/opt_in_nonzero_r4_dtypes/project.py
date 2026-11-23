"""Reversible exact f74029d rank4 public numeric dtype nonzero source-only projection."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
src,checker,out=map(Path,sys.argv[1:]);src=src.resolve(strict=True)
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('source_checker',checker)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(src/'manifest.json').read_bytes()
assert sha(mraw)=='ba28d8d9b67d4b7cbb9ae2bb9856915c6db858ee59250a7d9010a32a03466f07'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
a,units,_=b.check_sources(m,src,b.read_file)
assert sha(a)=='b9bf71e3224190ce9db80a34a30620aa8488d27047472601f5070b995fa92891'
unit=next(x for x in m['units'] if x['name']=='runtime775_11'); p=unit['path']
rows={row['path']:raw for row,raw,_ in units}
assert sha(rows[p])=='7cbd402961dffd7781f0aa17a37b013c0048de37f5dff08cf77e40ae21c56259'
A=b.parse_source(a)
names=['nz_rank2']
v=A['definitions']['nz_rank2'].decode()
left,tail=v.split('block g4(',1)
g4,right=tail.split('block limit(',1)
def rep(x,y):
 global g4
 assert g4.count(x)==1,x
 g4=g4.replace(x,y,1)
rep('  ffour:u64 = const() {"value":4}\n','')
rep('  fbool:bool = compare.eq(tag4,fone)',"""  fpv:u64 = const() {"value":3}
  fw:u64 = call(tag4) {"function":"dtype_width"}
  fhas:bool = compare.ne(fw,fzero)
  fpub:bool = compare.ne(tag4,fpv)
  fbool:bool = bool.and(fhas,fpub)""")
rep('flim,fone,fzero,ffour)','flim,fone,fzero,fw,tag4)')
rep('l4z:u64,l4four:u64)','l4z:u64,l4w:u64,l4tag:u64)')
rep('l4cap,l4o,l4z,l4four,zany)','l4cap,l4z,l4w,l4tag,zany)')
rep('v4cap:u64,v4o:u64,v4z:u64,v4f:u64,v4e:bool)','v4cap:u64,v4z:u64,v4w:u64,v4tag:u64,v4e:bool)')
rep('v4cap,v4o,v4z,v4f,v4z)','v4cap,v4w,v4tag,v4z,v4z)')
rep('v4cap,v4o,v4f)','v4cap,v4w,v4tag)')
rep('u4cap:u64,u4o:u64,u4f:u64)','u4cap:u64,u4w:u64,u4tag:u64)')
rep('u4cap,u4o,u4f,block_size,tail_size)','u4cap,u4w,u4tag,block_size,tail_size)')
rep('m4cap:u64,m4o:u64,m4f:u64,m4blk:u64','m4cap:u64,m4w:u64,m4tag:u64,m4blk:u64')
rep('m4cap,m4o,n_total,m4f,m4blk)','m4cap,m4w,m4tag,n_total,m4blk)')
rep('e4cap:u64,e4o:u64,e4sum:u64,e4f:u64,e4blk:u64)','e4cap:u64,e4w:u64,e4tag:u64,e4sum:u64,e4blk:u64)\n  ecol:u64 = const() {"value":4}')
rep('e4cap,e4o,e4o,e4f,e4blk)','e4cap,e4w,e4tag,ecol,e4blk)')
rep('fbool:bool = bool.and(fhas,fpub)','ftype:bool = bool.and(fhas,fpub)')
rep('bool.and(fdim,fbool)','bool.and(fdim,ftype)')
edited={'nz_rank2':(left+'block g4('+g4+'block limit('+right).encode()}
before=A['definitions']['nz_rank2']
assert before.split(b'block g4(',1)[0]==edited['nz_rank2'].split(b'block g4(',1)[0]
assert before.split(b'block limit(',1)[1]==edited['nz_rank2'].split(b'block limit(',1)[1]
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

m['name']='opt_in_exact_nonzero_rank4_public_dtypes'
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
print(json.dumps(dict(base_git='f74029da1be493811859c1b3fd9512a54c34d3a5',base_tree='a7c63ab6a9dc9b79e816fa8b026663cf6ac62125',
 base=sha(a),combined=sha(c),unit11=sha(u),manifest=sha(b.json_bytes(m)),unit11_counts={k:U[k] for k in ('functions','blocks','instructions')},
 exact_inverse=True,changed_bodies=names,added_bodies=[],other_14_units_exact=True,wire_unchanged=135,
 proven_ssa_call_and_branch_types=True),sort_keys=True))
