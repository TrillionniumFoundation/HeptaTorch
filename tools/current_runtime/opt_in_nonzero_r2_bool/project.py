"""Reversible exact dceb788 rank2 Bool nonzero source-only projection."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
src,checker,out=map(Path,sys.argv[1:]);src=src.resolve(strict=True)
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('source_checker',checker)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(src/'manifest.json').read_bytes()
assert sha(mraw)=='d1380df541520501f5791a33357ad83484a5af88289f863ff73fd45942db684c'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
a,units,_=b.check_sources(m,src,b.read_file)
assert sha(a)=='4b3274114665152bb66544a88b49b11047b1aac6d0b5438c3dc07727b01d0055'
unit=next(x for x in m['units'] if x['name']=='runtime775_11'); p=unit['path']
rows={row['path']:raw for row,raw,_ in units}
assert sha(rows[p])=='d54bb2df6592e23d5aa91073f52407b45ea552676a770c3a33d6cfae48f71983'
inc=Path(__file__).with_name('rank2.hepta.inc').read_bytes()
names=re.findall(rb'^fn (\w+)\(',inc,re.M);assert len(names)==5
ad=b'fn nz_command('
old=b'  made:u64 = call(es,ex) {"function":"nz_tensor"}'
new=b'  made:u64 = call(es,ex) {"function":"nz_rank_route"}'
uhead=b'exports(nz_truth,'
vhead=b'exports('+b','.join(names)+b',nz_truth,'
def change(v,split=False,inverse=False):
 xs=[(ad,inc+ad),(old,new)]+([(uhead,vhead)] if split else [])
 for x,y in (reversed(xs) if inverse else xs):
  if inverse:x,y=y,x
  assert v.count(x)==1
  v=v.replace(x,y,1)
 return v
c=change(a);u=change(rows[p],True);assert change(c,inverse=True)==a and change(u,True,True)==rows[p]
A=b.parse_source(a);C=b.parse_source(c);U=b.parse_source(u,unit=True)
assert set(C['definitions'])-set(A['definitions'])=={f.decode() for f in names}
assert {f for f in A['definitions'] if A['definitions'][f]!=C['definitions'][f]}=={'nz_command'}
assert all(C['signatures'][f]==A['signatures'][f] for f in A['signatures'])
assert C['definitions']['nz_command']==A['definitions']['nz_command'].replace(old,new)
def decl(v):
 return tuple(tuple(x.split(b':',1)) for x in v.split(b',') if x)
def args(v):
 return tuple(x for x in v.split(b',') if x)
for f in names+[b'nz_command']:
 body=C['definitions'][f.decode()]
 targets={x[1]:decl(x[2]) for x in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
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
assert b'_ = call(' not in inc
assert inc.count(b'{"function":"typed_store"}')==2 and inc.count(b'{"function":"typed_descriptor"}')==1
assert b'call(s,two,p0,z,one,two) {"function":"typed_descriptor"}' in inc
assert b'branch te layout(ts,to,tk) loop(ts,tx,tn,tc,tk,to,tz,tz)' in inc
assert b'call(ss,so,stridej) {"function":"taddr"}' in inc
assert b'int.add.wrap(sj,sk)' in inc
m['name']='opt_in_exact_nonzero_rank2_bool'
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
print(json.dumps(dict(base_git='dceb788712c7af08f00ea61c7c507dbbe3493dc7',base_tree='84a833773eee3dbed33c5f599c689a702a535f7e',
 base=sha(a),combined=sha(c),unit11=sha(u),manifest=sha(b.json_bytes(m)),unit11_counts={k:U[k] for k in ('functions','blocks','instructions')},
 exact_inverse=True,only_changed_body='nz_command',five_added=[f.decode() for f in names],other_14_units_exact=True,wire_unchanged=135,
 proven_ssa_call_and_branch_types=True),sort_keys=True))
