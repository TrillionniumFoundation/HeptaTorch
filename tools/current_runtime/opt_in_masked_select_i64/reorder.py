"""One-line exact allocator argument repair; no compiler or process launch."""
import hashlib,json,sys,importlib.util
from pathlib import Path
p,check,dst=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha(check.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',check);B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
q=(p/'manifest.json').read_bytes();assert sha(q)=='15b41ee1fb2cf448e8b063cec129dde5088b567b4aa5dcf1dafbb7066e55e670'
m=json.loads(q,object_pairs_hook=B.unique_object);R,U,_=B.check_sources(m,p,B.read_file)
assert sha(R)=='ed41a096e533b6dfc6d875ea6c68ee38698c3e86de90c379c0f64e8ba5217d78'
nodes={k['path']:v for k,v,_ in U}
unit=next(k for k in m['units'] if k['name']=='runtime775_11'); filepath=unit['path'];u0=nodes[filepath]
assert sha(u0)=='6a0ab809cf88f4b73bf98adf6ab5fa84d88b686699dd89aea89f27bf6f1cfc50'
old=b'  out:u64 = call(as,a1,ap0,value_tag,a1,a0) {"function":"typed_descriptor"}\n'
new=b'  out:u64 = call(as,a1,ap0,a0,a1,value_tag) {"function":"typed_descriptor"}\n'
assert new.replace(b'value_tag',b'a0')==b'  out:u64 = call(as,a1,ap0,a0,a1,a0) {"function":"typed_descriptor"}\n'
def fix(v):
 assert v.count(old)==1 and v.count(new)==0
 n=v.replace(old,new);assert n.count(new)==1 and n.replace(new,old)==v
 return n
n=fix(R);u=fix(u0)
O=B.parse_source(R);N=B.parse_source(n)
assert set(N['definitions'])==set(O['definitions'])
assert {k for k in O['definitions'] if N['definitions'][k]!=O['definitions'][k]}=={'ms_tensor'}
assert O['signatures']==N['signatures'] and all(O[k]==N[k] for k in ('functions','blocks','instructions'))
assert B.parse_source(u,unit=True)['definitions']['ms_tensor']==N['definitions']['ms_tensor']
assert b'call(mrr,mro,mrs,mrat,mrop,mrp)' in N['definitions']['entry']
unit['sha256']=sha(u);rp=m['runtime']['path'];m['runtime']['sha256']=sha(n)
nodes.update({rp:n,filepath:u});dst.mkdir();(dst/'units').mkdir()
for k,v in nodes.items():(dst/k).write_bytes(v)
(dst/'manifest.json').write_bytes(B.json_bytes(m)); B.check_sources(m,dst,B.read_file)
print(json.dumps(dict(parent_held=sha(R),combined=sha(n),unit11=sha(u),manifest=sha(B.json_bytes(m)),byte_inverse=True,changed_lines=1,changed_functions=['ms_tensor'],static_split=True)))
