from pathlib import Path
import hashlib,re,sys,json
D=Path(__file__).parent; a=Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(a).hexdigest()=='27ca2a8a6da366214cd4d611f0c1aeffff993f325db87b158ca67f2916f96cfa'
a=a.decode()
section=lambda n:a.split('def '+n+'(',1)[1].split('\ndef ',1)[0]
size=0; reserved={}
for l in section('keyword').splitlines():
 if z:=re.search(r'if size == (\d+)',l):size=int(z[1])
 if z:=re.search(r'choose\((.*?), (\d+), result\)',l):
  x={i:int(v) for i,v in re.findall(r'([abcd]) == (\d+)',z[1])}
  key=b''.join(x.get(k,0).to_bytes(8,'little') for k in 'abcd')[:size].decode()
  reserved[key]=int(z[2])
rules={}
for name in ('reserved_function','export_loader_hook'):
 rules[name]=set()
 for match in re.finditer(r'match\(s, begin, size, (\d+), (\d+), (\d+), (\d+), (\d+)\)',section(name)):
  n,*words=map(int,match.groups());rules[name].add(b''.join(i.to_bytes(8,'little') for i in words)[:n].decode())
assert reserved['read']==305 and len(reserved)>=91
runtime=Path(sys.argv[2]).read_text()
functions=set(['nz_rank_route','nz_rank2'])
allnew='\n'.join(re.search(r'^fn '+f+r'\(.*?^end$',runtime,re.M|re.S)[0] for f in functions)
assert not functions&set.union(*rules.values())
labels=set(re.findall(r'^block (\w+)\(',allnew,re.M))
variables=set(re.findall(r'\b(\w+):(?:u64|bool|readbuf|mutbuf|usize)\b',allnew))
names=functions|variables|labels
assert not names&reserved.keys(),names&reserved.keys()
assert max(map(len,names))<=128 and not any(k.startswith('__hia_') for k in names)
assert '_ = call(' not in allnew
print(json.dumps(dict(reserved_keywords=len(reserved),functions=len(functions),labels=len(labels),variables=len(variables),
 max_identifier_bytes=max(map(len,names)),name_collisions=[],discarded_scalar_calls=0),sort_keys=True))
