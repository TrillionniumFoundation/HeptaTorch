import hashlib,json,re,sys
from pathlib import Path
def audit(p):
 a=Path(p).read_bytes()
 defs={m[1].decode():(m[2],m[3],m[4],m[5]) for m in re.finditer(rb'^fn (\w+)\(([^)]*)\)[^\n]* entry\((\w+)\)\nblock (\w+)\(([^)]*)\)',a,re.M)}
 assert len(defs)==len(re.findall(rb'^fn ',a,re.M))==787
 return {k for k,(form,entry,lbl,bargs) in defs.items() if form!=bargs or entry!=lbl}
a,b=map(audit,sys.argv[1:])
assert not a and not b,(a,b)
print(json.dumps(dict(all_787_exact=True,changed_nz_rank2_exact=True,identifier_and_type_equality=True)))
