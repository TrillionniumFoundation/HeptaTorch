"""Independent framed binary comparison after native replay, stdlib only."""
import sys,struct,hashlib,json
from pathlib import Path
old,new,t=map(Path,sys.argv[1:])
j=json.loads(t.read_bytes()); assert j['retained_count']==1043
def rows(path,tag):
 b=path.read_bytes(); pos=0
 while pos<len(b):
  magic,n=struct.unpack_from('<8sQ',b,pos);pos+=16
  assert tag==magic and 32<=n<=1<<20 and pos+n<=len(b)
  yield b[pos:pos+n];pos+=n
 assert pos==len(b)
for g,x in j['retained'].items():
 for suf,mark in [('requests',b'HTOWNR01'),('responses',b'HTOWNS01')]:
  f=suf+'.frames.bin'; oldrows=list(rows(old/g/f,mark)); newrows=list(rows(new/g.replace('/','-')/f,mark))
  assert len(newrows)==x['replayed'] and [r for i,r in enumerate(oldrows,1) if i not in x['skipped']]==newrows
for g,v in j['witness'].items():
 oldv,newv=v['old'],v['new']; assert (oldv['status'],newv['status'])==(7,0)
 fr=oldv['frame']; assert fr==newv['frame']
 for which,rec in [('old',oldv),('new',newv)]:
  p=new/('witness-'+g+'-'+which)
  for typ,x,tag in [('request','requests',b'HTOWNR01'),('response','responses',b'HTOWNS01')]:
   got=list(rows(p/(x+'.frames.bin'),tag)); ref=list(rows(old/g/(x+'.frames.bin'),tag))
   assert len(got)==fr and ref[:fr-1]==got[:fr-1]
   assert hashlib.sha256(got[-1]).hexdigest()==rec[typ+'_sha256']
   if which=='old' or typ=='request': assert got[-1]==ref[fr-1]
print(json.dumps(dict(status='PASS',retained=j['retained_count'],changed_duplicate_witnesses=2),sort_keys=True))
