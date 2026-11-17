#!/usr/bin/env python3
"""Pure archive and custody verification; never runs the saved reader."""
import argparse,hashlib,json,lzma,stat,tarfile
from pathlib import Path,PurePosixPath
def sha(b):return hashlib.sha256(b).hexdigest()
def meta(p):
 h=hashlib.sha256();n=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b);n+=len(b)
 return {'bytes':n,'sha256':h.hexdigest(),'mode':stat.S_IMODE(p.stat().st_mode)}
def canonical(n):
 q=PurePosixPath(n)
 assert n and not q.is_absolute() and '..' not in q.parts and '.' not in q.parts and str(q)==n,n
 return q
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--archive',required=True,type=Path);ap.add_argument('--manifest',required=True,type=Path)
 ap.add_argument('--restore',type=Path);ap.add_argument('--report',required=True,type=Path)
 ap.add_argument('--original-reference',type=Path)
 a=ap.parse_args()
 m=json.loads(a.manifest.read_bytes())
 assert meta(a.archive)['sha256']==m['archive']['sha256'] and meta(a.archive)['bytes']==m['archive']['bytes']
 expected=m['files'];assert expected
 for n in expected:canonical(n)
 if a.restore:a.restore.mkdir(parents=True,exist_ok=False)
 d=lzma.LZMADecompressor(format=lzma.FORMAT_XZ);h=hashlib.sha256();n=0
 with a.archive.open('rb') as f:
  while not d.eof:
   raw=f.read(65536);assert raw,'truncated XZ'
   out=d.decompress(raw);h.update(out);n+=len(out)
  assert not d.unused_data and not f.read(1),'multiple XZ streams or trailing bytes'
 assert n==m['tar']['bytes'] and h.hexdigest()==m['tar']['sha256']
 seen=set()
 with tarfile.open(a.archive,'r:xz') as t:
  for x in t:
   canonical(x.name);assert x.isfile() and not x.issym() and not x.islnk()
   assert x.name in expected and x.name not in seen,x.name;seen.add(x.name)
   want=expected[x.name]
   assert x.size==want['bytes'] and stat.S_IMODE(x.mode)==want['mode'],x.name
   hh=hashlib.sha256();count=0
   dest=None
   if a.restore:
    p=a.restore.joinpath(*PurePosixPath(x.name).parts);p.parent.mkdir(parents=True,exist_ok=True);dest=p.open('xb')
   try:
    with t.extractfile(x) as f:
     for raw in iter(lambda:f.read(1048576),b''):
      hh.update(raw);count+=len(raw)
      if dest:dest.write(raw)
   finally:
    if dest:dest.close()
   assert count==want['bytes'] and hh.hexdigest()==want['sha256'],x.name
   if a.restore:
    p.chmod(want['mode']);assert meta(p)=={k:want[k] for k in ('bytes','sha256','mode')},x.name
 assert seen==set(expected)
 if a.original_reference:
  assert meta(a.original_reference)['sha256']=='fecf1956790f78a9b1238f535c1c7cce9f1237b211c389ac0736d88687bbb9c9'
 report={'status':'PASS_TRANSPORT_AND_CUSTODY_ONLY','members':len(seen),'single_XZ_EOF':True,
 'whole_tar_hash_verified':True,'all_member_bytes_size_mode_verified':True,
 'restore':str(a.restore) if a.restore else None,'original_reference_checked':bool(a.original_reference),
 'native_executed':False,'Torch_executed':False,'reader_executed':False,'identity_admission_executed':False,
 'original_reference_not_contained':True,'external_original_runtime_inputs_are_bound_not_recreated':True}
 with a.report.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report))
if __name__=='__main__':main()
