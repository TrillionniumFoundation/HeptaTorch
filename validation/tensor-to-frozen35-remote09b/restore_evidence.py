"""Verify the unchanged original ROG raw bytes from opaque transport parts."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,io,json,lzma,stat,tarfile
ARCHIVE_SHA='f5eecc4c1d1e21e377689e5b90f46edeb38ff4795ce2a3fcbec67019ef36a4eb'
MEMBERS_SHA='43a8725ca800da0fc9f856fecafd4174d6224bad9ff6e8b2a7d9a3da8e77f7be'
def sha(b):return hashlib.sha256(b).hexdigest()
def verify(root):
 root=Path(root).resolve()
 index=json.loads((root/'transport/index.json').read_bytes())
 assert index['whole_sha256']==ARCHIVE_SHA and index['whole_bytes']==9418900
 content=[];next_offset=0
 for part in index['parts']:
  rel=PurePosixPath(part['path']);assert not rel.is_absolute() and '..' not in rel.parts and str(rel)==part['path']
  p=root/rel;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
  b=p.read_bytes();assert len(b)==part['bytes'] and sha(b)==part['sha256'] and part['offset']==next_offset
  assert stat.S_IMODE(p.stat().st_mode)==part['installed_mode']==420
  next_offset+=len(b);content.append(b)
 raw=b''.join(content);assert next_offset==len(raw)==index['whole_bytes'] and sha(raw)==ARCHIVE_SHA
 mb=(root/'original/evidence-manifest.json').read_bytes();assert sha(mb)==MEMBERS_SHA
 m=json.loads(mb);d=lzma.LZMADecompressor();tb=d.decompress(raw)
 assert d.eof and not d.unused_data and len(tb)==m['decompressed_tar_bytes'] and sha(tb)==m['decompressed_tar_sha256']
 with tarfile.open(fileobj=io.BytesIO(tb),mode='r:') as t:
  rows=t.getmembers();assert len(rows)==245==len(m['members']) and len({r.name for r in rows})==len(rows) and {r.name for r in rows}==set(m['members'])
  total=0
  for r in rows:
   rel=PurePosixPath(r.name);assert r.isfile() and not rel.is_absolute() and '..' not in rel.parts and str(rel)==r.name
   b=t.extractfile(r).read();v=m['members'][r.name];assert len(b)==r.size==v['bytes'] and r.mode==v['mode'] and sha(b)==v['sha256'];total+=len(b)
 assert total==42306933
 return {'status':'PASS_TRANSPORT_ONLY','native_executed':False,'Torch_executed':False,'whole_sha256':ARCHIVE_SHA,'whole_bytes':len(raw),'member_count':245,'member_bytes':total,'single_XZ_EOF':True,'all_member_bytes_sizes_modes':True}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
 print(json.dumps(verify(a.root),sort_keys=True))
if __name__=='__main__':main()
