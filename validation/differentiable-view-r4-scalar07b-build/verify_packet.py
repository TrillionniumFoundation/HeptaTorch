"""Seal original Scalar07b compile bytes; only filesystem reads, hashing and compression."""
from pathlib import Path
import hashlib,io,json,lzma,os,stat,sys,tarfile
R=Path('/home/qian-qi/heptatorch-continuation-20261116/runs')
BUILD=R/'differentiable-view-r4-scalar07b-native-build-01'
OUT=R/'differentiable-view-r4-scalar07b-build-packet-01'
SOURCE='c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a'
IMAGE='0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74'
def sha(b):return hashlib.sha256(b).hexdigest()
def info(p):
 p=Path(p);s=p.stat();assert p.is_file() and not p.is_symlink()
 b=p.read_bytes();return {'bytes':len(b),'sha256':sha(b),'mode':stat.S_IMODE(s.st_mode)}
def write(p,j):
 p.write_bytes((json.dumps(j,indent=2,sort_keys=True)+'\n').encode());p.chmod(0o600)
def verify(packet):
 manifest=json.loads((packet/'manifest.json').read_bytes());archive=(packet/'complete.tar.xz').read_bytes()
 assert len(archive)==manifest['archive']['bytes'] and sha(archive)==manifest['archive']['sha256']
 dec=lzma.LZMADecompressor();raw=dec.decompress(archive);assert dec.eof and dec.unused_data==b''
 assert len(raw)==manifest['tar']['bytes'] and sha(raw)==manifest['tar']['sha256']
 files=manifest['files'];seen=set();total=0
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as t:
  for m in t:
   assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts and m.name not in seen
   seen.add(m.name);b=t.extractfile(m).read();e=files[m.name]
   assert len(b)==m.size==e['bytes'] and m.mode==e['mode'] and sha(b)==e['sha256'],m.name
   total+=len(b)
  assert not any(raw[t.offset:]),'nonzero bytes after tar members'
 assert seen==set(files) and len(seen)==manifest['member_count'] and total==manifest['expanded_bytes']
 return {'schema':'heptatorch.scalar07b.compile-packet-verification.v1','status':'PASS_BYTES_ONLY',
         'members':len(seen),'expanded_bytes':total,'archive_sha256':sha(archive),
         'all_member_bytes_sizes_modes_verified':True,'single_XZ_stream_EOF':True,'complete_tar_stream_verified':True,
         'new_compiler_executed_here':False,'new_native_business_executed_here':False,'new_Torch_executed_here':False}
def main():
 if len(sys.argv)==3 and sys.argv[1]=='--verify-only':
  print(json.dumps(verify(Path(sys.argv[2])),sort_keys=True));return
 assert len(sys.argv)==1
 OUT.mkdir(exist_ok=False)
 before=json.loads((BUILD/'inputs-before.json').read_bytes());after=json.loads((BUILD/'inputs-after.json').read_bytes())
 execution=json.loads((BUILD/'execution.json').read_bytes())
 assert before==after and execution['all_bound_inputs_unchanged'] is True
 assert execution['status']=='PASS' and execution['compiler_driver_started'] is execution['compiler_driver_completed'] is True
 assert execution['driver_returncode']==0 and execution['source_sha256']==SOURCE and execution['native_sha256']==IMAGE
 assert execution['all_13_units_fresh_compiler_outputs_verified'] is execution['all_prepared_unit_source_pins_matched'] is True
 assert (execution['units'],execution['functions'],execution['blocks'],execution['instructions'])==(13,658,8661,30980)
 status=json.loads((BUILD/'compile-terminal-status.original.json').read_bytes());logs=json.loads((BUILD/'compile-terminal-read.original.json').read_bytes())
 assert status['job_id']=='1a1e76ea9f4-392-3d7b' and status['state']=='succeeded' and status['exit_code']==0
 assert status['output_complete'] is status['output_durable'] is True
 assert logs['job_id']==status['job_id'] and logs['stdout']['eof'] is logs['stderr']['eof'] is True
 assert logs['stdout']['offset']==logs['stderr']['offset']==0 and logs['stderr']['size']==0
 assert bytes.fromhex(logs['stdout']['bytes_hex']).decode()==logs['stdout']['text']
 assert json.loads(logs['stdout']['text'])['native_sha256']==IMAGE
 paths={};mapping={}
 for i,(name,e) in enumerate(sorted(before.items())):
  p=Path(name);actual=info(p);assert all(actual[k]==e[k] for k in actual),name
  rel='bound-inputs/%04d-%s'%(i,p.name);paths[rel]=p
  mapping[name]={'archive_path':rel,**actual}
 for p in sorted(BUILD.rglob('*')):
  if p.is_file():
   assert not p.is_symlink();paths['execution/'+p.relative_to(BUILD).as_posix()]=p
 for p in [Path(__file__).resolve(),R/'differentiable-view-r4-scalar07b-seal-request-01.json',R/'differentiable-view-r4-scalar07b-wrapper-adaptation-01.json',R/'scalar07b-wrapper-preparation-correction.json',R/'differentiable-view-r4-scalar07b-execute-compile-once.py',R/'differentiable-view-r4-scalar07b-compile-request-01.json',R/'hia-wrapped-scalar07b-source-review-01/report.json',R/'hia-wrapped-scalar07b-source-review-01/manifest.json']:
  paths['custody/'+p.name]=p
 files={name:{**info(p),'source_path':str(p)} for name,p in paths.items()}
 raw_io=io.BytesIO()
 with tarfile.open(fileobj=raw_io,mode='w:',format=tarfile.PAX_FORMAT) as t:
  for name,p in sorted(paths.items()):
   e=files[name];b=p.read_bytes();assert len(b)==e['bytes'] and sha(b)==e['sha256'] and info(p)=={k:e[k] for k in ('bytes','sha256','mode')}
   m=tarfile.TarInfo(name);m.size=len(b);m.mode=e['mode'];m.mtime=0;m.uid=m.gid=0;t.addfile(m,io.BytesIO(b))
 raw=raw_io.getvalue();compressed=lzma.compress(raw,format=lzma.FORMAT_XZ,preset=6)
 (OUT/'complete.tar.xz').write_bytes(compressed);(OUT/'complete.tar.xz').chmod(0o600)
 manifest={'schema':'heptatorch.scalar07b.original-compile-packet.v1','status':'ORIGINAL_COMPILE_BYTES',
   'source_sha256':SOURCE,'native_sha256':IMAGE,'build_report_sha256':'b31a45a850f06196432b2786ad0a39b56088ccf6c7b100d0da3517a44de5ba48',
   'original_execution_job':status['job_id'],'original_execution_status':'PASS_COMPILE_ONLY',
   'member_count':len(files),'expanded_bytes':sum(e['bytes'] for e in files.values()),'bound_input_count':len(mapping),
   'archive':{'bytes':len(compressed),'sha256':sha(compressed)},'tar':{'bytes':len(raw),'sha256':sha(raw)},
   'files':files,'bound_input_map':mapping,'business_runtime_qualification_claimed':False,
   'previous06_compile_packet':{'archive_sha256':'49c500af46d6c10975fd68878d5084fd4bed6a71a575c1924e38285171c06845','manifest_sha256':'d543cb9869b8e512d8b88ad17eae42add73abe7cf341588cfa1845298ddd1f8e','modified':False}}
 write(OUT/'manifest.json',manifest)
 result=verify(OUT);write(OUT/'verification.json',result)
 assert all(info(Path(name))=={k:e[k] for k in ('bytes','sha256','mode')} for name,e in before.items())
 outer={'schema':'heptatorch.scalar07b.original-compile-outer.v1','status':'PASS_BYTES_ONLY','files':{n:info(OUT/n) for n in ('complete.tar.xz','manifest.json','verification.json')},'Torch_executed_here':False,'native_business_executed_here':False,'compiler_executed_here':False}
 write(OUT/'outer.json',outer)
 print(json.dumps({'status':'PASS_BYTES_ONLY','members':len(files),'bound_inputs':len(mapping),'files':{n:info(OUT/n) for n in ('complete.tar.xz','manifest.json','verification.json','outer.json')}}),flush=True)
if __name__=='__main__':main()
