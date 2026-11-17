#!/usr/bin/env python3
"""Restore a frozen Torch observation packet and replay its original pure admission."""
from pathlib import Path
import argparse,hashlib,json,os,stat,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[2]
SPECS={
 'view26':{
  'packet':'validation/differentiable-view-shape-reference26',
  'verifier':'tests/differentiable_view_shape_reference_review/verify_packet.py',
  'verifier_sha256':'a0729545bb74382a0ca7d2c5d6943869569f7cfa5688e647658535ca57030dd1',
  'archive_sha256':'6a17f7862b7d73c2b992982d862ceb108ec4cf85115542bea91718926c3d88cb',
  'manifest_sha256':'9241f5131d4b4bed3049b343ad764d6a9a078be528c893380d6540ece69389e4',
  'admission_sha256':'deb686ea4af865bbf03b3978cfbf6154f0780488d3e460555f6fba97484eb014',
  'reference_sha256':'fecf1956790f78a9b1238f535c1c7cce9f1237b211c389ac0736d88687bbb9c9',
  'reference_bytes':18335446,'probe_sha256':'4dd374419e4b21a9c814c855d846481730a12ea57a30dcf229a6b962f07021c5','probe_bytes':34613,'cases':26},
 'scalar32':{
  'packet':'validation/wrapped-f64-reference32',
  'verifier':'validation/wrapped-f64-reference32/verify_packet.py',
  'verifier_sha256':'6190cf0dbb34ecc73e4d5b74930df8d51fbf433ac5e72eedf3023e1381df5742',
  'archive_sha256':'bffcdc7025f5000c472727ac4d68d358d9b18a9868bc4b7e7eb49c2620fedc52',
  'manifest_sha256':'07ac12ad8a73561f070d72ed5de0fc8e1236e1e69f7a17b1ff0ee063459f9ad3',
  'admission_sha256':'767923408156d36d1f2247ce85b16f70f228bcf222a6f2068bf1b18c118f1e6d',
  'reference_sha256':'f3ecc9dd7db2ce8fbd0e424557ad5b95ac2e899a1c09cf38dad47e4fecc2b0c6',
  'reference_bytes':2867710,'probe_sha256':'c0ddafbae6ea81bd98f6066bc867f71e0b6e8d4f73e6c5d7f15f7fe3d29569a3','probe_bytes':17863,'cases':32}}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1,('single regular input',str(p))
 raw=p.read_bytes();return raw,{'bytes':len(raw),'sha256':sha(raw),'mode':stat.S_IMODE(s.st_mode)}
def save(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def canonical_input(payload,output,pin,size):
 matches=[]
 for p in sorted(payload.rglob('*')):
  if p.is_file() and not p.is_symlink() and p.stat().st_size==size:
   raw,meta=read(p)
   if meta['sha256']==pin:matches.append((p,raw,meta))
 assert matches,('original hash-bound input absent',pin,size)
 selected,raw,meta=matches[0]
 assert all(candidate==raw for _,candidate,_ in matches)
 with output.open('xb') as f:f.write(raw)
 os.chmod(output,0o644)
 assert output.read_bytes()==raw
 return {'sha256':pin,'bytes':size,'canonical_path':str(output),
         'source_path':str(selected),'source_mode':meta['mode'],'copied_mode':0o644,
         'identical_verified_source_paths':[str(p) for p,_,_ in matches]}
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--suite',choices=sorted(SPECS),required=True)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();spec=SPECS[args.suite];out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 packet=ROOT/spec['packet'];verifier=ROOT/spec['verifier'];inputs={}
 report={'schema':'heptatorch.fixed-r4-reference-preparation.v1','suite':args.suite,'status':'RUNNING','cases':spec['cases'],'Torch_executed':False,'native_executed':False,'compiler_executed':False,'closed_schema_count':0,'full_pytorch_rewrite_complete':False,'whole_pytorch_gate':'HOLD'}
 try:
  for p in sorted({Path(__file__).resolve(),verifier,*packet.iterdir()}):
   if p.is_file():
    raw,meta=read(p);inputs[str(p)]=meta
  assert inputs[str(verifier)]['sha256']==spec['verifier_sha256']
  assert inputs[str(packet/'complete.tar.xz')]['sha256']==spec['archive_sha256']
  assert inputs[str(packet/'manifest.json')]['sha256']==spec['manifest_sha256']
  command=[sys.executable,'-I','-S','-B',str(verifier),'--packet',str(packet),
           '--restore',str(out/'payload'),'--admission-output',str(out/'admission')]
  save(out/'command.json',{'argv':command,'stdin':'DEVNULL'})
  child=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,timeout=300)
  (out/'verifier.stdout').write_bytes(child.stdout);(out/'verifier.stderr').write_bytes(child.stderr)
  report['child']={'exit_code':child.returncode,'stdout_bytes':len(child.stdout),'stdout_sha256':sha(child.stdout),'stderr_bytes':len(child.stderr),'stderr_sha256':sha(child.stderr),'process_completed':True}
  assert child.returncode==0 and not child.stderr,('original pure verifier refused',child.returncode,child.stderr.decode(errors='replace'))
  parsed=json.loads(child.stdout);assert parsed['status']=='PASS'
  admission,meta=read(out/'admission/report.json');assert meta['sha256']==spec['admission_sha256']
  reference=canonical_input(out/'payload',out/'actual.json',spec['reference_sha256'],spec['reference_bytes'])
  probe=canonical_input(out/'payload',out/'probe.py',spec['probe_sha256'],spec['probe_bytes'])
  assert json.loads((out/'actual.json').read_bytes())['cases']==spec['cases']
  for path,expected in inputs.items():assert read(Path(path))[1]==expected,('input changed',path)
  report.update(status='PASS',all_inputs_unchanged=True,original_pure_admission_report_sha256=meta['sha256'],original_pure_admission_report_bytes=meta['bytes'],verifier_result=parsed,reference=reference,probe=probe,inputs=inputs,qualification_scope='Original immutable reference bytes and original pure admission only; no target runtime qualification.')
  code=0
 except BaseException as error:report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc(),inputs=inputs);code=1
 save(out/'report.json',report);print(json.dumps({'status':report['status'],'suite':args.suite,'cases':spec['cases'],'report_sha256':sha((out/'report.json').read_bytes()),'error':report.get('error')}),flush=True)
 return code
if __name__=='__main__':raise SystemExit(main())
