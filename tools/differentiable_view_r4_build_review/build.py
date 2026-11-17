#!/usr/bin/env python3
"""Run one frozen R4 authoring/build stage and validate every fixed source, object and ELF."""
from pathlib import Path
import argparse,hashlib,json,os,stat,struct,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
STAGES={
 'shape':{'package':'differentiable_view_shape_r4','original_report':'shape-original-build.json',
          'original_report_sha256':'6f96f137a82ae128efcd3e143784ea7f5df438b50bbfe71adc8ec321f8dedf54',
          'image':'tensor-shape-view-session','parent_image':'tensor-differentiable-view-session','parent_functions':627},
 'scalar07b':{'package':'differentiable_view_scalar_r4','original_report':'scalar07b-original-build.json',
          'original_report_sha256':'b31a45a850f06196432b2786ad0a39b56088ccf6c7b100d0da3517a44de5ba48',
          'image':'tensor-shape-scalar-session','parent_image':'tensor-shape-view-session','parent_functions':638}}
FIXED_FIELDS=('schema','orchestration_revision','parent_source_sha256','parent_native_sha256','source_sha256',
              'descriptor_bytes','state_format','backing_format','functions','blocks','instructions',
              'compiler_sha256','linker_sha256','changed_parent_functions','added_functions',
              'authoring_inputs_sha256','runtime_qualified','torch_executed',
              'full_pytorch_rewrite_complete','pytorch_api_denominator')
def sha(b):return hashlib.sha256(b).hexdigest()
def file(p):
 p=Path(p);s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise AssertionError(('single regular file required',str(p)))
 b=p.read_bytes();return b,{'bytes':len(b),'mode':stat.S_IMODE(s.st_mode),'sha256':sha(b)}
def dump(p,v):
 with p.open('xb') as f:f.write((json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
def main():
 if sys.flags.optimize:raise RuntimeError('assertions must be enabled')
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--stage',choices=sorted(STAGES),required=True)
 for name in ('previous','upstream','compiler','linker','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--prepare-only',action='store_true')
 a=p.parse_args();spec=STAGES[a.stage];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 native=out/'native';previous=a.previous.resolve();upstream=a.upstream.resolve()
 package=ROOT/'tools'/spec['package'];compiler=a.compiler.resolve();linker=a.linker.resolve()
 inputs={};mapped_expected={};report={'schema':'heptatorch.r4.portable-build-admission.v1','stage':a.stage,'status':'RUNNING',
  'compiler_executed':False,'native_business_executed':False,'Torch_executed':False,'runtime_qualified':False,
  'closed_schema_count':0,'full_pytorch_rewrite_complete':False,'whole_pytorch_gate':'HOLD',
  'prepare_only':a.prepare_only,'child_process_started':False,'child_process_completed':False}
 def bind(path,pin=None):
  path=Path(path).resolve();raw,v=file(path)
  if pin is not None:assert v['sha256']==pin,('fixed input SHA',str(path),v['sha256'],pin)
  if str(path) in inputs:assert inputs[str(path)]==v,('input changed',str(path))
  inputs[str(path)]=v
  return raw
 try:
  bind(Path(__file__).resolve());basis=json.loads(bind(HERE/spec['original_report'],spec['original_report_sha256']))
  assert basis['status']=='PASS' and basis['link_exit']==0 and len(basis['units'])==13
  assert len({u['name'] for u in basis['units']})==13
  bind(compiler,basis['compiler_sha256']);bind(linker,basis['linker_sha256'])
  for name,pin in basis['authoring_inputs_sha256'].items():
   assert Path(name).name==name;bind(package/name,pin)
  parent=json.loads(bind(previous/'build.json'))
  assert parent['status']=='PASS' and parent['link_exit']==0 and parent['functions']==spec['parent_functions']
  assert parent['source_sha256']==basis['parent_source_sha256'] and parent['native_sha256']==basis['parent_native_sha256']
  assert parent['descriptor_bytes']==672 and len(parent['units'])==13
  assert len({u['name'] for u in parent['units']})==13
  bind(previous/'runtime.hepta',basis['parent_source_sha256'])
  bind(previous/spec['parent_image'],basis['parent_native_sha256'])
  for u in parent['units']:
   assert Path(u['name']).name==u['name'] and u['compiler_exit']==0
   bind(previous/(u['name']+'.hepta'),u['source_sha256']);bind(previous/(u['name']+'.o'),u['object_sha256'])
  original_parent=[Path(path).parent for path,pin in basis['inputs_sha256'].items()
                   if Path(path).name=='runtime.hepta' and pin==basis['parent_source_sha256']]
  assert len(original_parent)==1;original_parent=original_parent[0]
  original_repo=Path('/home/qian-qi/heptatorch-continuation-20261116/runs/to-view-recovery-integration-01/repo')
  original_upstream=Path('/home/qian-qi/heptatorch-continuation-20261116/toolchain')
  relocation=[]
  for original,pin in basis['inputs_sha256'].items():
   old=Path(original)
   if old.is_relative_to(original_parent):
    new=previous/old.relative_to(original_parent)
    # Build report paths differ by checkout, while its actual source and all product identities are checked above.
    dynamic_parent_report=old==original_parent/'build.json'
   elif old.is_relative_to(original_repo):
    new=ROOT/old.relative_to(original_repo);dynamic_parent_report=False
   elif old.is_relative_to(original_upstream):
    new=upstream/old.relative_to(original_upstream);dynamic_parent_report=False
   else:raise AssertionError(('unmapped original input',original))
   bind(new,None if dynamic_parent_report else pin)
   mapped_expected[str(new)]=inputs[str(new)]['sha256']
   relocation.append({'original_path':original,'actual_path':str(new),'original_sha256':pin,
    'actual_sha256':inputs[str(new)]['sha256'],'path_dependent_parent_report':dynamic_parent_report})
  for u in basis['units']:bind(package/'generated/units'/(u['name']+'.hepta'),u['source_sha256'])
  bind(package/'generated/runtime.hepta',basis['source_sha256'])
  dump(out/'inputs-before.json',inputs);dump(out/'input-relocations.json',relocation)
  command=[sys.executable,'-E','-S','-B',str(package/'build.py'),'--previous',str(previous),
   '--repository',str(ROOT),'--upstream',str(upstream),'--compactor',str(ROOT/'tools/owned_optimizers/compact_functions.py'),
   '--compiler',str(compiler),'--linker',str(linker),'--output',str(native)]
  if a.prepare_only:command.append('--prepare-only')
  dump(out/'command.json',{'argv':command,'cwd':str(ROOT),'stdin':'DEVNULL','fresh_output_required':True})
  assert not native.exists();report['child_process_attempted']=True;t0=time.monotonic()
  report['child_process_started']=None
  report['compiler_executed']=False if a.prepare_only else None
  try:
   child=subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,capture_output=True,timeout=1800)
  except subprocess.TimeoutExpired as error:
   report['child_process_started']=True
   (out/'driver.stdout').write_bytes(error.stdout or b'');(out/'driver.stderr').write_bytes(error.stderr or b'')
   report['driver_timed_out']=True;raise
  report['child_process_started']=True;report['child_process_completed']=True
  (out/'driver.stdout').write_bytes(child.stdout);(out/'driver.stderr').write_bytes(child.stderr)
  report['child']={'exit_code':child.returncode,'elapsed_seconds':time.monotonic()-t0,
   'stdout_bytes':len(child.stdout),'stdout_sha256':sha(child.stdout),'stderr_bytes':len(child.stderr),
   'stderr_sha256':sha(child.stderr),'stdout_EOF':True,'stderr_EOF':True}
  if (native/'build.json').is_file():
   built=json.loads(file(native/'build.json')[0])
   if any('compiler_exit' in u for u in built.get('units',[])):report['compiler_executed']=True
  assert child.returncode==0 and child.stderr==b'',('original builder refused',child.returncode)
  built=json.loads(file(native/'build.json')[0]);printed=json.loads(child.stdout)
  for k in FIXED_FIELDS:assert type(built[k]) is type(basis[k]) and built[k]==basis[k],('fixed build field',k)
  assert built['inputs_sha256']==mapped_expected,'complete relocated builder input denominator'
  assert built['authoring_inputs_sha256']==basis['authoring_inputs_sha256']
  assert sha((native/'runtime.hepta').read_bytes())==basis['source_sha256']
  assert (native/'runtime.hepta').read_bytes()==(package/'generated/runtime.hepta').read_bytes()
  assert [u['name'] for u in built['units']]==[u['name'] for u in basis['units']]
  unit_results=[]
  for got,expected in zip(built['units'],basis['units']):
   name=expected['name'];raw,meta=file(native/(name+'.hepta'))
   assert meta['sha256']==got['source_sha256']==expected['source_sha256']
   assert raw==(package/'generated/units'/(name+'.hepta')).read_bytes()
   row={'name':name,'source':meta,'source_byte_equal_to_published':True}
   if a.prepare_only:
    assert got=={'name':name,'source_sha256':expected['source_sha256'],'native_compile_started':False}
    assert not (native/(name+'.o')).exists()
   else:
    assert got==expected,('all fixed unit fields',name)
    obj,om=file(native/(name+'.o'));err,em=file(native/(name+'.stderr'))
    assert obj.startswith(b'\x7fELF') and om['sha256']==expected['object_sha256'] and not err
    row.update(object=om,compiler_stderr=em,compiler_exit=0)
   unit_results.append(row)
  if a.prepare_only:
   assert built['status']=='PREPARED_NOT_COMPILED' and built['units_planned']==13
   assert not (native/spec['image']).exists() and not (native/'link.request.bin').exists()
   report['compiler_executed']=False
   status='PASS_SOURCE_PREPARATION_ONLY'
  else:
   assert built['status']=='PASS' and built['link_exit']==0
   app,im=file(native/spec['image'])
   assert im['sha256']==basis['native_sha256']==built['native_sha256']
   assert im['bytes']==basis['native_bytes']==built['native_bytes'] and app.startswith(b'\x7fELF')
   assert im['mode']==0o500
   objects=[(native/(u['name']+'.o')).read_bytes() for u in basis['units']]
   expected_link=b'HIASTKS1'+struct.pack('<3Q',13,4,0)+b'step'+b''.join(struct.pack('<Q',len(o))+o for o in objects)
   actual_link,lm=file(native/'link.request.bin');link_err,le=file(native/'link.stderr')
   assert actual_link==expected_link and not link_err
   report.update(compiler_executed=True,compiler_processes=13,linker_processes=1,
    native_image=im,link_request=lm,link_stderr=le,objects_and_link_image_reused=False)
   status='PASS'
  for k,v in printed.items():assert k in built and built[k]==v,('complete builder stdout',k)
  for path,v in inputs.items():assert file(path)[1]==v,('all before inputs unchanged',path)
  dump(out/'inputs-after.json',inputs)
  report.update(status=status,all_bound_inputs_unchanged=True,bound_input_count=len(inputs),units=unit_results,
   source_sha256=basis['source_sha256'],expected_native_sha256=basis['native_sha256'],functions=basis['functions'],
   blocks=basis['blocks'],instructions=basis['instructions'],native_unit_count=13,descriptor_bytes=672,
   original_build_report_sha256=spec['original_report_sha256'],new_build_report=file(native/'build.json')[1],
   qualification_scope='Exact source preparation or fresh compilation only. Runtime evidence is required separately.')
  code=0
 except BaseException as error:
  report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());code=1
 inventory={}
 for path in sorted(out.rglob('*')):
  if path.is_file():inventory[path.relative_to(out).as_posix()]=file(path)[1]
 dump(out/'output-inventory.json',inventory)
 report['output_inventory']=file(out/'output-inventory.json')[1]
 dump(out/'report.json',report)
 if code==0:
  dump(out/('prepared-success.json' if a.prepare_only else 'success.json'),{
   'status':report['status'],'report_sha256':file(out/'report.json')[1]['sha256'],
   'source_sha256':report['source_sha256'],'expected_native_sha256':report['expected_native_sha256'],
   'stage':a.stage,'native_business_executed':False,'Torch_executed':False})
 print(json.dumps({'status':report['status'],'report_sha256':file(out/'report.json')[1]['sha256'],
  'stage':a.stage,'error':report.get('error')}),flush=True)
 return code
if __name__=='__main__':raise SystemExit(main())
