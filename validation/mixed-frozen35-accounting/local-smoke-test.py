#!/usr/bin/env python3
"""Limited native accounting smoke, separate from the original actualTorch35."""
import argparse,hashlib,importlib.util,json,struct,sys,traceback
from pathlib import Path

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('observer','build','protocol','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 inputs={}
 def bind(path):
  path=path.resolve();b=path.read_bytes();inputs[str(path)]={'bytes':len(b),'sha256':sha(b)};return b
 def module(name,path):
  bind(path);s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
 m=module('mixed_fixed35_smoke_observer',a.observer)
 wire=module('mixed_fixed35_smoke_wire',a.protocol)
 bind(a.protocol.with_name('owner_client.py'));bind(Path(__file__))
 image=a.build/'tensor-mixed-autograd-session';src=a.build/'runtime.hepta';bind(image);bind(src)
 profile=m.pin_profile(image,src)
 for path in m.dependency_paths().values():bind(path)
 report={'status':'RUNNING','profile':profile,'frames':[],'Torch_executed':False,'original35_executed':False,
         'full_pytorch_rewrite_complete':False};session=None;journal=None
 try:
  session=wire.Session(image,directory=out/'wire');journal=m.Journal(out/'journals','native-smoke-construction-backward-reset-rollback')
  def snapshot():
   raw=session.owner();v=m.structural.decode_owner(raw['bytes'],raw['pointer']);m.verify_snapshot(v);return v
  def call(label,command,created=0,status=0):
   before=snapshot();packet=wire.packet([command]);words=list(struct.unpack_from('<8Q',packet,32))
   raw=session.exchange(packet);actual,rows=wire.parse(raw);after=snapshot()
   assert actual==status,(label,actual,status)
   tickets=[{'record':0,'opcode':words[0],'count':created,'logical':1}] if created else []
   result=m.verify_transition(before,after,tickets,[words],status)
   journal.event(before,after,tickets,[words],status,result=result)
   report['frames'].append({'label':label,'status':actual,'result':result})
   return rows[0] if rows else None
  x=call('public requires leaf',wire.full([2],2.,1),1)['value']
  z=call('public ordinary leaf',wire.full([2],3.),1)['value']
  call('MSE real three-node construction',(16,x,z),3)
  y=call('two original Saved Tensor slots',(2,3,x,x),1)['value']
  seed=call('public backward seed',wire.full([2],1.),1)['value']
  call('real private gradient materialization',(29,y,seed))
  call('reset retains only exact roots',(33,2))
  call('ordinary unknown command full rollback',(999,),status=4)
  stats=call('complete actual physical stats',(54,))
  assert list(struct.unpack('<12Q',stats['payload']))==m.verify_snapshot(snapshot())['stats']
  report['status']='PASS_NATIVE_SMOKE_ONLY';report['frame_count']=session.frames
 except BaseException as e:report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
 finally:
  if session is not None:
   try:session.finish();report['EOF_exit_code']=session.exit_code
   except BaseException as e:report.update(status='FAIL',transport_error=repr(e))
  if journal is not None:journal.close()
  for name,entry in inputs.items():
   b=Path(name).read_bytes();assert len(b)==entry['bytes'] and sha(b)==entry['sha256']
  report['inputs']=inputs;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report.get(k) for k in ('status','frame_count','error')}))
 return int(report['status']!='PASS_NATIVE_SMOKE_ONLY')
if __name__=='__main__':raise SystemExit(main())
