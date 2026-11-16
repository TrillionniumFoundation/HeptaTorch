#!/usr/bin/env python3
"""Joint Generator/Tensor/partial-backward transactions in one native owner."""
import argparse,hashlib,importlib.util,json,os,signal,struct,sys,traceback
from pathlib import Path
sys.dont_write_bytecode=True

SHA=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('executable','rng-package','cpp-reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    package=a.rng_package.resolve();inputs={Path(__file__).resolve():Path(__file__).read_bytes(),a.executable.resolve():a.executable.read_bytes()}
    def module(name,file):
        path=package/file;raw=path.read_bytes();inputs[path]=raw
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    wire=module('joint_wire','protocol.py');mt=module('joint_mt','reference.py');fp=module('joint_fp','reference_fp.py')
    manifest=(package/'source-pins.sha256').read_text()
    for line in manifest.splitlines():
        pin,name=line.split('  ',1);assert SHA((package/name).read_bytes())==pin,name
    report={'schema':'heptatorch.tensor-rng-partial-joint.v1','status':'RUNNING','executable_sha256':SHA(a.executable.read_bytes()),
        'single_native_process_per_case':True,'pytorch_executed_here':False,'cases':[],'frames':0,
        'full_generator_states':0,'joint_owner_rollbacks':0,'numeric_values':0,'stage_zero_observations':0}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def observe_stage_zero(s):
        raw=s.owner()['bytes'];assert raw[64:72]==b'HTRNGP01'
        slots=struct.unpack_from('<Q',raw,139264+16)[0];data_base=139264+144+488*slots
        checked=0
        for i in range(slots):
            record=139264+128+i*64;live=struct.unpack_from('<Q',raw,record+8)[0]
            if not live:continue
            node=struct.unpack_from('<Q',raw,record+48)[0];assert node
            logical=struct.unpack_from('<Q',raw,4096+(node//256)*8)[0];assert logical>>63
            descriptor=data_base+(logical&((1<<63)-1))+(node%256)*384
            assert all(struct.unpack_from('<Q',raw,descriptor+off)[0]==0 for off in (336,368,376))
            checked+=1
        report['stage_zero_observations']+=checked
    def exact_state(s,g,model):
        raw=s.run([(63,g)])[0]['payload'];assert raw==model.state();report['full_generator_states']+=1
    def exact_values(s,t,expected):
        raw=s.run([(39,t,0,len(expected))])[0]['payload'];assert raw==struct.pack('<'+'I'*len(expected),*expected)
        report['numeric_values']+=len(expected)
    def rollback(s,commands,error):
        s.error(commands,error);report['joint_owner_rollbacks']+=1;observe_stage_zero(s)
    def partial(s,commands,error,count):
        status,records=wire.parse(s.exchange(wire.packet(commands)))
        assert status==0 and len(records)==count and records[-1]['opcode']==58 and records[-1]['value']==error
        assert len(records[-1]['payload'])==32 and struct.unpack_from('<Q',records[-1]['payload'])[0]==error
        observe_stage_zero(s);return records
    def case(name,fn):
        s=wire.Session(a.executable,directory=out/name);row={'name':name,'status':'RUNNING'};report['cases'].append(row);save()
        try:
            details=fn(s);s.run([(53,)]);observe_stage_zero(s);s.finish()
            row.update(status='PASS',frames=s.frames,**details);report['frames']+=s.frames
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());report['status']='FAIL';save()
            if s.poll() is None:os.kill(s.pid,signal.SIGKILL);os.waitpid(s.pid,0)
            raise
        finally:
            (out/name/'assertions.json').write_text(json.dumps(row,indent=2)+'\n');save()
    def create(s,seed):
        rows=s.run([wire.full((),2,1),wire.full((),3),wire.full((4,),0),(60,seed)])
        x,c,t,g=[r['value'] for r in rows]
        y=s.run([(2,3,x,c)])[0]['value'];return x,c,t,g,y,mt.Reference(a.cpp_reference.resolve(),seed)

    def version_error(s):
        x,c,t,g,y,model=create(s,17);s.run([(12,c,0,wire.bits(5))])
        commands=[(68,t,g,0,wire.bits(1)),(58,y,0,0),(68,t,g,0,wire.bits(1)),(60,999)]
        records=partial(s,commands,6,2)
        expected=[fp.uniform(model.word(),0,wire.bits(1)) for _ in range(4)]
        exact_values(s,t,expected);exact_state(s,g,model)
        assert wire.meta(s.run([(13,t)])[0])['version']==1 and s.run([(59,x)])[0]['value']==0
        rollback(s,[(68,t,g,0,wire.bits(1)),(57,y,0,0),(68,t,g,0,wire.bits(1))],6)
        exact_values(s,t,expected);exact_state(s,g,model)
        rollback(s,[(68,t,g,0,wire.bits(1)),(58,y,0,2)],7)
        s.run([(33,3),(53,)]);exact_state(s,g,model)
        # Reclaimed private Saved records never include the Generator Hold.
        s.run([(68,x,g,0,wire.bits(1),1)]);value=fp.uniform(model.word(),0,wire.bits(1))
        exact_values(s,x,[value]);exact_state(s,g,model)
        r=s.run([(2,3,x,c),(57,wire.reference(1),0,0),(59,x)])
        assert r[-1]['value']==1 and s.run([(40,x,0,1)])[0]['payload']==struct.pack('<f',5)
        return {'backward_runtime_error':6,'response_records':2,'input_records':4,
            'prefix_draws_committed':4,'suffix_draws_committed':0,'atomic57_and_admission58_rollbacks':2,
            'generator_survives_graph_reset_and_saved_release':True}
    case('version-error58-commits-uniform-prefix-but57-rolls-back',version_error)

    def spent_error(s):
        x,c,t,g,y,model=create(s,19)
        done=s.run([(58,y,0,0),(59,x)])
        assert done[0]['value']==0 and done[1]['value']==1
        records=partial(s,[(68,t,g,0,wire.bits(1)),(58,y,0,0),(68,t,g,0,wire.bits(1))],10,2)
        expected=[fp.uniform(model.word(),0,wire.bits(1)) for _ in range(4)]
        exact_values(s,t,expected);exact_state(s,g,model)
        assert s.run([(40,x,0,1)])[0]['payload']==struct.pack('<f',3)
        rollback(s,[(68,t,g,0,wire.bits(1)),(57,y,0,0),(68,t,g,0,wire.bits(1))],10)
        exact_state(s,g,model);exact_values(s,t,expected)
        s.run([(33,3),(53,)])
        stats=struct.unpack('<12Q',s.run([(54,)])[0]['payload']);assert stats[9]==0
        clone=s.run([(65,g)])[0]['value'];exact_state(s,clone,model)
        s.run([(66,g)]);rollback(s,[(63,g)],1);exact_state(s,clone,model)
        return {'backward_runtime_error':10,'prior_leaf_gradient_retained':3,'prefix_draws_committed':4,
            'clone_after_reset_and_old_generator_release':True,'all_private_saved_reclaimed':True}
    case('spent-error58-and-generator-clone-after-saved-reclamation',spent_error)

    def growth_error(s):
        x,c,t,g,y,model=create(s,23);s.run([(12,c,0,wire.bits(7))]);old=s.owner()
        commands=[(68,t,g,0,wire.bits(1))]+[(60,1000+i) for i in range(100)]+[(58,y,0,0),(68,t,g,0,wire.bits(1))]
        rows=partial(s,commands,6,102);now=s.owner()
        assert now['length']>old['length'] and now['pointer']!=old['pointer'] and not s.mapped(old['pointer'])
        expected=[fp.uniform(model.word(),0,wire.bits(1)) for _ in range(4)]
        exact_values(s,t,expected);exact_state(s,g,model)
        assert wire.meta(s.run([(13,t)])[0])['version']==1
        for i in (0,49,99):exact_state(s,rows[i+1]['value'],mt.Reference(a.cpp_reference.resolve(),1000+i))
        # A later ordinary command still rolls back successful58 and prior RNG.
        good=s.run([(2,3,x,c)])[0]['value'];before=s.owner()
        rollback(s,[(68,t,g,0,wire.bits(1)),(58,good,0,0),(999,)],4)
        assert s.owner()==before;exact_state(s,g,model)
        successful=s.run([(58,good,0,0),(59,x)])
        assert successful[0]['value']==0 and successful[1]['value']==1
        for row in rows[1:101]:s.run([(66,row['value'])])
        s.run([(33,3),(53,)]);exact_state(s,g,model)
        return {'capacity_growth_before_partial_failure':True,'native_resize_bytes':[old['length'],now['length']],
            'committed_new_generators':100,'original_generator_draws':4,'suffix_draws':0,
            'successful58_then_ordinary_error_rolls_back_entire_frame':True}
    case('growth-replay-partial-failure-and-no-double-rng-consumption',growth_error)

    for path,raw in inputs.items():assert path.read_bytes()==raw,path
    report.update(status='PASS',case_count=len(report['cases']),input_sha256={str(p):SHA(raw) for p,raw in inputs.items()});save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','input_sha256')}))

if __name__=='__main__':main()
