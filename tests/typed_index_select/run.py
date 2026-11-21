#!/usr/bin/env python3
"""Execute native index_select and replay every original frame and owner snapshot."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import struct
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from typed_index_select.cases import CASES,CASE_SHA256,WIDTHS,raw_input
from numeric_predicates.native import Observer as BaseObserver,Live,Replay,Memory,load_wire,dump
from numeric_predicates.evidence import SUPPORT,LOCK

sha=lambda raw:hashlib.sha256(raw).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


class Inputs:
    def __init__(self):self.files={};self.records=self.files
    def bind(self,path,expected=None):
        path=Path(path).absolute();before=path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),'regular physical input: '+str(path))
        raw=path.read_bytes();after=path.stat()
        fields=lambda value:(value.st_dev,value.st_ino,value.st_mode,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        require(fields(before)==fields(after),'input changed during read: '+str(path))
        row=dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(after.st_mode))
        require(expected is None or row['sha256']==expected,'exact input hash: '+str(path))
        require(str(path) not in self.files or self.files[str(path)]==row,'input changed: '+str(path))
        self.files[str(path)]=row
        return raw
    def finish(self):
        for path,row in list(self.files.items()):self.bind(path,row['sha256'])
    recheck=finish


def module(path,name,bind):
    bind(path);spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def profile(repository,build,build_sha,inputs):
    custody=module(repository/'tests/pinned_reference_custody.py','typed_index_select_custody_profile',inputs.bind)
    for path in custody.source_files(repository):inputs.bind(path)
    qualification=module(repository/'tools/current_runtime/qualification.py','typed_index_select_current_profile',inputs.bind)
    built=qualification.admit_build(build,inputs.bind,build_sha)
    files=[repository/'tests/typed_index_select'/name for name in ('__init__.py','cases.py','reference.py','run.py')]
    files += [repository/'tests/numeric_predicates'/name for name in ('__init__.py','cases.py','native.py','reference.py','evidence.py')]
    files += [repository/'tests/half_logical'/name for name in ('__init__.py','native.py')]
    for file in files:inputs.bind(file)
    combined=module(repository/'tools/current_runtime/index_signbit_source_transition.py','typed_index_select_combined_source_proof',inputs.bind)
    source=inputs.bind(repository/'tools/current_runtime/source/runtime.hepta')
    parent,proof=combined.restore_source(source,inputs.bind)
    require(proof['source_sha256']==built['source_sha256'] and proof['functions']==797
        and proof['parent_functions']==775 and proof['unchanged_parent_functions']==760
        and proof['changed_functions']==['entry', 'linear', 'reduce_axis', 'softmax', 'layernorm', 'batchnorm_eval', 'conv2d', 'pool2d', 'matmul', 'index_select', 'gather', 'tensor_join', 'indexed_add', 'predicate_bits', 'predicate_command']
        and proof['appended_functions']==['typed_index_select_previous', 'typed_index_select', 'typed_index_select_offset', 'indexed_gather_forward_source', 'indexed_add_forward_inputs', 'indexed_add_forward_location', 'matmul_forward_index', 'matmul_forward_point', 'softmax_forward_group', 'axis_source_forward', 'conv_term_forward', 'conv_no_grad_forward', 'pool_point_forward', 'pool_no_grad_forward', 'matmul_output_version', 'concat_forward_target', 'concat_forward_part', 'concat_forward_copy', 'concat_forward_layout', 'concat_forward', 'linear_bias_output_version', 'normalization_output_layout']
        and proof['whole_source_inverse_exact'] is True and proof['whole_round_trip_exact'] is True
        and proof['original_f32_index_select_function_retained'] is True
        and proof['original_arithmetic_and_backward_functions_retained'] is True,
        'complete797 source inverse and all original F32 AD bodies retained')
    return built


class Observer(BaseObserver):
    def setup(self,spec):
        raw=raw_input(spec)
        if spec.get('legacy_bool'):raw=bytes(math.prod(spec['shape']))
        dtype=0 if spec.get('cast_from_f32') else spec['dtype']
        if spec['strides'] is None:
            source=self.create(dtype,spec['shape'],raw,spec.get('requires_grad',False));handles=[source]
        else:
            backing=self.create(dtype,[spec['storage_count']],raw)
            shape,strides=spec['shape'],spec['strides']
            cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),spec['offset'])
            source=self.one(111,backing,self.wire.Payload(cfg),80)['value'];handles=[source,backing]
        if spec.get('cast_from_f32'):
            source=self.one(108,source,spec['dtype'],0,0)['value'];handles.insert(0,source)
        return source,handles
    def command(self,spec,source):
        data=struct.pack('<'+'Q'*len(spec['indices']),*(v&((1<<64)-1) for v in spec['indices']))
        return (30,source,spec['axis'],len(spec['indices']),self.wire.Payload(data))


def expected_status(row):
    if row['accepted']:return 0
    text=row['error']['message']
    if text.startswith('Dimension out of range') or 'Index to scalar can have only 1 value' in text:return 2
    if 'out of' in text or 'indexing axis dim should be positive' in text:return 7
    raise ValueError('unmapped actual Torch error: '+text)


def transaction(n,row,invalid=False):
    spec=row['spec'];source=spec['input'];raw=raw_input(source)
    count=3 if invalid else 2;end=32+64*count;data_at=end+64
    indices_at=(data_at+len(raw)+7)&~7
    indices=struct.pack('<'+'Q'*len(spec['indices']),*(v&((1<<64)-1) for v in spec['indices']))
    body=b'HTORCH03'+struct.pack('<3Q',count,indices_at+len(indices),0)
    body+=struct.pack('<8Q',97,source['dtype'],end,0,0,0,0,0)
    body+=struct.pack('<8Q',30,n.wire.reference(1),spec['axis']&((1<<64)-1),len(spec['indices']),indices_at,0,0,0)
    if invalid:body+=struct.pack('<8Q',125,n.wire.reference(1),0,0,0,0,0,0)
    shape=source['shape']
    body+=struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),data_at,len(raw))+raw
    body+=bytes(indices_at-len(body))+indices
    return body


def controls(n,row,source):
    spec=row['spec'];command=n.command(spec,source);count=0
    for token in (0,1):
        status,_,_,_=n.issue(n.wire.packet([n.command(spec,token)]));n.check('invalid handle exact rollback',status,1);count+=1
    stale=n.create(spec['input']['dtype'],[1],bytes(WIDTHS[spec['input']['dtype']]))
    n.one(41,stale)
    status,_,_,_=n.issue(n.wire.packet([n.command(spec,stale)]));n.check('stale handle exact rollback',status,1);count+=1
    good=n.wire.packet([command])
    for offset in (0,32,97,len(good)+8,(1<<64)-8):
        packet=bytearray(good);struct.pack_into('<Q',packet,32+4*8,offset)
        status,_,_,_=n.issue(bytes(packet));n.check('unaligned or out-of-range index payload rejected',status,100);count+=1
    packet=bytearray(good);struct.pack_into('<Q',packet,32+3*8,len(spec['indices'])+1)
    status,_,_,_=n.issue(bytes(packet));n.check('truncated index payload rejected',status,100);count+=1
    packet=bytearray(good);struct.pack_into('<Q',packet,32+3*8,262145)
    status,_,_,_=n.issue(bytes(packet));n.check('index capacity limit checked before payload read',status,3);count+=1
    maps=n.s.maps();status,_,_,_=n.issue(n.wire.packet([command,(125,source)]))
    n.check('successful index selection and invalid suffix rollback',status,4)
    n.check('all mappings restored after suffix failure',n.s.maps(),maps);count+=1
    status,records,_,_=n.issue(transaction(n,row))
    n.check('same-frame factory and FE index_select succeed',status,0)
    n.check('same-frame result actual reference',n.tensor(records[1]['value'])[0],n.expected(row['result']))
    n.one(41,records[0]['value']);n.one(41,records[1]['value']);count+=1
    maps=n.s.maps();status,_,_,_=n.issue(transaction(n,row,True))
    n.check('same-frame factory/index-select prefix rolls back',status,4)
    n.check('same-frame mappings fully restored',n.s.maps(),maps);count+=1
    return count


def compare(n,row):
    spec=row['spec'];generator,rng=n.begin();source,handles=n.setup(spec['input'])
    original,source_node=n.tensor(source)
    n.check('complete independently observed source bytes and metadata',original,n.expected(row['input']))
    status,records,before,after=n.issue(n.wire.packet([n.command(spec,source)]))
    n.check('actual Torch acceptance/error',status,expected_status(row))
    n.check('source bytes, shape, layout and version unchanged',n.tensor(source)[0],original)
    if not spec['input'].get('requires_grad'):
        n.check('every source descriptor word unchanged',Memory(before['bytes']).node(source),Memory(after['bytes']).node(source))
    control_count=0;elements=0
    if status==0:
        result=records[0]['value'];actual,node=n.tensor(result)
        n.check('complete actual Torch result and storage bytes',actual,n.expected(row['result']))
        n.check('result has independent Tensor Storage Version',[result!=source,node[27]!=source_node[27],node[28]!=source_node[28]],[True]*3)
        if spec['input']['dtype']!=0:
            edges=(32,40,48,56,72,80,88,176,184,248,280,288,296,312,360,384,392,400,408,584,592,600,608,616,624,640,648,656,664)
            n.check('typed output has no input, view or gradient retention',[node[o//8] for o in edges],[0]*len(edges))
        if spec.get('controls'):control_count=controls(n,row,source)
        if row.get('gradient'):
            seed=n.create(0,actual['shape'],struct.pack('<I',0x3f800000)*actual['numel'])
            n.one(29,result,seed)
            gradient=n.one(11,source)
            n.check('actual F32 autograd gradient bytes',gradient['payload'],n.reference['blob'](row['gradient']['logical']))
            n.check('actual F32 autograd gradient length',gradient['value'],row['gradient']['numel'])
            n.one(41,seed)
        for token in handles:n.one(41,token)
        n.check('complete output survives releasing all source handles',n.tensor(result)[0],actual)
        n.one(41,result);elements=actual['numel']
    else:
        for token in handles:n.one(41,token)
    n.end(generator,rng)
    return dict(index=row['index'],accepted=status==0,status=status,elements=elements,controls=control_count,gradient_checked=bool(row.get('gradient')))


def allocation(n,row):
    spec=row['spec'];generator,rng=n.begin();source,handles=n.setup(spec['input']);original=n.tensor(source)[0]
    packet=n.wire.packet([n.command(spec,source)]);live=[]
    n.s.lower_limit(n.s.owner())
    for attempt in range(32):
        status,records,_,_=n.issue(packet)
        if status:
            n.check('real child allocation failure',status,3);break
        live.append(records[0]['value'])
    else:raise ValueError('no actual OOM within 32 attempts')
    n.s.restore_limit();status,records,_,_=n.issue(packet)
    n.check('identical request succeeds after restoring real child limit',status,0)
    result=records[0]['value'];live.append(result)
    n.check('recovery result equals original actual Torch',n.tensor(result)[0],n.expected(row['result']))
    n.check('source unchanged across OOM and recovery',n.tensor(source)[0],original)
    for token in live+handles:n.one(41,token)
    n.end(generator,rng)
    return dict(index=row['index'],attempts=attempt+1,actual_allocation_failures=1,identical_request_recovered=True)


def original_bool_successor(n,row):
    """Retain the original setup and three-command Bool admission request."""
    w=n.wire
    def one(*args):return n.one(*args)
    def boolean(shape):return n.create(1,shape,bytes(math.prod(shape)))
    def cfg(*values):return w.Payload(struct.pack('<'+'Q'*len(values),*values))
    def group(algorithm,parameter,include_count=True):
        words=([1] if include_count else [])+[1,0]+[0]*(6 if algorithm==0 else 8)+[parameter]
        return cfg(*words)
    def defaults(algorithm):
        if algorithm==0:return cfg(w.bits(.125),w.bits(.5),0,0,0,0)
        return cfg(w.bits(.125),w.bits(.5),w.bits(.75),w.bits(.001),0,0,0,0)
    # This is the original dtype-admission101 setup, in its original allocation order.
    b=boolean([2,2]);bv=boolean([2]);b4=boolean([1,1,2,2]);be=boolean([0])
    mutation_target=boolean([1]);bool_grad=one(18,b)['value']
    f=one(*w.full([2,2],.25))['value'];p=one(*w.full([2,2],.25,1))['value']
    v=one(*w.full([2],1))['value'];f4=one(*w.full([1,1,2,2],.25))['value']
    fkernel=one(*w.full([1,1,1,1],1))['value'];bkernel=boolean([1,1,1,1])
    state1=one(*w.full([2,2],0))['value'];state2=one(*w.full([2,2],0))['value']
    gen=one(60,1729)['value'];complete_rng=one(63,gen)['payload']
    weak=one(42,b)['value'];hold=one(43,weak)['value'];saved=one(46,b)['value']
    empty_weak=one(42,be)['value'];empty_hold=one(43,empty_weak)['value'];empty_saved=one(46,be)['value']
    opt=one(71,0,defaults(0),group(0,p))['value']
    source_before,node=n.tensor(b)
    n.check('original Bool source is actual all-zero reference',source_before,n.expected(row['input']))
    prefix=row['legacy_prefix'];n.check('original complete Generator state',complete_rng,bytes.fromhex(prefix['rng_before']))
    command=(30,b,0,2,cfg(0,1))
    commands=[(102,mutation_target),(68,f,gen,0,w.bits(1),1),command]
    packet=w.packet(commands)
    # The literal fields, allocation order, index payload and mutation prefix are preserved.
    status,records,_,_=n.issue(packet)
    n.check('original three-command Bool request now succeeds numerically',status,0)
    n.check('original three operation responses',[r['opcode'] for r in records],[102,68,30])
    result=records[2]['value'];actual,result_node=n.tensor(result)
    n.check('original Bool request actual output',actual,n.expected(row['result']))
    n.check('original Bool output independent Tensor Storage Version',
        [result!=b,result_node[27]!=node[27],result_node[28]!=node[28]],[True]*3)
    n.check('original logical-not prefix actually commits',n.tensor(mutation_target)[0],n.expected(prefix['mutation_result']))
    n.check('original uniform prefix actual Torch values/layout/version',n.tensor(f)[0],n.expected(prefix['floating_result']))
    n.check('original uniform prefix complete Generator state',one(63,gen)['payload'],bytes.fromhex(prefix['rng_after']))
    n.check('index-select preserves complete original Bool input',n.tensor(b)[0],source_before)
    maps=n.s.maps();status,_,_,_=n.issue(w.packet(commands+[(125,b)]))
    n.check('original successful prefix plus invalid suffix rolls back',status,4)
    n.check('entire original-prefix mapping set restores',n.s.maps(),maps)
    n.check('entire Generator state restores',one(63,gen)['payload'],bytes.fromhex(prefix['rng_after']))
    n.check('original current owner audit',one(53)['value'],1)
    return dict(index=row['index'],original_request_bytes=len(packet),original_request_sha256=sha(packet),
        original_three_command_fields_preserved=True,original_setup_allocation_order_preserved=True,
        actual_uniform_and_logical_not_prefix_checked=True,complete_suffix_rollback_checked=True)


def reference(repository,directory,execution,inputs,allow=False):
    raw=inputs.bind(directory/'report.json');report=json.loads(raw)
    require(report['schema']=='heptatorch.typed-index-select.reference.v1' and report['case_sha256']==CASE_SHA256
        and report['expected_cases']==report['cases']==len(report['rows'])==len(CASES)
        and report['Torch_executed'] is True and report['native_executed'] is False
        and report['compiler_executed'] is False and report['failures']==[]
        and report['host_expected_values_computed'] is False and report['inputs_unchanged'] is True
        and report['rng_unchanged'] is True and report['cpu_capability']=='AVX2' and report['torch_threads']==1
        and report['all_observed_blobs_unchanged'] is True,
        'complete actual index_select reference')
    wanted='OBSERVED_EXPLORATORY_INDEX_SELECT' if allow else 'OBSERVED_TYPED_INDEX_SELECT'
    require(report['status']==wanted and report['production_oracle_qualified'] is (not allow),
        'strict fixed-wheel provenance is mandatory unless explicitly exploratory')
    provenance=report['oracle_provenance']
    require(provenance['status']==('EXPLORATORY_NOT_PINNED_WHEEL_QUALIFICATION' if allow else 'PASS')
        and len(provenance['checks'])==(5 if allow else 11)
        and all(value is True for value in provenance['checks'].values())
        and provenance['lock_sha256']==LOCK,'actual installed Torch identity and original fixed-wheel controls')
    inputs.bind(repository/'tests/pytorch_oracle.py',SUPPORT)
    lock=json.loads(inputs.bind(repository/'compatibility/pytorch-oracle-lock.json',LOCK))
    require(provenance['expected']==lock['oracle'],'original official wheel lock unchanged')
    if allow:
        require(provenance['full_installation_provenance_verified'] is False,'exploratory results grant no CI qualification')
        for key in ('package','extension'):
            row=provenance[key];path=Path(row['path']);data=inputs.bind(path,row['sha256'])
            require(row==dict(path=str(path),bytes=len(data),sha256=sha(data),mode=stat.S_IMODE(path.stat().st_mode)),
                'actual local Torch binary and package retained unchanged')
    custody=module(repository/'tests/pinned_reference_custody.py','typed_index_select_custody_reader',inputs.bind)
    custody_proof=custody.read(repository,directory,execution,report,inputs.bind,'tests/typed_index_select/reference.py',allow)
    process=json.loads(inputs.bind(execution/'process.json'))
    require(process['started'] is True and process['exit_code']==0
        and process['stdout_EOF'] is True and process['stderr_EOF'] is True
        and process['forced_cleanup'] is False and not process.get('timed_out')
        and not process.get('error') and not process.get('execution_error')
        and not process['cleanup_errors'] and not process['running_descendants_after_cleanup']
        and not process['terminated_process_groups'],'actual reference exit, both EOFs and normal completion')
    expected_args=['-B',str(repository/'tests/typed_index_select/reference.py'),'--output',str(directory)]
    if allow:expected_args.append('--allow-unverified-local')
    argv=process['argv']
    require(argv[1:]==expected_args and report['actual_process']['argv']==argv
        and report['actual_process']['pid']==process['pid'],'exact supervised independent Torch argv and PID')
    interpreter=Path(argv[0]).resolve(strict=True);python=report['actual_process']['python'];data=inputs.bind(interpreter)
    require(python==dict(path=str(interpreter),bytes=len(data),sha256=sha(data),mode=stat.S_IMODE(interpreter.stat().st_mode)),
        'actual exact independent Torch interpreter')
    for channel in ('stdout','stderr'):
        stream=inputs.bind(execution/('runner.'+channel),process[channel+'_sha256'])
        require(len(stream)==process[channel+'_bytes'],'complete actual reference '+channel)
    require(inputs.bind(execution/'runner.stderr')==b'','clean actual Torch reference stderr')
    require(json.loads(inputs.bind(execution/'runner.stdout'))==dict(status=wanted,cases=866,expected_cases=866,
        production_oracle_qualified=not allow,error=None),'complete original reference stdout')
    files=[repository/'tests/typed_index_select'/name for name in ('__init__.py','cases.py','reference.py')]
    files += [repository/'tests/numeric_predicates'/name for name in ('__init__.py','cases.py','reference.py')]
    files += [repository/'tests/pytorch_oracle.py',repository/'compatibility/pytorch-oracle-lock.json',
        repository/'compatibility/upstream-lock.json',repository/'tests/pinned_reference_custody.py',interpreter]
    files += custody.source_files(repository)
    if not allow:files.append(repository/'build/pytorch-install.json')
    require(set(report['inputs'])=={str(path.resolve()) for path in files},'complete actual reference input census')
    for path,record in report['inputs'].items():
        data=inputs.bind(path,record['sha256'])
        require(record==dict(path=path,bytes=len(data),sha256=sha(data),mode=stat.S_IMODE(Path(path).stat().st_mode)),
            'complete unchanged original reference input bytes and modes')
    def blob(item):
        require(set(item)=={'path','bytes','sha256'} and item['path']=='blobs/'+item['sha256']+'.bin','canonical actual reference blob')
        raw=inputs.bind(directory/item['path'],item['sha256']);require(len(raw)==item['bytes'],'full actual reference blob');return raw
    for index,(row,spec) in enumerate(zip(report['rows'],CASES)):
        require(row==json.loads(inputs.bind(directory/('case-%04d.json'%index))) and row['index']==index and row['spec']==spec,
            'exact literal case input and original observed record')
        require(row['input_unchanged'] is True and row['rng_unchanged'] is True,
            'actual reference preserved original input and global RNG')
        if row['accepted']:require(row['fresh_tensor'] is True and row['fresh_storage'] is True,'actual independent result ownership')
        for key in ('input','result','gradient'):
            if key in row:
                for field in ('logical','storage'):blob(row[key][field])
        if 'legacy_prefix' in row:
            for key in ('mutation_result','floating_result'):
                for field in ('logical','storage'):blob(row['legacy_prefix'][key][field])
    return dict(report=report,blob=blob,root=directory,execution=execution,report_sha256=sha(raw),custody=custody_proof)


def recipes(ref):
    rows=ref['report']['rows']
    result=[('case-%04d'%row['index'],compare,row) for row in rows]
    result += [('allocation-%04d'%row['index'],allocation,row) for row in rows if row['spec']['allocation']]
    result += [('legacy-%04d'%row['index'],original_bool_successor,row) for row in rows if row['spec']['input'].get('legacy_bool')]
    require(len(result)==869,'866 numerical cases, two allocation recoveries and one exact Bool packet successor')
    return result


def qualify(report,repository,build,build_sha,output,inputs,allow_unverified_local=False):
    """Reapply every numerical assertion to original physical frames and Owner bytes."""
    built=profile(repository,build,build_sha,inputs)
    require(report==json.loads(inputs.bind(output/'report.json')),'unchanged original native report')
    wanted='PASS_EXPLORATORY_NATIVE_ONLY' if allow_unverified_local else 'PASS'
    require(report['schema']=='heptatorch.typed-index-select.native.v1' and report['group']=='typed_index_select'
        and report['status']==wanted and report['cases']==report['expected_cases']==len(CASES)==866
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['source_sha256']==built['source_sha256'] and report['native_sha256']==built['native_sha256']
        and report['actual_build_sha256']==build_sha and report['native_executed'] is True
        and report['Torch_executed'] is True and report['compiler_executed'] is False
        and report['host_expected_values_computed'] is False and report['original_reference_modified'] is False
        and report['all_inputs_unchanged'] is True and report['full_pytorch_rewrite_complete'] is False
        and report['allow_unverified_local'] is allow_unverified_local
        and report['production_oracle_qualified'] is (not allow_unverified_local)
        and report['full_installation_provenance_verified'] is (not allow_unverified_local),
        'exact actual independent observation identity and qualification scope')
    for path,row in report['inputs'].items():
        raw=inputs.bind(path,row['sha256'])
        require(row==dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(Path(path).stat().st_mode)),
            'every original observer input retained unchanged')
    fixed=reference(repository,Path(report['reference_directory']),Path(report['reference_process_directory']),
        inputs,allow_unverified_local)
    require(report['reference_custody']==fixed['custody'],'complete raw fixed-wheel and actual process custody retained')
    require(fixed['report_sha256']==report['reference_sha256'],'exact original independent actual Torch reference')
    wire=load_wire(repository,inputs.bind);cases=recipes(fixed)
    require(len(report['results'])==len(cases),'complete original numerical and control observation census')
    totals=dict(frames=0,owner_reads=0,mapping_reads=0,checks=0)
    for saved,(name,runner,row) in zip(report['results'],cases):
        require(saved==json.loads(inputs.bind(output/name/'observation.json'))
            and saved['name']==name and saved['status']=='PASS','exact original per-session observation')
        session=Replay(build/'heptatorch',wire,output/name,inputs.bind);observer=Observer(session,wire,fixed)
        result=runner(observer,row);finish=session.finish()
        require(result==saved['result'] and finish==saved['finish'] and observer.checks==saved['checks'],
            'every original numerical metadata ownership lifetime and resource assertion reapplied')
        for key in totals:totals[key]+=observer.checks if key=='checks' else finish[key]
    require(totals==report['totals'] and all(report[key]==totals[key] for key in totals)
        and totals['mapping_reads']==26,'complete physical frame Owner and mapping census')
    ordinary=report['results'][:866];allocations=report['results'][866:868];legacy=report['results'][868]
    require(sum(row['result']['controls'] for row in ordinary)==78
        and sum(row['result']['gradient_checked'] for row in ordinary)==5
        and all(row['result']['actual_allocation_failures']==1
            and row['result']['identical_request_recovered'] is True for row in allocations)
        and legacy['result']['original_three_command_fields_preserved'] is True
        and legacy['result']['original_setup_allocation_order_preserved'] is True
        and legacy['result']['actual_uniform_and_logical_not_prefix_checked'] is True
        and legacy['result']['complete_suffix_rollback_checked'] is True,
        'complete original control, actual autograd, OOM and Bool successor coverage')
    return dict(cases=866,frames=totals['frames'],assertions=totals['checks'],native_sessions=869,
        owner_snapshots=totals['owner_reads'],mapping_snapshots=totals['mapping_reads'],physical_replay=True,
        native_executed_by_qualifier=False,Torch_executed_by_qualifier=False,
        complete_ci_admission=not allow_unverified_local,full_installation_provenance_verified=not allow_unverified_local,
        production_oracle_qualified=not allow_unverified_local,reference_sha256=fixed['report_sha256'],
        additional_protocol_controls=78,additional_allocation_recoveries=2,original_bool_packet_numerical_successors=1,
        actual_autograd_cases=5,host_expected_values_computed=False,original_f32_ad_function_preserved=True,
        original_eleven_checks_recomputed_from_raw_inputs=True,
        raw_reference_entities=fixed['custody']['raw_reference_entities'],
        original_installation_checks=11,
        original_installation_checks_passed=sum(fixed['custody']['original_checks'].values()),
        actual_supervisor_intent_verified=True,actual_environment_transition_verified=True,
        original_reference_script_bytes_verified=True,portable_reference_custody_replayed=True,
        **{key:fixed['custody'][key] for key in ('official_wheel_sha256', 'official_member_anchor_sha256', 'independent_official_wheel_member_anchor_verified', 'complete_official_wheel_archive_verified', 'installed_wheel_members_verified', 'installed_wheel_bytes_verified', 'torch_shared_libraries_verified', 'actual_mapped_torch_cpu_code_verified', 'all_actual_installed_inputs_unchanged', 'complete_installed_raw_snapshot_verified')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-directory',type=Path,required=True)
    p.add_argument('--expected-build-sha256',required=True)
    p.add_argument('--torch-python',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-unverified-local',action='store_true',help='Exploratory work only; cannot pass complete CI admission.')
    args=p.parse_args();require(not sys.flags.optimize,'independent numerical predicates must be enabled')
    output=args.output.resolve();build=args.build_directory.resolve();output.mkdir(parents=True,exist_ok=False)
    inputs=Inputs();started=time.monotonic()
    report=dict(schema='heptatorch.typed-index-select.native.v1',group='typed_index_select',status='FAIL',results=[],failures=[],
        cases=len(CASES),expected_cases=len(CASES),case_sha256=CASE_SHA256,frames=0,owner_reads=0,mapping_reads=0,checks=0,
        native_executed=False,Torch_executed=False,compiler_executed=False,host_expected_values_computed=False,
        original_reference_modified=False,actual_build_sha256=args.expected_build_sha256,
        full_pytorch_rewrite_complete=False,full_project_status='HOLD',runtime_qualification_transferred=False,
        allow_unverified_local=args.allow_unverified_local,production_oracle_qualified=False,
        full_installation_provenance_verified=False)
    try:
        inputs.bind(Path(sys.executable).resolve())
        built=profile(ROOT,build,args.expected_build_sha256,inputs)
        report.update(source_sha256=built['source_sha256'],native_sha256=built['native_sha256'])
        supervisor=module(ROOT/'tests/current_runtime_process.py','index_select_reference_supervisor',inputs.bind)
        execution=output/'reference-execution';execution.mkdir()
        argv=[str(args.torch_python.absolute()),'-B',str(ROOT/'tests/typed_index_select/reference.py'),
            '--output',str(output/'reference')]
        if args.allow_unverified_local:argv.append('--allow-unverified-local')
        custody=module(ROOT/'tests/pinned_reference_custody.py','typed_index_select_custody_supervisor',inputs.bind)
        process=custody.supervise(supervisor,argv,execution,300,ROOT,inputs.bind)
        dump(execution/'process.json',process)
        fixed=reference(ROOT,output/'reference',execution,inputs,args.allow_unverified_local)
        report.update(Torch_executed=True,reference_directory=str(fixed['root']),
            reference_process_directory=str(execution),reference_sha256=fixed['report_sha256'],reference_custody=fixed['custody'],
            production_oracle_qualified=fixed['report']['production_oracle_qualified'],
            full_installation_provenance_verified=fixed['report']['production_oracle_qualified'])
        wire=load_wire(ROOT,inputs.bind);cases=recipes(fixed)
        totals=dict(frames=0,owner_reads=0,mapping_reads=0,checks=0)
        for ordinal,(name,runner,case) in enumerate(cases):
            row=dict(name=name,status='FAIL');session=observer=None
            try:
                session=Live(build/'heptatorch',wire,output/name,inputs.bind);report['native_executed']=True
                observer=Observer(session,wire,fixed);row['result']=runner(observer,case);row['status']='PASS'
            except BaseException as error:
                row['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            finally:
                if session is not None:
                    try:row['finish']=session.finish()
                    except BaseException as error:
                        row['status']='FAIL';row['finish_error']=dict(type=type(error).__name__,message=str(error))
                row['checks']=observer.checks if observer is not None else 0
                if (output/name).exists():dump(output/name/'observation.json',row)
            report['results'].append(row)
            for key in totals:totals[key]+=row['checks'] if key=='checks' else row.get('finish',{}).get(key,0)
            if row['status']!='PASS':report['failures'].append(name)
            if ordinal%64==0 or row['status']!='PASS':
                print(json.dumps(dict(group='typed_index_select',recipe=name,status=row['status'],error=row.get('error'))),flush=True)
        inputs.finish();report.update(all_inputs_unchanged=True,totals=totals,**totals)
        if not report['failures'] and len(report['results'])==869:
            report['status']='PASS_EXPLORATORY_NATIVE_ONLY' if args.allow_unverified_local else 'PASS'
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    report.update(inputs=inputs.files,elapsed_seconds=time.monotonic()-started)
    dump(output/'report.json',report)
    if report['status'] in ('PASS','PASS_EXPLORATORY_NATIVE_ONLY'):
        try:
            replay_inputs=Inputs()
            proof=qualify(report,ROOT,build,args.expected_build_sha256,output,replay_inputs,args.allow_unverified_local)
            replay_inputs.finish();dump(output/'physical-qualification.json',proof)
        except BaseException as error:
            dump(output/'physical-qualification.json',dict(status='FAIL',error=str(error),traceback=traceback.format_exc()))
            print(json.dumps(dict(status='FAIL_PHYSICAL_REPLAY',error=str(error))),flush=True)
            return 1
    print(json.dumps({key:report.get(key) for key in ('group','status','cases','frames','checks','failures','error')}),flush=True)
    return 0 if report['status'] in ('PASS','PASS_EXPLORATORY_NATIVE_ONLY') else 1


if __name__=='__main__':raise SystemExit(main())
