#!/usr/bin/env python3
"""Read old wide-neg physically on the exact joint image, with explicit abs successors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import types

BASE = '083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'
DUAL = '5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'
RUNNER = 'tests/wide_neg/run.py'
ADMIT = 'tools/current_runtime/opt_in_admission/admit.py'
PINS = {
    'compatibility/pytorch-oracle-lock.json': 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944',
    'compatibility/upstream-lock.json': 'd204c5a8268a0d0f0a1dd162b5e0ab184beee83310bcad643b084deb691f3f49',
    'tests/current_runtime_process.py': '36e3432deac48085ba0586c85547cee8432d1533c5915238db2439b43dbca523',
    'tests/current_runtime_typed_relu_controls.py': 'b0cdca3b07c72023388622bb7499bfc487333c91c1eabf97a53c2b214d1d92e6',
    'tests/differentiable_view_native_review/owner_client.py': 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b',
    'tests/differentiable_view_native_review/protocol.py': '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9',
    'tests/half_logical/__init__.py': '4c5fbdf0c7f8da54986ba8c0aef475abc92284941bcf215c0545a3528dc8b8fb',
    'tests/half_logical/native.py': '9c2cd19a5bb04eaa037f5905a52bce2dcc31042101f72d07e71205745160a14f',
    'tests/half_logical/reference.py': '5bb6a3804717968dba37f3174809a7c1eee5b225f0a3085b44fcec7656155528',
    'tests/half_logical/run.py': 'f358bc08f4d35ad48c16ebd024f2341fb4bde8a8f58cab2746635912a0fc7199',
    'tests/half_neg/__init__.py': 'f02584b81106f7440ca54bf8f59beb9a3e05feb8c1123964325d5e946b839163',
    'tests/half_neg/cases.py': '643da7cd50f0b9364444bac11a091b99b1fd6670b91979dca40e3227ee07cd69',
    'tests/half_neg/reference.py': 'b6f7ee4cd1ec107800abb650ce0aabbd63fd7edbbbb642a2290c5ae4d178a2bf',
    'tests/half_neg/run.py': 'f36fe06d82b539d5f4d3324a91ce2d0d9ab6a9d2033892e274d8c1377c53bec2',
    'tests/numeric_predicates/__init__.py': '7d9b42b2f3b86a533773fdb030f2cb539632f68ffdb22874a5071d0a4cc1b64f',
    'tests/numeric_predicates/cases.py': 'e69bebed091db88ef81eda2ea2052af6b2d88c74840909ee27e2cbdf6a9ec694',
    'tests/numeric_predicates/evidence.py': 'a3d39ee66d5511d49bb484bc9b95a131504085679546d25266b281e26b879c84',
    'tests/numeric_predicates/native.py': 'c0e9d1ba6f0dac5499f091333611a5bea0fb6f4a25e74faa39f35486fd404d2d',
    'tests/numeric_predicates/reference.py': '3ec85934110f2d8f273bc1e209a9c9ce13331cc4ff915dc9d7058d81913d0abd',
    'tests/numeric_predicates/run.py': 'e9fe4a7361cef25aca14c24ecd17c5413f058e5ccb3c35190d66b0be3b479991',
    'tests/pytorch_oracle.py': 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae',
    'tests/typed_relu/__init__.py': '838cea508ffc398ae7319d22d3666cea7a0a06050b24fc67aa025d1d43d2a481',
    'tests/typed_relu/cases.py': '385968517dbd6f5ecd5606384fb8b8920544e0195fc4bf349dc0df76524538ac',
    'tests/typed_relu/reference.py': 'f0ecb1e92449743714c24d7ef50ede74477075c3298fbde3426e397617ef4b26',
    'tests/typed_relu/run.py': '4bcaad8ebe594c7e09985159f0f13c5ab3f18fbc43cdb0b03c9a5dde08789edf',
    'tests/wide_neg/__init__.py': '01ea777820ede90061e80091e3f765f6eac9844d2c43d60dab95dcb3c48a2ba6',
    'tests/wide_neg/cases.py': '5796a0561637004f3cd5a5ac16cc0612e0ed51bd0c381c0440c2ced666ec0b84',
    'tests/wide_neg/reference.py': 'b40d811d0d3a030f45a8d086b6cf7df1e549cb1be778a6f7ea430346ec342ba3',
    'tests/wide_neg/run.py': '2ea0b003bdc371446da8efee479fb1b501808410fe9e48de8232babcbd4ad35f',
    'tools/current_runtime/opt_in_admission/admit.py': '51b32b1835fe8a294f1f42b98ca3b87b10a5096dec08d34840ac7acab61f672d',
    'tools/current_runtime/qualification.py': 'f4be6d41c29659c3132f1f64014ad9d8ebcf612db3a9cb556ee5ddd178a540b6',
}
REFUSAL = 'unimplemented I64/F64 unary remains an explicit refusal'
TARGETS = {'i64-protocol-controls', 'successor-unsupported-F64-neg'}
OLD = b"""            else:
                refused,_,_,_=n.issue(n.wire.packet([(3,operation,source)]))
                n.check('unimplemented I64/F64 unary remains an explicit refusal',refused,7)
"""
NEW = b"""            elif operation==7:
                check_abs(n,row,source)
            else:
                refused,_,_,_=n.issue(n.wire.packet([(3,operation,source)]))
                n.check('unimplemented I64/F64 unary remains an explicit refusal',refused,7)
"""
ORACLE = r"""
import json, os, runpy, sys, hashlib
from pathlib import Path
repo, output, original = map(Path, sys.argv[1:])
os.environ['ATEN_CPU_CAPABILITY']='avx2'
sys.argv=[str(repo/'tests/pytorch_oracle.py'),str(output/'NO_NATIVE_EXECUTION'),
          str(output/'oracle-provenance.json'),str(repo/'compatibility/pytorch-oracle-lock.json')]
ctx=runpy.run_path(sys.argv[0],run_name='wide_neg_abs_reference_only')
torch=ctx['torch']
assert torch.backends.cpu.get_cpu_capability()=='AVX2' and torch.get_num_threads()==1
native=runpy.run_path(str(repo/'tests/wide_neg/reference.py'),run_name='wide_neg_abs_reference_source')
report=json.loads(original.read_bytes())
rows={}
rng=bytes(torch.get_rng_state().tolist())
for row in report['rows']:
    if not row['spec'].get('controls'):
        continue
    x=native['setup'](torch,row['spec']['input'])
    obs=native['tensor'](torch,x,output)
    assert obs==row['input']
    y=torch.abs(x)
    assert native['tensor'](torch,x,output)==obs and y.untyped_storage()._cdata!=x.untyped_storage()._cdata
    rows[row['spec']['name']]=dict(input=obs,result=native['tensor'](torch,y,output))
assert bytes(torch.get_rng_state().tolist())==rng
print(json.dumps(dict(rows=rows,torch_version=torch.__version__,cpu_capability='AVX2',
                      torch_threads=1,provenance=ctx['PROVENANCE'])))
"""


def need(ok,message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path,pin=None):
    need(path.resolve()==path and stat.S_ISREG(path.lstat().st_mode),f'ordinary input required: {path}')
    a=path.stat()
    raw=path.read_bytes()
    b=path.stat()
    def identity(x):
        return (x.st_dev,x.st_ino,x.st_size,x.st_mode,x.st_mtime_ns,x.st_ctime_ns)
    need(identity(a)==identity(b) and b.st_size==len(raw),f'input changed: {path}')
    need(pin is None or sha(raw)==pin,f'wrong bytes: {path}')
    return raw


def dump(path,obj):
    with path.open('x') as f:
        f.write(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')


def load(path,raw,nspace=None):
    m=types.ModuleType('wide_neg_dual_projection')
    m.__file__=str(path)
    m.__dict__.update(nspace or {})
    exec(compile(raw,path,'exec',dont_inherit=True,optimize=0),m.__dict__)
    return m


def main(a):
    repo,base,build,scoped,out=a.repository,a.baseline_run,a.dual_build,a.dual_scoped,a.output
    paths={p:read(repo/p,h) for p,h in PINS.items()}
    sources={repo/p for p in paths if p.endswith('.py')}
    need(all(not (p.parent/'__pycache__').exists() for p in sources),
         'use a clean repository without observer bytecode caches')
    origin=read(base/'report.json')
    baseline=json.loads(origin)
    original_physical=json.loads(read(base/'physical-qualification.json'))
    need(baseline['status']=='PASS' and baseline['native_sha256']==BASE and
         baseline['cases']==len(baseline['results'])==405 and not baseline['failures'] and
         original_physical['physical_replay'] is True and original_physical['cases']==405 and
         original_physical['frames']==baseline['frames'] and
         original_physical['assertions']==baseline['checks'] and
         original_physical['reference_sha256']==baseline['reference_sha256'],
         'admitted complete original #61 baseline physical replay required')
    baseline_build=read(a.baseline_build/'build.json',baseline['actual_build_sha256'])
    need(sha(read(a.baseline_build/'heptatorch'))==BASE and
         json.loads(baseline_build)['native_sha256']==BASE, 'baseline binary/build identity')
    reference=Path(baseline['reference_directory'])
    original_reference=reference/'report.json'
    ref_raw=read(original_reference,baseline['reference_sha256'])
    ref=json.loads(ref_raw)
    need(a.torch_python.resolve()==Path(ref['actual_process']['python']['path']),
         'same exact original Torch interpreter required')
    targets={f'case-{r["index"]:03}' for r in ref['rows'] if r['spec']['name'] in TARGETS}
    need(len(targets)==2 and {r['spec']['name'] for r in ref['rows'] if r['spec'].get('controls')}==TARGETS,
         'both and only original unary 7 control recipes')
    am=load(repo/ADMIT,paths[ADMIT])
    def physical():
        return am.admit('dual',repo/'tools/current_runtime/source/manifest.json',build,{'dual_scoped':scoped})
    identity=physical()
    need(identity['native_image_sha256']==DUAL,'exact joint image')
    if a.mode=='successor':
        old=read(a.retained/'report.json')
        retained=json.loads(old)
        precheck(retained,baseline,targets,identity,successor=False)
        side=out/'torch-abs'
        side.mkdir();(side/'blobs').mkdir()
        obs=subprocess.run([str(a.torch_python),'-B','-c',ORACLE,str(repo),str(side),str(original_reference)],
                           cwd=repo,env={'PATH':'/bin:/usr/bin','ATEN_CPU_CAPABILITY':'avx2',
                           'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,check=True,timeout=120)
        need(not obs.stderr,'supplemental Torch emitted stderr')
        torch=json.loads(obs.stdout)
        need(set(torch['rows'])==TARGETS and torch['torch_version']=='2.15.0+cpu' and
             torch['provenance']['status']=='PASS' and len(torch['provenance']['checks'])==11 and
             all(v is True for v in torch['provenance']['checks'].values()),'complete supplemental Torch oracle')
        (side/'stdout').write_bytes(obs.stdout)
        (side/'stderr').write_bytes(obs.stderr)
    visits=[]
    bindings=[]
    raw=paths[RUNNER]
    if a.mode=='successor':
        need(raw.count(OLD)==1 and NEW not in raw,'exact reversible successor')
        raw=raw.replace(OLD,NEW,1)
        need(raw.replace(NEW,OLD,1)==paths[RUNNER],'successor inverse')
    m=load(repo/RUNNER,raw)
    from half_logical.native import split_frames
    def packets(archive,image):
        wire=m.load_wire(repo,m.Inputs().bind)
        found={}
        for target in sorted(targets):
            path=archive/target/'raw'
            trans=json.loads(read(path/'transport.json'))
            requests=split_frames(read(path/'requests.frames.bin'),b'HTOWNR01')
            responses=split_frames(read(path/'responses.frames.bin'),b'HTOWNS01')
            need(len(requests)==len(responses)==trans['frames'] and trans['executable_sha256']==image and
                 len(trans['rows'])==trans['frames'],'exact original physical control transport')
            rows=[]
            for index,(q,r) in enumerate(zip(requests,responses,strict=True)):
                if len(q)==96 and q[:8]==b'HTORCH03' and struct.unpack_from('<Q',q,8)==(1,) and struct.unpack_from('<2Q',q,32)==(3,7):
                    status,records=wire.parse(r)
                    rows.append(dict(frame=index+1,status=status,opcodes=[x['opcode'] for x in records],
                                     request_sha256=sha(q),response_sha256=sha(r)))
            need(len(rows)==1,'expected one unary 7 control in '+target)
            found[target]=rows[0]
        return found
    base_packets=packets(base,BASE)
    need(all(row['status']==7 and row['opcodes']==[] for row in base_packets.values()),
         'original baseline wire did not refuse unary 7')
    if a.mode=='successor':
        retained_packets=packets(a.retained,DUAL)
        need(all(row['status']==0 and row['opcodes']==[3] and
                 row['request_sha256']==base_packets[k]['request_sha256'] for k,row in retained_packets.items()),
             'retained candidate failed predicate not bound to exact native request/success')
    original=m.module
    def module(path,name,bind):
        x=original(path,name,bind)
        if path!=repo/'tools/current_runtime/qualification.py':
            return x
        old_admit=x.admit_build
        def admit(chosen,binder,build_sha):
            need(chosen==build and build_sha==identity['build_report_sha256'],'exact joint build identity')
            try:
                old_admit(chosen,binder,build_sha)
            except ValueError as e:
                need(str(e)=='build used another source manifest','original admission refusal changed')
                bindings.append(name)
            else:
                raise ValueError('original admission accepted the joint executable')
            need(physical()==identity,'physical build identity changed')
            binder(chosen/'heptatorch',DUAL)
            binder(chosen/'runtime.hepta',identity['source_sha256'])
            return json.loads(binder(chosen/'build.json',build_sha))
        x.admit_build=admit
        return x
    m.module=module
    if a.mode=='successor':
        def materialize(value):
            result=dict(value)
            for key in ('logical','storage'):
                item=result[key]
                need(item['path']=='blobs/'+item['sha256']+'.bin','Torch blob path')
                result[key]=read(side/item['path'],item['sha256'])
                need(len(result[key])==item['bytes'],'Torch blob extent')
            return result
        def check_abs(n,row,source):
            recipe=row['spec']['name']
            witness=torch['rows'][recipe]
            n.check('abs uses the unchanged original reference input',row['input'],witness['input'])
            request=n.wire.packet([(3,7,source)])
            status,records,before,after=n.issue(request)
            n.check('actual Torch abs acceptance',status,0)
            n.check('exact abs response',[r['opcode'] for r in records],[3])
            result=records[0]['value']
            input_node=m.Memory(before['bytes']).node(source)
            n.check('all 84 input descriptor words unchanged',m.Memory(after['bytes']).node(source),input_node)
            n.check('complete original input unchanged',n.tensor(source)[0],materialize(witness['input']))
            actual,node=n.tensor(result)
            n.check('all output bytes and complete metadata match pinned Torch abs',actual,materialize(witness['result']))
            n.check('fresh Tensor Storage Version',[result!=source,node[27]!=input_node[27],node[28]!=input_node[28]],[True]*3)
            no_edges=(32,40,48,56,72,80,88,176,184,248,280,288,296,312,360,384,392,400,408,584,592,600,608,616,624,640,648,656,664)
            n.check('no AD, Saved, input or view retention edges',[node[o//8] for o in no_edges],[0]*len(no_edges))
            visits.append(dict(recipe=recipe, request_sha256=sha(request),status=status,
                               result_sha256=sha(actual['logical']),physical_values=actual['numel']))
            n.one(41,result)
        m.check_abs=check_abs
    argv=sys.argv[:]
    try:
        sys.argv=[str(repo/RUNNER),'--build-directory',str(build),'--expected-build-sha256',
            identity['build_report_sha256'],'--torch-python',str(a.torch_python),'--output',str(out/'observer'),
            '--reference-directory',str(reference),'--reference-execution',baseline['reference_process_directory'],
            '--typed-relu-reference-directory',baseline['typed_relu_reference_directory'],
            '--typed-relu-reference-execution',baseline['typed_relu_reference_process_directory']]
        code=m.main()
    finally:
        sys.argv=argv
    after=read(out/'observer/report.json')
    report=json.loads(after)
    succeeded=a.mode=='successor'
    precheck(report,baseline,targets,identity,succeeded)
    candidate_packets=packets(out/'observer',DUAL)
    need(all(row['status']==0 and row['opcodes']==[3] and
             row['request_sha256']==base_packets[k]['request_sha256'] for k,row in candidate_packets.items()),
         'only the explicit same unary 7 packets acquire native abs results')
    evidence=dict(schema='heptatorch.dual-wide-neg.replay.v1',canonical_gate_modified=False,full_candidate_suite=False,
                  baseline_wire_probes=base_packets,candidate_wire_probes=candidate_packets,
                  status='PASS_WIDE_NEG_ABS_SUCCESSORS' if succeeded else 'EXPECTED_LEGACY_UNARY7_FAILURE',
                  mode=a.mode,observer_exit_code=code,baseline_report_sha256=sha(origin),
                  baseline_physical_sha256=sha(read(base/'physical-qualification.json')),
                  candidate_report_sha256=sha(after),reference_sha256=sha(ref_raw),
                  typed_relu_reference_sha256=baseline['typed_relu_reference_sha256'],
                  observer_sha256=sha(raw),candidate_image_sha256=DUAL,baseline_image_sha256=BASE,
                  candidate_build_report_sha256=identity['build_report_sha256'],unchanged_rows=403,
                  frames=report['frames'],checks=report['checks'],admissions=bindings,
                  observations={x['name']:dict(checks=x['checks'],status=x['status'],finish=x.get('finish'),
                    error=x.get('error',{}).get('message')) for x in report['results'] if x['name'] in targets})
    need(code==(0 if succeeded else 1) and bindings==(['wide_neg_build','wide_neg_readback_build'] if succeeded else ['wide_neg_build']),
         'actual process status/legacy-admission path')
    if succeeded:
        proof=read(out/'observer/physical-qualification.json')
        p=json.loads(proof)
        need(p['physical_replay'] is True and p['cases']==405 and p['protocol_controls']==46 and
             p['actual_allocation_refusals']==2 and p['identical_request_recoveries']==2 and
             p['frames']==report['frames'] and p['assertions']==report['checks'], 'complete independent physical readback')
        need(len(visits)==4 and visits[:2]==visits[2:] and {v['recipe'] for v in visits}==TARGETS and
             {v['request_sha256'] for v in visits}=={row['request_sha256'] for row in candidate_packets.values()} and
             candidate_packets==retained_packets,
             'exact two successors executed and physically read back from matching wire probes')
        evidence.update(successors=visits[:2],physical_report_sha256=sha(proof),retained_failure_sha256=sha(old),
                        torch_abs_reference_sha256=sha(obs.stdout))
    need(read(base/'report.json')==origin and
         all(read(repo/p,h)==paths[p] for p,h in PINS.items()) and physical()==identity,'fixed inputs changed')
    return evidence


def precheck(report,baseline,targets,identity,successor):
    need(report['native_sha256']==DUAL and report['source_sha256']==identity['source_sha256'] and
         report['actual_build_sha256']==identity['build_report_sha256'] and report['cases']==len(report['results'])==405 and
         report['all_inputs_unchanged'] is True and report['native_executed'] is True and
         report['reference_sha256']==baseline['reference_sha256'] and
         report['typed_relu_reference_sha256']==baseline['typed_relu_reference_sha256'],
         'exact original roster, current joint image and original references required')
    need(report['status']==('PASS' if successor else 'FAIL') and
         report['failures']==([] if successor else sorted(targets)), 'unexpected retained/successor result')
    for actual,old in zip(report['results'],baseline['results'],strict=True):
        name=actual['name']
        need(name==old['name'],'original order differs')
        if name not in targets:
            need(actual==old,'original unaffected row changed: '+name)
        elif not successor:
            need(actual['status']=='FAIL' and actual['error']['message']==REFUSAL,'unexpected legacy row failure')
        else:
            need(actual['status']=='PASS' and actual['result']==old['result'],'original changed-row predicates differ')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('repository','baseline-build','baseline-run','dual-build','dual-scoped','torch-python','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--retained',type=Path)
    p.add_argument('--mode',choices=('retained','successor'),required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,'run with -I -B')
    for k,v in vars(a).items():
        if isinstance(v,Path):
            need(k=='torch_python' or not v.is_symlink(),'symlinked artifact argument')
            setattr(a,k,v.absolute() if k=='torch_python' else v.resolve())
    a.output.mkdir()
    try:
        result=main(a)
    except Exception as e:
        result=dict(status='BLOCKED_OR_FAILED',error=str(e),type=type(e).__name__)
    dump(a.output/'receipt.json',result)
    print(result['status'])
    sys.exit({'PASS_WIDE_NEG_ABS_SUCCESSORS':0,'EXPECTED_LEGACY_UNARY7_FAILURE':2}.get(result['status'],1))
