#!/usr/bin/env python3
"""Run original live32 with complete PR39 transport/owner readback on current HIA."""
import argparse
import ast
import json
from pathlib import Path
import sys
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
# An isolated candidate may explicitly use the same frozen repository inputs.
if '--repository' in sys.argv:
    ROOT = Path(sys.argv[sys.argv.index('--repository')+1]).resolve()
sys.path.insert(0,str(ROOT/'tests'))
sys.path.insert(0,str(ROOT/'tests/current_runtime_original35'))
from binding import Inputs, bind_build, dump, require, sha
from current_runtime_process import execute
from current_runtime_admission import complete_runner, current35_live_runner
from current_runtime_admission import SOURCE, IMAGE, FUNCTIONS, UNITS

HERE = Path(__file__).resolve().parent
READBACK = HERE/'current_runtime_live32_readback.py'
READBACK_SHA = '5167d530c3cdc69b2a49401801dd85f959fc4ebebf32c4cb156bd345f7064a0b'
RUNNER_SHA = 'd736aa026f10ca5db6cbb887eab328412b1befb3c5788fce5269b128b3dae7f0'
MANIFEST_SHA = '8064f242e8a4923fe6c86adce43dc1d6e30668abfc46c28fafe0ec83ee52557b'
ADMISSION_OLD_SOURCE = 'de4fdf964a74be11acfeddcf5fe16656c6cfa671fc7917585a72364918387a92'
ADMISSION_OLD_IMAGE = '75e29e13c0a7f14395fd21acb82686279dc962615ae6bcde3176b72083cac971'


def definitions(raw):
    lines=raw.splitlines(keepends=True)
    return {n.name:b''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(raw).body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}


def transform(raw, edits, allowed):
    before=raw;record=[]
    for label,old,new in edits:
        old,new=old.encode(),new.encode()
        require(old!=new and raw.count(old)==1,'one exact live32 profile edit: '+label)
        raw=raw.replace(old,new,1)
        record.append(dict(label=label,before_hex=old.hex(),after_hex=new.hex(),count=1))
    inverse=raw
    for edit in reversed(record):
        old,new=bytes.fromhex(edit['before_hex']),bytes.fromhex(edit['after_hex'])
        require(inverse.count(new)==1,'unique complete live32 inverse: '+edit['label'])
        inverse=inverse.replace(new,old,1)
    require(inverse==before,'complete live32 original source inverse differs')
    old_defs,new_defs=definitions(before),definitions(raw)
    changed=[name for name in old_defs if old_defs[name]!=new_defs.get(name)]
    require(old_defs.keys()==new_defs.keys() and set(changed)==set(allowed),
            'unexpected original live32 function/class change: '+repr(changed))
    return raw,dict(original_sha256=sha(before),executed_sha256=sha(raw),
        whole_file_inverse_exact=True,edits=record,changed_definitions=changed,
        unchanged_definitions={name:sha(body) for name,body in old_defs.items() if name not in changed})


def inventory_edits():
    require(FUNCTIONS==800 and UNITS==15,'this reviewed owner requires canonical current800/15')
    return [("built['complete_core_bodies_matched'] == 701",
             "built['complete_core_bodies_matched'] == "+str(FUNCTIONS)),
            ('len(functions) == len(bodies) == 701',
             'len(functions) == len(bodies) == '+str(FUNCTIONS)),
            ("'functions': 701,", "'functions': "+str(FUNCTIONS)+','),
            ("built['units_planned'] == 14", "built['units_planned'] == 15"),
            ("len(units) == len(manifest['units']) == 14", "len(units) == len(manifest['units']) == 15"),
            ('len(processes) == 15', 'len(processes) == 16'),
            ("len({u['name'] for u in units}) == 14", "len({u['name'] for u in units}) == 15"),
            ("struct.pack('<3Q', 14, 4, 0)", "struct.pack('<3Q', 15, 4, 0)"),
            ("'units': 14,", "'units': 15,"),
            ("'compiler_processes': 14", "'compiler_processes': 15"),
            ('aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16',
             '761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e'),
            ('b9d039c365a8a3988e3add120e8cffa46a4e1297878f4cced30103953b030bfb',
             '9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'),
            ('a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3',
             '7c32981c916bc277e16221cce1c3dd2381ce0018642e21b20836d861cf585865'),
            ('9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884',
             '867a27b3a86634ea6a873909eb8bc16d5d8027c98c7cb9340376fba51312ec86'),
            ("b'HIASTKS1'", "b'HIACGS01'")]

def runner_source(package, inputs):
    raw=inputs.bind(package/'run.py',RUNNER_SHA)
    old="               ('"+ADMISSION_OLD_IMAGE+"', IMAGE)]"
    new=old[:-1]+',\n'+',\n'.join('               '+repr(pair) for pair in inventory_edits())+']'
    edits=[('current source assignment',"SOURCE = '82360ce87dae6f9a004ef50be4f991b6f83905a83f4b17ca8d6634827b7bd313'",
            'SOURCE = '+repr(SOURCE)),
        ('current executable assignment',"IMAGE = '80445a8c433f20314d3bc7b00ebb68f6a70fa200ee2e6aba193cff21aabfbc17'",
            'IMAGE = '+repr(IMAGE)),
        ('unchanged immutable package location','HERE = Path(__file__).resolve().parent',
            'HERE = Path('+repr(str(package))+')'),
        ('complete function inventory identities',old,new),
        ('explicit inventory admission change disclosure',"'build_predicates_changed': False}",
            "'build_predicates_changed': True, 'function_inventory_literals_changed': 3,\n"
            "                    'cold_unit_process_inventory_literals_changed': 7,\n"
            "                    'tool_support_and_ABI_literals_changed': 5,\n"
            "                    'total_identity_and_inventory_edits': 17,\n"
            "                    'predicate_operators_and_checks_retained': True}"),
        ('current owner function inventory',"('functions == 699', 'functions == 701', 1)",
            "('functions == 699', 'functions == "+str(FUNCTIONS)+"', 1)"),
        ('current unbound owner profile',"'live32-current701-UNBOUND'",
            repr('live32-current'+str(FUNCTIONS)+'-UNBOUND')),
        ('bind current complete owner','module.bind_profile(SOURCE, IMAGE, 701)',
            'module.bind_profile(SOURCE, IMAGE, '+str(FUNCTIONS)+')')]
    return transform(raw,edits,('admission_module','owner_module'))


def readback_source(inputs):
    raw=inputs.bind(READBACK,READBACK_SHA)
    old="            ['"+ADMISSION_OLD_IMAGE+"', IMAGE, 1]])"
    new=old[:-2]+',\n'+',\n'.join('            '+repr([a,b,1]) for a,b in inventory_edits())+'])'
    edits=[('current owner inventory inverse',"['functions == 699', 'functions == 701', 1]",
            "['functions == 699', 'functions == "+str(FUNCTIONS)+"', 1]"),
        ('current unbound owner inverse',"'live32-current701-UNBOUND'",
            repr('live32-current'+str(FUNCTIONS)+'-UNBOUND')),
        ('execute only the proven current runner',
            "reader = load_source(package / 'run.py', runner_raw, 'current_runtime_original_live32_reader')",
            'reader = load_current_runner(directory, package, inputs)'),
        ('explicit function inventory proof',
            "and admission_proof['build_predicates_changed'] is False, 'only the exact cold-build identities changed')",
            "and admission_proof['build_predicates_changed'] is True\n"
            "            and admission_proof['function_inventory_literals_changed'] == 3\n"
            "            and admission_proof['cold_unit_process_inventory_literals_changed'] == 7\n"
            "            and admission_proof['tool_support_and_ABI_literals_changed'] == 5\n"
            "            and admission_proof['total_identity_and_inventory_edits'] == 17\n"
            "            and admission_proof['predicate_operators_and_checks_retained'] is True,\n"
            "            'only source/image, function/unit/process inventories and five exact tool/support/ABI identities changed')"),
        ('complete admission inventory inverse',old,new),
        ('actual cold-build coverage disclosure',
            'repeat the original complete fourteen-unit/fifteen-process cold-build admission',
            'repeat complete fifteen-unit/sixteen-process admission with every original byte and process predicate'),
        ('read back the current complete owner','owner.bind_profile(SOURCE, IMAGE, 701)',
            'owner.bind_profile(SOURCE, IMAGE, '+str(FUNCTIONS)+')')]
    return transform(raw,edits,('qualify_live32',))


def load_source(path, raw, name, extra=None):
    value=types.ModuleType(name);value.__file__=str(path)
    if extra:value.__dict__.update(extra)
    exec(compile(raw,str(path),'exec'),value.__dict__)
    return value


def check_package(package, inputs):
    raw=inputs.bind(package/'package-manifest.json',MANIFEST_SHA)
    manifest=json.loads(raw)
    paths={str(path.relative_to(package)) for path in package.rglob('*') if path.is_file()}
    require(paths==set(manifest['files'])|{'package-manifest.json'},'complete immutable PR39 live32 package')
    for name,row in manifest['files'].items():
        path=package/name;content=inputs.bind(path,row['sha256'])
        require(len(content)==row['bytes'] and path.stat().st_mode&511==row['mode'],
                'original PR39 live32 package bytes/mode changed')


def prepare(package, adaptation, inputs):
    check_package(package,inputs)
    adaptation.mkdir(parents=True,exist_ok=False)
    run,run_proof=runner_source(package,inputs)
    reader,reader_proof=readback_source(inputs)
    (adaptation/'executed-run.py').write_bytes(run)
    (adaptation/'executed-readback.py').write_bytes(reader)
    proof=dict(schema='heptatorch.current-live32.pr39-identity-inverses.v1',
        source_sha256=SOURCE,native_sha256=IMAGE,functions=FUNCTIONS,units=UNITS,
        runner=run_proof,readback=reader_proof,original_oracle_modified=False,
        original_transport_and_owner_capture_modified=False,
        original_frame_decoder_and_owner_readback_modified=False,
        old_pass_transferred=False)
    dump(adaptation/'source-inverses.json',proof)
    for path in adaptation.iterdir():inputs.bind(path)
    return proof


def qualify_live32(report, build_sha, directory, inputs):
    directory=Path(directory)
    require(report['schema']=='heptatorch.current-runtime.live32.v1' and report['status']=='PASS'
        and report['actual_build_sha256']==build_sha and report['source_sha256']==SOURCE
        and report['native_sha256']==IMAGE and report['functions']==FUNCTIONS and report['units']==UNITS,
        'same complete current live32 qualification')
    require(report['native_executed'] is True and report['Torch_executed'] is True
        and report['all_inputs_unchanged'] is True and report['changed_inputs']==[] and report['failures']==[],
        'actual full live32 execution/custody')
    build=Path(report['build_directory']);package=Path(report['original_source_package'])
    bind_build(build,build_sha,inputs)
    check_package(package,inputs)
    adaptation=directory/'adaptation'
    run,run_proof=runner_source(package,inputs)
    reader_raw,reader_proof=readback_source(inputs)
    require(inputs.bind(adaptation/'executed-run.py')==run
        and inputs.bind(adaptation/'executed-readback.py')==reader_raw,
        'both actual executed live32 sources retain their whole original inverses')
    proof=json.loads(inputs.bind(adaptation/'source-inverses.json'))
    require(proof['runner']==run_proof and proof['readback']==reader_proof,
            'the actual source inverse records differ')

    def load_current_runner(original_directory,original_package,custody):
        require(Path(original_package)==package,'original live32 package location changed')
        custody.bind(package/'run.py',RUNNER_SHA)
        raw=custody.bind(adaptation/'executed-run.py',sha(run))
        return load_source(adaptation/'executed-run.py',raw,'actual_current_live32_runner')

    reader=load_source(adaptation/'executed-readback.py',reader_raw,'actual_current_live32_readback',
                       dict(load_current_runner=load_current_runner))
    original_raw=inputs.bind(directory/'original/report.json',report['original_report']['sha256'])
    require(len(original_raw)==report['original_report']['bytes'],'complete original enhanced live32 report')
    original=json.loads(original_raw)
    counts=reader.qualify_live32(original,build,build_sha,directory/'original',package,inputs)
    facts=current35_live_runner(report['execution'],directory,inputs)
    require(report['live_reference_stderr_admission']==facts,'exact pinned live Torch process/source observation')
    require(report['old_pass_transferred'] is False and report['full_project_status']=='HOLD'
        and report['full_pytorch_rewrite_complete'] is False,'finite original live32 qualification only')
    return dict(**counts, native_exit_zero_count=32, native_EOF_count=64, actual_Torch_reference=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=ROOT)
    parser.add_argument('--source-package',type=Path)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--expected-build-sha256',required=True)
    parser.add_argument('--torch-python',type=Path,default=Path(sys.executable))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    require(not sys.flags.optimize and args.repository.resolve()==ROOT,'exact repository and enabled assertions')
    build,out=args.build_directory.resolve(),args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    inputs,failures=Inputs(),[]
    report=dict(schema='heptatorch.current-runtime.live32.v1',status='RUNNING',
        source_sha256=SOURCE,native_sha256=IMAGE,functions=FUNCTIONS,units=UNITS,
        actual_build_sha256=args.expected_build_sha256,build_directory=str(build),
        expected_cases=32,cases=0,native_executed=False,Torch_executed=False,
        old_pass_transferred=False,original_comparison_source_unchanged=True,
        full_project_status='HOLD',full_pytorch_rewrite_complete=False,failures=failures)
    try:
        inputs.bind(Path(__file__).resolve())
        inputs.bind(READBACK,READBACK_SHA)
        for relative in ('tests/current_runtime_original35/binding.py','tests/current_runtime_process.py',
                         'tests/current_runtime_admission.py','tests/current_runtime_torch.py'):
            inputs.bind(ROOT/relative)
        bind_build(build,args.expected_build_sha256,inputs)
        if args.source_package:
            package=args.source_package.resolve()
        else:
            import current_runtime_fixtures
            inputs.bind(ROOT/'tests/current_runtime_fixtures.py')
            package=current_runtime_fixtures.restore_live32(ROOT,out/'source',inputs)
        report['original_source_package']=str(package)
        prepare(package,out/'adaptation',inputs)
        target=[str(out/'adaptation/executed-run.py'),'--repository',str(ROOT),
            '--binary',str(build/'heptatorch'),'--source',str(build/'runtime.hepta'),
            '--build-directory',str(build),'--expected-build-sha256',args.expected_build_sha256,
            '--expected-source-sha256',SOURCE,'--expected-native-sha256',IMAGE,
            '--output',str(out/'original')]
        if args.prepare_only:
            command=[sys.executable,'-I','-S','-B',*target,'--prepare-only']
        else:
            inputs.bind(ROOT/'build/pytorch-install.json')
            inputs.bind(args.torch_python.resolve())
            command=[str(args.torch_python.absolute()),'-B',str(ROOT/'tests/current_runtime_torch.py'),
                     '--source-observation',str(out/'torch-source.json'),'--',*target]
        execution=out/'native_and_live_reference-execution';execution.mkdir()
        try:
            receipt=execute(command,execution,180 if args.prepare_only else 1800,cwd=ROOT,
                            mirror_stderr=args.prepare_only)
            report['execution']=receipt
        finally:
            original_path=out/'original/report.json'
            if original_path.is_file():
                raw=inputs.bind(original_path);original=json.loads(raw)
                report['native_executed']=original.get('native_executed')
                report['Torch_executed']=original.get('Torch_executed')
                report['original_report']=dict(path='original/report.json',bytes=len(raw),sha256=sha(raw))
            elif not args.prepare_only:
                report['native_executed']=report['Torch_executed']=None
        if args.prepare_only:
            complete_runner(receipt)
            require(original['status']=='PREPARED_NOT_RUN' and original['native_executed'] is False
                    and original['Torch_executed'] is False,'only original source/build preparation')
            report['status']='PREPARED_NOT_RUN'
        else:
            report['live_reference_stderr_admission']=current35_live_runner(receipt,out,inputs)
            report.update(status='PASS',cases=32,frames=original['frames'],
                exact_binary32_words=original['exact_binary32_words'],
                serialized_state_bytes=original['serialized_state_bytes'],oracle_provenance_checks=11)
        report['changed_inputs']=inputs.recheck();report['all_inputs_unchanged']=not report['changed_inputs']
        if not args.prepare_only:
            report['readback']=qualify_live32(report,args.expected_build_sha256,out,inputs)
    except BaseException as error:
        failures.append(repr(error))
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc()))
    finally:
        report['changed_inputs']=inputs.recheck();report['all_inputs_unchanged']=not report['changed_inputs']
        report['inputs']=inputs.records
        if failures or report['changed_inputs']:report['status']='FAIL'
        dump(out/'report.json',report)
    print(json.dumps({key:report.get(key) for key in ('status','cases','frames','native_executed','failures')}),flush=True)
    return 0 if report['status'] in ('PASS','PREPARED_NOT_RUN') else 1


if __name__=='__main__':
    raise SystemExit(main())
