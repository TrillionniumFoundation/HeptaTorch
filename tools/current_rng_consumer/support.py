"""Pure custody reader and raw-evidence helpers for the current RNG consumer."""
from pathlib import Path
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import struct
import sys
import types

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import (RNG_OBJECT, RNG_SOURCE, OLD_SOURCE, COMPILER, LINKER, INSPECTOR,
                   adapter, current_admission, input_roles, require)

IMAGE = '084c855df6fa5bd387d051228744a9ded60a598334f1c237ae61d4759ad782a6'
ADAPTER_OBJECT = '16491b2d40d959b4cb0cc62f8f4fa3393b0a6ce1c4fc9599932ecf14773e3bf8'
ENTRY = 28752
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def admit(repository, directory, current_build):
    """Validate current cold build, reused object, exact adapter, link and EOFs.

    Current source/image/function counts come from the canonical manifest and
    its existing full reader. No historical 775/778 identity or local receipt
    timestamp/path is accepted as a substitute for that reader.
    """
    require(not sys.flags.optimize, 'original source, object and test assertions must remain enabled')
    repo, root, current = map(lambda p:Path(p).resolve(), (repository,directory,current_build))
    inputs = {}
    def bind(path, expected=None):
        path=Path(path).resolve(strict=True);raw=path.read_bytes()
        require(expected is None or sha(raw)==expected,'input identity: '+str(path))
        require(path not in inputs or inputs[path]==raw,'input changed: '+str(path))
        inputs[path]=raw
        return raw
    implementation=Path(__file__).resolve().with_name('build.py')
    bind(implementation)
    bind(implementation.with_name('process_evidence.py'))
    built=current_admission(repo,current,bind)
    build_raw=bind(current/'build.json')
    old=bind(repo/'tools/rng-cpu/rng.hepta',OLD_SOURCE)
    rng=bind(current/'rng03.hepta',RNG_SOURCE)
    require(rng==bind(repo/'tools/current_runtime/source/units/rng03.hepta',RNG_SOURCE),'canonical current RNG source')
    obj=bind(current/'rng03.o',RNG_OBJECT)
    spec_raw=bind(repo/'tools/stack_budget/profile.json')
    spec=json.loads(spec_raw)
    require(spec['compiler']['executable_sha256']==COMPILER
            and spec['linkers']['runtime']['executable_sha256']==LINKER,'fixed compiler/flat-linker profile')
    bind(current/'toolchain/stack-module-compiler',COMPILER)
    flat=root/'linker';bind(flat,LINKER)
    inspector_raw=bind(repo/'tools/stack_budget/inspect_stack.py',INSPECTOR)
    expected_bindings=input_roles(inputs,repo,current,implementation,flat)
    checked={}
    def output(name, expected=None):
        path=root/name;raw=path.read_bytes()
        require(not path.is_symlink() and (expected is None or sha(raw)==expected),'candidate output identity: '+name)
        checked[path]=raw
        return raw
    raw=output('build.json');report=json.loads(raw)
    require(report['schema']=='heptatorch.current-object-rng-packet-build.v1'
            and report['status']=='PASS_NATIVE_BUILD_ONLY','complete successful packet build')
    require(report['current_build_sha256']==sha(build_raw)
            and report['current_source_sha256']==built['source_sha256']
            and report['current_native_sha256']==built['native_sha256']
            and report['current_functions']==built['functions']
            and report['current_units']==built['units_planned'],'same complete current cold build')
    require(report['input_bindings']==expected_bindings,'complete portable build input closure')
    source,proof=adapter(old,rng)
    require(report['source_transition']==proof and output('packet_adapter.hepta')==source,'exact five original packet bodies and complete current kernel closure')
    require(output('rng03.hepta')==rng and output('rng03.o')==obj,'reuse the exact current actual RNG object')
    adapter_obj=output('packet_adapter.o',ADAPTER_OBJECT)
    require(report['current_rng_object_sha256']==RNG_OBJECT and report['adapter_object_sha256']==ADAPTER_OBJECT,'fixed native object identities')
    output('compiler',COMPILER);output('linker',LINKER)
    require(report['compiler_sha256']==COMPILER and report['flat_linker_sha256']==LINKER,'executed native tool identities')
    inspector=types.ModuleType('rng_consumer_stack_inspector')
    exec(compile(inspector_raw,'<fixed-inspector>','exec'),inspector.__dict__)
    views=[inspector.inspect(rng,obj),inspector.inspect(source,adapter_obj)]
    stack=inspector.longest(views)
    require(report['stack']==stack and stack['maximum_path_bytes']==19952
            and stack['maximum_path_bytes']<=spec['linkers']['runtime']['stack_budget'],'complete actual source/object stack graph')
    request=b'HIASTKR1'+struct.pack('<3Q',2,5,0)+b'entry'
    request+=b''.join(struct.pack('<Q',len(x))+x for x in [obj,adapter_obj])
    image=output('rng-native',IMAGE)
    require(report['native_sha256']==IMAGE and report['native_bytes']==len(image)==77824
            and report['native_interface_abi']==3 and report['native_entry_offset']==ENTRY,'reviewed adapter ELF/entry identity')
    require(image[477]==0xe8 and 482+struct.unpack_from('<i',image,478)[0]==ENTRY
            and image[ENTRY:ENTRY+15].hex()==report['native_entry_prologue_hex']=='554889e54881ec00100000488b0424','reviewed native ABI3 entry target')
    require([p['name'] for p in report['processes']]==['compile-0','compile-1','link-0','link-1'],'complete native tool executions')
    from process_evidence import normalized_absolute
    producer = report['actual_process']
    origins = report['producer_paths']
    require(set(origins) == {'repository','current_build','output'}
            and all(normalized_absolute(p) for p in origins.values())
            and producer['name'] == 'build' and type(producer['pid']) is int and producer['pid'] > 0,
            'actual native-build producer and original path labels')
    require(producer['argv'][1:4] == ['-I','-S','-B']
            and producer['argv'][5:] == ['--repository',origins['repository'],
                '--current-build',origins['current_build'],'--flat-linker',producer['argv'][10],
                '--output',origins['output']] and producer['cwd'] == origins['repository'],
            'exact supervised native build command')
    pids = set()
    for proc in report['processes']:
        name=proc['name'];compiling=name.startswith('compile-')
        wanted_in=source if compiling else request
        wanted_out=adapter_obj if compiling else image
        executable_name = 'compiler' if compiling else 'linker'
        executable_raw = output(executable_name, COMPILER if compiling else LINKER)
        executable = {'path':origins['output']+'/'+executable_name, 'bytes':len(executable_raw),
                      'sha256':sha(executable_raw), 'mode':0o500}
        intent = {'name':name, 'argv':[executable['path']], 'cwd':origins['output'],
                  'environment':{'PATH':''}, 'timeout_seconds':120, 'executable':executable,
                  'supervisor_pid':producer['pid'], 'started':False,
                  'stdout_EOF':False, 'stderr_EOF':False}
        require(json.loads(output(name+'.intent.json')) == intent
                and all(proc.get(k) == v for k,v in intent.items() if k not in ('started','stdout_EOF','stderr_EOF'))
                and proc['started'] is True and type(proc['pid']) is int and proc['pid'] > 0
                and proc['pid'] not in pids and proc['timed_out'] is False,
                'exact native tool original intent, process and environment')
        pids.add(proc['pid'])
        require(output(name+'.request.bin')==wanted_in and output(name+'.stdout.bin')==wanted_out
                and output(name+'.stderr.bin')==b'','complete actual tool raw input/output')
        require(proc['executable_sha256']==(COMPILER if compiling else LINKER)
                and proc['request_sha256']==sha(wanted_in) and proc['stdout_sha256']==sha(wanted_out)
                and proc['request_bytes']==len(wanted_in) and proc['stdout_bytes']==len(wanted_out)
                and proc['stderr_bytes']==0
                and proc['stderr_sha256']==sha(b'') and proc['exit_code']==0
                and proc['stdout_EOF'] is True and proc['stderr_EOF'] is True,'native tool result and complete dual EOF')
    for flag in ('all_inputs_unchanged','kernel_object_is_current_actual_object'):
        require(report[flag] is True,'positive build provenance: '+flag)
    for flag in ('full_current_image_executed','legacy143_restored','old_native_rng_image_executed',
                 'host_numerical_codegen','qualified','whole_pytorch_rewrite_complete'):
        require(report[flag] is False,'build scope: '+flag)
    for path,before in {**inputs,**checked}.items():
        require(path.read_bytes()==before,'bound input/output changed during pure admission')
    return {'candidate_build_sha256':sha(raw),'current_build_sha256':sha(build_raw),
            'current_source_sha256':built['source_sha256'],'current_image_sha256':built['native_sha256'],
            'current_functions':built['functions'],'current_units':built['units_planned'],
            'current_rng_object_sha256':RNG_OBJECT,'native_sha256':IMAGE,
            'adapter_source_sha256':sha(source),'same_current_rng_object':True,
            'original_packet_and_error_bodies_byte_identical':True,
            'native_interface_abi':3,'native_entry_offset':ENTRY,
            'legacy143_source_restored':False,'old_rng_runtime_executed':False,
            'host_numerical_codegen':False}


def capture(directory, number, request, response, stderr, returncode):
    directory = Path(directory)/'raw'
    directory.mkdir(exist_ok=True)
    stem = '%04d' % number
    for name, raw in [('request', request), ('response', response), ('stderr', stderr)]:
        (directory/(stem+'.'+name+'.bin')).write_bytes(raw)
    row = {'index': number, 'returncode': returncode,
           'request_sha256': sha(request), 'request_bytes': len(request),
           'response_sha256': sha(response), 'response_bytes': len(response),
           'stderr_sha256': sha(stderr), 'stderr_bytes': len(stderr)}
    (directory/(stem+'.json')).write_text(json.dumps(row, indent=2)+'\n')
    return row


def capture_flat(image, request, input_after, return_raw, out_cap, work_cap, output, workspace, observed):
    directory=Path(os.environ['HEPTATORCH_RNG_FLAT_EVIDENCE'])
    directory.mkdir(parents=True,exist_ok=True)
    index=len(list(directory.glob('*.json')))
    stem='%04d'%index
    blobs={}
    for name,raw in [('request',request),('input_after',input_after),('return',return_raw),('output',output),('workspace',workspace)]:
        path=directory/(stem+'.'+name+'.bin');path.write_bytes(raw)
        blobs[name]={'path':path.name,'bytes':len(raw),'sha256':sha(raw)}
    row={'index':index,'native_sha256':sha(image),'output_capacity':out_cap,
         'workspace_capacity':work_cap,'blobs':blobs,'observed':observed}
    (directory/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n')


def observed_torch(repo, output):
    """Explicit exploratory path; never substitutes for original 11 checks."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        import torch
    output = Path(output)
    (output/'torch-import.stdout').write_text(stdout.getvalue())
    (output/'torch-import.stderr').write_text(stderr.getvalue())
    lock_raw = (Path(repo)/'compatibility/pytorch-oracle-lock.json').read_bytes()
    require(sha(lock_raw) == 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944', 'original Torch lock unchanged')
    oracle = json.loads(lock_raw)['oracle']
    dist = importlib.metadata.distribution('torch')
    checks = {'python': sys.implementation.name == oracle['python']['implementation'] and list(sys.version_info[:2]) == oracle['python']['version'],
              'platform': {'sys_platform':sys.platform, 'machine':platform.machine()} == oracle['platform'],
              'version': str(torch.__version__) == oracle['version'] == dist.version,
              'git': torch.version.git_version == oracle['git_commit'],
              'cpu_backends': {k:getattr(torch.version,k,'MISSING') for k in oracle['backend_versions']} == oracle['backend_versions']}
    require(all(checks.values()), 'observed installed Torch identity mismatch')
    def identity(path):
        path = Path(path).resolve(strict=True); raw = path.read_bytes()
        return {'path':str(path), 'bytes':len(raw), 'sha256':sha(raw)}
    receipt = {'status':'EXPLORATORY_OBSERVED_TORCH_ONLY', 'observed_checks':checks,
               'actual_process':{'pid':os.getpid(), 'argv':sys.orig_argv, 'python':identity(sys.executable)},
               'version':str(torch.__version__), 'git_version':torch.version.git_version,
               'package':identity(torch.__file__), 'extension':identity(torch._C.__file__),
               'cpu_capability':torch.backends.cpu.get_cpu_capability(), 'reference_device':'cpu',
               'direct_url':dist.read_text('direct_url.json'),
               'import_stdout_sha256':sha(stdout.getvalue().encode()), 'import_stderr_sha256':sha(stderr.getvalue().encode()),
               'full_installation_provenance_verified':False, 'original_eleven_checks_passed':False,
               'original_pip_install_report_available':False, 'production_oracle_qualified':False}
    (output/'observed-torch.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return torch, receipt
