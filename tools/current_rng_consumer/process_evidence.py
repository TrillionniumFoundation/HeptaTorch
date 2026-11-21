"""Record and read exact process identities without executing an evidence file.

Children receive a small explicit environment. The archive keeps both the
supervisor's intent/result and a receipt written by the actual Python child.
Producer paths are labels: the reader reconstructs every command from these
labels and reads bytes through evidence-relative paths, so relocation does not
silently rewrite the original command line.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import platform
import posixpath
import stat
import sys

NAMES = ('build', 'prepare', 'prepare-custody', 'custody',
         'cpython74', 'flat10', 'independent25', 'pytorch76')
FLAGS = ('isolated', 'no_site', 'no_user_site', 'ignore_environment',
         'dont_write_bytecode', 'optimize')
ENVIRONMENT = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
               'LC_ALL': 'C.UTF-8', 'TZ': 'UTC'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), 'regular process input')
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': sha(raw),
            'mode': stat.S_IMODE(path.stat().st_mode)}


def actual_process(script, include_environment=True):
    return {'schema': 'heptatorch.rng-consumer.actual-process.v2',
            'pid': os.getpid(), 'ppid': os.getppid(),
            'argv': list(sys.orig_argv), 'cwd': str(Path.cwd().resolve()),
            'python': identity(sys.executable), 'script': identity(script),
            'flags': {name: getattr(sys.flags, name) for name in FLAGS},
            'python_runtime': {'implementation': sys.implementation.name,
                               'version': list(sys.version_info[:2]),
                               'full_version': sys.version},
            'platform': {'sys_platform': sys.platform, 'machine': platform.machine()},
            'environment': dict(os.environ) if include_environment else None}


def record_actual_process(script):
    """Called in each actual child; pure source regeneration has no receipt env."""
    destination = os.environ.get('HEPTATORCH_RNG_PROCESS_RECEIPT')
    if destination is None:
        return None
    require(os.environ.get('HEPTATORCH_RNG_PROCESS_NAME') in NAMES,
            'known supervised process name')
    row = actual_process(script)
    row['name'] = os.environ['HEPTATORCH_RNG_PROCESS_NAME']
    path = Path(destination)
    require(path.is_absolute() and path.parent.is_dir() and not path.exists()
            and not path.is_symlink(), 'new supervised actual-process receipt')
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
    return row


def plan(context, name):
    """All eight commands, environments and output locations, reconstructed."""
    require(name in NAMES, 'known production process')
    paths = context['paths']
    repo, current, out, implementation, linker = (
        PurePosixPath(paths[key]) for key in
        ('repository', 'current_build', 'output', 'implementation', 'flat_linker'))
    python = context['producer']['python']['path']
    iso = [python, '-I', '-S', '-B']
    ordinary = [python, '-B']
    code = out / 'code/adapted'
    common = ['--repository', str(repo), '--native-build', str(out/'packet-build'),
              '--current-build', str(current)]
    commands = {
        'build': iso + [str(implementation/'build.py'), '--repository', str(repo),
                       '--current-build', str(current), '--flat-linker', str(linker),
                       '--output', str(out/'packet-build')],
        'prepare': iso + [str(implementation/'prepare.py'), '--repository', str(repo),
                         '--build', str(out/'packet-build'), '--current-build', str(current),
                         '--output', str(code)],
        'prepare-custody': iso + [str(implementation/'prepare_custody.py'),
                                 '--repository', str(repo), '--output', str(code/'check_rng_custody.py')],
        'custody': iso + [str(code/'check_rng_custody.py'), str(repo)],
        'cpython74': iso + [str(code/'rng_cpu_oracle.py'), *common,
                           '--reference', 'cpython', '--output', str(out/'cpython74')],
        'flat10': iso + [str(code/'flat_probe.py'), *common,
                        '--native', str(out/'packet-build/rng-native'),
                        '--expect-short', 'return-zero', '--output', str(out/'flat10')],
        'independent25': iso + [str(code/'review_rng.py'), *common,
                               '--native', str(out/'packet-build/rng-native'),
                               '--source', str(repo/'tools/rng-cpu/rng.hepta'),
                               '--output', str(out/'independent25')],
        'pytorch76': ordinary + [str(code/'rng_cpu_oracle.py'), *common,
                                '--reference', 'pytorch', '--output', str(out/'pytorch76')]
                     + (['--allow-unverified-local'] if context['local_observed_oracle'] else []),
    }
    overrides = {'HEPTATORCH_RNG_PROCESS_NAME': name,
                 'HEPTATORCH_RNG_PROCESS_RECEIPT': str(out/'processes'/(name+'.actual.json'))}
    if name in ('cpython74', 'flat10', 'independent25', 'pytorch76'):
        overrides['HEPTATORCH_RNG_FLAT_EVIDENCE'] = str(out/name/'flat-evidence')
    if name == 'pytorch76':
        # Torch initialization may otherwise add a default cache path after the
        # startup receipt. Bind the same explicit path before and after import.
        overrides['TORCHINDUCTOR_CACHE_DIR'] = str(out/name/'torchinductor-cache')
    return {'name': name, 'argv': commands[name], 'cwd': str(repo),
            'environment': dict(ENVIRONMENT, HOME=context['home'], **overrides),
            'environment_overrides': overrides, 'timeout_seconds': 600,
            'actual_process_path': 'processes/'+name+'.actual.json',
            'stdout_path': 'processes/'+name+'.stdout',
            'stderr_path': 'processes/'+name+'.stderr'}


def normalized_absolute(path):
    return isinstance(path, str) and path.startswith('/') and posixpath.normpath(path) == path and '\\' not in path


def _resolve(label, cwd):
    return posixpath.normpath(label if label.startswith('/') else posixpath.join(cwd, label))


def read_context(read, local):
    context = json.loads(read('execution-context.json'))
    require(context['schema'] == 'heptatorch.rng-consumer.execution-context.v2'
            and type(context['local_observed_oracle']) is bool
            and context['local_observed_oracle'] is local, 'immutable execution mode')
    require(set(context['paths']) == {'repository', 'current_build', 'output', 'implementation',
                                     'flat_linker', 'original_package'}
            and all(normalized_absolute(p) for p in context['paths'].values())
            and normalized_absolute(context['home']), 'complete normalized producer path labels')
    producer = context['producer']
    require(producer['schema'] == 'heptatorch.rng-consumer.actual-process.v2'
            and producer['environment'] is None and normalized_absolute(producer['cwd'])
            and type(producer['pid']) is int and producer['pid'] > 0
            and type(producer['ppid']) is int and producer['ppid'] > 0,
            'actual isolated supervisor process identity')
    wanted_flags = dict(isolated=1, no_site=1, no_user_site=1, ignore_environment=1,
                        dont_write_bytecode=1, optimize=0)
    require(producer['flags'] == wanted_flags, 'isolated non-optimized production supervisor')
    argv = producer['argv']
    require(len(argv) >= 13 and argv[1:4] == ['-I', '-S', '-B'], 'exact supervisor Python flags')
    paths = context['paths']
    require(_resolve(argv[4], producer['cwd']) == paths['implementation']+'/run.py',
            'supervisor executed the bound runner')
    options, seen = {}, set()
    args = list(argv[5:])
    while args:
        flag = args.pop(0)
        require(flag not in seen, 'unique producer option')
        seen.add(flag)
        if flag == '--local-observed-oracle':
            options[flag] = True
        else:
            require(flag in ('--repository', '--current-build', '--flat-linker', '--output', '--original-package')
                    and args, 'known complete producer option')
            options[flag] = _resolve(args.pop(0), producer['cwd'])
    require(set(options) - {'--local-observed-oracle', '--original-package'} ==
            {'--repository', '--current-build', '--flat-linker', '--output'}, 'all required producer options')
    for flag, key in (('--repository','repository'), ('--current-build','current_build'),
                      ('--flat-linker','flat_linker'), ('--output','output')):
        require(options[flag] == paths[key], 'exact producer path option: '+flag)
    require(options.get('--original-package', paths['repository']) == paths['original_package']
            and options.get('--local-observed-oracle', False) is local, 'original package and explicit producer mode')
    python_raw = read('inputs/runtime/python.bin', producer['python']['sha256'])
    require(len(python_raw) == producer['python']['bytes'] and python_raw[:4] == b'\x7fELF'
            and normalized_absolute(producer['python']['path'])
            and type(producer['python']['mode']) is int and producer['python']['mode'] & 0o111,
            'complete actual executable interpreter entity')
    script = read('inputs/implementation/run.py', producer['script']['sha256'])
    require(len(script) == producer['script']['bytes']
            and producer['script']['path'] == paths['implementation']+'/run.py', 'bound actual runner script')
    return context


def read_processes(read, report, context):
    processes = report['processes']
    require([p['name'] for p in processes] == list(NAMES), 'all eight unique original child processes')
    pids = set()
    actuals = {}
    for row in processes:
        name = row['name']
        wanted = plan(context, name)
        require(all(row.get(k) == v for k, v in wanted.items()),
                'exact process argv/cwd/environment/timeout/artifact intent: '+name)
        require(json.loads(read('processes/'+name+'.json')) == row, 'original supervisor result: '+name)
        intent = json.loads(read('processes/'+name+'.intent.json'))
        require(intent == dict(wanted, python=context['producer']['python'],
                               supervisor_pid=context['producer']['pid'], started=False,
                               stdout_EOF=False, stderr_EOF=False), 'original pre-spawn intent: '+name)
        require(row['python'] == context['producer']['python']
                and row['python_sha256'] == row['python']['sha256']
                and row['supervisor_pid'] == context['producer']['pid']
                and row['started'] is True and type(row['pid']) is int and row['pid'] > 0
                and row['pid'] not in pids and row['exit_code'] == 0
                and row['stdout_EOF'] is True and row['stderr_EOF'] is True
                and row['timed_out'] is False and row['execution_error'] is None,
                'actual unique successful child with dual EOF: '+name)
        pids.add(row['pid'])
        for channel in ('stdout', 'stderr'):
            raw = read(wanted[channel+'_path'], row[channel+'_sha256'])
            require(len(raw) == row[channel+'_bytes'], 'complete child stream: '+name+'/'+channel)
            if channel == 'stderr':
                require(raw == b'', 'clean child stderr: '+name)
        actual_raw = read(wanted['actual_process_path'], row['actual_process_sha256'])
        actual = json.loads(actual_raw)
        flags = dict(isolated=0, no_site=0, no_user_site=0, ignore_environment=0,
                     dont_write_bytecode=1, optimize=0) if name == 'pytorch76' else context['producer']['flags']
        require(actual['schema'] == 'heptatorch.rng-consumer.actual-process.v2'
                and actual['name'] == name and actual['pid'] == row['pid']
                and actual['ppid'] == row['supervisor_pid']
                and actual['argv'] == wanted['argv'] and actual['cwd'] == wanted['cwd']
                and actual['environment'] == wanted['environment']
                and actual['python'] == row['python'] and actual['flags'] == flags
                and actual['python_runtime'] == context['producer']['python_runtime']
                and actual['platform'] == context['producer']['platform'],
                'actual child identity independently matches supervisor: '+name)
        if name in ('build', 'prepare', 'prepare-custody'):
            script = {'build':'build.py', 'prepare':'prepare.py', 'prepare-custody':'prepare_custody.py'}[name]
            relative = 'inputs/implementation/'+script
            script_path = context['paths']['implementation']+'/'+script
        else:
            script = {'custody':'check_rng_custody.py','cpython74':'rng_cpu_oracle.py',
                      'pytorch76':'rng_cpu_oracle.py','flat10':'flat_probe.py','independent25':'review_rng.py'}[name]
            relative = 'code/adapted/'+script
            script_path = context['paths']['output']+'/'+relative
        script_raw = read(relative, actual['script']['sha256'])
        require(actual['script']['path'] == script_path and actual['script']['bytes'] == len(script_raw)
                and type(actual['script']['mode']) is int, 'actual exact child script bytes: '+name)
        actuals[name] = actual
    return actuals
