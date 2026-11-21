"""Capture actual installed reference files; replay the original eleven checks.

The reader never imports Torch or executes the original oracle. Its CHECKS
expression is extracted from the fixed original source and evaluated only on
the captured observed fields and raw lock/pip/PEP610 bytes. A list of eleven
true booleans is not accepted as installation or source evidence.
"""
from pathlib import Path, PurePosixPath
import ast
import atexit
import base64
import csv
import email.parser
import importlib.metadata
import io
import json
import sys

from process_evidence import actual_process, identity, require, sha

SUPPORT = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
INVENTORY = 'd204c5a8268a0d0f0a1dd162b5e0ab184beee83310bcad643b084deb691f3f49'
REQUIREMENTS = 'cd49cdce73d64b096d08b346f171d13c29fd1c40b40e3845383b05c9ac4a0b89'
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CHECK_NAMES = ('python_matches', 'platform_matches', 'version_backend_matches',
               'git_matches_oracle_build_pin', 'distribution_version_matches',
               'direct_url_matches', 'direct_url_archive_sha256_matches',
               'pip_report_url_matches', 'pip_report_archive_sha256_matches',
               'pip_report_version_matches', 'pip_report_requested')
BASE_INPUTS = {'python.bin', 'torch-init.py', 'torch-version.py', 'torch-extension.bin',
               'torch-METADATA', 'torch-RECORD', 'torch-WHEEL', 'torch-INSTALLER',
               'support.py', 'oracle-lock.json', 'inventory-lock.json', 'reference-requirements.txt'}
OPTIONAL_INPUTS = {'torch-direct_url.json', 'pip-install.json', 'numpy-pip-install.json'}


def checks_from_original(support, observed, expected):
    require(sha(support) == SUPPORT, 'exact original eleven-check oracle source')
    assignments = [n for n in ast.parse(support).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'CHECKS' for t in n.targets)]
    require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict),
            'one original CHECKS dictionary expression')
    expression = ast.Expression(assignments[0].value)
    allowed = {'OBSERVED', 'EXPECTED', 'DIRECT', 'DOWNLOAD'}
    require({n.id for n in ast.walk(expression) if isinstance(n, ast.Name)} <= allowed,
            'only original observed/expected data in provenance expression')
    context = {'OBSERVED': observed, 'EXPECTED': expected,
               'DIRECT': observed.get('direct_url', {}),
               'DOWNLOAD': observed.get('install_report', {}).get('download_info', {})}
    result = eval(compile(expression, '<fixed-original-CHECKS>', 'eval'), {'__builtins__': {}}, context)
    require(tuple(result) == CHECK_NAMES and all(type(v) is bool for v in result.values()),
            'exact original eleven names and boolean predicates')
    return result


def begin_torch_capture(directory):
    capture = {'directory': Path(directory), 'stdout': io.StringIO(), 'stderr': io.StringIO(),
               'original_stdout': sys.stdout, 'original_stderr': sys.stderr, 'finished': False}
    sys.stdout, sys.stderr = capture['stdout'], capture['stderr']
    atexit.register(finish_torch_capture, capture)
    return capture


def finish_torch_capture(capture):
    if capture['finished']:
        return
    sys.stdout, sys.stderr = capture['original_stdout'], capture['original_stderr']
    for channel in ('stdout', 'stderr'):
        (capture['directory']/('torch-import.'+channel)).write_bytes(capture[channel].getvalue().encode())
    capture['finished'] = True


def _installed_record(pip):
    require(isinstance(pip, dict) and isinstance(pip.get('install'), list), 'complete original pip install report')
    records = [item for item in pip['install']
               if item.get('metadata', {}).get('name', '').lower() == 'torch']
    require(len(records) == 1, 'exactly one original requested torch installation record')
    row = records[0]
    return {'path': 'build/pytorch-install.json', 'pip_version': pip.get('pip_version'),
            'download_info': row['download_info'], 'metadata_name': row['metadata']['name'],
            'metadata_version': row['metadata']['version'], 'requested': row.get('requested')}


def _observed_base(process, version_values):
    return {'python': process['python_runtime'], 'platform': process['platform'],
            'version': version_values['__version__'], 'git_commit': version_values['git_version'],
            'backend_versions': {k: version_values[k] for k in ('cuda', 'hip', 'rocm', 'xpu')}}


def record_reference(repository, directory, torch, provenance, local):
    """Called only by the actual Torch child after the unchanged original branch."""
    repo, out = Path(repository).resolve(), Path(directory).resolve()
    destination = out/'reference-inputs'
    destination.mkdir(exist_ok=False)
    files = {}
    def capture(name, path):
        record = identity(path)
        raw = Path(record['path']).read_bytes()
        require(sha(raw) == record['sha256'], 'reference file changed before capture')
        (destination/name).write_bytes(raw)
        files[name] = record
        return raw
    support = capture('support.py', repo/'tests/pytorch_oracle.py')
    lock = json.loads(capture('oracle-lock.json', repo/'compatibility/pytorch-oracle-lock.json'))
    capture('inventory-lock.json', repo/'compatibility/upstream-lock.json')
    capture('reference-requirements.txt', repo/'compatibility/pytorch-reference-requirements.txt')
    capture('python.bin', sys.executable)
    capture('torch-init.py', torch.__file__)
    capture('torch-version.py', torch.version.__file__)
    capture('torch-extension.bin', torch._C.__file__)
    distribution = importlib.metadata.distribution('torch')
    dist_info = Path(distribution._path).resolve(strict=True)
    for name in ('METADATA', 'RECORD', 'WHEEL', 'INSTALLER'):
        capture('torch-'+name, dist_info/name)
    direct_path = dist_info/'direct_url.json'
    direct = json.loads(capture('torch-direct_url.json', direct_path)) if direct_path.is_file() else None
    pip_path = repo/lock['oracle']['install_report']
    pip = json.loads(capture('pip-install.json', pip_path)) if pip_path.is_file() else None
    support_pip = repo/'build/pytorch-reference-support-install.json'
    if support_pip.is_file():
        capture('numpy-pip-install.json', support_pip)
    process = actual_process(Path(sys.argv[0]))
    process['name'] = 'pytorch76'
    values = {'__version__': str(torch.__version__), 'git_version': torch.version.git_version,
              **{k: getattr(torch.version, k, 'MISSING') for k in lock['oracle']['backend_versions']}}
    observed = _observed_base(process, values)
    error = None
    if direct is None:
        error = 'ValueError: Installed torch lacks PEP610 direct_url.json'
    elif pip is None:
        error = 'ValueError: Original pip install report is absent'
    else:
        installed = _installed_record(pip)
        observed.update(distribution_version=distribution.version, direct_url=direct, install_report=installed)
    checks = checks_from_original(support, observed, lock['oracle'])
    if not local:
        require(error is None and observed == provenance['observed'] and checks == provenance['checks']
                and all(checks.values()), 'actual raw installation records must reproduce all original strict predicates')
    streams = {}
    for channel in ('stdout', 'stderr'):
        raw = (out/('torch-import.'+channel)).read_bytes()
        streams[channel] = {'bytes': len(raw), 'sha256': sha(raw)}
    receipt = {'schema': 'heptatorch.rng-consumer.reference-inputs.v2',
               'local_observed_oracle': local, 'actual_process': process,
               'distribution': {'name': 'torch', 'version': distribution.version,
                                'root': str(Path(distribution.locate_file('')).resolve()),
                                'dist_info': str(dist_info)},
               'torch_values': values, 'observed': observed, 'checks': checks,
               'metadata_error': error, 'original_provenance': provenance,
               'reference_device': 'cpu', 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
               'inputs': files, 'missing_inputs': sorted(OPTIONAL_INPUTS-set(files)),
               'import_streams': streams, 'original_eleven_checks_recomputed': True,
               'full_installation_provenance_verified': not local,
               'production_oracle_qualified': not local}
    (out/'reference-inputs.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    return receipt


def _version_values(raw):
    values = {}
    for node in ast.parse(raw).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            if isinstance(target, ast.Name) and target.id in ('__version__','git_version','cuda','hip','rocm','xpu'):
                require(target.id not in values, 'unique installed Torch version field')
                values[target.id] = ast.literal_eval(node.value)
    require(set(values) == {'__version__','git_version','cuda','hip','rocm','xpu'},
            'complete installed Torch version source fields')
    return values


def read_reference(read, report, context, actual):
    local = context['local_observed_oracle']
    receipt = json.loads(read('pytorch76/reference-inputs.json'))
    require(receipt['schema'] == 'heptatorch.rng-consumer.reference-inputs.v2'
            and receipt['local_observed_oracle'] is local
            and receipt['actual_process'] == actual
            and receipt['original_eleven_checks_recomputed'] is True
            and receipt['full_installation_provenance_verified'] is (not local)
            and receipt['production_oracle_qualified'] is (not local)
            and receipt['reference_device'] == 'cpu', 'actual independent reference process and scope')
    inputs = receipt['inputs']
    require(BASE_INPUTS <= set(inputs) <= BASE_INPUTS | OPTIONAL_INPUTS
            and receipt['missing_inputs'] == sorted(OPTIONAL_INPUTS-set(inputs)),
            'complete actual reference raw-input census')
    if not local:
        require(set(inputs) == BASE_INPUTS | OPTIONAL_INPUTS, 'all strict raw reference inputs are present')
    raw = {}
    for name, record in inputs.items():
        raw[name] = read('pytorch76/reference-inputs/'+name, record['sha256'])
        require(len(raw[name]) == record['bytes'] and type(record['mode']) is int
                and str(PurePosixPath(record['path'])) == record['path']
                and record['path'].startswith('/'), 'actual complete reference input entity: '+name)
    for name, relative, pin in (
        ('support.py','tests/pytorch_oracle.py',SUPPORT),
        ('oracle-lock.json','compatibility/pytorch-oracle-lock.json',LOCK),
        ('inventory-lock.json','compatibility/upstream-lock.json',INVENTORY),
        ('reference-requirements.txt','compatibility/pytorch-reference-requirements.txt',REQUIREMENTS)):
        require(sha(raw[name]) == pin and raw[name] == read('inputs/repository/'+relative)
                and inputs[name]['path'] == context['paths']['repository']+'/'+relative,
                'actual exact original source/lock input: '+relative)
    require(raw['python.bin'] == read('inputs/runtime/python.bin') and inputs['python.bin'] == actual['python'],
            'actual reference interpreter entity equals supervised interpreter')
    lock, inventory = json.loads(raw['oracle-lock.json']), json.loads(raw['inventory-lock.json'])
    expected = lock['oracle']
    require(lock['format'] == 'heptatorch.pytorch-oracle-lock.v1'
            and lock['inventory_commit'] == inventory['commit'] == PIN
            and lock['inventory_tag'] == 'v2.15.0'
            and lock['release_source_equivalence']['inventory_commit'] == PIN
            and lock['release_source_equivalence']['wheel_build_commit'] == expected['git_commit'],
            'original exact source-inventory/wheel equivalence preconditions')
    distribution = receipt['distribution']
    require(distribution['name'] == 'torch' and PurePosixPath(distribution['dist_info']).parent == PurePosixPath(distribution['root'])
            and PurePosixPath(distribution['dist_info']).name.endswith('.dist-info'), 'actual installed distribution paths')
    meta = email.parser.BytesParser().parsebytes(raw['torch-METADATA'])
    require(meta['Name'].lower() == 'torch' and meta['Version'] == distribution['version'],
            'raw installed distribution metadata')
    package_root = PurePosixPath(distribution['root'])
    require(inputs['torch-init.py']['path'] == str(package_root/'torch/__init__.py')
            and inputs['torch-version.py']['path'] == str(package_root/'torch/version.py')
            and PurePosixPath(inputs['torch-extension.bin']['path']).parent == package_root/'torch'
            and PurePosixPath(inputs['torch-extension.bin']['path']).name.startswith('_C.')
            and raw['torch-extension.bin'][:4] == b'\x7fELF', 'actual imported Torch package and extension entities')
    records = list(csv.reader(io.StringIO(raw['torch-RECORD'].decode())))
    require(all(len(r) == 3 for r in records) and len({r[0] for r in records}) == len(records),
            'complete unique installed wheel RECORD')
    record_map = {r[0]: r[1:] for r in records}
    for name in ('torch-METADATA','torch-WHEEL','torch-INSTALLER','torch-direct_url.json'):
        if name in inputs:
            suffix = name.removeprefix('torch-')
            require(inputs[name]['path'] == distribution['dist_info']+'/'+suffix,
                    'actual metadata input path: '+name)
    require(inputs['torch-RECORD']['path'] == distribution['dist_info']+'/RECORD', 'actual installed RECORD path')
    for name in ('torch-init.py','torch-version.py','torch-extension.bin','torch-METADATA','torch-WHEEL','torch-INSTALLER','torch-direct_url.json'):
        if name not in inputs:
            continue
        relative = str(PurePosixPath(inputs[name]['path']).relative_to(package_root))
        encoded = base64.urlsafe_b64encode(bytes.fromhex(sha(raw[name]))).decode().rstrip('=')
        require(record_map.get(relative) == ['sha256='+encoded, str(len(raw[name]))],
                'actual installed input is byte-bound by full wheel RECORD: '+name)
    values = _version_values(raw['torch-version.py'])
    require(values == receipt['torch_values'] and values['__version__'] == distribution['version'],
            'actual observed Torch identity matches installed version source and metadata')
    observed = _observed_base(actual, values)
    error = None
    direct = json.loads(raw['torch-direct_url.json']) if 'torch-direct_url.json' in raw else None
    pip = json.loads(raw['pip-install.json']) if 'pip-install.json' in raw else None
    if direct is None:
        error = 'ValueError: Installed torch lacks PEP610 direct_url.json'
    elif pip is None:
        error = 'ValueError: Original pip install report is absent'
    else:
        observed.update(distribution_version=distribution['version'], direct_url=direct,
                        install_report=_installed_record(pip))
    for name, path in (('pip-install.json','build/pytorch-install.json'),
                       ('numpy-pip-install.json','build/pytorch-reference-support-install.json')):
        if name in raw:
            require(raw[name] == read('inputs/repository/'+path)
                    and inputs[name]['path'] == context['paths']['repository']+'/'+path,
                    'actual same-execution raw pip report: '+path)
    require(observed == receipt['observed'] and error == receipt['metadata_error'],
            'all observed metadata fields reconstructed from raw interpreter/version/PEP610/pip entities')
    checks = checks_from_original(raw['support.py'], observed, expected)
    require(receipt['checks'] == checks and set(receipt['checks']) == set(CHECK_NAMES)
            and all(type(v) is bool for v in receipt['checks'].values()), 'exact original checks independently recomputed')
    provenance = report['oracle_provenance']
    require(receipt['original_provenance'] == provenance, 'actual raw reference receipt and comparison report agree')
    for channel in ('stdout','stderr'):
        stream = read('pytorch76/torch-import.'+channel, receipt['import_streams'][channel]['sha256'])
        require(len(stream) == receipt['import_streams'][channel]['bytes'] and stream == b'',
                'complete silent actual reference import stream: '+channel)
    if local:
        require(provenance['status'] == 'EXPLORATORY_OBSERVED_TORCH_ONLY'
                and provenance['full_installation_provenance_verified'] is False
                and provenance['original_eleven_checks_passed'] is False
                and provenance['production_oracle_qualified'] is False
                and report['allow_unverified_local'] is True
                and '--allow-unverified-local' in actual['argv'], 'exploratory receipt cannot gain strict qualification')
        require(json.loads(read('pytorch76/observed-torch.json')) == provenance, 'actual original exploratory receipt')
        require(provenance['actual_process']['pid'] == actual['pid']
                and provenance['actual_process']['argv'] == actual['argv']
                and provenance['actual_process']['python'] == {k:actual['python'][k] for k in ('path','bytes','sha256')}
                and provenance['package'] == {k:inputs['torch-init.py'][k] for k in ('path','bytes','sha256')}
                and provenance['extension'] == {k:inputs['torch-extension.bin'][k] for k in ('path','bytes','sha256')}
                and provenance['version'] == values['__version__'] and provenance['git_version'] == values['git_version'],
                'exploratory observed package/extension/interpreter entities are actual')
    else:
        require(report['allow_unverified_local'] is False and '--allow-unverified-local' not in actual['argv']
                and provenance['status'] == 'PASS' and provenance['metadata_error'] is None
                and error is None and all(checks.values())
                and set(provenance['checks']) == set(CHECK_NAMES) and provenance['checks'] == checks
                and all(type(v) is bool for v in provenance['checks'].values())
                and provenance['lock_path'] == context['paths']['repository']+'/compatibility/pytorch-oracle-lock.json'
                and provenance['lock_sha256'] == LOCK and provenance['expected'] == expected
                and provenance['observed'] == observed
                and provenance['release_source_equivalence'] == lock['release_source_equivalence'],
                'all eleven original strict conditions and raw input/source identities independently recomputed')
        require(sha(raw['torch-METADATA']) == expected['wheel']['metadata_sha256'], 'original official wheel metadata entity')
        support_rows = [x for x in json.loads(raw['numpy-pip-install.json'])['install']
                        if x.get('metadata',{}).get('name','').lower() == 'numpy']
        requirement = [s.strip() for s in raw['reference-requirements.txt'].decode().splitlines()
                       if s.strip() and not s.lstrip().startswith('#')]
        require(len(requirement) == len(support_rows) == 1, 'one fixed NumPy support requirement and actual install row')
        url, digest = requirement[0].split('#sha256=')
        require(support_rows[0]['download_info']['url'] == url
                and support_rows[0]['download_info']['archive_info']['hashes']['sha256'] == digest
                and support_rows[0]['metadata']['version'] == '2.3.5' and support_rows[0]['requested'] is True,
                'same-execution fixed support dependency install receipt')
    return {'original_check_names': list(CHECK_NAMES), 'original_checks': checks,
            'raw_reference_entities': len(inputs), 'actual_process_verified': True,
            'original_eleven_checks_recomputed_from_raw_inputs': True,
            'full_installation_provenance_verified': not local}



WHEEL_HELPER_SHA = 'fa91a644b8743493ed6ed8f29da966494983cf5eed7300faa715ee4e08c7212e'


def wheel_custody(repository):
    """Load only the reviewed common wheel reader, including the executed copy."""
    import importlib.util
    path=Path(__file__).resolve().with_name('pinned_reference_custody.py')
    if not path.is_file():
        path=Path(repository).resolve()/'tests/pinned_reference_custody.py'
    require(not path.is_symlink() and sha(path.read_bytes())==WHEEL_HELPER_SHA,
            'exact independently reviewed complete official wheel reader')
    spec=importlib.util.spec_from_file_location('rng_fixed_wheel_custody',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _wheel_process():
    process=actual_process(Path(sys.argv[0]));process['name']='pytorch76'
    return process


def begin_reference_wheel(repository,directory):
    module=wheel_custody(repository)
    out=Path(directory)/'reference-wheel';out.mkdir(exist_ok=False)
    state=module.capture_wheel_entities(repository,out,_wheel_process())
    state['reader_module']=module
    return state


def finish_reference_wheel(state):
    state['reader_module'].finish_wheel_entities(state,_wheel_process())


def _shared_wheel_reader(repository,module):
    repo=Path(repository).resolve();records={}
    sources={path.relative_to(repo).as_posix() for path in module.source_files(repo)}
    def bind(path,expected=None):
        path=Path(path).absolute()
        require(path.is_relative_to(repo),'contained shared reference input')
        relative=path.relative_to(repo).as_posix()
        require(relative in sources or relative.startswith(module.INSTALLED_PATH+'/'),
                'one fixed official wheel or installed snapshot role')
        raw,identity_row=module.stable_bytes(path)
        require(expected is None or sha(raw)==expected,'independent shared wheel entity hash')
        row={key:identity_row[key] for key in ('bytes','sha256','mode')}
        require(relative not in records or records[relative]==row,'shared original wheel input changed')
        records[relative]=row
        return raw
    return bind,records


def reference_wheel_inputs(repository):
    """Declare the complete shared raw corpus without importing Torch."""
    module=wheel_custody(repository);repo=Path(repository).resolve()
    bind,records=_shared_wheel_reader(repo,module)
    anchor=module.wheel_anchor(repo,bind)
    for name,row in module.immutable_members(anchor).items():
        raw=bind(repo/module.INSTALLED_PATH/name,row['sha256'])
        require(len(raw)==row['bytes'] and records[module.INSTALLED_PATH+'/'+name]['mode']==module.installed_mode(row),
                'entire immutable original wheel snapshot and exact installed mode')
    require(len(records)==12569,'exact two source inputs plus all12567 installed members')
    return records


def read_reference_wheel(repository,directory,read,actual):
    """Add independent official member anchors to the unchanged raw eleven checks."""
    module=wheel_custody(repository);repo=Path(repository).resolve();out=Path(directory).resolve()
    shared,records=_shared_wheel_reader(repo,module)
    def bind(path,expected=None):
        path=Path(path).absolute()
        if path.is_relative_to(out):
            return read(path.relative_to(out).as_posix(),expected)
        return shared(path,expected)
    proof=module.read_wheel_entities(repo,out/'pytorch76/reference-wheel',bind,actual)
    anchor=module.wheel_anchor(repo,bind)
    before=json.loads(read('pytorch76/reference-wheel/installed-before.json'))
    receipt=json.loads(read('pytorch76/reference-inputs.json'))
    require(receipt['actual_process']==actual==before['actual_process']
            and receipt['distribution']['root']==before['distribution_root']
            and receipt['distribution']['dist_info']==before['dist_info'],
            'the original strict eleven entities and complete wheel came from the same actual process and installation')
    for name,member in anchor['captured_core_members'].items():
        if name=='original-wheel-RECORD':
            continue
        raw=read('pytorch76/reference-inputs/'+name)
        row=receipt['inputs'][name];expected=anchor['members'][member]
        require(row['path']==before['distribution_root']+'/'+member
                and row['sha256']==sha(raw)==expected['sha256']
                and row['bytes']==len(raw)==expected['bytes']
                and row['mode']==module.installed_mode(expected)
                and raw==bind(repo/module.INSTALLED_PATH/member,expected['sha256']),
                'original core reference entity equals the independent official archive and shared bytes: '+name)
    require(len(records)==12569,'all complete original wheel inputs are bound by the pure reader')
    for relative,row in list(records.items()):
        bind(repo/relative,row['sha256'])
    return dict(proof=proof,shared_inputs=records)
