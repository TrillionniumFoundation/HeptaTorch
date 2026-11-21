"""Actual reference process and raw fixed-wheel custody, with a pure reader.

The original eleven predicates are re-evaluated from exact lock, installed
version/metadata/RECORD/PEP610, and original pip bytes. Source paths in receipts
are producer labels; archive reads use local evidence paths. No Torch operation
or subprocess is started by read().
"""
from pathlib import Path, PurePosixPath
import ast
import atexit
import base64
import csv
import email.parser
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import posixpath
import re
import stat
import sys

FLAGS=('isolated','no_site','no_user_site','ignore_environment','dont_write_bytecode','optimize')
SUPERVISOR='36e3432deac48085ba0586c85547cee8432d1533c5915238db2439b43dbca523'
BASE_ENVIRONMENT={'PATH':'/bin:/usr/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8'}

def require(value,message):
    if not value:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def identity(path):
    path=Path(path).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(),'regular reference input entity')
    raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(path.stat().st_mode))

def actual_process(script,include_environment=True):
    return dict(schema='heptatorch.pinned-reference.actual-process.v1',name='fixed_reference',
        pid=os.getpid(),ppid=os.getppid(),argv=list(sys.orig_argv),cwd=str(Path.cwd().resolve()),
        python=identity(sys.executable),script=identity(script),
        flags={name:getattr(sys.flags,name) for name in FLAGS},
        python_runtime=dict(implementation=sys.implementation.name,version=list(sys.version_info[:2]),full_version=sys.version),
        platform=dict(sys_platform=sys.platform,machine=platform.machine()),
        environment=dict(os.environ) if include_environment else None)

def normalized_absolute(value):
    return isinstance(value,str) and value.startswith('/') and posixpath.normpath(value)==value and '\\' not in value

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
WHEEL_MEMBERS='0e5377c5bd2767e35bfbcb81346dbc522ad9864e0d2d68b6a3705e0befd042f0'
WHEEL_SHA256='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3'
WHEEL_BYTES=209530490
WHEEL_MEMBER_PATH='compatibility/pytorch-official-wheel-members.json'
WHEEL_ARCHIVE_PATH='build/pytorch-reference-wheel.whl'
INSTALLED_PATH='build/pytorch-reference-installed'


def source_files(repository):
    """The shared immutable source inputs used by every new reference capsule."""
    repo=Path(repository)
    return [repo/WHEEL_MEMBER_PATH,repo/WHEEL_ARCHIVE_PATH]


def stable_bytes(path):
    path=Path(path).absolute();before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink() and path.resolve()==path,
            'regular original installed source entity: '+str(path))
    raw=path.read_bytes();after=path.stat()
    fields=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    require(fields(before)==fields(after),'original source entity changed during read: '+str(path))
    return raw,dict(path=str(path),bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(after.st_mode),
                    device=after.st_dev,inode=after.st_ino)


def wheel_anchor(repository,bind=None):
    repo=Path(repository)
    read=bind if bind is not None else lambda path,expected=None:stable_bytes(path)[0]
    anchor_raw=read(repo/WHEEL_MEMBER_PATH,WHEEL_MEMBERS)
    require(sha(anchor_raw)==WHEEL_MEMBERS,'independent complete official wheel member anchor')
    anchor=json.loads(anchor_raw)
    require(anchor['schema']=='heptatorch.pinned-pytorch-wheel-members.v1'
            and anchor['wheel']['sha256']==WHEEL_SHA256 and anchor['wheel']['bytes']==WHEEL_BYTES
            and anchor['member_count']==len(anchor['members'])==13337
            and anchor['file_count']==12568 and anchor['directory_count']==769
            and len(anchor['shared_libraries'])==12,'complete independently verified official wheel census')
    wheel=read(repo/WHEEL_ARCHIVE_PATH,WHEEL_SHA256)
    require(len(wheel)==WHEEL_BYTES and sha(wheel)==WHEEL_SHA256,'complete original official wheel archive bytes')
    return anchor


def immutable_members(anchor):
    original_record=anchor['captured_core_members']['original-wheel-RECORD']
    result={name:row for name,row in anchor['members'].items() if row['kind']=='file' and name!=original_record}
    require(len(result)==12567,'all official immutable members except the pip-rewritten installed RECORD')
    return result


def installed_mode(member):
    # pip's regular file extraction applies the runner's fixed022 umask;
    # the official archive contains65 group-writable0664 source entries.
    # Keep original ZIP modes in the independent anchor and constrain the
    # actual installed mode separately instead of accepting a self-report.
    return member['mode'] & ~0o022


def _publish_snapshot(path,raw,mode):
    """One physical shared snapshot; never overwrite an existing entity."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old,record=stable_bytes(path)
        require(old==raw and record['mode']==mode,'shared installed snapshot retained unchanged')
        return
    temporary=path.with_name(path.name+'.capturing-'+str(os.getpid()))
    try:
        with temporary.open('xb') as stream:stream.write(raw)
        temporary.chmod(mode)
        try:os.link(temporary,path)
        except FileExistsError:
            old,record=stable_bytes(path)
            require(old==raw and record['mode']==mode,'concurrent shared installed snapshot agrees')
    finally:
        if temporary.exists():temporary.unlink()


def prepare_installed(repository,wheel=None,snapshot_pool=None):
    """Prepare one shared byte snapshot after the original URL-based pip install.

    This command imports no Torch. Existing pip URL/provenance records are not
    modified. Local isolated consumers may hardlink the same immutable pool.
    """
    repo=Path(repository).resolve();target=repo/WHEEL_ARCHIVE_PATH;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        candidates=[Path(wheel).resolve()] if wheel is not None else sorted((Path.home()/'.cache/pip').glob('http*/**/*.body'))
        matches=[]
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size==WHEEL_BYTES:
                raw,_=stable_bytes(candidate)
                if sha(raw)==WHEEL_SHA256:matches.append(candidate)
        require(matches,'exact original wheel absent; pass --wheel pointing to the official archive')
        try:os.link(matches[0],target)
        except OSError:_publish_snapshot(target,matches[0].read_bytes(),stat.S_IMODE(matches[0].stat().st_mode))
    anchor=wheel_anchor(repo);members=immutable_members(anchor)
    distribution=importlib.metadata.distribution('torch')
    root=Path(distribution.locate_file('')).resolve(strict=True)
    require(Path(distribution._path).resolve()==root/anchor['dist_info'],'actual fixed distribution root')
    pool=Path(snapshot_pool).resolve() if snapshot_pool is not None else repo/INSTALLED_PATH
    total=0
    for name,expected in members.items():
        raw,record=stable_bytes(root/name)
        require(record['bytes']==expected['bytes'] and record['sha256']==expected['sha256']
                and record['mode']==installed_mode(expected),'installed original member equals independent official wheel: '+name)
        pooled=pool/name;_publish_snapshot(pooled,raw,record['mode'])
        destination=repo/INSTALLED_PATH/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination!=pooled:
            if not destination.exists():os.link(pooled,destination)
            saved,saved_record=stable_bytes(destination)
            require(saved==raw and saved_record['mode']==record['mode'],'same shared original installed entity')
        total+=len(raw)
    return dict(schema='heptatorch.pinned-reference.shared-installed-preparation.v1',
        official_wheel_sha256=WHEEL_SHA256,official_member_anchor_sha256=WHEEL_MEMBERS,
        installed_members=len(members),installed_bytes=total,Torch_executed=False,native_executed=False)


def _installation_observation(repository,distribution_root,anchor):
    repo,root=Path(repository),Path(distribution_root);members=immutable_members(anchor);observed={}
    for name,expected in members.items():
        raw,record=stable_bytes(root/name)
        require(record['bytes']==expected['bytes'] and record['sha256']==expected['sha256']
                and record['mode']==installed_mode(expected),'actual imported installation equals independent official wheel: '+name)
        saved,snapshot=stable_bytes(repo/INSTALLED_PATH/name)
        require(saved==raw and snapshot['mode']==record['mode'],'actual installed bytes equal shared physical snapshot: '+name)
        observed[name]=record
    return observed


def capture_wheel_entities(repository,directory,actual):
    """Capture the original physical installed entities used by the actual process."""
    repo,out=Path(repository).resolve(),Path(directory).resolve();anchor=wheel_anchor(repo)
    require(actual['pid']==os.getpid() and actual['argv']==list(sys.orig_argv)
        and actual['cwd']==str(Path.cwd().resolve()),'wheel capture belongs to this actual reference process')
    distribution=importlib.metadata.distribution('torch');root=Path(distribution.locate_file('')).resolve(strict=True)
    require(Path(distribution._path).resolve()==root/anchor['dist_info'],'same exact imported distribution tree')
    observed=_installation_observation(repo,root,anchor)
    maps=Path('/proc/self/maps').read_bytes();(out/'installed-before.maps').write_bytes(maps)
    record=dict(schema='heptatorch.pinned-reference.installed-wheel-entities.v1',
        official_wheel_sha256=WHEEL_SHA256,official_member_anchor_sha256=WHEEL_MEMBERS,
        snapshot_role=INSTALLED_PATH,distribution_root=str(root),dist_info=str(root/anchor['dist_info']),
        actual_process=actual,members=observed,
        maps=dict(path='installed-before.maps',bytes=len(maps),sha256=sha(maps)))
    (out/'installed-before.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    return dict(repository=repo,directory=out,anchor=anchor,record=record)


def finish_wheel_entities(state,actual):
    """Independently re-read actual installed files after the numerical program."""
    record=state['record'];observed=_installation_observation(state['repository'],record['distribution_root'],state['anchor'])
    require(observed==record['members'] and actual==record['actual_process'],
            'all original installed byte entities and complete process identity survived actual reference execution')
    maps=Path('/proc/self/maps').read_bytes();out=state['directory'];(out/'installed-after.maps').write_bytes(maps)
    after=dict(record,maps=dict(path='installed-after.maps',bytes=len(maps),sha256=sha(maps)))
    (out/'installed-after.json').write_text(json.dumps(after,sort_keys=True,indent=2)+'\n')


def _verify_mapped_torch_entities(raw,record,anchor):
    root=record['distribution_root']+'/' ;seen=set();executable=set();previous_end=0
    wheel_roots={name.split('/',1)[0] for name in anchor['members']}
    native_names={PurePosixPath(name).name for name in anchor['shared_libraries']}
    native_names.add(PurePosixPath(anchor['captured_core_members']['torch-extension.bin']).name)
    require(raw and raw.endswith(b'\n') and b'\r' not in raw,'complete original Linux mapping text')
    for line in raw.decode().splitlines():
        match=re.fullmatch(r'([0-9a-f]+)-([0-9a-f]+) +([r-][w-][x-][ps]) +([0-9a-f]+) +([0-9a-f]+):([0-9a-f]+) +([0-9]+)(?: +(.*))?',line)
        require(match is not None,'complete original mapping address permissions offset device inode grammar')
        start,end=int(match[1],16),int(match[2],16);permissions=match[3];offset=int(match[4],16)
        major,minor=int(match[5],16),int(match[6],16);inode=int(match[7]);path=match[8] or ''
        require(0<start<end<=1<<64 and start>=previous_end
            and start%4096==end%4096==offset%4096==0 and 0<=offset<1<<64
            and 0<=major<1<<32 and 0<=minor<1<<32 and 0<=inode<1<<64,
            'actual ordered non-overlapping page-aligned mapping address/device extents')
        previous_end=end;original_path=path
        while path.endswith(' (deleted)'):path=path[:-10]
        if PurePosixPath(path).name in native_names:
            require(path==original_path and path.startswith(root),
                    'mapped Torch computation code is neither deleted nor from a foreign installation')
        if not path.startswith(root):continue
        relative=path[len(root):]
        if relative.split('/',1)[0] not in wheel_roots:continue
        require(path==original_path and relative in record['members'],
                'every mapped Torch file is a live original immutable wheel member')
        row=record['members'][relative]
        require(inode==row['inode'] and (major,minor)==(os.major(row['device']),os.minor(row['device'])),
                'mapped actual Torch code inode/device matches the physically read original installation')
        seen.add(relative)
        if permissions[2]=='x':executable.add(relative)
    required={anchor['captured_core_members']['torch-extension.bin'],
        'torch/lib/libtorch_cpu.so','torch/lib/libtorch_python.so','torch/lib/libc10.so'}
    require(required<=seen and required<=executable,
            'actual Torch CPU extension and computation libraries each have an executable mapping in the same process')
    return sorted(seen)


def read_wheel_entities(repository,directory,bind,actual=None):
    """Pure readback of every shared installed file and both actual mapping records."""
    require(isinstance(actual,dict),'independently admitted actual process is mandatory for installed-byte custody')
    repo,out=Path(repository),Path(directory);anchor=wheel_anchor(repo,bind);members=immutable_members(anchor)
    before=json.loads(bind(out/'installed-before.json'));after=json.loads(bind(out/'installed-after.json'))
    require(set(before)==set(after)=={'schema','official_wheel_sha256','official_member_anchor_sha256','snapshot_role',
        'distribution_root','dist_info','actual_process','members','maps'}
        and before['schema']=='heptatorch.pinned-reference.installed-wheel-entities.v1'
        and before['official_wheel_sha256']==WHEEL_SHA256 and before['official_member_anchor_sha256']==WHEEL_MEMBERS
        and before['snapshot_role']==INSTALLED_PATH and normalized_absolute(before['distribution_root'])
        and before['dist_info']==before['distribution_root']+'/'+anchor['dist_info']
        and (actual is None or before['actual_process']==actual)
        and {k:v for k,v in before.items() if k!='maps'}=={k:v for k,v in after.items() if k!='maps'}
        and set(before['members'])==set(members),'complete before/after original installed member census and actual process')
    total=0
    for name,expected in members.items():
        row=before['members'][name]
        require(set(row)=={'path','bytes','sha256','mode','device','inode'}
            and row['path']==before['distribution_root']+'/'+name
            and {k:row[k] for k in ('bytes','sha256','mode')}==dict(bytes=expected['bytes'],sha256=expected['sha256'],mode=installed_mode(expected))
            and type(row['inode']) is int and row['inode']>0 and type(row['device']) is int and row['device']>=0,
            'actual installed member has the independent complete official identity: '+name)
        raw=bind(repo/INSTALLED_PATH/name,expected['sha256'])
        require(len(raw)==expected['bytes'] and stat.S_IMODE((repo/INSTALLED_PATH/name).stat().st_mode)==installed_mode(expected),
                'complete shared original installed entity and mode: '+name)
        total+=len(raw)
    mapped=[]
    for label,record in (('before',before),('after',after)):
        item=record['maps'];require(set(item)=={'path','bytes','sha256'} and item['path']=='installed-'+label+'.maps','exact installed mapping role')
        raw=bind(out/item['path'],item['sha256']);require(len(raw)==item['bytes'],'complete actual installed code mappings')
        mapped.append(_verify_mapped_torch_entities(raw,record,anchor))
    return dict(official_wheel_sha256=WHEEL_SHA256,official_member_anchor_sha256=WHEEL_MEMBERS,
        independent_official_wheel_member_anchor_verified=True,complete_official_wheel_archive_verified=True,
        installed_wheel_members_verified=len(members),installed_wheel_bytes_verified=total,
        torch_shared_libraries_verified=len(anchor['shared_libraries']),actual_mapped_torch_cpu_code_verified=True,
        all_actual_installed_inputs_unchanged=True,complete_installed_raw_snapshot_verified=True)


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
    process['name'] = 'fixed_reference'
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
    # Installed RECORD is mutable. Its identities must independently match the
    # complete fixed official wheel, whose original RECORD is a different file.
    anchor_raw=read('inputs/repository/'+WHEEL_MEMBER_PATH)
    require(sha(anchor_raw)==WHEEL_MEMBERS,'independent official wheel anchor cannot be rewritten with installed RECORD')
    anchor=json.loads(anchor_raw);wheel=read('inputs/repository/'+WHEEL_ARCHIVE_PATH)
    require(sha(wheel)==WHEEL_SHA256 and len(wheel)==WHEEL_BYTES
        and anchor['wheel']['sha256']==WHEEL_SHA256,'complete independent original official wheel archive')
    for name,member in immutable_members(anchor).items():
        encoded=base64.urlsafe_b64encode(bytes.fromhex(member['sha256'])).decode().rstrip('=')
        require(record_map.get(name)==['sha256='+encoded,str(member['bytes'])],
                'installed RECORD preserves every immutable original official member: '+name)
    for name,member in anchor['captured_core_members'].items():
        if name=='original-wheel-RECORD':continue
        expected_member=anchor['members'][member]
        require(sha(raw[name])==expected_member['sha256'] and len(raw[name])==expected_member['bytes']
            and inputs[name]['mode']==installed_mode(expected_member),
            'captured imported code/metadata independently matches original official wheel: '+name)
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
        signbit=context.get('script_relative')=='tests/signbit/reference.py'
        local_checks=dict(
            python_matches=observed['python']['implementation']==expected['python']['implementation']
                and observed['python']['version']==expected['python']['version'],
            platform_matches=observed['platform']==expected['platform'],
            version_backend_matches=values['__version__']==expected['version']
                and {key:values[key] for key in expected['backend_versions']}==expected['backend_versions'],
            git_matches_oracle_build_pin=values['git_version']==expected['git_commit'],
            distribution_version_matches=distribution['version']==expected['version'])
        require(provenance['status']==('OBSERVED_INSTALLED_WHEEL_IDENTITY_ONLY' if signbit else 'EXPLORATORY_NOT_PINNED_WHEEL_QUALIFICATION')
                and provenance['full_installation_provenance_verified'] is False
                and report['allow_unverified_local'] is True
                and ('--local-observed-oracle' if signbit else '--allow-unverified-local') in actual['argv']
                and provenance['checks']==local_checks and all(local_checks.values())
                and all(type(value) is bool for value in provenance['checks'].values())
                and provenance['expected']==expected and provenance['lock_sha256']==LOCK,
                'actual exploratory process and five raw-derived local identity predicates cannot gain strict qualification')
        for label,name in (('package','torch-init.py'),('extension','torch-extension.bin')):
            require(provenance[label]==inputs[name],'actual exploratory package and extension entities')
        if signbit:
            require(provenance['version']==values['__version__'] and provenance['git_commit']==values['git_version']
                and provenance['direct_url']==direct and provenance['direct_url_record_available'] is (direct is not None)
                and provenance['original_pip_report_available'] is False and provenance['cpu_capability']==receipt['cpu_capability'],
                'exact unchanged legacy signbit local identity and explicit limited scope')
        else:
            require(provenance['version']==values['__version__'] and provenance['git_version']==values['git_version']
                    and provenance['metadata_location']==distribution['dist_info']
                    and provenance['direct_url_present'] is ('torch-direct_url.json' in inputs),
                    'actual exploratory installed metadata identity')
    else:
        require(report['allow_unverified_local'] is False and '--allow-unverified-local' not in actual['argv']
                and '--local-observed-oracle' not in actual['argv']
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
            'full_installation_provenance_verified': not local,
            'independent_official_wheel_member_anchor_verified': True,
            'complete_official_wheel_archive_verified': True,
            'official_wheel_sha256': WHEEL_SHA256,'official_member_anchor_sha256': WHEEL_MEMBERS}



def begin(repository,directory,script,allow_unverified_local=False):
    """Called before any environment setup or Torch import in the real child."""
    repo,out=Path(repository).resolve(),Path(directory).resolve()
    capsule=out/'reference-custody';capsule.mkdir(exist_ok=False)
    initial=actual_process(script)
    require(initial['cwd']==str(repo) and initial['environment']==BASE_ENVIRONMENT,
            'reference starts with the exact supervised clean environment and cwd')
    require(initial['flags']==dict(isolated=0,no_site=0,no_user_site=0,ignore_environment=0,dont_write_bytecode=1,optimize=0),
            'actual reference starts with exact non-optimized -B interpreter flags')
    os.environ['ATEN_CPU_CAPABILITY']='avx2'
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(capsule/'torchinductor-cache')
    initialized=actual_process(script)
    context=dict(schema='heptatorch.pinned-reference.context.v1',local_observed_oracle=allow_unverified_local,
        paths=dict(repository=str(repo),output=str(out),script=str(Path(script).resolve())),
        initial_process=initial,initialized_process=initialized,helper=identity(__file__))
    (capsule/'context.json').write_text(json.dumps(context,indent=2,sort_keys=True)+'\n')
    state=begin_torch_capture(capsule);state['context']=context;return state


def finish_capture(state):
    finish_torch_capture(state)


def capture(repository,directory,script,torch,provenance,allow_unverified_local,state):
    finish_capture(state)
    require(state['context']['local_observed_oracle'] is allow_unverified_local,
            'same explicit reference mode before and after import')
    receipt=record_reference(repository,Path(directory)/'reference-custody',torch,provenance,allow_unverified_local)
    require(receipt['actual_process']==state['context']['initialized_process'],
            'complete actual reference process identity unchanged by import')
    require(all(row['bytes']==0 for row in receipt['import_streams'].values()),'complete silent actual reference import')
    state['wheel']=capture_wheel_entities(repository,Path(directory)/'reference-custody',receipt['actual_process'])
    return dict(path='reference-custody/reference-inputs.json',
        sha256=sha((Path(directory)/'reference-custody/reference-inputs.json').read_bytes()),
        original_eleven_checks_recomputed_from_raw_inputs=True,
        full_installation_provenance_verified=not allow_unverified_local,
        production_oracle_qualified=not allow_unverified_local)


def finish_reference(state):
    """Finish only after the original numerical reference has really completed."""
    require('wheel' in state and not state.get('reference_finished'),'one complete actual reference wheel capture')
    finish_wheel_entities(state['wheel'],actual_process(state['context']['paths']['script']))
    state['reference_finished']=True


def supervise(supervisor,argv,directory,timeout,repository,bind):
    """Record pre-spawn intent, then use the unchanged process-tree supervisor."""
    out,repo=Path(directory).resolve(),Path(repository).resolve()
    require(out.is_dir() and not (out/'reference-intent.json').exists(),'new reference supervisor intent')
    actual=actual_process(Path(sys.argv[0]),include_environment=False);actual['name']='supervisor'
    require(actual['flags']['optimize']==0 and os.defpath==BASE_ENVIRONMENT['PATH'],
            'unaltered non-optimized reference supervisor')
    raw=Path(actual['python']['path']).read_bytes();(out/'supervisor-python.bin').write_bytes(raw)
    raw_script=Path(actual['script']['path']).read_bytes();(out/'supervisor-script.py').write_bytes(raw_script)
    intent=dict(schema='heptatorch.pinned-reference.intent.v1',argv=list(map(str,argv)),cwd=str(repo),
        environment=BASE_ENVIRONMENT,timeout_seconds=timeout,python=identity(argv[0]),
        supervisor=actual,started=False,stdout_EOF=False,stderr_EOF=False,
        source_supervisor=identity(repo/'tests/current_runtime_process.py'))
    bind(repo/'tests/current_runtime_process.py',SUPERVISOR)
    (out/'reference-intent.json').write_text(json.dumps(intent,indent=2,sort_keys=True)+'\n')
    return supervisor.execute(argv,out,timeout,cwd=repo)


def read(repository,directory,execution,report,bind,script_relative,allow=False):
    """Pure readback of raw installation entities, exact process intent, and EOF."""
    repo,out,execution=map(Path,(repository,directory,execution));capsule=out/'reference-custody'
    require(script_relative.startswith('tests/') and script_relative.endswith('/reference.py')
            and '..' not in PurePosixPath(script_relative).parts,'fixed reference script role')
    context=json.loads(bind(capsule/'context.json'))
    require(context['schema']=='heptatorch.pinned-reference.context.v1'
            and context['local_observed_oracle'] is allow and type(allow) is bool
            and set(context['paths'])=={'repository','output','script'}
            and all(normalized_absolute(value) for value in context['paths'].values()),
            'complete original reference producer labels and explicit mode')
    paths=context['paths'];initial=context['initial_process'];actual=context['initialized_process']
    expected_environment=dict(BASE_ENVIRONMENT,ATEN_CPU_CAPABILITY='avx2',
        TORCHINDUCTOR_CACHE_DIR=paths['output']+'/reference-custody/torchinductor-cache')
    require(initial['environment']==BASE_ENVIRONMENT
            and actual==dict(initial,environment=expected_environment),
            'only the two exact declared child environment initializations are permitted')
    require(actual['schema']=='heptatorch.pinned-reference.actual-process.v1'
            and actual['name']=='fixed_reference' and actual['cwd']==paths['repository']
            and actual['flags']==dict(isolated=0,no_site=0,no_user_site=0,ignore_environment=0,dont_write_bytecode=1,optimize=0)
            and type(actual['pid']) is int and actual['pid']>0
            and type(actual['ppid']) is int and actual['ppid']>0,
            'complete exact actual non-optimized reference process')
    intent=json.loads(bind(execution/'reference-intent.json'))
    argv=intent['argv']
    signbit=script_relative=='tests/signbit/reference.py'
    expected=[argv[0],'-B',paths['repository']+'/'+script_relative]
    if signbit:expected+=['--repository',paths['repository']]
    expected+=['--output',paths['output']]
    if allow:expected.append('--local-observed-oracle' if signbit else '--allow-unverified-local')
    require(argv==expected==actual['argv'] and normalized_absolute(argv[0])
            and paths['script']==expected[2] and intent['schema']=='heptatorch.pinned-reference.intent.v1'
            and intent['cwd']==paths['repository'] and intent['environment']==BASE_ENVIRONMENT
            and intent['started'] is False and intent['stdout_EOF'] is False and intent['stderr_EOF'] is False
            and intent['python']==actual['python'] and intent['timeout_seconds']==300,
            'exact pre-spawn reference argv cwd interpreter environment timeout and mode')
    source=bind(repo/script_relative,actual['script']['sha256'])
    require(actual['script']==dict(path=paths['script'],bytes=len(source),sha256=sha(source),mode=stat.S_IMODE((repo/script_relative).stat().st_mode)),
            'actual reference script equals complete current source entity')
    helper=bind(Path(__file__))
    require(context['helper']==dict(path=paths['repository']+'/tests/pinned_reference_custody.py',
        bytes=len(helper),sha256=sha(helper),mode=stat.S_IMODE(Path(__file__).stat().st_mode)),
        'same complete reference custody implementation entity')
    parent=intent['supervisor'];parent_python=bind(execution/'supervisor-python.bin',parent['python']['sha256'])
    parent_script=bind(execution/'supervisor-script.py',parent['script']['sha256'])
    require(parent['schema']=='heptatorch.pinned-reference.actual-process.v1' and parent['name']=='supervisor'
            and parent['environment'] is None and parent['pid']==actual['ppid']
            and type(parent['ppid']) is int and parent['ppid']>0 and parent['flags']['optimize']==0
            and len(parent_python)==parent['python']['bytes'] and parent_python[:4]==b'\x7fELF'
            and parent['python']['mode']&0o111 and len(parent_script)==parent['script']['bytes']
            and parent_script==bind(repo/PurePosixPath(script_relative).with_name('run.py'))
            and parent['script']['path']==paths['repository']+'/'+str(PurePosixPath(script_relative).with_name('run.py')),
            'actual supervising Python process and complete runner executable/source entities')
    supervisor_source=bind(repo/'tests/current_runtime_process.py',SUPERVISOR)
    require(intent['source_supervisor']==dict(path=paths['repository']+'/tests/current_runtime_process.py',
        bytes=len(supervisor_source),sha256=SUPERVISOR,mode=stat.S_IMODE((repo/'tests/current_runtime_process.py').stat().st_mode)),
        'exact unchanged process-tree supervisor implementation')
    process=json.loads(bind(execution/'process.json'))
    require(process==json.loads(bind(execution/'runner.json')) and process['argv']==argv
            and process['pid']==actual['pid'] and process['timeout_seconds']==intent['timeout_seconds']
            and process['started'] is True and process['exit_code']==0
            and process['stdout_EOF'] is True and process['stderr_EOF'] is True
            and process['forced_cleanup'] is False and not process.get('timed_out')
            and not process.get('error') and not process.get('execution_error')
            and process['cleanup_errors']==[] and process['running_descendants_after_cleanup']==[]
            and process['terminated_process_groups']==[],
            'same actual child exited normally with both complete EOF streams')
    for channel in ('stdout','stderr'):
        stream=bind(execution/('runner.'+channel),process[channel+'_sha256'])
        require(len(stream)==process[channel+'_bytes'],'complete actual reference '+channel)
        if channel=='stderr':require(stream==b'','clean actual reference process stderr')
    require(report['actual_process']==dict(pid=actual['pid'],argv=argv,python=actual['python']),
            'original actual numerical-reference receipt matches full process custody')
    def read_raw(relative,expected=None):
        if relative.startswith('pytorch76/'):
            path=capsule/relative.removeprefix('pytorch76/')
        elif relative=='inputs/runtime/python.bin':path=capsule/'reference-inputs/python.bin'
        elif relative.startswith('inputs/repository/'):
            label=relative.removeprefix('inputs/repository/')
            aliases={'build/pytorch-install.json':'pip-install.json',
                     'build/pytorch-reference-support-install.json':'numpy-pip-install.json'}
            path=capsule/'reference-inputs'/aliases[label] if label in aliases else repo/label
        else:raise ValueError('unknown fixed reference raw-input role: '+relative)
        return bind(path,expected)
    core_context=dict(local_observed_oracle=allow,paths=dict(repository=paths['repository']),script_relative=script_relative)
    core_report=dict(oracle_provenance=report['oracle_provenance'],allow_unverified_local=allow)
    result=read_reference(read_raw,core_report,core_context,actual)
    installed=read_wheel_entities(repo,capsule,bind,actual)
    receipt=json.loads(bind(capsule/'reference-inputs.json'))
    require(receipt['distribution']['root']==json.loads(bind(capsule/'installed-before.json'))['distribution_root'],
            'the original core entities and every complete installed member came from the same actual distribution')
    result.update(installed)
    receipt_raw=bind(capsule/'reference-inputs.json')
    require(report['reference_custody']==dict(path='reference-custody/reference-inputs.json',sha256=sha(receipt_raw),
        original_eleven_checks_recomputed_from_raw_inputs=True,
        full_installation_provenance_verified=not allow,production_oracle_qualified=not allow),
        'numerical report binds the same complete original raw reference custody receipt')
    result.update(actual_supervisor_intent_verified=True,actual_environment_transition_verified=True,
        original_reference_script_bytes_verified=True,portable_producer_path_roles=True,
        native_executed_by_reader=False,Torch_executed_by_reader=False)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Prepare one shared original fixed-wheel installation snapshot, without importing Torch.')
    parser.add_argument('--prepare-installed',action='store_true',required=True)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--wheel',type=Path)
    parser.add_argument('--snapshot-pool',type=Path)
    args=parser.parse_args()
    result=prepare_installed(args.repository,args.wheel,args.snapshot_pool)
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
