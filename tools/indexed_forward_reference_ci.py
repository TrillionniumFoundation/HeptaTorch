#!/usr/bin/env python3
"""Install one pinned reference environment, observe, and seal original bytes.

This entry point never builds or launches HIA. The unchanged process supervisor
runs the installer and the frozen reference producer. Sealing is a separate,
always-run operation so a failed attempt keeps its original evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys
import tarfile
import traceback
import types


PACKAGE = 'tests/indexed_forward_reference42/producer'
MANIFEST = '5ffaf7df1d843c97a7a23386e308bd79742bedec3b5a52941a4b8d25361cec0d'
SUPPORT = {
    'tests/current_runtime_process.py': '36e3432deac48085ba0586c85547cee8432d1533c5915238db2439b43dbca523',
    'tests/pytorch_oracle.py': 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae',
    'compatibility/pytorch-oracle-lock.json': 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944',
    'compatibility/upstream-lock.json': 'd204c5a8268a0d0f0a1dd162b5e0ab184beee83310bcad643b084deb691f3f49',
}
CI_SOURCE = ('tools/indexed_forward_reference_ci.py', '.github/workflows/indexed-forward-reference.yml')
WHEEL = '0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3'
TORCH_GIT = 'd0f2af8b66b20da4baa50644c0cd3f887a2eae2e'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def info(path):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'unique ordinary file: ' + str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(raw)
            digest.update(raw)
    after = path.lstat()
    require(all(getattr(before, key) == getattr(after, key) for key in
        ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
        'file changed while reading: ' + str(path))
    return dict(bytes=size, sha256=digest.hexdigest(), mode=stat.S_IMODE(after.st_mode))


def bind(rows, path, expected=None):
    path = Path(path).absolute()
    row = info(path)
    require(expected is None or row == expected, 'input identity differs: ' + str(path))
    require(str(path) not in rows or rows[str(path)] == row, 'previous input changed: ' + str(path))
    rows[str(path)] = row
    return row


def source_inputs(repo):
    rows = {}
    raw = (repo / PACKAGE / 'manifest.json').read_bytes()
    require(sha(raw) == MANIFEST, 'frozen indexed42 producer manifest')
    manifest = parse(raw)
    require(manifest['expected_observations'] == 42 and manifest['expected_outputs_included'] is False,
            'reference contains literal inputs and no expected outputs')
    root = repo / PACKAGE
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'manifest.json'}, 'complete 42-entity producer')
    require(all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in root.rglob('*')),
            'ordinary producer tree without linked directories')
    require(len(manifest['files']) == 41 and sum(n.startswith('inputs/') for n in manifest['files']) == 34,
            'fixed source and literal blob census')
    for name, expected in manifest['files'].items():
        member = PurePosixPath(name)
        require(not member.is_absolute() and '..' not in member.parts and str(member) == name,
                'canonical producer member')
        bind(rows, root / name, expected)
    bind(rows, root / 'manifest.json')
    for name, expected in SUPPORT.items():
        require(bind(rows, repo / name)['sha256'] == expected, 'fixed support source: ' + name)
    for name in CI_SOURCE:
        bind(rows, repo / name)
    return rows


def complete_process(receipt):
    require(receipt['started'] is True and receipt['exit_code'] == 0
            and receipt['stdout_EOF'] is True and receipt['stderr_EOF'] is True
            and receipt['forced_cleanup'] is False and not receipt['cleanup_errors']
            and not receipt['running_descendants_after_cleanup'], 'successful process and both physical EOFs')


def run(args, repo, work):
    require(not work.exists() and not work.is_symlink(), 'fresh private reference workspace')
    work.mkdir(parents=False)
    evidence = work / 'evidence'
    evidence.mkdir()
    inputs = {}
    result = dict(schema='heptatorch.indexed-forward.reference-ci.v1', status='FAIL',
        source_commit=args.source_commit, producer_manifest_sha256=MANIFEST,
        expected_observations=42, native_executed=False, compiler_executed=False,
        runtime_qualification_transferred=False, full_project_status='HOLD', processes={})
    code = 1
    try:
        require(not sys.flags.optimize and sys.implementation.name == 'cpython'
                and sys.version_info[:2] == (3, 12) and sys.platform == 'linux'
                and platform.machine() == 'x86_64', 'actual CPython 3.12 Linux x86_64')
        inputs.update(source_inputs(repo))
        python = Path(sys.executable).resolve(strict=True)
        interpreter = dict(path=str(python), **bind(inputs, python))
        require(Path('/proc/self/exe').resolve() == python, 'actual orchestrator executable')
        result['orchestrator_python'] = interpreter
        shutil.copy2(python, evidence / 'orchestrator-python.bin')
        bind(inputs, evidence / 'orchestrator-python.bin', inputs[str(python)])
        raw = (repo / 'tests/current_runtime_process.py').read_bytes()
        supervisor = types.ModuleType('original_reference_supervisor')
        supervisor.__file__ = str(repo / 'tests/current_runtime_process.py')
        exec(compile(raw, supervisor.__file__, 'exec'), supervisor.__dict__)
        write_json(evidence / 'inputs-before.json', dict(inputs=inputs, interpreter=interpreter,
            source_commit=args.source_commit, producer_manifest_sha256=MANIFEST))

        def execute(label, command, timeout):
            directory = evidence / label
            directory.mkdir()
            executable = Path(command[0]).resolve(strict=True)
            executable_info = info(executable)
            captured = directory / 'executable.bin'
            shutil.copy2(executable, captured)
            bind(inputs, captured, executable_info)
            require(info(executable) == executable_info, 'executable unchanged during ordinary snapshot')
            write_json(directory / 'intent.json', dict(argv=command, cwd=str(repo),
                executable=dict(path=str(executable), **executable_info),
                captured_executable=dict(path=str(captured), **executable_info)))
            receipt = supervisor.execute(command, directory, timeout, cwd=repo, mirror_stderr=True)
            result['processes'][label] = receipt
            complete_process(receipt)
            return directory

        git = shutil.which('git')
        require(git is not None, 'checkout identity tool available')
        checkout = execute('checkout-identity', [git, 'rev-parse', 'HEAD', 'HEAD^{tree}'], 30)
        commit, tree = (checkout / 'runner.stdout').read_text().splitlines()
        require(commit == args.source_commit and re.fullmatch('[0-9a-f]{40}', tree),
                'actual checkout equals the GitHub execution SHA')
        result['checkout'] = dict(commit=commit, tree=tree)
        venv = work / 'environment'
        execute('create-environment', [str(python), '-I', '-B', '-m', 'venv', str(venv)], 120)
        selected = venv / 'bin/python'
        require(selected.resolve(strict=True) == python, 'fresh venv uses the prebound actual Python')
        bind(inputs, venv / 'pyvenv.cfg')
        expected = parse((repo / 'compatibility/pytorch-oracle-lock.json').read_bytes())['oracle']
        require(expected['version'] == '2.15.0+cpu' and expected['git_commit'] == TORCH_GIT
                and expected['wheel']['sha256'] == WHEEL
                and expected['install_report'] == 'build/pytorch-install.json', 'exact independent wheel lock')
        install_report = repo / expected['install_report']
        require(not install_report.exists() and not install_report.is_symlink(), 'no historical installation report reused')
        install_report.parent.mkdir(parents=True, exist_ok=True)
        execute('install-wheel', [str(selected), '-I', '-B', '-m', 'pip', 'install',
            '--disable-pip-version-check', '--force-reinstall', '--index-url',
            'https://download.pytorch.org/whl/cpu', '--report', str(install_report),
            expected['wheel']['url'] + '#sha256=' + WHEEL], 900)
        bind(inputs, install_report)
        command = [str(python), '-I', '-B', str(repo / PACKAGE / 'produce.py'),
            '--python', str(selected), '--repository', str(repo),
            '--expected-producer-sha256', MANIFEST, '--output', str(evidence / 'production')]
        execute('producer-execution', command, 1260)
        production = parse((evidence / 'production/production.json').read_bytes())
        require(production['status'] == 'OBSERVED_COMPLETE_INDEXED_FORWARD_REFERENCE_EXECUTION'
                and production['cases'] == 42 and production['all_physical_inputs_unchanged'] is True
                and production['producer_manifest_sha256'] == MANIFEST,
                'complete original producer observation')
        provenance = production['oracle_provenance']
        require(provenance['status'] == 'PASS' and len(provenance['checks']) == 11
                and all(value is True for value in provenance['checks'].values()), 'all original eleven provenance predicates')
        for path, row in list(inputs.items()):
            bind(inputs, path, row)
        result.update(status='OBSERVED_INDEXED_FORWARD42_REFERENCE_ONLY', cases=42,
            accepted=production['accepted'], observed_errors=production['observed_errors'],
            report=production['report'], oracle_provenance=provenance, all_bound_inputs_unchanged=True)
        code = 0
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    result['inputs'] = inputs
    write_json(evidence / 'ci.json', result)
    print(json.dumps({key:result.get(key) for key in ('status','cases','accepted','observed_errors','error')}, allow_nan=False))
    return code


def seal(args, repo, work):
    prefix = args.output_prefix.absolute()
    require(prefix.parent == work.parent and not prefix.is_symlink(), 'separate private artifact prefix')
    selected, declared, problems, mapping = {}, [], [], {}
    evidence, venv = work / 'evidence', work / 'environment'
    allowed_python = Path(sys.executable).resolve(strict=True)

    def add(path, member, expected=None):
        path = Path(path).absolute()
        safe = PurePosixPath(member)
        require(safe.parts and not safe.is_absolute() and '..' not in safe.parts and str(safe) == member,
                'safe archive member')
        row = info(path)
        require(expected is None or row == expected, 'original declared bytes and modes: ' + str(path))
        if member in selected:
            require(selected[member] == dict(path=str(path), **row), 'unique member source')
        selected[member] = dict(path=str(path), **row)
        mapping[str(path)] = member

    def mapped_member(path):
        if path == allowed_python:
            return 'interpreters/actual-orchestrator-python.bin'
        if path.is_relative_to(evidence):
            return 'evidence/' + path.relative_to(evidence).as_posix()
        if path == venv / 'pyvenv.cfg':
            return 'environment/pyvenv.cfg'
        if path.is_relative_to(repo / PACKAGE) or path in {repo / p for p in (*SUPPORT, *CI_SOURCE, 'build/pytorch-install.json')}:
            return 'repository/' + path.relative_to(repo).as_posix()
        site = venv / 'lib/python3.12/site-packages'
        allowed_torch = ('torch/__init__.py', 'torch/_C.cpython-312-x86_64-linux-gnu.so',
                        'torch-2.15.0+cpu.dist-info/METADATA', 'torch-2.15.0+cpu.dist-info/direct_url.json')
        if path in {site / name for name in allowed_torch}:
            return 'environment/' + path.relative_to(venv).as_posix()
        raise ValueError('unrecognized declared input; no broad external-path exception: ' + str(path))

    def attempt(operation, label):
        try:
            operation()
        except BaseException as error:
            problems.append(dict(label=label, type=type(error).__name__, message=str(error)))

    for root in (repo / PACKAGE, evidence):
        if root.is_dir() and not root.is_symlink():
            for path in sorted(root.rglob('*')):
                if not path.is_dir() or path.is_symlink():
                    attempt(lambda p=path: add(p, mapped_member(p)), str(path))
        else:
            problems.append(dict(label=str(root), type='MissingDirectory', message='required original directory unavailable'))
    for path in [repo / p for p in (*SUPPORT, *CI_SOURCE, 'build/pytorch-install.json')] + [allowed_python, venv / 'pyvenv.cfg']:
        attempt(lambda p=path: add(p, mapped_member(p)), str(path))
    for suffix in ('run.stdout', 'run.stderr'):
        path = Path(str(prefix) + '.' + suffix)
        attempt(lambda p=path, n=suffix: add(p, 'workflow-streams/' + n), str(path))
    reports = ('inputs-before.json', 'ci.json', 'production/execution/inputs-before.json',
               'production/production.json', 'production/reference/report.json')
    parsed = {}
    for relative in reports:
        path = evidence / relative
        try:
            value = parse(path.read_bytes())
            parsed[relative] = value
            table = value['inputs']
            require(isinstance(table, dict) and table, 'nonempty original input table: ' + relative)
            for spelling, row in table.items():
                require(Path(spelling).is_absolute() and set(row) == {'bytes','sha256','mode'}, 'exact original input row')
                require(type(row['bytes']) is int and row['bytes'] >= 0 and type(row['mode']) is int
                        and 0 <= row['mode'] <= 0o7777 and isinstance(row['sha256'], str)
                        and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'typed original input identity')
                declared.append(dict(report='evidence/' + relative, source_path=spelling, expected=row))
                attempt(lambda p=Path(spelling), r=row: add(p, mapped_member(p), r), spelling)
        except BaseException as error:
            problems.append(dict(label=relative, type=type(error).__name__, message=str(error)))
    attempt(lambda: source_inputs(repo), 'frozen-source-recheck')
    ci = parsed.get('ci.json', {})
    production = parsed.get('production/production.json', {})
    reference = parsed.get('production/reference/report.json', {})
    observed = (ci.get('status') == 'OBSERVED_INDEXED_FORWARD42_REFERENCE_ONLY'
        and ci.get('source_commit') == args.source_commit and ci.get('cases') == 42
        and production.get('status') == 'OBSERVED_COMPLETE_INDEXED_FORWARD_REFERENCE_EXECUTION'
        and production.get('producer_manifest_sha256') == MANIFEST
        and reference.get('status') == 'OBSERVED_INDEXED_FORWARD_REFERENCE' and reference.get('cases') == 42)
    archive = Path(str(prefix) + '.tar.xz')
    require(not archive.exists(), 'fresh lossless archive')
    with tarfile.open(archive, 'x:xz', format=tarfile.PAX_FORMAT, preset=6) as tar:
        for member, row in sorted(selected.items()):
            path = Path(row['path'])
            expected = {key:row[key] for key in ('bytes','sha256','mode')}
            require(info(path) == expected, 'selected input unchanged before archiving')
            header = tarfile.TarInfo(member)
            header.size, header.mode, header.mtime = row['bytes'], row['mode'], 0
            with path.open('rb') as stream:
                tar.addfile(header, stream)
            require(info(path) == expected, 'selected input unchanged after archiving')
    with tarfile.open(archive, 'r:xz') as tar:
        found = set()
        for member in tar:
            require(member.isfile() and member.name not in found and member.name in selected, 'ordinary unique archive member')
            found.add(member.name)
            row = selected[member.name]
            raw = tar.extractfile(member).read()
            require((len(raw), sha(raw), member.mode) == (row['bytes'], row['sha256'], row['mode']),
                    'complete sealed member bytes and original mode')
        require(found == set(selected), 'complete archive census')
    manifest_path = Path(str(prefix) + '.manifest.json')
    write_json(manifest_path, dict(schema='heptatorch.indexed-forward.reference-custody-manifest.v1',
        source_commit=args.source_commit, producer_manifest_sha256=MANIFEST,
        archive=dict(name=archive.name, **info(archive)), files=selected,
        member_count=len(selected), payload_bytes=sum(row['bytes'] for row in selected.values()),
        declared_input_bindings=declared, original_input_path_mapping=mapping,
        selected_original_bytes_and_modes_preserved=True, tar_uid_gid_mtime_normalized=True))
    custody = dict(schema='heptatorch.indexed-forward.reference-custody.v1',
        status='SEALED_OBSERVED_INDEXED_FORWARD42_REFERENCE' if observed and not problems else 'SEALED_FAILED_INDEXED_FORWARD_ATTEMPT',
        source_commit=args.source_commit, producer_manifest_sha256=MANIFEST,
        archive=dict(path=str(archive), **info(archive)), manifest=dict(path=str(manifest_path), **info(manifest_path)),
        original_reference_observed=observed, missing_or_changed=problems,
        declared_inputs=len(declared), unique_declared_inputs=len({row['source_path'] for row in declared}),
        native_executed_by_sealer=False, Torch_executed_by_sealer=False,
        reference_or_runtime_results_rewritten=False, runtime_qualification_transferred=False,
        actual_python_role=dict(original_path=str(allowed_python), member=mapping.get(str(allowed_python))),
        full_project_status='HOLD')
    custody_path = Path(str(prefix) + '.custody.json')
    write_json(custody_path, custody)
    with Path(str(prefix) + '.sha256').open('x') as stream:
        for path in (archive, manifest_path, custody_path):
            stream.write(info(path)['sha256'] + '  ' + path.name + '\n')
    print(json.dumps({key:custody[key] for key in ('status','declared_inputs','unique_declared_inputs','missing_or_changed')}))
    return 0 if observed and not problems else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('run','seal'))
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-prefix', type=Path)
    args = parser.parse_args()
    require(re.fullmatch('[0-9a-f]{40}', args.source_commit), 'full Git execution SHA')
    repo, work = args.repository.resolve(strict=True), args.workspace.absolute()
    require(not sys.flags.optimize and work.name == 'heptatorch-indexed-forward42'
            and work.parent == Path(os.environ['RUNNER_TEMP']).resolve(strict=True), 'owned fresh runner-temp namespace')
    if args.operation == 'run':
        return run(args, repo, work)
    require(args.output_prefix is not None, 'explicit custody artifact prefix')
    return seal(args, repo, work)


if __name__ == '__main__':
    raise SystemExit(main())
