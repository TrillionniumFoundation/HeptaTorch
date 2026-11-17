#!/usr/bin/env python3
"""Admit the fixed fresh 07b build and execute one backward private-cache resource control."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, stat, subprocess, sys, traceback

HERE = Path(__file__).resolve().parent
SOURCE = 'c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a'
IMAGE = '0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74'
BASIS_BUILD = 'b31a45a850f06196432b2786ad0a39b56088ccf6c7b100d0da3517a44de5ba48'
PARENT_WRAPPER = '00becd65f6897286e82723e0e4042eca6c0dbce18e2719855f57af0262aaca53'
PARENT_MANIFEST = 'f38c0cc3c2105e6180798a1fbccbe7fd84426bea18ef95ef14306cd8f63ce98d'
REFERENCE = 'f3ecc9dd7db2ce8fbd0e424557ad5b95ac2e899a1c09cf38dad47e4fecc2b0c6'
PROBE = 'c0ddafbae6ea81bd98f6066bc867f71e0b6e8d4f73e6c5d7f15f7fe3d29569a3'
CASE = 'backward_private_preserve_allocation_oom_recovery'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def file(path, pin=None):
    path = Path(path)
    st = path.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_nlink == 1, ('regular unique file', str(path))
    raw = path.read_bytes()
    meta = dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(st.st_mode))
    assert pin is None or meta['sha256'] == pin, (str(path), meta['sha256'], pin)
    return raw, meta

def dump(path, obj):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())

def exact(actual, expected, label):
    assert type(actual) is type(expected) and actual == expected, (label, actual, expected)

def resolved(repo, value):
    return (value if value.is_absolute() else repo / value).resolve()

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

def source_check(repo):
    return module(HERE / 'validate_package.py', 'backward_private_cache_package_check').verify_package(repo)

def main():
    if sys.flags.optimize:
        raise RuntimeError('assertions must be enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'builder-output', 'reference', 'probe', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args()
    assert len(a.expected_build_sha256) == 64 and all(c in '0123456789abcdef' for c in a.expected_build_sha256)
    repo = a.repo.resolve()
    assert HERE == repo / 'tests/backward_private_cache_resource_review'
    parent_path = repo / 'tests/wrapped_scalar_resource_review'
    builder_output = resolved(repo, a.builder_output)
    reference, probe = resolved(repo, a.reference), resolved(repo, a.probe)
    out = resolved(repo, a.output)
    assert out != builder_output and not out.is_relative_to(builder_output)
    assert not out.is_relative_to(HERE) and not out.is_relative_to(parent_path)
    assert not out.exists(), 'fresh wrapper output required'
    out.mkdir(parents=True, exist_ok=False)
    observed = out / 'observed'
    inputs = {}
    report = dict(
        schema='heptatorch.scalar07b.portable-backward-private-cache-resource.v1',
        status='RUNNING', prepare_only=a.prepare_only,
        native_executed=False, Torch_executed_here=False,
        compiler_executed=False, emitter_executed=False,
        source_sha256=SOURCE, native_sha256=IMAGE,
        original_build_sha256=BASIS_BUILD, expected_build_sha256=a.expected_build_sha256,
        whole_pytorch_gate='HOLD', closed_schema_count=0,
        original_resource3_reexecuted=False, numeric_oracle_computed_here=False,
        scope='One backward private preserve allocation refusal and identical-command resource recovery.',
        child_started=False, child_completed=False)
    code = 1
    try:
        report['source_check'] = source_check(repo)
        for rel, meta in report['source_check']['bound'].items():
            inputs[str(HERE / rel)] = meta
        inputs[str(HERE / 'package-manifest.json')] = file(HERE / 'package-manifest.json')[1]
        parent_file = parent_path / 'run_resources.py'
        file(parent_file, PARENT_WRAPPER)
        file(parent_path / 'package-manifest.json', PARENT_MANIFEST)
        parent = module(parent_file, 'frozen_wrapped_scalar_resource_wrapper')
        report['parent_source_check'] = parent.source_check()
        for rel, meta in report['parent_source_check']['bound'].items():
            inputs[str(parent_path / rel)] = meta
        inputs[str(parent_path / 'package-manifest.json')] = file(
            parent_path / 'package-manifest.json', PARENT_MANIFEST)[1]
        # These are the complete, byte-pinned parent's existing functions.
        # Its main() and all three original resource recipes are never invoked.
        native, admission = parent.fresh_build(repo, builder_output, a.expected_build_sha256, inputs)
        report['fresh_build_admission'] = admission
        for path, pin in ((reference, REFERENCE), (probe, PROBE)):
            inputs[str(path)] = file(path, pin)[1]
        dump(out / 'inputs-before.json', inputs)
        command = [
            sys.executable, '-S', '-B', str(HERE / 'driver.py'),
            '--resource-source', str(parent_path / 'portable07b'),
            '--native-build', str(native),
            '--binary', str(native / 'tensor-shape-scalar-session'),
            '--protocol', str(parent_path / 'portable07b/protocol.py'),
            '--reference', str(reference), '--probe', str(probe),
            '--obligations', str(HERE / 'obligations.json'),
            '--expected-native-sha256', IMAGE,
            '--expected-build-sha256', a.expected_build_sha256,
            '--output', str(observed)]
        if a.prepare_only:
            command.append('--prepare-only')
        dump(out / 'command.json', {
            'argv': command, 'cwd': str(repo), 'fresh_child_output': str(observed),
            'case_selection': CASE, 'original_three_recipes_selected': False})
        assert not observed.exists()
        report['child_started'] = None
        report['native_executed'] = False if a.prepare_only else None
        try:
            child = subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL,
                                   capture_output=True, timeout=1800)
        except subprocess.TimeoutExpired as error:
            report['child_started'] = True
            (out / 'driver.stdout').write_bytes(error.stdout or b'')
            (out / 'driver.stderr').write_bytes(error.stderr or b'')
            report['child_timed_out'] = True
            raise
        report['child_started'] = True
        report['child_completed'] = True
        (out / 'driver.stdout').write_bytes(child.stdout)
        (out / 'driver.stderr').write_bytes(child.stderr)
        report['child'] = dict(
            exit_code=child.returncode, stdout_EOF=True, stderr_EOF=True,
            stdout_bytes=len(child.stdout), stdout_sha256=sha(child.stdout),
            stderr_bytes=len(child.stderr), stderr_sha256=sha(child.stderr))
        raw, meta = file(observed / 'report.json')
        result = json.loads(raw)
        report['child_report'] = meta
        report['native_executed'] = result['native_executed']
        assert child.returncode == 0 and child.stderr == b''
        for key, expected in {
            'schema': 'heptatorch.scalar07b.backward-private-preserve-resource.v1',
            'source_sha256': SOURCE, 'native_sha256': IMAGE,
            'original_native_build_sha256': BASIS_BUILD,
            'native_build_sha256': a.expected_build_sha256,
            'unit_count': 13, 'function_count': 658, 'all_inputs_unchanged': True,
            'Torch_executed': False, 'compiler_executed': False,
            'original_resource3_reexecuted': False, 'numeric_oracle_computed_here': False,
            'whole_pytorch_gate': 'HOLD', 'closed_schema_count': 0}.items():
            exact(result[key], expected, 'actual child ' + key)
        if a.prepare_only:
            exact(result['status'], 'PREPARED_NOT_RUN', 'child preparation status')
            exact(result['native_executed'], False, 'no resource process during preparation')
            exact(result['cases'], [], 'no prepared case execution')
            exact(result['case_count'], 0, 'zero prepared cases')
            exact(result['passed'], 0, 'zero prepared passes')
            report['status'] = 'PREPARED_NOT_RUN'
        else:
            exact(result['status'], 'PASS', 'actual resource status')
            exact(result['native_executed'], True, 'actual resource process')
            exact(result['case_count'], 1, 'one resource case')
            exact(result['passed'], 1, 'one actual resource PASS')
            exact([row['name'] for row in result['cases']], [CASE], 'exact resource recipe')
            row = result['cases'][0]
            for key, expected in {
                'status': 'PASS', 'EOF_exit_code': 0,
                'target_allocation_qualified': True,
                'original_limit_restored': True}.items():
                exact(row[key], expected, 'actual case ' + key)
            exact(row['allocation_setup']['target_function'], 'incoming_reserve_preserve_storage',
                  'source-bound target allocation')
            exact(row['allocation_setup']['target_source_sha256'],
                  json.loads(file(HERE / 'obligations.json')[0])['functions']['incoming_reserve_preserve_storage'],
                  'exact target HIA body')
            child_case = json.loads(file(observed / CASE / 'report.json')[0])
            exact(child_case, row, 'complete per-case report')
            report['status'] = 'PASS'
        report['cases'] = result['cases']
        fields = {
            'frames_before_EOF': 'frames_before_EOF', 'checks': 'checks',
            'full_owner_rollback_bytes': 'rollback_bytes',
            'full_rng_bytes': 'full_rng_bytes',
            'committed_84Q_snapshots': 'committed_84Q_snapshots'}
        for total, child_field in fields.items():
            expected = sum(row.get(child_field, 0) for row in result['cases'])
            exact(result[total], expected, 'complete child aggregate ' + total)
            report[total] = expected
        code = 0
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    try:
        for path, meta in inputs.items():
            exact(file(path)[1], meta, 'all outer bound inputs unchanged')
        dump(out / 'inputs-after.json', inputs)
        report['all_inputs_unchanged'] = True
    except BaseException as error:
        report.update(status='FAIL', custody_error=repr(error), custody_traceback=traceback.format_exc())
        code = 1
    report['bound_inputs'] = inputs
    dump(out / 'report.json', report)
    print(json.dumps({k: report.get(k) for k in ('status', 'native_executed', 'error', 'custody_error')}))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
