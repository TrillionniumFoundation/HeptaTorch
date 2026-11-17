#!/usr/bin/env python3
"""Admit a fresh fixed 07b portable build, then invoke the three unchanged resource recipes."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, stat, struct, subprocess, sys, traceback

HERE = Path(__file__).resolve().parent
SOURCE = 'c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a'
IMAGE = '0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74'
BASIS_BUILD = 'b31a45a850f06196432b2786ad0a39b56088ccf6c7b100d0da3517a44de5ba48'
SHAPE_BUILD = '6f96f137a82ae128efcd3e143784ea7f5df438b50bbfe71adc8ec321f8dedf54'
BUILDER = '9f6f02a7d749206fb9b523d0bf7ddb308d695a3215fecd4982c5d11a74c5abf8'
REFERENCE = 'f3ecc9dd7db2ce8fbd0e424557ad5b95ac2e899a1c09cf38dad47e4fecc2b0c6'
PROBE = 'c0ddafbae6ea81bd98f6066bc867f71e0b6e8d4f73e6c5d7f15f7fe3d29569a3'
CASES = ('entry_guards_and_relative_tokens', 'saved_scalar_and_temporary_lifetime',
         'result_allocation_oom_recovery')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def dump(path, obj):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())

def file(path, pin=None):
    path = Path(path)
    st = path.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_nlink == 1, ('regular unique file', str(path))
    raw = path.read_bytes()
    meta = dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(st.st_mode))
    assert pin is None or meta['sha256'] == pin, (str(path), meta['sha256'], pin)
    return raw, meta

def resolved(repo, value):
    return (value if value.is_absolute() else repo / value).resolve()

def source_check():
    spec = importlib.util.spec_from_file_location('wrapped_resource_package_check',
                                                  HERE / 'validate_package.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_package()

def exact(actual, expected, label):
    assert type(actual) is type(expected) and actual == expected, (label, actual, expected)

def fresh_build(repo, output, expected_build, inputs):
    """Read current fixed products and the actual builder's full custody chain."""
    def bind(path, pin=None):
        path = Path(path).resolve()
        raw, meta = file(path, pin)
        key = str(path)
        if key in inputs:
            exact(meta, inputs[key], 'already bound input')
        inputs[key] = meta
        return raw, meta
    native = output / 'native'
    basis_path = HERE / 'provenance/original-build.json'
    basis = json.loads(bind(basis_path, BASIS_BUILD)[0])
    shape_basis = json.loads(bind(HERE / 'provenance/shape-original-build.json', SHAPE_BUILD)[0])
    builder = repo / 'tools/differentiable_view_r4_build_review/build.py'
    bind(builder, BUILDER)
    original_report_path = repo / 'tools/differentiable_view_r4_build_review/scalar07b-original-build.json'
    bind(original_report_path, BASIS_BUILD)
    bind(HERE / 'provenance/portable-builder-source.py', BUILDER)
    assert basis['status'] == 'PASS' and basis['link_exit'] == 0
    assert basis['source_sha256'] == SOURCE and basis['native_sha256'] == IMAGE
    assert basis['functions'] == 658 and len(basis['units']) == 13
    assert len({u['name'] for u in basis['units']}) == 13
    success = json.loads(bind(output / 'success.json')[0])
    report_raw, report_meta = bind(output / 'report.json')
    report = json.loads(report_raw)
    exact(success, {'status': 'PASS', 'report_sha256': report_meta['sha256'],
                    'source_sha256': SOURCE, 'expected_native_sha256': IMAGE,
                    'stage': 'scalar07b', 'native_business_executed': False,
                    'Torch_executed': False}, 'actual success receipt')
    required_report = {
        'schema': 'heptatorch.r4.portable-build-admission.v1', 'stage': 'scalar07b',
        'status': 'PASS', 'prepare_only': False, 'compiler_executed': True,
        'native_business_executed': False, 'Torch_executed': False,
        'runtime_qualified': False, 'closed_schema_count': 0,
        'full_pytorch_rewrite_complete': False, 'whole_pytorch_gate': 'HOLD',
        'child_process_attempted': True, 'child_process_started': True,
        'child_process_completed': True, 'all_bound_inputs_unchanged': True,
        'compiler_processes': 13, 'linker_processes': 1,
        'objects_and_link_image_reused': False, 'source_sha256': SOURCE,
        'expected_native_sha256': IMAGE, 'functions': 658, 'blocks': 8661,
        'instructions': 30980, 'native_unit_count': 13, 'descriptor_bytes': 672,
        'original_build_report_sha256': BASIS_BUILD}
    for key, value in required_report.items():
        exact(report[key], value, 'builder report ' + key)
    assert not any(k in report for k in ('error', 'traceback', 'driver_timed_out'))
    built_raw, built_meta = bind(native / 'build.json', expected_build)
    built = json.loads(built_raw)
    exact(report['new_build_report'], built_meta, 'current build receipt metadata')
    assert set(built) == set(basis), 'complete build field denominator'
    for key in basis:
        if key != 'inputs_sha256':
            exact(built[key], basis[key], 'fixed original build field ' + key)

    command = json.loads(bind(output / 'command.json')[0])
    argv = command['argv']
    assert len(argv) == 19 and argv[1:4] == ['-E', '-S', '-B']
    assert Path(argv[0]).is_absolute() and Path(argv[0]).name.startswith('python')
    package = repo / 'tools/differentiable_view_scalar_r4'
    exact(argv[4], str(package / 'build.py'), 'fixed authoring entry')
    pairs = list(zip(argv[5::2], argv[6::2]))
    exact([key for key, value in pairs],
          ['--previous', '--repository', '--upstream', '--compactor',
           '--compiler', '--linker', '--output'], 'complete builder argument names')
    options = dict(pairs)
    previous, upstream = Path(options['--previous']), Path(options['--upstream'])
    compiler, linker = Path(options['--compiler']), Path(options['--linker'])
    for path in (previous, upstream, compiler, linker):
        assert path.is_absolute() and path.resolve() == path
    exact(options['--repository'], str(repo), 'builder repository')
    exact(options['--compactor'], str(repo / 'tools/owned_optimizers/compact_functions.py'), 'compactor path')
    exact(options['--output'], str(native), 'fresh native output')
    exact(command, {'argv': argv, 'cwd': str(repo), 'stdin': 'DEVNULL',
                    'fresh_output_required': True}, 'complete builder command record')

    before = json.loads(bind(output / 'inputs-before.json')[0])
    after = json.loads(bind(output / 'inputs-after.json')[0])
    assert before and before == after
    exact(report['bound_input_count'], len(before), 'builder input count')
    required = {}
    def require(path, pin):
        path = Path(path).resolve()
        raw, meta = bind(path, pin)
        key = str(path)
        if key in required:
            exact(required[key], meta, 'duplicate builder input')
        required[key] = meta
        return raw
    require(builder, BUILDER)
    require(original_report_path, BASIS_BUILD)
    require(compiler, basis['compiler_sha256'])
    require(linker, basis['linker_sha256'])
    for name, pin in basis['authoring_inputs_sha256'].items():
        assert Path(name).name == name
        require(package / name, pin)
    parent_raw = require(previous / 'build.json', None)
    parent = json.loads(parent_raw)
    assert set(parent) == set(shape_basis)
    for key in shape_basis:
        if key != 'inputs_sha256':
            exact(parent[key], shape_basis[key], 'fixed shape parent field ' + key)
    assert parent['status'] == 'PASS' and parent['functions'] == 638
    require(previous / 'runtime.hepta', basis['parent_source_sha256'])
    require(previous / 'tensor-shape-view-session', basis['parent_native_sha256'])
    for unit in shape_basis['units']:
        assert Path(unit['name']).name == unit['name'] and unit['compiler_exit'] == 0
        require(previous / (unit['name'] + '.hepta'), unit['source_sha256'])
        require(previous / (unit['name'] + '.o'), unit['object_sha256'])
    old_parents = [Path(path).parent for path, pin in basis['inputs_sha256'].items()
                   if Path(path).name == 'runtime.hepta' and pin == basis['parent_source_sha256']]
    assert len(old_parents) == 1
    old_parent = old_parents[0]
    old_repo = Path('/home/qian-qi/heptatorch-continuation-20261116/runs/to-view-recovery-integration-01/repo')
    old_upstream = Path('/home/qian-qi/heptatorch-continuation-20261116/toolchain')
    relocated, mapped = [], {}
    for old_text, pin in basis['inputs_sha256'].items():
        old = Path(old_text)
        dynamic_parent = False
        if old.is_relative_to(old_parent):
            new = previous / old.relative_to(old_parent)
            dynamic_parent = old == old_parent / 'build.json'
        elif old.is_relative_to(old_repo):
            new = repo / old.relative_to(old_repo)
        elif old.is_relative_to(old_upstream):
            new = upstream / old.relative_to(old_upstream)
        else:
            raise AssertionError(('undeclared original path', old_text))
        require(new, None if dynamic_parent else pin)
        current = required[str(new.resolve())]['sha256']
        mapped[str(new)] = current
        relocated.append({'original_path': old_text, 'actual_path': str(new),
                          'original_sha256': pin, 'actual_sha256': current,
                          'path_dependent_parent_report': dynamic_parent})
    exact(json.loads(bind(output / 'input-relocations.json')[0]), relocated,
          'complete declared relocation map')
    exact(built['inputs_sha256'], mapped, 'complete current builder input hash map')
    for unit in basis['units']:
        require(package / 'generated/units' / (unit['name'] + '.hepta'), unit['source_sha256'])
    require(package / 'generated/runtime.hepta', SOURCE)
    exact(before, required, 'exact before/after builder custody set, bytes, mode and hash')

    exact(bind(native / 'runtime.hepta', SOURCE)[0],
          bind(package / 'generated/runtime.hepta', SOURCE)[0], 'complete emitted HIA')
    unit_rows, objects = [], []
    for unit in basis['units']:
        name = unit['name']
        assert Path(name).name == name
        source, sm = bind(native / (name + '.hepta'), unit['source_sha256'])
        exact(source, bind(package / 'generated/units' / (name + '.hepta'),
                           unit['source_sha256'])[0], 'fixed published unit ' + name)
        obj, om = bind(native / (name + '.o'), unit['object_sha256'])
        err, em = bind(native / (name + '.stderr'))
        assert obj.startswith(b'\x7fELF') and err == b''
        objects.append(obj)
        unit_rows.append({'name': name, 'source': sm, 'source_byte_equal_to_published': True,
                          'object': om, 'compiler_stderr': em, 'compiler_exit': 0})
    exact(report['units'], unit_rows, 'complete ordered fresh unit results')
    app, im = bind(native / 'tensor-shape-scalar-session', IMAGE)
    assert app.startswith(b'\x7fELF') and im['mode'] == 0o500 and im['bytes'] == basis['native_bytes']
    exact(report['native_image'], im, 'actual ELF metadata')
    link, lm = bind(native / 'link.request.bin')
    expected_link = b'HIASTKS1' + struct.pack('<3Q', 13, 4, 0) + b'step'
    expected_link += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    exact(link, expected_link, 'actual ABI3 ordered 13 object link input')
    err, em = bind(native / 'link.stderr')
    assert not err
    exact(report['link_request'], lm, 'link input metadata')
    exact(report['link_stderr'], em, 'link stderr metadata')
    stdout, out_meta = bind(output / 'driver.stdout')
    stderr, err_meta = bind(output / 'driver.stderr')
    assert not stderr
    child = report['child']
    for key, value in {'exit_code': 0, 'stdout_bytes': len(stdout), 'stdout_sha256': sha(stdout),
                       'stderr_bytes': 0, 'stderr_sha256': sha(b''),
                       'stdout_EOF': True, 'stderr_EOF': True}.items():
        exact(child[key], value, 'builder transport ' + key)
    assert isinstance(child['elapsed_seconds'], (int, float)) and child['elapsed_seconds'] >= 0
    printed = json.loads(stdout)
    assert printed
    for key, value in printed.items():
        assert key in built
        exact(value, built[key], 'complete child printed field ' + key)
    inventory_raw, inv_meta = bind(output / 'output-inventory.json')
    inventory = json.loads(inventory_raw)
    exact(report['output_inventory'], inv_meta, 'actual output inventory')
    actual = set()
    for path in output.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():
            rel = path.relative_to(output).as_posix()
            if rel not in ('report.json', 'output-inventory.json', 'success.json'):
                actual.add(rel)
    assert actual == set(inventory) and actual
    for rel, meta in inventory.items():
        p = Path(rel)
        assert not p.is_absolute() and '..' not in p.parts and p.as_posix() == rel
        exact(bind(output / rel)[1], meta, 'complete builder output member ' + rel)
    return native, dict(status='PASS_FRESH_BUILD_CUSTODY_ONLY',
                        original_build_sha256=BASIS_BUILD,
                        fresh_build_sha256=built_meta['sha256'],
                        fresh_builder_report=report_meta,
                        fixed_fields='all original build fields except the declared inputs_sha256 relocation',
                        fixed_unit_count=13, compiler_processes=13, linker_processes=1,
                        required_before_inputs=len(required), validated_builder_outputs=len(inventory),
                        source_sha256=SOURCE, native_sha256=IMAGE)

def main():
    if sys.flags.optimize:
        raise RuntimeError('assertions must be enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'builder-output', 'reference', 'probe', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Admit the already fresh build and run only the frozen driver input checks.')
    a = parser.parse_args()
    assert len(a.expected_build_sha256) == 64 and all(c in '0123456789abcdef' for c in a.expected_build_sha256)
    repo = a.repo.resolve()
    assert HERE == repo / 'tests/wrapped_scalar_resource_review'
    builder_output = resolved(repo, a.builder_output)
    reference, probe = resolved(repo, a.reference), resolved(repo, a.probe)
    out = resolved(repo, a.output)
    assert out != builder_output and not out.is_relative_to(builder_output)
    assert not out.exists(), 'fresh wrapper output required'
    out.mkdir(parents=True, exist_ok=False)
    observed = out / 'observed'
    inputs = {}
    report = dict(schema='heptatorch.wrapped-scalar07b.portable-resources.v1', status='RUNNING',
                  prepare_only=a.prepare_only, native_executed=False,
                  Torch_executed_here=False, compiler_executed=False, emitter_executed=False,
                  closed_schemas=0, whole_pytorch_gate='HOLD',
                  scope='Original three finite 07b recipes only; no backward preserve-cache OOM coverage.',
                  original_driver_sha256='48185e4657f0c57c0c79c82dd4b3192573fe7c5b8b6a8122bdb02f3f21b99d9c',
                  source_sha256=SOURCE, native_sha256=IMAGE, original_build_sha256=BASIS_BUILD,
                  expected_build_sha256=a.expected_build_sha256, child_started=False, child_completed=False)
    code = 1
    try:
        report['source_check'] = source_check()
        for rel, meta in report['source_check']['bound'].items():
            inputs[str(HERE / rel)] = meta
        inputs[str(HERE / 'package-manifest.json')] = file(HERE / 'package-manifest.json')[1]
        native, admission = fresh_build(repo, builder_output, a.expected_build_sha256, inputs)
        report['fresh_build_admission'] = admission
        for path, pin in ((reference, REFERENCE), (probe, PROBE)):
            inputs[str(path)] = file(path, pin)[1]
        dump(out / 'inputs-before.json', inputs)
        command = [sys.executable, '-S', '-B', str(HERE / 'portable07b/test_wrapped_scalar_resources.py'),
                   '--native-build', str(native), '--binary', str(native / 'tensor-shape-scalar-session'),
                   '--protocol', str(HERE / 'portable07b/protocol.py'),
                   '--reference', str(reference), '--probe', str(probe),
                   '--expected-native-sha256', IMAGE,
                   '--expected-build-sha256', a.expected_build_sha256,
                   '--output', str(observed)]
        if a.prepare_only:
            command.append('--prepare-only')
        dump(out / 'command.json', {'argv': command, 'cwd': str(repo),
                                   'fresh_child_output': str(observed), 'case_selection': 'all'})
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
        report['child'] = dict(exit_code=child.returncode, stdout_EOF=True, stderr_EOF=True,
                               stdout_bytes=len(child.stdout), stdout_sha256=sha(child.stdout),
                               stderr_bytes=len(child.stderr), stderr_sha256=sha(child.stderr))
        raw, meta = file(observed / 'report.json')
        result = json.loads(raw)
        report['original_child_report'] = meta
        report['native_executed'] = result['native_executed']
        assert child.returncode == 0 and child.stderr == b''
        exact(result['schema'], 'heptatorch.wrapped-scalar07b.resources.v1', 'original child schema')
        exact(result['source_sha256'], SOURCE, 'original child source')
        exact(result['native_sha256'], IMAGE, 'original child ELF')
        exact(result['native_build_sha256'], a.expected_build_sha256, 'actual child build receipt')
        exact(result['unit_count'], 13, 'actual child units')
        exact(result['all_inputs_unchanged'], True, 'original child inputs unchanged')
        for key in ('Torch_executed_here', 'compiler_executed', 'emitter_executed', 'old_resource_cases_executed'):
            exact(result[key], False, 'original child ' + key)
        if a.prepare_only:
            exact(result['status'], 'PREPARED_NOT_RUN', 'original preparation status')
            exact(result['native_executed'], False, 'no resources during preparation')
            exact(result['cases'], [], 'no prepared case execution')
            report['status'] = 'PREPARED_NOT_RUN'
        else:
            exact(result['status'], 'PASS', 'original three controls')
            exact(result['native_executed'], True, 'actual resource execution')
            exact([r['name'] for r in result['cases']], list(CASES), 'exact ordered original recipes')
            assert all(r['status'] == 'PASS' and r['EOF_exit_code'] == 0 for r in result['cases'])
            exact(result['passed'], 3, 'three original PASS cases')
            exact(result['case_count'], 3, 'three original case denominator')
            report['status'] = 'PASS'
        report['cases'] = result['cases']
        for key in ('frames_before_EOF', 'checks', 'full_owner_rollback_bytes',
                    'full_rng_bytes', 'committed_84Q_snapshots'):
            report[key] = result[key]
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
