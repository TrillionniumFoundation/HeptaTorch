#!/usr/bin/env python3
"""Regress source-admission failures with temporary synthetic inputs only.

The actual checkout's admission functions and SourceFileLoader hook are exercised
without running HIA or Torch. These controls do not add numerical cases or grant
runtime qualification. Only this script and the production checker are recorded
as physical inputs; synthetic files are destroyed before the report is written.
"""

import argparse
import ast
import contextlib
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import stat
import sys
import tempfile
import traceback
import types


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bound_source(path):
    path = Path(path).absolute()
    require(not path.is_symlink(), 'physical control source must not be a symlink')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'physical control source must be a regular file')
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                              value.st_mtime_ns, value.st_ctime_ns)
    require(identity(before) == identity(after), 'physical control source changed while reading')
    return raw, dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(after.st_mode))


def fresh(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def encoded(document):
    return (json.dumps(document, indent=2, sort_keys=True) + '\n').encode()


def controls(checker, checker_raw, root, results):
    Inputs = checker['Inputs']
    old, changed = b"VALUE = 'GOOD'\n", b"VALUE = 'BAD!'\n"
    require(len(old) == len(changed), 'same-size cache control required')

    def passed(name, **details):
        results.append(dict(name=name, status='PASS', **details))

    def stale(name):
        path = root / 'loader' / (name + '.py')
        fresh(path, old)
        os.utime(path, (1800000000, 1800000000))
        py_compile.compile(str(path), doraise=True,
                           invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        path.write_bytes(changed)
        os.utime(path, (1800000000, 1800000000))
        return path

    def via_spec(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    direct = stale('synthetic_direct')
    require(via_spec(direct, 'synthetic_old_cache_control').VALUE == 'GOOD',
            'the original SourceFileLoader must demonstrate the accepted stale cache')
    direct_inputs = Inputs()
    require(checker['load'](direct, direct_inputs).VALUE == 'BAD!',
            'production load executed stale bytecode instead of its bound source')
    require(direct_inputs.records[str(direct)]['sha256'] == sha(changed),
            'production load did not bind the changed source')
    passed('direct_load_executes_bound_source_despite_valid_stale_pyc')

    # Compile the exact nested production hook from the bound checker AST. Its
    # implementation is not copied or reimplemented in this control script.
    checker_tree = ast.parse(checker_raw)
    mains = [node for node in checker_tree.body
             if isinstance(node, ast.FunctionDef) and node.name == 'main']
    require(len(mains) == 1, 'one production checker main required')
    main = mains[0]
    hooks = [node for node in main.body
             if isinstance(node, ast.FunctionDef) and node.name == 'bound_source_code']
    require(len(hooks) == 1, 'the actual production SourceFileLoader hook is required')
    original_get_code = importlib.machinery.SourceFileLoader.get_code
    hook_inputs = Inputs()
    environment = dict(Path=Path, repository=root, inputs=hook_inputs,
                       original_get_code=original_get_code)
    exec(compile(ast.Module(body=hooks, type_ignores=[]), checker['__file__'], 'exec',
                 dont_inherit=True, optimize=0), environment)
    token = sha(str(root).encode())[:16]
    ordinary = stale('synthetic_ordinary_' + token)
    helper = stale('synthetic_helper_' + token)
    old_path = list(sys.path)
    require(ordinary.stem not in sys.modules, 'fresh ordinary import name required')
    try:
        importlib.machinery.SourceFileLoader.get_code = environment['bound_source_code']
        sys.path.insert(0, str(ordinary.parent))
        imported = importlib.import_module(ordinary.stem)
        require(imported.VALUE == 'BAD!', 'ordinary import executed stale bytecode')
        require(hook_inputs.records[str(ordinary)]['sha256'] == sha(changed),
                'ordinary import did not bind executed source')
        passed('ordinary_import_executes_bound_source')
        specified = via_spec(helper, 'synthetic_helper_spec_' + token)
        require(specified.VALUE == 'BAD!', 'helper spec loader executed stale bytecode')
        require(hook_inputs.records[str(helper)]['sha256'] == sha(changed),
                'helper spec loader did not bind executed source')
        passed('helper_spec_loader_executes_bound_source')
    finally:
        importlib.machinery.SourceFileLoader.get_code = original_get_code
        sys.path[:] = old_path
        sys.modules.pop(ordinary.stem, None)

    # Deliberately small synthetic inventory: these are not current numerical
    # suite counts and are never advertised as such in the resulting report.
    inventory = root / 'inventory'
    cases = {'synthetic_first': 2, 'synthetic_second': 3}
    groups, total, count, units = tuple(cases), sum(cases.values()), 11, 2
    source, image = 'a' * 64, 'b' * 64
    manifest = dict(runtime=dict(sha256=source, functions=count),
                    expected_native_sha256=image, units=list(range(units)))
    manifest_raw = encoded(manifest)
    manifest_sha = sha(manifest_raw)
    fresh(inventory / 'tools/current_runtime/source/manifest.json', manifest_raw)
    identity = dict(source_sha256=source, native_sha256=image,
                    manifest_sha256=manifest_sha, functions=count, units=units)
    sealer = ('GROUPS = ' + repr(groups) + '\nREQUIRED_CASES = ' + repr(cases) +
              '\nREQUIRED_CASES_TOTAL = ' + repr(total) + '\nSOURCE = ' + repr(source) +
              '\nIMAGE = ' + repr(image) + '\nSOURCE_MANIFEST = ' + repr(manifest_sha) + '\n')
    fresh(inventory / 'tools/seal_current_ci_evidence.py', sealer.encode())
    fresh(inventory / 'compatibility/capabilities.json', encoded(dict(current_runtime_profile=
        dict(current_image_qualification=dict(candidate_identity=identity, required_groups=cases)))))
    fresh(inventory / 'compatibility/full-acceptance.json', encoded(dict(current_runtime_integration=
        dict(candidate_identity=identity, required_cases=cases, required_suites=list(groups),
             required_current_image_cases=total, required_current_image_groups=len(groups),
             qualification_plan='validation/current/qualification-plan.json'))))
    fresh(inventory / 'validation/current/qualification-plan.json', encoded(dict(required_suites=cases,
        required_cases=total, required_groups=len(groups), required_functions=count, required_units=units,
        required_source_sha256=source, required_native_sha256=image, required_manifest_sha256=manifest_sha)))
    expected = dict(required_cases_total=total, executed_cases_total=total, required_suites=list(groups),
                    functions=count, units=units, source_sha256=source, native_sha256=image)
    comparisons = ' or '.join('qualification[' + repr(key) + '] != ' + repr(value)
                              for key, value in expected.items())
    baseline = ('import sys\nfailures = []\ntry:\n    if (' + comparisons +
                "):\n        failures.append('mismatch')\nexcept Exception:\n    failures.append('missing')\n"
                'sys.exit(1 if failures else 0)\n')
    coordinator_path = inventory / 'tests/qualify_current_runtime.py'
    coordinator_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path = inventory / '.github/workflows/current-runtime.yml'
    workflow_path.parent.mkdir(parents=True, exist_ok=True)

    def inventory_control(name, body=baseline, selected_cases=cases, selected_total=total, reject=False):
        coordinator_raw = ('GROUPS = ' + repr(groups) + '\ndef main():\n    result = {' +
            "'required_suites': list(GROUPS), 'required_cases': " + repr(selected_cases) +
            ", 'required_cases_total': " + repr(selected_total) + '}\n    return result\n')
        coordinator_path.write_text(coordinator_raw)
        workflow_path.write_text("          python3 -I -S -B - <<'SUMMARY'\n" +
            ''.join('          ' + line + '\n' for line in body.splitlines()) + '          SUMMARY\n')
        coordinator = checker['load'](coordinator_path, Inputs())
        try:
            actual = checker['inventory'](inventory, Inputs(), coordinator)
        except ValueError as error:
            require(reject, 'valid synthetic inventory unexpectedly rejected: ' + str(error))
            passed(name, rejected=True, rejection=str(error).replace(str(root), '<synthetic-directory>'))
        else:
            require(not reject, 'defective synthetic inventory or workflow was accepted: ' + name)
            require(actual == dict(groups=len(groups), cases=total, functions=count, units=units),
                    'valid synthetic inventory returned different counts')
            passed(name, rejected=False)

    inventory_control('valid_physical_inventory_and_connected_workflow')
    wrong = dict(cases)
    wrong['synthetic_second'] -= 1
    inventory_control('reject_actual_coordinator_wrong_case_map', selected_cases=wrong, reject=True)
    inventory_control('reject_actual_coordinator_wrong_total', selected_total=total - 1, reject=True)
    inventory_control('reject_workflow_dead_comparisons', body='import sys\nfailures=[]\nif False:\n    if (' +
        comparisons + "):\n        failures.append('mismatch')\nsys.exit(1 if failures else 0)\n", reject=True)
    inventory_control('reject_workflow_missing_failure_append',
        body=baseline.replace("failures.append('mismatch')", 'pass'), reject=True)
    inventory_control('reject_workflow_unconditional_success_exit',
        body=baseline.replace('sys.exit(1 if failures else 0)', 'sys.exit(0)'), reject=True)
    inventory_control('reject_workflow_and_guard', body=baseline.replace(' or ', ' and '), reject=True)

    extension_root = root / 'extension-catalogs'
    extension_groups = ('signbit', 'typed_index_select', 'indexed_f32',
                        'forward_nograd', 'join_forward', 'spatial_forward')
    extension_cases = {name: index + 1 for index, name in enumerate(extension_groups)}
    extension_hashes = {name: sha(('synthetic-' + name).encode()) for name in extension_groups}
    # Exercise the exact stale digest that previously survived admission. The
    # tiny synthetic recipes remain controls only, never numerical coverage.
    extension_hashes['spatial_forward'] = '64be7b4788f8e3c92b6466002df86c0c5ea84ffd002cca5d339ab2f591ae5a3f'
    extension_sealer = extension_root / 'tools/seal_current_ci_evidence.py'
    for name in extension_groups:
        fresh(extension_root / 'tests' / name / 'cases.py',
              ('CASES = ' + repr([dict(synthetic=index) for index in range(extension_cases[name])])
               + '\nCASE_SHA256 = ' + repr(extension_hashes[name]) + '\n').encode())

    def extension_source(hashes):
        return ('GROUPS = EXTENSION_GROUPS = ' + repr(extension_groups)
                + '\nREQUIRED_CASES = ' + repr(extension_cases)
                + '\nEXTENSION_CASE_HASHES = ' + repr(hashes) + '\n').encode()

    fresh(extension_sealer, extension_source(extension_hashes))
    extension_result = checker['extension_case_bindings'](extension_root, Inputs())
    require(extension_result == dict(groups=len(extension_groups), catalogs={
        name: dict(cases=extension_cases[name], case_sha256=extension_hashes[name])
        for name in extension_groups}), 'all synthetic extension catalogs must be checked')
    passed('valid_complete_required_extension_catalog_bindings', synthetic_catalogs=len(extension_groups))
    stale_extension_hashes = dict(extension_hashes)
    stale_extension_hashes['spatial_forward'] = '62f81aa0ed6626b6fe867c89549b04c395ee019114713b43fef2b8e8bf1d3cf3'
    extension_sealer.write_bytes(extension_source(stale_extension_hashes))
    try:
        checker['extension_case_bindings'](extension_root, Inputs())
    except ValueError as error:
        require(str(error) == 'actual extension catalog digest differs: spatial_forward',
                'stale spatial digest was rejected for an unrelated reason')
        passed('reject_frozen_spatial320_digest_after_catalog_extension', rejected=True,
               rejection=str(error))
    else:
        raise ValueError('the exact stale spatial 320-case digest was accepted')

    mode_root = root / 'mode'
    source_raw = b'fn sentinel()\n  return 0\nend\n'
    support = mode_root / 'support.hepta'
    support_raw = b'synthetic support\n'
    fresh(support, support_raw)
    support.chmod(0o644)
    support_row = dict(bytes=len(support_raw), sha256=sha(support_raw), mode=0o644)
    fresh(mode_root / 'tools/current_runtime/source/runtime.hepta', source_raw)
    fresh(mode_root / 'tools/current_runtime/half_addsub_fe_owner_contract.json', encoded(dict(
        function_identities={'sentinel': dict(bytes=len(source_raw), sha256=sha(source_raw))},
        support_unit=dict(path='support.hepta', **support_row))))
    fresh(mode_root / 'tools/current_runtime/half_addsub_fe_tombstone_contract.json', encoded(dict(
        functions={}, predicate=dict(path='support.hepta', **support_row))))
    fresh(mode_root / 'tests/current_runtime_resources/sources.json',
          encoded(dict(files={'support.hepta': support_row})))
    fresh(mode_root / 'tests/current_runtime_asstrided/sources.json', encoded(dict(repository_files={})))
    fresh(mode_root / 'tests/current_runtime_metadata10/source.json', encoded(dict(dependencies={})))
    require(checker['contracts'](mode_root, Inputs()) == dict(complete_HIA_functions=1, bound_source_files=1),
            'valid synthetic contract must be admitted before mode mutation')
    try:
        support.chmod(0o4644)
        require(stat.S_IMODE(support.stat().st_mode) == 0o4644, 'special-bit mode mutation was not applied')
        try:
            checker['contracts'](mode_root, Inputs())
        except ValueError as error:
            require('bound source mode' in str(error), 'mode control rejected for an unrelated reason')
            passed('reject_special_permission_bit_mismatch', declared_mode=0o644, actual_mode=0o4644)
        else:
            raise ValueError('same lower-nine-bit mode with changed special bits was accepted')
    finally:
        support.chmod(0o644)

    # Execute the exact production With statement against an instrumented
    # synthetic context manager. Check entry, exit and both identity guards.
    entered = [node for block in main.body if isinstance(block, ast.Try) for node in block.body
               if isinstance(node, ast.With) and len(node.items) == 1
               and isinstance(node.items[0].context_expr, ast.Call)
               and isinstance(node.items[0].context_expr.func, ast.Attribute)
               and node.items[0].context_expr.func.attr == 'frozen_reader']
    require(len(entered) == 1, 'the production frozen reader must be entered with a With statement')
    entered_code = compile(ast.Module(body=entered, type_ignores=[]), checker['__file__'], 'exec',
                           dont_inherit=True, optimize=0)
    context_inputs = Inputs()
    for wrong_field in (None, 'SOURCE_SHA', 'IMAGE_SHA'):
        events = []
        actual = dict(SOURCE_SHA=source, IMAGE_SHA=image)
        if wrong_field:
            actual[wrong_field] = 'c' * 64

        @contextlib.contextmanager
        def reader(selected_root, selected_inputs):
            require(selected_root == root and selected_inputs is context_inputs,
                    'the exact synthetic context arguments are required')
            events.append('enter')
            try:
                yield types.SimpleNamespace(run=types.SimpleNamespace(**actual))
            finally:
                events.append('exit')

        context_environment = dict(coordinator=types.SimpleNamespace(public_f64_logical192=
            types.SimpleNamespace(frozen_reader=reader)), repository=root, inputs=context_inputs,
            built=dict(source_sha256=source, native_sha256=image), require=checker['require'])
        try:
            exec(entered_code, context_environment)
        except ValueError:
            require(wrong_field is not None, 'valid entered synthetic reader was rejected')
        else:
            require(wrong_field is None, 'an entered reader with a changed identity was accepted')
        require(events == ['enter', 'exit'], 'the production statement must enter and exit its reader')
    passed('production_frozen_reader_enters_exits_and_checks_both_identities',
           synthetic_context_variants=3)
    require(len(results) == 14, 'the complete fourteen synthetic controls must run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    self_path = Path(__file__).resolve()
    checker_path = repository / 'tests/check_current_runtime_admission.py'
    report = dict(schema='heptatorch.current-source-admission.synthetic-controls.v1', status='FAIL',
        scope='synthetic source-admission controls only', runtime_executed=False,
        native_executed=False, Torch_executed=False, runtime_qualification_granted=False,
        full_pytorch_rewrite_complete=False, numerical_cases_added=0,
        repository_fixtures_modified=False, temporary_fixtures_retained=False,
        required_controls=14, controls=[], inputs={})
    originals = {}
    try:
        require(not sys.flags.optimize, 'synthetic controls require unoptimized Python assertions')
        require(not args.output.exists(), 'preserve every original synthetic-control report')
        sys.dont_write_bytecode = True
        for path in (self_path, checker_path):
            raw, metadata = bound_source(path)
            originals[str(path)] = raw
            report['inputs'][str(path)] = metadata
        checker_raw = originals[str(checker_path)]
        checker = {'__file__': str(checker_path), '__name__': 'synthetic_controls_production_checker'}
        exec(compile(checker_raw, str(checker_path), 'exec', dont_inherit=True, optimize=0), checker)
        sys.addaudithook(checker['prohibit_execution'])
        with tempfile.TemporaryDirectory(prefix='heptatorch-admission-controls-') as temporary:
            controls(checker, checker_raw, Path(temporary), report['controls'])
        require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
                'synthetic source controls imported Torch')
        for spelling, original in originals.items():
            raw, metadata = bound_source(Path(spelling))
            require(raw == original and metadata == report['inputs'][spelling],
                    'physical control source changed during the run')
        report.update(status='PASS_SYNTHETIC_ADMISSION_CONTROLS', all_inputs_unchanged=True)
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
    report['executed_controls'] = len(report['controls'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fresh(args.output, encoded(report))
    print(json.dumps({key: value for key, value in report.items() if key not in ('inputs', 'traceback')}, sort_keys=True))
    return 0 if report['status'] == 'PASS_SYNTHETIC_ADMISSION_CONTROLS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
