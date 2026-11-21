#!/usr/bin/env python3
"""Exercise F64 custody failure paths using disposable synthetic records only."""
import argparse
import copy
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import sys
import tempfile
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONTROLS = 19


def controls(seal, results):
    def passed(name, rejected=False):
        results.append(dict(name=name, status='PASS', rejected=rejected))

    native = dict(schema='heptatorch.current-object-f64-run.v1', reference='cpython',
        status='PASS', cases=55, failed_cases=0, oracle_provenance=None,
        rows=[dict(case='synthetic-' + str(i), status='PASS', mode=0) for i in range(55)],
        current_source_sha256=seal.SOURCE, native_sha256='synthetic-native-image',
        rne_accuracy=dict(status='PARTIAL', qualification_status='PASS', cases=55,
                          integer_checker_boundary_controls=18),
        sqrt_diagnostic=dict(native_errors=[], rows=[{}] * 5),
        strict_failures_reclassified=False, legacy143_restored=False,
        sqrt_public_tensor_support=False, whole_pytorch_rewrite_complete=False)
    physical = dict(schema='heptatorch.current-object-f64-physical-readback.v1',
        reference='cpython', status='PASS', cases=55, failed_cases=0,
        current_source_sha256=seal.SOURCE, native_sha256='synthetic-native-image',
        current_objects_verified=15, native_or_reference_execution_by_reader=False,
        component_qualification_status='PARTIAL', rne_repeat_cases=55,
        integer_boundary_controls=18, sqrt_diagnostic_cases=5,
        strict_failures_reclassified=False, whole_pytorch_rewrite_complete=False)
    strict = copy.deepcopy(native)
    strict.update(reference='pytorch', oracle_provenance=dict(status='PASS',
                  checks={'synthetic-' + str(i): True for i in range(11)}))
    strict_physical = dict(physical, reference='pytorch')
    hashes = {name: 'synthetic-hash' for name in
              ('cpython-readback.json', 'pytorch-readback.json', 'pytorch/execution/report.json')}
    baseline = dict(build=dict(schema='heptatorch.current-object-f64-build.v1',
        status='PASS_NATIVE_BUILD_ONLY', current_source_sha256=seal.SOURCE,
        current_native_sha256=seal.IMAGE, current_function_count=797,
        current_units=[dict(synthetic=i) for i in range(15)],
        native_sha256='synthetic-native-image', actual_current_objects_reused=True,
        legacy143_restored=False, runtime_qualified=False, whole_pytorch_rewrite_complete=False),
        cp=native, strict=strict, cp_reader=physical, strict_reader=strict_physical,
        component=dict(schema='heptatorch.current-object-f64-component.v1', status='PARTIAL',
            component_qualification_status='PASS', strict_pytorch_status='PASS',
            strict_pytorch_failed_cases=0, strict_gate_reclassified=False,
            current_source_sha256=seal.SOURCE, native_sha256='synthetic-native-image',
            original_strict_cases=55, original_rne_repeat_cases=55,
            original_rational_diagnostic_groups=5, legacy143_restored=False,
            whole_pytorch_rewrite_complete=False, reports=hashes),
        strict_exit=b'0\n', reader_exit=b'0\n', report_hashes=hashes)

    def check(name, mutate, reject=True):
        candidate = copy.deepcopy(baseline)
        mutate(candidate)
        try:
            actual = seal.f64_component_status(**candidate)
        except ValueError:
            assert reject, name
        else:
            assert not reject, name
            assert actual['status'] == 'PARTIAL'
            assert actual['strict_pytorch_status'] == candidate['strict']['status']
            assert actual['strict_gate_reclassified'] is False
        passed(name, reject)

    check('valid_component_retains_PARTIAL_scope', lambda value: None, False)

    def sqrt_difference(value):
        value['strict'].update(status='FAIL', failed_cases=1)
        value['strict']['rows'][0].update(status='FAIL', mode=4)
        value['strict_reader'].update(status='FAIL', failed_cases=1)
        value['component'].update(strict_pytorch_status='FAIL', strict_pytorch_failed_cases=1)
        value.update(strict_exit=b'1\n', reader_exit=b'1\n')

    check('strict_sqrt_FAIL_and_exit1_remain_FAIL', sqrt_difference, False)
    check('reject_failed_CPython_predecessor', lambda value: value['cp'].update(status='FAIL'))
    check('reject_non_sqrt_numeric_failure', lambda value: value['strict']['rows'][0].update(status='FAIL'))
    check('reject_reclassified_strict_sqrt_failure',
          lambda value: value['strict']['rows'][0].update(status='FAIL', mode=4))
    check('reject_stale_current_source', lambda value: value['build'].update(current_source_sha256='old-source'))
    check('reject_missing_actual_current_object', lambda value: value['build']['current_units'].pop())
    check('reject_failed_RNE_qualification', lambda value: value['strict']['rne_accuracy'].update(qualification_status='FAIL'))
    check('reject_missing_integer_boundary_control', lambda value: value['strict_reader'].update(integer_boundary_controls=17))
    check('reject_unexpected_strict_producer_exit', lambda value: value.update(strict_exit=b'2\n'))
    check('reject_unexpected_strict_reader_exit', lambda value: value.update(reader_exit=b'2\n'))
    check('reject_incomplete_original_wheel_provenance', lambda value: value['strict']['oracle_provenance']['checks'].popitem())
    check('reject_physical_readback_status_mismatch', lambda value: value['strict_reader'].update(status='FAIL'))

    def change_hash(value):
        value['component']['reports'] = dict(value['component']['reports'], **{'pytorch-readback.json': 'changed'})

    check('reject_component_report_hash_mismatch', change_hash)

    with tempfile.TemporaryDirectory(prefix='heptatorch-f64-sealer-controls-') as temporary:
        root = Path(temporary)
        try:
            seal.current_f64_consumer_custody(root / 'repository', root / 'current')
        except FileNotFoundError as error:
            assert str(root / 'current/f64-consumer/build/build.json') in str(error)
        else:
            raise ValueError('a missing actual F64 consumer entered complete custody')
        passed('reject_missing_F64_consumer_before_sealing', True)

        required = {'f64-consumer/build/build.json', 'f64-consumer/cpython/execution/report.json',
            'f64-consumer/pytorch/execution/report.json', 'f64-consumer/cpython-readback.json',
            'f64-consumer/pytorch-readback.json', 'f64-consumer/strict-exit.txt',
            'f64-consumer/readback-exit.txt', 'f64-consumer/component-status.json'}
        assert required <= set(seal.F64_CONSUMER_OUTPUTS) <= set(seal.REQUIRED_OUTPUTS)
        assert {path for path in seal.F64_CONSUMER_OUTPUTS if path.endswith('.json')} <= set(seal.INPUT_REPORTS)
        assert {'tools/current_f64_consumer', 'tools/dtype-f64/f64.hepta',
                'tools/dtype-f64/profile.json', 'tools/stack_budget/profile.json',
                'tests', 'tools/current_runtime', 'compatibility'} <= set(seal.REPOSITORY_PATHS)
        passed('complete_F64_outputs_reports_and_source_inventory_are_mandatory')

        source = root / 'build.py'
        source.write_bytes(b"VALUE = 'GOOD'\n")
        os.utime(source, (1800000000, 1800000000))
        py_compile.compile(str(source), doraise=True,
                           invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        source.write_bytes(b"VALUE = 'BAD!'\n")
        os.utime(source, (1800000000, 1800000000))
        spec = importlib.util.spec_from_file_location('synthetic_stale_cache_probe', source)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        assert old.VALUE == 'GOOD'
        saved = sys.modules.get('build')
        sentinel = types.ModuleType('synthetic_existing_build_namespace')
        original_bytes, original_text = Path.read_bytes, Path.read_text
        original_loader = importlib.machinery.SourceFileLoader.get_code
        original_path = list(sys.path)
        sys.modules['build'] = sentinel
        bindings = {}
        try:
            with seal.f64_reader_scope(root, bindings) as load:
                actual = load(source, 'build')
                assert actual.VALUE == 'BAD!' and sys.modules['build'] is actual
                assert source.read_text() == "VALUE = 'BAD!'\n"
            assert sys.modules['build'] is sentinel
            assert Path.read_bytes is original_bytes and Path.read_text is original_text
            assert importlib.machinery.SourceFileLoader.get_code is original_loader and sys.path == original_path
            assert bindings[str(source)]['sha256'] == seal.digest(source.read_bytes())
        finally:
            if saved is None:
                sys.modules.pop('build', None)
            else:
                sys.modules['build'] = saved
        passed('reader_executes_bound_source_and_restores_import_namespace')

        changing = root / 'changing.bin'
        changing.write_bytes(b'original')
        try:
            with seal.f64_reader_scope(root, {}):
                changing.read_bytes()
                changing.write_bytes(b'modified')
                changing.read_bytes()
        except ValueError as error:
            assert 'previously bound physical input changed' in str(error)
        else:
            raise ValueError('reader accepted a changed physical input')
        passed('reader_rejects_changed_physical_input', True)

        with seal.f64_reader_scope(root, {}):
            for event, values in (('subprocess.Popen', ('blocked', [], None, {})),
                                  ('import', ('torch', None, [], [], []))):
                try:
                    sys.audit(event, *values)
                except RuntimeError as error:
                    assert 'current F64 physical reader cannot' in str(error)
                else:
                    raise ValueError('reader allowed a forbidden execution event')
        passed('reader_refuses_process_and_Torch_execution_events', True)
    assert len(results) == REQUIRED_CONTROLS
    assert tuple(row['name'] for row in results) == seal.F64_SEALER_CONTROL_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    report = dict(schema='heptatorch.current-f64-sealer.synthetic-controls.v1', status='FAIL',
        required_controls=REQUIRED_CONTROLS, controls=[], inputs={},
        native_executed=False, Torch_executed=False, compiler_executed=False,
        numerical_cases_added=0, runtime_qualification_granted=False,
        whole_pytorch_rewrite_complete=False, repository_fixtures_modified=False)
    try:
        assert __debug__ and repository == ROOT
        assert not args.output.exists()
        path = repository / 'tools/seal_current_ci_evidence.py'
        spec = importlib.util.spec_from_file_location('current_f64_sealer_controls_production', path)
        seal = importlib.util.module_from_spec(spec)
        exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True, optimize=0), seal.__dict__)
        for source in (Path(__file__).resolve(), path):
            report['inputs'][str(source)] = seal.info(source)[1]
        controls(seal, report['controls'])
        for spelling, metadata in report['inputs'].items():
            assert seal.info(Path(spelling))[1] == metadata
        report.update(status='PASS_F64_SEALER_CONTROLS', all_inputs_unchanged=True)
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
    report['executed_controls'] = len(report['controls'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('inputs', 'traceback')}, sort_keys=True))
    return 0 if report['status'] == 'PASS_F64_SEALER_CONTROLS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
