#!/usr/bin/env python3
"""Execute the full original35 live-reference gate and its unchanged controls."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from current_runtime_fixtures import restore_current35
from current_runtime_process import execute
from current_runtime_admission import SOURCE, IMAGE, require, sha, complete_runner, current35_live_runner, qualify_current35


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--torch-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not sys.flags.optimize, 'all original Python assertions must remain enabled')
    from qualify_current_runtime import Inputs, build_binding
    build, output, torch_python = args.build_directory.resolve(), args.output.resolve(), args.torch_python.absolute()
    output.mkdir(parents=True, exist_ok=False)
    inputs, failures = Inputs(), []
    report = dict(schema='heptatorch.current-runtime.original35.v1', status='RUNNING',
        source_sha256=SOURCE, native_sha256=IMAGE, actual_build_sha256=args.expected_build_sha256,
        cases=0, frames=0, compared_values=0, native_executed=None, Torch_executed=None,
        executions={}, reports={}, failures=failures, full_project_status='HOLD',
        full_pytorch_rewrite_complete=False, previous_qualification_reused=False)
    try:
        inputs.bind(Path(__file__))
        for name in ('current_runtime_fixtures.py', 'current_runtime_process.py', 'current_runtime_admission.py',
                     'current_runtime_torch.py', 'qualify_current_runtime.py'):
            inputs.bind(ROOT / 'tests' / name)
        require(build_binding(build, inputs) == args.expected_build_sha256, 'same complete actual build')
        inputs.bind(torch_python.resolve())
        lock = json.loads(inputs.bind(ROOT / 'compatibility/pytorch-oracle-lock.json',
            'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'))
        inputs.bind(ROOT / lock['oracle']['install_report'])
        package = restore_current35(ROOT, output / 'source-package', inputs)
        commands = [
            ('native_and_live_reference', 730, [str(torch_python), '-B', str(ROOT / 'tests/current_runtime_torch.py'),
                '--source-observation', str(output / 'torch-source.json'), '--', str(package / 'run.py'),
                '--legacy-directory', str(ROOT / 'tests/tensor_object_oracle'),
                '--native-source', str(build / 'runtime.hepta'), '--build-directory', str(build),
                '--expected-build-sha256', args.expected_build_sha256,
                '--observer-output', str(output / 'observer'),
                '--observer-report', str(output / 'observer-report.json'), '--',
                str(build / 'heptatorch'), str(output / 'original-report.json'),
                '--owned-reference', str(ROOT / 'tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py'),
                '--session-reference', str(ROOT / 'tests/pytorch_session_oracle.py'),
                '--reference', str(ROOT / 'tests/pytorch_oracle.py'),
                '--lock', str(ROOT / 'compatibility/pytorch-oracle-lock.json'), '--repo', str(ROOT),
                '--trace-directory', str(output / 'wire'), '--expected-native-sha256', IMAGE]),
            ('strict_report', 60, [sys.executable, '-I', '-S', '-B', str(package / 'check_report.py'),
                '--observer-report', str(output / 'observer-report.json'),
                '--original-report', str(output / 'original-report.json')]),
            ('copied_controls', 90, [sys.executable, '-I', '-S', '-B', str(package / 'controls.py'),
                '--observer', str(package / 'observer701.py'),
                '--expected-profile', 'current701-cat-empty-original35-F32-ViewInfo-Saved',
                '--journals', str(output / 'observer/owners'), '--output', str(output / 'copied-controls')]),
        ]
        for name, budget, command in commands:
            directory = output / (name + '-execution')
            directory.mkdir()
            try:
                receipt = execute(command, directory, budget, cwd=ROOT,
                                  mirror_stderr=name != 'native_and_live_reference')
                report['executions'][name] = receipt
                if name == 'native_and_live_reference':
                    report['live_reference_stderr_admission'] = current35_live_runner(receipt, output, inputs)
                else:
                    complete_runner(receipt)
            except Exception as error:
                if (directory / 'runner.json').exists():
                    report['executions'][name] = json.loads((directory / 'runner.json').read_text())
                failures.append(name + ': ' + repr(error))
        paths = {'original': 'original-report.json', 'observer': 'observer-report.json',
                 'strict': 'strict_report-execution/runner.stdout', 'controls': 'copied-controls/report.json'}
        decoded = {}
        for name, relative in paths.items():
            raw = inputs.bind(output / relative)
            decoded[name] = json.loads(raw)
            report['reports'][name] = dict(path=relative, bytes=len(raw), sha256=sha(raw))
        original, strict, controls = (decoded[name] for name in ('original', 'strict', 'controls'))
        report.update(cases=original['cases'], frames=original['frames'],
            compared_values=original['compared_values'],
            native_executed=bool(original.get('rows')), Torch_executed=bool(original.get('oracle_provenance')),
            oracle_provenance_checks=strict['oracle_provenance_checks'],
            preserved_explicit_state_version_differences=original['explicit_state_absolute_version_differences'],
            positive_controls=controls['positive_cases'], negative_controls=controls['negative_cases'])
        report['all_inputs_unchanged'] = not inputs.recheck()
        report['status'] = 'PASS' if not failures else 'FAIL'
        qualify_current35(report, args.expected_build_sha256, output, inputs)
    except Exception as error:
        failures.append(repr(error))
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        errors = inputs.recheck()
        if errors:
            failures.append('bound current35 inputs changed')
        report.update(all_inputs_unchanged=not errors, input_recheck_errors=errors, checked_inputs=inputs.records,
                      status='PASS' if not failures and report.get('status') == 'PASS' else 'FAIL')
        (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'cases', 'frames', 'failures')}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
