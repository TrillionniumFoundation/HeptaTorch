#!/usr/bin/env python3
"""Launch the separate Half reference worker in an explicitly selected fixed venv.

Keep the requested venv Python spelling in argv: resolving that symlink for
execution would lose its environment. Independently snapshot and pre-bind the
resolved executable entity, then retain the original supervisor's two streams.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
import traceback
import types


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SUPERVISOR = '36e3432deac48085ba0586c85547cee8432d1533c5915238db2439b43dbca523'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def load(raw, name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True, help='Existing fixed-wheel venv Python; never auto-installed here.')
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-producer-sha256', required=True)
    args = parser.parse_args()
    require(not sys.flags.optimize, 'source assertions remain enabled')
    repo = args.repository.resolve(strict=True)
    destination = args.output.absolute()
    require(not destination.exists() and not destination.is_symlink(), 'fresh reference production directory')
    destination.mkdir(parents=True, exist_ok=False)
    execution, reference = destination / 'execution', destination / 'reference'
    execution.mkdir()
    result = dict(schema='heptatorch.half-logical.actual-reference-production.v1', status='FAIL',
        reference_production_started=False, native_executed=False, compiler_executed=False,
        original_736_runtime_and_prior_reports_modified=False, full_project_status='HOLD')
    code = 1
    try:
        manifest_raw = (HERE / 'manifest.json').read_bytes()
        require(sha(manifest_raw) == args.expected_producer_sha256, 'explicit fixed producer manifest')
        manifest = json.loads(manifest_raw)
        worker_path = HERE / 'reference.py'
        worker_raw = worker_path.read_bytes()
        require(sha(worker_raw) == manifest['files']['reference.py']['sha256'], 'fixed complete worker source')
        worker = load(worker_raw, 'half_logical_reference_source_only_api', worker_path)
        inputs = worker.Inputs()
        worker.verify_package(HERE, args.expected_producer_sha256, inputs)
        inputs.bind(Path(__file__).resolve())
        supervisor_path = repo / 'tests/current_runtime_process.py'
        supervisor_raw = inputs.bind(supervisor_path, SUPERVISOR)
        runtime = load(supervisor_raw, 'unchanged_half_reference_supervisor', supervisor_path)
        requested_python = args.python.absolute()
        python = requested_python.resolve(strict=True)
        inputs.bind(python)
        interpreter = dict(path=str(python), **inputs.rows[str(python)])
        require(interpreter['mode'] & 0o111, 'selected Python is executable')
        captured = execution / 'observer-python.bin'
        shutil.copy2(python, captured)
        inputs.bind(captured, interpreter['sha256'], interpreter['bytes'], interpreter['mode'])
        command = [str(requested_python), '-I', '-B', str(worker_path), '--repository', str(repo),
            '--output', str(reference), '--expected-producer-sha256', args.expected_producer_sha256,
            '--execution-intent', str(execution / 'intent.json')]
        write_json(execution / 'intent.json', dict(argv=command, observer_python=interpreter))
        inputs.bind(execution / 'intent.json')
        for name, expected in worker.SUPPORT.items():
            inputs.bind(repo / name, expected)
        inputs.bind(repo / 'build/pytorch-install.json')
        write_json(execution / 'inputs-before.json', dict(argv=command, cwd=str(repo), inputs=dict(inputs.rows),
            actual_observer_python_prebound=True, interpreter=interpreter,
            requested_venv_spelling_retained=str(requested_python), producer_manifest_sha256=args.expected_producer_sha256))
        result['reference_production_started'] = True
        receipt = runtime.execute(command, execution, 1200, cwd=repo, mirror_stderr=True)
        result['process'] = receipt
        require(receipt['started'] is True and receipt['exit_code'] == 0
                and receipt['stdout_EOF'] is True and receipt['stderr_EOF'] is True
                and receipt['forced_cleanup'] is False and not receipt['cleanup_errors']
                and not receipt['running_descendants_after_cleanup'], 'complete actual reference process with two physical EOFs')
        streams = {}
        for name in ('stdout', 'stderr'):
            raw = inputs.bind(execution / ('runner.' + name))
            require(len(raw) == receipt[name + '_bytes'] and sha(raw) == receipt[name + '_sha256'],
                    'complete actual reference ' + name + ' bytes')
            streams[name] = dict(path=str(execution / ('runner.' + name)), bytes=len(raw), sha256=sha(raw), EOF=True)
        observed_raw = inputs.bind(reference / 'report.json')
        observed = worker.parse(observed_raw)
        require(observed['status'] == 'OBSERVED_HALF_LOGICAL_REFERENCE'
                and observed['cases'] == len(observed['rows']) == 264
                and observed['Torch_executed'] is True and observed['native_executed'] is False
                and observed['compiler_executed'] is False and observed['inputs_unchanged'] is True
                and observed['rng_unchanged'] is True and observed['all_observed_blobs_unchanged'] is True
                and observed['host_expected_values_computed'] is False,
                'complete new actual Torch observations only')
        require(observed['actual_process']['pid'] == receipt['pid']
                and observed['actual_process']['argv'] == receipt['argv']
                and observed['actual_process']['python'] == interpreter
                and any(row['pid'] == receipt['pid'] and row['start_ticks'] == observed['actual_process']['start_ticks']
                        for row in receipt['tracked_processes']), 'actual PID incarnation and prebound interpreter match the worker')
        for path, row in observed['inputs'].items():
            inputs.bind(path, row['sha256'], row['bytes'], row['mode'])
        for name, row in observed['blob_manifest'].items():
            inputs.bind(reference / name, row['sha256'], row['bytes'], row['mode'])
        for index, row in enumerate(observed['rows']):
            require(worker.parse(inputs.bind(reference / ('case-%03d.json' % index))) == row,
                    'complete physical standalone observation equals aggregate')
        summary = worker.parse((execution / 'runner.stdout').read_bytes())
        require(summary == {key:observed.get(key) for key in ('status','cases','accepted','observed_errors','error')},
                'actual worker stdout is the complete reference summary')
        inputs.recheck()
        result.update(status='OBSERVED_COMPLETE_HALF_LOGICAL_REFERENCE_EXECUTION',
            producer_manifest_sha256=args.expected_producer_sha256,
            report=dict(path=str(reference / 'report.json'), bytes=len(observed_raw), sha256=sha(observed_raw)),
            cases=264, accepted=observed['accepted'], observed_errors=observed['observed_errors'],
            streams=streams, interpreter=interpreter, actual_interpreter_bound_before_execution=True,
            oracle_provenance=observed['oracle_provenance'], all_physical_inputs_unchanged=True,
            inputs=inputs.rows, new_runtime_qualification_transferred=False)
        code = 0
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    write_json(destination / 'production.json', result)
    print(json.dumps({key:result.get(key) for key in ('status','cases','accepted','observed_errors','error')}, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
