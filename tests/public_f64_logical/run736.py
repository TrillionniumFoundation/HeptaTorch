#!/usr/bin/env python3
"""Admit one cold HIA build and compare 192 public F64 logical observations.

The actual Torch producer and its two allocation observations are independent
inputs. Six Torch observations beyond the dense no-gradient scope remain
observations; the two native boundary controls do not claim compatibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import cases
import common
import native

SOURCE_SHA = 'c73a7cb474981056404ac96bc214979d57aaa9a86e6474b1c140c3d98eb05779'
IMAGE_SHA = '1ea247df76ff96c307e293790aa38de4a0bf0ff94b8148a480653c924658679d'
PYTHON_SHA = 'e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f'
BUILDER_SHA = '9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
ADMISSION_SHA = '9a45cee45a289dae0a35b0a256f66f6a1ea2d3c8becd30bb2174da1ffb51e9bb'
SOURCE_FILES = {
    'cases.py': 'e689699b519e57eec44edac5d4c319d3ea718a3777056d1514d1f0e43a08bef3',
    'reference.py': '2531920b4b13c991b33a3608ea2c168b184e4e7d6891b870ae874e95b1c9e296',
    'common.py': '29d25567c08e309c25d96d56f4476e3842ceda164c9647eb5c9397a04fad181a',
    'native.py': 'cc48760e9dad13dfdd834dca5086bc5fb96803dc29914d43b20f29cfb239320e',
    'run.py': 'a155cea27556e7c8647835f11641b4732557dd882d3ec024820a06cd96aeab5f',
    'qualify.py': '98b150d9295451c3be9ff4209d9f42350430e46ec0c3cd42ab3e84eab91b84f2',
    'qualify736.py': '5b89e78607e070cc8475d468078fe00a3cbb1643fc756d70c8d97689cbccb7ee',
    'derive736.py': '1de09fc1578bc3c886fdf08555a947822c4b6db4dd7e5757aa4e4241ef5e7483',
}
SUPPORT_FILES = {
    'tests/pytorch_oracle.py': 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae',
    'compatibility/pytorch-oracle-lock.json': 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    common.dump(path, value)


def pin(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'an explicit full SHA256 is required')
    return value


def local_receipt_file(directory, row, inputs):
    relative = Path(row['path'])
    require(not relative.is_absolute() and len(relative.parts) == 1 and relative.name not in ('.', '..'),
            'execution stream must have an explicit local filename')
    raw = inputs.bind(directory / relative, pin(row['sha256']))
    require(len(raw) == row['bytes'], 'execution stream extent changed')
    return raw


def admit_reference(args, inputs):
    """Read the actual reference, its original process receipt, and both streams."""
    raw = inputs.bind(args.reference, pin(args.expected_reference_sha256))
    reference = json.loads(raw)
    require(reference['schema'] == 'heptatorch.public-f64-dense-logical.reference.v1'
            and reference['status'] == 'OBSERVED_PUBLIC_F64_DENSE_REFERENCE'
            and reference['torch_executed'] is True and reference['native_executed'] is False
            and reference['inputs_unchanged'] is True and reference['rng_unchanged'] is True
            and reference['wrapped_scalar_used'] is False and reference['original147_modified'] is False
            and reference['full_schema_closure'] is False,
            'the complete independent actual public-F64 reference is required')
    require(reference['case_sha256'] == cases.CASE_SHA256
            and reference['allocation_sha256'] == cases.ALLOCATION_SHA256
            and reference['boundary_sha256'] == cases.BOUNDARY_SHA256
            and reference['cases'] == reference['expected_cases'] == len(reference['rows']) == len(cases.CASES) == 192
            and reference['allocation_observations'] == reference['expected_allocation_observations']
            == len(reference['allocation_rows']) == len(cases.ALLOCATION_CASES) == 2
            and reference['boundary_observations'] == reference['expected_boundary_observations']
            == len(reference['boundary_rows']) == len(cases.BOUNDARY_OBSERVATIONS) == 6
            and reference['total_actual_observations'] == 200,
            'all 192 plus 2 plus 6 literal observations are required')
    require([row['spec'] for row in reference['rows']] == list(cases.CASES)
            and [row['index'] for row in reference['rows']] == list(range(192))
            and [row['spec'] for row in reference['allocation_rows']] == list(cases.ALLOCATION_CASES)
            and all(row['accepted'] is True for row in reference['allocation_rows']),
            'ordered complete dense and allocation recipes changed')
    for row in reference['rows'] + reference['allocation_rows']:
        require(row['input_unchanged'] is True and row['rng_unchanged'] is True,
                'reference source or generator changed')
        if row['accepted']:
            require(row['fresh_tensor'] is True and row['fresh_storage'] is True and row['error'] is None,
                    'accepted reference did not return independent output objects')
        else:
            require(row['result'] is None and type(row['error']) is dict,
                    'reference rejection is incomplete')
        native.expected_status(row)
    for spec, row in zip(cases.BOUNDARY_OBSERVATIONS, reference['boundary_rows']):
        require(row['spec']['name'] == spec['name'] and row['spec']['operation'] == spec['operation']
                and row['scope'] == 'OUTSIDE_DENSE_NO_GRAD_NATIVE_EXTENSION'
                and row['native_comparison_required_here'] is False
                and row['input_unchanged'] is True and row['rng_unchanged'] is True
                and row['input']['dtype'] == 'torch.float64'
                and row['input']['requires_grad'] == (spec['kind'] == 'requires_grad_public_float64')
                and row['input']['is_view'] == (spec['kind'] == 'as_strided_public_float64'),
                'six explicitly out-of-scope actual observations must remain visible')
    provenance = reference['oracle_provenance']
    require(provenance['status'] == 'PASS' and len(provenance['checks']) == 11
            and all(value is True for value in provenance['checks'].values()),
            'all eleven fixed-wheel provenance checks are required')
    require(reference['source_sha256'] == SOURCE_FILES['reference.py'], 'reference producer changed')
    reference_inputs = reference['inputs']
    require(len(reference_inputs) == 6, 'exact six reference source/provenance inputs required')
    for basename, expected in [('reference.py', SOURCE_FILES['reference.py']), ('cases.py', SOURCE_FILES['cases.py'])]:
        matches = [row for path, row in reference_inputs.items() if Path(path).name == basename]
        require(len(matches) == 1 and matches[0]['sha256'] == expected
                and matches[0]['bytes'] == len(inputs.bind(HERE / basename, expected)),
                'reference producer or literal cases were not bound')
    for relative, expected in SUPPORT_FILES.items():
        matches = [row for path, row in reference_inputs.items() if path.endswith('/' + relative)]
        require(len(matches) == 1 and matches[0]['sha256'] == expected,
                'fixed reference provenance support changed: ' + relative)
    require(len(bytes.fromhex(reference['rng_hex'])) == 5056
            and sha(bytes.fromhex(reference['rng_hex'])) == reference['rng_sha256'],
            'complete actual reference generator state is required')

    receipt_raw = inputs.bind(args.reference_execution, pin(args.expected_execution_sha256))
    execution = json.loads(receipt_raw)
    process = execution['process']
    require(execution['schema'] == 'heptatorch.torch-reference-execution-public.v2'
            and execution['status'] == reference['status'] and process['state'] == 'succeeded'
            and process['exit_code'] == 0 and process['timed_out'] is False
            and process['terminal_publication_confirmed'] is True,
            'actual reference process must have completed successfully')
    require(execution['native_executed'] is False and execution['inputs_unchanged'] is True
            and execution['rng_unchanged'] is True
            and execution['source']['sha256'] == SOURCE_FILES['reference.py']
            and execution['source']['bytes'] == len(inputs.bind(HERE / 'reference.py', SOURCE_FILES['reference.py']))
            and execution['report']['sha256'] == sha(raw) and execution['report']['bytes'] == len(raw)
            and execution['oracle_provenance'] == provenance
            and execution['rng_sha256'] == reference['rng_sha256']
            and execution['python']['sha256'] == PYTHON_SHA
            and execution['python']['bytes'] == 8020928,
            'actual reference process/source/report/Python binding changed')
    require(execution['cases'] == execution['expected_cases'] == 192,
            'process receipt dense denominator differs')
    staged = execution['staged_inputs'] + execution['support_inputs']
    for path, row in reference_inputs.items():
        matches = [entry for entry in staged if entry.get('staged_path', entry['path']) == path]
        require(len(matches) == 1 and matches[0]['sha256'] == row['sha256']
                and matches[0]['bytes'] == row['bytes'], 'reference input lacks actual staging identity: ' + path)
    stream_data = {}
    for stream in ('stdout', 'stderr'):
        identity = execution['streams'][stream]
        require(identity['eof'] is True and identity['offset'] == 0
                and identity['next_offset'] == identity['bytes'],
                'complete original reference stream required: ' + stream)
        stream_data[stream] = local_receipt_file(args.reference_execution.parent, identity, inputs)
    staging_raw = local_receipt_file(args.reference_execution.parent, execution['staging_report'], inputs)
    lines = stream_data['stdout'].splitlines()
    require(len(lines) == 2, 'both original staging and reference stdout records are required')
    staged_summary, summary = [json.loads(line) for line in lines]
    require(staged_summary['status'] == 'STAGED_PUBLIC_DTYPE4_REFERENCE192'
            and staged_summary['files'] == 2
            and staged_summary['input_bytes'] == execution['source']['bytes'] + execution['case_source']['bytes']
            and staged_summary['staging_sha256'] == sha(staging_raw)
            and execution['case_source']['sha256'] == SOURCE_FILES['cases.py']
            and summary['status'] == reference['status'] and summary['cases'] == 192
            and summary['total_actual_observations'] == 200 and summary.get('error') is None,
            'the two original stdout records do not describe the bound staging and reference')
    return reference, execution


def admit(args, inputs):
    require(args.repository == ROOT, 'reader and repository must be the same physical checkout')
    inputs.bind(Path(__file__).resolve())
    for name, expected in SOURCE_FILES.items():
        inputs.bind(HERE / name, expected)
    inputs.bind(ROOT / 'tools/current_runtime/build.py', BUILDER_SHA)
    inputs.bind(ROOT / 'tools/current_runtime/qualification.py', ADMISSION_SHA)
    inputs.bind(ROOT / 'tests/differentiable_view_native_review/protocol.py', native.PROTOCOL_SHA)
    inputs.bind(ROOT / 'tests/differentiable_view_native_review/owner_client.py',
                'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b')
    cold_inputs = common.Inputs()
    built, bodies = common.bind_build(args.build_directory, pin(args.expected_build_sha256), cold_inputs)
    require(len(cold_inputs.records) == 73, 'all 73 physical cold-build inputs are required')
    for path, identity in cold_inputs.records.items():
        inputs.bind(Path(path), identity['sha256'])
        require(inputs.records[path] == identity, 'cold-build physical input identity changed')
    require(built['source_sha256'] == SOURCE_SHA and built['native_sha256'] == IMAGE_SHA
            and built['functions'] == len(bodies) == 797 and built['units_planned'] == 15,
            'this reader requires the exact integrated 797-function source and image')
    reference, execution = admit_reference(args, inputs)
    return built, reference, execution, cold_inputs.records


def scope_controls(args, wire, folder):
    """Retain current native boundaries without comparing them to Torch acceptance."""
    n = native.Native(args, wire, folder)
    rows = []
    try:
        source = n.create(4, [2], struct.pack('<2Q', 0, 0x8000000000000000), 0)
        before, _ = n.tensor(source)
        gen = n.one(60, 1729)['value']
        rng = n.one(63, gen)['payload']
        n.check('boundary-complete-RNG-width', len(rng), 5056)
        cfg = struct.pack('<8Q', 1, 1, 2, 0, 0, 0, 160, 16)
        payload = wire.Payload(cfg + struct.pack('<2Q', 0, 0x8000000000000000))
        view = wire.Payload(struct.pack('<10Q', 1, 2, 0, 0, 0, 1, 0, 0, 0, 0))
        controls = [('requires-grad-public-f64-create', (97, 4, payload), 5),
                    ('public-f64-as-strided', (111, source, view, 80), 7)]
        for name, command, expected in controls:
            status, records, _, _ = n.issue([command], name)
            n.check(name + '/status', status, expected)
            n.check(name + '/no-records', records, [])
            after, _ = n.tensor(source)
            n.check(name + '/source-unchanged', after, before)
            rows.append(dict(name=name, status='PASS_NATIVE_BOUNDARY_ONLY', error_status=status,
                             complete_owner_rollback=True, torch_compatibility_claimed=False))
        n.check('boundary-complete-RNG-unchanged', n.one(63, gen)['payload'], rng)
        n.one(41, source)
        n.one(66, gen)
        return dict(status='PASS_NATIVE_BOUNDARIES_ONLY', controls=2, rows=rows,
                    checks=n.checks, frames=n.s.frames, torch_compatibility_claimed=False)
    finally:
        n.finish()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--expected-reference-sha256', required=True)
    parser.add_argument('--reference-execution', type=Path, required=True)
    parser.add_argument('--expected-execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    for key in ('repository', 'build_directory', 'reference', 'reference_execution', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    args.binary = args.build_directory / 'heptatorch'
    return args


def main():
    args = arguments()
    require(not sys.flags.optimize, 'invariant assertions must remain enabled')
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = common.Inputs()
    report = dict(schema='heptatorch.public-f64-dense-logical.native.v1', status='RUNNING',
                  case_sha256=cases.CASE_SHA256, allocation_sha256=cases.ALLOCATION_SHA256,
                  boundary_sha256=cases.BOUNDARY_SHA256, expected_cases=192, rows=[],
                  native_executed=False, native_attempted=False, torch_executed=False,
                  compiler_executed=False, original147_executed=False,
                  full_pytorch_rewrite_complete=False, full_schema_closure=False,
                  prepare_only=args.prepare_only)
    code = 0
    try:
        built, reference, execution, cold_inputs = admit(args, inputs)
        report.update(source_sha256=built['source_sha256'], native_sha256=built['native_sha256'],
                      actual_build_sha256=args.expected_build_sha256,
                      reference_sha256=args.expected_reference_sha256,
                      reference_execution_sha256=args.expected_execution_sha256,
                      cold_admission_inputs=cold_inputs,
                      boundary_observations=reference['boundary_rows'],
                      boundary_observations_not_counted_as_native_pass=True)
        if args.prepare_only:
            report['status'] = 'PREPARED_NOT_EXECUTED'
        else:
            wire = native.load_wire(ROOT)
            report['native_attempted'] = True
            for index, row in enumerate(reference['rows']):
                observed = native.execute_case(args, wire, row, args.output / ('case-%03d' % index))
                report['rows'].append(observed)
                dump(args.output / 'report.json', report)
            report['packet_controls'] = native.controls(args, wire, args.output / 'packet-controls')
            report['allocation_controls'] = [native.allocation_control(
                args, wire, args.output / ('allocation-%d' % op), op, row)
                for op, row in zip((118, 119), reference['allocation_rows'])]
            report['scope_controls'] = scope_controls(args, wire, args.output / 'scope-controls')
            require(len(report['rows']) == 192 and all(row['status'] == 'PASS' for row in report['rows']),
                    'every independent dense reference case must pass')
            report.update(status='PASS_PUBLIC_F64_LOGICAL192_ONLY', cases=192,
                          checks=sum(row['checks'] for row in report['rows']),
                          frames=sum(row['frames'] for row in report['rows']) + report['packet_controls']['frames']
                          + sum(row['frames'] for row in report['allocation_controls']) + report['scope_controls']['frames'])
    except BaseException as error:
        report.update(status='FAIL', error=dict(type=type(error).__name__, message=str(error),
                                               traceback=traceback.format_exc()))
        code = 1
    finally:
        changed = inputs.recheck()
        report.update(inputs=inputs.records, all_inputs_unchanged=not changed, input_recheck_errors=changed)
        report['native_executed'] = any(path.stat().st_size > 0 for path in args.output.glob('*/raw/responses.frames.bin'))
        if changed:
            report['status'] = 'FAIL'
            code = 1
        dump(args.output / 'report.json', report)
    print(json.dumps({key: report.get(key) for key in ('status', 'cases', 'expected_cases', 'checks', 'frames', 'error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
