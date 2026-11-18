#!/usr/bin/env python3
"""Exercise four fixed cat/stack packets on the admitted current HIA image.

Three successful legacy responses must round-trip exactly. The fourth packet
checks successful native capacity growth, retaining its original capacity
failure fixture as historical evidence rather than claiming error equivalence.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import stat
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import packet
import transport


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    packet.require(not sys.flags.optimize, 'invariant assertions must remain enabled')
    repo, build, out = args.repository.resolve(), args.build_directory.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binding = load(repo / 'tests/current_runtime_original35/binding.py', 'cat4_current_binding')
    inputs = binding.Inputs()
    cases, error = [], None
    report = dict(schema='heptatorch.current-runtime.cat4.v1', status='RUNNING',
                  prepare_only=args.prepare_only, cases=cases, native_executed=False,
                  Torch_executed=False, compiler_executed=False, original_expected_bytes_modified=False,
                  complete_legacy461_qualified=False, full_pytorch_rewrite_complete=False,
                  schema_closure=False, full_project_status='HOLD')

    def save():
        report['native_executed'] = any(row.get('native_executed') is True for row in cases)
        dump(out / 'report.json', report)

    save()
    try:
        for name in ('run.py', 'packet.py', 'transport.py', 'cases.json'):
            inputs.bind(HERE / name)
        definitions = json.loads(inputs.bind(HERE / 'cases.json'))
        packet.require(definitions['schema'] == 'heptatorch.cat4.fixed-fixtures.v1'
                       and len(definitions['cases']) == 4, 'four fixed regression packets')
        packet.require(packet.sha(inputs.bind(HERE / 'packet.py')) == definitions['packet_sha256']
                       and packet.sha(inputs.bind(HERE / 'transport.py')) == definitions['transport_sha256'],
                       'fixed reversible packet conversion and complete owner transport')
        adapter = load(repo / 'tests/current_runtime270.py', 'cat4_current270_owner_binding')
        inputs.bind(repo / 'tests/current_runtime270.py')
        inputs.bind(repo / 'tests/current_runtime_original35/binding.py')
        fixtures = load(repo / 'tests/current_runtime_fixtures.py', 'cat4_current_fixtures')
        inputs.bind(repo / 'tests/current_runtime_fixtures.py')
        package = fixtures.restore_current270(repo, out / 'original-owner-package', inputs)
        built, source_proof = adapter.make_bindings(package, build, args.expected_build_sha256, inputs)
        owner, owner_proof, owner_raw = sys.modules['prepare'].owner_module(
            built['source_sha256'], built['native_sha256'])
        (out / 'owner-current.py').write_bytes(owner_raw)
        inputs.bind(out / 'owner-current.py')
        dump(out / 'source-binding.json', dict(current=source_proof, owner=owner_proof,
                                              complete_original_owner_predicates_unchanged=True))
        report.update(source_sha256=built['source_sha256'], native_sha256=built['native_sha256'],
                      actual_build_sha256=args.expected_build_sha256, functions=built['functions'],
                      units=built['units_planned'])
        prepared = []
        for item in definitions['cases']:
            blobs = {}
            for suffix in ('input', 'output'):
                pin = item[suffix]
                path = repo / 'tests/fixtures' / (item['name'] + '.' + suffix + '.hex')
                raw = inputs.bind(path, pin['file_sha256'])
                packet.require(len(raw) == pin['file_bytes']
                               and stat.S_IMODE(path.stat().st_mode) == pin['mode'],
                               'fixed fixture file extent/mode changed')
                blobs[suffix] = bytes.fromhex(raw.decode('ascii'))
                packet.require(packet.sha(blobs[suffix]) == pin['decoded_sha256'],
                               'fixed complete packet bytes changed')
            body, inverse, commands, ledger = packet.convert_request(blobs['input'])
            packet.require(inverse['whole_original_packet_inverse_exact']
                           and not inverse['unpublished_composite_inputs'], 'complete original request inverse')
            packet.require(all(command[0] != 13 for command in commands), 'fixed raw-response comparison scope')
            packet.require(packet.response(blobs['output'], 1)[0] == item['legacy_status'],
                           'original expected status changed')
            prepared.append((item, blobs, body, inverse, commands, ledger))
        if not args.prepare_only:
            for ordinal, (item, blobs, body, inverse, commands, ledger) in enumerate(prepared):
                directory = out / ('%02d-' % ordinal + item['name'])
                directory.mkdir()
                row = dict(name=item['name'], status='FAIL', native_executed=False,
                           mode=item['mode'], original_status=item['legacy_status'],
                           original_failure_reference_reclassified=False)
                cases.append(row)
                save()
                try:
                    (directory / 'original-request.bin').write_bytes(blobs['input'])
                    (directory / 'original-expected.bin').write_bytes(blobs['output'])
                    (directory / 'converted-request.bin').write_bytes(body)
                    dump(directory / 'request-inverse.json', inverse)
                    actual, snapshot, actual_transport = transport.execute(
                        body, build / 'heptatorch', built['native_sha256'], directory / 'native', owner)
                    row['native_executed'] = actual_transport['native_executed']
                    row['transport'] = actual_transport
                    packet.require(actual_transport['status'] == 'PASS_ACTUAL_TRANSPORT',
                                   'actual image, owner snapshots, one response and both EOF streams required')
                    status, records = packet.response(actual, 3)
                    packet.require(status == 0, 'current source must complete this fixed packet successfully')
                    projected, projection = packet.project_success(actual, commands, ledger, snapshot)
                    (directory / 'projected-response.bin').write_bytes(projected)
                    dump(directory / 'response-inverse.json', projection)
                    packet.require(projection['complete_actual_response_inverse_exact'], 'complete actual response inverse')
                    if item['mode'] == 'original-success-exact':
                        packet.require(projected == blobs['output'], 'complete original successful response differs')
                        row.update(status='PASS_ORIGINAL_SUCCESS_BYTES', legacy_success_exact=True)
                    else:
                        packet.require(item['mode'] == 'expanded-capacity-control' and item['legacy_status'] == 3,
                                       'explicit independently named capacity control')
                        packet.require([command[0] for command in commands] == [1, 34]
                                       and len(records) == 2, 'fixed empty tensor and large cat request')
                        generation, slot = packet.typed(records[-1]['value'], 1)
                        tensor = snapshot['tables']['1'][slot - 1]
                        packet.require(tensor[0] == generation and tensor[1] == 1,
                                       'returned cat Tensor is live in the actual owner')
                        node = snapshot['nodes'][str(tensor[6])]
                        packet.require(len(node) == 84 and node[0] == 1 and node[1] == 0
                                       and node[4] == 1 and node[5] == 37 and node[12] == 0,
                                       'actual capacity result is the requested empty rank-one differentiable cat')
                        packet.require(snapshot['header']['pool'] > 1 << 20,
                                       'actual owner grew past the original bounded arena')
                        row.update(status='PASS_EXPANDED_CAPACITY_CONTROL', legacy_failure_qualified=False,
                                   actual_owner_pool_bytes=snapshot['header']['pool'], actual_tensor_descriptor=node)
                    row.update(actual_status=0, exact_request_inverse=True,
                               exact_actual_response_inverse=True, response_sha256=packet.sha(actual))
                except BaseException as caught:
                    row['error'] = dict(type=type(caught).__name__, message=str(caught),
                                        traceback=traceback.format_exc())
                dump(directory / 'case.json', row)
                save()
        good = args.prepare_only or (len(cases) == 4 and all(row['status'].startswith('PASS_') for row in cases))
        report['status'] = 'PREPARED_NOT_EXECUTED' if args.prepare_only else ('PASS_CAT4_ONLY' if good else 'FAIL')
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught), traceback=traceback.format_exc())
        report.update(status='FAIL', error=error)
    finally:
        changed = inputs.recheck()
        report.update(inputs=inputs.records, all_inputs_unchanged=not changed, input_recheck_errors=changed)
        if changed:
            report['status'] = 'FAIL'
        report['original_success_cases_qualified'] = sum(row.get('legacy_success_exact', False) for row in cases)
        report['capacity_controls_passed'] = sum(row['status'] == 'PASS_EXPANDED_CAPACITY_CONTROL' for row in cases)
        save()
    print(json.dumps({key: report.get(key) for key in ('status', 'source_sha256', 'native_sha256',
                    'original_success_cases_qualified', 'capacity_controls_passed', 'all_inputs_unchanged')}))
    return 0 if report['status'] in ('PREPARED_NOT_EXECUTED', 'PASS_CAT4_ONLY') else 1


if __name__ == '__main__':
    raise SystemExit(main())
