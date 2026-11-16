"""Native replay of six frozen, actually executed pinned-PyTorch observations.

Numerical/exception expectations come from the original hosted report, not a
local formula or target result. The mutation is expressed as individual native
element writes; that is a deliberate wire difference from one PyTorch add_.
Only mutation invalidation, changed values, and backward results are compared.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import signal
import traceback
from review_native_objects import Review, sha, tensor_command, fbits
from review_object_boundaries import words


OBSERVATIONS_SHA = '43aadde8aef7f6406b6d0bd96f7b72fc724583e9bdcee823fa971ef8558f4d0c'
PROBE_SHA = '15c3f5f230a4cd30a42fe987b71078aa74c422b7e14b099e269bd80fede6c93a'


def flatten(value):
    if isinstance(value, list):
        return [x for part in value for x in flatten(part)]
    return [value]


def execute(review, observation):
    r = review.exchange
    before = observation['before']
    mode = observation['configuration']['mode']
    expected_a, expected_b = before['leaves']
    if mode == 3:
        base, = r([tensor_command([2, 3, 4], list(range(24)))], label='pre-transpose A')
        a, = r([(6, base['value'], 0, 1)], label='exact noncontiguous A layout')
    elif mode == 5:
        base, = r([tensor_command([3, 2], list(range(6)))], label='pre-transpose A')
        a, = r([(6, base['value'], 0, 1)], label='exact noncontiguous A layout')
    else:
        a, = r([tensor_command(expected_a['shape'], expected_a['values'], expected_a['requires_grad'])], label='A')
    a = a['value']
    b, = r([tensor_command(expected_b['shape'], expected_b['values'], expected_b['requires_grad'])], label='B')
    b = b['value']
    for name, handle, expected in [('A', a, expected_a), ('B', b, expected_b)]:
        meta, values = r([(13, handle), (10, handle)], label='original '+name)
        record = words(meta)
        rank = len(expected['shape'])
        review.eq(record[3:3+rank], tuple(expected['shape']), name+' original shape')
        review.eq(record[7:7+rank], tuple(expected['strides']), name+' original strides')
        review.bits(values, [fbits(v) for v in expected['values']], name+' original values')
    output, = r([(32, a, b)], label='matmul')
    output = output['value']
    values, = r([(10, output)], label='forward output')
    review.bits(values, [fbits(v) for v in before['output']['values']], 'real Torch forward')
    changed = b if mode == 1 else a
    changed_values = observation['after_changed']['values']
    r([(12, changed, i, fbits(value)) for i, value in enumerate(changed_values)],
      label='same changed values via explicit element writes')
    values, = r([(10, changed)], label='mutated operand')
    review.bits(values, [fbits(v) for v in changed_values], 'real Torch changed values')
    loss, = r([(4, 7, output)], label='sum for backward')
    status = 0 if observation['outcome'] == 'BACKWARD_SUCCEEDED' else 6
    r([(7, loss['value'])], status, 'actual pinned Torch backward outcome')
    if status == 0:
        for name, handle, expected in [('A', a, observation['gradients'][0]),
                                       ('B', b, observation['gradients'][1])]:
            if expected is not None:
                gradient, = r([(11, handle)], label='gradient '+name)
                review.bits(gradient, [fbits(v) for v in flatten(expected)], 'real Torch gradient '+name)
    audit, = r([(53,)], label='matmul replay audit')
    review.eq(audit['value'], 1, 'matmul replay audit')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--transport-observer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    native_sha = sha(args.binary.read_bytes())
    assert native_sha == args.expected_sha256
    assert sha(args.observations.read_bytes()) == OBSERVATIONS_SHA
    observations = json.loads(args.observations.read_text())
    assert observations['status'] == 'OBSERVED' and observations['cases'] == 22
    assert observations['probe_sha256'] == PROBE_SHA
    assert observations['oracle_provenance']['status'] == 'PASS'
    assert len(observations['oracle_provenance']['checks']) == 11 and all(observations['oracle_provenance']['checks'].values())
    chosen = [row for row in observations['rows'] if row['case'].startswith('matmul-')]
    assert len(chosen) == 6
    spec = importlib.util.spec_from_file_location('matmul_observation_transport', args.transport_observer)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    root = Path(__file__).resolve().parent
    sources = [Path(__file__), root/'review_native_objects.py', root/'review_object_boundaries.py',
               root/'reference_model.py', args.transport_observer, args.observations]
    hashes = {path.name: sha(path.read_bytes()) for path in sources}
    archive = args.output/'executed-sources'
    archive.mkdir()
    for path in sources:
        (archive/path.name).write_bytes(path.read_bytes())
    rows = []
    for index, case in enumerate(chosen):
        directory = args.output/f'case-{index+1:02d}'
        session = transport.OwnerSession(args.binary, directory=directory, address_limit=64 << 20)
        review = Review(session)
        result = {'case': case['case'], 'status': 'FAIL',
                  'actual_hosted_outcome': case['observation']['outcome'],
                  'actual_hosted_exception': case['observation']['exception']}
        try:
            execute(review, case['observation'])
            session.finish()
            result['status'] = 'PASS'
        except Exception as exc:
            result.update(error=repr(exc), traceback=traceback.format_exc())
            if session.poll() is None:
                os.kill(session.pid, signal.SIGKILL)
            try:
                session.finish(expected=-signal.SIGKILL)
            except Exception:
                pass
        result.update(frames=review.frames, exact_scalar_or_binary32_fields=review.compared_fields,
                      entire_owner_rollback_bytes=review.rollback_bytes)
        (directory/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        rows.append(result)
    assert native_sha == sha(args.binary.read_bytes())
    assert hashes == {path.name: sha(path.read_bytes()) for path in sources}
    report = {'schema': 'heptatorch.native-replay-hosted-matmul-lifetimes.v1',
              'status': 'PASS' if all(row['status']=='PASS' for row in rows) else 'FAIL',
              'native_executable_sha256': native_sha, 'source_hashes': hashes,
              'hosted_observations_sha256': OBSERVATIONS_SHA,
              'source_of_reference': 'Previously executed exact pinned Torch report; all 11 provenance checks retained.',
              'pinned_pytorch_executed_during_this_replay': False, 'native_target_executed': True,
              'mutation_interface': 'N native single-element set commands reproduce one Torch add_ value change; absolute mutation counts are not compared.',
              'cases': len(rows), 'passed': sum(row['status']=='PASS' for row in rows), 'rows': rows,
              'frames': sum(len(row['frames']) for row in rows),
              'exact_scalar_or_binary32_fields': sum(row['exact_scalar_or_binary32_fields'] for row in rows),
              'entire_owner_rollback_bytes': sum(row['entire_owner_rollback_bytes'] for row in rows)}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['status','cases','passed','frames','exact_scalar_or_binary32_fields','entire_owner_rollback_bytes']}))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
