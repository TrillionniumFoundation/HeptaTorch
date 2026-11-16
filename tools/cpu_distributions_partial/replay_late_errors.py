#!/usr/bin/env python3
"""Replay all four immutable late-error observations through explicit commands86/87.

The predecessor atomic commands are also executed unchanged. Original observed
probabilities, complete states, Tensor words and Versions are never replaced.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback

SHA = lambda data: hashlib.sha256(data).hexdigest()
REFERENCE = 'c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'protocol', 'executable', 'build-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {p.resolve(): p.read_bytes() for p in (a.reference, a.protocol, a.executable, a.build_report, Path(__file__))}
    assert SHA(inputs[a.reference.resolve()]) == REFERENCE
    assert SHA(inputs[a.protocol.resolve()]) == PROTOCOL
    source = json.loads(inputs[a.reference.resolve()])
    build = json.loads(inputs[a.build_report.resolve()])
    assert build['status'] == 'PASS' and SHA(inputs[a.executable.resolve()]) == build['executable_sha256']
    assert source['cases'] == 183 and len(source['oracle_provenance']['checks']) == 11
    assert all(source['oracle_provenance']['checks'].values())
    selected = [(i, row) for i, row in enumerate(source['rows']) if not row['accepted'] and not row['state_unchanged']]
    assert [i for i, _ in selected] == [101, 103, 105, 179]
    states = {}
    for digest, row in source['states'].items():
        raw = bytes.fromhex(row['hex'])
        assert len(raw) == 5056 and SHA(raw) == digest
        states[digest] = raw
    spec = importlib.util.spec_from_file_location('distribution_partial_wire', a.protocol.resolve())
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    report = {'schema': 'heptatorch.cpu-distributions-late-error-actual-replay.v1', 'status': 'RUNNING',
              'reference_sha256': REFERENCE, 'executable_sha256': build['executable_sha256'],
              'runtime_source_sha256': build['runtime_source_sha256'], 'cases': [], 'frames_before_EOF': 0,
              'exact_words': 0, 'complete_state_comparisons': 0, 'rollback_bytes': 0,
              'all_rng_complete': False, 'all_tensor_dtypes_complete': False, 'full_pytorch_rewrite_complete': False}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def tensor(snapshot):
        shape = snapshot['shape']
        return (1, len(shape), int(snapshot['requires_grad']), *shape, *([0] * (4 - len(shape))),
                wire.Payload(struct.pack('<' + 'I' * len(snapshot['words']), *snapshot['words']), 4))

    def get_state(session, generator, expected):
        actual = session.run([(63, generator)])[0]['payload']
        assert actual == expected
        report['complete_state_comparisons'] += 1
        return actual

    for index, observation in selected:
        row = {'reference_index': index, 'name': observation['name'], 'status': 'RUNNING'}
        report['cases'].append(row)
        session = wire.Session(a.executable.resolve(), directory=out / ('case-' + str(index)))
        try:
            x, generator = [r['value'] for r in session.run([tensor(observation['input_before']), (60, 0)])]
            session.run([(64, generator, wire.Payload(states[observation['state_before']]), 5056)])
            if observation['family'] == 'tensor-invalid-and-empty':
                probability = session.run([tensor(observation['probabilities'])])[0]['value']
                partial = (86, x, generator, probability, 1)
                atomic = (81, x, generator, probability, 1)
                error, writes, consumed = 7, 2, 2
            else:
                partial = (87, x, generator, observation['probability_double_bits'], 1, 0)
                atomic = (84, x, generator, observation['probability_double_bits'], 1, 0)
                error, writes, consumed = 5, 0, 8
            # Old atomic operation keeps its original transaction contract.
            report['rollback_bytes'] += session.error([atomic, (61, generator, 77)], error)
            get_state(session, generator, states[observation['state_before']])
            before_stats = struct.unpack('<12Q', session.run([(54,)])[0]['payload'])
            status, records = wire.parse(session.exchange(wire.packet(
                [(60, 10000 + index), partial, (61, generator, 77), (80, x, generator, 0, 1)])))
            assert status == 0 and len(records) == 2 and records[1]['opcode'] == partial[0]
            assert records[1]['value'] == error
            payload = struct.unpack('<4Q', records[1]['payload'])
            assert payload == (error, x, writes, consumed), payload
            prefix = records[0]['value']
            assert session.run([(62, prefix)])[0]['value'] == 10000 + index
            actual = list(struct.unpack('<' + 'I' * len(observation['input_after']['words']),
                                        session.run([(39, x, 0, len(observation['input_after']['words']))])[0]['payload']))
            assert actual == observation['input_after']['words']
            assert actual == observation['physical_after']
            report['exact_words'] += len(actual)
            metadata = wire.meta(session.run([(13, x)])[0])
            assert metadata['version'] == observation['input_after']['version'] == 0
            assert metadata['shape'][:metadata['rank']] == observation['input_after']['shape']
            assert metadata['strides'][:metadata['rank']] == observation['input_after']['strides']
            assert metadata['storage_offset'] == 0
            if observation['input_after']['requires_grad']:
                assert session.run([(59, x)])[0]['value'] == 0
            get_state(session, generator, states[observation['state_after']])
            session.run([(53,), (66, prefix)])
            after_stats = struct.unpack('<12Q', session.run([(54,)])[0]['payload'])
            assert before_stats[5:11] == after_stats[5:11], (before_stats, after_stats)
            assert session.owner()['bytes'][64:72] == b'HTDIST02'
            # Success followed by an ordinary error remains one atomic frame.
            if partial[0] == 86:
                good = session.run([wire.full((4,), .5)])[0]['value']
                success = (86, x, generator, good, 1)
            else:
                success = (87, x, generator, observation['probability_double_bits'], 1, 1)
            report['rollback_bytes'] += session.error([success, (999,)], 4)
            get_state(session, generator, states[observation['state_after']])
            # The returned owner remains usable, including nullable gradients
            # and an independent Generator clone after the failed operation.
            control = session.run([(65, generator)])[0]['value']
            session.run([(80, x, generator, 0x3ff0000000000000, 1), (67, control, 8)])
            assert session.run([(63, generator)])[0]['payload'] == session.run([(63, control)])[0]['payload']
            report['complete_state_comparisons'] += 1
            assert session.run([(39, x, 0, 4)])[0]['payload'] == struct.pack('<4I', *([1065353216] * 4))
            report['exact_words'] += 4
            assert wire.meta(session.run([(13, x)])[0])['version'] == 1
            session.run([(53,)])
            session.finish()
            row.update(status='PASS', atomic_opcode=atomic[0], explicit_partial_opcode=partial[0],
                       error=error, completed_writes=writes, consumed_raw32=consumed,
                       Version_after_error=0, prefix_commit_suffix_stop=True,
                       original_reference_words_and_complete_state_exact=True,
                       old_atomic_contract_unchanged=True, temporary_mask_live_rows_reclaimed=True,
                       frames=session.frames)
            report['frames_before_EOF'] += session.frames
        except BaseException as failure:
            row.update(status='FAIL', error=repr(failure), traceback=traceback.format_exc())
            try:
                session.finish()
            except BaseException as cleanup:
                row['cleanup_error'] = repr(cleanup)
            report['status'] = 'FAIL'
            save()
            raise
        save()
    for path, data in inputs.items():
        assert path.read_bytes() == data, path
    report.update(status='PASS', case_count=len(selected), input_custody_after=True,
                  inputs={str(p): SHA(data) for p, data in inputs.items()})
    save()
    print(json.dumps({key: report[key] for key in ('status', 'case_count', 'frames_before_EOF', 'exact_words',
                                                 'complete_state_comparisons', 'rollback_bytes')}, indent=2))


if __name__ == '__main__':
    main()
