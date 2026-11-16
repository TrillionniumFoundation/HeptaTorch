#!/usr/bin/env python3
"""Capture native observations for the frozen extreme-stride probe plan.

No predicted Torch output is installed here. A later comparison must use the
actual pinned reference report, including its metadata and admission result.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


SHA = lambda raw: hashlib.sha256(raw).hexdigest()
PROBE_SHA = '568ece8996db71f5d1a16b6dfb71efd963a258f457e73cfdc738f7d60e226f89'
WIRE_SHA = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT_SHA = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'protocol', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-source', 'expected-image'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    binary = build / 'tensor-gradient-adam-distributions-session'
    source = build / 'runtime.hepta'
    custody = {path.resolve(): path.read_bytes() for path in
               (Path(__file__), binary, source, args.protocol,
                args.protocol.with_name('owner_client.py'), args.prepared)}
    assert SHA(custody[source]) == args.expected_source
    assert SHA(custody[binary]) == args.expected_image
    assert SHA(custody[args.protocol.resolve()]) == WIRE_SHA
    assert SHA(custody[args.protocol.with_name('owner_client.py').resolve()]) == CLIENT_SHA
    plan = json.loads(custody[args.prepared.resolve()])
    assert plan['probe_sha256'] == PROBE_SHA and plan['status'] == 'PREPARED_NOT_RUN'
    assert plan['expected_cases'] == len(plan['case_plan']) == 8
    spec = importlib.util.spec_from_file_location('extreme_stride_wire', args.protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'executed-test.py').write_bytes(custody[Path(__file__).resolve()])
    report = {'schema': 'heptatorch.native-extreme-stride-observations.v1',
              'status': 'RUNNING', 'source_sha256': args.expected_source,
              'executable_sha256': args.expected_image, 'reference_probe_sha256': PROBE_SHA,
              'test_sha256': SHA(custody[Path(__file__).resolve()]),
              'actual_torch_compared': False, 'torch_executed_here': False,
              'native_executed': True, 'numerical_qualification_claim': False,
              'rows': [], 'full_rewrite_complete': False}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    save()
    for case in plan['case_plan']:
        session = wire.Session(binary, directory=out / case['case'] / 'transport')
        one = lambda *command: session.run([command])[0]
        base = one(*wire.tensor((6,), [1, 2, 3, 4, 5, 6]))['value']
        other = one(*wire.tensor((1, 2, 3), [10, 11, 12, 13, 14, 15]))['value']
        config = wire.Payload(struct.pack('<10Q', 3, 1, 2, 3, 0, case['stride'], 1, 2, 0, 0))
        owner_before = session.owner()
        status, records = wire.parse(session.exchange(wire.packet([(69, base, config, 80)])))
        row = {**case, 'view_accepted': status == 0, 'view_status': status}

        def info(handle):
            meta, data = session.run([(13, handle), (39, handle, 0, 6)])
            value = wire.meta(meta)
            return {'shape': value['shape'][:value['rank']],
                    'strides': value['strides'][:value['rank']],
                    'storage_offset_bytes': value['storage_offset'],
                    'storage_bytes': value['storage_bytes'], 'version': value['version'],
                    'requires_grad': bool(value['requires_grad']),
                    'binary32_words': list(struct.unpack('<6I', data['payload'])),
                    'tensor_identity': handle, 'storage_identity': value['storage_id'],
                    'version_identity': value['version_id']}

        if status:
            assert session.owner() == owner_before
        else:
            assert len(records) == 1
            view = records[0]['value']
            left, right = (other, view) if case['reverse_inputs'] else (view, other)
            before_left, before_right = info(left), info(right)
            owner_before = session.owner()
            status, records = wire.parse(session.exchange(wire.packet([(2, 1, left, right)])))
            row.update(left_before=before_left, right_before=before_right,
                       operation_accepted=status == 0, operation_status=status)
            if status:
                assert session.owner() == owner_before
            else:
                assert len(records) == 1
                output = records[0]['value']
                row.update(left_after=info(left), right_after=info(right), output=info(output))
                row['checks'] = {
                    'left_unchanged': row['left_before'] == row['left_after'],
                    'right_unchanged': row['right_before'] == row['right_after'],
                    'fresh_output_tensor': output not in (left, right),
                    'fresh_output_storage': row['output']['storage_identity'] not in
                        (before_left['storage_identity'], before_right['storage_identity']),
                    'same_output_shape': row['output']['shape'] == [1, 2, 3],
                    'new_output_version_zero': row['output']['version'] == 0}
                assert all(row['checks'].values())
        row.update(audit=one(53)['value'], frames=session.frames, status='OBSERVED')
        assert row['audit'] == 1
        session.finish()
        report['rows'].append(row)
        save()
    for path, raw in custody.items():
        assert path.read_bytes() == raw, path
    report.update(status='OBSERVED_NOT_TORCH_QUALIFIED', cases=len(report['rows']),
                  frames=sum(row['frames'] for row in report['rows']), input_custody_after=True,
                  input_sha256={str(path): SHA(raw) for path, raw in custody.items()})
    save()
    print(json.dumps({'status': report['status'], 'cases': report['cases'],
                      'frames': report['frames'], 'report_sha256': SHA((out / 'report.json').read_bytes())}))


if __name__ == '__main__':
    main()
