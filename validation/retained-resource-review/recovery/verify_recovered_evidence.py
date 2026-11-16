#!/usr/bin/env python3
"""Bind recovered complete bytes to the unchanged pre-outage reports."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def frames(raw, magic):
    offset = 0
    result = []
    while offset < len(raw):
        assert raw[offset:offset + 8] == magic and offset + 16 <= len(raw)
        size = struct.unpack_from('<Q', raw, offset + 8)[0]
        offset += 16
        assert size <= 1 << 20 and offset + size <= len(raw)
        result.append(raw[offset:offset + size])
        offset += size
    assert offset == len(raw)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recovered', 'validation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    reports = {}
    pins = {'resources.json': '52e9cad44f6027d26473f06f6e9c18a69616aa6285bc1c4e1eaeec824dccad1e',
            'typed-handoff.json': '57608933c7e2899903ce03d084540317ce0816b0e7301f83a66a79c25526cc90',
            'source-review.json': '7d417899c39c9a1574580ca888e6daae92722d723b6b6545dc1c32a1ae042363'}
    for name, expected in pins.items():
        raw = (a.validation / name).read_bytes()
        assert sha(raw) == expected
        reports[name] = json.loads(raw)
        assert reports[name]['status'] == 'PASS'
    units = []
    for label, prefix, source, image, image_pin in (
        ('old428', 'previous-combined-05',
         '1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae',
         'tensor-gradient-adam-distributions-session',
         '6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe'),
        ('retained446', 'retained-candidate-01',
         '666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b',
         'tensor-retained-session',
         '7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5'),
    ):
        directory = a.recovered / prefix
        assert sha((directory / 'runtime.hepta').read_bytes()) == source
        assert sha((directory / image).read_bytes()) == image_pin
        build = json.loads((directory / 'build.json').read_bytes())
        assert build['status'] == 'PASS' and build['source_sha256'] == source
        assert build['native_sha256'] == image_pin and build['module_count'] == len(build['units']) == 9
        for row in build['units']:
            assert sha((directory / (row['name'] + '.hepta')).read_bytes()) == row['source_sha256']
            assert sha((directory / (row['name'] + '.o')).read_bytes()) == row['object_sha256']
        units.append({'role': label, 'source_sha256': source, 'image_sha256': image_pin,
                      'module_count': 9, 'all_source_object_bytes_match_original_build': True})
    resources = reports['resources.json']
    raw_root = a.recovered / 'review-runs/native-01-first'
    assert sha((raw_root / 'report.json').read_bytes()) == pins['resources.json']
    assert sha((raw_root / 'executed-test.py').read_bytes()) == resources['test_sha256']
    case_rows = []
    total = 0
    for case in resources['cases']:
        directory = raw_root / case['name']
        requests = frames((directory / 'requests.frames.bin').read_bytes(), b'HTOWNR01')
        responses = frames((directory / 'responses.frames.bin').read_bytes(), b'HTOWNS01')
        transport = json.loads((directory / 'transport.json').read_bytes())
        assert len(requests) == len(responses) == case['frames'] == transport['frames']
        assert transport['exit_code'] == 0 and transport['responses_before_stdin_eof'] is True
        assert transport['executable_sha256'] == resources['native_sha256']
        assert len(case['frame_records']) == len(requests)
        for index, row in enumerate(case['frame_records']):
            assert row['frame'] == index + 1
            assert row['request_sha256'] == sha(requests[index])
            assert row['response_sha256'] == sha(responses[index])
        assert len(transport['rows']) == len(responses)
        for index, row in enumerate(transport['rows']):
            assert row['frame'] == index + 1 and row['stdin_open'] is True
            assert row['response_bytes'] == len(responses[index])
        total += len(requests)
        case_rows.append({'name': case['name'], 'frames': len(requests),
                          'all_original_frame_hashes_match': True,
                          'exit_code': transport['exit_code']})
    assert total == 320 and len(case_rows) == 8
    oom = raw_root / 'real-ENOMEM-and-recovery'
    assert (oom / 'before.owner.bin').read_bytes() == (oom / 'after.owner.bin').read_bytes()
    typed = reports['typed-handoff.json']
    typed_root = a.recovered / 'review-runs/typed-handoff-01-first'
    assert sha((typed_root / 'handoff-session').read_bytes()) == typed['qualification_image_sha256']
    assert sha((typed_root / 'controller.hepta').read_bytes()) == typed['controller_sha256']
    assert sha((typed_root / 'executed-test.py').read_bytes()) == typed['test_sha256']
    assert not (typed_root / 'transport').exists()
    source = reports['source-review.json']
    assert sha((a.recovered / 'review-runs/source-review-portable-01/report.json').read_bytes()) == pins['source-review.json']
    assert sha((a.recovered / 'peer/replay_retained_nonleaf.py').read_bytes()) == source['peer_runner_sha256']
    result = {'schema': 'heptatorch.retained-resource-recovered-bindings.v1',
              'status': 'PRESERVED_REPORT_BINDINGS_VERIFIED_WITH_TYPED_RAW_TAIL_LOSS',
              'native_executed_by_this_check': False, 'Torch_executed_by_this_check': False,
              'old_report_bytes_unchanged': pins, 'old_report_results_reclassified': False,
              'runtime_units': units, 'resource_cases': case_rows,
              'original_resource_frames_verified': total,
              'original_resource_owner_rollback_bytes_recorded': resources['whole_owner_rollback_bytes'],
              'OOM_before_after_owner_bytes_identical': True,
              'original_typed_handoff_image_verified': typed['qualification_image_sha256'],
              'original_typed_report_preserved': True, 'original_typed_raw_frames_available': False,
              'original_typed_raw_frames_reconstructed': False,
              'original_typed_frame_hashes_still_in_preserved_report': len(typed['frames']),
              'source_review_original_report_and_peer_runner_verified': True,
              'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'resource_frames_verified': total,
                      'report_sha256': sha(a.output.read_bytes())}))


if __name__ == '__main__':
    main()
