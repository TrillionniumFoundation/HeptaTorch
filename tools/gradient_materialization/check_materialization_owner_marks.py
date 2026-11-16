"""Replay fixed native traffic and inspect committed staging-field cleanup.

This is a native ownership/control check, not a new Torch numerical test. The
observer only reads its own child process. It never writes target memory or
replaces a numerical operation. Supplied trace directories must be from the
same exact executable being replayed.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import traceback


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def frames(raw, magic):
    result = []
    at = 0
    while at < len(raw):
        assert at + 16 <= len(raw) and raw[at:at+8] == magic
        size, = struct.unpack_from('<Q', raw, at + 8)
        assert 32 <= size <= 1 << 20 and at + 16 + size <= len(raw)
        result.append(raw[at+16:at+16+size])
        at += 16 + size
    assert at == len(raw)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'transport-observer', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--expected-source-sha256', required=True)
    p.add_argument('--trace', type=Path, action='append', required=True)
    a = p.parse_args()
    for name in ('binary', 'source', 'transport_observer', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    pins = {
        a.binary: a.expected_sha256,
        a.source: a.expected_source_sha256,
        a.transport_observer: 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b',
    }
    assert all(sha(path.read_bytes()) == pin for path, pin in pins.items())
    spec = importlib.util.spec_from_file_location('owner_marks_transport', a.transport_observer)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    a.output.mkdir(parents=True, exist_ok=False)
    archive = a.output / 'executed-sources'
    archive.mkdir()
    for path in (Path(__file__), a.transport_observer):
        (archive / path.name).write_bytes(path.read_bytes())
    rows = []
    for index, directory in enumerate(a.trace, 1):
        directory = directory.resolve()
        original_transport = json.loads((directory / 'transport.json').read_bytes())
        assert original_transport['executable_sha256'] == a.expected_sha256
        request_raw = (directory / 'requests.frames.bin').read_bytes()
        response_raw = (directory / 'responses.frames.bin').read_bytes()
        requests = frames(request_raw, b'HTOWNR01')
        responses = frames(response_raw, b'HTOWNS01')
        assert len(requests) == len(responses) == original_transport['frames']
        case_dir = a.output / ('case-' + str(index).zfill(2))
        session = transport.OwnerSession(a.binary, directory=case_dir, address_limit=64 << 20)
        row = {'case': index, 'status': 'FAIL', 'source_request_sha256': sha(request_raw),
               'source_response_sha256': sha(response_raw), 'source_transport_sha256': sha((directory / 'transport.json').read_bytes()),
               'frames': []}
        try:
            for number, (request, expected_response) in enumerate(zip(requests, responses), 1):
                actual = session.exchange(request)
                assert actual == expected_response, ('response bytes', index, number)
                state = session.owner()
                data = state['bytes']
                assert data and data[64:72] == b'HTSTATE5'
                u64 = lambda at: struct.unpack_from('<Q', data, at)[0]
                next_node = u64(8)
                slots = u64(139264 + 16)
                pool = u64(139264 + 24)
                assert 1 <= next_node <= 65536 and 4 <= slots <= 65536
                pool_base = 139264 + 144 + 424 * slots
                assert pool_base + pool == len(data)
                examined = 0
                for node in range(1, next_node):
                    page = u64(4096 + (node >> 8) * 8)
                    assert page >= 1 << 63
                    base = pool_base + (page & ((1 << 63) - 1)) + (node & 255) * 384
                    assert base + 384 <= len(data)
                    assert u64(base + 304) in (0, 1)
                    assert u64(base + 336) == 0, ('incoming flag leaked', index, number, node)
                    assert u64(base + 368) == 0, ('bitmap pointer leaked', index, number, node)
                    assert u64(base + 376) == 0, ('old work pointer leaked', index, number, node)
                    examined += 1
                row['frames'].append({'frame': number, 'state_sha256': sha(data), 'owner_bytes': len(data),
                    'live_owner_count': state['live'], 'nodes_inspected': examined,
                    'zero_ephemeral_words': examined * 3, 'response_equal_to_frozen_native_trace': True})
            session.finish()
            row['status'] = 'PASS'
        except Exception as error:
            row.update(exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            if session.poll() is None:
                os.kill(session.pid, signal.SIGKILL)
            try:
                session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup:
                row['cleanup_exception'] = repr(cleanup)
        (case_dir / 'report.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row)
    assert all(sha(path.read_bytes()) == pin for path, pin in pins.items())
    report = {
        'schema': 'heptatorch.gradient-materialization-committed-owner-control.v1',
        'status': 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
        'native_executable_sha256': a.expected_sha256,
        'native_source_sha256': a.expected_source_sha256,
        'test_sha256': sha(Path(__file__).read_bytes()),
        'transport_sha256': pins[a.transport_observer],
        'native_control_cases': len(rows),
        'frames': sum(len(row['frames']) for row in rows),
        'zero_ephemeral_word_observations': sum(f['zero_ephemeral_words'] for row in rows for f in row['frames']),
        'cases': rows,
        'scope': 'Exact same-binary trace replay plus read-only /proc observation after each complete frame. This checks committed owners and ephemeral descriptor fields, not a new numerical oracle.',
        'pinned_Torch_executed': False,
        'new_Torch_reference_cases': 0,
        'memory_written_by_observer': False,
        'full_rewrite_complete': False,
        'pytorch_api_denominator': 2590,
    }
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'native_control_cases', 'frames', 'zero_ephemeral_word_observations')}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
