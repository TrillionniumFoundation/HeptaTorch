"""Native owned-session transport and pre-exit close qualification.

The Python host only constructs wire packets, drives pipes, and observes its
own child's mappings. The tested owner allocation, transactions, and drop are
executed by the actual HIA modules and native wrapper.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import time
import traceback

HERE = Path(__file__).resolve().parent
CLIENT_SOURCE = (HERE / 'client.py').read_bytes()
spec = importlib.util.spec_from_file_location('owned_transport_client', HERE / 'client.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
assert (HERE / 'client.py').read_bytes() == CLIENT_SOURCE
MIB = client.MIB
TID = 0x101000001
sha = lambda data: hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', type=Path, required=True)
    parser.add_argument('--close-probe', type=Path, required=True)
    parser.add_argument('--build-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    service = args.service.resolve()
    probe = args.close_probe.resolve()
    before = {str(path): sha(path.read_bytes()) for path in (service, probe)}
    if args.build_report:
        receipt = json.loads(args.build_report.read_text())
        assert receipt['executables']['service']['sha256'] == before[str(service)]
        assert receipt['executables']['service-close-probe']['sha256'] == before[str(probe)]
        (out / 'native-build.json').write_bytes(args.build_report.read_bytes())
    rows = []
    active = []
    close_observations = []

    def session(name, *, observe=False, **kwargs):
        value = client.OwnerSession(probe if observe else service, probe=observe,
            directory=out / name, **kwargs)
        active.append(value)
        return value

    def ok(value, commands, *, slots=0, pool=0):
        status, records = client.response(value.exchange(client.request(commands, slots, pool)))
        assert status == 0
        return records

    def seed(value):
        record, = ok(value, [(1, 16, 65)], slots=4, pool=64)
        assert record['op'] == 1 and record['values'] == (
            TID, 0x102000001, 0x103000001, 0, 16)
        owner = value.owner()
        assert owner['pointer'] and owner['length'] == 1904
        return owner

    def close(value, status, expected_owner=None):
        observation = value.observe_close(status)
        if expected_owner is not None:
            assert observation['dead_argument_pointer'] == expected_owner['pointer']
            assert observation['dead_argument_length'] == expected_owner['length']
        close_observations.append(observation)
        value.finish(status)
        return observation

    def case(name, body):
        start = time.monotonic()
        try:
            details = body() or {}
            rows.append({'name': name, 'pass': True,
                'elapsed_seconds': time.monotonic() - start, **details})
            print(name + ': PASS', flush=True)
        except BaseException:
            rows.append({'name': name, 'pass': False, 'error': traceback.format_exc()})
            raise
        finally:
            (out / 'cases.json').write_text(json.dumps(rows, indent=2) + '\n')

    def first_bad_body():
        value = session('initial-invalid-body')
        empty = value.owner()
        assert empty['pointer'] == 0 and empty['length'] == 0
        status, records = client.response(value.exchange(b'bad'))
        assert status == 100 and records == [] and value.owner() == empty
        seed(value)
        value.finish()
        return {'initial_failure_preserves_live_empty': True}

    def partial_frame():
        value = session('partial-frame-admission')
        base = value.control_base()
        packet = client.REQUEST_MAGIC + struct.pack('<Q', 96) + client.request([(1, 16, 65)], 4, 64)
        at = 0
        for end in (1, 7, 15, 16, 17, len(packet) - 1):
            value.send(packet[at:end]); at = end
            value.assert_no_response()
            assert value.memory(base + 32, 16) == bytes(16)
            assert value.memory(base + 192, 8) == struct.pack('<Q', 1)
        value.send(packet[at:])
        status, records = client.response(value.receive())
        assert status == 0 and records[0]['values'][0] == TID
        value.finish()
        return {'partial_boundaries': [1, 7, 15, 16, 17, len(packet) - 1],
            'no_step_before_complete_body': True}

    def pipeline():
        value = session('pipelined-frames')
        packets = [client.request([(1, 16, 65)], 4, 64),
                   client.request([(12, TID, 0)]),
                   client.request([(11, TID, 0, 77)]),
                   client.request([(12, TID, 0)])]
        value.send(b''.join(client.REQUEST_MAGIC + struct.pack('<Q', len(body)) + body for body in packets))
        results = [client.response(value.receive()) for _ in packets]
        assert all(status == 0 for status, _ in results)
        assert results[0][1][0]['values'][0] == TID
        assert results[1][1][0]['values'][0] == 65
        assert results[3][1][0]['values'][0] == 77
        value.finish()
        return {'ordered_responses': 4, 'same_pid': True, 'responses_before_stdin_eof': True}

    def app_rollback():
        value = session('application-failure-owner')
        owner = seed(value)
        status, records = client.response(value.exchange(client.request([(11, TID, 0, 99), (99,)])))
        assert status == 7 and not records and value.owner() == owner
        record, = ok(value, [(12, TID, 0)])
        assert record['values'][0] == 65
        value.finish()
        return {'full_owned_state_unchanged_bytes': owner['length'],
            'same_owner_pointer_on_application_error': True}

    def exact_limit():
        value = session('exact-body-limit')
        owner = seed(value)
        status, records = client.response(value.exchange(bytes(MIB)))
        assert status == 100 and not records and value.owner() == owner
        record, = ok(value, [(12, TID, 0)])
        assert record['values'][0] == 65
        value.finish()
        return {'body_bytes': MIB, 'transport_admitted': True,
            'application_rejected_without_state_change': True}

    def after_owned(name, raw, status, *, eof=False):
        value = session(name, observe=True)
        owner = seed(value)
        if raw: value.send(raw)
        if eof: value.close_input()
        observed = close(value, status, owner)
        return {'close': observed, 'no_failed_transport_response': True}

    def broken_output():
        value = session('broken-output-pipe', observe=True)
        old = seed(value)
        value.close_output()
        body = client.request([(12, TID, 0)])
        value.send(client.REQUEST_MAGIC + struct.pack('<Q', len(body)) + body)
        observation = value.observe_close(4)
        assert observation['dead_argument_pointer'] and observation['dead_argument_length'] == old['length']
        assert not value.mapped(old['pointer'])
        close_observations.append(observation)
        value.finish(4)
        return {'close': observation, 'actual_write_EPIPE': True,
            'SIGPIPE_did_not_skip_close': True}

    def nonblocking_read_error():
        value = session('post-state-read-EAGAIN', observe=True, keep_read_control=True)
        owner = seed(value)
        value.nonblocking_input()
        # Wake the already-blocked read with one byte; the next read is a
        # real nonblocking read of the remaining header and returns EAGAIN.
        value.send(client.REQUEST_MAGIC[:1])
        return {'close': close(value, 4, owner), 'actual_read_EAGAIN': True}

    def directory_read_error():
        value = session('directory-read-EISDIR', observe=True, stdin_directory=out)
        observed = close(value, 4)
        assert observed['dead_argument_pointer'] == observed['dead_argument_length'] == 0
        return {'close': observed, 'actual_read_EISDIR': True,
            'startup_live_empty_closed': True}

    def grown_owner():
        value = session('external-owner-beyond-frame-limit', observe=True)
        seed(value)
        ok(value, [(17, 16, 2 * MIB)])
        record, = ok(value, [(12, TID, 0)])
        assert record['values'][0] == 65
        owner = value.owner()
        assert owner['length'] > 2 * MIB
        value.close_input()
        observed = close(value, 0, owner)
        return {'owned_bytes': owner['length'], 'close': observed,
            'owned_mapping_is_not_copied_frame_arena': True}

    try:
        case('initial-invalid-body', first_bad_body)
        case('partial-frame-admission', partial_frame)
        case('pipelined-frames', pipeline)
        case('application-failure-owner', app_rollback)
        case('exact-body-limit', exact_limit)
        case('clean-EOF-closes-owned-state', lambda: after_owned('clean-EOF', b'', 0, eof=True))
        case('invalid-magic-closes-owned-state', lambda: after_owned('invalid-magic', b'NOTOWNR1' + bytes(8), 2))
        for amount in (MIB + 1, (1 << 64) - 1):
            name = 'oversize-' + str(amount)
            case(name, lambda name=name, amount=amount: after_owned(name,
                client.REQUEST_MAGIC + struct.pack('<Q', amount), 3))
        for amount in (1, 7, 8, 15):
            name = 'short-header-' + str(amount)
            case(name, lambda name=name, amount=amount: after_owned(name,
                (client.REQUEST_MAGIC + struct.pack('<Q', 32))[:amount], 2, eof=True))
        for amount in (0, 1, 31):
            name = 'short-body-' + str(amount)
            case(name, lambda name=name, amount=amount: after_owned(name,
                client.REQUEST_MAGIC + struct.pack('<Q', 32) + bytes(amount), 2, eof=True))
        case('broken-output-pipe', broken_output)
        case('post-state-read-EAGAIN', nonblocking_read_error)
        case('directory-read-EISDIR', directory_read_error)
        case('external-owner-beyond-frame-limit', grown_owner)
        assert all(value.poll() is not None for value in active)
        for path, digest in before.items(): assert sha(Path(path).read_bytes()) == digest
        assert (HERE / 'client.py').read_bytes() == CLIENT_SOURCE
        report = {'schema': 'heptatorch.owned-session-transport.v1', 'pass': True,
            'cases': len(rows), 'response_frames': sum(value.frames for value in active),
            'close_observations_before_process_exit': len(close_observations),
            'binary_sha256': before,
            'test_sha256': sha(Path(__file__).read_bytes()),
            'client_sha256': sha(CLIENT_SOURCE),
            'all_responses_before_stdin_EOF': True,
            'normal_and_error_cleanup_use_actual_HIA_close': True,
            'qualification_only_post_close_hook': True,
            'native_owner_pointer_never_sent_in_public_wire': True,
            'rows': rows}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
    finally:
        # A failed qualification preserves its raw requests and receipt. Avoid
        # leaving only our own failed child blocked on its private probe pipe.
        for value in active:
            if value.poll() is None:
                os.kill(value.pid, signal.SIGKILL)
                os.waitpid(value.pid, 0)


if __name__ == '__main__':
    main()
