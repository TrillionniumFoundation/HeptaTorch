"""Observe real kernel EINTR and positive short writes with live HIA owners.

The separately pinned qualification wrapper installs a SIGUSR1 observer. It
copies saved ucontext registers without modifying them; all 124 bytes of its
read/write loops are identical to production. No ptrace or fault emulation.
"""
from pathlib import Path
import argparse
import ctypes
import fcntl
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
spec = importlib.util.spec_from_file_location('owned_signal_client', HERE / 'client.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
assert (HERE / 'client.py').read_bytes() == CLIENT_SOURCE
sha = lambda data: hashlib.sha256(data).hexdigest()
TID = 0x101000001


class KernelStack(ctypes.Structure):
    _fields_ = [('pointer', ctypes.c_void_p), ('flags', ctypes.c_int), ('size', ctypes.c_ulong)]


class KernelUcontextPrefix(ctypes.Structure):
    _fields_ = [('flags', ctypes.c_ulong), ('link', ctypes.c_void_p),
        ('stack', KernelStack), ('gregs', ctypes.c_ulonglong * 23)]


def observation(session):
    count, rax, rip, rdx, rsi = struct.unpack('<QQQQQ', session.memory(session.base + 256, 40))
    return {'signal_count': count, 'saved_rax': rax - (1 << 64) if rax >> 63 else rax,
        'saved_rip': rip, 'saved_rdx': rdx, 'saved_rsi': rsi}


def wait_blocked(session, number, predicate=lambda values: True):
    until = time.monotonic() + 5
    while True:
        assert session.poll() is None
        raw = Path(f'/proc/{session.pid}/syscall').read_text().split()
        channel = Path(f'/proc/{session.pid}/wchan').read_text().strip()
        if raw and raw[0] not in ('running', '-1'):
            values = [int(value, 0) for value in raw]
            if values[0] == number and predicate(values) and channel in ('anon_pipe_read', 'anon_pipe_write'):
                return {'syscall': values, 'wait_channel': channel}
        assert time.monotonic() < until, ('native process did not block', number, raw, channel)
        time.sleep(.001)


def interrupt(session, number, predicate=lambda values: True):
    before = observation(session)
    blocked = wait_blocked(session, number, predicate)
    os.kill(session.pid, signal.SIGUSR1)
    until = time.monotonic() + 5
    while True:
        seen = observation(session)
        if seen['signal_count'] == before['signal_count'] + 1:
            after = wait_blocked(session, number)
            seen = observation(session)
            return {'blocked_before_signal': blocked, 'observed_kernel_ucontext': seen,
                'blocked_again_after_sigreturn': after}
        assert time.monotonic() < until, ('handler did not run', before, seen)
        time.sleep(.001)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', type=Path, required=True)
    parser.add_argument('--signal-probe', type=Path, required=True)
    parser.add_argument('--signal-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    service = args.service.resolve(); probe = args.signal_probe.resolve()
    native = {str(path): sha(path.read_bytes()) for path in (service, probe)}
    metadata_bytes = args.signal_metadata.read_bytes()
    metadata = json.loads(metadata_bytes)
    assert metadata['qualification_only'] and metadata['production_io_loop_bytes_identical']
    assert metadata['signal'] == 10 and metadata['sa_restart'] is False
    assert metadata['observer_controls'] == {'count': 256, 'saved_rax': 264, 'saved_rip': 272,
        'saved_rdx': 280, 'saved_rsi': 288, 'action': [320, 352]}
    assert ctypes.sizeof(KernelStack) == 24 and KernelUcontextPrefix.gregs.offset == 40
    native_header = Path('/usr/include/x86_64-linux-gnu/sys/ucontext.h')
    header = native_header.read_bytes()
    assert all(name in header for name in (b'REG_RAX', b'REG_RDX', b'REG_RIP', b'gregset_t __ctx(gregs)'))
    # Both profiles must contain byte-identical linked HIA object text; only
    # the separately admitted startup prefix can differ.
    assert service.read_bytes()[4096:] == probe.read_bytes()[4096:]
    (out / 'signal-metadata.json').write_bytes(metadata_bytes)
    rows = []; active = []

    def new(name):
        value = client.OwnerSession(probe, directory=out / name)
        active.append(value); value.control_base()
        return value

    def ok(value, commands, *, slots=0, pool=0):
        status, records = client.response(value.exchange(client.request(commands, slots, pool)))
        assert status == 0
        return records

    try:
        value = new('read-interruptions-with-owned-state')
        record, = ok(value, [(1, 16, 65)], slots=4, pool=64)
        assert record['values'][0] == TID
        state = value.owner()
        header_event = interrupt(value, 0, lambda values: values[1] == 0 and values[3] == 16)
        assert header_event['observed_kernel_ucontext']['saved_rax'] == -4
        assert value.owner() == state
        body = client.request([(11, TID, 0, 80)])
        frame = client.REQUEST_MAGIC + struct.pack('<Q', len(body)) + body
        value.send(frame[:26])
        body_event = interrupt(value, 0, lambda values: values[1] == 0 and
            values[2] == value.base + 4096 + 10 and values[3] == len(body) - 10)
        assert body_event['observed_kernel_ucontext']['saved_rax'] == -4
        pointer, length = struct.unpack('<QQ', value.memory(value.base + 32, 16))
        assert pointer == state['pointer'] and length == state['length']
        assert value.memory(pointer, length) == state['bytes']
        value.send(frame[26:])
        status, records = client.response(value.receive())
        assert status == 0 and len(records) == 1
        record, = ok(value, [(12, TID, 0)])
        assert record['values'][0] == 80
        value.finish()
        rows.append({'name': 'real-header-and-body-EINTR', 'pass': True,
            'events': [header_event, body_event], 'owned_state_unchanged_until_complete_frame': True,
            'updated_byte_after_complete_frame': 80, 'same_pid': value.pid})

        value = new('short-write-and-write-interruption')
        record, = ok(value, [(1, 16, 65)], slots=4, pool=1 << 18)
        assert record['values'][0] == TID
        before = value.owner()
        assert before['length'] > 1 << 18
        pipe_capacity = fcntl.fcntl(value.output, fcntl.F_GETPIPE_SZ)
        body = client.request([(18,)])
        value.send(client.REQUEST_MAGIC + struct.pack('<Q', len(body)) + body)
        short_event = interrupt(value, 1, lambda values: values[1] == 1 and values[3] > pipe_capacity)
        seen = short_event['observed_kernel_ucontext']
        assert 0 < seen['saved_rax'] < seen['saved_rdx'], short_event
        after = short_event['blocked_again_after_sigreturn']['syscall']
        assert after[2] == seen['saved_rsi'] + seen['saved_rax']
        assert after[3] == seen['saved_rdx'] - seen['saved_rax']
        eintr_event = interrupt(value, 1, lambda values: values[1] == 1 and values[2:4] == after[2:4])
        assert eintr_event['observed_kernel_ucontext']['saved_rax'] == -4
        assert eintr_event['blocked_again_after_sigreturn']['syscall'][2:4] == after[2:4]
        status, records = client.response(value.receive())
        assert status == 0 and len(records) == 1 and records[0]['op'] == 18
        assert records[0]['payload'] == before['bytes']
        after_state = value.owner()
        assert after_state['bytes'] == before['bytes'] and after_state['length'] == before['length']
        record, = ok(value, [(12, TID, 0)])
        assert record['values'][0] == 65
        value.finish()
        rows.append({'name': 'real-positive-short-write-and-EINTR', 'pass': True,
            'events': [short_event, eintr_event], 'pipe_capacity_bytes': pipe_capacity,
            'payload_bytes_compared': before['length'], 'same_pid': value.pid,
            'remaining_write_pointer_and_length_exact': True,
            'next_request_still_reads_original_owner_value': 65})
        for path, digest in native.items(): assert sha(Path(path).read_bytes()) == digest
        assert args.signal_metadata.read_bytes() == metadata_bytes
        assert (HERE / 'client.py').read_bytes() == CLIENT_SOURCE
        report = {'schema': 'heptatorch.owned-session-real-signals.v1', 'pass': True,
            'cases': len(rows), 'signals': 4, 'response_frames': sum(value.frames for value in active),
            'actual_kernel_EINTR_observations': 3, 'actual_positive_short_write_observations': 1,
            'test_sha256': sha(Path(__file__).read_bytes()), 'client_sha256': sha(CLIENT_SOURCE),
            'binary_sha256': native, 'signal_prefix_sha256': metadata['prefix_sha256'],
            'metadata_sha256': sha(metadata_bytes), 'ucontext_header_sha256': sha(header),
            'production_io_loop_bytes_identical': True,
            'linked_HIA_object_text_identical': True,
            'no_ptrace_or_syscall_emulation': True,
            'handler_only_observes_kernel_saved_registers': True,
            'signals_sent_only_at_verified_blocked_read_or_write_syscalls': True,
            'qualification_handler_not_used_during_HIA_calls': True,
            'all_responses_before_stdin_EOF': True, 'rows': rows}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        for value in active:
            if value.poll() is None:
                os.kill(value.pid, signal.SIGKILL)
                os.waitpid(value.pid, 0)


if __name__ == '__main__':
    main()
