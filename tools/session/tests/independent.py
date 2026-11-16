"""Independent wire/transaction review; every operation executes native HIA."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import select
import struct
import subprocess
import time

MIB = 1 << 20
REQUEST = b'HTOSES01'
RESPONSE = b'HTOSRP01'
BODY = b'HTORCH01'
RESULT = b'HTORSP01'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]

def command(op, *args):
    assert len(args) <= 7
    return [op, *args, *([0] * (7 - len(args)))]

def packet(commands, payload=b''):
    return (struct.pack('<8sQQQ', BODY, len(commands), 32+64*len(commands)+len(payload), 0)
            + b''.join(struct.pack('<8Q', *row) for row in commands) + payload)

def tensor(values, requires=False):
    return packet([command(1, 1, int(requires), len(values), 0, 0, 0, 96)],
                  struct.pack('<' + str(len(values)) + 'f', *values))

def framed(body):
    return REQUEST + struct.pack('<Q', len(body)) + body

def parse(raw):
    assert len(raw) >= 32
    magic, version, count, status = struct.unpack_from('<8sQQQ', raw)
    assert magic == RESULT and version == 1
    offset, rows = 32, []
    for _ in range(count):
        op, error, value, length = struct.unpack_from('<4Q', raw, offset)
        offset += 32
        data = raw[offset:offset+length]
        assert len(data) == length
        rows.append((op, error, value, data))
        offset += length
    assert offset == len(raw)
    if status:
        assert count == 0 and raw == struct.pack('<8sQQQ', RESULT, 1, 0, status)
    return status, rows

class Client:
    def __init__(self, executable):
        self.child = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      bufsize=0, env={'PATH': ''})
        self.frames = 0

    def send(self, data):
        view = memoryview(data)
        while view:
            count = os.write(self.child.stdin.fileno(), view)
            assert count > 0
            view = view[count:]

    def read(self, size):
        parts, received = [], 0
        deadline = time.monotonic() + 20
        while received < size:
            remaining = deadline - time.monotonic()
            assert remaining > 0 and select.select([self.child.stdout], [], [], remaining)[0], ('response timeout', size, received)
            part = os.read(self.child.stdout.fileno(), size-received)
            assert part, ('premature EOF', self.child.poll(), size, received)
            parts.append(part)
            received += len(part)
        return b''.join(parts)

    def receive(self):
        assert not self.child.stdin.closed
        magic, length = struct.unpack('<8sQ', self.read(16))
        assert magic == RESPONSE and 32 <= length <= MIB
        result = self.read(length)
        self.frames += 1
        assert self.child.poll() is None, 'process exited before input EOF'
        return result

    def exchange(self, body, status=0):
        self.send(framed(body))
        raw = self.receive()
        observed, rows = parse(raw)
        assert observed == status, (observed, status)
        return rows, raw

    def workspace(self):
        # Observe only this review's own child and synthetic tensor data.
        entries = Path(f'/proc/{self.child.pid}/maps').read_text().splitlines()
        candidates = []
        for line in entries:
            fields = line.split()
            begin, end = [int(x, 16) for x in fields[0].split('-')]
            if fields[1] == 'rw-p' and end-begin == 4096+4*MIB and len(fields) == 5:
                candidates.append(begin)
        assert len(candidates) == 1, candidates
        base = candidates[0]
        fd = os.open(f'/proc/{self.child.pid}/mem', os.O_RDONLY)
        try:
            pointer = struct.unpack('<Q', os.pread(fd, 8, base+32))[0]
            assert pointer in (base+4096+2*MIB, base+4096+3*MIB)
            data = os.pread(fd, MIB, pointer)
            assert len(data) == MIB
            return data
        finally:
            os.close(fd)

    def close(self):
        if not self.child.stdin.closed:
            self.child.stdin.close()
        status = self.child.wait(timeout=10)
        stdout = self.child.stdout.read()
        stderr = self.child.stderr.read()
        assert status == 0 and stdout == stderr == b'', (status, stdout[:32], stderr[:128])

    def kill(self):
        if self.child.poll() is None:
            self.child.kill()
        self.child.wait(timeout=10)

def run(executable, fixtures, output):
    output.mkdir(exist_ok=False, parents=True)
    executable_before = executable.read_bytes()
    rows = []
    scalar = packet([command(14, 0, bits(1.), bits(2.))])
    scalar_response = (struct.pack('<8sQQQ', RESULT, 1, 1, 0)
                       + struct.pack('<4Q', 14, 0, bits(3.), 0))

    # Existing expected bytes come from independently qualified repository data.
    # A failed session transaction publishes only its error header by contract.
    for request in sorted(fixtures.glob('*.input.hex')):
        body = bytes.fromhex(request.read_text())
        expected = bytes.fromhex(request.with_name(request.name.replace('.input.hex', '.output.hex')).read_text())
        expected_status = struct.unpack_from('<Q', expected, 24)[0]
        if expected_status:
            expected = struct.pack('<8sQQQ', RESULT, 1, 0, expected_status)
        client = Client(executable)
        try:
            _, raw = client.exchange(body, expected_status)
            assert raw == expected, request.name
            _, next_raw = client.exchange(scalar)
            assert next_raw == scalar_response, request.name
            client.close()
        finally:
            client.kill()
        rows.append({'case': request.name[:-10], 'kind': 'fixture-interoperation',
                     'status': 'PASS', 'input_sha256': sha(body), 'response_sha256': sha(raw),
                     'original_packet_status': expected_status, 'interactive_frames': 2})

    malformed = [('empty-stream', b'', 0)]
    header = REQUEST + struct.pack('<Q', len(scalar))
    for length in (1, 7, 8, 15):
        malformed.append((f'short-header-{length}', header[:length], 2))
    for length in (0, 1, 31, len(scalar)-1):
        malformed.append((f'short-body-{length}', header+scalar[:length], 2))
    malformed.extend([
        ('invalid-frame-magic', b'BADFRAME'+struct.pack('<Q', 0), 2),
        ('oversized-frame', REQUEST+struct.pack('<Q', MIB+1), 3),
        ('u64-maximum-frame', REQUEST+struct.pack('<Q', (1 << 64)-1), 3),
    ])
    for name, raw, expected_exit in malformed:
        result = subprocess.run([str(executable)], input=raw, capture_output=True,
                                timeout=10, env={'PATH': ''})
        assert result.returncode == expected_exit and result.stdout == result.stderr == b'', (name, result)
        rows.append({'case': name, 'kind': 'transport-refusal', 'status': 'PASS',
                     'exit_status': result.returncode, 'input_sha256': sha(raw)})

    client = Client(executable)
    try:
        _, raw = client.exchange(b'', 100)
        assert client.workspace() == bytes(MIB), 'first malformed body initialized state'
        maximum = struct.pack('<8sQQQ', BODY, 0, MIB, 0) + bytes(MIB-32)
        _, raw = client.exchange(maximum)
        assert raw == struct.pack('<8sQQQ', RESULT, 1, 0, 0)
        stream = framed(scalar) * 3
        for at in range(0, len(stream), 7):
            client.send(stream[at:at+7])
        for _ in range(3):
            assert client.receive() == scalar_response
        frames = client.frames
        client.close()
    finally:
        client.kill()
    rows.append({'case': 'exact-capacity-fragmented-pipeline', 'kind': 'transport-boundary',
                 'status': 'PASS', 'interactive_frames': frames})

    client = Client(executable)
    rollback_checks = []
    try:
        created, _ = client.exchange(tensor([1., 2.], True))
        assert created[0][2] == 1
        baseline = client.workspace()
        client.exchange(packet([command(12, 1, 0, bits(9.)), command(10, 999)]), 1)
        assert client.workspace() == baseline
        rollback_checks.append('value-and-version')

        bad_create = packet([command(1, 1, 0, 1, 0, 0, 0, 224),
                             command(12, 1, 0, bits(77.)), command(10, 999)],
                            struct.pack('<f', 3.))
        client.exchange(bad_create, 1)
        assert client.workspace() == baseline
        rollback_checks.append('allocation-cursor-and-provisional-handle')
        created, _ = client.exchange(tensor([4.]))
        assert created[0][2] == 2
        before_reset = client.workspace()
        client.exchange(packet([command(33, 1), command(10, 2)]), 1)
        assert client.workspace() == before_reset
        rollback_checks.append('reset-generation-and-descriptor-retirement')
        read_rows, _ = client.exchange(packet([command(10, 2)]))
        assert read_rows[0][3] == struct.pack('<f', 4.)
        client.exchange(packet([command(33, 1)]))
        client.exchange(packet([command(10, 2)]), 1)
        created, _ = client.exchange(tensor([5.]))
        assert created[0][2] == 258
        before_reset = client.workspace()
        client.exchange(packet([command(33, 1), command(10, 999)]), 1)
        assert client.workspace() == before_reset
        rollback_checks.append('published-generation-remains-live-after-rollback')
        read_rows, _ = client.exchange(packet([command(10, 258)]))
        assert read_rows[0][3] == struct.pack('<f', 5.)
        client.exchange(packet([command(33, 1)]))
        summed, _ = client.exchange(packet([command(4, 7, 1)]))
        assert summed[0][2] == 514
        client.exchange(packet([command(7, 514)]))
        grads, _ = client.exchange(packet([command(11, 1)]))
        assert grads[0][3] == struct.pack('<2f', 1., 1.)
        before_gradient = client.workspace()
        client.exchange(packet([command(8, 1), command(9, 1, bits(.125)), command(10, 999)]), 1)
        assert client.workspace() == before_gradient
        rollback_checks.append('gradient-and-optimizer-mutation')
        assert client.exchange(packet([command(10, 1)]))[0][0][3] == struct.pack('<2f', 1., 2.)
        frames = client.frames
        client.close()
    finally:
        client.kill()
    rows.append({'case': 'independent-whole-frame-transactions', 'kind': 'state-atomicity',
                 'status': 'PASS', 'interactive_frames': frames,
                 'full_workspace_bytes_per_check': MIB, 'checks': rollback_checks})

    client = Client(executable)
    try:
        client.exchange(tensor([1., 2., 3., 4.], True))
        for step in range(64):
            client.exchange(packet([command(8, 1)]))
            reduced, _ = client.exchange(packet([command(4, 7, 1)]))
            handle = (step << 8) | 2
            assert reduced[0][2] == handle
            client.exchange(packet([command(7, handle)]))
            updated, _ = client.exchange(packet([command(9, 1, bits(.125)), command(10, 1), command(11, 1)]))
            assert updated[1][3] == struct.pack('<4f', *(v-.125*(step+1) for v in (1., 2., 3., 4.)))
            assert updated[2][3] == struct.pack('<4f', 1., 1., 1., 1.)
            client.exchange(packet([command(33, 1)]))
        frames = client.frames
        client.close()
    finally:
        client.kill()
    rows.append({'case': 'phase-separated-SGD-exact-recurrence', 'kind': 'training',
                 'status': 'PASS', 'updates': 64, 'exact_binary32_comparisons': 512,
                 'interactive_frames': frames})

    # Actual EPIPE, not a forged kernel result: close the child's stdout reader.
    process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0, env={'PATH': ''})
    try:
        process.stdout.close()
        process.stdin.write(framed(scalar))
        process.stdin.close()
        code = process.wait(timeout=10)
        assert code == 4 and process.stderr.read() == b'', code
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
    rows.append({'case': 'real-broken-output-pipe', 'kind': 'native-io-error', 'status': 'PASS', 'exit_status': code})

    assert executable.read_bytes() == executable_before
    report = {'schema': 'heptatorch.independent-persistent-session-review.v1', 'status': 'PASS',
              'executable_sha256': sha(executable_before), 'reviewer_sha256': sha(Path(__file__).read_bytes()),
              'cases': len(rows), 'fixture_interoperation_cases': sum(r['kind']=='fixture-interoperation' for r in rows),
              'executable_unchanged': True, 'host_numerical_runtime': False,
              'every_successful_exchange_observed_before_stdin_eof': True,
              'rows': rows}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.executable.resolve(), args.fixtures.resolve(), args.output.resolve())
