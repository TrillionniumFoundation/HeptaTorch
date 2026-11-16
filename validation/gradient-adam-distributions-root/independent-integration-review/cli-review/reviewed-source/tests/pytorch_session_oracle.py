"""Pinned PyTorch comparisons across real interactive native HIA requests."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import select
import struct
import subprocess
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('executable', type=Path)
parser.add_argument('report', type=Path)
parser.add_argument('--reference', type=Path, default=Path(__file__).with_name('pytorch_oracle.py'))
parser.add_argument('--lock', type=Path, default=Path('compatibility/pytorch-oracle-lock.json'))
args = parser.parse_args()
EXE, REPORT = args.executable.resolve(), args.report.resolve()
original_argv = sys.argv
try:
    sys.argv = [str(args.reference), str(EXE), str(REPORT.with_name(REPORT.stem+'-provenance.json')), str(args.lock)]
    support = runpy.run_path(str(args.reference), run_name='heptatorch_session_reference_support')
finally:
    sys.argv = original_argv
torch, F = support['torch'], support['F']
Program, bits, f32 = support['Program'], support['bits'], support['f32']
PROVENANCE = support['PROVENANCE']
assert PROVENANCE['status'] == 'PASS' and len(PROVENANCE['checks']) == 11
assert all(value is True for value in PROVENANCE['checks'].values())
ROWS = []
MIB = 1 << 20

class Session:
    def __init__(self, row):
        self.row = row
        self.process = subprocess.Popen([str(EXE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, env={'PATH': ''})
        self.next_id, self.generation, self.keep = 1, 0, 0

    def program(self):
        p = Program()
        p.next_id, p.generation, p.keep = self.next_id, self.generation, self.keep
        return p

    def read(self, length):
        data = bytearray()
        deadline = time.monotonic()+120
        while len(data) < length:
            delay = deadline-time.monotonic()
            assert delay > 0 and select.select([self.process.stdout], [], [], delay)[0], 'native response timeout'
            part = os.read(self.process.stdout.fileno(), length-len(data))
            assert part, ('native premature EOF', self.process.poll())
            data.extend(part)
        return bytes(data)

    def exchange(self, program, expected=None, status=0):
        body = program.encode()
        raw = b'HTOSES01'+struct.pack('<Q', len(body))+body
        data = memoryview(raw)
        while data:
            written = os.write(self.process.stdin.fileno(), data)
            assert written > 0
            data = data[written:]
        assert not self.process.stdin.closed
        magic, length = struct.unpack('<8sQ', self.read(16))
        assert magic == b'HTOSRP01' and 32 <= length <= MIB
        response = self.read(length)
        assert self.process.poll() is None and not self.process.stdin.closed
        header, version, count, observed = struct.unpack_from('<8sQQQ', response)
        assert header == b'HTORSP01' and version == 1 and observed == status, (observed, status)
        self.row['frames'] += 1
        self.row['request_bytes'] += len(raw)
        self.row['response_bytes'] += 16+len(response)
        if status:
            assert count == 0 and length == 32
            return []
        assert count == len(program.commands)
        at, records = 32, []
        for index, command in enumerate(program.commands):
            op, error, value, size = struct.unpack_from('<4Q', response, at)
            at += 32
            payload = response[at:at+size]
            at += size
            assert op == command[0] and error == 0 and len(payload) == size
            records.append((op, value, payload))
            if expected is not None and index in expected:
                assert size == value*4
                actual = torch.tensor(struct.unpack('<'+str(value)+'f', payload), dtype=torch.float32)
                reference = expected[index].detach().cpu().to(dtype=torch.float32).reshape(-1)
                assert actual.shape == reference.shape
                finite = torch.isfinite(actual) & torch.isfinite(reference)
                if bool(finite.any()):
                    delta = (actual[finite].double()-reference[finite].double()).abs()
                    self.row['max_absolute_error'] = max(self.row['max_absolute_error'], delta.max().item())
                self.row['compared_values'] += reference.numel()
                assert bool(torch.isclose(actual, reference, rtol=self.row['rtol'], atol=self.row['atol'], equal_nan=True).all()), (index, actual, reference)
        assert at == len(response)
        self.next_id, self.generation, self.keep = program.next_id, program.generation, program.keep
        return records

    def metadata(self, record, reference):
        fields = struct.unpack('<13Q', record[2])
        assert fields[0] == reference.ndim and fields[1] == reference.numel()
        assert fields[2] == int(reference.requires_grad)
        assert fields[3:3+reference.ndim] == tuple(reference.shape)
        assert fields[7:7+reference.ndim] == tuple(reference.stride())
        self.row['metadata_checks'] += 1

    def close(self):
        self.process.stdin.close()
        assert self.process.wait(timeout=20) == 0
        assert self.process.stdout.read() == self.process.stderr.read() == b''

    def stop(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)

def case(name, fn, rtol=4e-5, atol=5e-6):
    row = {'case': name, 'status': 'PASS', 'rtol': rtol, 'atol': atol,
           'frames': 0, 'request_bytes': 0, 'response_bytes': 0,
           'compared_values': 0, 'metadata_checks': 0, 'max_absolute_error': 0.,
           'all_responses_before_stdin_eof': True, 'single_native_process': True}
    session = Session(row)
    try:
        fn(session)
        session.close()
    except Exception as error:
        row.update(status='FAIL', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    finally:
        session.stop()
    ROWS.append(row)

def aliases(s):
    leaf = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    p = s.program(); native = p.tensor(leaf); s.exchange(p)
    p = s.program(); copied, shared = p.clone(native), p.detach(native); s.exchange(p)
    clone, detached = leaf.clone(), leaf.detach()
    p = s.program()
    indices = [p.command(13, value) for value in (native, copied, shared)]
    records = s.exchange(p)
    before = [struct.unpack('<13Q', records[index][2]) for index in indices]
    reference_before = [value._version for value in (leaf, clone, detached)]
    assert before[0][11] == before[2][11] and before[1][11] != before[0][11]
    assert before[0][12] == before[2][12]
    p = s.program(); p.command(12, native, 2, bits(7.5)); s.exchange(p)
    leaf.reshape(-1)[2] = 7.5
    p = s.program(); expected = {p.read(native): leaf, p.read(copied): clone, p.read(shared): detached}
    indices = [p.command(13, value) for value in (native, copied, shared)]
    records = s.exchange(p, expected)
    after = [struct.unpack('<13Q', records[index][2]) for index in indices]
    references = (leaf, clone, detached)
    for index, reference in zip(indices, references):
        s.metadata(records[index], reference)
    assert after[0][11] == after[2][11] and after[1][11] != after[0][11]
    assert after[0][12] == after[2][12]
    native_deltas = [a[12]-b[12] for a, b in zip(after, before)]
    reference_deltas = [value._version-old for value, old in zip(references, reference_before)]
    assert native_deltas == reference_deltas == [1, 0, 1]
    s.row['version_observation'] = {
        'native_initial': [value[12] for value in before],
        'pytorch_initial': reference_before,
        'native_after_write': [value[12] for value in after],
        'pytorch_after_write': [value._version for value in references],
        'exact_shared_increment_and_clone_independence': True,
        'absolute_counter_parity_qualified': False,
        'scope': 'Packet owner counters include initial payload writes. Absolute torch._version parity remains an open compatibility gap.'}

def joined(s):
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_(True)
    p = s.program(); nx = p.tensor(x, True); s.exchange(p)
    p = s.program(); ny = p.join([nx, nx], 1); s.exchange(p)
    y = torch.cat([x, x], 1)
    p = s.program(); half = p.tensor(.5); nh = p.binary(3, ny, half); s.exchange(p)
    p = s.program(); nz = p.join([ny, nh], 0, stack=True); s.exchange(p)
    z = torch.stack([y, y*.5], 0)
    seed = torch.linspace(-1., 1., z.numel()).reshape(z.shape)
    p = s.program(); ns = p.tensor(seed); s.exchange(p)
    for _ in range(2):
        z.backward(seed, retain_graph=True)
        p = s.program(); p.seeded(nz, ns); s.exchange(p)
        p = s.program(); s.exchange(p, {p.read(nz): z, p.read(nx, True): x.grad})

def indexed(s):
    base = torch.arange(6, dtype=torch.float32).reshape(3, 2).requires_grad_(True)
    other = torch.linspace(-.3, .7, 6).reshape(2, 3).requires_grad_(True)
    p = s.program(); nb, no = p.tensor(base, True), p.tensor(other, True); s.exchange(p)
    p = s.program(); nx = p.transpose(nb, 0, 1); s.exchange(p)
    x = base.transpose(0, 1)
    index = torch.tensor([[0, 1, 1], [1, 2, 0]], dtype=torch.int64)
    p = s.program(); ny = p.scatter_add(nx, 1, index, no); s.exchange(p)
    y = torch.scatter_add(x, 1, index, other)
    selected = torch.tensor([1, 1], dtype=torch.int64)
    p = s.program(); nz = p.index_add(ny, 0, selected, nx, .5); s.exchange(p)
    z = torch.index_add(y, 0, selected, x, alpha=.5)
    seed = torch.tensor([[.25, -.5, .75], [1., -1.25, 1.5]])
    p = s.program(); ns = p.tensor(seed); s.exchange(p)
    for _ in range(2):
        z.backward(seed, retain_graph=True)
        p = s.program(); p.seeded(nz, ns); s.exchange(p)
        p = s.program(); s.exchange(p, {p.read(nz): z, p.read(nb, True): base.grad, p.read(no, True): other.grad})

def empty(s):
    x = torch.empty((262144, 262144, 0, 262144), requires_grad=True)
    p = s.program(); nx = p.tensor(x, True); s.exchange(p)
    p = s.program(); ny = p.join([nx, nx], 2); s.exchange(p)
    y = torch.cat([x, x], 2)
    p = s.program(); nl = p.reduce(7, ny); s.exchange(p)
    y.sum().backward()
    p = s.program(); p.backward(nl); s.exchange(p)
    p = s.program(); metadata = p.command(13, ny)
    records = s.exchange(p, {p.read(nl): y.sum(), p.read(nx, True): x.grad})
    s.metadata(records[metadata], y)
    p = s.program(); p.reset_graph(1); s.exchange(p)
    p = s.program(); metadata = p.command(13, nx); records = s.exchange(p)
    s.metadata(records[metadata], x)

def cumulative(s):
    for iteration in range(8):
        reference = torch.arange(60000, dtype=torch.float32)*(iteration+1)/16
        p = s.program(); value = p.tensor(reference); expected = {p.read(value): reference}
        p.reset_graph(0); s.exchange(p, expected)
    assert s.row['request_bytes'] > MIB and s.row['response_bytes'] > MIB

def transaction(s):
    parameter = torch.tensor([1., 2.], requires_grad=True)
    p = s.program(); native = p.tensor(parameter, True); s.exchange(p)
    p = s.program(); loss = p.reduce(7, p.binary(3, native, native)); s.exchange(p)
    reference = (parameter*parameter).sum()
    failed = parameter.detach().clone().requires_grad_(True)
    failed_loss = (failed*failed).sum()
    with torch.no_grad(): failed[0] = 9.
    try:
        failed_loss.backward()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Pinned PyTorch accepted changed saved input')
    p = s.program(); provisional = p.clone(native); p.command(12, native, 0, bits(9.)); p.backward(loss)
    s.exchange(p, status=6)
    reference.backward()
    p = s.program(); p.backward(loss); s.exchange(p)
    p = s.program(); committed = p.clone(native); assert committed == provisional
    s.exchange(p, {p.read(native): parameter, p.read(native, True): parameter.grad, p.read(committed): parameter.detach()})

def optimizer_session(s, mode, momentum=0., dampening=0., wd=0., nesterov=False):
    parameter = torch.nn.Parameter(torch.tensor([-.7, .3, 1.2]))
    lr, b1, b2, eps = map(f32, [.02, .9, .999, 1e-8])
    momentum, dampening, wd = map(f32, [momentum, dampening, wd])
    p = s.program(); native = p.tensor(parameter, True)
    if mode == 'SGD':
        state = p.tensor(torch.zeros_like(parameter)); keep = 2
        optimizer = torch.optim.SGD([parameter], lr=lr, momentum=momentum, dampening=dampening,
                                    weight_decay=wd, nesterov=nesterov, foreach=False, fused=False)
        steps = 64
    else:
        first, second = p.tensor(torch.zeros_like(parameter)), p.tensor(torch.zeros_like(parameter)); keep = 3
        cls = torch.optim.AdamW if mode == 'AdamW' else torch.optim.Adam
        optimizer = cls([parameter], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd, foreach=False, fused=False)
        steps = 128
    s.exchange(p)
    generator = torch.Generator().manual_seed(2731)
    for step in range(1, steps+1):
        gradient = torch.randn(parameter.shape, generator=generator)
        optimizer.zero_grad(set_to_none=False)
        p = s.program(); p.zero(native); s.exchange(p)
        p = s.program(); ng = p.tensor(gradient); loss = p.reduce(7, p.binary(3, native, ng))
        reference = (parameter*gradient).sum()
        s.exchange(p, {p.read(loss): reference.detach()})
        reference.backward()
        p = s.program(); p.backward(loss); s.exchange(p, {p.read(native, True): parameter.grad})
        optimizer.step()
        p = s.program()
        if mode == 'SGD':
            if momentum == wd == dampening == 0. and not nesterov:
                p.sgd(native, lr)
            else:
                p.momentum(native, state, step, lr, momentum, dampening, wd, nesterov)
            expected = {p.read(native): parameter.detach().clone()}
            if momentum:
                expected[p.read(state)] = optimizer.state[parameter]['momentum_buffer'].clone()
        else:
            p.adam(native, first, second, step, lr, b1, b2, eps, wd, mode == 'AdamW')
            expected = {p.read(native): parameter.detach().clone(), p.read(first): optimizer.state[parameter]['exp_avg'].clone(),
                        p.read(second): optimizer.state[parameter]['exp_avg_sq'].clone()}
        s.exchange(p, expected)
        p = s.program(); p.reset_graph(keep); s.exchange(p)
    s.row['optimizer_steps'] = steps

def network(s):
    data = torch.tensor([[-2., -1.], [-1., 1.], [0., -1.], [1., 1.], [2., 0.], [3., 1.]])
    target = data @ torch.tensor([[1.5], [-.75]])+.3
    generator = torch.Generator().manual_seed(1729)
    parameters = [torch.nn.Parameter(torch.randn((4, 2), generator=generator)*.15), torch.nn.Parameter(torch.zeros(4)),
                  torch.nn.Parameter(torch.randn((1, 4), generator=generator)*.15), torch.nn.Parameter(torch.zeros(1))]
    p = s.program(); nx, nt = p.tensor(data), p.tensor(target); ids = [p.tensor(v, True) for v in parameters]; s.exchange(p)
    lr = f32(.04)
    optimizer = torch.optim.SGD(parameters, lr=lr, foreach=False, fused=False)
    for _ in range(24):
        optimizer.zero_grad(set_to_none=False)
        p = s.program()
        for native in ids: p.zero(native)
        s.exchange(p)
        p = s.program(); prediction = p.linear(p.unary(15, p.linear(nx, ids[0], ids[1])), ids[2], ids[3]); loss = p.mse(prediction, nt)
        reference = F.mse_loss(F.linear(torch.tanh(F.linear(data, parameters[0], parameters[1])), parameters[2], parameters[3]), target)
        s.exchange(p, {p.read(loss): reference.detach()})
        reference.backward()
        p = s.program(); p.backward(loss); s.exchange(p, {p.read(native, True): parameter.grad for native, parameter in zip(ids, parameters)})
        optimizer.step()
        p = s.program()
        for native in ids: p.sgd(native, lr)
        s.exchange(p, {p.read(native): parameter.detach().clone() for native, parameter in zip(ids, parameters)})
        p = s.program(); p.reset_graph(6); s.exchange(p)
    s.row['optimizer_steps'] = 24

def matrix(s, left_shape, right_shape):
    generator = torch.Generator().manual_seed(3981)
    left = torch.randn(left_shape, generator=generator).requires_grad_(True)
    right = torch.randn(right_shape, generator=generator).requires_grad_(True)
    p = s.program(); nl, nr = p.tensor(left, True), p.tensor(right, True); s.exchange(p)
    p = s.program(); output = p.matmul(nl, nr); s.exchange(p)
    reference = torch.matmul(left, right)
    seed = torch.randn(reference.shape, generator=generator)
    p = s.program()
    if seed.ndim >= 2:
        native_seed = p.transpose(p.tensor(seed.transpose(-1, -2).contiguous()), -1, -2)
    else:
        native_seed = p.tensor(seed)
    s.exchange(p)
    for _ in range(2):
        reference.backward(seed, retain_graph=True)
        p = s.program(); p.seeded(output, native_seed); s.exchange(p)
        p = s.program(); s.exchange(p, {p.read(output): reference.detach(), p.read(nl, True): left.grad, p.read(nr, True): right.grad})

def channels_last(s):
    base = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5).requires_grad_(True)
    p = s.program(); native = p.tensor(base, True); s.exchange(p)
    p = s.program(); view = p.transpose(p.transpose(native, 1, 3), 2, 3); s.exchange(p)
    reference_view = base.transpose(1, 3).transpose(2, 3)
    p = s.program(); output = p.join([view, view], 1); s.exchange(p)
    reference = torch.cat([reference_view, reference_view], 1)
    assert reference.is_contiguous(memory_format=torch.channels_last)
    seed = torch.linspace(-.75, 1.25, reference.numel()).reshape(reference.shape)
    p = s.program(); ns = p.tensor(seed); s.exchange(p)
    reference.backward(seed)
    p = s.program(); p.seeded(output, ns); s.exchange(p)
    p = s.program(); meta = p.command(13, output)
    records = s.exchange(p, {p.read(output): reference.detach(), p.read(native, True): base.grad})
    s.metadata(records[meta], reference)

started = time.monotonic()
before = hashlib.sha256(EXE.read_bytes()).hexdigest()
case('cross-frame-storage-alias-and-version', aliases, 0., 0.)
case('cross-frame-cat-stack-shared-VJP', joined)
case('cross-frame-strided-scatter-index-VJP', indexed)
case('cross-frame-late-zero-empty-shape', empty, 0., 0.)
case('cross-frame-cumulative-payload-beyond-one-wrapper', cumulative, 0., 0.)
case('cross-frame-failed-version-transaction-continuation', transaction, 0., 0.)
sgd_cases = [('plain', 0., 0., 0., False), ('weight-decay', 0., 0., .04, False),
             ('momentum', .9, 0., 0., False), ('dampening', .9, .2, 0., False),
             ('nesterov', .9, 0., 0., True), ('momentum-weight-decay', .9, 0., .04, False)]
for name, momentum, dampening, wd, nesterov in sgd_cases:
    case('cross-frame-SGD-'+name, lambda s, m=momentum, d=dampening, w=wd, n=nesterov: optimizer_session(s, 'SGD', m, d, w, n), 5e-5, 5e-6)
for mode in ('Adam', 'AdamW'):
    for wd in (0., .07):
        case('cross-frame-'+mode+'-wd-'+str(wd), lambda s, m=mode, w=wd: optimizer_session(s, m, wd=w), 1e-4, 4e-6)
case('cross-frame-MLP-autograd-SGD-24-steps', network, 6e-4, 4e-5)
for left, right in [((4,), (4,)), ((2, 3, 4), (4, 5)), ((2, 1, 3, 4), (5, 4, 2))]:
    case('cross-frame-matmul-'+str(left)+'-'+str(right), lambda s, a=left, b=right: matrix(s, a, b), 8e-5, 8e-6)
case('cross-frame-channels-last-cat-layout-and-VJP', channels_last)
after = hashlib.sha256(EXE.read_bytes()).hexdigest()
passed = len(ROWS) == len({r['case'] for r in ROWS}) == 21 and all(r['status'] == 'PASS' for r in ROWS) and before == after
report = {'schema': 'heptatorch.live-pytorch-persistent-session.v1', 'status': 'PASS' if passed else 'FAIL',
          'native_executable_sha256': before, 'native_unchanged_during_run': before == after,
          'oracle_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'reference_support_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
          'oracle_provenance': PROVENANCE, 'expected_cases': 21, 'cases': len(ROWS),
          'failed_cases': sum(r['status'] == 'FAIL' for r in ROWS),
          'compared_values': sum(r['compared_values'] for r in ROWS),
          'frames': sum(r['frames'] for r in ROWS), 'rows': ROWS,
          'wall_seconds': time.monotonic()-started,
          'scope': 'Real pinned CPU PyTorch comparisons across interactive native requests; each case owns one native process. Transaction rollback is the session contract, not a claim that PyTorch itself supplies transactions.'}
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps(report, allow_nan=False))
raise SystemExit(0 if passed else 1)
