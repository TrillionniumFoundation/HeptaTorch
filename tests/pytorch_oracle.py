"""Standalone CI oracle. PyTorch is a test dependency, never the native runtime.

Usage: python pytorch_oracle.py build/heptatorch build/pytorch-differential.json
This file can be embedded verbatim in an isolated CI Python step.
"""
import hashlib
import importlib.metadata
import platform
import json
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

EXE = Path(sys.argv[1] if len(sys.argv) > 1 else 'build/heptatorch').resolve()
REPORT = Path(sys.argv[2] if len(sys.argv) > 2 else 'build/pytorch-differential.json')
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
LOCK_PATH = Path(sys.argv[3] if len(sys.argv) > 3 else 'compatibility/pytorch-oracle-lock.json')
ROWS = []


def emit(report):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False))


try:
    lock_bytes = LOCK_PATH.read_bytes()
    LOCK = json.loads(lock_bytes)
    EXPECTED = LOCK['oracle']
    inventory = json.loads(Path('compatibility/upstream-lock.json').read_text())
    if (LOCK['format'] != 'heptatorch.pytorch-oracle-lock.v1'
            or LOCK['inventory_commit'] != PIN or inventory['commit'] != PIN
            or LOCK['inventory_tag'] != 'v2.15.0'
            or LOCK['release_source_equivalence']['inventory_commit'] != PIN
            or LOCK['release_source_equivalence']['wheel_build_commit'] != EXPECTED['git_commit']):
        raise ValueError('Oracle lock does not bind the recorded inventory/build pair')
except (OSError, ValueError, KeyError, TypeError) as exc:
    emit({'status': 'NOT_RUN', 'reason': 'Invalid oracle lock: ' + str(exc),
          'upstream_commit': PIN, 'oracle_lock': str(LOCK_PATH)})
    raise SystemExit(2)

OBSERVED = {'python': {'implementation': sys.implementation.name,
                       'version': list(sys.version_info[:2]), 'full_version': sys.version},
            'platform': {'sys_platform': sys.platform, 'machine': platform.machine()}}
try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    emit({'status': 'NOT_RUN', 'reason': str(exc), 'required_oracle': EXPECTED,
          'observed_oracle': OBSERVED, 'upstream_commit': PIN})
    raise SystemExit(2)

OBSERVED.update(version=str(torch.__version__), git_commit=torch.version.git_version,
                backend_versions={key: getattr(torch.version, key, 'MISSING')
                                  for key in EXPECTED['backend_versions']})
metadata_error = None
try:
    distribution = importlib.metadata.distribution(EXPECTED['distribution'])
    direct_url_text = distribution.read_text('direct_url.json')
    if direct_url_text is None:
        raise ValueError('Installed torch lacks PEP610 direct_url.json')
    direct_url = json.loads(direct_url_text)
    install_report = json.loads(Path(EXPECTED['install_report']).read_text())
    torch_records = [item for item in install_report['install']
                     if item.get('metadata', {}).get('name', '').lower() == 'torch']
    if len(torch_records) != 1:
        raise ValueError('pip report must contain exactly one requested torch distribution')
    installed_record = torch_records[0]
    OBSERVED.update(distribution_version=distribution.version,
                    direct_url=direct_url,
                    install_report={
                        'path': EXPECTED['install_report'],
                        'pip_version': install_report.get('pip_version'),
                        'download_info': installed_record['download_info'],
                        'metadata_name': installed_record['metadata']['name'],
                        'metadata_version': installed_record['metadata']['version'],
                        'requested': installed_record.get('requested')})
except (OSError, ValueError, KeyError, TypeError, importlib.metadata.PackageNotFoundError) as exc:
    metadata_error = type(exc).__name__ + ': ' + str(exc)

DIRECT = OBSERVED.get('direct_url', {})
DOWNLOAD = OBSERVED.get('install_report', {}).get('download_info', {})
CHECKS = {
    'python_matches': (OBSERVED['python']['implementation'] == EXPECTED['python']['implementation']
                       and OBSERVED['python']['version'] == EXPECTED['python']['version']),
    'platform_matches': OBSERVED['platform'] == EXPECTED['platform'],
    'version_backend_matches': (OBSERVED['version'] == EXPECTED['version']
                                and OBSERVED['backend_versions'] == EXPECTED['backend_versions']),
    'git_matches_oracle_build_pin': OBSERVED['git_commit'] == EXPECTED['git_commit'],
    'distribution_version_matches': OBSERVED.get('distribution_version') == EXPECTED['version'],
    'direct_url_matches': DIRECT.get('url') == EXPECTED['wheel']['url'],
    'direct_url_archive_sha256_matches': DIRECT.get('archive_info', {}).get('hashes', {}).get('sha256') == EXPECTED['wheel']['sha256'],
    'pip_report_url_matches': DOWNLOAD.get('url') == EXPECTED['wheel']['url'],
    'pip_report_archive_sha256_matches': DOWNLOAD.get('archive_info', {}).get('hashes', {}).get('sha256') == EXPECTED['wheel']['sha256'],
    'pip_report_version_matches': OBSERVED.get('install_report', {}).get('metadata_version') == EXPECTED['version'],
    'pip_report_requested': OBSERVED.get('install_report', {}).get('requested') is True,
}
PROVENANCE = {'status': 'PASS' if metadata_error is None and all(CHECKS.values()) else 'FAIL',
              'lock_path': str(LOCK_PATH), 'lock_sha256': hashlib.sha256(lock_bytes).hexdigest(),
              'expected': EXPECTED, 'observed': OBSERVED, 'checks': CHECKS,
              'metadata_error': metadata_error,
              'release_source_equivalence': LOCK['release_source_equivalence']}
if PROVENANCE['status'] != 'PASS':
    emit({'status': 'NOT_RUN', 'reason': 'Pinned oracle provenance mismatch',
          'upstream_commit': PIN, 'torch_git_version': torch.version.git_version,
          'torch_git_matches_upstream_pin': torch.version.git_version == PIN,
          'oracle_provenance': PROVENANCE})
    raise SystemExit(2)
torch.set_num_threads(1)
torch.manual_seed(1729)
torch.backends.mkldnn.enabled = False
torch.use_deterministic_algorithms(True)


def bits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def f32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


class Program:
    """Only wire serialization and tensor-ID bookkeeping; no numerical kernels."""
    def __init__(self):
        self.commands, self.blobs, self.configs, self.extra, self.next_id = [], {}, {}, [], 1
        self.generation, self.keep = 0, 0

    def command(self, code, *args):
        assert len(args) <= 7
        at = len(self.commands)
        self.commands.append([code] + [int(a) & ((1 << 64) - 1) for a in args] + [0] * (7 - len(args)))
        return at

    def make(self, code, *args, count=1):
        self.command(code, *args)
        self.next_id += count
        return self.handle(self.next_id - 1)

    def tensor(self, value, requires=False):
        value = torch.as_tensor(value, dtype=torch.float32, device='cpu').detach().contiguous()
        assert value.ndim <= 4
        at = self.command(1, value.ndim, int(requires), *value.shape, *([0] * (4 - value.ndim)), 0)
        self.blobs[at] = struct.pack('<' + str(value.numel()) + 'f', *value.reshape(-1).tolist())
        result = self.next_id
        self.next_id += 1
        return self.handle(result)

    def binary(self, op, a, b): return self.make(2, op, a, b)
    def unary(self, op, a): return self.make(3, op, a)
    def reduce(self, op, a): return self.make(4, op, a)
    def mm(self, a, b): return self.make(5, a, b)
    def transpose(self, a, x=0, y=1): return self.make(6, a, x, y)
    def backward(self, a): return self.command(7, a)
    def zero(self, a): return self.command(8, a)
    def sgd(self, a, lr): return self.command(9, a, bits(lr))
    def read(self, a, grad=False): return self.command(11 if grad else 10, a)
    def linear(self, x, w, b): return self.make(15, x, w, b, count=3)
    def mse(self, a, b): return self.make(16, a, b, count=3)
    def view(self, a, shape): return self.make(17, a, len(shape), *shape, *([0] * (4 - len(shape))))
    def clone(self, a): return self.make(18, a)
    def detach(self, a): return self.make(19, a)
    def axis(self, op, a, dim, keep=False): return self.make(20, op, a, dim, int(keep))
    def softmax(self, a, dim, log=False): return self.make(21, 21 if log else 20, a, dim)
    def conv(self, x, w, stride, pad, dilation, groups):
        pack = lambda v: (v[0] << 32) | v[1]
        return self.make(22, x, w, pack(stride), pack(pad), pack(dilation), groups)
    def pool(self, x, kernel, stride, pad, average, include):
        pack = lambda v: (v[0] << 32) | v[1]
        return self.make(23, 23 if average else 24, x, pack(kernel), pack(stride), pack(pad), int(include))
    def adam(self, x, m, v, step, lr, b1, b2, eps, wd, decoupled):
        at = self.command(24, x, m, v, step, 0)
        self.configs[at] = struct.pack('<6Q', bits(lr), bits(b1), bits(b2), bits(eps), bits(wd), int(decoupled))
    def momentum(self, x, state, step, lr, momentum, dampening, wd, nesterov):
        at = self.command(25, x, state, 0, step, 0)
        self.configs[at] = struct.pack('<5Q', bits(lr), bits(momentum), bits(dampening), bits(wd), int(nesterov))
    def layernorm(self, x, w, b, eps): return self.make(26, x, w, b, bits(eps), count=10)
    def cross_entropy(self, x, target): return self.make(27, x, target, count=5)
    def batchnorm(self, x, m, v, w, b, eps): return self.make(28, x, m, v, w, b, bits(eps), count=11)

    def handle(self, slot):
        return slot if slot <= self.keep else (self.generation << 8) | slot

    def reset_graph(self, keep):
        self.command(33, keep)
        self.keep, self.next_id = keep, keep + 1
        self.generation += 1

    def seeded(self, output, seed): return self.command(29, output, seed)
    def matmul(self, left, right): return self.make(32, left, right)

    def index_select(self, value, axis, index):
        index = torch.as_tensor(index, dtype=torch.int64).reshape(-1)
        at = self.command(30, value, axis, index.numel(), 0)
        self.extra.append((at, 4, struct.pack('<' + str(index.numel()) + 'q', *index.tolist())))
        result = self.next_id
        self.next_id += 1
        return self.handle(result)

    def gather(self, value, axis, index):
        index = torch.as_tensor(index, dtype=torch.int64)
        at = self.command(31, value, axis, 0)
        body = struct.pack('<' + str(index.ndim + 1) + 'Q', index.ndim, *index.shape)
        body += struct.pack('<' + str(index.numel()) + 'q', *index.reshape(-1).tolist())
        self.extra.append((at, 3, body))
        result = self.next_id
        self.next_id += 1
        return self.handle(result)

    def join(self, values, axis=0, stack=False):
        at = self.command(35 if stack else 34, axis, 0)
        body = struct.pack('<' + str(len(values) + 1) + 'Q', len(values), *values)
        self.extra.append((at, 2, body))
        result = self.next_id
        self.next_id += 1
        return self.handle(result)

    def _indexed_add(self, code, value, axis, index, source, alpha):
        index = torch.as_tensor(index, dtype=torch.int64, device='cpu')
        at = self.command(code, value, source, axis, 0, bits(alpha))
        body = struct.pack('<' + str(index.ndim + 1) + 'Q', index.ndim, *index.shape)
        body += struct.pack('<' + str(index.numel()) + 'q', *index.reshape(-1).tolist())
        self.extra.append((at, 4, body))
        result = self.next_id
        self.next_id += 1
        return self.handle(result)

    def scatter_add(self, value, axis, index, source):
        return self._indexed_add(36, value, axis, index, source, 1.)

    def index_add(self, value, axis, index, source, alpha=1.):
        return self._indexed_add(37, value, axis, index, source, alpha)

    def encode(self):
        offset = 32 + 64 * len(self.commands)
        payload, commands = bytearray(), [row.copy() for row in self.commands]
        for at, data in self.blobs.items():
            commands[at][7] = offset + len(payload)
            payload.extend(data)
        for at, field, data in self.extra:
            while (offset + len(payload)) % 8: payload.append(0)
            commands[at][field] = offset + len(payload)
            payload.extend(data)
        for at, data in self.configs.items():
            while (offset + len(payload)) % 8: payload.append(0)
            commands[at][5] = offset + len(payload)
            payload.extend(data)
        return struct.pack('<8sQQQ', b'HTORCH01', len(commands), offset + len(payload), 0) + b''.join(struct.pack('<8Q', *row) for row in commands) + payload


def compare(name, program, expected, rtol=4e-5, atol=5e-6):
    row = {'case': name, 'status': 'PASS', 'rtol': rtol, 'atol': atol, 'compared_values': 0,
           'max_absolute_error': 0., 'max_relative_error_scaled_by_atol': 0., 'mismatches': 0}
    try:
        result = subprocess.run([str(EXE)], input=program.encode(), capture_output=True, timeout=120, env={'PATH': ''})
        assert result.returncode == 0 and not result.stderr, (result.returncode, result.stderr.decode(errors='replace'))
        raw = result.stdout
        assert len(raw) >= 32
        magic, version, count, status = struct.unpack_from('<8sQQQ', raw)
        assert (magic, version, count, status) == (b'HTORSP01', 1, len(program.commands), 0)
        offset = 32
        for i in range(count):
            code, error, value, size = struct.unpack_from('<4Q', raw, offset)
            offset += 32
            payload = raw[offset:offset + size]
            offset += size
            assert code == program.commands[i][0] and error == 0 and len(payload) == size
            if i not in expected: continue
            assert size == value * 4
            actual = torch.tensor(struct.unpack('<' + str(value) + 'f', payload), dtype=torch.float32)
            reference = expected[i].detach().to(dtype=torch.float32, device='cpu').reshape(-1)
            assert actual.shape == reference.shape
            finite = torch.isfinite(actual) & torch.isfinite(reference)
            if bool(finite.any()):
                delta = (actual[finite].double() - reference[finite].double()).abs()
                scaled = delta / reference[finite].double().abs().clamp_min(atol)
                row['max_absolute_error'] = max(row['max_absolute_error'], delta.max().item())
                row['max_relative_error_scaled_by_atol'] = max(row['max_relative_error_scaled_by_atol'], scaled.max().item())
            row['compared_values'] += reference.numel()
            bad = torch.nonzero(~torch.isclose(actual, reference, rtol=rtol, atol=atol, equal_nan=True)).reshape(-1)
            row['mismatches'] += bad.numel()
            for j in bad[:max(0, 5 - len(row.get('examples', [])))].tolist():
                safe = lambda v: v if math.isfinite(v) else str(v)
                row.setdefault('examples', []).append({'command': i, 'index': j,
                    'actual': safe(actual[j].item()), 'reference': safe(reference[j].item())})
        assert offset == len(raw), (offset, len(raw))
        assert row['mismatches'] == 0, f"{row['mismatches']} values outside predeclared tolerances"
    except Exception as exc:
        row.update(status='FAIL', error=str(exc)[:2000])
    ROWS.append(row)


def function_case(name, data, reference_fn, native_fn, requires=None, rtol=4e-5, atol=5e-6):
    requires = requires or [True] * len(data)
    leaves = [a.detach().clone().requires_grad_(req) for a, req in zip(data, requires)]
    p = Program()
    ids = [p.tensor(a, req) for a, req in zip(leaves, requires)]
    reference = reference_fn(*leaves)
    output = native_fn(p, *ids)
    upstream = torch.randn_like(reference)
    (reference * upstream).sum().backward()
    p.backward(p.reduce(7, p.binary(3, output, p.tensor(upstream))))
    expected = {p.read(output): reference.detach().clone()}
    for leaf, tid, req in zip(leaves, ids, requires):
        if req:
            assert leaf.grad is not None, name
            expected[p.read(tid, True)] = leaf.grad.clone()
    compare(name, p, expected, rtol, atol)


def operators():
    shapes = [((), ()), ((2, 3), (3,)), ((2, 1, 3), (1, 4, 1)), ((2, 0, 3), (1, 1, 3)), ((2, 1, 2, 3), (1, 3, 1, 3))]
    for op, fn in [(1, torch.add), (2, torch.sub), (3, torch.mul), (4, torch.div)]:
        for i, (a, b) in enumerate(shapes):
            function_case(f'binary-{op}-{i}', [torch.rand(a) + .4, torch.rand(b) + .4], fn, lambda p, x, y: p.binary(op, x, y))
    for m, k, n in [(2, 3, 4), (2, 0, 3), (4, 5, 2)]:
        for trans in [False, True]:
            data = [torch.randn((k, m) if trans else (m, k)), torch.randn(k, n)]
            function_case(f'mm-{m}-{k}-{n}-transpose-{trans}', data,
                          lambda a, b: (a.t() if trans else a) @ b,
                          lambda p, a, b: p.mm(p.transpose(a) if trans else a, b))
    for op, fn in [(5, torch.relu), (6, torch.neg), (11, torch.sqrt), (12, torch.exp), (13, torch.log), (14, torch.sigmoid), (15, torch.tanh)]:
        x = torch.rand(2, 3) + .2 if op in [11, 13] else torch.randn(2, 3)
        function_case(f'unary-{op}', [x], fn, lambda p, a: p.unary(op, a), rtol=8e-5)
    for axis in [0, 1, -1]:
        for keep in [False, True]:
            for mean in [False, True]:
                function_case(f'axis-{axis}-{keep}-{mean}', [torch.randn(2, 3, 4)],
                              lambda x: x.mean(axis, keepdim=keep) if mean else x.sum(axis, keepdim=keep),
                              lambda p, a: p.axis(19 if mean else 18, a, axis, keep))
        for log in [False, True]:
            function_case(f'softmax-{axis}-{log}', [torch.randn(2, 3, 4) * 3],
                          lambda x: F.log_softmax(x, axis) if log else F.softmax(x, axis),
                          lambda p, a: p.softmax(a, axis, log), rtol=1e-4)


def spatial_modules():
    specs = [((1, 2, 5, 5), (3, 2, 3, 2), (1, 1), (1, 0), (1, 1), 1),
             ((1, 4, 5, 6), (4, 2, 2, 3), (2, 1), (1, 1), (1, 1), 2),
             ((1, 2, 5, 5), (2, 1, 2, 2), (1, 1), (1, 1), (2, 2), 2)]
    for i, (xs, ws, stride, pad, dilation, groups) in enumerate(specs):
        function_case(f'conv2d-{i}', [torch.randn(xs), torch.randn(ws)],
                      lambda x, w: F.conv2d(x, w, stride=stride, padding=pad, dilation=dilation, groups=groups),
                      lambda p, x, w: p.conv(x, w, stride, pad, dilation, groups), rtol=2e-4, atol=2e-5)
    for padded in [False, True]:
        kernel, stride, pad = (3, 2), (2, 1), ((1, 1) if padded else (0, 0))
        for average, include in [(False, True), (True, True), (True, False)]:
            function_case(f'pool-{padded}-{average}-{include}', [torch.randn(1, 2, 5, 6)],
                          lambda x: F.avg_pool2d(x, kernel, stride, pad, count_include_pad=include) if average else F.max_pool2d(x, kernel, stride, pad),
                          lambda p, x: p.pool(x, kernel, stride, pad, average, include), rtol=1e-4)
    eps = f32(1e-5)
    for shape in [(2, 3), (2, 2, 3)]:
        function_case(f'layernorm-{shape}', [torch.randn(shape), torch.randn(3), torch.randn(3)],
                      lambda x, w, b: F.layer_norm(x, (3,), w, b, eps),
                      lambda p, x, w, b: p.layernorm(x, w, b, eps), rtol=5e-4, atol=5e-5)
    for shape in [(2, 3), (2, 3, 2, 2)]:
        data = [torch.randn(shape), torch.randn(3), torch.rand(3) + .3, torch.randn(3), torch.randn(3)]
        function_case(f'batchnorm-eval-{shape}', data,
                      lambda x, m, v, w, b: F.batch_norm(x, m, v, w, b, training=False, eps=eps),
                      lambda p, x, m, v, w, b: p.batchnorm(x, m, v, w, b, eps),
                      requires=[True, False, False, True, True], rtol=2e-4, atol=2e-5)
    for shape in [(3, 4), (1, 3)]:
        target = torch.rand(shape)
        target /= target.sum(1, keepdim=True)
        function_case(f'cross-entropy-probabilities-{shape}', [torch.randn(shape), target],
                      F.cross_entropy, lambda p, x, t: p.cross_entropy(x, t), rtol=1e-4)
    function_case('view-clone', [torch.randn(2, 3)], lambda x: x.view(3, 2).clone(),
                  lambda p, x: p.clone(p.view(x, (3, 2))))


def optimizers():
    for decoupled in [False, True]:
        for wd in [0., f32(.07)]:
            p = Program()
            parameter = torch.nn.Parameter(torch.tensor([-.7, .3, 1.2]))
            x, m, v = p.tensor(parameter, True), p.tensor(torch.zeros(3)), p.tensor(torch.zeros(3))
            lr, b1, b2, eps = map(f32, [.02, .9, .999, 1e-8])
            cls = torch.optim.AdamW if decoupled else torch.optim.Adam
            optimizer = cls([parameter], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd, foreach=False, fused=False)
            expected = {}
            for step in range(1, 7):
                grad = torch.randn_like(parameter)
                p.zero(x)
                p.backward(p.reduce(7, p.binary(3, x, p.tensor(grad))))
                p.adam(x, m, v, step, lr, b1, b2, eps, wd, decoupled)
                parameter.grad = grad.clone()
                optimizer.step()
                expected[p.read(x)] = parameter.detach().clone()
            expected[p.read(m)] = optimizer.state[parameter]['exp_avg'].clone()
            expected[p.read(v)] = optimizer.state[parameter]['exp_avg_sq'].clone()
            compare(f'optim-AdamW-{decoupled}-wd-{wd}', p, expected, rtol=1e-4, atol=4e-6)
    for momentum, dampening, nesterov in [(f32(.9), 0., False), (f32(.9), f32(.2), False), (f32(.9), 0., True)]:
        p = Program()
        parameter = torch.nn.Parameter(torch.tensor([.1, -.3, .9]))
        x, state = p.tensor(parameter, True), p.tensor(torch.zeros(3))
        lr, wd = f32(.03), f32(.04)
        optimizer = torch.optim.SGD([parameter], lr=lr, momentum=momentum, dampening=dampening, weight_decay=wd, nesterov=nesterov, foreach=False, fused=False)
        expected = {}
        for step in range(1, 7):
            grad = torch.randn_like(parameter)
            p.zero(x)
            p.backward(p.reduce(7, p.binary(3, x, p.tensor(grad))))
            p.momentum(x, state, step, lr, momentum, dampening, wd, nesterov)
            parameter.grad = grad.clone()
            optimizer.step()
            expected[p.read(x)] = parameter.detach().clone()
        expected[p.read(state)] = optimizer.state[parameter]['momentum_buffer'].clone()
        compare(f'optim-SGD-{dampening}-{nesterov}', p, expected, rtol=5e-5)
    p = Program()
    parameter = torch.nn.Parameter(torch.ones(2))
    x, m, v = p.tensor(parameter, True), p.tensor(torch.zeros(2)), p.tensor(torch.zeros(2))
    grad = torch.tensor([2e19, -1e20])
    lr, b1, b2, eps = map(f32, [.02, .9, .999, 1e-8])
    optimizer = torch.optim.Adam([parameter], lr=lr, betas=(b1, b2), eps=eps, foreach=False, fused=False)
    p.backward(p.reduce(7, p.binary(3, x, p.tensor(grad))))
    p.adam(x, m, v, 1, lr, b1, b2, eps, 0., False)
    parameter.grad = grad
    optimizer.step()
    compare('optim-Adam-large-gradient', p, {p.read(x): parameter.detach().clone(), p.read(v): optimizer.state[parameter]['exp_avg_sq'].clone()}, rtol=1e-4)


def training():
    p = Program()
    data = torch.tensor([[-2., -1.], [-1., 1.], [0., -1.], [1., 1.], [2., 0.], [3., 1.]])
    target = data @ torch.tensor([[1.5], [-.75]]) + .3
    params = [torch.nn.Parameter(torch.randn(4, 2) * .15), torch.nn.Parameter(torch.zeros(4)),
              torch.nn.Parameter(torch.randn(1, 4) * .15), torch.nn.Parameter(torch.zeros(1))]
    x, t = p.tensor(data), p.tensor(target)
    ids = [p.tensor(a, True) for a in params]
    lr = f32(.04)
    optimizer = torch.optim.SGD(params, lr=lr, foreach=False, fused=False)
    expected, losses = {}, []
    for step in range(18):
        optimizer.zero_grad(set_to_none=False)
        for a in ids: p.zero(a)
        out = p.linear(p.unary(15, p.linear(x, ids[0], ids[1])), ids[2], ids[3])
        loss = p.mse(out, t)
        reference = F.mse_loss(F.linear(torch.tanh(F.linear(data, params[0], params[1])), params[2], params[3]), target)
        expected[p.read(loss)] = reference.detach().clone()
        losses.append(reference.item())
        p.backward(loss)
        reference.backward()
        for a in ids: p.sgd(a, lr)
        optimizer.step()
    for a, ref in zip(ids, params): expected[p.read(a)] = ref.detach().clone()
    compare('MLP-real-autograd-SGD-18-steps', p, expected, rtol=6e-4, atol=4e-5)
    return {'steps': 18, 'torch_initial_loss': losses[0], 'torch_last_pre_update_loss': losses[-1],
            'torch_loss_decreased': losses[-1] < losses[0]}



def seeded_case(name, data, reference_fn, native_fn, rtol=8e-5, atol=8e-6):
    leaves = [a.detach().clone().requires_grad_(True) for a in data]
    p = Program()
    ids = [p.tensor(a, True) for a in leaves]
    reference = reference_fn(*leaves)
    output = native_fn(p, *ids)
    upstream = torch.randn_like(reference)
    # Construct a real noncontiguous seed whenever there are at least two axes.
    if upstream.ndim >= 2:
        seed = p.transpose(p.tensor(upstream.transpose(-1, -2).contiguous()), -1, -2)
    else:
        seed = p.tensor(upstream)
    expected = {p.read(output): reference.detach().clone()}
    for iteration in range(2):
        reference.backward(upstream, retain_graph=True)
        p.seeded(output, seed)
        for leaf, tid in zip(leaves, ids):
            assert leaf.grad is not None, name
            expected[p.read(tid, True)] = leaf.grad.clone()
    compare(name, p, expected, rtol, atol)


def extensions():
    specs = [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 4)), ((2, 3), (3, 4)),
             ((2, 3, 4), (4, 5)), ((4,), (2, 3, 4, 5)), ((2, 1, 3, 4), (5, 4, 6)),
             ((1, 3, 2, 4), (2, 1, 4, 5)), ((2, 3, 4, 5), (5,)), ((2, 1, 3, 4), (1, 4, 5)),
             ((0,), (0,)), ((2, 0), (0, 3)), ((0, 2, 3), (1, 3, 4)),
             ((2, 1, 0, 3), (1, 4, 3, 5)), ((2, 3), (3, 0)), ((1, 2, 3), (0, 3, 4))]
    for number, (sa, sb) in enumerate(specs):
        for trans in [False, True]:
            a, b = torch.randn(sa), torch.randn(sb)
            if trans and a.ndim > 1: a = a.transpose(-1, -2).contiguous()
            if trans and b.ndim > 1: b = b.transpose(-1, -2).contiguous()
            seeded_case(f'extended-matmul-{number}-{trans}', [a, b],
                        lambda x, w: (x.transpose(-1, -2) if trans and x.ndim > 1 else x) @ (w.transpose(-1, -2) if trans and w.ndim > 1 else w),
                        lambda p, x, w: p.matmul(p.transpose(x, -1, -2) if trans and len(sa) > 1 else x,
                                                p.transpose(w, -1, -2) if trans and len(sb) > 1 else w))
    for number, shape in enumerate([(3,), (2, 3), (2, 4, 3), (2, 1, 4, 3), (0, 2, 3)]):
        for bias in [False, True]:
            data = [torch.randn(shape), torch.randn(5, 3)] + ([torch.randn(5)] if bias else [])
            seeded_case(f'extended-linear-{number}-{bias}', data,
                        lambda *xs: F.linear(*xs),
                        lambda p, *xs: p.make(15, xs[0], xs[1], xs[2] if bias else 0, count=3 if bias else 2))
    index_specs = [((4,), 0, [3, 0, 3]), ((2, 3), -1, [2, 0, 2]),
                   ((2, 3, 4), 1, [2, 0]), ((2, 1, 3, 4), -2, [2, 2, 0]),
                   ((2, 3), 0, []), ((0, 3), -1, [2, 0])]
    for number, (shape, axis, indices) in enumerate(index_specs):
        index = torch.tensor(indices, dtype=torch.int64)
        for trans in [False, True]:
            x = torch.randn(shape)
            if trans and len(shape) > 1: x = x.transpose(-1, -2).contiguous()
            seeded_case(f'extended-index-select-{number}-{trans}', [x],
                        lambda x: torch.index_select(x.transpose(-1, -2) if trans and x.ndim > 1 else x, axis, index),
                        lambda p, x: p.index_select(p.transpose(x, -1, -2) if trans and len(shape) > 1 else x, axis, index))
    gather_specs = [((), 0, torch.tensor(0)), ((3,), -1, torch.tensor([2, 0, 2])),
                    ((2, 3), 1, torch.tensor([[2, 1], [0, 0]])),
                    ((2, 3), 0, torch.tensor([[1, 0, 1]])),
                    ((2, 3, 4), 1, torch.tensor([[[2, 0, 1], [0, 2, 2]]])),
                    ((2, 1, 3, 4), -2, torch.arange(12).reshape(2, 1, 2, 3) % 3),
                    ((2, 0, 3), 1, torch.empty(2, 0, 3, dtype=torch.int64)),
                    ((0, 2), 0, torch.empty(0, 9, dtype=torch.int64))]
    for number, (shape, axis, index) in enumerate(gather_specs):
        for trans in [False, True]:
            x = torch.randn(shape)
            if trans and len(shape) > 1: x = x.transpose(-1, -2).contiguous()
            seeded_case(f'extended-gather-{number}-{trans}', [x],
                        lambda x: torch.gather(x.transpose(-1, -2) if trans and x.ndim > 1 else x, axis, index),
                        lambda p, x: p.gather(p.transpose(x, -1, -2) if trans and len(shape) > 1 else x, axis, index))
    seeded_case('extended-seeded-scalar-leaf', [torch.tensor(.75)], lambda x: x, lambda p, x: x)
    seeded_case('extended-seeded-vector-leaf', [torch.randn(4)], lambda x: x, lambda p, x: x)
    seeded_case('extended-seeded-transpose', [torch.randn(2, 3)], lambda x: x.t(), lambda p, x: p.transpose(x))
    seeded_case('extended-seeded-shared-matmul', [torch.randn(3, 3)], lambda x: x @ x, lambda p, x: p.matmul(x, x))
    # A composed no-mask/no-dropout attention path exercises all three gradients.
    attention_specs = [((2, 3), (4, 3), (4, 5)),
                       ((2, 2, 3), (2, 4, 3), (2, 4, 5)),
                       ((2, 1, 2, 3), (1, 3, 4, 3), (1, 3, 4, 5))]
    for number, shapes in enumerate(attention_specs):
        scale = f32(1.0 / math.sqrt(shapes[0][-1]))
        def native_attention(p, q, k, v):
            scores = p.matmul(q, p.transpose(k, -1, -2))
            scores = p.binary(3, scores, p.tensor(scale))
            return p.matmul(p.softmax(scores, -1, False), v)
        seeded_case(f'extended-attention-composition-{number}', [torch.randn(s) for s in shapes],
                    lambda q, k, v: F.scaled_dot_product_attention(q, k, v, dropout_p=0., scale=scale),
                    native_attention, rtol=2e-4, atol=2e-5)



def training_sessions():
    for mode, steps in [('SGD', 1024), ('Adam', 512), ('AdamW', 512)]:
        p = Program()
        parameter = torch.nn.Parameter(torch.tensor([-.75, .25, 1.5, -1.]))
        target = torch.tensor([.5, -.5, .25, .75])
        x = p.tensor(parameter, True)
        m = p.tensor(torch.zeros(4)) if mode != 'SGD' else 0
        v = p.tensor(torch.zeros(4)) if mode != 'SGD' else 0
        target_id = p.tensor(target)
        keep = p.next_id - 1
        lr, b1, b2, eps, wd = map(f32, [2**-8, .5, .75, 2**-14, 2**-6])
        if mode == 'SGD':
            optimizer = torch.optim.SGD([parameter], lr=lr, foreach=False, fused=False)
        else:
            cls = torch.optim.Adam if mode == 'Adam' else torch.optim.AdamW
            optimizer = cls([parameter], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd, foreach=False, fused=False)
        expected = {}
        for iteration in range(1, steps + 1):
            p.zero(x)
            loss = p.mse(x, target_id)
            p.backward(loss)
            optimizer.zero_grad(set_to_none=False)
            reference = F.mse_loss(parameter, target)
            reference.backward()
            if iteration == 1 or iteration % 128 == 0:
                expected[p.read(loss)] = reference.detach().clone()
                expected[p.read(x, True)] = parameter.grad.detach().clone()
            if mode == 'SGD':
                p.sgd(x, lr)
            else:
                p.adam(x, m, v, iteration, lr, b1, b2, eps, wd, mode == 'AdamW')
            optimizer.step()
            p.reset_graph(keep)
            if iteration == 1 or iteration % 128 == 0:
                expected[p.read(x)] = parameter.detach().clone()
        if mode != 'SGD':
            expected[p.read(m)] = optimizer.state[parameter]['exp_avg'].clone()
            expected[p.read(v)] = optimizer.state[parameter]['exp_avg_sq'].clone()
        compare(f'session-{mode}-{steps}-steps', p, expected, rtol=2e-4, atol=2e-5)

    p = Program()
    leaf = torch.tensor([.5, -.25, 1.], requires_grad=True)
    x = p.tensor(leaf, True)
    expected = {}
    for iteration in range(4):
        p.backward(p.reduce(7, p.binary(3, x, x)))
        (leaf * leaf).sum().backward()
        p.reset_graph(1)
        expected[p.read(x, True)] = leaf.grad.clone()
        expected[p.read(x)] = leaf.detach().clone()
    compare('session-reset-preserves-accumulated-gradients', p, expected)

    p = Program()
    expected = {}
    for iteration in range(4):
        leaf = torch.tensor([iteration + .5, -.25], requires_grad=True)
        x = p.tensor(leaf, True)
        p.backward(p.reduce(7, p.binary(3, x, x)))
        (leaf * leaf).sum().backward()
        expected[p.read(x, True)] = leaf.grad.clone()
        p.reset_graph(0)
    compare('session-empty-preserved-prefix', p, expected)

    p = Program()
    empty = torch.empty(0, 3, requires_grad=True)
    value = torch.tensor([.25, -.5], requires_grad=True)
    a, x = p.tensor(empty, True), p.tensor(value, True)
    expected = {}
    for iteration in range(4):
        p.seeded(a, p.tensor(torch.empty(0, 3)))
        empty.backward(torch.empty_like(empty))
        p.backward(p.reduce(7, p.binary(3, x, x)))
        (value * value).sum().backward()
        p.reset_graph(2)
        expected[p.read(a, True)] = empty.grad.clone()
        expected[p.read(x, True)] = value.grad.clone()
    compare('session-empty-kept-tensor', p, expected)


def compare_join_metadata(name, program, expected_metadata, empty_reads=()):
    row={'case':name,'status':'PASS','rtol':0.,'atol':0.,'compared_values':0,
         'max_absolute_error':0.,'max_relative_error_scaled_by_atol':0.,'mismatches':0}
    try:
        result=subprocess.run([str(EXE)],input=program.encode(),capture_output=True,
                              timeout=120,env={'PATH':''})
        assert result.returncode==0 and not result.stderr
        raw=result.stdout
        assert struct.unpack_from('<8sQQQ',raw)==(b'HTORSP01',1,len(program.commands),0)
        at=32
        for index,command in enumerate(program.commands):
            code,error,value,size=struct.unpack_from('<4Q',raw,at);at+=32
            payload=raw[at:at+size];at+=size
            assert code==command[0] and error==0 and len(payload)==size
            if index in expected_metadata:
                reference=expected_metadata[index]
                shape=tuple(reference.shape);stride=tuple(reference.stride())
                wanted=(reference.ndim,reference.numel(),int(reference.requires_grad),
                        *shape,*([0]*(4-len(shape))),*stride,*([0]*(4-len(stride))))
                actual=struct.unpack('<13Q',payload)
                assert actual[:11]==wanted,(index,actual[:11],wanted)
                row['compared_values']+=11
            if index in empty_reads:
                assert value==0 and size==0
        assert at==len(raw)
    except Exception as exc:
        row.update(status='FAIL',error=str(exc)[:2000])
    ROWS.append(row)

def concatenation():
    def values(shape,offset=0):
        return (torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)+offset-3)/8
    specs=[
        ('cat-vector',[values((0,)),values((3,)),values((2,),1)],False,-1),
        ('cat-matrix-axis0',[values((2,3)),values((0,3)),values((1,3),1)],False,0),
        ('cat-matrix-axis1',[values((2,1)),values((2,3),1)],False,1),
        ('cat-rank3',[values((2,1,3)),values((2,2,3),1)],False,-2),
        ('cat-rank4',[values((2,1,1,3)),values((2,1,2,3),1)],False,-2),
        ('cat-other-empty',[values((2,0,3)),values((3,0,3),1)],False,0),
        ('stack-scalars',[values(()),values((),1)],True,-1),
        ('stack-vector',[values((3,)),values((3,),1)],True,-1),
        ('stack-matrix',[values((2,3)),values((2,3),1)],True,1),
        ('stack-rank3',[values((2,1,3)),values((2,1,3),1)],True,0),
        ('stack-empty',[values((2,0,3)),values((2,0,3),1)],True,-1),
    ]
    for name,data,stack,axis in specs:
        operation=torch.stack if stack else torch.cat
        function_case(name,data,
                      lambda *args,operation=operation,axis=axis:operation(args,dim=axis),
                      lambda p,*ids,stack=stack,axis=axis:p.join(ids,axis,stack))
    function_case('cat-strided',[values((2,3)),values((2,3),1)],
                  lambda a,b:torch.cat([a.t(),b.t()],dim=-1),
                  lambda p,a,b:p.join([p.transpose(a),p.transpose(b)],-1))
    function_case('cat-skipped-only-grad',[torch.empty(0),values((2,3))],
                  lambda empty,x:torch.cat([empty,x,empty],dim=1),
                  lambda p,empty,x:p.join([empty,x,empty],1),requires=[True,False])
    for stack in (False,True):
        operation=torch.stack if stack else torch.cat
        function_case(('stack' if stack else 'cat')+'-shared-view-square',[values((2,2))],
                      lambda x,operation=operation:operation([x,x.t(),x*x],dim=1),
                      lambda p,x,stack=stack:p.join([x,p.transpose(x),p.binary(3,x,x)],1,stack))
    for axis in (-(1<<63),-999,-2,-1,0,1,999,(1<<63)-1):
        p=Program()
        leaf=torch.empty(0,requires_grad=True);constant=torch.empty(0)
        a,b=p.tensor(leaf,True),p.tensor(constant)
        reference=torch.cat([leaf,constant,leaf],dim=axis)
        output=p.join([a,b,a],axis)
        reference.backward(torch.empty_like(reference))
        assert leaf.grad is not None and leaf.grad.numel()==0
        p.seeded(output,p.tensor(torch.empty_like(reference)))
        meta=p.command(13,output)
        reads=[p.read(output),p.read(a,True)]
        compare_join_metadata('cat-all-empty-axis-'+str(axis),p,{meta:reference},reads)
    p=Program();expected={};empty_reads=[]
    for shape in ((2,0,3),(0,2,3),(2,3,0),
                  (0,262144,262144,262144),(262144,0,262144,262144),
                  (262144,262144,0,262144),(262144,262144,262144,0)):
        reference=torch.empty(shape)
        native=p.tensor(reference)
        expected[p.command(13,native)]=reference
        empty_reads.append(p.read(native))
    for shape in ((2,0,3),(0,2),(512,512,0)):
        a=torch.empty(shape,requires_grad=True);b=torch.empty(shape,requires_grad=True)
        ai,bi=p.tensor(a,True),p.tensor(b,True)
        for stack,axis in ((False,0),(False,-1),(True,0),(True,-1)):
            reference=(torch.stack if stack else torch.cat)([a,b],dim=axis)
            native=p.join([ai,bi],axis,stack)
            expected[p.command(13,native)]=reference
            empty_reads.append(p.read(native))
    compare_join_metadata('concat-empty-shape-and-stride-metadata',p,expected,empty_reads)
    for stack in (False,True):
        p=Program();expected={}
        leaf=values((2,2)).requires_grad_();native=p.tensor(leaf,True)
        lr=f32(2**-7)
        for step in range(12):
            p.zero(native);leaf.grad=None
            square=p.binary(3,native,native)
            output=p.join([native,p.transpose(native),square],1,stack)
            reference=(torch.stack if stack else torch.cat)([leaf,leaf.t(),leaf*leaf],dim=1)
            seed=(torch.arange(reference.numel(),dtype=torch.float32).reshape(reference.shape)%9-4)/16
            reference.backward(seed)
            p.seeded(output,p.tensor(seed))
            expected[p.read(output)]=reference.detach().clone()
            expected[p.read(native,True)]=leaf.grad.clone()
            p.sgd(native,lr)
            with torch.no_grad():leaf.sub_(lr*leaf.grad)
            p.reset_graph(1)
            expected[p.read(native)]=leaf.detach().clone()
        compare(('stack' if stack else 'cat')+'-training-12-resets',p,expected)
    p=Program();leaf=torch.tensor([2.],requires_grad=True)
    a=p.tensor(leaf,True);out=p.join([a]*256,0)
    reference=torch.cat([leaf]*256,dim=0);reference.sum().backward()
    p.seeded(out,p.tensor(torch.ones_like(reference)))
    compare('cat-256-repeated-parents',p,{p.read(out):reference.detach(),p.read(a,True):leaf.grad.clone()})

def _indexed_case(name, operation, av, bv, index, dim=0, alpha=1.,
                  requires=(True, True), shared=False, transposed=False,
                  repeat=1):
    av = torch.as_tensor(av, dtype=torch.float32)
    bv = torch.as_tensor(bv, dtype=torch.float32)
    index = torch.as_tensor(index, dtype=torch.int64)
    if transposed:
        av, bv = av.transpose(0, 1).contiguous(), bv.transpose(0, 1).contiguous()
    a = av.detach().clone().requires_grad_(requires[0])
    b = a if shared else bv.detach().clone().requires_grad_(requires[1])
    p = Program()
    ai = p.tensor(a, requires[0])
    bi = ai if shared else p.tensor(b, requires[1])
    aa, bb, aio, bio = a, b, ai, bi
    if transposed:
        aa, bb = a.transpose(0, 1), b.transpose(0, 1)
        aio = p.transpose(ai)
        bio = aio if shared else p.transpose(bi)
        if index.ndim == 2:
            index = index.transpose(0, 1).contiguous().transpose(0, 1)
    if operation == 'scatter_add':
        result = torch.scatter_add(aa, dim, index, bb)
        output = p.scatter_add(aio, dim, index, bio)
    else:
        result = torch.index_add(aa, dim, index, bb, alpha=alpha)
        output = p.index_add(aio, dim, index, bio, alpha)
    seed = ((torch.arange(result.numel(), dtype=torch.float32) % 13 - 6) / 4).reshape(result.shape)
    seed_id = p.tensor(seed)
    for _ in range(repeat):
        result.backward(seed, retain_graph=True)
        p.seeded(output, seed_id)
    expected = {p.read(output): result.detach().clone()}
    if requires[0]:
        expected[p.read(ai, True)] = a.grad.detach().clone()
    if requires[1] and not shared:
        expected[p.read(bi, True)] = b.grad.detach().clone()
    compare('indexed-' + name, p, expected, rtol=4e-5, atol=5e-6)

def _indexed_reject(name, operation, ashape, bshape, index, dim, phase, status):
    row = {'case': 'indexed-reject-' + name, 'status': 'PASS', 'compared_values': 0,
           'expected_native_status': status, 'expected_phase': phase,
           'self_shape': list(ashape), 'source_shape': list(bshape),
           'index_shape': list(index.shape), 'max_absolute_error': 0.,
           'max_relative_error_scaled_by_atol': 0., 'mismatches': 0}
    try:
        a = torch.zeros(ashape, dtype=torch.float32, requires_grad=True)
        b = torch.ones(bshape, dtype=torch.float32, requires_grad=True)
        observed = None
        try:
            out = (torch.scatter_add(a, dim, index, b) if operation == 'scatter_add'
                   else torch.index_add(a, dim, index, b, alpha=.25))
        except (RuntimeError, IndexError) as exc:
            observed = {'phase': 'forward', 'type': type(exc).__name__, 'message': str(exc)}
        if observed is None and phase == 'backward':
            try:
                out.backward(torch.ones_like(out))
            except (RuntimeError, IndexError) as exc:
                observed = {'phase': 'backward', 'type': type(exc).__name__, 'message': str(exc)}
        assert observed is not None and observed['phase'] == phase, observed
        row['pytorch_exception'] = observed
        row['pytorch_gradient_shapes'] = {'self': None if a.grad is None else list(a.grad.shape),
                                         'source': None if b.grad is None else list(b.grad.shape)}
        p = Program()
        ai, bi = p.tensor(a, True), p.tensor(b, True)
        output = (p.scatter_add(ai, dim, index, bi) if operation == 'scatter_add'
                  else p.index_add(ai, dim, index, bi, .25))
        expected = {}
        if phase == 'backward':
            expected[p.read(output)] = out.detach().clone()
            p.seeded(output, p.tensor(torch.ones_like(out)))
        result = subprocess.run([str(EXE)], input=p.encode(), capture_output=True,
                                timeout=120, env={'PATH': ''})
        assert result.returncode == 0 and not result.stderr
        raw = result.stdout
        magic, version, count, error = struct.unpack_from('<8sQQQ', raw)
        assert (magic, version, count, error) == (b'HTORSP01', 1, len(p.commands), status)
        offset = 32
        for i in range(count):
            code, error, value, size = struct.unpack_from('<4Q', raw, offset)
            offset += 32
            payload = raw[offset:offset + size]
            offset += size
            assert code == p.commands[i][0] and len(payload) == size
            assert error == (status if i == count - 1 else 0)
            if i in expected:
                actual = torch.tensor(struct.unpack('<' + str(value) + 'f', payload))
                target = expected[i].reshape(-1)
                assert actual.shape == target.shape and torch.equal(actual, target)
                row['compared_values'] += target.numel()
        assert offset == len(raw)
    except Exception as exc:
        row.update(status='FAIL', error=type(exc).__name__ + ': ' + str(exc))
    ROWS.append(row)

def scatter_index_extensions():
    start = len(ROWS)
    generator = torch.Generator().manual_seed(1847)
    for rank in range(1, 5):
        for dim in range(rank):
            shape = [2] * rank
            shape[dim] = 3
            a = torch.randn(shape, generator=generator)
            b = torch.randn(shape, generator=generator)
            index = torch.randint(3, shape, generator=generator)
            _indexed_case(f'scatter-rank{rank}-axis{dim}', 'scatter_add', a, b, index, dim - rank, repeat=2)
            _indexed_case(f'index-rank{rank}-axis{dim}', 'index_add', a, b, torch.tensor([2, 0, 2]), dim - rank, -.25)
    for operation in ('scatter_add', 'index_add'):
        a = torch.tensor([[1., -2., 3.], [0., 4., -1.]])
        b = torch.tensor([[3., 1., 2.], [-1., 0., 5.]])
        index = torch.tensor([[2, 0, 2], [1, 1, 0]]) if operation == 'scatter_add' else torch.tensor([2, 0, 2])
        _indexed_case(operation + '-noncontiguous', operation, a, b, index, -1, -.5, transposed=True)
        _indexed_case(operation + '-shared', operation, a, a, index, 1, .25, shared=True, repeat=3)
        _indexed_case(operation + '-shared-transpose', operation, a, a, index, 1, .5, shared=True, transposed=True)
        _indexed_case(operation + '-scalar', operation, 2., 4., torch.tensor(0), -1, .25)
        _indexed_case(operation + '-scalar-vector-index', operation, 2., 4., torch.tensor([0]), 0, -.5)
    _indexed_case('scatter-scalar-source', 'scatter_add', [1., 2.], 4., torch.tensor([1]))
    _indexed_case('scatter-empty-broadcast', 'scatter_add', torch.zeros(2, 3), torch.ones(1, 3), torch.empty(0, 3, dtype=torch.int64))
    _indexed_case('scatter-empty-rank4-scalar', 'scatter_add', torch.zeros(2, 3), 4., torch.empty(0, 2, 3, 4, dtype=torch.int64))
    _indexed_case('scatter-empty-rank4-self-only', 'scatter_add', torch.zeros(2, 3), torch.ones(4, 5), torch.empty(0, 2, 3, 4, dtype=torch.int64), requires=(True, False))
    _indexed_case('scatter-smaller-index-self-only', 'scatter_add', torch.ones(2, 3), torch.ones(3, 4), torch.tensor([[0, 1, 0], [1, 0, 1]]), requires=(True, False))
    _indexed_case('index-empty-indices', 'index_add', torch.ones(2, 3), torch.empty(0, 3), torch.empty(0, dtype=torch.int64), alpha=-.5)
    _indexed_case('index-empty-other-axis', 'index_add', torch.empty(2, 0), torch.empty(3, 0), torch.tensor([1, 0, 1]), alpha=-.5)
    _indexed_case('index-empty-bad-indices-self-only', 'index_add', torch.empty(2, 0), torch.empty(3, 0), torch.tensor([99, -1, 44]), requires=(True, False))
    for i, alpha in enumerate((0., -0., 1., -1., .25, 3.5)):
        _indexed_case(f'index-alpha-{i}', 'index_add', [1., -2., 3.], [2., 4., -1., 3.], torch.tensor([1, 0, 1, 1]), alpha=alpha)
    for i, (shape, dim) in enumerate([((), 0), ((1,), 0), ((33,), 0), ((1, 1), 0), ((1, 33), 0), ((33, 1), 1)]):
        index = torch.arange(shape[dim] if shape else 1)
        _indexed_case(f'index-alpha-overflow-cancellation-{i}', 'index_add', torch.full(shape, -2e38), torch.full(shape, 2e38), index, dim, 2.)
    for operation in ('scatter_add', 'index_add'):
        p = Program()
        a = torch.tensor([1., 2.], requires_grad=True)
        b = torch.tensor([3., 4.], requires_grad=True)
        ai, bi = p.tensor(a, True), p.tensor(b, True)
        expected = {}
        index = torch.tensor([1, 0])
        for _ in range(3):
            p.reset_graph(2)
            if operation == 'scatter_add':
                out, output = torch.scatter_add(a, 0, index, b), p.scatter_add(ai, 0, index, bi)
            else:
                out, output = torch.index_add(a, 0, index, b, alpha=.5), p.index_add(ai, 0, index, bi, .5)
            seed = torch.tensor([1., -2.])
            out.backward(seed)
            p.seeded(output, p.tensor(seed))
            expected[p.read(output)] = out.detach().clone()
            expected[p.read(ai, True)], expected[p.read(bi, True)] = a.grad.clone(), b.grad.clone()
        compare('indexed-' + operation + '-reset-accumulation', p, expected)
    for operation in ('scatter_add', 'index_add'):
        _indexed_case(operation + '-special-values', operation, [1., float('inf'), 3.],
                      [float('nan'), -float('inf'), 5.], torch.tensor([0, 1, 2]), alpha=-.5)
        _indexed_case(operation + '-source-only', operation, [1., 2.], [3., 4.], torch.tensor([1, 1]), alpha=.25, requires=(False, True))
    _indexed_case('index-zero-alpha-nonfinite', 'index_add', [1., 2.], [float('inf'), float('nan')], torch.tensor([0, 1]), alpha=0.)
    for operation in ('scatter_add', 'index_add'):
        p = Program()
        a = torch.tensor([[1., 2.], [-1., 3.]], requires_grad=True)
        b = torch.tensor([[2., 0.], [1., -1.]], requires_grad=True)
        ai, bi = p.tensor(a, True), p.tensor(b, True)
        matrix, mid = a @ b, p.matmul(ai, bi)
        index = torch.tensor([[1, 0], [1, 1]]) if operation == 'scatter_add' else torch.tensor([1, 1])
        out = (torch.scatter_add(matrix, 1, index, matrix) if operation == 'scatter_add'
               else torch.index_add(matrix, 1, index, matrix, alpha=.25))
        oid = (p.scatter_add(mid, 1, index, mid) if operation == 'scatter_add'
               else p.index_add(mid, 1, index, mid, .25))
        seed = torch.tensor([[1., -2.], [.5, 3.]])
        out.backward(seed)
        p.seeded(oid, p.tensor(seed))
        compare('indexed-' + operation + '-matmul-shared-chain', p,
                {p.read(oid): out.detach(), p.read(ai, True): a.grad, p.read(bi, True): b.grad})
    bad = [
        ('scatter-smaller-index', 'scatter_add', (2, 3), (3, 4), torch.tensor([[0, 1, 0], [1, 0, 1]]), 0, 'backward', 5),
        ('scatter-scalar-index-vector-source', 'scatter_add', (), (1,), torch.tensor(0), 0, 'backward', 5),
        ('scatter-empty-rank-mismatch', 'scatter_add', (2, 3), (4, 5), torch.empty(0, dtype=torch.int64), 0, 'backward', 5),
        ('scatter-negative-index', 'scatter_add', (2,), (1,), torch.tensor([-1]), 0, 'forward', 7),
        ('scatter-source-rank', 'scatter_add', (2, 3), (6,), torch.zeros(2, 3, dtype=torch.int64), 0, 'forward', 2),
        ('scatter-empty-axis', 'scatter_add', (2, 3), (0, 3), torch.empty(0, dtype=torch.int64), 2, 'forward', 2),
        ('index-empty-other-axis', 'index_add', (2, 0), (3, 0), torch.tensor([99, -1, (1 << 63) - 1]), 0, 'backward', 7),
        ('index-empty-selected-axis', 'index_add', (0, 3), (2, 3), torch.tensor([0, 1]), 0, 'backward', 7),
        ('index-rank2-empty', 'index_add', (2, 3), (0, 3), torch.empty(0, 2, dtype=torch.int64), 0, 'forward', 2),
        ('index-scalar-vector-source', 'index_add', (), (1,), torch.tensor([0]), 0, 'forward', 2),
        ('index-vector-scalar-source', 'index_add', (1,), (), torch.tensor([0]), 0, 'forward', 2),
        ('index-empty-shape', 'index_add', (2, 3), (0, 4), torch.empty(0, dtype=torch.int64), 0, 'forward', 2),
        ('index-count', 'index_add', (2, 3), (1, 3), torch.empty(0, dtype=torch.int64), 0, 'forward', 2),
        ('index-large-i64', 'index_add', (2,), (1,), torch.tensor([(1 << 63) - 1]), 0, 'forward', 7),
    ]
    for arguments in bad:
        _indexed_reject(*arguments)
    _indexed_case('scatter-late-zero-large-extents', 'scatter_add', torch.arange(1, 7, dtype=torch.float32).reshape(2, 3), 4., torch.empty((262144, 262144, 0, 262144), dtype=torch.int64))
    assert len(ROWS) - start == 74, len(ROWS) - start

def compare_join_layout(name, program, expected_values, expected_metadata):
    row={'case':name,'status':'PASS','rtol':0.,'atol':0.,'compared_values':0,
         'max_absolute_error':0.,'max_relative_error_scaled_by_atol':0.,'mismatches':0}
    try:
        result=subprocess.run([str(EXE)],input=program.encode(),capture_output=True,
                              timeout=120,env={'PATH':''})
        assert result.returncode==0 and not result.stderr
        raw=result.stdout
        assert struct.unpack_from('<8sQQQ',raw)==(b'HTORSP01',1,len(program.commands),0)
        offset=32
        for index,command in enumerate(program.commands):
            code,error,value,size=struct.unpack_from('<4Q',raw,offset);offset+=32
            payload=raw[offset:offset+size];offset+=size
            assert code==command[0] and error==0 and len(payload)==size
            if index in expected_values:
                assert size==value*4
                actual=torch.tensor(struct.unpack('<'+str(value)+'f',payload),dtype=torch.float32)
                reference=expected_values[index].detach().reshape(-1).to(dtype=torch.float32,device='cpu')
                torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                row['compared_values']+=reference.numel()
            if index in expected_metadata:
                reference=expected_metadata[index]
                shape=tuple(reference.shape);strides=tuple(reference.stride())
                wanted=(reference.ndim,reference.numel(),int(reference.requires_grad),
                        *shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(strides))))
                actual=struct.unpack('<13Q',payload)
                assert actual[:11]==wanted,(index,actual[:11],wanted)
                row['compared_values']+=11
        assert offset==len(raw)
    except Exception as exc:
        row.update(status='FAIL',error=str(exc)[:2000])
    ROWS.append(row)

def concatenation_layouts():
    specs=[('cat-channels',1),('cat-channels',3),('cat-mixed-empty',1),
           ('cat-mixed-row',1),('cat-singletons',1),
           ('stack-singletons',3),('stack-singletons',1),('stack-nontrivial',3)]
    for kind,axis in specs:
        p=Program();expected={};stack=kind.startswith('stack')
        if kind in ('cat-channels','cat-mixed-empty','cat-mixed-row'):
            leaf=(torch.arange(120,dtype=torch.float32).reshape(2,3,4,5)/8).requires_grad_()
            native=p.tensor(leaf,True)
            reference=leaf.transpose(1,3).transpose(2,3)
            view=p.transpose(p.transpose(native,1,3),2,3)
        elif kind=='cat-singletons':
            leaf=(torch.arange(6,dtype=torch.float32).reshape(2,3,1,1)/8).requires_grad_()
            native=p.tensor(leaf,True);reference=leaf.transpose(1,2);view=p.transpose(native,1,2)
        else:
            shape=(2,3,4) if kind=='stack-nontrivial' else (2,3,1)
            leaf=(torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)/8).requires_grad_()
            native=p.tensor(leaf,True);reference=leaf.transpose(1,2);view=p.transpose(native,1,2)
        ids=[view,view];values=[reference,reference]
        other=None;other_id=0
        if kind=='cat-mixed-empty':
            other=torch.empty(0,requires_grad=True);other_id=p.tensor(other,True)
            ids=[other_id,view];values=[other,reference]
        if kind=='cat-mixed-row':
            other=(reference.detach().contiguous()+1).requires_grad_();other_id=p.tensor(other,True)
            ids=[view,other_id];values=[reference,other]
        output=p.join(ids,axis,stack)
        result=(torch.stack if stack else torch.cat)(values,dim=axis)
        seed=(torch.arange(result.numel(),dtype=torch.float32).reshape(result.shape)%7-3)/8
        result.backward(seed)
        p.seeded(output,p.tensor(seed))
        expected[p.read(output)]=result.detach().clone()
        expected[p.read(native,True)]=leaf.grad.clone()
        if other is not None:expected[p.read(other_id,True)]=other.grad.clone()
        compare_join_layout('layout-'+kind+'-axis'+str(axis),p,expected,{p.command(13,output):result})

def main():
    started = time.monotonic()
    sha = hashlib.sha256(EXE.read_bytes()).hexdigest()
    training_result = None
    setup_error = None
    try:
        operators()
        spatial_modules()
        optimizers()
        training_result = training()
        extensions()
        training_sessions()
        concatenation()
        concatenation_layouts()
        scatter_index_extensions()
    except Exception as exc:
        setup_error = type(exc).__name__ + ': ' + str(exc)
    changed = sha != hashlib.sha256(EXE.read_bytes()).hexdigest()
    failed = sum(row['status'] == 'FAIL' for row in ROWS)
    passed = setup_error is None and not changed and failed == 0 and len(ROWS) == 268 and len({row["case"] for row in ROWS}) == 268 and PROVENANCE['status'] == 'PASS'
    report = {'schema': 'heptatorch.live-pytorch-differential.v1', 'status': 'PASS' if passed else 'FAIL',
              'oracle': 'real pinned CPU PyTorch forward, autograd and torch.optim; no copied derivative formulas',
              'torch_version': torch.__version__, 'torch_git_version': torch.version.git_version,
              'upstream_commit': PIN, 'torch_git_matches_upstream_pin': torch.version.git_version == PIN,
              'oracle_build_commit': EXPECTED['git_commit'],
              'torch_git_matches_oracle_build_pin': torch.version.git_version == EXPECTED['git_commit'],
              'oracle_provenance': PROVENANCE,
              'native_executable_sha256': sha, 'native_unchanged_during_run': not changed,
              'seed': 1729, 'expected_cases': 268, 'cases': len(ROWS), 'failed_cases': failed, 'setup_error': setup_error,
              'compared_values': sum(row['compared_values'] for row in ROWS), 'rows': ROWS,
              'training': training_result, 'wall_seconds': time.monotonic() - started}
    emit(report)
    if not passed: raise SystemExit(1)


if __name__ == '__main__':
    main()
