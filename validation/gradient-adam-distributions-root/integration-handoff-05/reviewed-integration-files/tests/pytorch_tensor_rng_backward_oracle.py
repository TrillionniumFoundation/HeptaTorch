"""Independent live PyTorch comparisons for one native Tensor/RNG/backward owner.

Python transports typed wire requests and executes the independent reference.
All target allocation, RNG, Tensor arithmetic and gradients execute in HIA.
"""
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

MASK = (1 << 64) - 1
SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
SEEDS = [0, 1, 17, 5489, (1 << 31)-1, (1 << 32)-1, 1 << 32,
         (1 << 32)+17, (1 << 63)-1, 1 << 63, MASK-1, MASK]
LAYOUTS = {
    'transpose': ((4, 3), (1, 4), 0),
    'interleaved': ((2, 3), (9, 2), 1),
    'overlap-nonzero-strides': ((2, 3), (1, 1), 2),
    'channels-last': ((1, 2, 2, 3), (12, 1, 6, 2), 0),
}
EMPTIES = {
    'empty-contiguous': ((0,), (1,), 0),
    'empty-offset-999': ((0,), (1,), 999),
    'empty-zero-strides': ((0, 3), (0, 0), 0),
}
GENERATOR_CASES = ['clone-across-twist', 'state-restore-with-live-tensors', 'import-normal-cache-state']
PARTIAL_CASES = ['version-error-prefix', 'spent-saved-prefix', 'ready-leaf-before-error', 'two-generators-prefix']
RANGES = {
    'equal-interval': (0x3f800000, 0x3f800000, False),
    'rounded-upper-clamp': (0x3f800000, 0x3f800001, False),
    'subnormal-interval': (0x00000001, 0x00000010, False),
    'signed-zero-interval': (0x80000000, 0x00000000, False),
    'binary64-maximum-admission': (0xbf800000, 0x7f7fffff, False),
    'invalid-ordered-interval': (0x40000000, 0x3f800000, True),
}
PLAN = ([f'train-{seed}' for seed in SEEDS] + list(LAYOUTS) + list(EMPTIES)
        + GENERATOR_CASES + PARTIAL_CASES + list(RANGES))
assert len(PLAN) == len(set(PLAN)) == 32


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def scalar(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


class Payload:
    def __init__(self, raw):
        self.raw = bytes(raw)


def reference(record):
    assert 1 <= record <= 16383
    return 0xfe000000 | record


def packet(commands):
    end = 32 + 64 * len(commands)
    body, payload = bytearray(), bytearray()
    for command in commands:
        assert 1 <= len(command) <= 8
        words = []
        for argument in command:
            if isinstance(argument, Payload):
                while (end + len(payload)) % 8:
                    payload.append(0)
                words.append(end + len(payload)); payload.extend(argument.raw)
            else:
                words.append(int(argument) & MASK)
        body.extend(struct.pack('<8Q', *words, *([0] * (8-len(words)))))
    result = b'HTORCH03' + struct.pack('<3Q', len(commands), end+len(payload), 0) + body + payload
    assert len(result) <= 1 << 20
    return bytes(result)


class Native:
    def __init__(self, executable):
        self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': ''})
        self.frames = 0
        self.requests, self.responses = hashlib.sha256(), hashlib.sha256()

    def read(self, count):
        deadline, result = time.monotonic()+60, bytearray()
        while len(result) < count:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError(('native response', count, len(result), self.process.poll()))
            block = os.read(self.process.stdout.fileno(), count-len(result))
            assert block, ('native EOF', self.process.poll(), count, len(result))
            result.extend(block)
        return bytes(result)

    def run(self, commands, status=0, returned=None):
        body = packet(commands)
        request = b'HTOWNR01' + struct.pack('<Q', len(body)) + body
        self.process.stdin.write(request); self.process.stdin.flush()
        prefix = self.read(16)
        assert prefix[:8] == b'HTOWNS01'
        length, = struct.unpack_from('<Q', prefix, 8)
        assert 32 <= length <= 1 << 20
        raw = self.read(length)
        magic, version, count, error = struct.unpack_from('<8s3Q', raw)
        assert magic == b'HTORSP03' and version == 3 and error == status, (commands, version, error, status)
        assert self.process.poll() is None, 'Every response must arrive before process EOF'
        self.frames += 1; self.requests.update(request); self.responses.update(prefix+raw)
        if status:
            assert count == 0 and len(raw) == 32
            return []
        assert count == (len(commands) if returned is None else returned), (count, commands)
        records, at = [], 32
        for index in range(count):
            assert at+32 <= len(raw)
            opcode, record_error, value, size = struct.unpack_from('<4Q', raw, at); at += 32
            assert opcode == commands[index][0] and record_error == 0 and at+size <= len(raw)
            records.append({'opcode': opcode, 'value': value, 'payload': raw[at:at+size]}); at += size
        assert at == len(raw)
        return records

    def call(self, command, **kwargs):
        rows = self.run([command], **kwargs)
        return rows[0] if rows else None

    def close(self, failed=False):
        if failed:
            if self.process.poll() is None:
                self.process.kill()
            self.process.communicate(); return
        self.process.stdin.close(); self.process.stdin = None
        out, err = self.process.communicate(timeout=30)
        assert self.process.returncode == 0 and not out and not err, (self.process.returncode, out, err)


class Comparison:
    def __init__(self, native, torch):
        self.n, self.t = native, torch
        self.tensors, self.generators = {}, {}
        self.exact_words = self.state_bytes = self.versions = self.presences = 0
        self.reference_words, self.reference_states = hashlib.sha256(), hashlib.sha256()
        self.details = {}

    def full(self, shape, value=0.0, requires=False):
        tensor = self.n.call((38, len(shape), int(requires), *shape, *([0]*(4-len(shape))), bits(value)))['value']
        self.tensors[tensor] = self.t.full(shape, value, dtype=self.t.float32, requires_grad=requires)
        self.check(tensor, gradient=requires)
        return tensor

    def alias(self, tensor, shape, strides, offset):
        config = struct.pack('<10Q', len(shape), *shape, *([0]*(4-len(shape))),
                             *strides, *([0]*(4-len(strides))), offset)
        alias = self.n.call((69, tensor, Payload(config), 80))['value']
        self.tensors[alias] = self.t.as_strided(self.tensors[tensor], shape, strides, offset).detach()
        return alias

    def state(self, generator):
        expected = bytes(self.generators[generator].get_state().tolist())
        actual = self.n.call((63, generator))['payload']
        assert len(actual) == len(expected) == 5056 and actual == expected, (generator, 'serialized CPU Generator state')
        self.state_bytes += len(expected); self.reference_states.update(expected)
        assert self.n.call((62, generator))['value'] == self.generators[generator].initial_seed()
        return expected

    def create_generator(self, seed):
        generator = self.n.call((60, seed))['value']
        self.generators[generator] = self.t.Generator(device='cpu').manual_seed(seed)
        self.state(generator)
        return generator

    def clone_generator(self, generator):
        clone = self.n.call((65, generator))['value']
        self.generators[clone] = self.generators[generator].clone_state()
        self.state(generator); self.state(clone)
        return clone

    def restore(self, generator, state):
        self.n.call((64, generator, Payload(state), len(state)))
        self.generators[generator].set_state(self.t.frombuffer(bytearray(state), dtype=self.t.uint8).clone())
        self.state(generator)

    def values(self, tensor, gradient=False):
        reference_tensor = self.tensors[tensor].grad if gradient else self.tensors[tensor]
        assert reference_tensor is not None
        words = [int(x) & 0xffffffff for x in reference_tensor.detach().contiguous().reshape(-1).view(self.t.int32).tolist()]
        raw = struct.pack('<'+'I'*len(words), *words)
        actual = self.n.call((40 if gradient else 39, tensor, 0, len(words)))
        assert actual['value'] == len(words) and actual['payload'] == raw, (
            'gradient' if gradient else 'tensor', tensor, words,
            list(struct.unpack('<'+'I'*(len(actual['payload'])//4), actual['payload']))[:32])
        self.exact_words += len(words); self.reference_words.update(raw)

    def check(self, tensor, gradient=False):
        expected = self.tensors[tensor]
        record = self.n.call((13, tensor))
        assert len(record['payload']) == 128
        meta = struct.unpack('<16Q', record['payload'])
        assert meta[0] == expected.dim() and meta[1] == expected.numel()
        assert meta[2] == int(expected.requires_grad)
        assert tuple(meta[3:3+expected.dim()]) == tuple(expected.shape)
        assert tuple(meta[7:7+expected.dim()]) == tuple(expected.stride())
        assert meta[13] == expected._version, ('absolute parameter Version', tensor, meta[13], expected._version)
        assert meta[14] == expected.storage_offset()*expected.element_size(), ('storage byte offset', tensor, meta[14], expected.storage_offset())
        assert meta[15] == expected.untyped_storage().nbytes(), ('storage bytes', tensor, meta[15], expected.untyped_storage().nbytes())
        self.versions += 1; self.values(tensor)
        if gradient:
            assert expected.is_leaf and expected.requires_grad
            actual = self.n.call((59, tensor))['value']
            assert actual == int(expected.grad is not None), ('gradient presence', tensor, actual, expected.grad)
            self.presences += 1
            if expected.grad is not None:
                self.values(tensor, True)

    def uniform(self, tensor, generator, low=0.0, high=1.0, no_grad=False):
        self.n.call((68, tensor, generator, bits(low), bits(high), int(no_grad)))
        if no_grad:
            with self.t.no_grad():
                self.tensors[tensor].uniform_(low, high, generator=self.generators[generator])
        else:
            self.tensors[tensor].uniform_(low, high, generator=self.generators[generator])

    def backward(self, tensor, opcode=58):
        result = self.n.call((opcode, tensor, 0, 0))
        if opcode == 58:
            assert result['value'] == 0 and len(result['payload']) == 32
            assert struct.unpack_from('<Q', result['payload'])[0] == 0
        self.tensors[tensor].backward()

    def binary(self, op, left, right):
        result = self.n.call((2, op, left, right))['value']
        a, b = self.tensors[left], self.tensors[right]
        self.tensors[result] = a+b if op == 1 else a*b
        return result


def train(c, seed):
    p = c.full((2, 4), requires=True); g = c.create_generator(seed)
    parameter = c.tensors[p]
    optimizer = c.t.optim.SGD([parameter], lr=0.125, foreach=False, fused=False)
    opcode = 57 if SEEDS.index(seed) % 2 == 0 else 58
    for iteration in range(12):
        c.n.call((8, p)); optimizer.zero_grad(set_to_none=False)
        target = p
        if iteration % 2:
            target = c.alias(p, (4, 2), (1, 4), 0)
        c.uniform(target, g, -1.0, 1.0, no_grad=True)
        if target != p:
            c.n.call((41, target)); del c.tensors[target]
        rows = c.n.run([(2, 3, p, p), (4, 7, reference(1)),
                        (opcode, reference(2), 0, 0), (9, p, bits(0.125))])
        if opcode == 58:
            assert rows[2]['value'] == 0 and len(rows[2]['payload']) == 32
        loss = (parameter*parameter).sum(); loss.backward(); optimizer.step()
        c.check(p, True); c.state(g)
        c.n.run([(33, 1), (53,)])
        del loss
    c.details.update(seed=seed, updates=12, backward_opcode=opcode,
                     alternating_contiguous_and_transposed_fill=True,
                     exact_parameter_and_gradient_bits=True)


def layout(c, shape, strides, offset):
    p = c.full((32,), -3.0); g = c.create_generator(1729)
    a = c.alias(p, shape, strides, offset)
    c.uniform(a, g, -1.25, 2.5)
    c.check(a); c.check(p); c.state(g)
    c.n.run([(41, a), (53,)])
    c.details.update(shape=shape, strides=strides, offset=offset,
                     whole_backing_tensor_compared=True)


def generator_case(c, name):
    p = c.full((625,), -2.0); q = c.full((625,), -3.0)
    g = c.create_generator(5489)
    c.uniform(p, g); c.check(p); saved = c.state(g)
    if name == 'clone-across-twist':
        other = c.clone_generator(g)
        for _ in range(3):
            c.uniform(p, g, -2, 3); c.uniform(q, other, -2, 3)
            c.check(p); c.check(q); c.state(g); c.state(other)
        c.n.call((66, g)); del c.generators[g]
        c.uniform(q, other); c.check(q); c.state(other)
    elif name == 'state-restore-with-live-tensors':
        c.uniform(q, g); c.check(q); c.state(g)
        c.restore(g, saved)
        c.uniform(p, g); c.check(p); c.state(g)
        c.restore(g, saved)
        c.uniform(q, g); c.check(q); c.state(g)
    else:
        c.t.randn(1, dtype=c.t.float64, generator=c.generators[g])
        cache_state = bytes(c.generators[g].get_state().tolist())
        c.restore(g, cache_state)
        c.uniform(p, g); c.check(p); c.state(g)
        clone = c.clone_generator(g)
        c.uniform(q, clone, -0.5, 0.5); c.check(q); c.state(clone)
        c.n.call((61, g, MASK)); c.generators[g].manual_seed(MASK)
        c.state(g); c.uniform(p, g); c.check(p); c.state(g)
        c.details['imported_actual_normal_state_sha256'] = sha(cache_state)


def partial(c, name):
    p = c.full((), 2.0, True); coefficient = c.full((), 3.0)
    noise = c.full((4,), -1.0); g = c.create_generator(17)
    bad = c.binary(3, p, coefficient)
    root, leaves = bad, [p]
    if name == 'ready-leaf-before-error':
        z = c.full((), 5.0, True); scale = c.full((), 7.0)
        good = c.binary(3, z, scale); root = c.binary(1, bad, good); leaves.append(z)
    expected = 10 if name == 'spent-saved-prefix' else 6
    if expected == 10:
        c.backward(root)
    else:
        c.n.call((12, coefficient, 0, bits(4.0)))
        c.tensors[coefficient].fill_(4.0)
    other = c.create_generator(123) if name == 'two-generators-prefix' else g
    commands = [(68, noise, g, bits(-1.0), bits(1.0), 0), (58, root, 0, 0),
                (68, noise, other, bits(2.0), bits(3.0), 0)]
    records = c.n.run(commands, returned=2)
    failure = records[-1]
    assert failure['value'] == expected and len(failure['payload']) == 32
    detail = struct.unpack('<4Q', failure['payload']); assert detail[0] == expected
    c.tensors[noise].uniform_(-1.0, 1.0, generator=c.generators[g])
    try:
        c.tensors[root].backward()
    except RuntimeError as error:
        message = str(error)
        assert ('backward through the graph a second time' if expected == 10 else 'modified by an inplace operation') in message
    else:
        raise AssertionError('actual PyTorch did not reject the observed backward')
    for tensor in leaves:
        c.check(tensor, True)
    c.check(noise); c.check(coefficient); c.state(g)
    if other != g:
        c.state(other)
    c.details.update(native_backward_error=expected, completed_nodes=detail[1],
                     completed_leaf_accumulators=detail[2], failing_ad_opcode=detail[3],
                     prefix_consumed_and_suffix_not_executed=True)


def range_case(c, low_word, high_word, invalid):
    p = c.full((64,), -3.0); g = c.create_generator(12345)
    low, high = scalar(low_word), scalar(high_word)
    before = c.state(g)
    if invalid:
        c.n.call((68, p, g, low_word, high_word, 0), status=7)
        try:
            c.tensors[p].uniform_(low, high, generator=c.generators[g])
        except RuntimeError:
            pass
        else:
            raise AssertionError('actual PyTorch accepted invalid uniform interval')
        assert c.state(g) == before
    else:
        c.uniform(p, g, low, high)
        assert c.state(g) != before
    c.check(p)
    c.details.update(from_bits=low_word, to_bits=high_word, rejected=invalid)


def emit(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'frames', 'exact_binary32_words', 'serialized_state_bytes') if key in report}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'source', 'repo', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--transport-smoke', action='store_true')
    args = parser.parse_args()
    paths = {'native': args.native.resolve(), 'source': args.source.resolve(),
             'oracle': Path(__file__).resolve(), 'reference_support': args.repo.resolve()/'tests/pytorch_oracle.py',
             'lock': args.repo.resolve()/'compatibility/pytorch-oracle-lock.json'}
    output = args.output.resolve()
    before = {key: sha(path.read_bytes()) for key, path in paths.items()}
    assert before['native'] == args.expected_native_sha256 and before['source'] == args.expected_source_sha256
    assert before['reference_support'] == SUPPORT_SHA and before['lock'] == LOCK_SHA
    base = {'schema': 'heptatorch.live-pytorch-tensor-rng-backward.v1',
            'source_custody': before, 'native_executable_sha256': before['native'],
            'native_source_sha256': before['source'], 'case_plan': PLAN, 'expected_cases': 32,
            'pytorch_api_denominator': 2590, 'whole_pytorch_rewrite_complete': False,
            'full_autograd_gate_closed': False, 'full_rng_gate_closed': False}
    if args.prepare_only:
        emit(output, dict(base, status='NOT_RUN', cases=0, reason='Independent 32-case plan and inputs pinned before actual execution.'))
        return 0
    if args.transport_smoke:
        n = Native(paths['native'])
        try:
            rows = n.run([(38, 1, 1, 8, 0, 0, 0, 0), (60, 5489)])
            tensor, generator = [row['value'] for row in rows]
            n.call((68, tensor, generator, bits(-1.0), bits(1.0), 1))
            rows = n.run([(2, 3, tensor, tensor), (4, 7, reference(1)), (58, reference(2), 0, 0)])
            assert rows[-1]['value'] == 0 and len(rows[-1]['payload']) == 32
            assert n.call((59, tensor))['value'] == 1
            assert len(n.call((63, generator))['payload']) == 5056
            n.run([(9, tensor, bits(0.125)), (33, 1), (53,)])
            n.close()
        except Exception:
            n.close(True); raise
        emit(output, dict(base, status='NATIVE_SMOKE_ONLY', cases=1, frames=n.frames, pytorch_executed=False))
        return 0
    previous_argv, previous_cwd = sys.argv, Path.cwd()
    provenance_path = output.with_name(output.stem+'-provenance.json')
    try:
        os.chdir(args.repo)
        sys.argv = [str(paths['reference_support']), str(paths['native']), str(provenance_path), str(paths['lock'])]
        support = runpy.run_path(str(paths['reference_support']), run_name='tensor_rng_backward_reference_support')
    except (SystemExit, ModuleNotFoundError) as error:
        emit(output, dict(base, status='NOT_RUN', cases=0, reason=str(error),
                         reference_initialization=json.loads(provenance_path.read_bytes()) if provenance_path.exists() else None))
        return 2
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    provenance = support['PROVENANCE']
    assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11 and all(x is True for x in provenance['checks'].values())
    rows = []
    for name in PLAN:
        n = Native(paths['native']); c = Comparison(n, support['torch'])
        row = {'case': name, 'status': 'PASS'}
        try:
            if name.startswith('train-'):
                train(c, int(name.removeprefix('train-')))
            elif name in LAYOUTS or name in EMPTIES:
                layout(c, *(LAYOUTS | EMPTIES)[name])
            elif name in GENERATOR_CASES:
                generator_case(c, name)
            elif name in PARTIAL_CASES:
                partial(c, name)
            else:
                range_case(c, *RANGES[name])
            n.call((53,)); n.close()
        except Exception:
            row.update(status='FAIL', error=traceback.format_exc()); n.close(True)
        row.update(frames=n.frames, exact_binary32_words=c.exact_words, serialized_state_bytes=c.state_bytes,
                   absolute_version_comparisons=c.versions, gradient_presence_comparisons=c.presences,
                   request_stream_sha256=n.requests.hexdigest(), response_stream_sha256=n.responses.hexdigest(),
                   reference_words_sha256=c.reference_words.hexdigest(), reference_states_sha256=c.reference_states.hexdigest(),
                   details=c.details)
        rows.append(row)
    unchanged = {key: sha(path.read_bytes()) == before[key] for key, path in paths.items()}
    passed = all(row['status'] == 'PASS' for row in rows) and all(unchanged.values())
    report = dict(base, status='PASS' if passed else 'FAIL', cases=len(rows),
                  failed_cases=sum(row['status'] == 'FAIL' for row in rows), oracle_provenance=provenance,
                  source_files_unchanged_during_run=unchanged, rows=rows,
                  pytorch_executed=True, one_native_process_per_trajectory=True,
                  scope='Rank zero through four CPU binary32 Tensor fill, owned Generator state, bounded backward 57/58 and plain SGD trajectories. Full API, tracked nonleaf uniform autograd and optimizer-object lifecycle remain open.')
    for field in ('frames', 'exact_binary32_words', 'serialized_state_bytes', 'absolute_version_comparisons', 'gradient_presence_comparisons'):
        report[field] = sum(row[field] for row in rows)
    emit(output, report)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
