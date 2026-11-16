"""Exact CPU RNG sequences and serialized state, with independent references.

The C++ standard MT19937 and CPython engine are qualification references only.
The PyTorch mode additionally checks the exact pinned CPU wheel and all eleven
provenance conditions. No host generator is used by the native executable.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import runpy
import shutil
import struct
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
MASK = (1 << 64)-1
SOURCE_SHA = 'c1de63f5f046bfb20aa4d4f65cbffa6217a38a36c0dc63b05f08c7ddc7612032'
IMAGE_SHA = 'cffc7c2f6574bf08a184e7d4f84e41a38ce0fb9be2f645bf9c332c93c2621eac'
COMPILER_SHA = '87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54'
SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
f32 = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
f64 = lambda x: struct.unpack('<Q', struct.pack('<d', x))[0]
signed = lambda x: x if x < (1 << 63) else x-(1 << 64)


def packet(commands, seed=0, state=None):
    prefix = b'HTRNG001'+struct.pack('<IIQQ', state is not None, len(commands),
                                   seed if state is None else 0, 0 if state is None else len(state))
    return prefix+(state or b'')+b''.join(struct.pack('<IIQQQ', op, count, low & MASK, high & MASK, 0)
                                        for op, count, low, high in commands)


class Model:
    def __init__(self, initial, seed=0, state=None):
        self.initial = initial
        if state is None:
            self.seed(seed)
        else:
            self.restore(state)

    def seed(self, value):
        initial = self.initial(value)
        self.rng = random.Random()
        self.rng.setstate((3, tuple(initial), None))
        self.seed_value, self.fresh = value, True
        self.double_cache, self.double_valid = 0, 0
        self.float_cache, self.float_valid = 0, 0

    def restore(self, raw):
        assert len(raw) == 5056
        self.seed_value, left, seeded, index = struct.unpack_from('<QIIQ', raw)
        index &= 0xffffffff
        assert seeded and 1 <= left <= 624 and index <= 624
        assert (left, index) == (1, 0) or index and left+index == 625
        words = [x & 0xffffffff for x in struct.unpack_from('<624Q', raw, 24)]
        self.rng = random.Random(); self.rng.setstate((3, tuple(words+[index if index else 624]), None))
        self.fresh = index == 0
        self.double_valid = int(bool(struct.unpack_from('<I', raw, 5040)[0]))
        self.double_cache = struct.unpack_from('<Q', raw, 5024)[0] if self.double_valid else 0
        self.float_valid = int(bool(raw[5052]))
        self.float_cache = struct.unpack_from('<I', raw, 5048)[0] if self.float_valid else 0

    def state(self):
        words = self.rng.getstate()[1]
        index = 0 if self.fresh else words[-1]
        left = 1 if self.fresh else 625-index
        raw = bytearray(5056)
        struct.pack_into('<QIIQ', raw, 0, self.seed_value, left, 1, index)
        struct.pack_into('<624Q', raw, 24, *words[:624])
        struct.pack_into('<Q', raw, 5024, self.double_cache)
        struct.pack_into('<I', raw, 5040, self.double_valid)
        struct.pack_into('<I', raw, 5048, self.float_cache)
        raw[5052] = self.float_valid
        return bytes(raw)

    def word32(self):
        self.fresh = False
        return self.rng.getrandbits(32)

    def word64(self):
        return (self.word32() << 32) | self.word32()

    def operation(self, op, count, low, high):
        if op == 7:
            self.seed(low & MASK); return []
        values = []
        for _ in range(count):
            if op in (0, 2, 5, 6, 8):
                value = self.word32()
            elif op in (1, 3):
                value = self.word64()
            else:
                width = high-low
                value = (self.word64() if width >= (1 << 28) else self.word32()) % width+low
            if op in (2, 5, 6):
                unit = (value & ((1 << 24)-1))/(1 << 24)
                if op == 2:
                    value = f32(unit)
                elif op == 5:
                    value = struct.unpack('<H', struct.pack('<e', unit))[0]
                    value = 0 if value == 0x3c00 else value
                else:
                    bits = f32(unit)
                    value = (bits+0x7fff+((bits >> 16) & 1)) >> 16
                    value = 0 if value == 0x3f80 else value
            elif op == 3:
                value = f64((value & ((1 << 53)-1))/(1 << 53))
            if op != 8:
                values.append(value & MASK)
        return values


class TorchReference:
    def __init__(self, torch, initial, seed=0, state=None):
        self.torch, self.initial = torch, initial
        self.generator = torch.Generator(device='cpu')
        if state is None:
            self.generator.manual_seed(seed)
        else:
            self.generator.set_state(torch.frombuffer(bytearray(state), dtype=torch.uint8).clone())

    def state(self):
        return bytes(self.generator.get_state().tolist())

    def operation(self, op, count, low, high):
        t, g = self.torch, self.generator
        if op == 7:
            g.manual_seed(low & MASK); return []
        if op in (0, 8):
            model = Model(self.initial, state=self.state())
            raw = model.operation(0, count, 0, 0)
            projection = t.empty(count, dtype=t.int32).random_(generator=g).tolist()
            assert projection == [x & 0x7fffffff for x in raw]
            assert self.state() == model.state()
            return raw if op == 0 else []
        if op == 1:
            values = t.empty(count, dtype=t.int64).random_(-(1 << 63), None, generator=g)
        elif op in (2, 3, 5, 6):
            dtype = {2: t.float32, 3: t.float64, 5: t.float16, 6: t.bfloat16}[op]
            values = t.rand(count, dtype=dtype, generator=g)
            values = values.view({2: t.int32, 3: t.int64, 5: t.int16, 6: t.int16}[op])
        else:
            values = t.randint(low, high, (count,), dtype=t.int64, generator=g)
        mask = {2: (1 << 32)-1, 5: (1 << 16)-1, 6: (1 << 16)-1}.get(op, MASK)
        return [x & mask for x in values.tolist()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--reference', choices=('cpython', 'pytorch'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = (ROOT/'tools/rng-cpu/rng.hepta').read_bytes()
    assert sha(source) == SOURCE_SHA and sha(a.compiler.read_bytes()) == COMPILER_SHA
    profile_bytes = (ROOT/'tools/rng-cpu/profile.json').read_bytes(); profile = json.loads(profile_bytes)
    assert profile['source_sha256'] == SOURCE_SHA and profile['native_executable_sha256'] == IMAGE_SHA
    images = []
    for i in range(2):
        proc = subprocess.run([str(a.compiler.resolve())], input=source, capture_output=True, timeout=30, env={'PATH': ''})
        assert proc.returncode == 0 and not proc.stderr and sha(proc.stdout) == IMAGE_SHA
        images.append(proc.stdout)
    assert images[0] == images[1]
    native = out/'rng-native'; native.write_bytes(images[0]); native.chmod(0o500)
    compiler = shutil.which('g++'); assert compiler, 'GNU C++ standard-library reference required'
    cpp = ROOT/'tests/rng_mt_reference.cpp'; reference = out/'mt-standard-reference'
    build = subprocess.run([compiler, '-std=c++17', '-O2', str(cpp), '-o', str(reference)], capture_output=True)
    assert build.returncode == 0 and not build.stderr
    cpp_version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    seeds, seed_proofs = {}, []

    def initial(seed):
        if seed not in seeds:
            r = subprocess.run([str(reference), str(seed), '4097'], capture_output=True, timeout=30, env={'PATH': ''})
            assert r.returncode == 0 and not r.stderr and len(r.stdout) == (625+4097+625)*8
            words = struct.unpack('<'+'Q'*(625+4097+625), r.stdout)
            first, values, last = words[:625], words[625:-625], words[-625:]
            rng = random.Random(); rng.setstate((3, tuple(first), None))
            assert tuple(rng.getrandbits(32) for _ in range(4097)) == values
            assert rng.getstate()[1] == last
            seeds[seed] = first
            seed_proofs.append({'seed': seed, 'independent_raw32_words': 4097, 'cpp_output_sha256': sha(r.stdout),
                               'status': 'PASS', 'cpp_and_cpython_final_state_equal': True})
        return seeds[seed]

    torch, provenance = None, None
    if a.reference == 'pytorch':
        support = ROOT/'tests/pytorch_oracle.py'; assert sha(support.read_bytes()) == SUPPORT_SHA
        argv = sys.argv
        try:
            sys.argv = [str(support), str(native), str(out/'provenance.json'), str(ROOT/'compatibility/pytorch-oracle-lock.json')]
            context = runpy.run_path(str(support), run_name='rng_reference_only')
        finally:
            sys.argv = argv
        torch, provenance = context['torch'], context['PROVENANCE']
        assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
        assert all(x is True for x in provenance['checks'].values())

    rows, frames, value_count, state_bytes_compared = [], 0, 0, 0

    def invoke(raw):
        result = subprocess.run([str(native)], input=raw, capture_output=True, timeout=90, env={'PATH': ''})
        assert result.returncode == 0 and not result.stderr, (result.returncode, result.stderr)
        return result.stdout

    def scenario(name, seed, batches, imported=None):
        nonlocal frames, value_count, state_bytes_compared
        model = Model(initial, seed=seed, state=imported)
        actual_reference = model if torch is None else TorchReference(torch, initial, seed, imported)
        state, rows_here = imported, []
        for commands in batches:
            request = packet(commands, seed, state)
            wanted = []
            for command in commands:
                wanted += actual_reference.operation(*command)
            if torch is not None:
                model_values = []
                for command in commands:
                    model_values += model.operation(*command)
                assert model_values == wanted, (name, 'independent model vs PyTorch values')
                assert model.state() == actual_reference.state(), (name, 'independent model vs PyTorch state')
            response = invoke(request); frames += 1
            assert len(response) >= 5080 and struct.unpack('<3Q', response[:24]) == (0, len(wanted), 5056)
            values = struct.unpack('<'+'Q'*len(wanted), response[5080:])
            assert list(values) == wanted, (name, [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(values, wanted)) if x != y][:8])
            state = response[24:5080]
            expected_state = actual_reference.state()
            assert state == expected_state, (name, [(i, x, y) for i, (x, y) in enumerate(zip(state, expected_state)) if x != y][:16])
            state_bytes_compared += 5056; value_count += len(wanted)
            rows_here.append({'commands': commands, 'values': len(wanted), 'state_bytes_exact': 5056,
                              'request_sha256': sha(request), 'response_sha256': sha(response),
                              'state_sha256': sha(state), 'reference_values_sha256': sha(b''.join(struct.pack('<Q', v) for v in wanted))})
        return {'case': name, 'status': 'PASS', 'frames': rows_here}

    def run(name, call):
        try:
            rows.append(call())
        except Exception:
            rows.append({'case': name, 'status': 'FAIL', 'error': traceback.format_exc()})

    mixed = [(0, 17, 0, 0), (1, 19, 0, 0), (2, 23, 0, 0), (3, 29, 0, 0),
             (5, 113, 0, 0), (6, 127, 0, 0), (4, 31, -9999, 10001)]
    for seed in (0, 1, 17, 5489, (1 << 31)-1, 1 << 31, (1 << 32)-1,
                 1 << 32, (1 << 32)+17, (1 << 63)-1, 1 << 63, MASK):
        name = 'mixed-seed-'+str(seed)
        run(name, lambda name=name, seed=seed: scenario(name, seed, [mixed]))
    for count in (0, 1, 2, 226, 227, 396, 397, 623, 624, 625, 1247, 1248, 1249):
        name = 'twist-boundary-'+str(count)
        run(name, lambda name=name, count=count: scenario(name, 5489, [[(0, count, 0, 0)], [(1, 2, 0, 0), (2, 3, 0, 0)]]))
    for op in range(7):
        name = 'packet-capacity-op-'+str(op)
        lo, hi = (-(1 << 63), (1 << 63)-1) if op == 4 else (0, 0)
        run(name, lambda name=name, op=op, lo=lo, hi=hi: scenario(name, 20261116, [[(op, 100000, lo, hi)]]))
    for width in (1, 2, 255, (1 << 28)-1, 1 << 28, (1 << 28)+1, 1 << 31,
                  1 << 32, 1 << 53, (1 << 63)-1, 1 << 63, MASK):
        low = -(1 << 63) if width >= (1 << 63) else -113
        name = 'signed-range-width-'+str(width)
        run(name, lambda name=name, low=low, width=width: scenario(name, 12345, [[(4, 301, low, low+width)], [(0, 3, 0, 0)]]))
    run('split-restore-reseed', lambda: scenario('split-restore-reseed', MASK, [
        [(2, 623, 0, 0)], [(3, 1, 0, 0)], [(0, 624, 0, 0)], [(5, 127, 0, 0), (6, 129, 0, 0)],
        [(7, 0, (1 << 32)+17, 0), (2, 0, 0, 0)], [(1, 31, 0, 0)], [(7, 0, 17, 0)], mixed]))
    run('maximum-consumption', lambda: scenario('maximum-consumption', 6789, [[(8, 1000000, 0, 0)], [(0, 8, 0, 0)]]))
    run('maximum-command-count', lambda: scenario('maximum-command-count', 6789, [[(0, 1, 0, 0)]*128]))
    base = Model(initial, seed=17); base.operation(0, 9, 0, 0); saved = base.state()
    # Canonicalization matches the actual legacy state import: upper word bits,
    # dead legacy fields and padding do not survive get_state().
    edited = bytearray(saved)
    for index in (0, 19, 623):
        value = struct.unpack_from('<Q', edited, 24+index*8)[0]
        struct.pack_into('<Q', edited, 24+index*8, value | 0xdeadbeef00000000)
    struct.pack_into('<Q', edited, 16, struct.unpack_from('<Q', edited, 16)[0] | (123 << 32))
    struct.pack_into('<Q', edited, 5016, 0x3ff0000000000000)
    struct.pack_into('<Q', edited, 5032, 0x4000000000000000)
    struct.pack_into('<Q', edited, 5024, 0xbff4000000000000); struct.pack_into('<I', edited, 5040, 1)
    struct.pack_into('<I', edited, 5048, 0x3eaaaaab); edited[5052] = 1
    run('import-normal-caches-and-legacy-truncation', lambda: scenario(
        'import-normal-caches-and-legacy-truncation', 0, [mixed, [(7, 0, 5489, 0)]], bytes(edited)))
    run('state-fork-left', lambda: scenario('state-fork-left', 0, [[(0, 313, 0, 0)], mixed], saved))
    run('state-fork-right', lambda: scenario('state-fork-right', 0, [[(0, 313, 0, 0)], mixed], saved))
    if torch is not None:
        for dtype in (torch.float32, torch.float64):
            g = torch.Generator().manual_seed(123)
            torch.randn(1, dtype=dtype, generator=g)
            state = bytes(g.get_state().tolist()); name = 'import-actual-torch-normal-cache-'+str(dtype)
            run(name, lambda name=name, state=state: scenario(name, 0, [mixed, [(7, 0, 7, 0)]], state))
    invalid = [('truncated-header-'+str(n), bytes(n), 1) for n in (0, 1, 7, 8, 15, 24, 31)]
    invalid += [('unknown-operation', packet([(9, 0, 0, 0)]), 1),
                ('too-many-commands', packet([(0, 0, 0, 0)]*129), 1),
                ('too-many-values', packet([(0, 100001, 0, 0)]), 3),
                ('too-much-work', packet([(8, 1000001, 0, 0)]), 3),
                ('late-invalid-operation', packet([(0, 32, 0, 0), (9, 1, 0, 0)]), 1),
                ('empty-range', packet([(4, 0, 12, 12)]), 1),
                ('reversed-signed-range', packet([(4, 5, 1, -1)]), 1),
                ('reseed-count', packet([(7, 1, 0, 0)]), 1),
                ('unused-argument', packet([(2, 1, 1, 0)]), 1),
                ('trailing-data', packet([])+b'x', 1)]
    for label, offset, fmt, value in [('left-zero', 8, '<I', 0), ('left-too-large', 8, '<I', 625),
                                      ('unseeded', 12, '<I', 0), ('next-too-large', 16, '<Q', 625),
                                      ('inconsistent-position', 16, '<Q', 624),
                                      ('nonboolean-double-cache', 5040, '<I', 2),
                                      ('nonboolean-float-cache', 5052, '<B', 2)]:
        bad = bytearray(saved); struct.pack_into(fmt, bad, offset, value)
        invalid.append((label, packet([], state=bytes(bad)), 2))
    for name, request, status in invalid:
        def check(name=name, request=request, status=status):
            nonlocal frames
            response = invoke(request); frames += 1
            assert response == struct.pack('<3Q', status, 0, 0)
            return {'case': name, 'status': 'PASS', 'request_sha256': sha(request), 'response_sha256': sha(response), 'native_status': status}
        run(name, check)
    assert sha((ROOT/'tools/rng-cpu/rng.hepta').read_bytes()) == SOURCE_SHA
    assert sha(native.read_bytes()) == IMAGE_SHA and (ROOT/'tools/rng-cpu/profile.json').read_bytes() == profile_bytes
    failed = sum(row['status'] != 'PASS' for row in rows)
    report = {'schema': 'heptatorch.cpu-rng-sequence-and-state.v1', 'status': 'FAIL' if failed else 'PASS',
              'reference': a.reference, 'source_sha256': SOURCE_SHA, 'native_executable_sha256': IMAGE_SHA,
              'compiler_sha256': COMPILER_SHA, 'profile_sha256': sha(profile_bytes),
              'oracle_sha256': sha(Path(__file__).read_bytes()), 'cpp_reference_source_sha256': sha(cpp.read_bytes()),
              'cpp_reference_compiler': cpp_version, 'cpp_reference_executable_sha256': sha(reference.read_bytes()),
              'independent_cpp_cpython_seed_proofs': seed_proofs, 'oracle_provenance': provenance,
              'cases': len(rows), 'failed_cases': failed, 'native_invocations': frames,
              'exact_values': value_count, 'serialized_state_bytes_compared': state_bytes_compared,
              'native_compilations_byte_identical': True, 'source_and_image_unchanged': True,
              'rows': rows, 'raw32_reference_boundary': 'Full std::mt19937 and CPython engine comparison; PyTorch raw32 projection additionally verifies low31 outputs and full serialized state. Raw64 is directly compared to PyTorch full signed random_ range.',
              'invalid_state_boundary': 'Malformed/inconsistent MT positions and nonboolean cache flags are explicitly refused; no claim of reproducing undefined or unsafe upstream malformed-state behavior.',
              'tensor_rng_integration_complete': False, 'all_distributions_implemented': False,
              'whole_pytorch_rewrite_complete': False}
    raw = json.dumps(report, indent=2)+'\n'; (out/'report.json').write_text(raw); print(raw)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
