"""Bit-exact independent references for native HIA binary64 arithmetic.

CPython and the exact locked PyTorch wheel are references only. The target is
compiled twice by the pinned native HIA compiler and runs with an empty PATH.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import re
import runpy
import struct
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
COMPILER_SHA = '87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54'
IMAGE_SHA = '4225dd162b5c9fcf2a44987839bd410b5e37aef32665683389fb1f4f281f25a2'
SOURCE_SHA = '44a6ff00413829d3cf8f7fbd91fb96a6baf090c3c754882d4361f49ed85a2e56'
REFERENCE_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
SIGN, MASK, INF, FRACTION = 1 << 63, (1 << 63)-1, 0x7ff0000000000000, (1 << 52)-1
LIMIT = 60000
sha = lambda raw: hashlib.sha256(raw).hexdigest()
word = lambda value: struct.unpack('<Q', struct.pack('<d', value))[0]
number = lambda value: struct.unpack('<d', struct.pack('<Q', value))[0]
word32 = lambda value: struct.unpack('<I', struct.pack('<f', value))[0]
number32 = lambda value: struct.unpack('<f', struct.pack('<I', value))[0]


def corpus():
    special = [0, 1, 2, 3, FRACTION-1, FRACTION, 1 << 52, (1 << 52)+1,
               word(.5), word(1.), word(1.)+1, word(1.5), word(2.), word(3.),
               INF-2, INF-1, INF, INF+1, INF+(1 << 51), MASK]
    special += [value | SIGN for value in special]
    fractions = (0, 1, (1 << 51)-1, 1 << 51, FRACTION-1, FRACTION)
    strata = [(exponent << 52) | fraction | sign for exponent in range(2048)
              for fraction in fractions for sign in (0, SIGN)]
    rng = random.Random(413779)
    random_pairs = [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(100000)]
    close = []
    for _ in range(100000):
        exponent = rng.randrange(1, 2047)
        other = min(2046, max(0, exponent+rng.randrange(-2, 3)))
        close.append(((rng.getrandbits(1) << 63) | (exponent << 52) | rng.getrandbits(52),
                      (rng.getrandbits(1) << 63) | (other << 52) | rng.getrandbits(52)))
    adjacent = []
    for value in strata:
        exponent = (value >> 52) & 2047
        other = (max(0, exponent-1) << 52) | ((value ^ FRACTION) & FRACTION)
        adjacent.extend(((value, other), (value, other | SIGN)))
    halfway = []
    for exponent in range(2, 2047):
        midpoint = word(math.ldexp(1., exponent-1076))
        for fraction in (0, 1, FRACTION-1, FRACTION):
            for sign in (0, SIGN):
                value = (exponent << 52) | fraction | sign
                for delta in (-1, 0, 1):
                    halfway.extend(((value, midpoint+delta), (value, (midpoint+delta) | SIGN)))
    for mode, name in enumerate(('add', 'sub', 'mul', 'div')):
        yield name+'-special-cross-product', mode, [(a, b) for a in special for b in special]
        yield name+'-every-exponent-and-adjacent-binade', mode, adjacent
        yield name+'-100000-random-bit-pairs', mode, random_pairs
        yield name+'-100000-close-exponent-pairs', mode, close
        if mode < 2:
            yield name+'-halfway-and-adjacent-inputs', mode, halfway
        else:
            partners = (.5, 1.5, 2., 3., math.ldexp(1., -52), math.ldexp(1., 52))
            yield name+'-scaled-exponent-boundaries', mode, [(a, word(b)) for a in strata for b in partners]
    yield 'sqrt-special', 4, [(a, 0) for a in special]
    yield 'sqrt-every-exponent', 4, [(a, 0) for a in strata]
    yield 'sqrt-100000-random-encodings', 4, [(rng.getrandbits(64), 0) for _ in range(100000)]
    yield 'sqrt-100000-positive-encodings', 4, [(rng.getrandbits(63), 0) for _ in range(100000)]
    squares = []
    for exponent in range(-511, 512):
        for mantissa in (1., 1.25, 1.5, 1.75):
            exact = math.ldexp(mantissa*mantissa, 2*exponent)
            encoded = word(exact)
            squares.extend((encoded+delta, 0) for delta in (-1, 0, 1))
    yield 'sqrt-square-boundaries-and-neighbors', 4, squares
    strata32 = [(exponent << 23) | fraction | sign for exponent in range(256)
                for fraction in (0, 1, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff)
                for sign in (0, 1 << 31)]
    yield 'widen-every-binary32-exponent', 5, [(a, 0) for a in strata32]
    yield 'widen-100000-random-binary32-encodings', 5, [(rng.getrandbits(32), 0) for _ in range(100000)]
    yield 'widen-all-half-encodings-through-binary32', 5, [
        (word32(struct.unpack('<e', struct.pack('<H', a))[0]), 0) for a in range(65536)]
    yield 'narrow-special', 6, [(a, 0) for a in special]
    yield 'narrow-every-binary64-exponent', 6, [(a, 0) for a in strata]
    yield 'narrow-100000-random-binary64-encodings', 6, [(rng.getrandbits(64), 0) for _ in range(100000)]
    narrow_midpoints = []
    for exponent in range(255):
        for fraction in (0, 1, 2, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff):
            lo = (exponent << 23) | fraction
            midpoint = ((number32(lo)+number32(lo+1))/2 if lo < 0x7f7fffff
                        else number32(lo)+math.ldexp(1., 103))
            encoded = word(midpoint)
            for sign in (0, SIGN):
                narrow_midpoints.extend(((encoded+delta) | sign, 0) for delta in (-1, 0, 1))
    yield 'narrow-representative-every-exponent-midpoints', 6, narrow_midpoints
    for mode in range(7):
        yield 'empty-operation-'+str(mode), mode, []


def host_expected(mode, pairs):
    result = []
    for a, b in pairs:
        x, y = number(a), number(b)
        if mode == 0:
            expected = word(x+y)
        elif mode == 1:
            expected = word(x-y)
        elif mode == 2:
            expected = word(x*y)
        elif mode == 3:
            if y == 0.:
                expected = word(math.nan) if x == 0. or math.isnan(x) else ((a ^ b) & SIGN) | INF
            else:
                expected = word(x/y)
        elif mode == 4:
            expected = word(math.nan if x < 0. else math.sqrt(x))
        elif mode == 5:
            expected = word(number32(a))
        else:
            try:
                expected = word32(x)
            except OverflowError:
                expected = ((a >> 32) & (1 << 31)) | 0x7f800000
        result.append(expected)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--reference', choices=('cpython', 'pytorch'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    source_path = ROOT / 'tools/dtype-f64/f64.hepta'; source = source_path.read_bytes()
    assert sha(source) == SOURCE_SHA
    profile_path = ROOT / 'tools/dtype-f64/profile.json'; profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    assert profile['source_sha256'] == SOURCE_SHA and profile['native_executable_sha256'] == IMAGE_SHA
    assert profile['compiler_sha256'] == COMPILER_SHA
    bodies = lambda raw: {m[2].decode(): m[1] for m in re.finditer(
        rb'^(fn (\w+)\([^\n]*\n.*?^end\n)', raw, re.M | re.S)}
    actual_functions = bodies(source)
    assert {name: sha(body) for name, body in actual_functions.items()} == profile['function_body_sha256']
    canonical = (ROOT / 'src/heptatorch.hepta').read_bytes()
    assert sha(canonical) == profile['canonical_source_sha256']
    canonical_functions = bodies(canonical)
    for name, identity in profile['reused_canonical_functions'].items():
        assert actual_functions[name] == canonical_functions[name] and sha(actual_functions[name]) == identity
    compiler = a.compiler.resolve(); assert sha(compiler.read_bytes()) == COMPILER_SHA
    images = []
    for repeat in range(2):
        proc = subprocess.run([str(compiler)], input=source, capture_output=True,
                              timeout=30, env={'PATH': ''})
        (out / ('compile-'+str(repeat)+'.stderr')).write_bytes(proc.stderr)
        assert proc.returncode == 0 and not proc.stderr and proc.stdout.startswith(b'\x7fELF')
        images.append(proc.stdout)
    assert images[0] == images[1] and sha(images[0]) == IMAGE_SHA
    executable = out / 'f64-native'; executable.write_bytes(images[0]); executable.chmod(0o500)
    torch, provenance = None, None
    if a.reference == 'pytorch':
        support_path = ROOT / 'tests/pytorch_oracle.py'
        assert sha(support_path.read_bytes()) == REFERENCE_SHA
        original_argv = sys.argv
        try:
            sys.argv = [str(support_path), str(executable), str(out / 'provenance.json'),
                        str(ROOT / 'compatibility/pytorch-oracle-lock.json')]
            support = runpy.run_path(str(support_path), run_name='binary64_reference_only')
        finally:
            sys.argv = original_argv
        torch, provenance = support['torch'], support['PROVENANCE']
        assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
        assert all(value is True for value in provenance['checks'].values())

    def invoke(raw):
        proc = subprocess.run([str(executable)], input=raw, capture_output=True,
                              timeout=30, env={'PATH': ''})
        assert proc.returncode == 0 and not proc.stderr, (proc.returncode, proc.stderr)
        return proc.stdout

    def reference(mode, pairs):
        if a.reference == 'cpython':
            return host_expected(mode, pairs)
        if not pairs:
            return []
        if mode == 5:
            raw = b''.join(struct.pack('<I', x) for x, _ in pairs)
            result = torch.frombuffer(bytearray(raw), dtype=torch.float32).to(torch.float64)
        else:
            left = torch.frombuffer(bytearray(b''.join(struct.pack('<Q', x) for x, _ in pairs)), dtype=torch.float64)
            if mode < 4:
                right = torch.frombuffer(bytearray(b''.join(struct.pack('<Q', y) for _, y in pairs)), dtype=torch.float64)
                result = (torch.add, torch.sub, torch.mul, torch.div)[mode](left, right)
            elif mode == 4:
                result = torch.sqrt(left)
            else:
                result = left.to(torch.float32)
        return [x & (0xffffffff if mode == 6 else 0xffffffffffffffff)
                for x in result.view(torch.int32 if mode == 6 else torch.int64).tolist()]

    rows = []
    for name, mode, pairs in corpus():
        row = {'case': name, 'mode': mode, 'status': 'FAIL', 'input_values': len(pairs),
               'non_nan_attempted': 0, 'exact_non_nan': 0, 'nan_classification': 0,
               'mismatches': [], 'native_invocations': 0}
        requests, responses, references = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        try:
            for offset in range(0, max(1, len(pairs)), LIMIT):
                batch = pairs[offset:offset+LIMIT]
                raw = struct.pack('<II', mode, len(batch)) + b''.join(
                    struct.pack('<QQ', x, y) if mode < 4 else struct.pack('<Q', x) for x, y in batch)
                response = invoke(raw)
                assert response[:8] == bytes(8) and len(response) == 8+8*len(batch)
                actual = struct.unpack('<'+'Q'*len(batch), response[8:])
                expected = reference(mode, batch)
                assert len(expected) == len(actual) == len(batch)
                requests.update(raw); responses.update(response)
                references.update(b''.join(struct.pack('<Q', x) for x in expected))
                row['native_invocations'] += 1
                for position, (got, wanted) in enumerate(zip(actual, expected)):
                    mask, inf, quiet = ((0x7fffffff, 0x7f800000, 1 << 22) if mode == 6
                                        else (MASK, INF, 1 << 51))
                    if wanted & mask > inf:
                        valid = got & mask > inf and bool(got & quiet)
                        if valid:
                            row['nan_classification'] += 1
                    else:
                        row['non_nan_attempted'] += 1
                        valid = got == wanted
                        if valid:
                            row['exact_non_nan'] += 1
                    if not valid and len(row['mismatches']) < 16:
                        x, y = batch[position]
                        row['mismatches'].append({'index': offset+position, 'a': hex(x), 'b': hex(y),
                                                   'actual': hex(got), 'expected': hex(wanted)})
            assert not row['mismatches'], row['mismatches']
            row['status'] = 'PASS'
        except Exception:
            row['error'] = traceback.format_exc()
        row.update(request_sha256=requests.hexdigest(), response_sha256=responses.hexdigest(),
                   reference_packed_sha256=references.hexdigest())
        rows.append(row)
    bad = [('truncated-header-'+str(n), bytes(n), 1) for n in range(8)]
    bad += [
        ('unknown-operation', struct.pack('<II', 7, 0), 1),
        ('u32-max-operation', struct.pack('<II', 0xffffffff, 0), 1),
        ('count-limit-plus-one', struct.pack('<II', 0, LIMIT+1), 1),
        ('u32-max-count', struct.pack('<II', 0, 0xffffffff), 1),
        ('missing-binary-operand', struct.pack('<IIQ', 0, 1, word(1.)), 1),
        ('trailing-input', struct.pack('<II', 0, 0)+b'x', 1),
        ('widen-high-input-bits', struct.pack('<IIQ', 5, 1, 1 << 32), 2),
        ('widen-late-bad-word', struct.pack('<IIQQ', 5, 2, 0x3f800000, 1 << 32), 2),
    ]
    for name, raw, status in bad:
        row = {'case': name, 'status': 'FAIL', 'request_sha256': sha(raw)}
        try:
            response = invoke(raw); row['response_sha256'] = sha(response)
            assert response == struct.pack('<Q', status)
            row['status'] = 'PASS'
        except Exception:
            row['error'] = traceback.format_exc()
        rows.append(row)
    assert len(rows) == 55 and len({r['case'] for r in rows}) == len(rows)
    unchanged = (source_path.read_bytes() == source and sha(compiler.read_bytes()) == COMPILER_SHA
                 and executable.read_bytes() == images[0] and profile_path.read_bytes() == profile_bytes)
    result = {'schema': 'heptatorch.native-binary64-review.v1',
              'status': 'PASS' if unchanged and all(r['status'] == 'PASS' for r in rows) else 'FAIL',
              'reference': a.reference, 'source_sha256': sha(source), 'compiler_sha256': COMPILER_SHA,
              'profile_sha256': sha(profile_bytes),
              'native_executable_sha256': IMAGE_SHA, 'oracle_script_sha256': sha(Path(__file__).read_bytes()),
              'native_compilations_byte_identical': True, 'source_producer_and_image_unchanged': unchanged,
              'oracle_provenance': provenance, 'cases': len(rows),
              'failed_cases': sum(r['status'] != 'PASS' for r in rows),
              'exact_non_nan': sum(r.get('exact_non_nan', 0) for r in rows),
              'nan_classification': sum(r.get('nan_classification', 0) for r in rows),
              'rows': rows, 'exact_nan_payload_parity_claimed': False,
              'torch_or_python_production_runtime': False, 'tensor_dtype_integration_complete': False,
              'whole_pytorch_rewrite_complete': False,
              'scope': 'RNE binary64 add/subtract/multiply/divide/sqrt and binary32 conversion; '
                       'signed zeros, infinities, gradual underflow, NaN classification and packet admission. '
                       'This does not qualify all binary64 bit patterns, dtype promotion, tensor kernels, '
                       'floating-point flags or alternate rounding modes.'}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
