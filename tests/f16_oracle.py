"""Exact native HIA IEEE binary16 conversion against independent host references.

CPython struct and pinned PyTorch are test references only. Production conversion
and packet admission execute in the separately compiled HIA ELF with empty PATH.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import runpy
import struct
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
COMPILER_SHA = '87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54'
IMAGE_SHA = '97c52928ef9dc201c02818c73bb2f82351dbc5fe0305f3b3dc58b32aa92031ac'
REFERENCE_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
sha = lambda data: hashlib.sha256(data).hexdigest()


def word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def half_value(bits):
    return struct.unpack('<e', struct.pack('<H', bits))[0]


def corpus():
    # Each adjacent finite positive binary16 pair has an exactly representable
    # binary32 midpoint. Include that midpoint and both neighboring binary32
    # encodings, with both signs. The overflow threshold follows max+half-ULP.
    midpoints = []
    for lower in range(0x7c00):
        lo = half_value(lower)
        middle = (lo + half_value(lower + 1)) / 2 if lower < 0x7bff else 65520.
        bits = word(middle)
        for sign in (0, 0x80000000):
            midpoints.extend(sign | (bits + delta) for delta in (-1, 0, 1))
    assert len(midpoints) == 190464
    rng = random.Random(43019)
    special = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
               0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
               0x7f800000, 0xff800000, 0x7f800001, 0xff800001,
               0x7fc00000, 0xffc00000, 0x7fffffff, 0xffffffff]
    return [
        ('widen-all-65536-encodings', 1, list(range(65536))),
        ('round-all-exact-half-values', 0, [word(half_value(i)) for i in range(65536)]),
        ('round-every-finite-half-boundary-and-neighbors', 0, midpoints),
        ('round-100000-seeded-f32-encodings', 0, [rng.getrandbits(32) for _ in range(100000)]),
        ('round-special-f32-encodings', 0, special),
        ('round-empty', 0, []), ('widen-empty', 1, []),
        ('round-exact-packet-count-limit', 0, [0x3f801000, 0x80000000, 0x33800000, 0x477ff000] * 50000),
        ('widen-exact-packet-count-limit', 1, [0x0001, 0x03ff, 0x0400, 0x7bff] * 50000),
    ]


def cpython_expected(mode, values):
    if mode == 1:
        return [word(half_value(value)) for value in values]
    result = []
    for value in values:
        number = struct.unpack('<f', struct.pack('<I', value))[0]
        try:
            rounded = struct.pack('<e', number)
        except OverflowError:
            rounded = struct.pack('<e', math.copysign(math.inf, number))
        result.append(int.from_bytes(rounded, 'little'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--reference', choices=('cpython', 'pytorch'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    source_path = ROOT / 'tools/dtype-f16/f16.hepta'
    source = source_path.read_bytes()
    compiler = a.compiler.resolve()
    assert sha(compiler.read_bytes()) == COMPILER_SHA
    images = []
    for generation in range(2):
        result = subprocess.run([str(compiler)], input=source, capture_output=True,
                                timeout=30, env={'PATH': ''})
        (a.output / f'compile-{generation}.stderr').write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        images.append(result.stdout)
    assert images[0] == images[1] and sha(images[0]) == IMAGE_SHA
    executable = a.output / 'f16-native'
    executable.write_bytes(images[0]); executable.chmod(0o700)
    provenance = None
    torch = None
    if a.reference == 'pytorch':
        reference = ROOT / 'tests/pytorch_oracle.py'
        assert sha(reference.read_bytes()) == REFERENCE_SHA
        original_argv = sys.argv
        try:
            sys.argv = [str(reference), str(executable), str(a.output / 'provenance.json'),
                        str(ROOT / 'compatibility/pytorch-oracle-lock.json')]
            support = runpy.run_path(str(reference), run_name='f16_pinned_reference_support')
        finally:
            sys.argv = original_argv
        torch, provenance = support['torch'], support['PROVENANCE']
        assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
        assert all(value is True for value in provenance['checks'].values())

    def invoke(packet):
        result = subprocess.run([str(executable)], input=packet, capture_output=True,
                                timeout=30, env={'PATH': ''})
        assert result.returncode == 0 and not result.stderr, (result.returncode, result.stderr)
        return result.stdout

    def reference(mode, values, raw):
        if a.reference == 'cpython':
            return cpython_expected(mode, values)
        if not values:
            return []
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32 if mode == 0 else torch.float16)
        converted = tensor.to(dtype=torch.float16 if mode == 0 else torch.float32)
        signed = converted.view(torch.int16 if mode == 0 else torch.int32).tolist()
        mask = 0xffff if mode == 0 else 0xffffffff
        return [value & mask for value in signed]

    rows = []
    for name, mode, values in corpus():
        row = {'case': name, 'status': 'FAIL', 'input_count': len(values),
               'exact_non_nan': 0, 'nan_classification': 0, 'mismatches': []}
        try:
            width, output_width = (4, 2) if mode == 0 else (2, 4)
            raw = b''.join(value.to_bytes(width, 'little') for value in values)
            packet = struct.pack('<II', mode, len(values)) + raw
            response = invoke(packet)
            assert response[:4] == bytes(4) and len(response) == 4 + output_width * len(values)
            want = reference(mode, values, raw)
            got = [int.from_bytes(response[4+i*output_width:4+(i+1)*output_width], 'little')
                   for i in range(len(values))]
            src_exp, src_fraction = (0x7f800000, 0x7fffff) if mode == 0 else (0x7c00, 0x3ff)
            dst_exp, dst_fraction, dst_quiet = (0x7c00, 0x3ff, 0x200) if mode == 0 else (0x7f800000, 0x7fffff, 0x400000)
            for index, (src, actual, expected) in enumerate(zip(values, got, want)):
                if src & src_exp == src_exp and src & src_fraction:
                    assert actual & dst_exp == dst_exp and actual & dst_fraction and actual & dst_quiet
                    assert expected & dst_exp == dst_exp and expected & dst_fraction
                    row['nan_classification'] += 1
                else:
                    if actual != expected and len(row['mismatches']) < 8:
                        row['mismatches'].append({'index': index, 'input': src, 'actual': actual, 'expected': expected})
                    row['exact_non_nan'] += 1
            assert not row['mismatches'], row['mismatches']
            row.update(status='PASS', request_sha256=sha(packet), response_sha256=sha(response),
                       reference_packed_sha256=sha(b''.join(value.to_bytes(output_width, 'little') for value in want)))
        except Exception as error:
            row.update(error=str(error), traceback=traceback.format_exc())
        rows.append(row)
    refusals = [
        ('refuse-short-header', bytes(7)),
        ('refuse-unknown-mode', struct.pack('<II', 2, 0)),
        ('refuse-truncated-f32', struct.pack('<II', 0, 1) + bytes(3)),
        ('refuse-truncated-half', struct.pack('<II', 1, 1) + bytes(1)),
        ('refuse-extra-payload', struct.pack('<II', 1, 0) + b'x'),
        ('refuse-count-limit-plus-one', struct.pack('<II', 0, 200001)),
        ('refuse-u32-max-count', struct.pack('<II', 1, 0xffffffff)),
    ]
    for name, packet in refusals:
        row = {'case': name, 'status': 'FAIL'}
        try:
            response = invoke(packet)
            assert response == b'\x01\0\0\0', response
            row.update(status='PASS', request_sha256=sha(packet), response_sha256=sha(response))
        except Exception as error:
            row.update(error=str(error), traceback=traceback.format_exc())
        rows.append(row)
    unchanged = source_path.read_bytes() == source and executable.read_bytes() == images[0] and sha(compiler.read_bytes()) == COMPILER_SHA
    success = len(rows) == 16 and len({row['case'] for row in rows}) == 16 and all(row['status'] == 'PASS' for row in rows) and unchanged
    report = {'schema': 'heptatorch.native-f16-conversion-review.v1', 'status': 'PASS' if success else 'FAIL',
              'reference': a.reference, 'oracle_provenance': provenance,
              'source_sha256': sha(source), 'compiler_sha256': COMPILER_SHA,
              'native_executable_sha256': IMAGE_SHA, 'source_producer_and_image_unchanged': unchanged,
              'oracle_script_sha256': sha(Path(__file__).read_bytes()),
              'native_compilations_byte_identical': True, 'cases': len(rows),
              'failed_cases': sum(row['status'] != 'PASS' for row in rows),
              'exact_non_nan': sum(row.get('exact_non_nan', 0) for row in rows),
              'nan_classification': sum(row.get('nan_classification', 0) for row in rows),
              'rows': rows, 'exact_nan_payload_parity_claimed': False,
              'torch_or_python_production_runtime': False,
              'scope': 'Native IEEE binary16 conversion, finite RNE boundaries and packet refusal; complete dtype/Tensor integration and full upstream gates remain open.'}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
