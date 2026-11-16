"""Observe binary64 sqrt against exact rational rounding cells and PyTorch.

This diagnostic never converts the frozen strict differential failure to PASS.
The mathematical check uses integer/Fraction comparisons, not math.sqrt.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import runpy
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SHA = lambda data: hashlib.sha256(data).hexdigest()
ORACLE_SHA = 'bb639f7ef669d34787986ed62e870cc61071adcba1ae4e44e325673e2f75bb03'
IMAGE_SHA = '4225dd162b5c9fcf2a44987839bd410b5e37aef32665683389fb1f4f281f25a2'
INF, SIGN = 0x7ff0000000000000, 1 << 63


def rational(bits):
    exponent, significand = (bits >> 52) & 2047, bits & ((1 << 52)-1)
    assert bits < INF
    if exponent:
        significand |= 1 << 52
    scale = (exponent or 1)-1075
    return (Fraction(significand << scale) if scale >= 0
            else Fraction(significand, 1 << -scale))


def correctly_rounded_sqrt(source, answer):
    if source == 0 or source == SIGN or source == INF:
        return answer == source
    if source & SIGN or source > INF:
        return (answer & (SIGN-1)) > INF and bool(answer & (1 << 51))
    if not 0 < answer < INF:
        return False
    x, y = rational(source), rational(answer)
    lower = (rational(answer-1)+y)/2
    upper = (y+rational(answer+1))/2
    even = not answer & 1
    return ((x > lower*lower or even and x == lower*lower)
            and (x < upper*upper or even and x == upper*upper))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--reference', choices=('cpython', 'pytorch'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    path = ROOT/'tests/f64_oracle.py'; assert SHA(path.read_bytes()) == ORACLE_SHA
    scope = runpy.run_path(str(path), run_name='sqrt_diagnostic_corpus_only')
    assert SHA(a.native.read_bytes()) == IMAGE_SHA
    torch, provenance = None, None
    if a.reference == 'pytorch':
        support = ROOT/'tests/pytorch_oracle.py'
        assert SHA(support.read_bytes()) == scope['REFERENCE_SHA']
        before = sys.argv
        try:
            sys.argv = [str(support), str(a.native.resolve()), str(a.output/'provenance.json'),
                        str(ROOT/'compatibility/pytorch-oracle-lock.json')]
            context = runpy.run_path(str(support), run_name='sqrt_reference_only')
        finally:
            sys.argv = before
        torch, provenance = context['torch'], context['PROVENANCE']
        assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
        assert all(v is True for v in provenance['checks'].values())

    def reference(inputs):
        if torch is None:
            return scope['host_expected'](4, [(x, 0) for x in inputs])
        raw = b''.join(struct.pack('<Q', x) for x in inputs)
        t = torch.frombuffer(bytearray(raw), dtype=torch.float64)
        return [v & ((1 << 64)-1) for v in torch.sqrt(t).view(torch.int64).tolist()]

    rows, mismatches, native_errors, reference_errors = [], [], [], []
    for name, mode, pairs in scope['corpus']():
        if mode != 4 or not pairs:
            continue
        row = {'case': name, 'values': len(pairs), 'exact_finite': 0,
               'nan_classification': 0, 'ulp_histogram': {}, 'native_rne_errors': 0,
               'reference_rne_errors': 0}
        for offset in range(0, len(pairs), 60000):
            inputs = [x for x, _ in pairs[offset:offset+60000]]
            req = struct.pack('<II', 4, len(inputs))+b''.join(struct.pack('<Q', x) for x in inputs)
            proc = subprocess.run([str(a.native.resolve())], input=req, capture_output=True,
                                  timeout=30, env={'PATH': ''})
            assert proc.returncode == 0 and not proc.stderr and proc.stdout[:8] == bytes(8)
            actual = struct.unpack('<'+'Q'*len(inputs), proc.stdout[8:])
            expected = reference(inputs)
            for index, (source, got, wanted) in enumerate(zip(inputs, actual, expected)):
                native_rne = correctly_rounded_sqrt(source, got)
                reference_rne = correctly_rounded_sqrt(source, wanted)
                evidence = {'case': name, 'index': offset+index, 'input': hex(source),
                            'native': hex(got), 'reference': hex(wanted),
                            'native_in_exact_rne_cell': native_rne,
                            'reference_in_exact_rne_cell': reference_rne}
                if not native_rne:
                    native_errors.append(evidence); row['native_rne_errors'] += 1
                if not reference_rne:
                    reference_errors.append(evidence); row['reference_rne_errors'] += 1
                if wanted & (SIGN-1) > INF:
                    assert got & (SIGN-1) > INF
                    row['nan_classification'] += 1
                else:
                    ulp = abs(got-wanted)
                    row['ulp_histogram'][str(ulp)] = row['ulp_histogram'].get(str(ulp), 0)+1
                    if ulp:
                        mismatches.append(evidence | {'ulp': ulp})
                    else:
                        row['exact_finite'] += 1
        rows.append(row)
    shape_observations = []
    for source in dict.fromkeys(int(row['input'], 16) for row in mismatches[:16]):
        for count in (1, 2, 3, 4, 8, 16, 128, 2048):
            values = reference([source]*count)
            shape_observations.append({'input': hex(source), 'elements': count,
                                       'distinct_reference_outputs': sorted(set(map(hex, values)))})
    mismatch_raw = (json.dumps(mismatches, indent=2)+'\n').encode()
    (a.output/'mismatches.json').write_bytes(mismatch_raw)
    report = {'schema': 'heptatorch.binary64-sqrt-exact-rounding-diagnostic.v1',
              'status': 'FAIL' if native_errors else ('OBSERVED_DIFFERENCE' if mismatches else 'PASS'),
              'reference': a.reference, 'oracle_provenance': provenance,
              'native_executable_sha256': IMAGE_SHA, 'frozen_oracle_sha256': ORACLE_SHA,
              'diagnostic_sha256': SHA(Path(__file__).read_bytes()),
              'rounding_check': 'exact rational squared half-way boundaries; ties-to-even; no floating-point sqrt in proof',
              'rows': rows, 'mismatch_count': len(mismatches),
              'mismatches_sha256': SHA(mismatch_raw), 'first_mismatches': mismatches[:32],
              'native_rne_error_count': len(native_errors), 'first_native_errors': native_errors[:16],
              'reference_rne_error_count': len(reference_errors),
              'shape_observations': shape_observations,
              'torch_build_configuration': torch.__config__.show() if torch is not None else None,
              'torch_cpu_capability': torch.backends.cpu.get_cpu_capability() if torch is not None else None,
              'torch_mkl_available': torch.backends.mkl.is_available() if torch is not None else None,
              'strict_pytorch_bitwise_gate_closed': False,
              'whole_pytorch_rewrite_complete': False}
    raw = json.dumps(report, indent=2)+'\n'; (a.output/'report.json').write_text(raw); print(raw)
    assert not native_errors, native_errors[:16]


if __name__ == '__main__':
    main()
