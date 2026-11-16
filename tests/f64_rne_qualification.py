"""Qualify a PARTIAL HIA binary64 component without relaxing the strict oracle.

Both original 55-case reports must exist. The frozen oracle is never changed.
This independently repeats its complete corpus and admission probes, binds
all request/response/reference hashes, requires every non-sqrt strict case to
PASS, and proves native sqrt RNE plus reference sqrt error <1 ulp with integer
arithmetic. Every sqrt bitwise mismatch is retained, not converted to equality.
"""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import struct
import subprocess
import sys

from f64_sqrt_integer import INF, MASK, SIGN, in_rne_cell, is_nan, self_check, sqrt_rne_bits, within_one_ulp

SHA = lambda b: hashlib.sha256(b).hexdigest()
ORACLE_SHA = 'bb639f7ef669d34787986ed62e870cc61071adcba1ae4e44e325673e2f75bb03'
REFERENCE_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
SOURCE_SHA = '44a6ff00413829d3cf8f7fbd91fb96a6baf090c3c754882d4361f49ed85a2e56'
IMAGE_SHA = '4225dd162b5c9fcf2a44987839bd410b5e37aef32665683389fb1f4f281f25a2'
COMPILER_SHA = '87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54'
PROFILE_SHA = '8865de92df8a2e6f55ef1b3d7a28d56e05dd228b21a1c53b3f485151f4027dd3'


def read_original(path, reference):
    raw = path.read_bytes()
    r = json.loads(raw)
    assert r['schema'] == 'heptatorch.native-binary64-review.v1'
    assert r['reference'] == reference and r['cases'] == 55
    assert len(r['rows']) == len({x['case'] for x in r['rows']}) == 55
    for key, value in [('source_sha256', SOURCE_SHA), ('native_executable_sha256', IMAGE_SHA),
                       ('compiler_sha256', COMPILER_SHA), ('profile_sha256', PROFILE_SHA),
                       ('oracle_script_sha256', ORACLE_SHA)]:
        assert r[key] == value, key
    assert r['native_compilations_byte_identical'] is True
    assert r['source_producer_and_image_unchanged'] is True
    bad = sum(x['status'] != 'PASS' for x in r['rows'])
    assert r['failed_cases'] == bad
    assert r['status'] == ('FAIL' if bad else 'PASS')
    if reference == 'cpython':
        assert r['status'] == 'PASS' and bad == 0
    else:
        provenance = r['oracle_provenance']
        assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
        assert all(x is True for x in provenance['checks'].values())
    return r, raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--reference', choices=('cpython', 'pytorch'), required=True)
    p.add_argument('--strict-report', type=Path, required=True)
    p.add_argument('--cpython-report', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    checker_controls = self_check()
    repo, native = a.repo.resolve(), a.native.resolve()
    paths = {'source': repo/'tools/dtype-f64/f64.hepta', 'profile': repo/'tools/dtype-f64/profile.json',
             'oracle': repo/'tests/f64_oracle.py', 'native': native,
             'qualification': Path(__file__).resolve(),
             'integer_checker': Path(__file__).resolve().with_name('f64_sqrt_integer.py'),
             'strict_report': a.strict_report.resolve()}
    custody = {key: value.read_bytes() for key, value in paths.items()}
    for key, value in [('source', SOURCE_SHA), ('profile', PROFILE_SHA), ('oracle', ORACLE_SHA), ('native', IMAGE_SHA)]:
        assert SHA(custody[key]) == value, key
    strict, strict_raw = read_original(a.strict_report, a.reference)
    cp_path = a.cpython_report if a.reference == 'pytorch' else a.strict_report
    assert cp_path is not None, '--cpython-report is required for PyTorch qualification'
    cp, cp_raw = read_original(cp_path, 'cpython')
    paths['cpython_report'] = cp_path.resolve(); custody['cpython_report'] = cp_raw
    old = {r['case']: r for r in strict['rows']}
    cp_rows = {r['case']: r for r in cp['rows']}
    scope = runpy.run_path(str(paths['oracle']), run_name='integer_qualification_corpus_only')
    torch, provenance = None, None
    if a.reference == 'pytorch':
        support = repo/'tests/pytorch_oracle.py'
        paths['reference_support'] = support; custody['reference_support'] = support.read_bytes()
        assert SHA(custody['reference_support']) == REFERENCE_SHA
        original_argv = sys.argv
        try:
            sys.argv = [str(support), str(native), str(a.output/'provenance.json'),
                        str(repo/'compatibility/pytorch-oracle-lock.json')]
            context = runpy.run_path(str(support), run_name='integer_qualification_reference_only')
        finally:
            sys.argv = original_argv
        torch, provenance = context['torch'], context['PROVENANCE']
        assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
        assert all(x is True for x in provenance['checks'].values())

    def invoke(request):
        result = subprocess.run([str(native)], input=request, capture_output=True,
                                timeout=30, env={'PATH': ''})
        assert result.returncode == 0 and not result.stderr
        return result.stdout

    def reference(mode, batch):
        if torch is None:
            return scope['host_expected'](mode, batch)
        if not batch:
            return []
        if mode == 5:
            data = b''.join(struct.pack('<I', x) for x, _ in batch)
            answer = torch.frombuffer(bytearray(data), dtype=torch.float32).to(torch.float64)
        else:
            data = b''.join(struct.pack('<Q', x) for x, _ in batch)
            left = torch.frombuffer(bytearray(data), dtype=torch.float64)
            if mode < 4:
                data = b''.join(struct.pack('<Q', y) for _, y in batch)
                right = torch.frombuffer(bytearray(data), dtype=torch.float64)
                answer = (torch.add, torch.sub, torch.mul, torch.div)[mode](left, right)
            else:
                answer = torch.sqrt(left) if mode == 4 else left.to(torch.float32)
        dtype, mask = (torch.int32, (1 << 32)-1) if mode == 6 else (torch.int64, (1 << 64)-1)
        return [x & mask for x in answer.view(dtype).tolist()]

    rows, mismatches = [], []
    for name, mode, pairs in scope['corpus']():
        row = {'case': name, 'mode': mode, 'input_values': len(pairs), 'exact_non_nan': 0,
               'non_nan_attempted': 0, 'nan_classification': 0, 'native_invocations': 0,
               'native_sqrt_integer_rne_checks': 0, 'native_sqrt_squared_cell_checks': 0,
               'reference_sqrt_strict_ha_checks': 0, 'reference_sqrt_rne_errors': 0,
               'strict_bitwise_mismatches': 0}
        requests, responses, references = (hashlib.sha256() for _ in range(3))
        strict_first = []
        for offset in range(0, max(1, len(pairs)), scope['LIMIT']):
            batch = pairs[offset:offset+scope['LIMIT']]
            request = struct.pack('<II', mode, len(batch)) + b''.join(
                struct.pack('<QQ', x, y) if mode < 4 else struct.pack('<Q', x) for x, y in batch)
            response = invoke(request)
            assert response[:8] == bytes(8) and len(response) == 8+8*len(batch)
            actual = struct.unpack('<'+'Q'*len(batch), response[8:])
            expected = reference(mode, batch)
            assert len(actual) == len(expected) == len(batch)
            requests.update(request); responses.update(response)
            references.update(b''.join(struct.pack('<Q', x) for x in expected))
            row['native_invocations'] += 1
            for index, ((x, y), got, wanted) in enumerate(zip(batch, actual, expected)):
                mask, inf, quiet = ((0x7fffffff, 0x7f800000, 1 << 22) if mode == 6
                                    else (MASK, INF, 1 << 51))
                if wanted & mask > inf:
                    assert got & mask > inf and got & quiet
                    row['nan_classification'] += 1
                    equal = True
                else:
                    row['non_nan_attempted'] += 1
                    equal = got == wanted
                    row['exact_non_nan'] += equal
                if mode == 4:
                    correct = sqrt_rne_bits(x)
                    assert is_nan(got) and is_nan(correct) or got == correct, ('native RNE', name, offset+index)
                    assert in_rne_cell(x, got), ('native midpoint cell', name, offset+index)
                    assert within_one_ulp(x, wanted), ('reference violates strict HA', name, offset+index)
                    row['native_sqrt_integer_rne_checks'] += 1
                    row['native_sqrt_squared_cell_checks'] += 1
                    row['reference_sqrt_strict_ha_checks'] += 1
                    reference_rne = in_rne_cell(x, wanted)
                    row['reference_sqrt_rne_errors'] += not reference_rne
                else:
                    assert equal, ('non-sqrt strict mismatch', name, offset+index, got, wanted)
                if not equal:
                    assert mode == 4 and not is_nan(got) and not is_nan(wanted)
                    item = {'case': name, 'index': offset+index, 'input': hex(x), 'native': hex(got),
                            'reference': hex(wanted), 'native_rne': True,
                            'reference_rne': reference_rne, 'reference_error_strictly_below_one_ulp': True,
                            'encoding_distance': abs(got-wanted)}
                    mismatches.append(item)
                    row['strict_bitwise_mismatches'] += 1
                    if len(strict_first) < 16:
                        strict_first.append({'index': offset+index, 'a': hex(x), 'b': hex(y),
                                             'actual': hex(got), 'expected': hex(wanted)})
        row.update(request_sha256=requests.hexdigest(), response_sha256=responses.hexdigest(),
                   reference_packed_sha256=references.hexdigest())
        for key in ['input_values', 'exact_non_nan', 'non_nan_attempted', 'nan_classification',
                    'native_invocations', 'request_sha256', 'response_sha256', 'reference_packed_sha256']:
            assert old[name][key] == row[key], ('strict report is not this execution', name, key)
        assert old[name]['mismatches'] == strict_first
        assert old[name]['status'] == ('FAIL' if strict_first else 'PASS')
        if mode != 4 or not pairs:
            assert old[name]['status'] == 'PASS'
        for key in ['input_values', 'native_invocations', 'request_sha256', 'response_sha256']:
            assert cp_rows[name][key] == row[key], ('CPython report is not this native execution', name, key)
        assert cp_rows[name]['status'] == 'PASS'
        row['qualification_status'] = 'PASS'
        row['original_strict_status'] = old[name]['status']
        rows.append(row)

    bad = [('truncated-header-'+str(n), bytes(n), 1) for n in range(8)]
    bad += [('unknown-operation', struct.pack('<II', 7, 0), 1),
            ('u32-max-operation', struct.pack('<II', 0xffffffff, 0), 1),
            ('count-limit-plus-one', struct.pack('<II', 0, scope['LIMIT']+1), 1),
            ('u32-max-count', struct.pack('<II', 0, 0xffffffff), 1),
            ('missing-binary-operand', struct.pack('<IIQ', 0, 1, 0x3ff0000000000000), 1),
            ('trailing-input', struct.pack('<II', 0, 0)+b'x', 1),
            ('widen-high-input-bits', struct.pack('<IIQ', 5, 1, 1 << 32), 2),
            ('widen-late-bad-word', struct.pack('<IIQQ', 5, 2, 0x3f800000, 1 << 32), 2)]
    for name, request, status in bad:
        response = invoke(request)
        assert response == struct.pack('<Q', status)
        for report in (old, cp_rows):
            assert report[name]['status'] == 'PASS'
            assert report[name]['request_sha256'] == SHA(request)
            assert report[name]['response_sha256'] == SHA(response)
        rows.append({'case': name, 'qualification_status': 'PASS', 'original_strict_status': 'PASS',
                     'request_sha256': SHA(request), 'response_sha256': SHA(response)})
    assert len(rows) == 55 and {r['case'] for r in rows} == set(old) == set(cp_rows)
    assert all(path.read_bytes() == custody[key] for key, path in paths.items())
    mismatch_raw = (json.dumps(mismatches, indent=2)+'\n').encode()
    (a.output/'mismatches.json').write_bytes(mismatch_raw)
    report = {'schema': 'heptatorch.binary64-rne-and-reference-accuracy.v1',
              'status': 'PARTIAL', 'qualification_status': 'PASS', 'reference': a.reference,
              'qualification_cases': len(rows), 'cpython_strict_cases': cp['cases'],
              'integer_checker_boundary_controls': checker_controls,
              'cpython_strict_status': cp['status'], 'original_strict_status': strict['status'],
              'original_strict_failed_cases': strict['failed_cases'],
              'original_strict_report_sha256': SHA(strict_raw), 'cpython_strict_report_sha256': SHA(cp_raw),
              'custody_sha256': {key: SHA(raw) for key, raw in custody.items()},
              'custody_before_after_unchanged': True, 'oracle_provenance': provenance,
              'rows': rows, 'sqrt_bitwise_mismatch_count': len(mismatches),
              'sqrt_bitwise_mismatches_sha256': SHA(mismatch_raw), 'sqrt_bitwise_mismatches': mismatches,
              'sqrt_native_rne_failures': 0, 'sqrt_reference_ha_failures': 0,
              'reference_bitwise_observed': not mismatches,
              'strict_pytorch_bitwise_gate_closed': False, 'full_dtype_gate_closed': False,
              'whole_pytorch_rewrite_complete': False,
              'accuracy_definition': 'strict |y-sqrt(x)| < 2**(floor(log2(x)/2)-52), proved with exact integer squared bounds; special values separately exact/classified',
              'limitations': 'Qualified scalar component and this corpus only; no claim of MKL algorithm replication, all binary64 patterns exhausted, floating-point flags, NaN payload parity, alternate rounding modes, dtype promotion or tensor integration.'}
    raw = json.dumps(report, indent=2, allow_nan=False)+'\n'
    (a.output/'report.json').write_text(raw)
    print(raw)


if __name__ == '__main__':
    main()
