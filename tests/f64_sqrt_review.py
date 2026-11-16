"""Independent source-review evidence, separate from the frozen 55-case corpus."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import math
import random
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
from f64_sqrt_integer import INF, SIGN, QNAN, decode, in_rne_cell, self_check, sqrt_rne_bits, within_one_ulp

SHA = lambda raw: hashlib.sha256(raw).hexdigest()


def rational(bits):
    m, e = decode(bits)
    return Fraction(m << e) if e >= 0 else Fraction(m, 1 << -e)


def fraction_rne(x, y):
    source, answer = rational(x), rational(y)
    lower = (rational(y-1)+answer)/2
    upper = (answer+rational(y+1))/2
    return ((source > lower*lower or source == lower*lower and not y & 1)
            and (source < upper*upper or source == upper*upper and not y & 1))


def fraction_ha(x, y):
    source, answer = rational(x), rational(y)
    p, q = source.numerator, source.denominator
    exponent = p.bit_length()-q.bit_length()
    if exponent >= 0:
        exponent -= p < q << exponent
    else:
        exponent -= p << -exponent < q
    shift = exponent//2-52
    unit = Fraction(1 << shift) if shift >= 0 else Fraction(1, 1 << -shift)
    return (answer-unit < 0 or (answer-unit)**2 < source) and source < (answer+unit)**2


def restoring_trace(bits):
    m, e = decode(bits)
    while m < 1 << 52:
        m <<= 1; e -= 1
    # decode's scale is an integer-significand exponent; unbiased is scale+52.
    unbiased = e+52
    if unbiased & 1:
        m <<= 1; unbiased -= 1
    assert (1 << 52) <= m < (1 << 54) and -1074 <= unbiased <= 1022
    n = m << 58
    high, low = m >> 6, (m << 58) & ((1 << 64)-1)
    assert (high << 64) | low == n
    root = remainder = prefix = maximum = 0
    for i in reversed(range(56)):
        position = i*2
        digit = ((high >> (position-64)) if position >= 64 else (low >> position)) & 3
        assert digit == n >> position & 3
        prefix = (prefix << 2) | digit
        pending, trial = (remainder << 2) | digit, (root << 2) | 1
        root <<= 1
        maximum = max(maximum, pending, trial, root)
        remainder = pending
        if pending >= trial:
            remainder -= trial; root |= 1
        assert prefix == root*root+remainder and 0 <= remainder < 2*root+1
        assert maximum < 1 << 64
    assert root == math.isqrt(n) and remainder == n-root*root
    sticky = root | bool(remainder)
    significand, tail = sticky >> 3, sticky & 7
    significand += tail > 4 or tail == 4 and significand & 1
    out_exp = unbiased//2
    if significand == 1 << 53:
        significand >>= 1; out_exp += 1
    answer = ((out_exp+1023) << 52) | (significand & ((1 << 52)-1))
    assert answer == sqrt_rne_bits(bits)
    return maximum.bit_length()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    native = a.native.resolve(); image = native.read_bytes()
    assert SHA(image) == '4225dd162b5c9fcf2a44987839bd410b5e37aef32665683389fb1f4f281f25a2'
    source = (a.repo/'tools/dtype-f64/f64.hepta').read_bytes()
    assert SHA(source) == '44a6ff00413829d3cf8f7fbd91fb96a6baf090c3c754882d4361f49ed85a2e56'
    profile = json.loads((a.repo/'tools/dtype-f64/profile.json').read_text())
    bodies = {m[1].decode():m[0] for m in re.finditer(rb'^fn (\w+)\([^\n]*\n.*?^end\n', source, re.M|re.S)}
    for name in ['sqrt64','pack64']:
        assert SHA(bodies[name]) == profile['function_body_sha256'][name]
    diagnostic_raw = a.diagnostic.read_bytes(); diagnostic = json.loads(diagnostic_raw)
    assert diagnostic['mismatch_count'] == 4008
    assert diagnostic['native_rne_error_count'] == 0 and diagnostic['reference_rne_error_count'] == 4008
    controls = self_check()
    rng = random.Random(918232)
    groups = {}
    fractions = [0,1,2,3,(1 << 51)-1,1 << 51,(1 << 52)-3,(1 << 52)-2,(1 << 52)-1]
    groups['every_normal_exponent_adjacencies'] = [(e << 52)|m for e in range(1,2047) for m in fractions]
    groups['subnormal_normalization_shifts'] = sorted({(1 << k)+d for k in range(53) for d in [-2,-1,0,1,2] if 0 < (1 << k)+d < INF})
    groups['independent_positive_random_bits'] = [rng.randrange(1,INF) for _ in range(20000)]
    near = []
    for _ in range(8192):
        m = (1 << 52)|rng.getrandbits(52)
        n = (2*m+1)**2  # exact square of a midpoint, scale 2**-106.
        shift = n.bit_length()-53
        q, r = divmod(n,1 << shift)
        q += 2*r > 1 << shift or 2*r == 1 << shift and q & 1
        e = n.bit_length()-1-106
        if q == 1 << 53:
            q >>= 1; e += 1
        center = ((e+1023) << 52)|(q-(1 << 52))
        near.extend([center-1,center,center+1])
    groups['independent_squared_midpoint_neighbors'] = near
    records, requests_hash, response_hash = [], hashlib.sha256(), hashlib.sha256()
    fraction_count = neighbor_rejections = 0
    for name, inputs in groups.items():
        for offset in range(0,len(inputs),60000):
            batch = inputs[offset:offset+60000]
            request = struct.pack('<II',4,len(batch))+struct.pack('<'+'Q'*len(batch),*batch)
            result = subprocess.run([str(native)],input=request,capture_output=True,timeout=30,env={'PATH':''})
            assert result.returncode == 0 and not result.stderr
            assert result.stdout[:8] == bytes(8) and len(result.stdout)==8+8*len(batch)
            actual=struct.unpack('<'+'Q'*len(batch),result.stdout[8:])
            requests_hash.update(request);response_hash.update(result.stdout)
            for x,y in zip(batch,actual):
                assert y == sqrt_rne_bits(x) and in_rne_cell(x,y) and within_one_ulp(x,y)
                assert not in_rne_cell(x,y-1) and not in_rne_cell(x,y+1)
                neighbor_rejections += 2
                assert not within_one_ulp(x,y-2) and not within_one_ulp(x,y+2)
                if fraction_count < 5000 or name=='independent_squared_midpoint_neighbors' and fraction_count < 10000:
                    for z in [y-2,y-1,y,y+1,y+2]:
                        assert in_rne_cell(x,z) == fraction_rne(x,z)
                        assert within_one_ulp(x,z) == fraction_ha(x,z)
                    fraction_count += 1
        records.append({'case':name,'native_values':len(inputs),'status':'PASS'})
    trace_inputs = groups['every_normal_exponent_adjacencies'][::9]+groups['independent_positive_random_bits'][:1024]+near[:1024]
    trace_max = max(map(restoring_trace,trace_inputs))
    witnesses = []
    for r in diagnostic['first_mismatches']:
        x, native_word, backend_word = (int(r[k],16) for k in ['input','native','reference'])
        assert sqrt_rne_bits(x)==native_word and in_rne_cell(x,native_word)
        assert not in_rne_cell(x,backend_word) and within_one_ulp(x,backend_word)
        assert fraction_ha(x,backend_word) and not fraction_rne(x,backend_word)
        witnesses.append({'input':hex(x),'native':hex(native_word),'reference':hex(backend_word),
                          'native_integer_rne':True,'reference_not_rne':True,'reference_strict_ha':True})
    # This witness is proved symbolically in REVIEW.md: x=1-2**-53,
    # midpoint=1-2**-54, midpoint**2-x=2**-108 > 0.
    x=0x3fefffffffffffff;midpoint=(rational(0x3fefffffffffffff)+1)/2
    assert midpoint**2-rational(x)==Fraction(1,1 << 108)
    assert native.read_bytes()==image
    report={'schema':'heptatorch.binary64-sqrt-independent-integer-review.v1','status':'PASS',
            'native_executable_sha256':SHA(image),'source_sha256':SHA(source),
            'sqrt_function_sha256':SHA(bodies['sqrt64']),'pack_function_sha256':SHA(bodies['pack64']),
            'diagnostic_sha256':SHA(diagnostic_raw),'script_sha256':SHA(Path(__file__).read_bytes()),
            'integer_checker_sha256':SHA((ROOT/'f64_sqrt_integer.py').read_bytes()),
            'controls':controls,'rows':records,'native_values':sum(len(v) for v in groups.values()),
            'wrong_neighbor_rne_rejections':neighbor_rejections,
            'independent_fraction_cells_and_ha_cases':fraction_count*5,
            'restoring_trace_inputs':len(trace_inputs),'restoring_invariant_checks':56*len(trace_inputs),
            'largest_observed_u64_intermediate_bits':trace_max,'strict_mkl_sample_witnesses':witnesses,
            'full_mkl_mismatches_replayed_here':False,'strict_pytorch_bitwise_gate_closed':False,
            'full_dtype_gate_closed':False,'whole_pytorch_rewrite_complete':False}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='strict_mkl_sample_witnesses'},indent=2))


if __name__=='__main__':
    main()
