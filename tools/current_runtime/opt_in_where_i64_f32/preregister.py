#!/usr/bin/env python3
"""Emit preregistered I64/F32 where cases; Python stdlib only, no PyTorch."""
import itertools
import json
import math

I64, F32 = 2, 0
F32_SPECIAL = (
    0, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x00800000,
    0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345,
    0xff812345, 0x7f7fffff, 0xff7fffff, 0x3f800000, 0xbf800000,
)


def i64_to_ieee32(n):
    """Compute bits by comparing integer distances to adjacent F32 numbers."""
    assert -(1 << 63) <= n < (1 << 63)
    if n == 0:
        return 0
    a = abs(n)
    e = a.bit_length() - 1
    if e < 24:
        significand = a * (1 << (23 - e))
    else:
        spacing = 1 << (e - 23)
        base = a - (a % spacing)
        upper = base + spacing
        lower_sig = base // spacing
        sigs = ((a - base, lower_sig & 1, lower_sig),
                (upper - a, (lower_sig + 1) & 1, lower_sig + 1))
        _, _, significand = min(sigs)
        if significand == (1 << 24):
            e += 1
            significand = 1 << 23
    return ((n < 0) << 31) | ((e + 127) << 23) | (significand - (1 << 23))


def array(dtype, shape, words):
    assert len(words) == math.prod(shape)
    width = {1: 1, I64: 8, F32: 4}[dtype]
    vals = [v & ((1 << (8 * width)) - 1) for v in words]
    byte_hex = b''.join(v.to_bytes(width, 'little') for v in vals).hex()
    return dict(dtype=dtype, shape=shape, words=vals, byte_hex=byte_hex)


def join(*shapes):
    rank = max(map(len, shapes))
    result = []
    for dim in zip(*[(1,) * (rank - len(s)) + tuple(s) for s in shapes]):
        extents = {v for v in dim if v != 1}
        assert len(extents) <= 1
        result.append(next(iter(extents), 1))
    return result


def fetch(a, coord):
    shape = a['shape']
    offset = 0
    aligned = coord[len(coord) - len(shape):]
    for c, d in zip(aligned, shape):
        offset = offset * d + (0 if d == 1 else c)
    w = a['words'][offset]
    if a['dtype'] != I64:
        return w
    return i64_to_ieee32(w - (1 << 64) if w >> 63 else w)


def case(label, m, x, y, order):
    shape = join(m['shape'], x['shape'], y['shape'])
    rows = [fetch(x if fetch(m, c) else y, c)
            for c in itertools.product(*(range(d) for d in shape))]
    return dict(label=label, mask=m, x=x, y=y, lifetime_order=order,
                result=array(F32, shape, rows))


def make():
    assert i64_to_ieee32(2**62 + 2**38 + 1) == 0x5e800001
    assert i64_to_ieee32(2**62 + 2**38) == 0x5e800000
    assert i64_to_ieee32(-(2**62 + 2**38 + 1)) == 0xde800001
    assert i64_to_ieee32(2**63 - 1) == 0x5f000000
    assert i64_to_ieee32(-2**63) == 0xdf000000
    seeds = [0, 1, -1, -2**63, -2**63+1, -2**63+2**38,
             -2**63+2**38+1, 2**63-1, 2**63-2**38, 2**63-2**38-1]
    traps = []
    for e in (24, 25, 32, 53, 54, 55, 56, 60, 62):
        midpoint = 1 << (e - 24)
        base = 1 << e
        for offset in (-1, 0, 1):
            for mult in (0, 1, 3, 5):
                for sign in (-1, 1):
                    seeds.append(sign * (base + mult*midpoint + offset))
        if e >= 54:
            for sign in (-1, 1):
                for off in (-1, 1):
                    for mult in (1, 3):
                        n = sign * (base + mult*midpoint + off)
                        if i64_to_ieee32(n) != i64_to_ieee32(sign * (base + mult*midpoint)):
                            traps.append(n)
    words = list(dict.fromkeys(seeds))
    assert len(traps) >= 20
    n = len(words)
    mask = array(1, [n, 2], [1, 0] * n)
    itr = array(I64, [n, 1], words)
    real = array(F32, [n, 2], [F32_SPECIAL[i % len(F32_SPECIAL)] for row in range(n) for i in (row + 3, row)])
    R = [case('i64-left-all-boundaries', mask, itr, real, 'inputs-first'),
         case('i64-right-all-boundaries', array(1, [n, 2], [0, 1] * n), real, itr, 'result-first'),
         case('three-independently-determined-axes', array(1, [2, 1, 1], [1,0]),
              array(I64, [1, 3, 1], [traps[0], traps[-1], -(1<<63)]),
              array(F32, [1, 1, 4], [0xff812345, 0x80000000, 0x7f800000, 0x00000001]), 'inputs-first'),
         case('same-shape-mixed-alternating', array(1,[4],[1,0,1,0]),
              array(F32,[4],[0x7f812345,0,0xff812345,0]),
              array(I64,[4],[7,2**24+3,8,-(2**24+3)]), 'result-first'),
         case('empty-value-axis', array(1,[],[1]), array(I64,[2,0],[]),
              array(F32,[1,1],[0xff812345]), 'inputs-first'),
         case('empty-mask-axis', array(1,[0,1],[]), array(F32,[1,4],[0,1,2,3]),
              array(I64,[1,1],[-1]), 'result-first')]
    denials = []
    for irank, frank in ((0, 1), (1, 0), (0, 0)):
        i = array(I64, [1]*irank, [2**24+1])
        f = array(F32, [1]*frank, [0x7f812345])
        for rev in (False, True):
            x, y = (f, i) if rev else (i, f)
            denials.append(dict(label=f'rank0-new-scope-excluded-i{irank}-f{frank}-rev{int(rev)}',
                                mask=array(1,[],[rev]), x=x, y=y, expected_native_status=7))
    return dict(scope='preregistered prediction only; no PyTorch or HIA qualification',
                positives=R, rank0_native_policy_denials=denials,
                all_positive_rows_expected_real_predecessor_denials=True,
                direct_double_rounding_traps=[dict(i64=n, direct_f32=i64_to_ieee32(n)) for n in traps])


if __name__ == '__main__':
    print(json.dumps(make(), indent=2, sort_keys=True))
