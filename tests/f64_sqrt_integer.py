"""Independent binary64 sqrt proofs using integers only.

No floating-point arithmetic, sqrt, Decimal, Fraction, or target implementation
is used here. A dyadic number is an exact (integer coefficient, power of two).
This is qualification code, never part of the HIA numerical runtime.
"""
from math import isqrt

SIGN = 1 << 63
INF = 0x7ff0000000000000
QNAN = 0x7ff8000000000000
MASK = SIGN - 1
FRAC = (1 << 52) - 1


def decode(bits):
    """Decode a finite nonnegative binary64 to an exact dyadic."""
    assert 0 <= bits < INF
    e, m = bits >> 52, bits & FRAC
    return ((m | (1 << 52), e - 1075) if e else (m, -1074))


def compare(a, b):
    scale = min(a[1], b[1])
    left, right = a[0] << (a[1] - scale), b[0] << (b[1] - scale)
    return (left > right) - (left < right)


def add(a, b):
    scale = min(a[1], b[1])
    return ((a[0] << (a[1] - scale)) + (b[0] << (b[1] - scale)), scale)


def square(a):
    return a[0] * a[0], 2 * a[1]


def is_nan(bits):
    return bits & MASK > INF


def is_quiet_nan(bits):
    return is_nan(bits) and bool(bits & (1 << 51))


def special_result(source, answer):
    """Return None for positive finite input; otherwise check exact semantics."""
    if source in (0, SIGN, INF):
        return answer == source
    if source & SIGN or source > INF:
        return is_quiet_nan(answer)
    return None


def sqrt_rne_bits(source):
    """Directly round sqrt(x)/2**q using integer isqrt, not guard/sticky bits.

    q = floor(log2(sqrt(x))) - 52; x/2**(2*q) is an integer N.
    If t=floor(sqrt(N)), compare 4*N with (2*t+1)**2 to choose the
    nearest representable integer significand. The equality branch is even.
    Every positive finite binary64 has a normal, finite square root.
    """
    if source in (0, SIGN, INF):
        return source
    if source & SIGN or source > INF:
        return QNAN
    m, scale = decode(source)
    exponent = (m.bit_length() - 1 + scale) // 2
    quantum = exponent - 52
    shift = scale - 2 * quantum
    assert 0 <= shift <= 105
    n = m << shift
    t = isqrt(n)
    assert (1 << 52) <= t < (1 << 53)
    midpoint_squared_times_four = (2 * t + 1) ** 2
    if 4 * n > midpoint_squared_times_four or (4 * n == midpoint_squared_times_four and t & 1):
        t += 1
    if t == 1 << 53:
        t >>= 1
        exponent += 1
    assert -537 <= exponent <= 512
    return ((exponent + 1023) << 52) | (t - (1 << 52))


def in_rne_cell(source, answer):
    """Exact squared midpoint test, independently of the isqrt result."""
    special = special_result(source, answer)
    if special is not None:
        return special
    if not 0 < answer < INF - 1:
        return False
    x, y = decode(source), decode(answer)
    lo, hi = add(decode(answer - 1), y), add(y, decode(answer + 1))
    lo, hi = (lo[0], lo[1] - 1), (hi[0], hi[1] - 1)
    low, high = compare(x, square(lo)), compare(x, square(hi))
    even = not answer & 1
    return (low > 0 or low == 0 and even) and (high < 0 or high == 0 and even)


def within_one_ulp(source, answer):
    """Prove strict error < 1 ulp of the exact result's binary binade.

    For positive finite x, e=floor(log2(x)/2), u=2**(e-52). The claim
    |answer-sqrt(x)| < u is equivalent to (answer-u)**2 < x <
    (answer+u)**2, with the lower inequality omitted when answer-u<0.
    In particular, a difference of one representable encoding is NOT used
    as a substitute for this test, and an exact one-ulp error is rejected.
    """
    special = special_result(source, answer)
    if special is not None:
        return special
    if not 0 < answer < INF:
        return False
    x, y = decode(source), decode(answer)
    exponent = (x[0].bit_length() - 1 + x[1]) // 2
    u = 1, exponent - 52
    lower, upper = add(y, (-u[0], u[1])), add(y, u)
    return ((lower[0] < 0 or compare(x, square(lower)) > 0)
            and compare(x, square(upper)) < 0)


def self_check():
    """Exact boundary/refusal controls; neither target nor host sqrt is used."""
    rows = [
        ('one-exact', 0x3ff0000000000000, 0x3ff0000000000000, True, True),
        ('one-plus-exact-ulp-refused', 0x3ff0000000000000, 0x3ff0000000000001, False, False),
        ('four-plus-exact-ulp-refused', 0x4010000000000000, 0x4000000000000001, False, False),
        ('below-one-rne', 0x3fefffffffffffff, 0x3fefffffffffffff, True, True),
        ('below-one-non-rne-but-ha', 0x3fefffffffffffff, 0x3ff0000000000000, False, True),
        ('below-one-binade-spacing-refused', 0x3fefffffffffffff, 0x3ff0000000000001, False, False),
        ('minimum-subnormal-exact', 1, 0x1e60000000000000, True, True),
        ('minimum-subnormal-plus-ulp-refused', 1, 0x1e60000000000001, False, False),
        ('positive-zero', 0, 0, True, True),
        ('negative-zero', SIGN, SIGN, True, True),
        ('negative-zero-sign-loss-refused', SIGN, 0, False, False),
        ('positive-zero-sign-flip-refused', 0, SIGN, False, False),
        ('positive-infinity', INF, INF, True, True),
        ('positive-infinity-finite-refused', INF, INF-1, False, False),
        ('negative-infinity-quiet-nan', SIGN | INF, QNAN, True, True),
        ('negative-real-quiet-nan', 0xbff0000000000000, QNAN, True, True),
        ('nan-quieted', INF+1, QNAN, True, True),
        ('signaling-result-refused', INF+1, INF+1, False, False),
    ]
    for name, source, answer, rne, ha in rows:
        assert in_rne_cell(source, answer) is rne, name
        assert within_one_ulp(source, answer) is ha, name
        correct = sqrt_rne_bits(source)
        assert in_rne_cell(source, correct) and within_one_ulp(source, correct), name
    return len(rows)
