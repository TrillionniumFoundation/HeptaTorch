def scalar_add_preserving_bits(a:u64, b:u64) -> u64:
    a = a & 4294967295
    b = b & 4294967295
    aa = a & 2147483647
    bb = b & 2147483647
    if bb > 2139095040:
        return b | 4194304
    if aa > 2139095040:
        return a | 4194304
    if aa == 2139095040 and bb == 2139095040 and ((a ^ b) & 2147483648) != 0:
        return 4290772992
    return addf(a,b)
