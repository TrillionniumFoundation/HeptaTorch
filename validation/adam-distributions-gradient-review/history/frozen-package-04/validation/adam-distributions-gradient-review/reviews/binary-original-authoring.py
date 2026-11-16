def binary(w: mutbuf, op: u64, a: u64, b: u64) -> u64:
    if valid(w, a) == 0 or valid(w, b) == 0:
        return 0
    if op < 1 or op > 4:
        return terror(w, 4)
    rank = bshape(w, a, b)
    req = getw(w, dptr(w, a) + 32) | getw(w, dptr(w, b) + 32)
    t = tnew(w, rank, 128, req)
    if t == 0:
        return 0
    n = getw(w, dptr(w, t) + 8)
    data = getw(w, dptr(w, t) + 16)
    i = 0
    while i < n:
        ai = bindex(w, t, a, i)
        bi = bindex(w, t, b, i)
        x = tget(w, a, ai)
        y = tget(w, b, bi)
        put4(w, data + i * 4, scalar_binary(op, x, y))
        i = i + 1
    return record(w, t, op, a, b)
