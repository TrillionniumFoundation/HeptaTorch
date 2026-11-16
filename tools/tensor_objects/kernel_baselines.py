"""Checked authoring baselines for the named canonical HIA functions only."""

def conv2d(w: mutbuf, x: u64, weight: u64, stride_pack: u64, pad_pack: u64, dilation_pack: u64, groups: u64) -> u64:
    if valid(w, x) == 0 or valid(w, weight) == 0:
        return 0
    px = dptr(w, x)
    pk = dptr(w, weight)
    if getw(w, px) != 4 or getw(w, pk) != 4:
        return terror(w, 2)
    if spatial_pair(stride_pack, 1) == 0 or spatial_pair(pad_pack, 0) == 0 or spatial_pair(dilation_pack, 1) == 0:
        return terror(w, 7)
    if groups == 0 or groups > 262144:
        return terror(w, 7)
    if getw(w, px + 104) == 0 or getw(w, px + 112) == 0 or getw(w, px + 120) == 0:
        return terror(w, 2)
    if getw(w, pk + 96) == 0 or getw(w, pk + 104) == 0 or getw(w, pk + 112) == 0 or (getw(w, pk + 120) == 0):
        return terror(w, 2)
    if irem(getw(w, px + 104), groups) != 0 or irem(getw(w, pk + 96), groups) != 0:
        return terror(w, 2)
    if idiv(getw(w, px + 104), groups) != getw(w, pk + 104):
        return terror(w, 2)
    config = alloc(w, 128)
    if config == 0:
        return 0
    putw(w, config, stride_pack >> 32)
    putw(w, config + 8, stride_pack & 4294967295)
    putw(w, config + 16, pad_pack >> 32)
    putw(w, config + 24, pad_pack & 4294967295)
    putw(w, config + 32, dilation_pack >> 32)
    putw(w, config + 40, dilation_pack & 4294967295)
    putw(w, config + 48, getw(w, pk + 104))
    putw(w, config + 56, idiv(getw(w, pk + 96), groups))
    putw(w, config + 64, getw(w, pk + 112))
    putw(w, config + 72, getw(w, pk + 120))
    putw(w, config + 80, getw(w, px + 112))
    putw(w, config + 88, getw(w, px + 120))
    putw(w, config + 96, getw(w, px + 104))
    putw(w, config + 104, getw(w, pk + 104) * getw(w, pk + 112) * getw(w, pk + 120))
    oh = spatial_extent(getw(w, config + 80), getw(w, config + 64), getw(w, config), getw(w, config + 16), getw(w, config + 32))
    ow = spatial_extent(getw(w, config + 88), getw(w, config + 72), getw(w, config + 8), getw(w, config + 24), getw(w, config + 40))
    if oh == 0 or ow == 0:
        return terror(w, 2)
    putw(w, 128, getw(w, px + 96))
    putw(w, 136, getw(w, pk + 96))
    putw(w, 144, oh)
    putw(w, 152, ow)
    t = tnew(w, 4, 128, getw(w, px + 32) | getw(w, pk + 32))
    if t == 0:
        return 0
    p = dptr(w, t)
    putw(w, p + 176, config)
    n = getw(w, p + 8)
    if n > idiv(16777216, getw(w, config + 104)):
        return terror(w, 3)
    record(w, t, 22, x, weight)
    i = 0
    while i < n:
        put4(w, getw(w, p + 16) + i * 4, conv_point(w, t, i, 0))
        i = i + 1
    return t

def pool2d(w: mutbuf, op: u64, x: u64, kernel_pack: u64, stride_pack: u64, pad_pack: u64, include_pad: u64) -> u64:
    if valid(w, x) == 0:
        return 0
    if op != 23 and op != 24:
        return terror(w, 4)
    if spatial_pair(kernel_pack, 1) == 0 or spatial_pair(stride_pack, 1) == 0 or spatial_pair(pad_pack, 0) == 0 or (include_pad > 1):
        return terror(w, 7)
    p = dptr(w, x)
    if getw(w, p) != 4:
        return terror(w, 2)
    if getw(w, p + 104) == 0 or getw(w, p + 112) == 0 or getw(w, p + 120) == 0:
        return terror(w, 2)
    if pad_pack >> 32 > idiv(kernel_pack >> 32, 2) or pad_pack & 4294967295 > idiv(kernel_pack & 4294967295, 2):
        return terror(w, 7)
    config = alloc(w, 128)
    if config == 0:
        return 0
    putw(w, config, stride_pack >> 32)
    putw(w, config + 8, stride_pack & 4294967295)
    putw(w, config + 16, pad_pack >> 32)
    putw(w, config + 24, pad_pack & 4294967295)
    putw(w, config + 48, include_pad)
    putw(w, config + 64, kernel_pack >> 32)
    putw(w, config + 72, kernel_pack & 4294967295)
    putw(w, config + 80, getw(w, p + 112))
    putw(w, config + 88, getw(w, p + 120))
    putw(w, config + 104, (kernel_pack >> 32) * (kernel_pack & 4294967295))
    oh = spatial_extent(getw(w, config + 80), getw(w, config + 64), getw(w, config), getw(w, config + 16), 1)
    ow = spatial_extent(getw(w, config + 88), getw(w, config + 72), getw(w, config + 8), getw(w, config + 24), 1)
    if oh == 0 or ow == 0:
        return terror(w, 2)
    putw(w, 128, getw(w, p + 96))
    putw(w, 136, getw(w, p + 104))
    putw(w, 144, oh)
    putw(w, 152, ow)
    t = tnew(w, 4, 128, getw(w, p + 32))
    if t == 0:
        return 0
    p = dptr(w, t)
    putw(w, p + 176, config)
    n = getw(w, p + 8)
    if op == 24 and getw(w, p + 32) != 0:
        indices = alloc(w, n * 8)
        if indices == 0:
            return 0
        putw(w, p + 184, indices)
    record(w, t, op, x, 0)
    i = 0
    while i < n:
        value = pool_point(w, t, i, 0)
        if getw(w, 0) != 0:
            return 0
        put4(w, getw(w, p + 16) + i * 4, value)
        i = i + 1
    return t

def index_select(s: readbuf, w: mutbuf, a: u64, axis: u64, count: u64, offset: u64, end: u64) -> u64:
    if valid(w, a) == 0:
        return 0
    ap = dptr(w, a)
    rank = getw(w, ap)
    if rank == 0:
        return terror(w, 2)
    if axis >= 9223372036854775808:
        axis = axis + rank
    if axis >= rank:
        return terror(w, 2)
    if count > 262144:
        return terror(w, 3)
    if offset < end or offset > length(s) or offset % 8 != 0:
        return terror(w, 100)
    if count > (length(s) - offset) // 8:
        return terror(w, 100)
    dim = getw(w, ap + 96 + axis * 8)
    j = 0
    while j < count:
        if getr(s, offset + j * 8) >= dim:
            return terror(w, 7)
        j = j + 1
    indices = alloc(w, count * 8)
    if indices == 0:
        return 0
    j = 0
    while j < count:
        putw(w, indices + j * 8, getr(s, offset + j * 8))
        j = j + 1
    j = 0
    while j < rank:
        dim = getw(w, ap + 96 + j * 8)
        if j == axis:
            dim = count
        putw(w, 2048 + j * 8, dim)
        j = j + 1
    t = tnew(w, rank, 2048, getw(w, ap + 32))
    if t == 0:
        return 0
    p = dptr(w, t)
    putw(w, p + 176, axis)
    putw(w, p + 184, indices)
    record(w, t, 26, a, 0)
    j = 0
    while j < getw(w, p + 8):
        source = index_select_source(w, t, j)
        put4(w, getw(w, p + 16) + j * 4, tget(w, a, source))
        j = j + 1
    return t

def tensor_join(s: readbuf, w: mutbuf, op: u64, axis: u64, config: u64, end: u64) -> u64:
    if getw(w, 0) != 0:
        return 0
    if op != 37 and op != 38:
        return terror(w, 4)
    if config < end or config > length(s) or config % 8 != 0:
        return terror(w, 100)
    if length(s) - config < 8:
        return terror(w, 100)
    count = getr(s, config)
    if count == 0:
        return terror(w, 7)
    if count > (length(s) - config - 8) // 8:
        return terror(w, 100)
    if concat_shape(s, w, op, axis, config + 8, count) == 0:
        return 0
    if getw(w, 8) >= 256:
        return terror(w, 3)
    cursor = getw(w, 16)
    if cursor > length(w):
        return terror(w, 3)
    required = 8 + count * 16 + getw(w, 2104) * 4 * (1 + getw(w, 2096) * 2)
    if required > length(w) - cursor:
        return terror(w, 3)
    saved = alloc(w, 8 + count * 16)
    putw(w, saved, count)
    j = 0
    while j < count:
        a = getr(s, config + 8 + j * 8)
        putw(w, saved + 8 + j * 16, a)
        putw(w, saved + 16 + j * 16, tversion(w, a))
        j = j + 1
    t = tnew(w, getw(w, 2080), 2048, getw(w, 2096))
    if t == 0:
        return 0
    p = dptr(w, t)
    putw(w, p + 176, getw(w, 2088))
    putw(w, p + 184, saved)
    record(w, t, op, getr(s, config + 8), 0)
    concat_layout(w, t)
    concat_copy(w, t, 0)
    return t

def concat_extent(s: readbuf, w: mutbuf, op: u64, axis: u64, offset: u64, count: u64, reference: u64) -> u64:
    rp = dptr(w, reference)
    rank = getw(w, rp)
    extent = 0
    j = 0
    while j < count:
        p = dptr(w, getr(s, offset + j * 8))
        skip = 0
        if op == 37 and getw(w, p) == 1 and (getw(w, p + 8) == 0):
            skip = 1
        if skip == 0:
            if getw(w, p) != rank:
                return terror(w, 2)
            k = 0
            while k < rank:
                if op == 38 or k != axis:
                    if getw(w, p + 96 + k * 8) != getw(w, rp + 96 + k * 8):
                        return terror(w, 2)
                k = k + 1
            if op == 37:
                extent = extent + getw(w, p + 96 + axis * 8)
                if extent > 262144:
                    return terror(w, 3)
        j = j + 1
    if op == 38:
        extent = count
    return extent

def concat_shape(s: readbuf, w: mutbuf, op: u64, axis: u64, offset: u64, count: u64) -> u64:
    reference = 0
    requires = 0
    j = 0
    while j < count:
        a = getr(s, offset + j * 8)
        if valid(w, a) == 0:
            return 0
        p = dptr(w, a)
        rank = getw(w, p)
        requires = requires | getw(w, p + 32)
        if op == 37 and rank == 0:
            return terror(w, 2)
        if reference == 0:
            if op == 38 or rank != 1 or getw(w, p + 8) != 0:
                reference = a
        j = j + 1
    if reference == 0:
        putw(w, 2048, 0)
        putw(w, 2056, 0)
        putw(w, 2064, 0)
        putw(w, 2072, 0)
        putw(w, 2080, 1)
        putw(w, 2088, 0)
        putw(w, 2096, requires)
        putw(w, 2104, 0)
        return 1
    rank = getw(w, dptr(w, reference))
    if op == 38:
        rank = rank + 1
    if rank > 4:
        return terror(w, 2)
    if axis >= 9223372036854775808:
        axis = axis + rank
    if axis >= rank:
        return terror(w, 2)
    extent = concat_extent(s, w, op, axis, offset, count, reference)
    if getw(w, 0) != 0:
        return 0
    has_zero = 0
    j = 0
    while j < rank:
        dim = concat_dimension(w, reference, op, axis, extent, j)
        if dim > 262144:
            return terror(w, 3)
        if dim == 0:
            has_zero = 1
        j = j + 1
    n = 0
    if has_zero == 0:
        n = 1
        j = 0
        while j < rank:
            n = n * concat_dimension(w, reference, op, axis, extent, j)
            if n > 262144:
                return terror(w, 3)
            j = j + 1
    j = 0
    while j < 4:
        dim = 0
        if j < rank:
            dim = concat_dimension(w, reference, op, axis, extent, j)
        putw(w, 2048 + j * 8, dim)
        j = j + 1
    putw(w, 2080, rank)
    putw(w, 2088, axis)
    putw(w, 2096, requires)
    putw(w, 2104, n)
    return 1

def back_unary(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    a = getw(w, p + 48)
    op = getw(w, p + 40)
    n = getw(w, p + 8)
    i = 0
    while i < n:
        g = get4(w, getw(w, p + 160) + i * 4)
        if op == 5:
            x = tget(w, a, i)
            if ltf(x, 0) != 0 or absf(x) == 0:
                g = 0
        if op == 6:
            g = negf(g)
        if op == 11:
            g = divf(g, mulf(1073741824, tget(w, t, i)))
        if op == 12:
            g = mulf(g, tget(w, t, i))
        if op == 13:
            g = divf(g, tget(w, a, i))
        if op == 14:
            y = tget(w, t, i)
            g = mulf(mulf(g, y), subf(1065353216, y))
        if op == 15:
            y = tget(w, t, i)
            g = mulf(g, subf(1065353216, mulf(y, y)))
        ai = i
        if op == 10:
            ai = transindex(w, t, i)
        gadd(w, a, ai, g)
        i = i + 1
    return 1

def back_matmul(w: mutbuf, t: u64) -> u64:
    n = getw(w, dptr(w, t) + 8)
    i = 0
    while i < n:
        back_matmul_point(w, t, i)
        i = i + 1
    return 1

def back_matmul_point(w: mutbuf, t: u64, index: u64) -> u64:
    p = dptr(w, t)
    a = getw(w, p + 48)
    b = getw(w, p + 56)
    bound = getw(w, p + 184)
    grad = get4(w, getw(w, p + 160) + index * 4)
    k = 0
    while k < bound:
        ai = matmul_index(w, t, 0, index, k)
        bi = matmul_index(w, t, 1, index, k)
        gadd(w, a, ai, mulf(grad, tget(w, b, bi)))
        gadd(w, b, bi, mulf(grad, tget(w, a, ai)))
        k = k + 1
    return 1

def indexed_add(s: readbuf, w: mutbuf, a: u64, b: u64, axis: u64, config: u64, alpha: u64, op: u64, end: u64) -> u64:
    if op != 39 and op != 40:
        return terror(w, 4)
    if valid(w, a) == 0 or valid(w, b) == 0:
        return 0
    rank = getw(w, dptr(w, a))
    if rank == 0:
        rank = 1
    if axis >= 9223372036854775808:
        axis = axis + rank
    if axis >= rank:
        return terror(w, 2)
    if alpha > 4294967295:
        return terror(w, 7)
    count = indexed_add_validate(s, w, a, b, axis, config, op, end)
    if getw(w, 0) != 0:
        return 0
    aux = alloc(w, 64 + count * 8)
    if aux == 0:
        return 0
    irank = getr(s, config)
    putw(w, aux, irank)
    putw(w, aux + 8, count)
    putw(w, aux + 16, alpha)
    j = 0
    while j < irank:
        putw(w, aux + 32 + j * 8, getr(s, config + 8 + j * 8))
        j = j + 1
    j = 0
    while j < count:
        putw(w, aux + 64 + j * 8, getr(s, config + 8 + irank * 8 + j * 8))
        j = j + 1
    ap = dptr(w, a)
    requires = getw(w, ap + 32) | getw(w, dptr(w, b) + 32)
    t = tnew(w, getw(w, ap), ap + 96, requires)
    if t == 0:
        return 0
    p = dptr(w, t)
    putw(w, p + 176, axis)
    putw(w, p + 184, aux)
    record(w, t, op, a, b)
    indexed_add_forward(w, t)
    return t

def index_select_command(s: readbuf, w: mutbuf, cp: u64, end: u64) -> u64:
    return index_select(s, w, getr(s, cp + 8), getr(s, cp + 16), getr(s, cp + 24), getr(s, cp + 32), end)

def gather_command(s: readbuf, w: mutbuf, cp: u64, end: u64) -> u64:
    return gather(s, w, getr(s, cp + 8), getr(s, cp + 16), getr(s, cp + 24), end)

def indexed_add_command(s: readbuf, w: mutbuf, cp: u64, end: u64) -> u64:
    op = 39
    alpha = 1065353216
    if getr(s, cp) == 37:
        op = 40
        alpha = getr(s, cp + 40)
    return indexed_add(s, w, getr(s, cp + 8), getr(s, cp + 16), getr(s, cp + 24), getr(s, cp + 32), alpha, op, end)
