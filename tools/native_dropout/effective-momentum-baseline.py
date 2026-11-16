def optimizer_sgd_occurrence(w: mutbuf, gp: u64, item: u64) -> u64:
    parameter = getw(w, item)
    if parameter == 0:
        return 1
    p = dptr(w, parameter)
    n = getw(w, p + 8)
    if getw(w, p + 360) == 0 or gradient_binding(w, parameter) == 0:
        return terror(w, 9)
    if version_can_bump(w, parameter, 1) == 0:
        return 0
    momentum = getw(w, gp + 8)
    buffer = getw(w, item + 8)
    created = 0
    if absf(momentum) != 0:
        if buffer == 0:
            buffer = optimizer_new_momentum(w, parameter)
            if buffer == 0:
                return 0
            putw(w, item + 8, buffer)
            putw(w, item + 24, 1)
            created = 1
        elif version_can_bump(w, buffer, 2) == 0:
            return 0
    i = 0
    while i < n:
        optimizer_sgd_element(w, gp, parameter, buffer, created, i)
        i = i + 1
    if absf(momentum) != 0 and created == 0:
        bump_version(w, buffer)
        bump_version(w, buffer)
    return bump_version(w, parameter)

def pointwise_layout(w: mutbuf, t: u64, a: u64, b: u64) -> u64:
    p = dptr(w, t)
    pa = dptr(w, a)
    pb = dptr(w, b)
    rank = getw(w, p)
    if incoming_same_shape(w, a, b) != 0:
        if is_contiguous(w, a) != 0 and is_contiguous(w, b) != 0:
            return 1
        if pointwise_channels_last(w, a) != 0 and pointwise_channels_last(w, b) != 0:
            return pointwise_write_order(w, p, 131841, 0)
        if is_dense(w, a) != 0 and is_dense(w, b) != 0:
            same = 1
            i = 0
            while i < rank:
                if getw(w, pa + 128 + i * 8) != getw(w, pb + 128 + i * 8):
                    same = 0
                i = i + 1
            if same != 0:
                i = 0
                while i < rank:
                    putw(w, p + 128 + i * 8, getw(w, pb + 128 + i * 8))
                    i = i + 1
                return 1
    order = pointwise_axis_order(w, p, pa, pb)
    inverted = 1
    i = 0
    while i < rank:
        if order >> i * 8 & 255 != rank - i - 1:
            inverted = 0
        i = i + 1
    if inverted != 0:
        return 1
    return pointwise_write_order(w, p, order, 1)
