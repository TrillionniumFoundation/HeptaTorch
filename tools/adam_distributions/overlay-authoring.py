def entry(s: readbuf, o: mutbuf, w: mutbuf, scratch: mutbuf) -> u32:
    if length(o) < 32:
        return 0
    reply_error(o, 0)
    error = frame_valid(s)
    if error != 0:
        reply_error(o, error)
        return 32
    putw(w, 0, 0)
    putw(w, 48, 0)
    putw(w, 56, 0)
    count = getr(s, 8)
    if count != 0:
        putw(w, 96, 8192)
    end = 32 + count * 64
    i = 0
    at = 32
    halted = 0
    while i < count and halted == 0:
        if at > length(o) - 32:
            putw(o, 24, 3)
            return at
        putw(w, 104, i)
        code = getr(s, 32 + i * 64)
        if code >= 71 and code <= 79 or code == 90 or code == 95:
            next = owned_optimizer_command(s, o, w, 32 + i * 64, at, end)
        elif code >= 80 and code <= 89:
            if code == 86 or code == 87:
                next = dist_partial_command(s, o, w, scratch, 32 + i * 64, at)
            else:
                next = dist_command(s, o, w, scratch, 32 + i * 64, at)
        elif code >= 60 and code <= 69:
            next = rng_command(s, o, w, scratch, 32 + i * 64, at, end)
        else:
            next = command(s, o, w, 32 + i * 64, at, end)
        error = getw(w, 0)
        if error != 0:
            putw(o, 24, error)
            return next
        putw(w, getw(w, 96) + i * 8, getw(o, at + 16))
        if (code == 58 or code == 86 or code == 87) and getw(o, at + 16) != 0:
            halted = 1
        at = next
        i = i + 1
        putw(o, 16, i)
    if count != 0:
        fill_range(w, 8192, count * 8, 0)
        putw(w, 96, 0)
        putw(w, 104, 0)
    putw(w, 80, 0)
    if getw(w, 0) != 0:
        reply_error(o, getw(w, 0))
        return 32
    return at

def partial_apply_node(w: mutbuf, t: u64) -> u64:
    if getw(w, dptr(w, t) + 40) == 41:
        dist_backward_bernoulli(w, t)
        if getw(w, 0) != 0:
            return 0
        return 1
    return partial_apply_node_adam(w, t)
