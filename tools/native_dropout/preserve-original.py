def incoming_mark_preserve_targets(w: mutbuf, limit: u64, epoch: u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 168) == epoch:
            if getw(w, p + 544) != 0 or getw(w, p + 552) != 0 or getw(w, p + 560) != 0 or (getw(w, p + 568) != 0):
                return terror(w, 9)
        i = i + 1
    i = 1
    while i <= limit:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 168) == epoch and (getw(w, p + 40) == 41):
            parent = gradient_parent_at(w, i, 0)
            if parent != 0:
                q = dptr(w, parent)
                if getw(w, q + 304) == 0 or getw(w, q + 168) != epoch:
                    return terror(w, 9)
                putw(w, q + 544, 1)
        i = i + 1
    return 1

def incoming_reserve_preserve_storage(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    if getw(w, p + 544) != 1:
        return terror(w, 9)
    address = alloc(w, getw(w, p + 8) * 4)
    if address == 0:
        return 0
    hold = getw(w, 80)
    hp = so_slot_lookup(w, 6, hold)
    if hp == 0:
        return terror(w, 9)
    sid = so_uget(w, hp + 16)
    if storage_capacity(w, 3, 0) == 0:
        return 0
    vid = so_slot_new(w, 3)
    if vid == 0:
        return terror(w, 9)
    vp = so_slot_lookup(w, 3, vid)
    so_uput(w, vp + 24, 1)
    putw(w, p + 544, address)
    putw(w, p + 552, hold)
    putw(w, p + 560, sid)
    putw(w, p + 568, vid)
    return 1
