def zero_grad_none(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0 or getw(w,p+32) == 0:
        return terror(w,5)
    if hold_release(w,getw(w,p+248)) == 0:
        return 0
    putw(w,p+248,0)
    putw(w,p+24,0)
    putw(w,p+360,0)
    return 1

def zero_grad(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0 or getw(w,p+32) == 0:
        return terror(w,5)
    if getw(w,p+360) == 0:
        return 1
    if getw(w,p+248) == 0 or getw(w,p+24) == 0:
        return terror(w,9)
    memory_fill(w,getw(w,p+24),getw(w,p+8)*4,0)
    return 1

def tgrad(w:mutbuf, t:u64, index:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0 or getw(w,p+32) == 0:
        return terror(w,5)
    if index >= getw(w,p+8):
        return terror(w,7)
    if getw(w,p+360) == 0:
        return 0
    if getw(w,p+248) == 0 or getw(w,p+24) == 0:
        return terror(w,9)
    return get4(w,getw(w,p+24)+index*4)

def ensure_backward_work(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if getw(w,p+40) == 0 and getw(w,p+248) == 0:
                if getw(w,p+360) != 0:
                    return terror(w,9)
                grad = alloc(w,getw(w,p+8)*4)
                if grad == 0:
                    return 0
                putw(w,p+24,grad)
                putw(w,p+248,getw(w,80))
            if getw(w,p+256) == 0:
                address = alloc(w,getw(w,p+8)*4)
                if address == 0:
                    return 0
                putw(w,p+160,address)
                putw(w,p+256,getw(w,80))
        i = i+1
    return 1

def release_backward_scratch(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if hold_release(w,getw(w,p+256)) == 0:
                return 0
            putw(w,p+256,0)
            putw(w,p+160,0)
            if getw(w,p+40) != 0 or getw(w,p+360) == 0:
                if hold_release(w,getw(w,p+248)) == 0:
                    return 0
                putw(w,p+248,0)
                putw(w,p+24,0)
        i = i+1
    return 1

def descriptor(w: mutbuf, rank: u64, shape: u64, requires: u64, own: u64) -> u64:
    if getw(w, 0) != 0:
        return 0
    if rank > 4 or requires > 1:
        return terror(w, 2)
    location = object_address(w, shape)
    if location > length(w) or 32 > length(w) - location:
        return terror(w, 2)
    slot = getw(w, 8)
    if slot >= 65536:
        return terror(w, 3)
    n = 1
    axis = 0
    while axis < rank:
        d = getw(w, shape + axis * 8)
        if d > 262144:
            return terror(w, 3)
        if d == 0:
            n = 0
        axis = axis + 1
    if n != 0:
        axis = 0
        while axis < rank:
            n = n * getw(w, shape + axis * 8)
            if n > 16777216:
                return terror(w, 3)
            axis = axis + 1
    if descriptor_page(w, slot) == 0:
        return 0
    if storage_capacity(w, 1, 0) == 0:
        return 0
    if own != 0 and storage_capacity(w, 3, 0) == 0:
        return 0
    p = dptr(w, slot)
    memory_fill(w, p, 384, 0)
    putw(w, p, rank)
    putw(w, p + 8, n)
    putw(w, p + 32, requires)
    putw(w, p + 64, choose(own != 0, slot, 0))
    putw(w, p + 200, 2)
    putw(w, p + 304, 1)
    axis = rank
    stride = 1
    while axis > 0:
        axis = axis - 1
        d = getw(w, shape + axis * 8)
        putw(w, p + 96 + axis * 8, d)
        putw(w, p + 128 + axis * 8, stride)
        if d != 0:
            stride = stride * d
    tid = so_slot_new(w, 1)
    tp = so_slot_lookup(w, 1, tid)
    putw(w, p + 192, tid)
    putw(w, p + 240, 1)
    so_uput(w, tp + 48, slot)
    if own != 0:
        data = alloc(w, n * 4)
        if getw(w, 0) != 0:
            return 0
        hid = getw(w, 80)
        hp = so_slot_lookup(w, 6, hid)
        sid = so_uget(w, hp + 16)
        vid = so_slot_new(w, 3)
        vp = so_slot_lookup(w, 3, vid)
        so_uput(w, vp + 24, 1)
        so_uput(w, tp + 16, sid)
        so_uput(w, tp + 24, vid)
        so_uput(w, tp + 40, n * 4)
        putw(w, p + 16, data)
        putw(w, p + 216, sid)
        putw(w, p + 224, vid)
        so_slot_drop(w, 6, hid)
    if requires != 0:
        work = alloc(w, n * 4)
        if getw(w, 0) != 0:
            return 0
        putw(w, p + 160, work)
        putw(w, p + 256, getw(w, 80))
    putw(w, 8, slot + 1)
    return slot
def object_command(s: readbuf, o: mutbuf, w: mutbuf, cp: u64, at: u64) -> u64:
    code = getr(s, cp)
    a = resolve_handle(w, getr(s, cp + 8))
    b = getr(s, cp + 16)
    c = getr(s, cp + 24)
    result = 0
    payload = 0
    first = getw(w, 8)
    if code < 41 or (code > 59 and code != 70):
        terror(w, 4)
    if code == 41:
        result = drop_tensor(w, a)
    if code == 42:
        t = public_node(w, a)
        if t != 0:
            result = weak_create(w, t)
    if code == 43:
        result = weak_lock(w, a)
    if code == 44:
        result = weak_release(w, a)
    if code == 45:
        result = public_hold_release(w, a)
    if code == 46:
        t = public_node(w, a)
        if t != 0:
            result = save_public(w, t)
    if code == 47:
        result = saved_public_check(w, a)
    if code == 48:
        result = saved_public_release(w, a)
    if code == 49 or code == 50:
        n = 0
        if code == 49:
            p = public_saved(w, a)
            if p != 0:
                if graph_saved_check(w, a) != 0:
                    layout = private_hold(w, so_uget(w, p + 56), 128)
                    n = getw(w, layout + 8)
        else:
            p = public_hold(w, a)
            if p != 0:
                sp = so_storage_live(w, so_uget(w, p + 16))
                n = so_uget(w, sp + 24) // 4
        if b > n or c > n - b:
            terror(w, 2)
        if c > (length(o) - at - 32) // 4:
            terror(w, 3)
        if getw(w, 0) == 0:
            j = 0
            while j < c:
                value = 0
                if code == 49:
                    value = saved_logical(w, a, b + j)
                else:
                    value = hold_logical(w, a, b + j)
                put4(o, at + 32 + j * 4, value)
                j = j + 1
            result = c
            payload = c * 4
    if code == 51 or code == 52:
        kind = code - 49
        p = so_slot_lookup(w, kind, a)
        if p == 0:
            terror(w, 1)
        elif length(o) - at - 32 < 64:
            terror(w, 3)
        else:
            j = 0
            while j < 8:
                putw(o, at + 32 + j * 8, so_uget(w, p + j * 8))
                j = j + 1
            result = a
            payload = 64
    if code == 53:
        status = so_audit(w)
        if status != 0:
            terror(w, status)
        else:
            result = 1
    if code == 54:
        if length(o) - at - 32 < 96:
            terror(w, 3)
        else:
            payload = object_stats(w, o, at + 32)
            result = 1
    if code == 55:
        t = public_node(w, a)
        if t != 0:
            result = data_alias(w, t)
    if code == 56:
        result = unpack_saved(w, a)
    if code == 57:
        result = backward_command(w, a, b, c)
    if code == 58:
        result = partial_command(w, o, at + 32, a, b, c)
        payload = 32
    if code == 59:
        result = gradient_present(w, a)
    if code == 70:
        if b != 0 or c != 0 or getr(s,cp+32) != 0 or getr(s,cp+40) != 0 or getr(s,cp+48) != 0 or getr(s,cp+56) != 0:
            terror(w,7)
        else:
            t = public_node(w,a)
            if t != 0:
                result = zero_grad_none(w,t)
    if getw(w, 0) == 0:
        if code == 55 or code == 56:
            result = publish_tensor(w, result)
            finish_command(w, first)
    status = getw(w, 0)
    if status != 0:
        result = 0
        payload = 0
    putw(o, at, code)
    putw(o, at + 8, status)
    putw(o, at + 16, result)
    putw(o, at + 24, payload)
    return at + 32 + payload
