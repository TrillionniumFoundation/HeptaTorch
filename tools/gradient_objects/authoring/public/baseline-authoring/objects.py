def object_address(w: mutbuf, address: u64) -> u64:
    if address >= 9223372036854775808:
        return 139264 + so_data_base(so_uget(w, 16)) + (address & 9223372036854775807)
    return address

def getw(w: mutbuf, p: u64) -> u64:
    p = object_address(w, p)
    x = 0
    j = 0
    while j < 8:
        x = x | byte(w, p + j) << j * 8
        j = j + 1
    return x

def putw(w: mutbuf, p: u64, x: u64) -> u64:
    p = object_address(w, p)
    j = 0
    while j < 8:
        store(w, p + j, x >> j * 8)
        j = j + 1
    return x

def get4(w: mutbuf, p: u64) -> u64:
    p = object_address(w, p)
    return byte(w, p) | byte(w, p + 1) << 8 | byte(w, p + 2) << 16 | byte(w, p + 3) << 24

def put4(w: mutbuf, p: u64, x: u64) -> u64:
    p = object_address(w, p)
    store(w, p, x)
    store(w, p + 1, x >> 8)
    store(w, p + 2, x >> 16)
    store(w, p + 3, x >> 24)
    return x

def memory_fill(w: mutbuf, address: u64, size: u64, value: u64) -> u64:
    fill_range(w, object_address(w, address), size, value)
    return 0

def storage_capacity(w: mutbuf, kind: u64, amount: u64) -> u64:
    if getw(w, 0) != 0:
        return 0
    slots = so_uget(w, 16)
    pool = so_uget(w, 24)
    need_slots = slots
    need_pool = pool
    if kind != 0 and so_slot_available(w, kind) == 0:
        if slots == 65536:
            return terror(w, 3)
        need_slots = slots * 2
    if amount > 1073741824:
        return terror(w, 3)
    if so_extent_room(w, amount) == 0:
        amount = so_align16(amount)
        if amount > 1073741824 - pool:
            return terror(w, 3)
        while need_pool < pool + amount:
            need_pool = need_pool * 2
    if need_slots > slots or need_pool > pool:
        if need_slots > getw(w, 48):
            putw(w, 48, need_slots)
        if need_pool > getw(w, 56):
            putw(w, 56, need_pool)
        return terror(w, 3)
    return 1

def alloc(w: mutbuf, amount: u64) -> u64:
    if storage_capacity(w, 2, amount) == 0:
        return 0
    if storage_capacity(w, 6, 0) == 0:
        return 0
    extent = so_extent_allocate(w, amount)
    if extent == 0:
        return terror(w, 9)
    sid = so_slot_new(w, 2)
    hid = so_slot_new(w, 6)
    if sid == 0 or hid == 0:
        return terror(w, 9)
    sp = so_slot_lookup(w, 2, sid)
    hp = so_slot_lookup(w, 6, hid)
    so_uput(w, sp + 16, extent - 1)
    so_uput(w, sp + 24, amount)
    so_uput(w, sp + 32, so_align16(amount))
    so_uput(w, sp + 40, 1)
    so_uput(w, sp + 48, 1)
    so_uput(w, hp + 16, sid)
    so_uput(w, hp + 24, 1)
    putw(w, 80, hid)
    address = 9223372036854775808 + extent - 1
    memory_fill(w, address, amount, 0)
    return address

def hold_release(w: mutbuf, id: u64) -> u64:
    if id == 0:
        return 1
    hp = so_slot_lookup(w, 6, id)
    if hp == 0:
        return terror(w, 9)
    if so_storage_release(w, so_uget(w, hp + 16)) != 0:
        return terror(w, 9)
    if so_slot_drop(w, 6, id) != 0:
        return terror(w, 9)
    return 1

def init_state(w: mutbuf, slots: u64, pool: u64) -> u64:
    if length(w) != 139264 + so_state_size(slots, pool):
        return 0
    fill_range(w, 0, length(w), 0)
    so_state_init(w, slots, pool)
    putw(w, 8, 1)
    putw(w, 64, 3694451708891714632)
    return 1

def clone_state(source: readbuf, target: mutbuf, slots: u64, pool: u64) -> u64:
    if so_rget(source, 16) == slots and so_rget(source, 24) == pool:
        copy_range(target, source, 0, 0, length(source))
        return 0
    copy_range(target, source, 0, 0, 139264)
    return so_state_migrate(source, target, slots, pool)

def object_size(slots: u64, pool: u64) -> u64:
    return 139264 + so_state_size(slots, pool)

def state_header(w: mutbuf) -> u64:
    if length(w) == 0:
        return 0
    if length(w) < 139264 + 1840 + 131072:
        return 7
    if so_state_header(w) != 0 or getw(w, 64) != 3694451708891714632:
        return 7
    if getw(w, 0) != 0 or getw(w, 48) != 0 or getw(w, 56) != 0:
        return 7
    if getw(w, 88) != 0 or getw(w, 96) != 0 or getw(w, 104) != 0 or (getw(w, 112) != 0):
        return 7
    next = getw(w, 8)
    if next == 0 or next > 65536 or getw(w, 40) >= next:
        return 7
    if getw(w, 32) == 0 and getw(w, 40) != 0:
        return 7
    i = 0
    while i < 256:
        base = getw(w, 4096 + i * 8)
        hid = getw(w, 6144 + i * 8)
        if base != 0:
            hp = so_slot_lookup(w, 6, hid)
            if hp == 0 or so_uget(w, hp + 24) != 1:
                return 7
            sp = so_storage_live(w, so_uget(w, hp + 16))
            if sp == 0 or so_uget(w, sp + 24) != 98304:
                return 7
            if base != 9223372036854775808 + so_uget(w, sp + 16):
                return 7
        elif hid != 0:
            return 7
        i = i + 1
    return 0

def dptr(w: mutbuf, t: u64) -> u64:
    page = getw(w, 4096 + (t >> 8) * 8)
    return page + (t & 255) * 384

def live_id(w: mutbuf, slot: u64) -> u64:
    return slot

def valid(w: mutbuf, t: u64) -> u64:
    if getw(w, 0) != 0:
        return 0
    if t == 0 or t >= getw(w, 8):
        return terror(w, 1)
    if getw(w, dptr(w, t) + 304) == 0:
        return terror(w, 1)
    return 1

def public_node(w: mutbuf, id: u64) -> u64:
    if getw(w, 0) != 0:
        return 0
    id = resolve_handle(w, id)
    tp = so_tensor_live(w, id)
    if tp == 0:
        return terror(w, 1)
    node = so_uget(w, tp + 48)
    if node == 0 or node >= getw(w, 8):
        return terror(w, 1)
    p = dptr(w, node)
    if getw(w, p + 304) == 0 or getw(w, p + 192) != id or getw(w, p + 200) != 1:
        return terror(w, 1)
    return node

def public_optional(w: mutbuf, id: u64) -> u64:
    if id == 0:
        return 0
    return public_node(w, id)

def wire_node(s: readbuf, w: mutbuf, at: u64) -> u64:
    return public_node(w, getr(s, at))

def resolve_handle(w: mutbuf, id: u64) -> u64:
    if id >> 24 == 254:
        ordinal = id & 16777215
        if ordinal == 0 or ordinal > getw(w, 104) or getw(w, 96) == 0:
            return terror(w, 1)
        return getw(w, getw(w, 96) + (ordinal - 1) * 8)
    return id

def descriptor_page(w: mutbuf, slot: u64) -> u64:
    address = 4096 + (slot >> 8) * 8
    if getw(w, address) == 0:
        page = alloc(w, 98304)
        if getw(w, 0) != 0:
            return 0
        putw(w, address, page)
        putw(w, 6144 + (slot >> 8) * 8, getw(w, 80))
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
        grad = alloc(w, n * 4)
        if getw(w, 0) != 0:
            return 0
        putw(w, p + 24, grad)
        putw(w, p + 248, getw(w, 80))
        work = alloc(w, n * 4)
        if getw(w, 0) != 0:
            return 0
        putw(w, p + 160, work)
        putw(w, p + 256, getw(w, 80))
    putw(w, 8, slot + 1)
    return slot

def bind_alias(w: mutbuf, t: u64, source: u64) -> u64:
    p = dptr(w, t)
    q = dptr(w, source)
    sid = getw(w, q + 216)
    vid = getw(w, q + 224)
    if so_storage_retain(w, sid) != 0 or so_version_retain(w, vid) != 0:
        return terror(w, 9)
    tp = so_slot_lookup(w, 1, getw(w, p + 192))
    so_uput(w, tp + 16, sid)
    so_uput(w, tp + 24, vid)
    so_uput(w, tp + 32, getw(w, q + 232))
    sp = so_storage_live(w, sid)
    so_uput(w, tp + 40, so_uget(w, sp + 24) - getw(w, q + 232))
    putw(w, p + 216, sid)
    putw(w, p + 224, vid)
    putw(w, p + 232, getw(w, q + 232))
    return 1

def tget(w: mutbuf, t: u64, index: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if index >= getw(w, p + 8):
        return terror(w, 7)
    if so_storage_live(w, getw(w, p + 216)) == 0:
        return terror(w, 1)
    return get4(w, taddr(w, t, index))

def tversion(w: mutbuf, t: u64) -> u64:
    p = so_version_live(w, getw(w, dptr(w, t) + 224))
    if p == 0:
        return terror(w, 1)
    return so_uget(w, p + 16)

def version_can_bump(w: mutbuf, t: u64, amount: u64) -> u64:
    p = so_version_live(w, getw(w, dptr(w, t) + 224))
    if p == 0:
        return terror(w, 1)
    if amount > 18446744073709551615 - so_uget(w, p + 16):
        return terror(w, 8)
    return 1

def bump_version(w: mutbuf, t: u64) -> u64:
    if version_can_bump(w, t, 1) == 0:
        return 0
    p = so_version_live(w, getw(w, dptr(w, t) + 224))
    so_uput(w, p + 16, so_uget(w, p + 16) + 1)
    return 1

def tset(w: mutbuf, t: u64, index: u64, bits: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if index >= getw(w, p + 8) or bits > 4294967295:
        return terror(w, 7)
    if version_can_bump(w, t, 1) == 0:
        return 0
    put4(w, taddr(w, t, index), bits)
    return bump_version(w, t)

def zero_grad(w: mutbuf, t: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if getw(w, p + 40) != 0 or getw(w, p + 32) == 0:
        return terror(w, 5)
    memory_fill(w, getw(w, p + 24), getw(w, p + 8) * 4, 0)
    return 1

def sgd(w: mutbuf, t: u64, lr: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if getw(w, p + 40) != 0 or getw(w, p + 32) == 0:
        return terror(w, 5)
    if lr > 4294967295 or ltf(lr, 0) != 0 or absf(lr) >= 2139095040:
        return terror(w, 7)
    if version_can_bump(w, t, 1) == 0:
        return 0
    n = getw(w, p + 8)
    i = 0
    while i < n:
        address = taddr(w, t, i)
        g = get4(w, getw(w, p + 24) + i * 4)
        put4(w, address, subf(get4(w, address), mulf(lr, g)))
        i = i + 1
    return bump_version(w, t)

def is_dense(w: mutbuf, a: u64) -> u64:
    p = dptr(w, a)
    if getw(w, p + 8) == 0:
        return 1
    rank = getw(w, p)
    used = 0
    expected = 1
    done = 0
    while done < rank:
        choice = 4
        best = 18446744073709551615
        j = 0
        while j < rank:
            if used & 1 << j == 0 and getw(w, p + 96 + j * 8) > 1:
                stride = getw(w, p + 128 + j * 8)
                if stride < best:
                    best = stride
                    choice = j
            j = j + 1
        if choice == 4:
            return 1
        if best != expected:
            return 0
        expected = expected * getw(w, p + 96 + choice * 8)
        used = used | 1 << choice
        done = done + 1
    return 1

def clone(w: mutbuf, a: u64) -> u64:
    if valid(w, a) == 0:
        return 0
    p = dptr(w, a)
    t = tnew(w, getw(w, p), p + 96, getw(w, p + 32))
    if t == 0:
        return 0
    q = dptr(w, t)
    if is_dense(w, a) != 0:
        j = 0
        while j < getw(w, p):
            putw(w, q + 128 + j * 8, getw(w, p + 128 + j * 8))
            j = j + 1
    i = 0
    while i < getw(w, p + 8):
        put4(w, taddr(w, t, i), tget(w, a, i))
        i = i + 1
    return record(w, t, 17, a, 0)

def detach(w: mutbuf, a: u64) -> u64:
    if valid(w, a) == 0:
        return 0
    pa = dptr(w, a)
    rank = getw(w, pa)
    t = descriptor(w, rank, pa + 96, 0, 0)
    if t == 0:
        return 0
    pt = dptr(w, t)
    putw(w, pt + 16, getw(w, pa + 16))
    putw(w, pt + 64, getw(w, pa + 64))
    i = 0
    while i < rank:
        putw(w, pt + 128 + i * 8, getw(w, pa + 128 + i * 8))
        i = i + 1
    if bind_alias(w, t, a) == 0:
        return 0
    return t

def saved_create(w: mutbuf, t: u64, layout: u64) -> u64:
    if storage_capacity(w, 5, 0) == 0:
        return 0
    p = dptr(w, t)
    sid = getw(w, p + 216)
    vid = getw(w, p + 224)
    if so_storage_retain(w, sid) != 0 or so_version_retain(w, vid) != 0:
        return terror(w, 9)
    id = so_slot_new(w, 5)
    sp = so_slot_lookup(w, 5, id)
    so_uput(w, sp + 16, sid)
    so_uput(w, sp + 24, vid)
    so_uput(w, sp + 32, tversion(w, t))
    so_uput(w, sp + 40, getw(w, p + 232))
    storage = so_storage_live(w, sid)
    so_uput(w, sp + 48, so_uget(w, storage + 24) - getw(w, p + 232))
    so_uput(w, sp + 56, layout)
    return id

def saved_release(w: mutbuf, id: u64) -> u64:
    if id == 0:
        return 1
    p = so_saved_live(w, id)
    if p == 0:
        return terror(w, 9)
    sid = so_uget(w, p + 16)
    vid = so_uget(w, p + 24)
    layout = so_uget(w, p + 56)
    if so_version_release(w, vid) != 0 or so_storage_release(w, sid) != 0:
        return terror(w, 9)
    if so_slot_drop(w, 5, id) != 0:
        return terror(w, 9)
    return hold_release(w, layout)

def graph_saved_check(w: mutbuf, id: u64) -> u64:
    if id == 0:
        return 1
    status = so_saved_check(w, id)
    if status == 5:
        return terror(w, 6)
    if status != 0:
        return terror(w, 9)
    return 1

def node_retain(w: mutbuf, t: u64) -> u64:
    if t == 0:
        return 1
    p = dptr(w, t)
    refs = getw(w, p + 208)
    if refs >= 4294967295:
        return terror(w, 8)
    putw(w, p + 208, refs + 1)
    return 1

def node_release(w: mutbuf, t: u64) -> u64:
    if t == 0:
        return 1
    p = dptr(w, t)
    refs = getw(w, p + 208)
    if refs == 0:
        return terror(w, 9)
    putw(w, p + 208, refs - 1)
    return 1

def record(w: mutbuf, t: u64, op: u64, a: u64, b: u64) -> u64:
    if t == 0 or getw(w, 0) != 0:
        return 0
    p = dptr(w, t)
    putw(w, p + 40, op)
    putw(w, p + 48, a)
    putw(w, p + 56, b)
    if op == 10 or op == 16:
        if bind_alias(w, t, a) == 0:
            return 0
    if getw(w, p + 32) == 0:
        return t
    if op == 37 or op == 38:
        saved = getw(w, p + 184)
        count = getw(w, saved)
        j = 0
        while j < count:
            node_retain(w, getw(w, saved + 8 + j * 16))
            j = j + 1
    else:
        node_retain(w, a)
        if b != 0:
            node_retain(w, b)
    req_a = getw(w, dptr(w, a) + 32)
    req_b = 0
    if b != 0:
        req_b = getw(w, dptr(w, b) + 32)
    save_a = 0
    save_b = 0
    save_self = 0
    if op == 3:
        save_a = req_b
        save_b = req_a
    if op == 4:
        save_a = req_b
        save_b = req_a | req_b
    if op == 9 or op == 25:
        save_a = req_b
        save_b = req_a
    if op == 22:
        save_a = 1
        save_b = 1
    if op == 13 or op == 23 or op == 24 or (op == 27):
        save_a = 1
    if op == 40:
        save_b = req_b
    if op == 5 or op == 11 or op == 12 or (op == 14) or (op == 15) or (op == 20) or (op == 21):
        save_self = 1
    if save_a != 0:
        copy = 0
        if op == 25:
            copy = matmul_copy_side(w, a, b, 0)
        if copy != 0:
            putw(w, p + 280, saved_copy(w, a))
            putw(w, p + 344, 1)
        else:
            putw(w, p + 280, saved_create(w, a, 0))
    if save_b != 0:
        copy = 0
        if op == 25:
            copy = matmul_copy_side(w, a, b, 1)
        if copy != 0:
            putw(w, p + 288, saved_copy(w, b))
            putw(w, p + 352, 1)
        else:
            putw(w, p + 288, saved_create(w, b, 0))
    if save_self != 0:
        putw(w, p + 296, saved_create(w, t, 0))
    if getw(w, 0) != 0:
        return 0
    return t

def tensor_row_release(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    if getw(w, p + 240) == 0:
        return 1
    tid = getw(w, p + 192)
    tp = so_slot_lookup(w, 1, tid)
    if tp == 0:
        return terror(w, 9)
    if so_version_release(w, so_uget(w, tp + 24)) != 0:
        return terror(w, 9)
    if so_storage_release(w, so_uget(w, tp + 16)) != 0:
        return terror(w, 9)
    if so_slot_drop(w, 1, tid) != 0:
        return terror(w, 9)
    putw(w, p + 240, 0)
    return 1

def graph_node_release(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    if getw(w, p + 32) != 0 and getw(w, p + 40) != 0:
        op = getw(w, p + 40)
        if op == 37 or op == 38:
            saved = getw(w, p + 184)
            j = 0
            while j < getw(w, saved):
                node_release(w, getw(w, saved + 8 + j * 16))
                j = j + 1
        else:
            node_release(w, getw(w, p + 48))
            node_release(w, getw(w, p + 56))
    saved_release(w, getw(w, p + 280))
    saved_release(w, getw(w, p + 288))
    saved_release(w, getw(w, p + 296))
    tensor_row_release(w, t)
    hold_release(w, getw(w, p + 248))
    hold_release(w, getw(w, p + 256))
    hold_release(w, getw(w, p + 264))
    hold_release(w, getw(w, p + 272))
    hold_release(w, getw(w, p + 320))
    putw(w, p + 304, 0)
    return 1

def graph_collect_since(w: mutbuf, first: u64) -> u64:
    i = getw(w, 8) - 1
    while i >= first and i > 0:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 200) == 0:
            if getw(w, p + 208) == 0:
                graph_node_release(w, i)
            elif getw(w, p + 40) != 0 or getw(w, p + 32) == 0:
                tensor_row_release(w, i)
        i = i - 1
    return choose(getw(w, 0) == 0, 1, 0)

def graph_collect(w: mutbuf) -> u64:
    return graph_collect_since(w, 1)

def publish_tensor(w: mutbuf, t: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    putw(w, p + 200, 1)
    return getw(w, p + 192)

def finish_command(w: mutbuf, first: u64) -> u64:
    i = first
    while i < getw(w, 8):
        p = dptr(w, i)
        if getw(w, p + 200) == 2:
            putw(w, p + 200, 0)
        i = i + 1
    return graph_collect_since(w, first)

def drop_tensor(w: mutbuf, id: u64) -> u64:
    t = public_node(w, id)
    if t == 0:
        return 0
    putw(w, dptr(w, t) + 200, 0)
    return graph_collect(w)

def graph_reset(w: mutbuf, keep: u64) -> u64:
    if getw(w, 0) != 0:
        return 0
    old_next = getw(w, 8)
    if keep >= old_next or keep > 65535:
        return terror(w, 7)
    if getw(w, 32) != 0 and keep != getw(w, 40):
        return terror(w, 7)
    if getw(w, 32) == 18446744073709551615:
        return terror(w, 8)
    i = 1
    while i <= keep:
        p = dptr(w, i)
        if getw(w, p + 304) == 0 or getw(w, p + 200) != 1:
            return terror(w, 6)
        if getw(w, p + 40) != 0 or getw(w, p + 64) != i:
            return terror(w, 6)
        i = i + 1
    i = keep + 1
    while i < old_next:
        putw(w, dptr(w, i) + 200, 0)
        i = i + 1
    graph_collect(w)
    if getw(w, 0) != 0:
        return 0
    i = keep + 1
    while i < old_next:
        p = dptr(w, i)
        if getw(w, p + 304) != 0:
            return terror(w, 9)
        memory_fill(w, p, 384, 0)
        i = i + 1
    page = (keep >> 8) + 1
    while page < 256:
        hold_release(w, getw(w, 6144 + page * 8))
        putw(w, 4096 + page * 8, 0)
        putw(w, 6144 + page * 8, 0)
        page = page + 1
    putw(w, 8, keep + 1)
    putw(w, 40, keep)
    putw(w, 32, getw(w, 32) + 1)
    return 1

def mark_concat(w: mutbuf, t: u64, epoch: u64) -> u64:
    saved = getw(w, dptr(w, t) + 184)
    j = 0
    while j < getw(w, saved):
        a = getw(w, saved + 8 + j * 16)
        p = dptr(w, a)
        if getw(w, p + 32) != 0:
            putw(w, p + 168, epoch)
        j = j + 1
    return 1

def mark_graph(w: mutbuf, t: u64, epoch: u64) -> u64:
    putw(w, dptr(w, t) + 168, epoch)
    i = t
    while i > 0:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 168) == epoch and (getw(w, p + 40) != 0):
            if graph_saved_check(w, getw(w, p + 280)) == 0:
                return 0
            if graph_saved_check(w, getw(w, p + 288)) == 0:
                return 0
            if graph_saved_check(w, getw(w, p + 296)) == 0:
                return 0
            if indexed_add_backward_check(w, i) == 0:
                return 0
            op = getw(w, p + 40)
            if op == 37 or op == 38:
                mark_concat(w, i, epoch)
            else:
                a = getw(w, p + 48)
                b = getw(w, p + 56)
                if getw(w, dptr(w, a) + 32) != 0:
                    putw(w, dptr(w, a) + 168, epoch)
                if b != 0:
                    if getw(w, dptr(w, b) + 32) != 0:
                        putw(w, dptr(w, b) + 168, epoch)
        i = i - 1
    return 1

def back_binary(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    a = getw(w, p + 48)
    b = getw(w, p + 56)
    req_a = getw(w, dptr(w, a) + 32)
    req_b = getw(w, dptr(w, b) + 32)
    op = getw(w, p + 40)
    n = getw(w, p + 8)
    i = 0
    while i < n:
        ai = bindex(w, t, a, i)
        bi = bindex(w, t, b, i)
        g = get4(w, getw(w, p + 160) + i * 4)
        if req_a != 0:
            da = g
            if op == 3:
                da = mulf(g, tget(w, b, bi))
            if op == 4:
                da = divf(g, tget(w, b, bi))
            gadd(w, a, ai, da)
        if req_b != 0:
            db = g
            if op == 2:
                db = negf(g)
            if op == 3:
                db = mulf(g, tget(w, a, ai))
            if op == 4:
                x = tget(w, a, ai)
                y = tget(w, b, bi)
                db = negf(mulf(g, divf(divf(x, y), y)))
            gadd(w, b, bi, db)
        i = i + 1
    return 1

def private_hold(w: mutbuf, id: u64, size: u64) -> u64:
    p = so_slot_lookup(w, 6, id)
    if p == 0 or so_uget(w, p + 24) != 1:
        return 0
    sp = so_storage_live(w, so_uget(w, p + 16))
    if sp == 0 or so_uget(w, sp + 24) != size:
        return 0
    return 9223372036854775808 + so_uget(w, sp + 16)

def public_saved(w: mutbuf, id: u64) -> u64:
    p = so_saved_live(w, id)
    if p == 0 or so_uget(w, p + 56) == 0:
        return terror(w, 1)
    if private_hold(w, so_uget(w, p + 56), 128) == 0:
        return terror(w, 1)
    return p

def public_hold(w: mutbuf, id: u64) -> u64:
    p = so_slot_lookup(w, 6, id)
    if p == 0 or so_uget(w, p + 24) != 0:
        return terror(w, 1)
    if so_storage_live(w, so_uget(w, p + 16)) == 0:
        return terror(w, 1)
    return p

def save_public(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    layout = alloc(w, 128)
    if layout == 0:
        return 0
    hid = getw(w, 80)
    putw(w, layout, getw(w, p))
    putw(w, layout + 8, getw(w, p + 8))
    j = 0
    while j < 4:
        putw(w, layout + 32 + j * 8, getw(w, p + 96 + j * 8))
        putw(w, layout + 64 + j * 8, getw(w, p + 128 + j * 8))
        j = j + 1
    return saved_create(w, t, hid)

def saved_public_check(w: mutbuf, id: u64) -> u64:
    if public_saved(w, id) == 0:
        return 0
    return graph_saved_check(w, id)

def saved_public_release(w: mutbuf, id: u64) -> u64:
    if public_saved(w, id) == 0:
        return 0
    return saved_release(w, id)

def weak_create(w: mutbuf, t: u64) -> u64:
    sid = getw(w, dptr(w, t) + 216)
    p = so_storage_live(w, sid)
    if p == 0:
        return terror(w, 1)
    if so_uget(w, p + 48) >= 4294967295:
        return terror(w, 8)
    if storage_capacity(w, 4, 0) == 0:
        return 0
    id = so_slot_new(w, 4)
    so_uput(w, so_slot_lookup(w, 4, id) + 16, sid)
    so_uput(w, p + 48, so_uget(w, p + 48) + 1)
    return id

def weak_lock(w: mutbuf, id: u64) -> u64:
    wp = so_slot_lookup(w, 4, id)
    if wp == 0:
        return terror(w, 1)
    sid = so_uget(w, wp + 16)
    if so_storage_live(w, sid) == 0:
        return terror(w, 6)
    if storage_capacity(w, 6, 0) == 0:
        return 0
    status = so_storage_retain(w, sid)
    if status != 0:
        return terror(w, status)
    hid = so_slot_new(w, 6)
    so_uput(w, so_slot_lookup(w, 6, hid) + 16, sid)
    return hid

def weak_release(w: mutbuf, id: u64) -> u64:
    p = so_slot_lookup(w, 4, id)
    if p == 0:
        return terror(w, 1)
    if so_weak_storage_release(w, so_uget(w, p + 16)) != 0:
        return terror(w, 9)
    if so_slot_drop(w, 4, id) != 0:
        return terror(w, 9)
    return 1

def public_hold_release(w: mutbuf, id: u64) -> u64:
    if public_hold(w, id) == 0:
        return 0
    return hold_release(w, id)

def saved_logical(w: mutbuf, id: u64, index: u64) -> u64:
    p = public_saved(w, id)
    if p == 0:
        return 0
    if graph_saved_check(w, id) == 0:
        return 0
    layout = private_hold(w, so_uget(w, p + 56), 128)
    if index >= getw(w, layout + 8):
        return terror(w, 2)
    axis = getw(w, layout)
    off = 0
    while axis > 0:
        axis = axis - 1
        dim = getw(w, layout + 32 + axis * 8)
        co = index % dim
        index = index // dim
        off = off + co * getw(w, layout + 64 + axis * 8)
    sp = so_storage_live(w, so_uget(w, p + 16))
    return get4(w, 9223372036854775808 + so_uget(w, sp + 16) + so_uget(w, p + 40) + off * 4)

def hold_logical(w: mutbuf, id: u64, index: u64) -> u64:
    hp = public_hold(w, id)
    if hp == 0:
        return 0
    sp = so_storage_live(w, so_uget(w, hp + 16))
    if index >= so_uget(w, sp + 24) // 4:
        return terror(w, 2)
    return get4(w, 9223372036854775808 + so_uget(w, sp + 16) + index * 4)

def data_alias(w: mutbuf, t: u64) -> u64:
    if storage_capacity(w, 3, 0) == 0:
        return 0
    result = detach(w, t)
    if result == 0:
        return 0
    p = dptr(w, result)
    old = getw(w, p + 224)
    vid = so_slot_new(w, 3)
    if vid == 0:
        return terror(w, 9)
    so_uput(w, so_slot_lookup(w, 3, vid) + 24, 1)
    so_version_release(w, old)
    tp = so_slot_lookup(w, 1, getw(w, p + 192))
    so_uput(w, tp + 24, vid)
    putw(w, p + 224, vid)
    return result

def unpack_saved(w: mutbuf, id: u64) -> u64:
    p = public_saved(w, id)
    if p == 0:
        return 0
    if graph_saved_check(w, id) == 0:
        return 0
    layout = private_hold(w, so_uget(w, p + 56), 128)
    t = descriptor(w, getw(w, layout), layout + 32, 0, 0)
    if t == 0:
        return 0
    tp = dptr(w, t)
    sid = so_uget(w, p + 16)
    vid = so_uget(w, p + 24)
    if so_storage_retain(w, sid) != 0 or so_version_retain(w, vid) != 0:
        return terror(w, 9)
    record = so_slot_lookup(w, 1, getw(w, tp + 192))
    so_uput(w, record + 16, sid)
    so_uput(w, record + 24, vid)
    so_uput(w, record + 32, so_uget(w, p + 40))
    so_uput(w, record + 40, so_uget(w, p + 48))
    putw(w, tp + 216, sid)
    putw(w, tp + 224, vid)
    putw(w, tp + 232, so_uget(w, p + 40))
    sp = so_storage_live(w, sid)
    putw(w, tp + 16, 9223372036854775808 + so_uget(w, sp + 16) + so_uget(w, p + 40))
    j = 0
    while j < getw(w, layout):
        putw(w, tp + 128 + j * 8, getw(w, layout + 64 + j * 8))
        j = j + 1
    return t

def object_stats(w: mutbuf, o: mutbuf, at: u64) -> u64:
    putw(o, at, so_uget(w, 16))
    putw(o, at + 8, so_uget(w, 24))
    putw(o, at + 16, so_uget(w, 40))
    putw(o, at + 24, length(w))
    free = 128 + so_uget(w, 16) * 384
    available = 0
    j = 0
    while j < so_uget(w, 32):
        available = available + so_uget(w, free + j * 16 + 8)
        j = j + 1
    putw(o, at + 32, available)
    kind = 1
    while kind <= 6:
        live = 0
        j = 1
        while j <= so_uget(w, 16):
            if so_uget(w, so_slot_at(w, kind, j) + 8) == 1:
                live = live + 1
            j = j + 1
        putw(o, at + 32 + kind * 8, live)
        kind = kind + 1
    putw(o, at + 88, getw(w, 8) - 1)
    return 96

def command_argument(w: mutbuf, code: u64, field: u64, value: u64) -> u64:
    required = 0
    optional = 0
    if field == 1:
        required = 1654574268384
    if field == 2:
        required = 4713447484
        optional = 603979776
    if field == 3:
        required = 268435460
        optional = 67141632
    if code < 64:
        flag = 1 << code
        if required & flag != 0:
            return public_node(w, value)
        if optional & flag != 0:
            return public_optional(w, value)
    return value

def optimizer_command(s: readbuf, w: mutbuf, cp: u64, end: u64) -> u64:
    code = getr(s, cp)
    a = public_node(w, getr(s, cp + 8))
    b = public_node(w, getr(s, cp + 16))
    c = getr(s, cp + 24)
    if code == 24:
        c = public_node(w, c)
    step = getr(s, cp + 32)
    source = getr(s, cp + 40)
    amount = 40
    if code == 24:
        amount = 48
    if source < end or source > length(s):
        return terror(w, 100)
    if amount > length(s) - source or source % 8 != 0:
        return terror(w, 100)
    i = 0
    while i < amount:
        putw(w, 2080 + i, getr(s, source + i))
        i = i + 8
    if code == 24:
        return adam(w, a, b, c, step, 2080)
    return sgd_momentum(w, a, b, step, 2080)

def inspect_command(s: readbuf, o: mutbuf, w: mutbuf, cp: u64, at: u64) -> u64:
    code = getr(s, cp)
    id = resolve_handle(w, getr(s, cp + 8))
    a = public_node(w, id)
    result = 0
    payload = 0
    if a != 0:
        p = dptr(w, a)
        if code == 13:
            if length(o) - at - 32 < 128:
                terror(w, 3)
            else:
                putw(o, at + 32, getw(w, p))
                putw(o, at + 40, getw(w, p + 8))
                putw(o, at + 48, getw(w, p + 32))
                j = 0
                while j < 4:
                    putw(o, at + 56 + j * 8, getw(w, p + 96 + j * 8))
                    putw(o, at + 88 + j * 8, getw(w, p + 128 + j * 8))
                    j = j + 1
                putw(o, at + 120, getw(w, p + 216))
                putw(o, at + 128, getw(w, p + 224))
                putw(o, at + 136, tversion(w, a))
                putw(o, at + 144, getw(w, p + 232))
                sp = so_storage_live(w, getw(w, p + 216))
                putw(o, at + 152, so_uget(w, sp + 24))
                result = id
                payload = 128
        else:
            n = getw(w, p + 8)
            first = 0
            count = n
            if code == 39 or code == 40:
                first = getr(s, cp + 16)
                count = getr(s, cp + 24)
            if code == 11 or code == 40:
                if getw(w, p + 40) != 0 or getw(w, p + 32) == 0:
                    terror(w, 5)
            if first > n or count > n - first:
                terror(w, 2)
            if count > (length(o) - at - 32) // 4:
                terror(w, 3)
            if getw(w, 0) == 0:
                j = 0
                while j < count:
                    value = 0
                    if code == 10 or code == 39:
                        value = tget(w, a, first + j)
                    else:
                        value = tgrad(w, a, first + j)
                    put4(o, at + 32 + j * 4, value)
                    j = j + 1
                result = count
                payload = count * 4
    status = getw(w, 0)
    if status != 0:
        result = 0
        payload = 0
    putw(o, at, code)
    putw(o, at + 8, status)
    putw(o, at + 16, result)
    putw(o, at + 24, payload)
    return at + 32 + payload

def object_command(s: readbuf, o: mutbuf, w: mutbuf, cp: u64, at: u64) -> u64:
    code = getr(s, cp)
    a = resolve_handle(w, getr(s, cp + 8))
    b = getr(s, cp + 16)
    c = getr(s, cp + 24)
    result = 0
    payload = 0
    first = getw(w, 8)
    if code < 41 or code > 56:
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

def entry(s: readbuf, o: mutbuf, w: mutbuf) -> u32:
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
    while i < count:
        if at > length(o) - 32:
            putw(o, 24, 3)
            return at
        putw(w, 104, i)
        next = command(s, o, w, 32 + i * 64, at, end)
        error = getw(w, 0)
        if error != 0:
            putw(o, 24, error)
            return next
        putw(w, getw(w, 96) + i * 8, getw(o, at + 16))
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

def fold_view(w: mutbuf, t: u64, squeeze: u64, transpose: u64) -> u64:
    p = dptr(w, t)
    if getw(w, p + 8) == 0:
        return 1
    rank = getw(w, p) - squeeze
    axis = rank - 1
    expected = 0
    while axis > 0:
        axis = axis - 1
        source = axis + squeeze
        if transpose != 0 and axis == rank - 2:
            source = source + 1
        dim = getw(w, p + 96 + source * 8)
        stride = getw(w, p + 128 + source * 8)
        if dim > 1:
            if expected != 0 and stride != expected:
                return 0
            expected = stride * dim
    return 1

def matmul_copy_side(w: mutbuf, a: u64, b: u64, side: u64) -> u64:
    pa = dptr(w, a)
    pb = dptr(w, b)
    ra = getw(w, pa)
    rb = getw(w, pb)
    if getw(w, pa + 8) == 0 or getw(w, pb + 8) == 0:
        return 0
    sa = 0
    sb = 0
    if ra == 3 and rb == 3 and (getw(w, pa + 96) != getw(w, pb + 96)):
        if getw(w, pa + 96) == 1 and getw(w, pa + 32) != 0:
            sa = 1
            ra = 2
        elif getw(w, pb + 96) == 1 and getw(w, pb + 32) != 0:
            sb = 1
            rb = 2
    if ra <= 2 and rb <= 2:
        return 0
    large = a
    small = b
    squeeze = sa
    transpose = 0
    large_rank = ra
    small_rank = rb
    large_side = 0
    if rb > ra:
        large = b
        small = a
        squeeze = sb
        transpose = 1
        large_rank = rb
        small_rank = ra
        large_side = 1
    if large_rank >= 3 and small_rank <= 2:
        view = fold_view(w, large, squeeze, transpose)
        if getw(w, dptr(w, small) + 32) != 0:
            if side == large_side and view == 0:
                return 1
            return 0
        if ra != 2 and view != 0:
            return 0
    if ra < 4 and rb < 4:
        return 0
    target0 = 1
    target1 = 1
    if ra == 4:
        target0 = getw(w, pa + 96)
        target1 = getw(w, pa + 104)
    elif ra == 3:
        target1 = getw(w, pa + 96)
    if rb == 4:
        dim = getw(w, pb + 96)
        if dim > target0:
            target0 = dim
        dim = getw(w, pb + 104)
        if dim > target1:
            target1 = dim
    elif rb == 3:
        dim = getw(w, pb + 96)
        if dim > target1:
            target1 = dim
    if target0 <= 1 or target1 <= 1:
        return 0
    source = pa
    rank = ra
    if side != 0:
        source = pb
        rank = rb
    stride0 = 0
    stride1 = 0
    if rank == 4:
        if getw(w, source + 96) == target0:
            stride0 = getw(w, source + 128)
        if getw(w, source + 104) == target1:
            stride1 = getw(w, source + 136)
    if rank == 3:
        if getw(w, source + 96) == target1:
            stride1 = getw(w, source + 128)
    return choose(stride0 == stride1 * target1, 0, 1)

def saved_copy(w: mutbuf, t: u64) -> u64:
    if storage_capacity(w, 5, 0) == 0 or storage_capacity(w, 3, 0) == 0:
        return 0
    n = getw(w, dptr(w, t) + 8)
    data = alloc(w, n * 4)
    if data == 0:
        return 0
    hid = getw(w, 80)
    hp = so_slot_lookup(w, 6, hid)
    sid = so_uget(w, hp + 16)
    i = 0
    while i < n:
        put4(w, data + i * 4, tget(w, t, i))
        i = i + 1
    vid = so_slot_new(w, 3)
    id = so_slot_new(w, 5)
    if vid == 0 or id == 0:
        return terror(w, 9)
    so_uput(w, so_slot_lookup(w, 3, vid) + 24, 1)
    sp = so_slot_lookup(w, 5, id)
    so_uput(w, sp + 16, sid)
    so_uput(w, sp + 24, vid)
    so_uput(w, sp + 48, n * 4)
    so_slot_drop(w, 6, hid)
    return id

def matmul_saved_get(w: mutbuf, t: u64, side: u64, index: u64) -> u64:
    p = dptr(w, t)
    if getw(w, p + 344 + side * 8) != 0:
        saved = getw(w, p + 280 + side * 8)
        sp = so_saved_live(w, saved)
        storage = so_storage_live(w, so_uget(w, sp + 16))
        return get4(w, 9223372036854775808 + so_uget(w, storage + 16) + index * 4)
    return tget(w, getw(w, p + 48 + side * 8), index)

def so_rget(s: readbuf, at: u64) -> u64:
    value = 0
    i = 0
    while i < 8:
        value = value | byte(s, 139264 + at + i) << i * 8
        i = i + 1
    return value

def so_uget(w: mutbuf, at: u64) -> u64:
    value = 0
    i = 0
    while i < 8:
        value = value | byte(w, 139264 + at + i) << i * 8
        i = i + 1
    return value

def so_uput(w: mutbuf, at: u64, value: u64) -> u64:
    i = 0
    while i < 8:
        store(w, 139264 + at + i, value >> i * 8)
        i = i + 1
    return 0

def so_align16(value: u64) -> u64:
    return value + 15 & 18446744073709551600

def so_data_base(slots: u64) -> u64:
    return slots * 424 + 144

def so_state_size(slots: u64, pool: u64) -> u64:
    if slots < 4 or slots > 65536 or pool < 16 or (pool > 1073741824):
        return 0
    if slots & slots - 1 != 0 or pool & 15 != 0:
        return 0
    return so_data_base(slots) + pool

def so_state_init(w: mutbuf, slots: u64, pool: u64) -> u64:
    so_uput(w, 0, 3544425623597306952)
    so_uput(w, 8, 1)
    so_uput(w, 16, slots)
    so_uput(w, 24, pool)
    so_uput(w, 32, 1)
    so_uput(w, 48, 1)
    free = 128 + slots * 384
    so_uput(w, free, 0)
    so_uput(w, free + 8, pool)
    return 0

def so_state_header(w: mutbuf) -> u64:
    if length(w) < 144:
        return 9
    if so_uget(w, 0) != 3544425623597306952 or so_uget(w, 8) != 1 or so_uget(w, 48) != 1:
        return 9
    slots = so_uget(w, 16)
    pool = so_uget(w, 24)
    size = so_state_size(slots, pool)
    if size == 0 or size + 139264 != length(w) or so_uget(w, 32) > slots + 1:
        return 9
    i = 56
    while i < 128:
        if so_uget(w, i) != 0:
            return 9
        i = i + 8
    return 0

def so_handle(kind: u64, slot: u64, generation: u64) -> u64:
    return generation << 32 | kind << 24 | slot

def so_slot_at(w: mutbuf, kind: u64, slot: u64) -> u64:
    return 128 + ((kind - 1) * so_uget(w, 16) + slot - 1) * 64

def so_slot_lookup(w: mutbuf, kind: u64, id: u64) -> u64:
    if kind == 0 or kind > 6 or id >> 24 & 255 != kind:
        return 0
    slot = id & 16777215
    if slot == 0 or slot > so_uget(w, 16):
        return 0
    p = so_slot_at(w, kind, slot)
    generation = id >> 32
    if generation == 0 or so_uget(w, p) != generation or so_uget(w, p + 8) != 1:
        return 0
    return p

def so_slot_available(w: mutbuf, kind: u64) -> u64:
    i = 1
    while i <= so_uget(w, 16):
        p = so_slot_at(w, kind, i)
        if so_uget(w, p + 8) == 0 and so_uget(w, p) < 4294967295:
            return 1
        i = i + 1
    return 0

def so_slot_new(w: mutbuf, kind: u64) -> u64:
    i = 1
    while i <= so_uget(w, 16):
        p = so_slot_at(w, kind, i)
        if so_uget(w, p + 8) == 0 and so_uget(w, p) < 4294967295:
            generation = so_uget(w, p) + 1
            so_fill_range(w, p, 64, 0)
            so_uput(w, p, generation)
            so_uput(w, p + 8, 1)
            return so_handle(kind, i, generation)
        i = i + 1
    return 0

def so_slot_drop(w: mutbuf, kind: u64, id: u64) -> u64:
    p = so_slot_lookup(w, kind, id)
    if p == 0:
        return 1
    so_fill_range(w, p + 8, 56, 0)
    return 0

def so_extent_room(w: mutbuf, size: u64) -> u64:
    if size == 0:
        return 1
    size = so_align16(size)
    free = 128 + so_uget(w, 16) * 384
    i = 0
    while i < so_uget(w, 32):
        if so_uget(w, free + i * 16 + 8) >= size:
            return 1
        i = i + 1
    return 0

def so_extent_remove(w: mutbuf, index: u64) -> u64:
    free = 128 + so_uget(w, 16) * 384
    count = so_uget(w, 32)
    i = index
    while i + 1 < count:
        so_uput(w, free + i * 16, so_uget(w, free + (i + 1) * 16))
        so_uput(w, free + i * 16 + 8, so_uget(w, free + (i + 1) * 16 + 8))
        i = i + 1
    so_fill_range(w, free + (count - 1) * 16, 16, 0)
    so_uput(w, 32, count - 1)
    return 0

def so_extent_allocate(w: mutbuf, size: u64) -> u64:
    if size == 0:
        return 1
    size = so_align16(size)
    free = 128 + so_uget(w, 16) * 384
    i = 0
    while i < so_uget(w, 32):
        at = free + i * 16
        available = so_uget(w, at + 8)
        if available >= size:
            offset = so_uget(w, at)
            if available == size:
                so_extent_remove(w, i)
            else:
                so_uput(w, at, offset + size)
                so_uput(w, at + 8, available - size)
            return offset + 1
        i = i + 1
    return 0

def so_extent_release(w: mutbuf, offset: u64, size: u64) -> u64:
    if size == 0:
        return 0
    pool = so_uget(w, 24)
    if offset > pool or size > pool - offset or offset & 15 != 0 or (size & 15 != 0):
        return 9
    free = 128 + so_uget(w, 16) * 384
    count = so_uget(w, 32)
    i = 0
    while i < count:
        start = so_uget(w, free + i * 16)
        length = so_uget(w, free + i * 16 + 8)
        if offset < start + length and start < offset + size:
            return 9
        i = i + 1
    i = 0
    while i < count:
        start = so_uget(w, free + i * 16)
        length = so_uget(w, free + i * 16 + 8)
        if start + length == offset:
            length = length + size
            if i + 1 < count and offset + size == so_uget(w, free + (i + 1) * 16):
                length = length + so_uget(w, free + (i + 1) * 16 + 8)
                so_extent_remove(w, i + 1)
            so_uput(w, free + i * 16 + 8, length)
            return 0
        if offset + size == start:
            so_uput(w, free + i * 16, offset)
            so_uput(w, free + i * 16 + 8, length + size)
            return 0
        i = i + 1
    if count >= so_uget(w, 16) + 1:
        return 9
    i = count
    moving = 1
    while i > 0 and moving == 1:
        start = so_uget(w, free + (i - 1) * 16)
        if start < offset:
            moving = 0
        else:
            so_uput(w, free + i * 16, start)
            so_uput(w, free + i * 16 + 8, so_uget(w, free + (i - 1) * 16 + 8))
            i = i - 1
    so_uput(w, free + i * 16, offset)
    so_uput(w, free + i * 16 + 8, size)
    so_uput(w, 32, count + 1)
    i = 0
    while i + 1 < so_uget(w, 32):
        start = so_uget(w, free + i * 16)
        length = so_uget(w, free + i * 16 + 8)
        next_start = so_uget(w, free + (i + 1) * 16)
        if start + length == next_start:
            so_uput(w, free + i * 16 + 8, length + so_uget(w, free + (i + 1) * 16 + 8))
            so_extent_remove(w, i + 1)
        else:
            i = i + 1
    return 0

def so_state_migrate(source: readbuf, target: mutbuf, slots: u64, pool: u64) -> u64:
    old_slots = so_rget(source, 16)
    old_pool = so_rget(source, 24)
    so_state_init(target, slots, pool)
    so_fill_range(target, 128 + slots * 384, (slots + 1) * 16, 0)
    so_uput(target, 32, so_rget(source, 32))
    so_uput(target, 40, so_rget(source, 40) + 1)
    kind = 0
    while kind < 6:
        so_copy_range(target, source, 128 + kind * slots * 64, 128 + kind * old_slots * 64, old_slots * 64)
        kind = kind + 1
    so_copy_range(target, source, 128 + slots * 384, 128 + old_slots * 384, so_rget(source, 32) * 16)
    so_copy_range(target, source, so_data_base(slots), so_data_base(old_slots), old_pool)
    if pool > old_pool:
        return so_extent_release(target, old_pool, pool - old_pool)
    return 0

def so_storage_live(w: mutbuf, id: u64) -> u64:
    p = so_slot_lookup(w, 2, id)
    if p == 0:
        return 0
    if so_uget(w, p + 40) == 0 or so_uget(w, p + 48) == 0 or so_uget(w, p + 56) != 0:
        return 0
    return p

def so_version_live(w: mutbuf, id: u64) -> u64:
    p = so_slot_lookup(w, 3, id)
    if p == 0:
        return 0
    if so_uget(w, p + 24) == 0:
        return 0
    return p

def so_tensor_live(w: mutbuf, id: u64) -> u64:
    p = so_slot_lookup(w, 1, id)
    if p == 0:
        return 0
    storage = so_storage_live(w, so_uget(w, p + 16))
    if storage == 0 or so_version_live(w, so_uget(w, p + 24)) == 0:
        return 0
    offset = so_uget(w, p + 32)
    size = so_uget(w, p + 40)
    if offset > so_uget(w, storage + 24) or size > so_uget(w, storage + 24) - offset:
        return 0
    return p

def so_saved_live(w: mutbuf, id: u64) -> u64:
    p = so_slot_lookup(w, 5, id)
    if p == 0:
        return 0
    storage = so_storage_live(w, so_uget(w, p + 16))
    if storage == 0 or so_version_live(w, so_uget(w, p + 24)) == 0:
        return 0
    offset = so_uget(w, p + 40)
    size = so_uget(w, p + 48)
    if offset > so_uget(w, storage + 24) or size > so_uget(w, storage + 24) - offset:
        return 0
    return p

def so_storage_retain(w: mutbuf, id: u64) -> u64:
    p = so_storage_live(w, id)
    if p == 0:
        return 1
    value = so_uget(w, p + 40)
    if value >= 4294967295:
        return 8
    so_uput(w, p + 40, value + 1)
    return 0

def so_version_retain(w: mutbuf, id: u64) -> u64:
    p = so_version_live(w, id)
    if p == 0:
        return 1
    value = so_uget(w, p + 24)
    if value >= 4294967295:
        return 8
    so_uput(w, p + 24, value + 1)
    return 0

def so_version_release(w: mutbuf, id: u64) -> u64:
    p = so_version_live(w, id)
    if p == 0:
        return 1
    value = so_uget(w, p + 24)
    if value == 1:
        return so_slot_drop(w, 3, id)
    so_uput(w, p + 24, value - 1)
    return 0

def so_storage_release(w: mutbuf, id: u64) -> u64:
    p = so_storage_live(w, id)
    if p == 0:
        return 1
    value = so_uget(w, p + 40)
    if value > 1:
        so_uput(w, p + 40, value - 1)
        return 0
    status = so_extent_release(w, so_uget(w, p + 16), so_uget(w, p + 32))
    if status != 0:
        return status
    weak = so_uget(w, p + 48)
    if weak == 1:
        return so_slot_drop(w, 2, id)
    so_uput(w, p + 16, 0)
    so_uput(w, p + 32, 0)
    so_uput(w, p + 40, 0)
    so_uput(w, p + 48, weak - 1)
    so_uput(w, p + 56, 1)
    return 0

def so_weak_storage_release(w: mutbuf, id: u64) -> u64:
    p = so_slot_lookup(w, 2, id)
    if p == 0 or so_uget(w, p + 48) == 0:
        return 1
    weak = so_uget(w, p + 48)
    strong = so_uget(w, p + 40)
    if weak == 1:
        if strong != 0:
            return 9
        return so_slot_drop(w, 2, id)
    so_uput(w, p + 48, weak - 1)
    return 0

def so_saved_check(w: mutbuf, id: u64) -> u64:
    p = so_saved_live(w, id)
    if p == 0:
        return 1
    version = so_version_live(w, so_uget(w, p + 24))
    if so_uget(w, p + 32) != so_uget(w, version + 16):
        return 5
    return 0

def so_audit_count(w: mutbuf, kind: u64, id: u64, field: u64) -> u64:
    if so_slot_lookup(w, kind, id) == 0:
        return 9
    slot = id & 16777215
    p = 144 + so_uget(w, 16) * 400 + (slot - 1) * 24 + field
    value = so_uget(w, p)
    if value >= 4294967295:
        return 9
    so_uput(w, p, value + 1)
    return 0

def so_audit_records(w: mutbuf) -> u64:
    kind = 1
    while kind <= 6:
        i = 1
        while i <= so_uget(w, 16):
            p = so_slot_at(w, kind, i)
            generation = so_uget(w, p)
            live = so_uget(w, p + 8)
            if generation > 4294967295 or live > 1:
                return 9
            if live == 0:
                j = 16
                while j < 64:
                    if so_uget(w, p + j) != 0:
                        return 9
                    j = j + 8
            else:
                if generation == 0:
                    return 9
                id = so_handle(kind, i, generation)
                if kind == 1 or kind == 5:
                    if kind == 1 and so_tensor_live(w, id) == 0 or (kind == 5 and so_saved_live(w, id) == 0):
                        return 9
                    if so_audit_count(w, 2, so_uget(w, p + 16), 0) != 0 or so_audit_count(w, 3, so_uget(w, p + 24), 16) != 0:
                        return 9
                    if kind == 1 and (so_uget(w, p + 56) != 0 or so_uget(w, p + 48) == 0 or so_uget(w, p + 48) >= getw(w, 8)):
                        return 9
                    if kind == 5 and so_uget(w, p + 56) != 0 and (private_hold(w, so_uget(w, p + 56), 128) == 0):
                        return 9
                if kind == 4 or kind == 6:
                    field = 0
                    if kind == 4:
                        field = 8
                    if so_audit_count(w, 2, so_uget(w, p + 16), field) != 0:
                        return 9
                    j = 24
                    if kind == 6:
                        if so_uget(w, p + 24) > 1:
                            return 9
                        j = 32
                    while j < 64:
                        if so_uget(w, p + j) != 0:
                            return 9
                        j = j + 8
                if kind == 3:
                    j = 32
                    while j < 64:
                        if so_uget(w, p + j) != 0:
                            return 9
                        j = j + 8
            i = i + 1
        kind = kind + 1
    return 0

def so_audit_references(w: mutbuf) -> u64:
    i = 1
    while i <= so_uget(w, 16):
        counters = 144 + so_uget(w, 16) * 400 + (i - 1) * 24
        storage = so_slot_at(w, 2, i)
        version = so_slot_at(w, 3, i)
        if so_uget(w, storage + 8) == 0:
            if so_uget(w, counters) != 0 or so_uget(w, counters + 8) != 0:
                return 9
        else:
            strong = so_uget(w, counters)
            weak = so_uget(w, counters + 8)
            if strong > 0:
                weak = weak + 1
            if weak == 0 or so_uget(w, storage + 40) != strong or so_uget(w, storage + 48) != weak:
                return 9
            if so_uget(w, storage + 24) > 1073741824:
                return 9
            if strong == 0:
                if so_uget(w, storage + 16) != 0 or so_uget(w, storage + 32) != 0 or so_uget(w, storage + 56) != 1:
                    return 9
            else:
                if so_uget(w, storage + 56) != 0 or so_uget(w, storage + 32) != so_align16(so_uget(w, storage + 24)):
                    return 9
                offset = so_uget(w, storage + 16)
                size = so_uget(w, storage + 32)
                if offset > so_uget(w, 24) or size > so_uget(w, 24) - offset or offset & 15 != 0:
                    return 9
                if size == 0 and offset != 0:
                    return 9
        refs = so_uget(w, counters + 16)
        if so_uget(w, version + 8) == 0:
            if refs != 0:
                return 9
        elif refs == 0 or so_uget(w, version + 24) != refs:
            return 9
        i = i + 1
    return 0

def so_audit_free(w: mutbuf) -> u64:
    free = 128 + so_uget(w, 16) * 384
    i = 0
    previous = 0
    while i < so_uget(w, 32):
        start = so_uget(w, free + i * 16)
        size = so_uget(w, free + i * 16 + 8)
        if size == 0 or start > so_uget(w, 24) or size > so_uget(w, 24) - start:
            return 9
        if start & 15 != 0 or size & 15 != 0:
            return 9
        if i > 0 and start <= previous:
            return 9
        previous = start + size
        i = i + 1
    while i < so_uget(w, 16) + 1:
        if so_uget(w, free + i * 16) != 0 or so_uget(w, free + i * 16 + 8) != 0:
            return 9
        i = i + 1
    return 0

def so_audit_extent(w: mutbuf, index: u64) -> u64:
    storage = so_slot_at(w, 2, index)
    if so_uget(w, storage + 8) == 0 or so_uget(w, storage + 40) == 0 or so_uget(w, storage + 32) == 0:
        return 0
    start = so_uget(w, storage + 16)
    size = so_uget(w, storage + 32)
    j = index + 1
    while j <= so_uget(w, 16):
        other = so_slot_at(w, 2, j)
        if so_uget(w, other + 8) == 1 and so_uget(w, other + 40) > 0 and (so_uget(w, other + 32) > 0):
            offset = so_uget(w, other + 16)
            if start < offset + so_uget(w, other + 32) and offset < start + size:
                return 9
        j = j + 1
    free = 128 + so_uget(w, 16) * 384
    j = 0
    while j < so_uget(w, 32):
        offset = so_uget(w, free + j * 16)
        if start < offset + so_uget(w, free + j * 16 + 8) and offset < start + size:
            return 9
        j = j + 1
    return 0

def so_audit_extents(w: mutbuf) -> u64:
    if so_audit_free(w) != 0:
        return 9
    total = 0
    i = 1
    while i <= so_uget(w, 16):
        if so_audit_extent(w, i) != 0:
            return 9
        p = so_slot_at(w, 2, i)
        if so_uget(w, p + 8) == 1:
            total = total + so_uget(w, p + 32)
        i = i + 1
    free = 128 + so_uget(w, 16) * 384
    i = 0
    while i < so_uget(w, 32):
        total = total + so_uget(w, free + i * 16 + 8)
        i = i + 1
    if total != so_uget(w, 24):
        return 9
    return 0

def so_audit(w: mutbuf) -> u64:
    if so_state_header(w) != 0:
        return 9
    slots = so_uget(w, 16)
    if slots > 4096:
        return 3
    scratch = 144 + slots * 400
    so_fill_range(w, scratch, slots * 24, 0)
    status = so_audit_records(w)
    if status == 0:
        status = so_audit_references(w)
    if status == 0:
        status = so_audit_extents(w)
    so_fill_range(w, scratch, slots * 24, 0)
    return status

def so_fill_range(w: mutbuf, at: u64, size: u64, value: u64) -> u64:
    fill_range(w, 139264 + at, size, value)
    return 0

def so_copy_range(w: mutbuf, s: readbuf, at: u64, source: u64, size: u64) -> u64:
    copy_range(w, s, 139264 + at, 139264 + source, size)
    return 0

def back_unary(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    a = getw(w, p + 48)
    op = getw(w, p + 40)
    n = getw(w, p + 8)
    i = 0
    while i < n:
        g = get4(w, getw(w, p + 160) + i * 4)
        if op == 5:
            x = tget(w, t, i)
            if ltf(x, 0) != 0 or absf(x) == 0:
                g = 0
        g = choose(op == 6, negf(g), g)
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

def conv2d(w: mutbuf, x: u64, weight: u64, stride_pack: u64, pad_pack: u64, dilation_pack: u64, groups: u64) -> u64:
    config_hold = 0
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
    config_hold = getw(w, 80)
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
    putw(w, dptr(w, t) + 264, config_hold)
    record(w, t, 22, x, weight)
    i = 0
    while i < n:
        put4(w, getw(w, p + 16) + i * 4, conv_point(w, t, i, 0))
        i = i + 1
    return t

def pool2d(w: mutbuf, op: u64, x: u64, kernel_pack: u64, stride_pack: u64, pad_pack: u64, include_pad: u64) -> u64:
    config_hold = 0
    indices_hold = 0
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
    config_hold = getw(w, 80)
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
    if n > idiv(16777216, getw(w, config + 104)):
        return terror(w, 3)
    if op == 24 and getw(w, p + 32) != 0:
        indices = alloc(w, n * 8)
        indices_hold = getw(w, 80)
        if indices == 0:
            return 0
        putw(w, p + 184, indices)
    putw(w, dptr(w, t) + 264, config_hold)
    putw(w, dptr(w, t) + 272, indices_hold)
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
    indices_hold = 0
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
    indices_hold = getw(w, 80)
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
    putw(w, dptr(w, t) + 264, indices_hold)
    record(w, t, 26, a, 0)
    j = 0
    while j < getw(w, p + 8):
        source = index_select_source(w, t, j)
        put4(w, getw(w, p + 16) + j * 4, tget(w, a, source))
        j = j + 1
    return t

def gather(s: readbuf, w: mutbuf, a: u64, axis: u64, config: u64, end: u64) -> u64:
    indices_hold = 0
    if valid(w, a) == 0:
        return 0
    if config < end or config > length(s) or config % 8 != 0:
        return terror(w, 100)
    if length(s) - config < 8:
        return terror(w, 100)
    rank = getr(s, config)
    ap = dptr(w, a)
    if rank > 4 or rank != getw(w, ap):
        return terror(w, 2)
    arank = rank
    if arank == 0:
        arank = 1
    if axis >= 9223372036854775808:
        axis = axis + arank
    if axis >= arank:
        return terror(w, 2)
    if length(s) - config < 8 + rank * 8:
        return terror(w, 100)
    count = 1
    j = 0
    while j < rank:
        dim = getr(s, config + 8 + j * 8)
        if dim > 262144:
            return terror(w, 3)
        if dim == 0:
            count = 0
        putw(w, 2048 + j * 8, dim)
        j = j + 1
    if count != 0:
        j = 0
        while j < rank:
            count = count * getw(w, 2048 + j * 8)
            if count > 16777216:
                return terror(w, 3)
            j = j + 1
    offset = config + 8 + rank * 8
    if count > (length(s) - offset) // 8:
        return terror(w, 100)
    if count != 0:
        j = 0
        while j < rank:
            if j != axis and getw(w, 2048 + j * 8) > getw(w, ap + 96 + j * 8):
                return terror(w, 2)
            j = j + 1
    extent = 1
    if rank != 0:
        extent = getw(w, ap + 96 + axis * 8)
    j = 0
    while j < count:
        if getr(s, offset + j * 8) >= extent:
            return terror(w, 7)
        j = j + 1
    indices = alloc(w, count * 8)
    indices_hold = getw(w, 80)
    if indices == 0:
        return 0
    j = 0
    while j < count:
        putw(w, indices + j * 8, getr(s, offset + j * 8))
        j = j + 1
    t = tnew(w, rank, 2048, getw(w, ap + 32))
    if t == 0:
        return 0
    p = dptr(w, t)
    putw(w, p + 176, axis)
    putw(w, p + 184, indices)
    putw(w, dptr(w, t) + 264, indices_hold)
    record(w, t, 27, a, 0)
    j = 0
    while j < count:
        source = gather_source(w, t, j)
        put4(w, getw(w, p + 16) + j * 4, tget(w, a, source))
        j = j + 1
    return t

def tensor_join(s: readbuf, w: mutbuf, op: u64, axis: u64, config: u64, end: u64) -> u64:
    saved_hold = 0
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
    if getw(w, 8) >= 65536:
        return terror(w, 3)
    saved = alloc(w, 8 + count * 16)
    saved_hold = getw(w, 80)
    putw(w, saved, count)
    j = 0
    while j < count:
        a = wire_node(s, w, config + 8 + j * 8)
        putw(w, saved + 8 + j * 16, a)
        putw(w, saved + 16 + j * 16, tversion(w, a))
        j = j + 1
    t = tnew(w, getw(w, 2080), 2048, getw(w, 2096))
    if t == 0:
        return 0
    p = dptr(w, t)
    putw(w, p + 176, getw(w, 2088))
    putw(w, p + 184, saved)
    putw(w, dptr(w, t) + 264, saved_hold)
    record(w, t, op, wire_node(s, w, config + 8), 0)
    concat_layout(w, t)
    concat_copy(w, t, 0)
    return t

def indexed_add(s: readbuf, w: mutbuf, a: u64, b: u64, axis: u64, config: u64, alpha: u64, op: u64, end: u64) -> u64:
    aux_hold = 0
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
    aux_hold = getw(w, 80)
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
    putw(w, dptr(w, t) + 264, aux_hold)
    record(w, t, op, a, b)
    indexed_add_forward(w, t)
    return t

def concat_extent(s: readbuf, w: mutbuf, op: u64, axis: u64, offset: u64, count: u64, reference: u64) -> u64:
    rp = dptr(w, reference)
    rank = getw(w, rp)
    extent = 0
    j = 0
    while j < count:
        p = dptr(w, wire_node(s, w, offset + j * 8))
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
        a = wire_node(s, w, offset + j * 8)
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
            if n > 16777216:
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

def index_select_command(s: readbuf, w: mutbuf, cp: u64, end: u64) -> u64:
    return index_select(s, w, wire_node(s, w, cp + 8), getr(s, cp + 16), getr(s, cp + 24), getr(s, cp + 32), end)

def gather_command(s: readbuf, w: mutbuf, cp: u64, end: u64) -> u64:
    return gather(s, w, wire_node(s, w, cp + 8), getr(s, cp + 16), getr(s, cp + 24), end)

def indexed_add_command(s: readbuf, w: mutbuf, cp: u64, end: u64) -> u64:
    op = 39
    alpha = 1065353216
    if getr(s, cp) == 37:
        op = 40
        alpha = getr(s, cp + 40)
    return indexed_add(s, w, wire_node(s, w, cp + 8), wire_node(s, w, cp + 16), getr(s, cp + 24), getr(s, cp + 32), alpha, op, end)

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
        if getw(w, dptr(w, a) + 32) != 0:
            gadd(w, a, ai, mulf(grad, matmul_saved_get(w, t, 1, bi)))
        if getw(w, dptr(w, b) + 32) != 0:
            gadd(w, b, bi, mulf(grad, matmul_saved_get(w, t, 0, ai)))
        k = k + 1
    return 1

def backward_seed(w: mutbuf, t: u64, seed: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if getw(w, p + 32) == 0:
        return terror(w, 5)
    if seed == 0:
        if getw(w, p + 8) != 1:
            return terror(w, 5)
    else:
        if valid(w, seed) == 0:
            return 0
        q = dptr(w, seed)
        rank = getw(w, p)
        if getw(w, q) != rank:
            return terror(w, 2)
        axis = 0
        while axis < rank:
            if getw(w, p + 96 + axis * 8) != getw(w, q + 96 + axis * 8):
                return terror(w, 2)
            axis = axis + 1
    epoch = getw(w, 24) + 1
    putw(w, 24, epoch)
    if mark_graph(w, t, epoch) == 0:
        return 0
    limit = t
    i = 1
    while i <= limit:
        current = live_id(w, i)
        pi = dptr(w, current)
        if getw(w, pi + 168) == epoch:
            memory_fill(w, getw(w, pi + 160), getw(w, pi + 8) * 4, 0)
        i = i + 1
    if seed == 0:
        put4(w, getw(w, p + 160), 1065353216)
    else:
        j = 0
        n = getw(w, p + 8)
        dst = getw(w, p + 160)
        while j < n:
            put4(w, dst + j * 4, tget(w, seed, j))
            j = j + 1
    i = limit
    while i > 0:
        current = live_id(w, i)
        pi = dptr(w, current)
        if getw(w, pi + 168) == epoch:
            op = getw(w, pi + 40)
            if op >= 1 and op <= 4:
                back_binary(w, current)
            if op == 5 or op == 6 or op == 10 or (op >= 11 and op <= 17):
                back_unary(w, current)
            if op == 7 or op == 8:
                back_reduce(w, current)
            if op == 18 or op == 19:
                back_axis(w, current)
            if op == 20 or op == 21:
                back_softmax(w, current)
            if op >= 22 and op <= 24:
                back_spatial(w, current)
            if op == 25:
                back_matmul(w, current)
            if op == 26:
                back_index_select(w, current)
            if op == 27:
                back_gather(w, current)
            if op == 37 or op == 38:
                back_concat(w, current)
            if op == 39 or op == 40:
                back_indexed_add(w, current)
        i = i - 1
    i = 1
    while i <= limit:
        current = live_id(w, i)
        pi = dptr(w, current)
        if getw(w, pi + 168) == epoch and getw(w, pi + 40) == 0:
            j = 0
            n = getw(w, pi + 8)
            dst = getw(w, pi + 24)
            src = getw(w, pi + 160)
            while j < n:
                put4(w, dst + j * 4, addf(get4(w, dst + j * 4), get4(w, src + j * 4)))
                j = j + 1
        i = i + 1
    return 1

def frame_valid(s: readbuf) -> u64:
    if length(s) < 32 or length(s) > 1048576:
        return 100
    if getr(s, 0) != 3688527548797375560 or getr(s, 16) != length(s) or getr(s, 24) != 0:
        return 100
    if getr(s, 8) > (length(s) - 32) // 64:
        return 100
    return 0

def reply_error(o: mutbuf, status: u64) -> u64:
    putw(o, 0, 3688536413609874504)
    putw(o, 8, 3)
    putw(o, 16, 0)
    putw(o, 24, status)
    return 32

def tensor_initialize(w: mutbuf, t: u64, index: u64, bits: u64) -> u64:
    put4(w, taddr(w, t, index), bits)
    return 1

def full_tensor(w: mutbuf, rank: u64, requires: u64, shape: u64, bits: u64) -> u64:
    if bits > 4294967295:
        return terror(w, 7)
    t = tnew(w, rank, shape, requires)
    if t == 0:
        return 0
    p = dptr(w, t)
    n = getw(w, p + 8)
    data = getw(w, p + 16)
    i = 0
    while i < n:
        put4(w, data + i * 4, bits)
        i = i + 1
    return t

def command(s: readbuf, o: mutbuf, w: mutbuf, cp: u64, at: u64, end: u64) -> u64:
    code = getr(s, cp)
    if code >= 41:
        return object_command(s, o, w, cp, at)
    if code == 10 or code == 11 or code == 13 or (code == 39) or (code == 40):
        return inspect_command(s, o, w, cp, at)
    putw(w, 88, getw(w, 8))
    a = getr(s, cp + 8)
    b = getr(s, cp + 16)
    c = getr(s, cp + 24)
    a = command_argument(w, code, 1, a)
    b = command_argument(w, code, 2, b)
    c = command_argument(w, code, 3, c)
    result = 0
    payload = 0
    recognized = 0
    if code == 1:
        recognized = 1
        putw(w, 2048, c)
        putw(w, 2056, getr(s, cp + 32))
        putw(w, 2064, getr(s, cp + 40))
        putw(w, 2072, getr(s, cp + 48))
        result = tnew(w, a, 2048, b)
        if getw(w, 0) == 0:
            dp = dptr(w, result)
            n = getw(w, dp + 8)
            source = getr(s, cp + 56)
            if source < end or source > length(s):
                terror(w, 100)
            elif n > (length(s) - source) // 4 or source % 4 != 0:
                terror(w, 100)
            else:
                i = 0
                while i < n:
                    tensor_initialize(w, result, i, read4(s, source + 4 * i))
                    i = i + 1
    if code == 2:
        recognized = 1
        result = binary(w, a, b, c)
    if code == 3:
        recognized = 1
        result = unary(w, a, b)
    if code == 4:
        recognized = 1
        result = reduce(w, a, b)
    if code == 5:
        recognized = 1
        result = mm(w, a, b)
    if code == 6:
        recognized = 1
        result = transpose(w, a, b, c)
    if code == 7:
        recognized = 1
        result = backward(w, a)
    if code == 8:
        recognized = 1
        result = zero_grad(w, a)
    if code == 9:
        recognized = 1
        result = sgd(w, a, b)
    if code == 12:
        recognized = 1
        result = tset(w, a, b, c)
    if code == 14:
        recognized = 1
        result = scalar(w, a, b, c)
    if code == 15:
        recognized = 1
        result = linear(w, a, b, c)
    if code == 16:
        recognized = 1
        result = mse(w, a, b)
    if code == 17:
        recognized = 1
        putw(w, 2048, c)
        putw(w, 2056, getr(s, cp + 32))
        putw(w, 2064, getr(s, cp + 40))
        putw(w, 2072, getr(s, cp + 48))
        result = view(w, a, b, 2048)
    if code == 18:
        recognized = 1
        result = clone(w, a)
    if code == 19:
        recognized = 1
        result = detach(w, a)
    if code == 20:
        recognized = 1
        result = reduce_axis(w, a, b, c, getr(s, cp + 32))
    if code == 21:
        recognized = 1
        result = softmax(w, a, b, c)
    if code == 22:
        recognized = 1
        result = conv2d(w, a, b, c, getr(s, cp + 32), getr(s, cp + 40), getr(s, cp + 48))
    if code == 23:
        recognized = 1
        result = pool2d(w, a, b, c, getr(s, cp + 32), getr(s, cp + 40), getr(s, cp + 48))
    if code == 24 or code == 25:
        recognized = 1
        result = optimizer_command(s, w, cp, end)
    if code == 26:
        recognized = 1
        result = layernorm(w, a, b, c, getr(s, cp + 32))
    if code == 27:
        recognized = 1
        result = cross_entropy_probs(w, a, b)
    if code == 28:
        recognized = 1
        result = batchnorm_eval(w, a, b, c, public_optional(w, getr(s, cp + 32)), public_optional(w, getr(s, cp + 40)), getr(s, cp + 48))
    if code == 29:
        recognized = 1
        result = backward_seed(w, a, b)
    if code == 30:
        recognized = 1
        result = index_select_command(s, w, cp, end)
    if code == 31:
        recognized = 1
        result = gather_command(s, w, cp, end)
    if code == 32:
        recognized = 1
        result = matmul(w, a, b)
    if code == 33:
        recognized = 1
        result = graph_reset(w, a)
    if code == 34 or code == 35:
        recognized = 1
        result = concat_command(s, w, cp, end)
    if code == 36 or code == 37:
        recognized = 1
        result = indexed_add_command(s, w, cp, end)
    if code == 38:
        recognized = 1
        putw(w, 2048, c)
        putw(w, 2056, getr(s, cp + 32))
        putw(w, 2064, getr(s, cp + 40))
        putw(w, 2072, getr(s, cp + 48))
        result = full_tensor(w, a, b, 2048, getr(s, cp + 56))
    if recognized == 0:
        terror(w, 4)
    if getw(w, 0) == 0:
        if 540578652286 & 1 << code != 0:
            result = publish_tensor(w, result)
        if getw(w, 8) != getw(w, 88):
            finish_command(w, getw(w, 88))
        putw(w, 88, 0)
    status = getw(w, 0)
    if status != 0:
        payload = 0
        result = 0
    putw(o, at, code)
    putw(o, at + 8, status)
    putw(o, at + 16, result)
    putw(o, at + 24, payload)
    return at + 32 + payload
