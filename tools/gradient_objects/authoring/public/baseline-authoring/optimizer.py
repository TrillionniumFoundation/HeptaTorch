def object_stats(w: mutbuf, o: mutbuf, at: u64) -> u64:
    putw(o, at, so_uget(w, 16))
    putw(o, at + 8, so_uget(w, 24))
    putw(o, at + 16, so_uget(w, 40))
    putw(o, at + 24, length(w))
    free = 128 + so_uget(w, 16) * 512
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

def so_data_base(slots: u64) -> u64:
    return slots * 552 + 144

def so_state_init(w: mutbuf, slots: u64, pool: u64) -> u64:
    so_uput(w, 0, 3616483204683224136)
    so_uput(w, 8, 2)
    so_uput(w, 16, slots)
    so_uput(w, 24, pool)
    so_uput(w, 32, 1)
    so_uput(w, 48, 1)
    free = 128 + slots * 512
    so_uput(w, free, 0)
    so_uput(w, free + 8, pool)
    return 0

def so_state_header(w: mutbuf) -> u64:
    if length(w) < 144:
        return 9
    if so_uget(w, 0) != 3616483204683224136 or so_uget(w, 8) != 2 or so_uget(w, 48) != 1:
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

def so_slot_lookup(w: mutbuf, kind: u64, id: u64) -> u64:
    if kind == 0 or kind > 8 or id >> 24 & 255 != kind:
        return 0
    slot = id & 16777215
    if slot == 0 or slot > so_uget(w, 16):
        return 0
    p = so_slot_at(w, kind, slot)
    generation = id >> 32
    if generation == 0 or so_uget(w, p) != generation or so_uget(w, p + 8) != 1:
        return 0
    return p

def so_extent_room(w: mutbuf, size: u64) -> u64:
    if size == 0:
        return 1
    size = so_align16(size)
    free = 128 + so_uget(w, 16) * 512
    i = 0
    while i < so_uget(w, 32):
        if so_uget(w, free + i * 16 + 8) >= size:
            return 1
        i = i + 1
    return 0

def so_extent_remove(w: mutbuf, index: u64) -> u64:
    free = 128 + so_uget(w, 16) * 512
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
    free = 128 + so_uget(w, 16) * 512
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
    free = 128 + so_uget(w, 16) * 512
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
    so_fill_range(target, 128 + slots * 512, (slots + 1) * 16, 0)
    so_uput(target, 32, so_rget(source, 32))
    so_uput(target, 40, so_rget(source, 40) + 1)
    kind = 0
    while kind < 8:
        so_copy_range(target, source, 128 + kind * slots * 64, 128 + kind * old_slots * 64, old_slots * 64)
        kind = kind + 1
    so_copy_range(target, source, 128 + slots * 512, 128 + old_slots * 512, so_rget(source, 32) * 16)
    so_copy_range(target, source, so_data_base(slots), so_data_base(old_slots), old_pool)
    if pool > old_pool:
        return so_extent_release(target, old_pool, pool - old_pool)
    return 0

def so_audit_count(w: mutbuf, kind: u64, id: u64, field: u64) -> u64:
    if so_slot_lookup(w, kind, id) == 0:
        return 9
    slot = id & 16777215
    p = 144 + so_uget(w, 16) * 528 + (slot - 1) * 24 + field
    value = so_uget(w, p)
    if value >= 4294967295:
        return 9
    so_uput(w, p, value + 1)
    return 0

def so_audit_records(w: mutbuf) -> u64:
    kind = 1
    while kind <= 8:
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
                if kind == 7:
                    hid = so_uget(w, p + 16)
                    address = private_hold(w, hid, 5056)
                    if address == 0 or rng_state_canonical(w, address) == 0:
                        return 9
                    j = 24
                    while j < 64:
                        if so_uget(w, p + j) != 0:
                            return 9
                        j = j + 8
                    j = 1
                    while j < i:
                        other = so_slot_at(w, 7, j)
                        if so_uget(w, other + 8) == 1 and so_uget(w, other + 16) == hid:
                            return 9
                        j = j + 1
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
        counters = 144 + so_uget(w, 16) * 528 + (i - 1) * 24
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
    free = 128 + so_uget(w, 16) * 512
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
    free = 128 + so_uget(w, 16) * 512
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
    free = 128 + so_uget(w, 16) * 512
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
    scratch = 144 + slots * 528
    so_fill_range(w, scratch, slots * 24, 0)
    status = optimizer_audit(w)
    if status == 0:
        status = so_audit_records(w)
    if status == 0:
        status = so_audit_references(w)
    if status == 0:
        status = so_audit_extents(w)
    so_fill_range(w, scratch, slots * 24, 0)
    return status

def init_state(w: mutbuf, slots: u64, pool: u64) -> u64:
    if length(w) != 139264 + so_state_size(slots, pool):
        return 0
    fill_range(w, 0, length(w), 0)
    so_state_init(w, slots, pool)
    putw(w, 8, 1)
    putw(w, 64, 3616443639461270600)
    return 1

def state_header(w: mutbuf) -> u64:
    if length(w) == 0:
        return 0
    if length(w) < 139264 + 2352 + 131072:
        return 7
    if so_state_header(w) != 0 or getw(w, 64) != 3616443639461270600:
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
    if getw(w,120) == 18446744073709551615:
        return terror(w,8)
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
    putw(w,120,getw(w,120)+1)
    putw(w,p+320,getw(w,120))
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
        if (code >= 71 and code <= 79) or code == 90:
            next = owned_optimizer_command(s,o,w,32+i*64,at,end)
        elif code >= 60 and code <= 69:
            next = rng_command(s, o, w, scratch, 32 + i * 64, at, end)
        else:
            next = command(s, o, w, 32 + i * 64, at, end)
        error = getw(w, 0)
        if error != 0:
            putw(o, 24, error)
            return next
        putw(w, getw(w, 96) + i * 8, getw(o, at + 16))
        if getr(s, 32 + i * 64) == 58 and getw(o, at + 16) != 0:
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
                if getw(w, p + 40) != 0:
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

def partial_apply_node(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    op = getw(w, p + 40)
    if op == 0:
        if getw(w,p+32) == 0:
            return 1
        present = getw(w, p + 360)
        j = 0
        n = getw(w, p + 8)
        dst = getw(w, p + 24)
        src = getw(w, p + 160)
        while j < n:
            value = get4(w, src + j * 4)
            if present != 0:
                value = addf(get4(w, dst + j * 4), value)
            put4(w, dst + j * 4, value)
            j = j + 1
        putw(w, p + 360, 1)
    if op >= 1 and op <= 4:
        back_binary(w, t)
    if op == 5 or op == 6 or op == 10 or (op >= 11 and op <= 17):
        back_unary(w, t)
    if op == 7 or op == 8:
        back_reduce(w, t)
    if op == 18 or op == 19:
        back_axis(w, t)
    if op == 20 or op == 21:
        back_softmax(w, t)
    if op >= 22 and op <= 24:
        back_spatial(w, t)
    if op == 25:
        back_matmul(w, t)
    if op == 26:
        back_index_select(w, t)
    if op == 27:
        back_gather(w, t)
    if op == 37 or op == 38:
        back_concat(w, t)
    if op == 39 or op == 40:
        back_indexed_add(w, t)
    return choose(getw(w, 0) == 0, 1, 0)

def ensure_backward_work(w: mutbuf, limit: u64, epoch: u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 168) == epoch:
            if getw(w, p + 40) == 0 and getw(w,p+32) != 0 and getw(w, p + 248) == 0:
                if getw(w, p + 360) != 0:
                    return terror(w, 9)
                grad = alloc(w, getw(w, p + 8) * 4)
                if grad == 0:
                    return 0
                putw(w, p + 24, grad)
                putw(w, p + 248, getw(w, 80))
            if getw(w, p + 256) == 0:
                address = alloc(w, getw(w, p + 8) * 4)
                if address == 0:
                    return 0
                putw(w, p + 160, address)
                putw(w, p + 256, getw(w, 80))
        i = i + 1
    return 1
def private_tensor(w:mutbuf, node:u64, incarnation:u64) -> u64:
    if node == 0 or node >= getw(w,8) or incarnation == 0:
        return 0
    p = dptr(w,node)
    if getw(w,p+304) == 0 or getw(w,p+320) != incarnation or getw(w,p+240) != 1:
        return 0
    if so_tensor_live(w,getw(w,p+192)) == 0 or getw(w,p+208) < getw(w,p+312):
        return 0
    return p

def private_tensor_retain(w:mutbuf, node:u64, incarnation:u64) -> u64:
    p = private_tensor(w,node,incarnation)
    if p == 0:
        return terror(w,9)
    if getw(w,p+312) >= 4294967295 or getw(w,p+208) >= 4294967295:
        return terror(w,8)
    putw(w,p+312,getw(w,p+312)+1)
    putw(w,p+208,getw(w,p+208)+1)
    return 1

def private_tensor_release(w:mutbuf, node:u64, incarnation:u64) -> u64:
    p = private_tensor(w,node,incarnation)
    if p == 0 or getw(w,p+312) == 0 or getw(w,p+208) == 0:
        return terror(w,9)
    putw(w,p+312,getw(w,p+312)-1)
    putw(w,p+208,getw(w,p+208)-1)
    return 1

def publish_tensor(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+240) != 1:
        return terror(w,1)
    if getw(w,p+200) == 0:
        old = getw(w,p+192)
        tp = so_slot_lookup(w,1,old)
        if tp == 0:
            return terror(w,9)
        sid = so_uget(w,tp+16)
        vid = so_uget(w,tp+24)
        offset = so_uget(w,tp+32)
        size = so_uget(w,tp+40)
        if so_slot_drop(w,1,old) != 0:
            return terror(w,9)
        if storage_capacity(w,1,0) == 0:
            return 0
        fresh = so_slot_new(w,1)
        tp = so_slot_lookup(w,1,fresh)
        if tp == 0:
            return terror(w,9)
        so_uput(w,tp+16,sid)
        so_uput(w,tp+24,vid)
        so_uput(w,tp+32,offset)
        so_uput(w,tp+40,size)
        so_uput(w,tp+48,t)
        putw(w,p+192,fresh)
    putw(w,p+200,1)
    return getw(w,p+192)

def graph_collect_since(w:mutbuf, first:u64) -> u64:
    i = getw(w,8)-1
    while i >= first and i > 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+200) == 0:
            if getw(w,p+208) == 0:
                if getw(w,p+312) != 0:
                    return terror(w,9)
                graph_node_release(w,i)
            elif getw(w,p+312) == 0 and getw(w,p+88) == 0:
                tensor_row_release(w,i)
        i = i-1
    return choose(getw(w,0) == 0,1,0)

def graph_reset(w:mutbuf, keep:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    old_next = getw(w,8)
    if keep >= old_next or keep > 65535:
        return terror(w,7)
    if getw(w,32) != 0 and keep != getw(w,40):
        return terror(w,7)
    if getw(w,32) == 18446744073709551615:
        return terror(w,8)
    i = 1
    while i <= keep:
        p = dptr(w,i)
        if getw(w,p+304) == 0 or getw(w,p+200) != 1:
            return terror(w,6)
        if getw(w,p+40) != 0 or getw(w,p+64) != i:
            return terror(w,6)
        i = i+1
    i = keep+1
    while i < old_next:
        putw(w,dptr(w,i)+200,0)
        i = i+1
    graph_collect(w)
    if getw(w,0) != 0:
        return 0
    highest = keep
    i = keep+1
    while i < old_next:
        p = dptr(w,i)
        if getw(w,p+304) != 0:
            if getw(w,p+312) == 0 or getw(w,p+40) != 0:
                return terror(w,9)
            highest = i
        else:
            memory_fill(w,p,384,0)
        i = i+1
    page = (highest >> 8)+1
    while page < 256:
        hold_release(w,getw(w,6144+page*8))
        putw(w,4096+page*8,0)
        putw(w,6144+page*8,0)
        page = page+1
    putw(w,8,highest+1)
    putw(w,40,keep)
    putw(w,32,getw(w,32)+1)
    return 1

def leaf_requires_grad(w:mutbuf, id:u64, enabled:u64) -> u64:
    t = public_node(w,id)
    if t == 0:
        return 0
    if enabled > 1:
        return terror(w,7)
    p = dptr(w,t)
    if getw(w,p+40) != 0:
        return terror(w,5)
    putw(w,p+32,enabled)
    return getw(w,p+192)

def gradient_present(w:mutbuf, id:u64) -> u64:
    t = public_node(w,id)
    if t == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0:
        return 0
    return getw(w,p+360)

def tgrad(w:mutbuf, t:u64, index:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0:
        return terror(w,5)
    if index >= getw(w,p+8):
        return terror(w,7)
    if getw(w,p+360) == 0:
        return 0
    if getw(w,p+248) == 0 or getw(w,p+24) == 0:
        return terror(w,9)
    return get4(w,getw(w,p+24)+index*4)

def zero_grad_none(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0:
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
    if getw(w,p+40) != 0:
        return terror(w,5)
    if getw(w,p+360) == 0:
        return 1
    if getw(w,p+248) == 0 or getw(w,p+24) == 0:
        return terror(w,9)
    memory_fill(w,getw(w,p+24),getw(w,p+8)*4,0)
    return 1

def optimizer_sgd_element(w:mutbuf, gp:u64, parameter:u64, buffer:u64, created:u64, index:u64) -> u64:
    p = dptr(w,parameter)
    grad = get4(w,getw(w,p+24)+index*4)
    if getw(w,gp+40) != 0:
        grad = negf(grad)
    address = taddr(w,parameter,index)
    parameter_value = get4(w,address)
    decay = getw(w,gp+24)
    if absf(decay) != 0:
        grad = addf(grad,mulf(decay,parameter_value))
    if buffer != 0:
        value = grad
        state_address = taddr(w,buffer,index)
        if created == 0:
            value = addf(mulf(get4(w,state_address),getw(w,gp+8)),mulf(subf(1065353216,getw(w,gp+16)),grad))
        put4(w,state_address,value)
        if getw(w,gp+32) != 0:
            grad = addf(grad,mulf(getw(w,gp+8),value))
        else:
            grad = value
    put4(w,address,addf(parameter_value,mulf(negf(getw(w,gp)),grad)))
    return 1

def optimizer_blob(w:mutbuf, id:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    id = resolve_handle(w,id)
    p = so_slot_lookup(w,8,id)
    if p == 0:
        return terror(w,1)
    groups = so_uget(w,p+24)
    count = so_uget(w,p+32)
    if groups == 0 or groups > 1024 or count > 4096 or so_uget(w,p+56) != 0:
        return terror(w,9)
    address = private_hold(w,so_uget(w,p+16),64+groups*64+count*48)
    if address == 0 or getw(w,address) != groups or getw(w,address+8) != count:
        return terror(w,9)
    return address

def optimizer_group_at(w:mutbuf, blob:u64, group:u64) -> u64:
    if group >= getw(w,blob):
        return terror(w,7)
    return blob+64+group*64

def optimizer_entry_at(w:mutbuf, blob:u64, index:u64) -> u64:
    return blob+64+getw(w,blob)*64+index*48

def optimizer_occurrence(w:mutbuf, blob:u64, group:u64, occurrence:u64) -> u64:
    gp = optimizer_group_at(w,blob,group)
    if gp == 0:
        return 0
    if occurrence >= getw(w,gp+56):
        return terror(w,7)
    ep = optimizer_entry_at(w,blob,getw(w,gp+48)+occurrence)
    if private_tensor(w,getw(w,ep),getw(w,ep+8)) == 0:
        return terror(w,9)
    return ep

def optimizer_key(w:mutbuf, blob:u64, occurrence:u64) -> u64:
    index = getw(w,occurrence+16)
    if index >= getw(w,blob+8):
        return terror(w,9)
    key = optimizer_entry_at(w,blob,index)
    if getw(w,key) != getw(w,occurrence) or getw(w,key+8) != getw(w,occurrence+8) or getw(w,key+16) != index:
        return terror(w,9)
    return key

def optimizer_word(w:mutbuf, field:u64, value:u64) -> u64:
    if field < 4:
        if value > 4294967295:
            return terror(w,7)
    elif value > 1:
        return terror(w,7)
    return 1

def optimizer_payload(s:readbuf, w:mutbuf, offset:u64, size:u64, end:u64) -> u64:
    if offset < end or offset > length(s) or size > length(s)-offset or (offset & 7) != 0:
        return terror(w,100)
    return 1

def optimizer_defaults(s:readbuf, w:mutbuf, offset:u64, end:u64) -> u64:
    if optimizer_payload(s,w,offset,48,end) == 0:
        return 0
    j = 0
    while j < 6:
        if optimizer_word(w,j,getr(s,offset+j*8)) == 0:
            return 0
        j = j+1
    lr = getr(s,offset)
    momentum = getr(s,offset+8)
    dampening = getr(s,offset+16)
    decay = getr(s,offset+24)
    if ltf(lr,0) != 0 or ltf(momentum,0) != 0 or ltf(decay,0) != 0:
        return terror(w,7)
    if getr(s,offset+32) != 0:
        if ltf(momentum,0) != 0 or absf(momentum) == 0 or absf(dampening) != 0:
            return terror(w,7)
    return 1

def optimizer_group_payload(s:readbuf, w:mutbuf, offset:u64, end:u64) -> u64:
    if optimizer_payload(s,w,offset,64,end) == 0:
        return 0
    count = getr(s,offset)
    mask = getr(s,offset+8)
    if count > 4096 or mask > 63:
        return terror(w,7)
    if optimizer_payload(s,w,offset,64+count*8,end) == 0:
        return 0
    j = 0
    while j < 6:
        value = getr(s,offset+16+j*8)
        if (mask & (1 << j)) == 0:
            if value != 0:
                return terror(w,7)
        elif optimizer_word(w,j,value) == 0:
            return 0
        j = j+1
    return offset+64+count*8

def optimizer_fill_group(s:readbuf, w:mutbuf, blob:u64, group:u64, offset:u64, first:u64) -> u64:
    gp = blob+64+group*64
    mask = getr(s,offset+8)
    count = getr(s,offset)
    j = 0
    while j < 6:
        value = getw(w,blob+16+j*8)
        if (mask & (1 << j)) != 0:
            value = getr(s,offset+16+j*8)
        putw(w,gp+j*8,value)
        j = j+1
    putw(w,gp+48,first)
    putw(w,gp+56,count)
    i = 0
    duplicates = 0
    while i < count:
        t = public_node(w,getr(s,offset+64+i*8))
        if t == 0:
            return 0
        p = dptr(w,t)
        if getw(w,p+40) != 0:
            return terror(w,5)
        incarnation = getw(w,p+320)
        index = first+i
        key = index
        j = 0
        while j < index:
            previous = optimizer_entry_at(w,blob,j)
            if getw(w,previous) == t and getw(w,previous+8) == incarnation:
                if j < first:
                    return terror(w,7)
                key = getw(w,previous+16)
                j = index
            else:
                j = j+1
        ep = optimizer_entry_at(w,blob,index)
        putw(w,ep,t)
        putw(w,ep+8,incarnation)
        putw(w,ep+16,key)
        if key != index:
            duplicates = duplicates+1
        if private_tensor_retain(w,t,incarnation) == 0:
            return 0
        i = i+1
    return duplicates+1

def optimizer_create(s:readbuf, w:mutbuf, algorithm:u64, defaults:u64, groups_at:u64, end:u64) -> u64:
    if algorithm != 0:
        return terror(w,7)
    if optimizer_defaults(s,w,defaults,end) == 0 or optimizer_payload(s,w,groups_at,8,end) == 0:
        return 0
    groups = getr(s,groups_at)
    if groups == 0 or groups > 1024:
        return terror(w,7)
    offset = groups_at+8
    count = 0
    i = 0
    while i < groups:
        next = optimizer_group_payload(s,w,offset,end)
        if next == 0:
            return 0
        count = count+getr(s,offset)
        if count > 4096:
            return terror(w,3)
        offset = next
        i = i+1
    if storage_capacity(w,8,0) == 0:
        return 0
    blob = alloc(w,64+groups*64+count*48)
    if blob == 0:
        return 0
    hold = getw(w,80)
    putw(w,blob,groups)
    putw(w,blob+8,count)
    j = 0
    while j < 6:
        putw(w,blob+16+j*8,getr(s,defaults+j*8))
        j = j+1
    offset = groups_at+8
    first = 0
    duplicates = 0
    i = 0
    while i < groups:
        result = optimizer_fill_group(s,w,blob,i,offset,first)
        if result == 0:
            return 0
        duplicates = duplicates+result-1
        amount = getr(s,offset)
        first = first+amount
        offset = offset+64+amount*8
        i = i+1
    id = so_slot_new(w,8)
    p = so_slot_lookup(w,8,id)
    if p == 0:
        return terror(w,9)
    so_uput(w,p+16,hold)
    so_uput(w,p+24,groups)
    so_uput(w,p+32,count)
    so_uput(w,p+40,count-duplicates)
    so_uput(w,p+48,duplicates)
    return id

def optimizer_add_group(s:readbuf, w:mutbuf, id:u64, offset:u64, end:u64) -> u64:
    old = optimizer_blob(w,id)
    if old == 0 or optimizer_group_payload(s,w,offset,end) == 0:
        return 0
    id = resolve_handle(w,id)
    p = so_slot_lookup(w,8,id)
    groups = getw(w,old)
    count = getw(w,old+8)
    extra = getr(s,offset)
    if groups >= 1024 or extra > 4096-count:
        return terror(w,3)
    blob = alloc(w,64+(groups+1)*64+(count+extra)*48)
    if blob == 0:
        return 0
    hold = getw(w,80)
    move_within(w,object_address(w,blob),object_address(w,old),64+groups*64)
    putw(w,blob,groups+1)
    putw(w,blob+8,count+extra)
    move_within(w,object_address(w,optimizer_entry_at(w,blob,0)),object_address(w,optimizer_entry_at(w,old,0)),count*48)
    result = optimizer_fill_group(s,w,blob,groups,offset,count)
    if result == 0:
        return 0
    if hold_release(w,so_uget(w,p+16)) == 0:
        return 0
    so_uput(w,p+16,hold)
    so_uput(w,p+24,groups+1)
    so_uput(w,p+32,count+extra)
    duplicates = so_uget(w,p+48)+result-1
    so_uput(w,p+40,count+extra-duplicates)
    so_uput(w,p+48,duplicates)
    return 1

def optimizer_drop(w:mutbuf, id:u64) -> u64:
    blob = optimizer_blob(w,id)
    if blob == 0:
        return 0
    id = resolve_handle(w,id)
    p = so_slot_lookup(w,8,id)
    hold = so_uget(w,p+16)
    i = 0
    while i < getw(w,blob+8):
        ep = optimizer_entry_at(w,blob,i)
        if private_tensor_release(w,getw(w,ep),getw(w,ep+8)) == 0:
            return 0
        if getw(w,ep+16) == i and getw(w,ep+24) != 0:
            if private_tensor_release(w,getw(w,ep),getw(w,ep+8)) == 0:
                return 0
            if private_tensor_release(w,getw(w,ep+32),getw(w,ep+40)) == 0:
                return 0
        i = i+1
    if hold_release(w,hold) == 0 or so_slot_drop(w,8,id) != 0:
        return terror(w,9)
    return graph_collect(w)

def optimizer_zero(w:mutbuf, id:u64, none:u64) -> u64:
    if none > 1:
        return terror(w,7)
    blob = optimizer_blob(w,id)
    if blob == 0:
        return 0
    i = 0
    while i < getw(w,blob+8):
        ep = optimizer_entry_at(w,blob,i)
        node = getw(w,ep)
        if private_tensor(w,node,getw(w,ep+8)) == 0:
            return terror(w,9)
        if none != 0:
            zero_grad_none(w,node)
        else:
            zero_grad(w,node)
        if getw(w,0) != 0:
            return 0
        i = i+1
    return 1

def optimizer_new_momentum(w:mutbuf, parameter:u64) -> u64:
    p = dptr(w,parameter)
    t = descriptor(w,getw(w,p),p+96,0,1)
    if t == 0:
        return 0
    q = dptr(w,t)
    if is_dense(w,parameter) != 0:
        j = 0
        while j < getw(w,p):
            putw(w,q+128+j*8,getw(w,p+128+j*8))
            j = j+1
    if private_tensor_retain(w,t,getw(w,q+320)) == 0:
        return 0
    putw(w,q+200,0)
    return t

def optimizer_plan_group(w:mutbuf, blob:u64, group:u64, plan:u64) -> u64:
    gp = blob+64+group*64
    first = getw(w,gp+48)
    count = getw(w,gp+56)
    momentum = getw(w,gp+8)
    i = 0
    while i < count:
        ep = optimizer_entry_at(w,blob,first+i)
        node = getw(w,ep)
        p = private_tensor(w,node,getw(w,ep+8))
        if p == 0:
            return terror(w,9)
        item = plan+i*32
        if getw(w,p+360) != 0:
            putw(w,item,node)
            if absf(momentum) != 0:
                key = optimizer_key(w,blob,ep)
                if key == 0:
                    return 0
                if getw(w,key+24) == 0:
                    if private_tensor_retain(w,node,getw(w,ep+8)) == 0:
                        return 0
                    putw(w,key+24,1)
                else:
                    buffer = getw(w,key+32)
                    if buffer != 0:
                        if private_tensor(w,buffer,getw(w,key+40)) == 0:
                            return terror(w,9)
                        putw(w,item+8,buffer)
                putw(w,item+16,key)
        i = i+1
    return 1

def optimizer_sgd_occurrence(w:mutbuf, gp:u64, item:u64) -> u64:
    parameter = getw(w,item)
    if parameter == 0:
        return 1
    p = dptr(w,parameter)
    n = getw(w,p+8)
    if getw(w,p+360) == 0 or getw(w,p+248) == 0 or getw(w,p+24) == 0:
        return terror(w,9)
    if version_can_bump(w,parameter,1) == 0:
        return 0
    momentum = getw(w,gp+8)
    buffer = getw(w,item+8)
    created = 0
    if absf(momentum) != 0:
        if buffer == 0:
            buffer = optimizer_new_momentum(w,parameter)
            if buffer == 0:
                return 0
            putw(w,item+8,buffer)
            putw(w,item+24,1)
            created = 1
        elif version_can_bump(w,buffer,2) == 0:
            return 0
    i = 0
    while i < n:
        optimizer_sgd_element(w,gp,parameter,buffer,created,i)
        i = i+1
    if absf(momentum) != 0 and created == 0:
        bump_version(w,buffer)
        bump_version(w,buffer)
    return bump_version(w,parameter)

def optimizer_commit_group(w:mutbuf, gp:u64, plan:u64) -> u64:
    i = 0
    while i < getw(w,gp+56):
        item = plan+i*32
        key = getw(w,item+16)
        if getw(w,item) != 0 and key != 0:
            buffer = getw(w,item+8)
            incarnation = getw(w,dptr(w,buffer)+320)
            old = getw(w,key+32)
            if buffer != old:
                if private_tensor_retain(w,buffer,incarnation) == 0:
                    return 0
                if old != 0 and private_tensor_release(w,old,getw(w,key+40)) == 0:
                    return 0
                putw(w,key+32,buffer)
                putw(w,key+40,incarnation)
            if getw(w,item+24) != 0:
                if private_tensor_release(w,buffer,incarnation) == 0:
                    return 0
        i = i+1
    return 1

def optimizer_step(w:mutbuf, id:u64) -> u64:
    blob = optimizer_blob(w,id)
    if blob == 0:
        return 0
    maximum = 0
    i = 0
    while i < getw(w,blob):
        count = getw(w,blob+64+i*64+56)
        if count > maximum:
            maximum = count
        i = i+1
    plan = 0
    hold = 0
    if maximum != 0:
        plan = alloc(w,maximum*32)
        if plan == 0:
            return 0
        hold = getw(w,80)
    i = 0
    while i < getw(w,blob):
        gp = blob+64+i*64
        count = getw(w,gp+56)
        if count != 0:
            memory_fill(w,plan,maximum*32,0)
            if optimizer_plan_group(w,blob,i,plan) == 0:
                return 0
            j = 0
            while j < count:
                if optimizer_sgd_occurrence(w,gp,plan+j*32) == 0:
                    return 0
                j = j+1
            if optimizer_commit_group(w,gp,plan) == 0:
                return 0
        i = i+1
    if hold_release(w,hold) == 0:
        return 0
    return graph_collect(w)

def optimizer_inventory(w:mutbuf, o:mutbuf, at:u64, id:u64) -> u64:
    p = so_slot_lookup(w,8,resolve_handle(w,id))
    if p == 0:
        return terror(w,1)
    putw(o,at,so_uget(w,p+24))
    putw(o,at+8,so_uget(w,p+32))
    putw(o,at+16,so_uget(w,p+40))
    putw(o,at+24,so_uget(w,p+48))
    return 32

def optimizer_group_inventory(w:mutbuf, o:mutbuf, at:u64, blob:u64, group:u64) -> u64:
    gp = optimizer_group_at(w,blob,group)
    if gp == 0:
        return 0
    count = getw(w,gp+56)
    size = 64+count*16
    if size > length(o)-at:
        return terror(w,3)
    putw(o,at,group)
    putw(o,at+8,count)
    j = 0
    while j < 6:
        putw(o,at+16+j*8,getw(w,gp+j*8))
        j = j+1
    j = 0
    while j < count:
        ep = optimizer_entry_at(w,blob,getw(w,gp+48)+j)
        p = private_tensor(w,getw(w,ep),getw(w,ep+8))
        if p == 0:
            return terror(w,9)
        putw(o,at+64+j*16,getw(w,ep+8))
        token = 0
        if getw(w,p+200) == 1:
            token = getw(w,p+192)
        putw(o,at+72+j*16,token)
        j = j+1
    return size

def optimizer_state_inventory(w:mutbuf, o:mutbuf, at:u64, key:u64) -> u64:
    putw(o,at,getw(w,key+8))
    putw(o,at+8,getw(w,key+24))
    putw(o,at+16,getw(w,key+40))
    token = 0
    if getw(w,key+24) != 0:
        p = private_tensor(w,getw(w,key+32),getw(w,key+40))
        if p == 0:
            return terror(w,9)
        if getw(w,p+200) == 1:
            token = getw(w,p+192)
    putw(o,at+24,token)
    return 32

def owned_optimizer_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64, end:u64) -> u64:
    code = getr(s,cp)
    a = getr(s,cp+8)
    b = getr(s,cp+16)
    c = getr(s,cp+24)
    d = getr(s,cp+32)
    result = 0
    payload = 0
    first = getw(w,8)
    if getr(s,cp+40) != 0 or getr(s,cp+48) != 0 or getr(s,cp+56) != 0:
        terror(w,7)
    if d != 0:
        terror(w,7)
    if code != 71 and code != 75 and code != 76 and code != 77 and c != 0:
        terror(w,7)
    if (code == 72 or code == 78) and b != 0:
        terror(w,7)
    if getw(w,0) == 0:
        if code == 90:
            result = leaf_requires_grad(w,a,b)
        elif code == 71:
            if length(o)-at-32 < 32:
                terror(w,3)
            else:
                result = optimizer_create(s,w,a,b,c,end)
                if result != 0:
                    payload = optimizer_inventory(w,o,at+32,result)
        elif code == 72:
            result = optimizer_step(w,a)
        elif code == 73:
            result = optimizer_zero(w,a,b)
        elif code == 78:
            result = optimizer_drop(w,a)
        elif code == 79:
            if length(o)-at-32 < 32:
                terror(w,3)
            else:
                result = optimizer_add_group(s,w,a,b,end)
                if result != 0:
                    payload = optimizer_inventory(w,o,at+32,a)
        else:
            blob = optimizer_blob(w,a)
            if blob != 0:
                if code == 74:
                    payload = optimizer_group_inventory(w,o,at+32,blob,b)
                    if payload != 0:
                        result = getw(o,at+40)
                elif code == 75 or code == 76 or code == 77:
                    ep = optimizer_occurrence(w,blob,b,c)
                    if ep != 0:
                        if code == 75:
                            result = publish_tensor(w,getw(w,ep))
                        else:
                            key = optimizer_key(w,blob,ep)
                            if key != 0:
                                if code == 76:
                                    if length(o)-at-32 < 32:
                                        terror(w,3)
                                    else:
                                        payload = optimizer_state_inventory(w,o,at+32,key)
                                        result = getw(w,key+24)
                                elif getw(w,key+24) != 0:
                                    result = publish_tensor(w,getw(w,key+32))
                else:
                    terror(w,4)
    if getw(w,0) == 0:
        finish_command(w,first)
    status = getw(w,0)
    if status != 0:
        result = 0
        payload = 0
    putw(o,at,code)
    putw(o,at+8,status)
    putw(o,at+16,result)
    putw(o,at+24,payload)
    return at+32+payload

def optimizer_audit_entry(w:mutbuf, blob:u64, i:u64, first:u64) -> u64:
    ep = optimizer_entry_at(w,blob,i)
    node = getw(w,ep)
    incarnation = getw(w,ep+8)
    p = private_tensor(w,node,incarnation)
    if p == 0 or getw(w,p+40) != 0:
        return 9
    if optimizer_audit_reference(w,node,incarnation) != 0:
        return 9
    key_index = getw(w,ep+16)
    if key_index < first or key_index > i:
        return 9
    key = optimizer_entry_at(w,blob,key_index)
    if getw(w,key) != node or getw(w,key+8) != incarnation or getw(w,key+16) != key_index:
        return 9
    if key_index != i:
        if getw(w,ep+24) != 0 or getw(w,ep+32) != 0 or getw(w,ep+40) != 0:
            return 9
    else:
        j = 0
        while j < i:
            other = optimizer_entry_at(w,blob,j)
            if getw(w,other) == node and getw(w,other+8) == incarnation:
                return 9
            j = j+1
        present = getw(w,key+24)
        if present > 1:
            return 9
        if present == 0:
            if getw(w,key+32) != 0 or getw(w,key+40) != 0:
                return 9
        else:
            buffer = getw(w,key+32)
            bp = private_tensor(w,buffer,getw(w,key+40))
            if bp == 0 or getw(w,bp+40) != 0 or getw(w,bp) != getw(w,p):
                return 9
            j = 0
            while j < getw(w,p):
                if getw(w,bp+96+j*8) != getw(w,p+96+j*8):
                    return 9
                j = j+1
            if optimizer_audit_reference(w,node,incarnation) != 0 or optimizer_audit_reference(w,buffer,getw(w,key+40)) != 0:
                return 9
    return choose(key_index == i,1,2)

def optimizer_audit_reference(w:mutbuf, node:u64, incarnation:u64) -> u64:
    p = private_tensor(w,node,incarnation)
    if p == 0 or getw(w,p+336) >= 4294967295:
        return 9
    putw(w,p+336,getw(w,p+336)+1)
    return 0

def optimizer_audit_record(w:mutbuf, row:u64) -> u64:
    groups = so_uget(w,row+24)
    count = so_uget(w,row+32)
    if groups == 0 or groups > 1024 or count > 4096 or so_uget(w,row+56) != 0:
        return 9
    if so_uget(w,row+48) > count or so_uget(w,row+40) != count-so_uget(w,row+48):
        return 9
    blob = private_hold(w,so_uget(w,row+16),64+groups*64+count*48)
    if blob == 0 or getw(w,blob) != groups or getw(w,blob+8) != count:
        return 9
    j = 0
    while j < 6:
        value = getw(w,blob+16+j*8)
        if (j < 4 and value > 4294967295) or (j >= 4 and value > 1):
            return 9
        j = j+1
    first = 0
    duplicate_count = 0
    group = 0
    while group < groups:
        gp = blob+64+group*64
        amount = getw(w,gp+56)
        if getw(w,gp+48) != first or first > count or amount > count-first:
            return 9
        j = 0
        while j < 6:
            value = getw(w,gp+j*8)
            if (j < 4 and value > 4294967295) or (j >= 4 and value > 1):
                return 9
            j = j+1
        i = first
        while i < first+amount:
            entry_status = optimizer_audit_entry(w,blob,i,first)
            if entry_status != 1 and entry_status != 2:
                return 9
            duplicate_count = duplicate_count+entry_status-1
            i = i+1
        first = first+amount
        group = group+1
    if first != count or duplicate_count != so_uget(w,row+48):
        return 9
    return 0

def optimizer_audit(w:mutbuf) -> u64:
    slots = so_uget(w,16)
    budget = 0
    j = 1
    while j <= slots:
        row = so_slot_at(w,8,j)
        if so_uget(w,row+8) == 1:
            count = so_uget(w,row+32)
            if count > 4096:
                return 9
            if count*count > 16777216-budget:
                return 3
            budget = budget+count*count
        j = j+1
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if getw(w,p+336) != 0 or getw(w,p+368) != 0 or getw(w,p+376) != 0:
            return 9
        if getw(w,p+304) != 0:
            if getw(w,p+320) == 0 or getw(w,p+320) > getw(w,120):
                return 9
        i = i+1
    status = graph_audit(w)
    j = 1
    while j <= slots and status == 0:
        row = so_slot_at(w,8,j)
        if so_uget(w,row+8) == 1:
            status = optimizer_audit_record(w,row)
            other_index = 1
            while other_index < j and status == 0:
                other = so_slot_at(w,8,other_index)
                if so_uget(w,other+8) == 1 and so_uget(w,other+16) == so_uget(w,row+16):
                    status = 9
                other_index = other_index+1
            other_index = 1
            while other_index <= slots and status == 0:
                other = so_slot_at(w,7,other_index)
                if so_uget(w,other+8) == 1 and so_uget(w,other+16) == so_uget(w,row+16):
                    status = 9
                other_index = other_index+1
        j = j+1
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if getw(w,p+304) != 0:
            if getw(w,p+336) != getw(w,p+312) or getw(w,p+208) < getw(w,p+312):
                status = 9
        elif getw(w,p+312) != 0 or getw(w,p+336) != 0:
            status = 9
        putw(w,p+336,0)
        i = i+1
    return status

def graph_parent_at(w:mutbuf, t:u64, index:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 37 or op == 38:
        return getw(w,getw(w,p+184)+8+index*16)
    return getw(w,p+48+index*8)

def graph_edge_enabled(w:mutbuf, t:u64, index:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 0:
        return 0
    if op == 37 or op == 38:
        return getw(w,getw(w,p+184)+16+index*16)
    return getw(w,p+72+index*8)

def record(w:mutbuf, t:u64, op:u64, a:u64, b:u64) -> u64:
    result = record_body(w,t,op,a,b)
    if result == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+32) == 0:
        putw(w,p+40,0)
        putw(w,p+48,0)
        putw(w,p+56,0)
        return result
    count = gradient_parent_count(w,t)
    i = 0
    while i < count:
        parent = graph_parent_at(w,t,i)
        flag = 0
        if parent != 0:
            q = dptr(w,parent)
            flag = getw(w,q+32)
            if flag != 0 and getw(w,q+40) == 0:
                if getw(w,q+88) >= 4294967295:
                    return terror(w,8)
                putw(w,q+88,getw(w,q+88)+1)
        if op == 37 or op == 38:
            putw(w,getw(w,p+184)+16+i*16,flag)
        else:
            putw(w,p+72+i*8,flag)
        i = i+1
    return result

def graph_release_edge(w:mutbuf, parent:u64, captured:u64) -> u64:
    if parent == 0:
        return 1
    q = dptr(w,parent)
    if captured != 0 and getw(w,q+40) == 0:
        refs = getw(w,q+88)
        if refs == 0:
            return terror(w,9)
        putw(w,q+88,refs-1)
    return node_release(w,parent)

def graph_node_release(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+312) != 0 or getw(w,p+88) != 0:
        return terror(w,9)
    if getw(w,p+40) != 0:
        j = 0
        while j < gradient_parent_count(w,t):
            if graph_release_edge(w,graph_parent_at(w,t,j),graph_edge_enabled(w,t,j)) == 0:
                return 0
            j = j+1
    saved_release(w,getw(w,p+280))
    saved_release(w,getw(w,p+288))
    saved_release(w,getw(w,p+296))
    tensor_row_release(w,t)
    hold_release(w,getw(w,p+248))
    hold_release(w,getw(w,p+256))
    hold_release(w,getw(w,p+264))
    hold_release(w,getw(w,p+272))
    putw(w,p+304,0)
    return choose(getw(w,0) == 0,1,0)

def gradient_parent_at(w:mutbuf, t:u64, index:u64) -> u64:
    if graph_edge_enabled(w,t,index) == 0:
        return 0
    return graph_parent_at(w,t,index)

def backward_getw(w:mutbuf, address:u64) -> u64:
    t = getw(w,112)
    if t != 0:
        i = 0
        count = gradient_parent_count(w,t)
        while i < count:
            parent = graph_parent_at(w,t,i)
            if parent != 0 and address == dptr(w,parent)+32:
                return graph_edge_enabled(w,t,i)
            i = i+1
    return getw(w,address)

def gadd(w:mutbuf, t:u64, index:u64, value:u64) -> u64:
    p = dptr(w,t)
    flags = getw(w,p+368)
    if flags == 0:
        return 0
    address = getw(w,p+160)+index*4
    if flags > 1:
        flag = object_address(w,flags+index)
        if byte(w,flag) == 0:
            store(w,flag,1)
            put4(w,address,value)
            return 1
    put4(w,address,addf(get4(w,address),value))
    return 1

def mark_concat(w:mutbuf, t:u64, epoch:u64) -> u64:
    j = 0
    while j < gradient_parent_count(w,t):
        parent = gradient_parent_at(w,t,j)
        if parent != 0:
            putw(w,dptr(w,parent)+168,epoch)
        j = j+1
    return 1

def mark_graph(w:mutbuf, t:u64, epoch:u64) -> u64:
    putw(w,dptr(w,t)+168,epoch)
    i = t
    while i > 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch and getw(w,p+40) != 0:
            if getw(w,p+328) != 0:
                return terror(w,10)
            if graph_saved_check(w,getw(w,p+280)) == 0:
                return 0
            if graph_saved_check(w,getw(w,p+288)) == 0:
                return 0
            if graph_saved_check(w,getw(w,p+296)) == 0:
                return 0
            if indexed_add_backward_check(w,i) == 0:
                return 0
            mark_concat(w,i,epoch)
        i = i-1
    return 1

def partial_mark(w:mutbuf, t:u64, epoch:u64) -> u64:
    putw(w,dptr(w,t)+168,epoch)
    i = t
    while i > 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch and getw(w,p+40) != 0:
            mark_concat(w,i,epoch)
        i = i-1
    return 1

def partial_parent(w:mutbuf, queue:u64, t:u64, apply:u64) -> u64:
    if t == 0:
        return 1
    at = queue+24+t*8
    pending = getw(w,at)
    if apply == 0:
        putw(w,at,pending+1)
    else:
        if pending == 0:
            return terror(w,8)
        pending = pending-1
        putw(w,at,pending)
        if pending == 0:
            return partial_push(w,queue,t)
    return 1

def partial_edges(w:mutbuf, queue:u64, t:u64, apply:u64) -> u64:
    j = 0
    while j < gradient_parent_count(w,t):
        if partial_parent(w,queue,gradient_parent_at(w,t,j),apply) == 0:
            return 0
        j = j+1
    return 1

def gradient_workspace_size(w:mutbuf, limit:u64, epoch:u64) -> u64:
    maximum = 0
    t = 1
    while t <= limit:
        p = dptr(w,t)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            amount = 0
            j = 0
            count = gradient_parent_count(w,t)
            while j < count:
                parent = gradient_parent_at(w,t,j)
                if parent != 0:
                    part = getw(w,dptr(w,parent)+8)*5
                    if part > 1073741824-amount:
                        return terror(w,3)
                    amount = amount+part
                j = j+1
            if amount > maximum:
                maximum = amount
        t = t+1
    return maximum

def gradient_prepare(w:mutbuf, t:u64, scratch:u64) -> u64:
    cursor = scratch
    j = 0
    count = gradient_parent_count(w,t)
    while j < count:
        parent = gradient_parent_at(w,t,j)
        if parent != 0:
            p = dptr(w,parent)
            if getw(w,p+368) == 0:
                n = getw(w,p+8)
                memory_fill(w,cursor,n*5,0)
                putw(w,p+376,getw(w,p+160))
                putw(w,p+160,cursor)
                putw(w,p+368,1)
                if gradient_direct(w,t,parent) != 0:
                    putw(w,p+368,cursor+n*4)
                cursor = cursor+n*5
        j = j+1
    return 1

def gradient_apply(w:mutbuf, t:u64, scratch:u64) -> u64:
    if getw(w,dptr(w,t)+40) == 0:
        return partial_apply_node(w,t)
    putw(w,112,t)
    gradient_prepare(w,t,scratch)
    partial_apply_node(w,t)
    gradient_finish(w,t,choose(getw(w,0) == 0,1,0))
    putw(w,112,0)
    return choose(getw(w,0) == 0,1,0)

def indexed_add_backward_check(w:mutbuf, t:u64) -> u64:
    putw(w,112,t)
    result = indexed_add_backward_check_body(w,t)
    putw(w,112,0)
    return result

def release_backward_saved(w:mutbuf, t:u64) -> u64:
    putw(w,112,t)
    result = release_backward_saved_body(w,t)
    putw(w,112,0)
    return result

def graph_audit(w:mutbuf) -> u64:
    status = 0
    budget = 0
    i = 1
    while i < getw(w,8) and status == 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0:
            count = gradient_parent_count(w,i)
            if count > 16777216-budget:
                status = 3
            else:
                budget = budget+count
                j = 0
                while j < count and status == 0:
                    parent = graph_parent_at(w,i,j)
                    flag = graph_edge_enabled(w,i,j)
                    if flag > 1 or parent >= i:
                        status = 9
                    elif parent != 0:
                        q = dptr(w,parent)
                        if getw(w,q+304) == 0:
                            status = 9
                        else:
                            putw(w,q+376,getw(w,q+376)+1)
                            if flag != 0 and getw(w,q+40) == 0:
                                putw(w,q+368,getw(w,q+368)+1)
                    elif flag != 0:
                        status = 9
                    j = j+1
        i = i+1
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if status == 0 and getw(w,p+304) != 0:
            if getw(w,p+88) != getw(w,p+368) or getw(w,p+208) != getw(w,p+376)+getw(w,p+312):
                status = 9
        putw(w,p+368,0)
        putw(w,p+376,0)
        i = i+1
    return status

