def init_state(w: mutbuf, slots: u64, pool: u64) -> u64:
    if length(w) != 139264 + so_state_size(slots, pool):
        return 0
    fill_range(w, 0, length(w), 0)
    so_state_init(w, slots, pool)
    putw(w, 8, 1)
    putw(w, 64, 3544421173927498824)
    return 1

def state_header(w: mutbuf) -> u64:
    if length(w) == 0:
        return 0
    if length(w) < 139264 + 2096 + 131072:
        return 7
    if so_state_header(w) != 0 or getw(w, 64) != 3544421173927498824:
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
    remaining = 0
    if getw(w,q+232) <= so_uget(w,sp+24):
        remaining = so_uget(w,sp+24)-getw(w,q+232)
    so_uput(w,tp+40,remaining)
    putw(w, p + 216, sid)
    putw(w, p + 224, vid)
    putw(w, p + 232, getw(w, q + 232))
    return 1

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
    remaining = 0
    if getw(w,p+232) <= so_uget(w,storage+24):
        remaining = so_uget(w,storage+24)-getw(w,p+232)
    so_uput(w,sp+48,remaining)
    so_uput(w, sp + 56, layout)
    return id

def object_stats(w: mutbuf, o: mutbuf, at: u64) -> u64:
    putw(o, at, so_uget(w, 16))
    putw(o, at + 8, so_uget(w, 24))
    putw(o, at + 16, so_uget(w, 40))
    putw(o, at + 24, length(w))
    free = 128 + so_uget(w, 16) * 448
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
    return slots * 488 + 144

def so_state_init(w: mutbuf, slots: u64, pool: u64) -> u64:
    so_uput(w, 0, 3553418507709273160)
    so_uput(w, 8, 2)
    so_uput(w, 16, slots)
    so_uput(w, 24, pool)
    so_uput(w, 32, 1)
    so_uput(w, 48, 1)
    free = 128 + slots * 448
    so_uput(w, free, 0)
    so_uput(w, free + 8, pool)
    return 0

def so_state_header(w: mutbuf) -> u64:
    if length(w) < 144:
        return 9
    if so_uget(w, 0) != 3553418507709273160 or so_uget(w, 8) != 2 or so_uget(w, 48) != 1:
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
    if kind == 0 or kind > 7 or id >> 24 & 255 != kind:
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
    free = 128 + so_uget(w, 16) * 448
    i = 0
    while i < so_uget(w, 32):
        if so_uget(w, free + i * 16 + 8) >= size:
            return 1
        i = i + 1
    return 0

def so_extent_remove(w: mutbuf, index: u64) -> u64:
    free = 128 + so_uget(w, 16) * 448
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
    free = 128 + so_uget(w, 16) * 448
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
    free = 128 + so_uget(w, 16) * 448
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
    so_fill_range(target, 128 + slots * 448, (slots + 1) * 16, 0)
    so_uput(target, 32, so_rget(source, 32))
    so_uput(target, 40, so_rget(source, 40) + 1)
    kind = 0
    while kind < 7:
        so_copy_range(target, source, 128 + kind * slots * 64, 128 + kind * old_slots * 64, old_slots * 64)
        kind = kind + 1
    so_copy_range(target, source, 128 + slots * 448, 128 + old_slots * 448, so_rget(source, 32) * 16)
    so_copy_range(target, source, so_data_base(slots), so_data_base(old_slots), old_pool)
    if pool > old_pool:
        return so_extent_release(target, old_pool, pool - old_pool)
    return 0

def so_tensor_live(w: mutbuf, id: u64) -> u64:
    p = so_slot_lookup(w, 1, id)
    if p == 0:
        return 0
    storage = so_storage_live(w, so_uget(w, p + 16))
    if storage == 0 or so_version_live(w, so_uget(w, p + 24)) == 0:
        return 0
    offset = so_uget(w, p + 32)
    size = so_uget(w, p + 40)
    if offset > 1073741824:
        return 0
    if offset > so_uget(w,storage+24):
        node = so_uget(w,p+48)
        if size != 0 or node == 0 or node >= getw(w,8):
            return 0
        q = dptr(w,node)
        if getw(w,q+8) != 0 or getw(w,q+232) != offset:
            return 0
    else:
        if size > so_uget(w,storage+24)-offset:
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
    if offset > 1073741824:
        return 0
    if offset > so_uget(w,storage+24):
        if size != 0:
            return 0
    else:
        if size > so_uget(w,storage+24)-offset:
            return 0
    return p

def so_audit_count(w: mutbuf, kind: u64, id: u64, field: u64) -> u64:
    if so_slot_lookup(w, kind, id) == 0:
        return 9
    slot = id & 16777215
    p = 144 + so_uget(w, 16) * 464 + (slot - 1) * 24 + field
    value = so_uget(w, p)
    if value >= 4294967295:
        return 9
    so_uput(w, p, value + 1)
    return 0

def so_audit_records(w: mutbuf) -> u64:
    kind = 1
    while kind <= 7:
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
                    hid = so_uget(w,p+16)
                    address = private_hold(w,hid,5056)
                    if address == 0 or rng_state_canonical(w,address) == 0:
                        return 9
                    j = 24
                    while j < 64:
                        if so_uget(w,p+j) != 0:
                            return 9
                        j = j+8
                    j = 1
                    while j < i:
                        other = so_slot_at(w,7,j)
                        if so_uget(w,other+8) == 1 and so_uget(w,other+16) == hid:
                            return 9
                        j = j+1
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
        counters = 144 + so_uget(w, 16) * 464 + (i - 1) * 24
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
    free = 128 + so_uget(w, 16) * 448
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
    free = 128 + so_uget(w, 16) * 448
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
    free = 128 + so_uget(w, 16) * 448
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
    scratch = 144 + slots * 464
    so_fill_range(w, scratch, slots * 24, 0)
    status = so_audit_records(w)
    if status == 0:
        status = so_audit_references(w)
    if status == 0:
        status = so_audit_extents(w)
    so_fill_range(w, scratch, slots * 24, 0)
    return status

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
        code = getr(s,32+i*64)
        if code >= 60 and code <= 69:
            next = rng_command(s,o,w,scratch,32+i*64,at,end)
        else:
            next = command(s,o,w,32+i*64,at,end)
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

def rng_object(w:mutbuf,id:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    id = resolve_handle(w,id)
    p = so_slot_lookup(w,7,id)
    if p == 0:
        return terror(w,1)
    address = private_hold(w,so_uget(w,p+16),5056)
    if address == 0:
        return terror(w,9)
    return address

def rng_load_scratch(w:mutbuf,scratch:mutbuf,address:u64) -> u64:
    copy_from_mut(scratch,w,0,object_address(w,address),5056)
    return 1

def rng_save_scratch(w:mutbuf,scratch:mutbuf,address:u64) -> u64:
    copy_from_mut(w,scratch,object_address(w,address),0,5056)
    return 1

def rng_create_object(w:mutbuf,scratch:mutbuf,seed:u64) -> u64:
    if storage_capacity(w,7,0) == 0:
        return 0
    data = alloc(w,5056)
    if data == 0:
        return 0
    hold = getw(w,80)
    id = so_slot_new(w,7)
    if id == 0:
        return terror(w,9)
    so_uput(w,so_slot_lookup(w,7,id)+16,hold)
    rng_seed(scratch,seed)
    rng_save_scratch(w,scratch,data)
    return id

def rng_clone_object(w:mutbuf,id:u64) -> u64:
    source = rng_object(w,id)
    if source == 0 or storage_capacity(w,7,0) == 0:
        return 0
    data = alloc(w,5056)
    if data == 0:
        return 0
    hold = getw(w,80)
    result = so_slot_new(w,7)
    if result == 0:
        return terror(w,9)
    so_uput(w,so_slot_lookup(w,7,result)+16,hold)
    move_within(w,object_address(w,data),object_address(w,source),5056)
    return result

def rng_drop_object(w:mutbuf,id:u64) -> u64:
    if rng_object(w,id) == 0:
        return 0
    id = resolve_handle(w,id)
    p = so_slot_lookup(w,7,id)
    if hold_release(w,so_uget(w,p+16)) == 0:
        return 0
    if so_slot_drop(w,7,id) != 0:
        return terror(w,9)
    return 1

def rng_interval(from_bits:u64,to_bits:u64) -> u64:
    if from_bits > 4294967295 or to_bits > 4294967295:
        return 0
    left = absf(from_bits)
    right = absf(to_bits)
    if left >= 2139095040 or right >= 2139095040 or ltf(to_bits,from_bits) != 0:
        return 0
    if (from_bits >> 31) != 0 and (to_bits >> 31) == 0:
        larger = choose(left>right,left,right)
        smaller = choose(left>right,right,left)
        if larger == 2139095039:
            if smaller > 1686110208:
                return 0
        else:
            if larger > 2130706431:
                if smaller > subf(2139095039,larger):
                    return 0
    return 1

def rng_axis_order(w:mutbuf,p:u64) -> u64:
    rank = getw(w,p)
    perm = 0
    i = 0
    while i < rank:
        perm = perm | ((rank-i-1) << (i*8))
        i = i+1
    i = 1
    while i < rank:
        anchor = i
        j = i
        done = 0
        while j > 0 and done == 0:
            j = j-1
            left = (perm >> (j*8)) & 255
            right = (perm >> (anchor*8)) & 255
            ls = getw(w,p+128+left*8)
            rs = getw(w,p+128+right*8)
            if ls != 0 and rs != 0:
                if ls > rs or (ls == rs and getw(w,p+96+left*8) > getw(w,p+96+right*8)):
                    difference = left ^ right
                    perm = perm ^ (difference << (j*8)) ^ (difference << (anchor*8))
                    anchor = j
                else:
                    if ls < rs:
                        done = 1
        i = i+1
    return perm

def rng_iteration_address(w:mutbuf,p:u64,perm:u64,index:u64) -> u64:
    offset = 0
    i = 0
    while i < getw(w,p):
        axis = (perm >> (i*8)) & 255
        dim = getw(w,p+96+axis*8)
        coordinate = index % dim
        index = index // dim
        offset = offset+coordinate*getw(w,p+128+axis*8)
        i = i+1
    return getw(w,p+16)+offset*4

def rng_uniform_tensor(w:mutbuf,scratch:mutbuf,tensor:u64,generator:u64,from_bits:u64,to_bits:u64,no_grad:u64) -> u64:
    t = public_node(w,tensor)
    data = rng_object(w,generator)
    if t == 0 or data == 0:
        return 0
    p = dptr(w,t)
    if no_grad > 1:
        return terror(w,7)
    if no_grad == 0 and getw(w,p+32) != 0:
        return terror(w,5)
    if rng_interval(from_bits,to_bits) == 0:
        return terror(w,7)
    n = getw(w,p+8)
    if n > 1000000:
        return terror(w,3)
    if version_can_bump(w,t,1) == 0:
        return 0
    if n != 0:
        i = 0
        while i < getw(w,p):
            if getw(w,p+96+i*8) > 1 and getw(w,p+128+i*8) == 0:
                return terror(w,2)
            i = i+1
        rng_load_scratch(w,scratch,data)
        perm = rng_axis_order(w,p)
        width = subf(to_bits,from_bits)
        i = 0
        while i < n:
            unit = rng_unit32(rng_next(scratch))
            value = fmaf(unit,width,from_bits)
            if value == to_bits or (absf(value) == 0 and absf(to_bits) == 0):
                value = from_bits
            put4(w,rng_iteration_address(w,p,perm,i),value)
            i = i+1
        rng_save_scratch(w,scratch,data)
    bump_version(w,t)
    return getw(w,p+192)

def rng_strided_view(s:readbuf,w:mutbuf,id:u64,config:u64,end:u64) -> u64:
    t = public_node(w,id)
    if t == 0:
        return 0
    if config < end or config > length(s) or 80 > length(s)-config or (config & 7) != 0:
        return terror(w,100)
    rank = getr(s,config)
    offset = getr(s,config+72)
    if rank > 4:
        return terror(w,2)
    p = dptr(w,t)
    sp = so_storage_live(w,getw(w,p+216))
    if sp == 0:
        return terror(w,1)
    size = so_uget(w,sp+24)//4
    if offset > 268435456:
        return terror(w,3)
    n = 1
    i = 0
    while i < 4:
        dim = getr(s,config+8+i*8)
        stride = getr(s,config+40+i*8)
        if i >= rank:
            if dim != 0 or stride != 0:
                return terror(w,2)
        else:
            if dim > 262144 or stride >= 9223372036854775808:
                return terror(w,3)
            if dim == 0:
                n = 0
        putw(w,2048+i*8,dim)
        i = i+1
    if n != 0:
        if offset > size:
            return terror(w,2)
        extent = 1
        i = 0
        while i < rank:
            dim = getr(s,config+8+i*8)
            stride = getr(s,config+40+i*8)
            if n > 16777216//dim:
                return terror(w,3)
            n = n*dim
            if extent > size-offset:
                return terror(w,2)
            if stride != 0:
                if dim-1 > (size-offset-extent)//stride:
                    return terror(w,2)
                extent = extent+(dim-1)*stride
            i = i+1
        if extent > size-offset:
            return terror(w,2)
    result = descriptor(w,rank,2048,0,0)
    if result == 0:
        return 0
    q = dptr(w,result)
    if bind_alias(w,result,t) == 0:
        return 0
    putw(w,q+16,9223372036854775808+so_uget(w,sp+16)+offset*4)
    putw(w,q+64,getw(w,p+64))
    putw(w,q+232,offset*4)
    tp = so_slot_lookup(w,1,getw(w,q+192))
    so_uput(w,tp+32,offset*4)
    remaining = 0
    if offset <= size:
        remaining = (size-offset)*4
    so_uput(w,tp+40,remaining)
    i = 0
    while i < rank:
        putw(w,q+128+i*8,getr(s,config+40+i*8))
        i = i+1
    return result

def rng_command(s:readbuf,o:mutbuf,w:mutbuf,scratch:mutbuf,cp:u64,at:u64,end:u64) -> u64:
    code = getr(s,cp)
    a = getr(s,cp+8)
    b = getr(s,cp+16)
    c = getr(s,cp+24)
    d = getr(s,cp+32)
    result = 0
    payload = 0
    first = getw(w,8)
    if code < 60 or code > 69:
        terror(w,4)
    if getr(s,cp+48) != 0 or getr(s,cp+56) != 0:
        terror(w,7)
    if code != 68 and getr(s,cp+40) != 0:
        terror(w,7)
    if code != 68 and d != 0:
        terror(w,7)
    if code != 64 and code != 68 and code != 69 and c != 0:
        terror(w,7)
    if code == 60 or code == 62 or code == 63 or code == 65 or code == 66:
        if b != 0:
            terror(w,7)
    if getw(w,0) == 0:
        if code == 60:
            result = rng_create_object(w,scratch,a)
        if code == 65:
            result = rng_clone_object(w,a)
        if code == 66:
            result = rng_drop_object(w,a)
        if code == 69:
            if c != 80:
                terror(w,100)
            else:
                result = rng_strided_view(s,w,a,b,end)
                if getw(w,0) == 0:
                    result = publish_tensor(w,result)
                    finish_command(w,first)
        if code == 68:
            result = rng_uniform_tensor(w,scratch,a,b,c,d,getr(s,cp+40))
        if code == 61 or code == 62 or code == 63 or code == 64 or code == 67:
            address = rng_object(w,a)
            if address != 0:
                if code == 61:
                    rng_seed(scratch,b)
                    result = rng_save_scratch(w,scratch,address)
                if code == 62:
                    result = getw(w,address)
                if code == 63:
                    if length(o)-at-32 < 5056:
                        terror(w,3)
                    else:
                        copy_from_mut(o,w,at+32,object_address(w,address),5056)
                        result = 5056
                        payload = 5056
                if code == 64:
                    if b < end or b > length(s) or c != 5056 or 5056 > length(s)-b or (b & 7) != 0:
                        terror(w,100)
                    else:
                        if rng_restore(s,scratch,b) != 0:
                            terror(w,7)
                        else:
                            result = rng_save_scratch(w,scratch,address)
                if code == 67:
                    if b > 1000000:
                        terror(w,3)
                    else:
                        rng_load_scratch(w,scratch,address)
                        i = 0
                        while i < b:
                            rng_next(scratch)
                            i = i+1
                        result = rng_save_scratch(w,scratch,address)
    status = getw(w,0)
    if status != 0:
        result = 0
        payload = 0
    putw(o,at,code)
    putw(o,at+8,status)
    putw(o,at+16,result)
    putw(o,at+24,payload)
    return at+32+payload

def rng_state_canonical(w:mutbuf,address:u64) -> u64:
    left = get4(w,address+8)
    next = getw(w,address+16)
    if get4(w,address+12) != 1 or left == 0 or left > 624 or next > 624:
        return 0
    if (left != 1 or next != 0) and (next == 0 or left+next != 625):
        return 0
    if getw(w,address+5016) != 0 or getw(w,address+5032) != 0 or get4(w,address+5044) != 0:
        return 0
    double_flag = get4(w,address+5040)
    float_flag = get4(w,address+5052)
    if double_flag > 1 or float_flag > 1:
        return 0
    if double_flag == 0 and getw(w,address+5024) != 0:
        return 0
    if float_flag == 0 and get4(w,address+5048) != 0:
        return 0
    i = 0
    while i < 624:
        if getw(w,address+24+i*8) > 4294967295:
            return 0
        i = i+1
    return 1
