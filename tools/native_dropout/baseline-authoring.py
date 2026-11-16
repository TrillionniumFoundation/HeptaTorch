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
    if getw(w, 120) == 18446744073709551615:
        return terror(w, 8)
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
    memory_fill(w, p, 576, 0)
    putw(w, p, rank)
    putw(w, p + 8, n)
    putw(w, p + 32, requires)
    putw(w, p + 64, choose(own != 0, slot, 0))
    putw(w, p + 200, 2)
    putw(w, p + 304, 1)
    putw(w, 120, getw(w, 120) + 1)
    putw(w, p + 320, getw(w, 120))
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
    if getw(w, q + 232) <= so_uget(w, sp + 24):
        remaining = so_uget(w, sp + 24) - getw(w, q + 232)
    so_uput(w, tp + 40, remaining)
    putw(w, p + 216, sid)
    putw(w, p + 224, vid)
    putw(w, p + 232, getw(w, q + 232))
    return 1

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

def rng_strided_view(s: readbuf, w: mutbuf, id: u64, config: u64, end: u64) -> u64:
    t = public_node(w, id)
    if t == 0:
        return 0
    if config < end or config > length(s) or 80 > length(s) - config or (config & 7 != 0):
        return terror(w, 100)
    rank = getr(s, config)
    offset = getr(s, config + 72)
    if rank > 4:
        return terror(w, 2)
    p = dptr(w, t)
    sp = so_storage_live(w, getw(w, p + 216))
    if sp == 0:
        return terror(w, 1)
    size = so_uget(w, sp + 24) // 4
    if offset > 268435456:
        return terror(w, 3)
    n = 1
    i = 0
    while i < 4:
        dim = getr(s, config + 8 + i * 8)
        stride = getr(s, config + 40 + i * 8)
        if i >= rank:
            if dim != 0 or stride != 0:
                return terror(w, 2)
        else:
            if dim > 262144 or stride >= 9223372036854775808:
                return terror(w, 3)
            if dim == 0:
                n = 0
        putw(w, 2048 + i * 8, dim)
        i = i + 1
    if n != 0:
        if offset > size:
            return terror(w, 2)
        extent = 1
        i = 0
        while i < rank:
            dim = getr(s, config + 8 + i * 8)
            stride = getr(s, config + 40 + i * 8)
            if n > 16777216 // dim:
                return terror(w, 3)
            n = n * dim
            if extent > size - offset:
                return terror(w, 2)
            if stride != 0:
                if dim - 1 > (size - offset - extent) // stride:
                    return terror(w, 2)
                extent = extent + (dim - 1) * stride
            i = i + 1
        if extent > size - offset:
            return terror(w, 2)
    result = descriptor(w, rank, 2048, 0, 0)
    if result == 0:
        return 0
    q = dptr(w, result)
    if bind_alias(w, result, t) == 0:
        return 0
    putw(w, q + 16, 9223372036854775808 + so_uget(w, sp + 16) + offset * 4)
    putw(w, q + 64, getw(w, p + 64))
    putw(w, q + 232, offset * 4)
    tp = so_slot_lookup(w, 1, getw(w, q + 192))
    so_uput(w, tp + 32, offset * 4)
    remaining = 0
    if offset <= size:
        remaining = (size - offset) * 4
    so_uput(w, tp + 40, remaining)
    i = 0
    while i < rank:
        putw(w, q + 128 + i * 8, getr(s, config + 40 + i * 8))
        i = i + 1
    return result

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
        if code >= 91 and code <= 94:
            next = gradient_object_command(s, o, w, 32 + i * 64, at)
        elif code >= 71 and code <= 79 or code == 90 or code == 95:
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
        if getw(w, 0) == 0 and getw(w, 3968) != 0:
            graph_collect_since(w, getw(w, 8))
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

def gradient_assign(w: mutbuf, t: u64, g: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if g == 0:
        return gradient_unbind(w, t)
    if valid(w, g) == 0:
        return 0
    if t == g or gradient_shape_matches(w, t, g) == 0:
        return terror(w, 2)
    if gradient_binding(w, t) == g:
        return 1
    if gradient_cycle_check(w, t, g) == 0:
        return 0
    incarnation = getw(w, dptr(w, g) + 320)
    if gradient_reference_retain(w, g, incarnation) == 0:
        return 0
    if gradient_unbind(w, t) == 0:
        return 0
    putw(w, p + 384, g)
    putw(w, p + 392, incarnation)
    putw(w, p + 360, 1)
    return 1

def gradient_shape_matches(w: mutbuf, t: u64, g: u64) -> u64:
    p = dptr(w, t)
    q = dptr(w, g)
    rank = getw(w, p)
    if getw(w, q) != rank:
        return 0
    j = 0
    while j < rank:
        if getw(w, p + 96 + j * 8) != getw(w, q + 96 + j * 8):
            return 0
        j = j + 1
    return 1

def optimizer_fill_group(s: readbuf, w: mutbuf, blob: u64, group: u64, offset: u64, first: u64) -> u64:
    gp = blob + 64 + group * 64
    mask = getr(s, offset + 8)
    count = getr(s, offset)
    j = 0
    while j < 6:
        value = getw(w, blob + 16 + j * 8)
        if mask & 1 << j != 0:
            value = getr(s, offset + 16 + j * 8)
        putw(w, gp + j * 8, value)
        j = j + 1
    putw(w, gp + 48, first)
    putw(w, gp + 56, count)
    i = 0
    duplicates = 0
    while i < count:
        t = public_node(w, getr(s, offset + 64 + i * 8))
        if t == 0:
            return 0
        p = dptr(w, t)
        if retained_parameter_blocker(w, t) != 0:
            return terror(w, 5)
        incarnation = getw(w, p + 320)
        index = first + i
        key = index
        j = 0
        while j < index:
            previous = optimizer_entry_at(w, blob, j)
            if getw(w, previous) == t and getw(w, previous + 8) == incarnation:
                if j < first:
                    return terror(w, 7)
                key = getw(w, previous + 16)
                j = index
            else:
                j = j + 1
        ep = optimizer_entry_at(w, blob, index)
        putw(w, ep, t)
        putw(w, ep + 8, incarnation)
        putw(w, ep + 16, key)
        if key != index:
            duplicates = duplicates + 1
        if private_tensor_retain(w, t, incarnation) == 0:
            return 0
        i = i + 1
    return duplicates + 1

def owned_adam_fill_group(s: readbuf, w: mutbuf, blob: u64, group: u64, offset: u64, first: u64) -> u64:
    gp = blob + 96 + group * 80
    mask = getr(s, offset + 8)
    count = getr(s, offset)
    j = 0
    while j < 8:
        value = getw(w, blob + 32 + j * 8)
        if mask & 1 << j != 0:
            value = getr(s, offset + 16 + j * 8)
        putw(w, gp + j * 8, value)
        j = j + 1
    putw(w, gp + 64, first)
    putw(w, gp + 72, count)
    i = 0
    duplicates = 0
    while i < count:
        t = public_node(w, getr(s, offset + 80 + i * 8))
        if t == 0:
            return 0
        p = dptr(w, t)
        if retained_parameter_blocker(w, t) != 0:
            return terror(w, 5)
        incarnation = getw(w, p + 320)
        index = first + i
        key = index
        j = 0
        while j < index:
            previous = owned_adam_entry_at(w, blob, j)
            if getw(w, previous) == t and getw(w, previous + 8) == incarnation:
                if j < first:
                    return terror(w, 7)
                key = getw(w, previous + 16)
                j = index
            else:
                j = j + 1
        ep = owned_adam_entry_at(w, blob, index)
        putw(w, ep, t)
        putw(w, ep + 8, incarnation)
        putw(w, ep + 16, key)
        if key != index:
            duplicates = duplicates + 1
        if private_tensor_retain(w, t, incarnation) == 0:
            return 0
        i = i + 1
    return duplicates + 1

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

def incoming_describe(w: mutbuf, t: u64, parent: u64, edge: u64) -> u64:
    memory_fill(w, 3808, 88, 0)
    p = dptr(w, t)
    op = getw(w, p + 40)
    putw(w, 3840, parent)
    putw(w, 3848, 2)
    incoming_contiguous_layout(w, parent, 3856)
    if op == 1 or (op == 2 and edge == 0) or op == 17 or ((op == 39 or op == 40) and edge == 0):
        if incoming_same_shape(w, t, parent) != 0:
            return incoming_describe_alias(w, t, getw(w, p + 472), getw(w, p + 488))
        return 1
    if op == 10:
        if incoming_describe_alias(w, t, 2, parent) == 0:
            return 0
        a = getw(w, p + 176)
        b = getw(w, p + 184)
        putw(w, 3856 + a * 8, getw(w, p + 496 + b * 8))
        putw(w, 3856 + b * 8, getw(w, p + 496 + a * 8))
        return 1
    if op == 16:
        if incoming_describe_alias(w, t, 2, parent) == 0:
            return 0
        if incoming_reshape_layout(w, t, parent, 3856) != 0:
            return 1
        putw(w, 3808, 0)
        putw(w, 3816, 0)
        putw(w, 3824, 0)
        putw(w, 3832, 0)
        return incoming_contiguous_layout(w, parent, 3856)
    if op == 7:
        if incoming_describe_alias(w, t, 2, parent) == 0:
            return 0
        j = 0
        while j < getw(w, dptr(w, parent)):
            putw(w, 3856 + j * 8, 0)
            j = j + 1
        return 1
    if op == 37 or op == 38:
        return incoming_describe_cat(w, t, parent, edge)
    if op == 41:
        return incoming_zero_layout(w, t, 3856)
    if op == 26 or op == 27:
        putw(w, 3888, 1)
        return 1
    if op == 2 or op == 3 or op == 4 or (op == 5) or (op == 6) or (op >= 11 and op <= 15) or (op == 20) or (op == 21) or (op == 22) or (op == 25) or (op == 39) or (op == 40):
        return 1
    putw(w, 3848, 1)
    return 1

def ensure_backward_work(w: mutbuf, limit: u64, epoch: u64) -> u64:
    if incoming_mark_preserve_targets(w, limit, epoch) == 0:
        return 0
    i = 1
    while i <= limit:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 168) == epoch:
            if getw(w, p + 448) != 0 or getw(w, p + 480) != 0 or getw(w, p + 528) != 0:
                return terror(w, 9)
            if getw(w, p + 40) == 0 and getw(w, p + 32) != 0 and (getw(w, p + 384) == 0):
                if gradient_new(w, i) == 0:
                    return 0
            if getw(w, p + 256) == 0:
                address = alloc(w, getw(w, p + 8) * 4)
                if address == 0:
                    return 0
                putw(w, p + 160, address)
                putw(w, p + 256, getw(w, 80))
            if storage_capacity(w, 3, 0) == 0:
                return 0
            vid = so_slot_new(w, 3)
            if vid == 0:
                return terror(w, 9)
            vp = so_slot_lookup(w, 3, vid)
            so_uput(w, vp + 24, 1)
            putw(w, p + 480, vid)
            hp = so_slot_lookup(w, 6, getw(w, p + 256))
            if hp == 0:
                return terror(w, 9)
            putw(w, p + 536, so_uget(w, hp + 16))
            if getw(w, p + 544) == 1:
                if incoming_reserve_preserve_storage(w, i) == 0:
                    return 0
        i = i + 1
    return 1

def graph_audit(w: mutbuf) -> u64:
    status = 0
    budget = 0
    i = 1
    while i < getw(w, 8) and status == 0:
        p = dptr(w, i)
        if getw(w, p + 304) != 0:
            count = gradient_parent_count(w, i)
            if count > 16777216 - budget:
                status = 3
            else:
                budget = budget + count
                j = 0
                while j < count and status == 0:
                    parent = graph_parent_at(w, i, j)
                    flag = graph_edge_enabled(w, i, j)
                    if flag > 1 or parent >= i:
                        status = 9
                    elif parent != 0:
                        q = dptr(w, parent)
                        if getw(w, q + 304) == 0:
                            status = 9
                        else:
                            putw(w, q + 376, getw(w, q + 376) + 1)
                            if flag != 0 and getw(w, q + 40) == 0:
                                putw(w, q + 368, getw(w, q + 368) + 1)
                    elif flag != 0:
                        status = 9
                    j = j + 1
        i = i + 1
    i = 1
    while i < getw(w, 8):
        p = dptr(w, i)
        if status == 0 and getw(w, p + 304) != 0:
            if getw(w, p + 88) != getw(w, p + 368) or getw(w, p + 208) != getw(w, p + 376) + getw(w, p + 312) + getw(w, p + 400):
                status = 9
        putw(w, p + 368, 0)
        putw(w, p + 376, 0)
        i = i + 1
    return status
