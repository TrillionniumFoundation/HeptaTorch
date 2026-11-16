def partial_execute(w: mutbuf, o: mutbuf, info: u64, t: u64, seed: u64, retain: u64) -> u64:
    if partial_admit(w, t, seed) == 0:
        return 0
    epoch = getw(w, 24) + 1
    if epoch == 0:
        return terror(w, 8)
    putw(w, 24, epoch)
    partial_mark(w, t, epoch)
    queue_bytes = 24 + (t + 1) * 16
    stage_bytes = gradient_workspace_size(w, t, epoch)
    if getw(w, 0) != 0:
        return 0
    queue = alloc(w, queue_bytes + stage_bytes)
    if queue == 0:
        return 0
    queue_owner = getw(w, 80)
    memory_fill(w, queue, queue_bytes, 0)
    putw(w, queue + 8, t)
    if ensure_backward_work(w, t, epoch) == 0:
        return 0
    i = 1
    while i <= t:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 168) == epoch:
            if partial_edges(w, queue, i, 0) == 0:
                return 0
            memory_fill(w, getw(w, p + 160), getw(w, p + 8) * 4, 0)
            putw(w, p + 336, 0)
        i = i + 1
    p = dptr(w, t)
    if seed == 0:
        put4(w, getw(w, p + 160), 1065353216)
    else:
        i = 0
        while i < getw(w, p + 8):
            put4(w, getw(w, p + 160) + i * 4, tget(w, seed, i))
            i = i + 1
    if incoming_seed(w, t, seed) == 0:
        return 0
    putw(w, p + 336, 1)
    if partial_push(w, queue, t) == 0:
        return 0
    error = 0
    failed_op = 0
    completed = 0
    leaves = 0
    while getw(w, queue) != 0 and error == 0:
        current = partial_pop(w, queue)
        if getw(w, 0) == 0:
            partial_check_node(w, current)
        if getw(w, 0) == 0:
            gradient_apply(w, current, queue + queue_bytes)
        if getw(w, 0) == 0 and retain == 0:
            release_backward_saved(w, current)
        if getw(w, 0) == 0:
            completed = completed + 1
            if getw(w, dptr(w, current) + 40) == 0:
                leaves = leaves + 1
            partial_edges(w, queue, current, 1)
        error = getw(w, 0)
        if error != 0:
            failed_op = getw(w, dptr(w, current) + 40)
    putw(w, 0, 0)
    if gradient_clear_marks(w, t, epoch) == 0:
        return 0
    if release_backward_scratch(w, t, epoch) == 0:
        return 0
    if hold_release(w, queue_owner) == 0:
        return 0
    putw(o, info, error)
    putw(o, info + 8, completed)
    putw(o, info + 16, leaves)
    putw(o, info + 24, failed_op)
    return error

def backward_execute(w: mutbuf, t: u64, seed: u64, retain: u64) -> u64:
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
    if epoch == 0:
        return terror(w, 8)
    putw(w, 24, epoch)
    if mark_graph(w, t, epoch) == 0:
        return 0
    limit = t
    if ensure_backward_work(w, limit, epoch) == 0:
        return 0
    stage_bytes = gradient_workspace_size(w, limit, epoch)
    if getw(w, 0) != 0:
        return 0
    scratch = alloc(w, stage_bytes + 1)
    if scratch == 0:
        return 0
    scratch_owner = getw(w, 80)
    i = 1
    while i <= limit:
        current = live_id(w, i)
        pi = dptr(w, current)
        if getw(w, pi + 168) == epoch:
            memory_fill(w, getw(w, pi + 160), getw(w, pi + 8) * 4, 0)
            putw(w, pi + 336, 0)
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
    if incoming_seed(w, t, seed) == 0:
        return 0
    putw(w, p + 336, 1)
    i = limit
    while i > 0:
        current = live_id(w, i)
        pi = dptr(w, current)
        if getw(w, pi + 168) == epoch:
            op = getw(w, pi + 40)
            if op != 0:
                gradient_apply(w, current, scratch)
            if getw(w, 0) != 0:
                return 0
            if retain == 0:
                if release_backward_saved(w, current) == 0:
                    return 0
        i = i - 1
    i = 1
    while i <= limit:
        current = live_id(w, i)
        pi = dptr(w, current)
        if getw(w, pi + 168) == epoch and getw(w, pi + 40) == 0:
            gradient_apply(w, current, scratch)
        i = i + 1
    if gradient_clear_marks(w, limit, epoch) == 0:
        return 0
    if hold_release(w, scratch_owner) == 0:
        return 0
    return release_backward_scratch(w, limit, epoch)

def tgrad(w: mutbuf, t: u64, index: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if getw(w, p + 40) != 0:
        return terror(w, 5)
    if index >= getw(w, p + 8):
        return terror(w, 7)
    if getw(w, p + 360) == 0:
        return 0
    g = gradient_binding(w, t)
    if g == 0:
        return 0
    return tget(w, g, index)

def zero_grad(w: mutbuf, t: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if getw(w, p + 40) != 0:
        return terror(w, 5)
    if getw(w, p + 360) == 0:
        return 1
    g = gradient_binding(w, t)
    if g == 0 or version_can_bump(w, g, 1) == 0:
        return 0
    if gradient_detach_inplace(w, g) == 0:
        return 0
    i = 0
    while i < getw(w, p + 8):
        put4(w, taddr(w, g, i), 0)
        i = i + 1
    return bump_version(w, g)

def zero_grad_none(w: mutbuf, t: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    if getw(w, dptr(w, t) + 40) != 0:
        return terror(w, 5)
    return gradient_unbind(w, t)

def gradient_object_command(s: readbuf, o: mutbuf, w: mutbuf, cp: u64, at: u64) -> u64:
    code = getr(s, cp)
    a = getr(s, cp + 8)
    b = getr(s, cp + 16)
    first = getw(w, 8)
    result = 0
    field = 3
    while field < 8:
        if getr(s, cp + field * 8) != 0:
            terror(w, 7)
        field = field + 1
    if code != 92 and b != 0:
        terror(w, 7)
    if getw(w, 0) == 0:
        if code == 91:
            result = gradient_get_public(w, a)
        elif code == 92:
            t = public_node(w, a)
            g = public_optional(w, b)
            if t != 0 and getw(w, 0) == 0:
                result = gradient_assign(w, t, g)
        else:
            terror(w, 4)
    if getw(w, 0) == 0:
        finish_command(w, first)
    status = getw(w, 0)
    if status != 0:
        result = 0
    putw(o, at, code)
    putw(o, at + 8, status)
    putw(o, at + 16, result)
    putw(o, at + 24, 0)
    return at + 32

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
        if getw(w, p + 40) != 0:
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

def optimizer_audit_entry(w: mutbuf, blob: u64, i: u64, first: u64) -> u64:
    ep = optimizer_entry_at(w, blob, i)
    node = getw(w, ep)
    incarnation = getw(w, ep + 8)
    p = private_tensor(w, node, incarnation)
    if p == 0 or getw(w, p + 40) != 0:
        return 9
    if optimizer_audit_reference(w, node, incarnation) != 0:
        return 9
    key_index = getw(w, ep + 16)
    if key_index < first or key_index > i:
        return 9
    key = optimizer_entry_at(w, blob, key_index)
    if getw(w, key) != node or getw(w, key + 8) != incarnation or getw(w, key + 16) != key_index:
        return 9
    if key_index != i:
        if getw(w, ep + 24) != 0 or getw(w, ep + 32) != 0 or getw(w, ep + 40) != 0:
            return 9
    else:
        j = 0
        while j < i:
            other = optimizer_entry_at(w, blob, j)
            if getw(w, other) == node and getw(w, other + 8) == incarnation:
                return 9
            j = j + 1
        present = getw(w, key + 24)
        if present > 1:
            return 9
        if present == 0:
            if getw(w, key + 32) != 0 or getw(w, key + 40) != 0:
                return 9
        else:
            buffer = getw(w, key + 32)
            bp = private_tensor(w, buffer, getw(w, key + 40))
            if bp == 0 or getw(w, bp + 40) != 0 or getw(w, bp) != getw(w, p):
                return 9
            j = 0
            while j < getw(w, p):
                if getw(w, bp + 96 + j * 8) != getw(w, p + 96 + j * 8):
                    return 9
                j = j + 1
            if optimizer_audit_reference(w, node, incarnation) != 0 or optimizer_audit_reference(w, buffer, getw(w, key + 40)) != 0:
                return 9
    return choose(key_index == i, 1, 2)

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
        if getw(w, p + 40) != 0:
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

def owned_adam_audit_entry(w: mutbuf, blob: u64, i: u64, first: u64) -> u64:
    ep = owned_adam_entry_at(w, blob, i)
    node = getw(w, ep)
    incarnation = getw(w, ep + 8)
    p = private_tensor(w, node, incarnation)
    if p == 0 or getw(w, p + 40) != 0:
        return 9
    if optimizer_audit_reference(w, node, incarnation) != 0:
        return 9
    key_index = getw(w, ep + 16)
    if key_index < first or key_index > i:
        return 9
    key = owned_adam_entry_at(w, blob, key_index)
    if getw(w, key) != node or getw(w, key + 8) != incarnation or getw(w, key + 16) != key_index:
        return 9
    if key_index != i:
        j = 24
        while j < 80:
            if getw(w, ep + j) != 0:
                return 9
            j = j + 8
    else:
        j = 0
        while j < i:
            other = owned_adam_entry_at(w, blob, j)
            if getw(w, other) == node and getw(w, other + 8) == incarnation:
                return 9
            j = j + 1
        if owned_adam_audit_state(w, key, p) != 0:
            return 9
    return choose(key_index == i, 1, 2)

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
