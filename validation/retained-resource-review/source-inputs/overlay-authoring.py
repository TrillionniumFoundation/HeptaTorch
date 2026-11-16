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
    retained_bytes = (t + 1) * 8
    queue = alloc(w, queue_bytes + retained_bytes + stage_bytes)
    if queue == 0:
        return 0
    queue_owner = getw(w, 80)
    memory_fill(w, queue, queue_bytes, 0)
    putw(w, queue + 8, t)
    if ensure_backward_work(w, t, epoch) == 0:
        return 0
    retained_table = queue + queue_bytes
    if retained_reserve(w, retained_table, t, epoch) == 0:
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
            retained_apply(w, retained_table, current)
        if getw(w, 0) == 0:
            partial_check_node(w, current)
        if getw(w, 0) == 0:
            gradient_apply(w, current, queue + queue_bytes + retained_bytes)
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
    if retained_release_reservations(w, retained_table, t) == 0:
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
    retained_bytes = (limit + 1) * 8
    scratch = alloc(w, stage_bytes + 1 + retained_bytes)
    if scratch == 0:
        return 0
    scratch_owner = getw(w, 80)
    if retained_reserve(w, scratch, limit, epoch) == 0:
        return 0
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
                if retained_apply(w, scratch, current) == 0:
                    return 0
                gradient_apply(w, current, scratch + retained_bytes)
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
            gradient_apply(w, current, scratch + retained_bytes)
        i = i + 1
    if gradient_clear_marks(w, limit, epoch) == 0:
        return 0
    if release_backward_scratch(w, limit, epoch) == 0:
        return 0
    if retained_release_reservations(w, scratch, limit) == 0:
        return 0
    return hold_release(w, scratch_owner)

def tgrad(w: mutbuf, t: u64, index: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if retained_parameter_blocker(w, t) != 0:
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
    if retained_parameter_blocker(w, t) != 0:
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
    if retained_parameter_blocker(w, t) != 0:
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
        elif code == 93:
            result = retained_enable(w, resolve_handle(w, a))
        elif code == 94:
            result = retained_query(w, resolve_handle(w, a))
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

def optimizer_audit_entry(w: mutbuf, blob: u64, i: u64, first: u64) -> u64:
    ep = optimizer_entry_at(w, blob, i)
    node = getw(w, ep)
    incarnation = getw(w, ep + 8)
    p = private_tensor(w, node, incarnation)
    if p == 0 or retained_parameter_blocker(w, node) != 0:
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

def owned_adam_audit_entry(w: mutbuf, blob: u64, i: u64, first: u64) -> u64:
    ep = owned_adam_entry_at(w, blob, i)
    node = getw(w, ep)
    incarnation = getw(w, ep + 8)
    p = private_tensor(w, node, incarnation)
    if p == 0 or retained_parameter_blocker(w, node) != 0:
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
                    if kind == 1 and (so_uget(w, p + 56) > 1 or so_uget(w, p + 48) == 0 or so_uget(w, p + 48) >= getw(w, 8)):
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

def retained_row(w:mutbuf, t:u64) -> u64:
    if t == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+304) == 0 or getw(w,p+240) == 0:
        return 0
    row = so_slot_lookup(w,1,getw(w,p+192))
    if row == 0 or so_uget(w,row+48) != t:
        return terror(w,9)
    return row

def retained_enabled(w:mutbuf, t:u64) -> u64:
    row = retained_row(w,t)
    if row == 0:
        return 0
    flags = so_uget(w,row+56)
    if flags > 3:
        return terror(w,9)
    return flags & 1

def retained_parameter_blocker(w:mutbuf, t:u64) -> u64:
    if getw(w,dptr(w,t)+40) == 0:
        return 0
    return choose(retained_enabled(w,t) != 0,0,1)

def retained_temporary_root(w:mutbuf, t:u64) -> u64:
    row = retained_row(w,t)
    if row == 0:
        return 0
    return so_uget(w,row+56) & 2

def gradient_gc_enqueue(w:mutbuf, t:u64) -> u64:
    if retained_temporary_root(w,t) != 0:
        return 1
    return retained_gc_enqueue_original(w,t)

def tensor_row_release(w:mutbuf, t:u64) -> u64:
    if retained_temporary_root(w,t) != 0:
        return 1
    return retained_row_release_original(w,t)

def publish_tensor(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    row = retained_row(w,t)
    if row == 0:
        return terror(w,1)
    flags = so_uget(w,row+56)
    if flags > 1:
        return terror(w,9)
    result = retained_publish_original(w,t)
    if result == 0:
        return 0
    row = retained_row(w,t)
    if row == 0:
        return terror(w,9)
    so_uput(w,row+56,flags)
    return result

def retained_enable(w:mutbuf, id:u64) -> u64:
    t = public_node(w,id)
    if t == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+32) == 0:
        return terror(w,5)
    if getw(w,p+40) == 0:
        return 1
    row = retained_row(w,t)
    if row == 0 or so_uget(w,row+56) > 1:
        return terror(w,9)
    so_uput(w,row+56,1)
    return 1

def retained_query(w:mutbuf, id:u64) -> u64:
    t = public_node(w,id)
    if t == 0:
        return 0
    return retained_enabled(w,t)

def retained_reserve_one(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    g = descriptor(w,getw(w,p),p+96,0,1)
    if g == 0:
        return 0
    row = retained_row(w,g)
    if row == 0 or so_uget(w,row+56) != 0:
        return terror(w,9)
    if node_retain(w,g) == 0:
        return 0
    so_uput(w,row+56,2)
    putw(w,dptr(w,g)+200,0)
    return g

def retained_release_one(w:mutbuf, g:u64) -> u64:
    row = retained_row(w,g)
    if row == 0 or so_uget(w,row+56) != 2:
        return terror(w,9)
    so_uput(w,row+56,0)
    return node_release(w,g)

def retained_reserve(w:mutbuf, table:u64, limit:u64, epoch:u64) -> u64:
    memory_fill(w,table,(limit+1)*8,0)
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch and getw(w,p+40) != 0:
            if retained_enabled(w,i) != 0:
                g = retained_reserve_one(w,i)
                if g == 0:
                    return 0
                putw(w,table+i*8,g)
        i = i+1
    return choose(getw(w,0) == 0,1,0)

def retained_release_reservations(w:mutbuf, table:u64, limit:u64) -> u64:
    i = 1
    while i <= limit:
        g = getw(w,table+i*8)
        if g != 0:
            putw(w,table+i*8,0)
            if retained_release_one(w,g) == 0:
                return 0
        i = i+1
    return graph_collect_since(w,1)

def retained_apply(w:mutbuf, table:u64, t:u64) -> u64:
    g = getw(w,table+t*8)
    if g == 0:
        return 1
    if retained_enabled(w,t) == 0:
        putw(w,table+t*8,0)
        return retained_release_one(w,g)
    p = dptr(w,t)
    if getw(w,p+40) == 0 or getw(w,p+448) == 0 or getw(w,p+336) == 0:
        return terror(w,9)
    old = gradient_binding(w,t)
    if getw(w,0) != 0:
        return 0
    if old != 0:
        if retained_sum_layout(w,g,old,t) == 0:
            return 0
    i = 0
    while i < getw(w,p+8):
        value = get4(w,incoming_address(w,t,i))
        if old != 0:
            value = addf(tget(w,old,i),value)
        put4(w,taddr(w,g,i),value)
        i = i+1
    if gradient_assign(w,t,g) == 0:
        return 0
    putw(w,table+t*8,0)
    return retained_release_one(w,g)


def retained_compare_axes(w:mutbuf, t:u64, old:u64, left:u64, right:u64) -> u64:
    p = dptr(w,t)
    q = dptr(w,old)
    operand = 0
    while operand < 2:
        layout = q+128
        if operand == 1:
            layout = p+496
        ls = getw(w,layout+left*8) << 2
        rs = getw(w,layout+right*8) << 2
        if ls != 0 and rs != 0:
            ordered_left = ls ^ 9223372036854775808
            ordered_right = rs ^ 9223372036854775808
            if ordered_left < ordered_right:
                return 1
            if ordered_left > ordered_right:
                return 2
            if getw(w,p+96+left*8) > getw(w,p+96+right*8):
                return 2
        operand = operand+1
    return 0

def retained_axis_order(w:mutbuf, t:u64, old:u64) -> u64:
    rank = getw(w,dptr(w,t))
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
            comparison = retained_compare_axes(w,t,old,left,right)
            if comparison == 2:
                difference = left ^ right
                perm = perm ^ (difference << (j*8)) ^ (difference << (anchor*8))
                anchor = j
            if comparison == 1:
                done = 1
        i = i+1
    return perm

def retained_incoming_channels_last(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p) != 4:
        return 0
    expected = 1
    i = 0
    while i < 4:
        axis = (131841 >> (i*8)) & 255
        size = getw(w,p+96+axis*8)
        if size > 1 and getw(w,p+496+axis*8) != expected:
            return 0
        expected = expected*size
        i = i+1
    return 1

def retained_sum_layout(w:mutbuf, g:u64, old:u64, t:u64) -> u64:
    out = dptr(w,g)
    p = dptr(w,t)
    q = dptr(w,old)
    rank = getw(w,p)
    if is_contiguous(w,old) != 0 and incoming_contiguous(w,t,p+496) != 0:
        return 1
    if pointwise_channels_last(w,old) != 0 and retained_incoming_channels_last(w,t) != 0:
        return pointwise_write_order(w,out,131841,0)
    if is_dense(w,old) != 0 and incoming_dense(w,t,p+496) != 0:
        same = 1
        i = 0
        while i < rank:
            if getw(w,q+128+i*8) != getw(w,p+496+i*8):
                same = 0
            i = i+1
        if same != 0:
            i = 0
            while i < rank:
                putw(w,out+128+i*8,getw(w,p+496+i*8))
                i = i+1
            return 1
    order = retained_axis_order(w,t,old)
    inverted = 1
    i = 0
    while i < rank:
        if ((order >> (i*8)) & 255) != rank-i-1:
            inverted = 0
        i = i+1
    if inverted != 0:
        return 1
    return pointwise_write_order(w,out,order,1)
