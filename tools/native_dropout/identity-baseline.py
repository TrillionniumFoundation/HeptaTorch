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

def partial_push(w: mutbuf, queue: u64, t: u64) -> u64:
    limit = getw(w, queue + 8)
    count = getw(w, queue) + 1
    if count > limit:
        return terror(w, 8)
    putw(w, queue, count)
    heap = queue + 24 + (limit + 1) * 8
    key = t
    if getw(w, dptr(w, t) + 40) == 0:
        key = key | 9223372036854775808
    i = count
    moving = 1
    while i > 1 and moving != 0:
        parent = i // 2
        previous = getw(w, heap + parent * 8)
        if previous < key:
            putw(w, heap + i * 8, previous)
            i = parent
        else:
            moving = 0
    putw(w, heap + i * 8, key)
    return 1

def partial_pop(w: mutbuf, queue: u64) -> u64:
    count = getw(w, queue)
    if count == 0:
        return terror(w, 8)
    heap = queue + 24 + (getw(w, queue + 8) + 1) * 8
    result = getw(w, heap + 8)
    key = getw(w, heap + count * 8)
    count = count - 1
    putw(w, queue, count)
    i = 1
    moving = 1
    while i * 2 <= count and moving != 0:
        child = i * 2
        if child + 1 <= count:
            if getw(w, heap + (child + 1) * 8) > getw(w, heap + child * 8):
                child = child + 1
        value = getw(w, heap + child * 8)
        if key < value:
            putw(w, heap + i * 8, value)
            i = child
        else:
            moving = 0
    if count != 0:
        putw(w, heap + i * 8, key)
    return result & 9223372036854775807

def partial_check_node(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    if getw(w, p + 40) == 0:
        return 1
    if getw(w, p + 328) != 0:
        return terror(w, 10)
    if graph_saved_check(w, getw(w, p + 280)) == 0:
        return 0
    if graph_saved_check(w, getw(w, p + 288)) == 0:
        return 0
    if graph_saved_check(w, getw(w, p + 296)) == 0:
        return 0
    return indexed_add_backward_check(w, t)

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

def partial_mark(w: mutbuf, t: u64, epoch: u64) -> u64:
    putw(w, dptr(w, t) + 168, epoch)
    i = t
    while i > 0:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 168) == epoch and (getw(w, p + 40) != 0):
            mark_concat(w, i, epoch)
        i = i - 1
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

def mark_graph(w: mutbuf, t: u64, epoch: u64) -> u64:
    putw(w, dptr(w, t) + 168, epoch)
    i = t
    while i > 0:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 168) == epoch and (getw(w, p + 40) != 0):
            if getw(w, p + 328) != 0:
                return terror(w, 10)
            if graph_saved_check(w, getw(w, p + 280)) == 0:
                return 0
            if graph_saved_check(w, getw(w, p + 288)) == 0:
                return 0
            if graph_saved_check(w, getw(w, p + 296)) == 0:
                return 0
            if indexed_add_backward_check(w, i) == 0:
                return 0
            mark_concat(w, i, epoch)
        i = i - 1
    return 1

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

def graph_collect_since(w: mutbuf, first: u64) -> u64:
    i = getw(w, 8) - 1
    while i >= first and i > 0:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 200) == 0:
            if getw(w, p + 208) == 0:
                gradient_gc_enqueue(w, i)
            elif getw(w, p + 312) == 0 and getw(w, p + 88) == 0 and (getw(w, p + 400) == 0):
                tensor_row_release(w, i)
        i = i - 1
    budget = 0
    while getw(w, 3968) != 0 and getw(w, 0) == 0:
        if budget >= 131070:
            return terror(w, 9)
        budget = budget + 1
        current = getw(w, 3968)
        p = dptr(w, current)
        putw(w, 3968, getw(w, p + 424))
        if getw(w, 3968) == 0:
            putw(w, 3976, 0)
        putw(w, p + 424, 0)
        putw(w, p + 432, 0)
        if getw(w, p + 304) != 0 and getw(w, p + 200) == 0 and (getw(w, p + 208) == 0):
            graph_node_release(w, current)
        elif getw(w, p + 304) != 0 and getw(w, p + 200) == 0 and (getw(w, p + 312) == 0) and (getw(w, p + 400) == 0) and (getw(w, p + 88) == 0):
            tensor_row_release(w, current)
    return choose(getw(w, 0) == 0, 1, 0)

def leaf_requires_grad(w: mutbuf, id: u64, enabled: u64) -> u64:
    t = public_node(w, id)
    if t == 0:
        return 0
    if enabled > 1:
        return terror(w, 7)
    p = dptr(w, t)
    if getw(w, p + 40) != 0:
        return terror(w, 5)
    putw(w, p + 32, enabled)
    return getw(w, p + 192)

def finish_command(w: mutbuf, first: u64) -> u64:
    i = first
    while i < getw(w, 8):
        p = dptr(w, i)
        if getw(w, p + 200) == 2:
            putw(w, p + 200, 0)
        i = i + 1
    return graph_collect_since(w, first)

def node_retain(w: mutbuf, t: u64) -> u64:
    if t == 0:
        return 1
    p = dptr(w, t)
    refs = getw(w, p + 208)
    if refs >= 4294967295:
        return terror(w, 8)
    putw(w, p + 208, refs + 1)
    return 1

def gradient_accumulate(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    if getw(w, p + 32) == 0:
        return 1
    g = gradient_binding(w, t)
    if g == 0 or getw(w, p + 448) == 0:
        return terror(w, 9)
    present = getw(w, p + 360)
    if present == 0 and getw(w, p + 472) == 2 and (getw(w, dptr(w, getw(w, p + 488)) + 528) == 1) and (incoming_obeys_contract(w, t) != 0):
        if incoming_adopt_gradient(w, t, g) == 0:
            return 0
        putw(w, p + 360, 1)
        return 1
    bumps = 1
    if present == 0 and is_dense(w, t) == 0:
        bumps = 0
    if version_can_bump(w, g, bumps) == 0:
        return 0
    src = getw(w, p + 160)
    i = 0
    while i < getw(w, p + 8):
        address = taddr(w, g, i)
        value = get4(w, src + i * 4)
        if present != 0:
            value = addf(get4(w, address), value)
        put4(w, address, value)
        i = i + 1
    if bumps != 0:
        bump_version(w, g)
    putw(w, p + 360, 1)
    return 1

def incoming_obeys_contract(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    if is_dense(w, t) == 0:
        return incoming_contiguous(w, t, p + 496)
    j = 0
    while j < getw(w, p):
        size = getw(w, p + 96 + j * 8)
        stride = getw(w, p + 496 + j * 8)
        if size != 1:
            if stride != getw(w, p + 128 + j * 8):
                return 0
        elif stride == 0:
            return 0
        j = j + 1
    return 1

def incoming_adopt_gradient(w: mutbuf, t: u64, g: u64) -> u64:
    p = dptr(w, t)
    q = dptr(w, g)
    sid = getw(w, p + 448)
    vid = getw(w, p + 456)
    offset = getw(w, p + 464)
    sp = so_storage_live(w, sid)
    if sp == 0 or so_version_live(w, vid) == 0:
        return terror(w, 9)
    size = so_uget(w, sp + 24)
    if getw(w, p + 8) != 0:
        if offset > size:
            return terror(w, 9)
        bound = offset + 4
        j = 0
        while j < getw(w, p):
            bound = bound + (getw(w, p + 96 + j * 8) - 1) * getw(w, p + 496 + j * 8) * 4
            j = j + 1
        if bound > size:
            return terror(w, 9)
    if so_storage_retain(w, sid) != 0 or so_version_retain(w, vid) != 0:
        return terror(w, 9)
    if so_storage_release(w, getw(w, q + 216)) != 0 or so_version_release(w, getw(w, q + 224)) != 0:
        return terror(w, 9)
    tp = so_slot_lookup(w, 1, getw(w, q + 192))
    if tp == 0:
        return terror(w, 9)
    so_uput(w, tp + 16, sid)
    so_uput(w, tp + 24, vid)
    so_uput(w, tp + 32, offset)
    remaining = 0
    if offset <= size:
        remaining = size - offset
    so_uput(w, tp + 40, remaining)
    putw(w, q + 16, 9223372036854775808 + so_uget(w, sp + 16) + offset)
    putw(w, q + 216, sid)
    putw(w, q + 224, vid)
    putw(w, q + 232, offset)
    j = 0
    while j < 4:
        putw(w, q + 128 + j * 8, getw(w, p + 496 + j * 8))
        j = j + 1
    return 1

def gradient_cycle_check(w: mutbuf, t: u64, g: u64) -> u64:
    if getw(w, 3984) != 0:
        return terror(w, 9)
    gradient_cycle_push(w, g, t)
    budget = 0
    while getw(w, 3984) != 0 and getw(w, 0) == 0:
        current = getw(w, 3984)
        p = dptr(w, current)
        putw(w, 3984, getw(w, p + 440))
        putw(w, p + 440, 0)
        gradient_cycle_push(w, getw(w, p + 384), t)
        count = gradient_parent_count(w, current)
        if count > 16777216 - budget:
            terror(w, 3)
        else:
            budget = budget + count
            j = 0
            while j < count and getw(w, 0) == 0:
                gradient_cycle_push(w, graph_parent_at(w, current, j), t)
                j = j + 1
    i = 1
    while i < getw(w, 8):
        p = dptr(w, i)
        putw(w, p + 416, 0)
        putw(w, p + 440, 0)
        i = i + 1
    putw(w, 3984, 0)
    return choose(getw(w, 0) == 0, 1, 0)

def retained_apply(w: mutbuf, table: u64, t: u64) -> u64:
    g = getw(w, table + t * 8)
    if g == 0:
        return 1
    if retained_enabled(w, t) == 0:
        putw(w, table + t * 8, 0)
        return retained_release_one(w, g)
    p = dptr(w, t)
    if getw(w, p + 40) == 0 or getw(w, p + 448) == 0 or getw(w, p + 336) == 0:
        return terror(w, 9)
    old = gradient_binding(w, t)
    if getw(w, 0) != 0:
        return 0
    if old != 0:
        if retained_sum_layout(w, g, old, t) == 0:
            return 0
    i = 0
    while i < getw(w, p + 8):
        value = get4(w, incoming_address(w, t, i))
        if old != 0:
            value = addf(tget(w, old, i), value)
        put4(w, taddr(w, g, i), value)
        i = i + 1
    if gradient_assign(w, t, g) == 0:
        return 0
    putw(w, table + t * 8, 0)
    return retained_release_one(w, g)

def retained_reserve(w: mutbuf, table: u64, limit: u64, epoch: u64) -> u64:
    memory_fill(w, table, (limit + 1) * 8, 0)
    i = 1
    while i <= limit:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 168) == epoch and (getw(w, p + 40) != 0):
            if retained_enabled(w, i) != 0:
                g = retained_reserve_one(w, i)
                if g == 0:
                    return 0
                putw(w, table + i * 8, g)
        i = i + 1
    return choose(getw(w, 0) == 0, 1, 0)

def retained_enable(w: mutbuf, id: u64) -> u64:
    t = public_node(w, id)
    if t == 0:
        return 0
    p = dptr(w, t)
    if getw(w, p + 32) == 0:
        return terror(w, 5)
    if getw(w, p + 40) == 0:
        return 1
    row = retained_row(w, t)
    if row == 0 or so_uget(w, row + 56) > 1:
        return terror(w, 9)
    so_uput(w, row + 56, 1)
    return 1

def release_backward_scratch(w: mutbuf, limit: u64, epoch: u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w, i)
        if getw(w, p + 304) != 0 and getw(w, p + 168) == epoch:
            if incoming_clear(w, i) == 0:
                return 0
            if getw(w, p + 360) == 0:
                if gradient_unbind(w, i) == 0:
                    return 0
        i = i + 1
    i = 1
    while i <= limit:
        if getw(w, dptr(w, i) + 528) != 0:
            return terror(w, 9)
        i = i + 1
    return 1
