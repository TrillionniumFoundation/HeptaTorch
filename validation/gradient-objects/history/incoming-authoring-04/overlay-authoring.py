def dptr(w: mutbuf, t: u64) -> u64:
    page = getw(w, 4096 + (t >> 8) * 8)
    return page + (t & 255) * 576

def descriptor_page(w: mutbuf, slot: u64) -> u64:
    address = 4096 + (slot >> 8) * 8
    if getw(w, address) == 0:
        page = alloc(w, 147456)
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
    highest = keep
    i = keep + 1
    while i < old_next:
        p = dptr(w, i)
        if getw(w, p + 304) != 0:
            highest = i
        else:
            memory_fill(w, p, 576, 0)
        i = i + 1
    page = (highest >> 8) + 1
    while page < 256:
        hold_release(w, getw(w, 6144 + page * 8))
        putw(w, 4096 + page * 8, 0)
        putw(w, 6144 + page * 8, 0)
        page = page + 1
    putw(w, 8, highest + 1)
    putw(w, 40, keep)
    putw(w, 32, getw(w, 32) + 1)
    return 1

def init_state(w: mutbuf, slots: u64, pool: u64) -> u64:
    if length(w) != 139264 + so_state_size(slots, pool):
        return 0
    fill_range(w, 0, length(w), 0)
    so_state_init(w, slots, pool)
    putw(w, 8, 1)
    putw(w, 64, 3616465548122477640)
    return 1

def state_header(w: mutbuf) -> u64:
    if length(w) == 0:
        return 0
    if length(w) < 139264 + 2352 + 131072:
        return 7
    if so_state_header(w) != 0 or getw(w, 64) != 3616465548122477640:
        return 7
    if getw(w, 0) != 0 or getw(w, 48) != 0 or getw(w, 56) != 0:
        return 7
    if getw(w, 88) != 0 or getw(w, 96) != 0 or getw(w, 104) != 0 or (getw(w, 112) != 0):
        return 7
    if getw(w, 3968) != 0 or getw(w, 3976) != 0 or getw(w, 3984) != 0:
        return 7
    if getw(w,3992) != 0 or getw(w,4000) != 0 or getw(w,4008) != 0:
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
            if sp == 0 or so_uget(w, sp + 24) != 147456:
                return 7
            if base != 9223372036854775808 + so_uget(w, sp + 16):
                return 7
        elif hid != 0:
            return 7
        i = i + 1
    return 0

def so_state_init(w: mutbuf, slots: u64, pool: u64) -> u64:
    so_uput(w, 0, 3622091821952226376)
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
    if so_uget(w, 0) != 3622091821952226376 or so_uget(w, 8) != 2 or so_uget(w, 48) != 1:
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

def gradient_audit(w: mutbuf) -> u64:
    status = 0
    if getw(w,3992) != 0 or getw(w,4000) != 0 or getw(w,4008) != 0:
        return 9
    if getw(w, 3968) != 0 or getw(w, 3976) != 0 or getw(w, 3984) != 0:
        return 9
    i = 1
    while i < getw(w, 8) and status == 0:
        p = dptr(w, i)
        field = 448
        while field < 576:
            if getw(w,p+field) != 0:
                status = 9
            field = field+8
        if getw(w, p + 416) != 0 or getw(w, p + 424) != 0 or getw(w, p + 432) != 0 or (getw(w, p + 440) != 0):
            status = 9
        i = i + 1
    i = 1
    while i < getw(w, 8) and status == 0:
        p = dptr(w, i)
        if getw(w, p + 304) != 0:
            if getw(w, p + 24) != 0 or getw(w, p + 248) != 0 or getw(w, p + 360) > 1:
                status = 9
            g = getw(w, p + 384)
            if g == 0:
                if getw(w, p + 392) != 0 or getw(w, p + 360) != 0:
                    status = 9
            else:
                q = private_tensor(w, g, getw(w, p + 392))
                if q == 0 or getw(w, p + 360) != 1 or gradient_shape_matches(w, i, g) == 0:
                    status = 9
                else:
                    putw(w, q + 416, getw(w, q + 416) + 1)
        i = i + 1
    i = 1
    while i < getw(w, 8):
        p = dptr(w, i)
        if getw(w, p + 304) != 0:
            if getw(w, p + 400) != getw(w, p + 416):
                status = 9
        elif getw(w, p + 400) != 0 or getw(w, p + 416) != 0:
            status = 9
        putw(w, p + 416, 0)
        i = i + 1
    return status

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
    if incoming_seed(w,t,seed) == 0:
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
    if incoming_seed(w,t,seed) == 0:
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
def incoming_version_bump(w:mutbuf, id:u64) -> u64:
    p = so_version_live(w,id)
    if p == 0:
        return terror(w,9)
    old = so_uget(w,p+16)
    if old == 18446744073709551615:
        return terror(w,8)
    so_uput(w,p+16,old+1)
    return 1

def incoming_drop_value(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    sid = getw(w,p+448)
    if sid != 0:
        origin = getw(w,p+488)
        if origin == 0:
            return terror(w,9)
        q = dptr(w,origin)
        refs = getw(w,q+528)
        if refs == 0:
            return terror(w,9)
        putw(w,q+528,refs-1)
        if so_storage_release(w,sid) != 0 or so_version_release(w,getw(w,p+456)) != 0:
            return terror(w,9)
    putw(w,p+448,0)
    putw(w,p+456,0)
    putw(w,p+464,0)
    putw(w,p+472,0)
    putw(w,p+488,0)
    j = 0
    while j < 4:
        putw(w,p+496+j*8,0)
        j = j+1
    return 1

def incoming_drop_cache(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if hold_release(w,getw(w,p+256)) == 0:
        return 0
    vid = getw(w,p+480)
    if vid != 0 and so_version_release(w,vid) != 0:
        return terror(w,9)
    putw(w,p+160,0)
    putw(w,p+256,0)
    putw(w,p+480,0)
    putw(w,p+536,0)
    return 1

def incoming_clear(w:mutbuf, t:u64) -> u64:
    if incoming_drop_value(w,t) == 0:
        return 0
    return incoming_drop_cache(w,t)

def incoming_bind(w:mutbuf, t:u64, sid:u64, vid:u64, offset:u64, mode:u64, origin:u64) -> u64:
    if sid == 0 or vid == 0 or origin == 0 or mode == 0 or mode > 2:
        return terror(w,9)
    if getw(w,dptr(w,origin)+528) >= 65535:
        return terror(w,8)
    if so_storage_retain(w,sid) != 0 or so_version_retain(w,vid) != 0:
        return terror(w,9)
    putw(w,dptr(w,origin)+528,getw(w,dptr(w,origin)+528)+1)
    if incoming_drop_value(w,t) == 0:
        return 0
    p = dptr(w,t)
    putw(w,p+448,sid)
    putw(w,p+456,vid)
    putw(w,p+464,offset)
    putw(w,p+472,mode)
    putw(w,p+488,origin)
    return 1

def incoming_contiguous_layout(w:mutbuf, t:u64, dest:u64) -> u64:
    p = dptr(w,t)
    j = 0
    while j < 4:
        putw(w,dest+j*8,0)
        j = j+1
    j = getw(w,p)
    stride = 1
    while j > 0:
        j = j-1
        putw(w,dest+j*8,stride)
        size = getw(w,p+96+j*8)
        if size > 0:
            stride = stride*size
    return 1

def incoming_contiguous(w:mutbuf, t:u64, layout:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+8) == 0:
        return 1
    j = getw(w,p)
    stride = 1
    while j > 0:
        j = j-1
        size = getw(w,p+96+j*8)
        if size > 1 and getw(w,layout+j*8) != stride:
            return 0
        stride = stride*size
    return 1

def incoming_dense(w:mutbuf, t:u64, layout:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+8) == 0:
        return 1
    used = 0
    expected = 1
    while used != 15:
        choice = 4
        best = 18446744073709551615
        j = 0
        while j < getw(w,p):
            if (used & (1 << j)) == 0 and getw(w,p+96+j*8) > 1:
                stride = getw(w,layout+j*8)
                if stride < best:
                    best = stride
                    choice = j
            j = j+1
        if choice == 4:
            return 1
        if best != expected:
            return 0
        expected = expected*getw(w,p+96+choice*8)
        used = used | (1 << choice)
    return 1

def incoming_same_shape(w:mutbuf, a:u64, b:u64) -> u64:
    p = dptr(w,a)
    q = dptr(w,b)
    if getw(w,p) != getw(w,q):
        return 0
    j = 0
    while j < getw(w,p):
        if getw(w,p+96+j*8) != getw(w,q+96+j*8):
            return 0
        j = j+1
    return 1

def incoming_reshape_layout(w:mutbuf, source:u64, target:u64, dest:u64) -> u64:
    p = dptr(w,source)
    q = dptr(w,target)
    oldrank = getw(w,p)
    newrank = getw(w,q)
    if oldrank == 0:
        j = 0
        while j < newrank:
            putw(w,dest+j*8,1)
            j = j+1
        return 1
    if getw(w,p+8) == 0:
        if incoming_same_shape(w,source,target) != 0:
            j = 0
            while j < newrank:
                putw(w,dest+j*8,getw(w,p+496+j*8))
                j = j+1
            return 1
        return incoming_contiguous_layout(w,target,dest)
    tensor_d = oldrank
    view_d = newrank
    chunk_stride = getw(w,p+496+(oldrank-1)*8)
    tensor_n = 1
    view_n = 1
    while tensor_d > 0:
        tensor_d = tensor_d-1
        tensor_n = tensor_n*getw(w,p+96+tensor_d*8)
        boundary = choose(tensor_d == 0,1,0)
        if tensor_d > 0:
            if getw(w,p+96+(tensor_d-1)*8) != 1 and getw(w,p+496+(tensor_d-1)*8) != tensor_n*chunk_stride:
                boundary = 1
        if boundary != 0:
            while view_d > 0 and (view_n < tensor_n or getw(w,q+96+(view_d-1)*8) == 1):
                view_d = view_d-1
                putw(w,dest+view_d*8,view_n*chunk_stride)
                view_n = view_n*getw(w,q+96+view_d*8)
            if view_n != tensor_n:
                return 0
            if tensor_d > 0:
                chunk_stride = getw(w,p+496+(tensor_d-1)*8)
                tensor_n = 1
                view_n = 1
    return choose(view_d == 0,1,0)

def incoming_seed(w:mutbuf, t:u64, seed:u64) -> u64:
    p = dptr(w,t)
    sid = getw(w,p+536)
    vid = getw(w,p+480)
    offset = 0
    layout = p+496
    if seed != 0:
        q = dptr(w,seed)
        sid = getw(w,q+216)
        vid = getw(w,q+224)
        offset = getw(w,q+232)
    if incoming_bind(w,t,sid,vid,offset,1,t) == 0:
        return 0
    if seed == 0:
        return incoming_contiguous_layout(w,t,layout)
    j = 0
    while j < 4:
        putw(w,layout+j*8,getw(w,dptr(w,seed)+128+j*8))
        j = j+1
    return 1

def incoming_describe_alias(w:mutbuf, t:u64, mode:u64, origin:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+448) == 0:
        return terror(w,9)
    putw(w,3808,1)
    putw(w,3816,getw(w,p+448))
    putw(w,3824,getw(w,p+456))
    putw(w,3832,getw(w,p+464))
    putw(w,3840,origin)
    putw(w,3848,mode)
    j = 0
    while j < 4:
        putw(w,3856+j*8,getw(w,p+496+j*8))
        j = j+1
    return 1

def incoming_describe_cat(w:mutbuf, t:u64, parent:u64, edge:u64) -> u64:
    p = dptr(w,t)
    q = dptr(w,parent)
    axis = getw(w,p+176)
    op = getw(w,p+40)
    if op == 37 and getw(w,p) != getw(w,q):
        putw(w,3848,1)
        return 1
    if incoming_describe_alias(w,t,2,parent) == 0:
        return 0
    offset = 0
    if op == 38:
        offset = edge
        j = 0
        while j < getw(w,q):
            source_axis = j
            if j >= axis:
                source_axis = j+1
            putw(w,3856+j*8,getw(w,p+496+source_axis*8))
            j = j+1
    else:
        j = 0
        while j < edge:
            previous = graph_parent_at(w,t,j)
            r = dptr(w,previous)
            if getw(w,r) == getw(w,p):
                offset = offset+getw(w,r+96+axis*8)
            j = j+1
    putw(w,3832,getw(w,p+464)+offset*getw(w,p+496+axis*8)*4)
    return 1

def incoming_describe(w:mutbuf, t:u64, parent:u64, edge:u64) -> u64:
    memory_fill(w,3808,88,0)
    p = dptr(w,t)
    op = getw(w,p+40)
    putw(w,3840,parent)
    putw(w,3848,2)
    incoming_contiguous_layout(w,parent,3856)
    if op == 1 or (op == 2 and edge == 0) or op == 17 or ((op == 39 or op == 40) and edge == 0):
        if incoming_same_shape(w,t,parent) != 0:
            return incoming_describe_alias(w,t,getw(w,p+472),getw(w,p+488))
        return 1
    if op == 10:
        if incoming_describe_alias(w,t,2,parent) == 0:
            return 0
        a = getw(w,p+176)
        b = getw(w,p+184)
        putw(w,3856+a*8,getw(w,p+496+b*8))
        putw(w,3856+b*8,getw(w,p+496+a*8))
        return 1
    if op == 16:
        if incoming_describe_alias(w,t,2,parent) == 0:
            return 0
        if incoming_reshape_layout(w,t,parent,3856) != 0:
            return 1
        putw(w,3808,0)
        putw(w,3816,0)
        putw(w,3824,0)
        putw(w,3832,0)
        return incoming_contiguous_layout(w,parent,3856)
    if op == 7:
        if incoming_describe_alias(w,t,2,parent) == 0:
            return 0
        j = 0
        while j < getw(w,dptr(w,parent)):
            putw(w,3856+j*8,0)
            j = j+1
        return 1
    if op == 37 or op == 38:
        return incoming_describe_cat(w,t,parent,edge)
    if op == 26 or op == 27:
        putw(w,3888,1)
        return 1
    if op == 2 or op == 3 or op == 4 or op == 5 or op == 6 or (op >= 11 and op <= 15) or op == 20 or op == 21 or op == 22 or op == 25 or op == 39 or op == 40:
        return 1
    putw(w,3848,1)
    return 1

def incoming_address(w:mutbuf, t:u64, index:u64) -> u64:
    p = dptr(w,t)
    sp = so_storage_live(w,getw(w,p+448))
    if sp == 0:
        return terror(w,9)
    address = getw(w,p+464)
    axis = getw(w,p)
    while axis > 0:
        axis = axis-1
        size = getw(w,p+96+axis*8)
        address = address+irem(index,size)*getw(w,p+496+axis*8)*4
        index = idiv(index,size)
    if address > so_uget(w,sp+24) or 4 > so_uget(w,sp+24)-address:
        return terror(w,9)
    return 9223372036854775808+so_uget(w,sp+16)+address

def incoming_can_accumulate(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+472) != 2:
        return 0
    origin = getw(w,p+488)
    if origin == 0 or getw(w,dptr(w,origin)+528) != 1 or incoming_dense(w,t,p+496) == 0:
        return 0
    sp = so_storage_live(w,getw(w,p+448))
    if sp == 0:
        return terror(w,9)
    expected = 1
    if getw(w,p+256) != 0 and getw(w,p+536) == getw(w,p+448):
        expected = expected+1
    return choose(so_uget(w,sp+40) == expected,1,0)

def incoming_construct(w:mutbuf, parent:u64) -> u64:
    p = dptr(w,parent)
    sid = getw(w,3816)
    vid = getw(w,3824)
    offset = getw(w,3832)
    mode = getw(w,3848)
    origin = getw(w,3840)
    if getw(w,3808) == 0:
        sid = getw(w,p+536)
        vid = getw(w,p+480)
        offset = 0
        origin = parent
    if incoming_bind(w,parent,sid,vid,offset,mode,origin) == 0:
        return 0
    j = 0
    while j < 4:
        putw(w,p+496+j*8,getw(w,3856+j*8))
        j = j+1
    if getw(w,3808) == 0 and getw(w,3888) != 0:
        return incoming_version_bump(w,vid)
    return 1

def incoming_merge_edge(w:mutbuf, t:u64, parent:u64, edge:u64, source:u64) -> u64:
    p = dptr(w,parent)
    if incoming_describe(w,t,parent,edge) == 0:
        return 0
    present = getw(w,p+336)
    inplace = 0
    if present != 0:
        inplace = incoming_can_accumulate(w,parent)
        if getw(w,0) != 0:
            return 0
    dest = getw(w,p+160)
    j = 0
    while j < getw(w,p+8):
        value = get4(w,source+j*4)
        if present != 0:
            value = addf(get4(w,dest+j*4),value)
        put4(w,dest+j*4,value)
        if inplace != 0 and getw(w,p+448) != getw(w,p+536):
            put4(w,incoming_address(w,parent,j),value)
        j = j+1
    if present == 0:
        if incoming_construct(w,parent) == 0:
            return 0
    elif inplace != 0:
        if incoming_version_bump(w,getw(w,p+456)) == 0:
            return 0
    else:
        if getw(w,p+448) == getw(w,p+536):
            return terror(w,9)
        if incoming_bind(w,parent,getw(w,p+536),getw(w,p+480),0,2,parent) == 0:
            return 0
        incoming_contiguous_layout(w,parent,p+496)
    putw(w,p+336,1)
    return 1

def gradient_workspace_size(w:mutbuf, limit:u64, epoch:u64) -> u64:
    maximum = 0
    t = 1
    while t <= limit:
        p = dptr(w,t)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch and getw(w,p+40) != 0:
            count = gradient_parent_count(w,t)
            amount = 8+count*32
            j = 0
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
    if getw(w,3992) != 0 or getw(w,4000) != 0 or getw(w,4008) != 0:
        return terror(w,9)
    count = gradient_parent_count(w,t)
    putw(w,scratch,count)
    putw(w,3992,scratch)
    putw(w,4000,count)
    cursor = scratch+8+count*32
    j = 0
    while j < count:
        row = scratch+8+j*32
        parent = gradient_parent_at(w,t,j)
        memory_fill(w,row,32,0)
        putw(w,row,parent)
        if parent != 0:
            n = getw(w,dptr(w,parent)+8)
            memory_fill(w,cursor,n*5,0)
            putw(w,row+8,cursor)
            if gradient_direct(w,t,parent) != 0:
                putw(w,row+16,cursor+n*4)
            putw(w,row+24,n)
            cursor = cursor+n*5
        j = j+1
    return 1

def incoming_gadd(w:mutbuf, edge:u64, t:u64, index:u64, value:u64) -> u64:
    table = getw(w,3992)
    if table == 0 or edge >= getw(w,4000):
        return terror(w,9)
    row = table+8+edge*32
    parent = getw(w,row)
    if parent == 0:
        return 0
    if parent != t or index >= getw(w,row+24):
        return terror(w,9)
    address = getw(w,row+8)+index*4
    flags = getw(w,row+16)
    if flags != 0:
        flag = object_address(w,flags+index)
        if byte(w,flag) == 0:
            store(w,flag,1)
            put4(w,address,value)
            return 1
    put4(w,address,addf(get4(w,address),value))
    return 1

def incoming_gadd_first(w:mutbuf, t:u64, index:u64, value:u64) -> u64:
    return incoming_gadd(w,0,t,index,value)

def incoming_gadd_second(w:mutbuf, t:u64, index:u64, value:u64) -> u64:
    return incoming_gadd(w,1,t,index,value)

def incoming_gadd_selected(w:mutbuf, t:u64, index:u64, value:u64) -> u64:
    return incoming_gadd(w,getw(w,4008),t,index,value)

def gadd(w:mutbuf, t:u64, index:u64, value:u64) -> u64:
    j = 0
    table = getw(w,3992)
    while j < getw(w,4000):
        if getw(w,table+8+j*32) == t:
            return incoming_gadd(w,j,t,index,value)
        j = j+1
    return 0

def incoming_concat_part(w:mutbuf, t:u64, a:u64, offset:u64, reverse:u64) -> u64:
    p = dptr(w,t)
    j = 0
    current = 0
    count = gradient_parent_count(w,t)
    axis = getw(w,p+176)
    while j < count:
        parent = graph_parent_at(w,t,j)
        if parent == a and current == offset:
            putw(w,4008,j)
            result = concat_part(w,t,a,offset,reverse)
            putw(w,4008,0)
            return result
        if getw(w,p+40) == 38:
            current = current+1
        elif getw(w,dptr(w,parent)) == getw(w,p):
            current = current+getw(w,dptr(w,parent)+96+axis*8)
        j = j+1
    return terror(w,9)

def gradient_finish(w:mutbuf, t:u64, commit:u64) -> u64:
    table = getw(w,3992)
    count = getw(w,4000)
    j = 0
    while j < count and getw(w,0) == 0:
        row = table+8+j*32
        parent = getw(w,row)
        if parent != 0 and commit != 0:
            if incoming_merge_edge(w,t,parent,j,getw(w,row+8)) == 0:
                return 0
        j = j+1
    putw(w,3992,0)
    putw(w,4000,0)
    putw(w,4008,0)
    memory_fill(w,3808,88,0)
    return choose(getw(w,0) == 0,1,0)

def gradient_apply(w:mutbuf, t:u64, scratch:u64) -> u64:
    if getw(w,dptr(w,t)+40) == 0:
        partial_apply_node(w,t)
    else:
        putw(w,112,t)
        gradient_prepare(w,t,scratch)
        if getw(w,0) == 0:
            partial_apply_node(w,t)
        if getw(w,0) == 0:
            gradient_finish(w,t,1)
        putw(w,112,0)
    if getw(w,0) == 0:
        incoming_clear(w,t)
    putw(w,3992,0)
    putw(w,4000,0)
    putw(w,4008,0)
    memory_fill(w,3808,88,0)
    return choose(getw(w,0) == 0,1,0)

def ensure_backward_work(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if getw(w,p+448) != 0 or getw(w,p+480) != 0 or getw(w,p+528) != 0:
                return terror(w,9)
            if getw(w,p+40) == 0 and getw(w,p+32) != 0 and getw(w,p+384) == 0:
                if gradient_new(w,i) == 0:
                    return 0
            if getw(w,p+256) == 0:
                address = alloc(w,getw(w,p+8)*4)
                if address == 0:
                    return 0
                putw(w,p+160,address)
                putw(w,p+256,getw(w,80))
            if storage_capacity(w,3,0) == 0:
                return 0
            vid = so_slot_new(w,3)
            if vid == 0:
                return terror(w,9)
            vp = so_slot_lookup(w,3,vid)
            so_uput(w,vp+24,1)
            putw(w,p+480,vid)
            hp = so_slot_lookup(w,6,getw(w,p+256))
            if hp == 0:
                return terror(w,9)
            putw(w,p+536,so_uget(w,hp+16))
        i = i+1
    return 1

def incoming_obeys_contract(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if is_dense(w,t) == 0:
        return incoming_contiguous(w,t,p+496)
    j = 0
    while j < getw(w,p):
        size = getw(w,p+96+j*8)
        stride = getw(w,p+496+j*8)
        if size != 1:
            if stride != getw(w,p+128+j*8):
                return 0
        elif stride == 0:
            return 0
        j = j+1
    return 1

def incoming_adopt_gradient(w:mutbuf, t:u64, g:u64) -> u64:
    p = dptr(w,t)
    q = dptr(w,g)
    sid = getw(w,p+448)
    vid = getw(w,p+456)
    offset = getw(w,p+464)
    sp = so_storage_live(w,sid)
    if sp == 0 or so_version_live(w,vid) == 0:
        return terror(w,9)
    size = so_uget(w,sp+24)
    if getw(w,p+8) != 0:
        if offset > size:
            return terror(w,9)
        bound = offset+4
        j = 0
        while j < getw(w,p):
            bound = bound+(getw(w,p+96+j*8)-1)*getw(w,p+496+j*8)*4
            j = j+1
        if bound > size:
            return terror(w,9)
    if so_storage_retain(w,sid) != 0 or so_version_retain(w,vid) != 0:
        return terror(w,9)
    if so_storage_release(w,getw(w,q+216)) != 0 or so_version_release(w,getw(w,q+224)) != 0:
        return terror(w,9)
    tp = so_slot_lookup(w,1,getw(w,q+192))
    if tp == 0:
        return terror(w,9)
    so_uput(w,tp+16,sid)
    so_uput(w,tp+24,vid)
    so_uput(w,tp+32,offset)
    remaining = 0
    if offset <= size:
        remaining = size-offset
    so_uput(w,tp+40,remaining)
    putw(w,q+16,9223372036854775808+so_uget(w,sp+16)+offset)
    putw(w,q+216,sid)
    putw(w,q+224,vid)
    putw(w,q+232,offset)
    j = 0
    while j < 4:
        putw(w,q+128+j*8,getw(w,p+496+j*8))
        j = j+1
    return 1

def gradient_accumulate(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+32) == 0:
        return 1
    g = gradient_binding(w,t)
    if g == 0 or getw(w,p+448) == 0:
        return terror(w,9)
    present = getw(w,p+360)
    if present == 0 and getw(w,p+472) == 2 and getw(w,dptr(w,getw(w,p+488))+528) == 1 and incoming_obeys_contract(w,t) != 0:
        if incoming_adopt_gradient(w,t,g) == 0:
            return 0
        putw(w,p+360,1)
        return 1
    bumps = 1
    if present == 0 and is_dense(w,t) == 0:
        bumps = 0
    if version_can_bump(w,g,bumps) == 0:
        return 0
    src = getw(w,p+160)
    i = 0
    while i < getw(w,p+8):
        address = taddr(w,g,i)
        value = get4(w,src+i*4)
        if present != 0:
            value = addf(get4(w,address),value)
        put4(w,address,value)
        i = i+1
    if bumps != 0:
        bump_version(w,g)
    putw(w,p+360,1)
    return 1

def release_backward_scratch(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if incoming_clear(w,i) == 0:
                return 0
            if getw(w,p+360) == 0:
                if gradient_unbind(w,i) == 0:
                    return 0
        i = i+1
    i = 1
    while i <= limit:
        if getw(w,dptr(w,i)+528) != 0:
            return terror(w,9)
        i = i+1
    return 1
