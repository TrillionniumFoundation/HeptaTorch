def partial_admit(w:mutbuf, t:u64, seed:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+32) == 0:
        return terror(w,5)
    if seed == 0:
        if getw(w,p+8) != 1:
            return terror(w,5)
    else:
        if valid(w,seed) == 0:
            return 0
        q = dptr(w,seed)
        rank = getw(w,p)
        if getw(w,q) != rank:
            return terror(w,2)
        axis = 0
        while axis < rank:
            if getw(w,p+96+axis*8) != getw(w,q+96+axis*8):
                return terror(w,2)
            axis = axis+1
    return 1

def partial_mark(w:mutbuf, t:u64, epoch:u64) -> u64:
    putw(w,dptr(w,t)+168,epoch)
    i = t
    while i > 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch and getw(w,p+40) != 0:
            op = getw(w,p+40)
            if op == 37 or op == 38:
                mark_concat(w,i,epoch)
            else:
                a = getw(w,p+48)
                b = getw(w,p+56)
                if getw(w,dptr(w,a)+32) != 0:
                    putw(w,dptr(w,a)+168,epoch)
                if b != 0:
                    if getw(w,dptr(w,b)+32) != 0:
                        putw(w,dptr(w,b)+168,epoch)
        i = i-1
    return 1

def partial_push(w:mutbuf, queue:u64, t:u64) -> u64:
    limit = getw(w,queue+8)
    count = getw(w,queue)+1
    if count > limit:
        return terror(w,8)
    putw(w,queue,count)
    heap = queue+24+(limit+1)*8
    key = t
    if getw(w,dptr(w,t)+40) == 0:
        key = key | 9223372036854775808
    i = count
    moving = 1
    while i > 1 and moving != 0:
        parent = i//2
        previous = getw(w,heap+parent*8)
        if previous < key:
            putw(w,heap+i*8,previous)
            i = parent
        else:
            moving = 0
    putw(w,heap+i*8,key)
    return 1

def partial_pop(w:mutbuf, queue:u64) -> u64:
    count = getw(w,queue)
    if count == 0:
        return terror(w,8)
    heap = queue+24+(getw(w,queue+8)+1)*8
    result = getw(w,heap+8)
    key = getw(w,heap+count*8)
    count = count-1
    putw(w,queue,count)
    i = 1
    moving = 1
    while i*2 <= count and moving != 0:
        child = i*2
        if child+1 <= count:
            if getw(w,heap+(child+1)*8) > getw(w,heap+child*8):
                child = child+1
        value = getw(w,heap+child*8)
        if key < value:
            putw(w,heap+i*8,value)
            i = child
        else:
            moving = 0
    if count != 0:
        putw(w,heap+i*8,key)
    return result & 9223372036854775807

def partial_parent(w:mutbuf, queue:u64, t:u64, apply:u64) -> u64:
    if t == 0:
        return 1
    if getw(w,dptr(w,t)+32) == 0:
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
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 0:
        return 1
    if op == 37 or op == 38:
        parents = getw(w,p+184)
        j = 0
        while j < getw(w,parents):
            if partial_parent(w,queue,getw(w,parents+8+j*16),apply) == 0:
                return 0
            j = j+1
    else:
        if partial_parent(w,queue,getw(w,p+48),apply) == 0:
            return 0
        if partial_parent(w,queue,getw(w,p+56),apply) == 0:
            return 0
    return 1

def partial_check_node(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+40) == 0:
        return 1
    if getw(w,p+328) != 0:
        return terror(w,10)
    if graph_saved_check(w,getw(w,p+280)) == 0:
        return 0
    if graph_saved_check(w,getw(w,p+288)) == 0:
        return 0
    if graph_saved_check(w,getw(w,p+296)) == 0:
        return 0
    return indexed_add_backward_check(w,t)

def partial_apply_node(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 0:
        j = 0
        n = getw(w,p+8)
        dst = getw(w,p+24)
        src = getw(w,p+160)
        while j < n:
            put4(w,dst+j*4,addf(get4(w,dst+j*4),get4(w,src+j*4)))
            j = j+1
        putw(w,p+360,1)
    if op >= 1 and op <= 4:
        back_binary(w,t)
    if op == 5 or op == 6 or op == 10 or (op >= 11 and op <= 17):
        back_unary(w,t)
    if op == 7 or op == 8:
        back_reduce(w,t)
    if op == 18 or op == 19:
        back_axis(w,t)
    if op == 20 or op == 21:
        back_softmax(w,t)
    if op >= 22 and op <= 24:
        back_spatial(w,t)
    if op == 25:
        back_matmul(w,t)
    if op == 26:
        back_index_select(w,t)
    if op == 27:
        back_gather(w,t)
    if op == 37 or op == 38:
        back_concat(w,t)
    if op == 39 or op == 40:
        back_indexed_add(w,t)
    return choose(getw(w,0) == 0,1,0)

def partial_execute(w:mutbuf, o:mutbuf, info:u64, t:u64, seed:u64, retain:u64) -> u64:
    if partial_admit(w,t,seed) == 0:
        return 0
    epoch = getw(w,24)+1
    if epoch == 0:
        return terror(w,8)
    putw(w,24,epoch)
    partial_mark(w,t,epoch)
    queue_bytes = 24+(t+1)*16
    queue = alloc(w,queue_bytes)
    if queue == 0:
        return 0
    queue_owner = getw(w,80)
    memory_fill(w,queue,queue_bytes,0)
    putw(w,queue+8,t)
    if ensure_backward_work(w,t,epoch) == 0:
        return 0
    i = 1
    while i <= t:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if partial_edges(w,queue,i,0) == 0:
                return 0
            memory_fill(w,getw(w,p+160),getw(w,p+8)*4,0)
        i = i+1
    p = dptr(w,t)
    if seed == 0:
        put4(w,getw(w,p+160),1065353216)
    else:
        i = 0
        while i < getw(w,p+8):
            put4(w,getw(w,p+160)+i*4,tget(w,seed,i))
            i = i+1
    if partial_push(w,queue,t) == 0:
        return 0
    error = 0
    failed_op = 0
    completed = 0
    leaves = 0
    while getw(w,queue) != 0 and error == 0:
        current = partial_pop(w,queue)
        if getw(w,0) == 0:
            partial_check_node(w,current)
        if getw(w,0) == 0:
            partial_apply_node(w,current)
        if getw(w,0) == 0 and retain == 0:
            release_backward_saved(w,current)
        if getw(w,0) == 0:
            completed = completed+1
            if getw(w,dptr(w,current)+40) == 0:
                leaves = leaves+1
            partial_edges(w,queue,current,1)
        error = getw(w,0)
        if error != 0:
            failed_op = getw(w,dptr(w,current)+40)
    putw(w,0,0)
    if release_backward_scratch(w,t,epoch) == 0:
        return 0
    if hold_release(w,queue_owner) == 0:
        return 0
    putw(o,info,error)
    putw(o,info+8,completed)
    putw(o,info+16,leaves)
    putw(o,info+24,failed_op)
    return error

def partial_command(w:mutbuf, o:mutbuf, info:u64, output:u64, seed:u64, flags:u64) -> u64:
    if flags > 1:
        return terror(w,7)
    if info > length(o) or length(o)-info < 32:
        return terror(w,3)
    t = public_node(w,output)
    v = public_optional(w,seed)
    if getw(w,0) != 0:
        return 0
    return partial_execute(w,o,info,t,v,flags)

def gradient_present(w:mutbuf, output:u64) -> u64:
    t = public_node(w,output)
    if t == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0 or getw(w,p+32) == 0:
        return 0
    return getw(w,p+360)

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
    halted = 0
    while i < count and halted == 0:
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
        if getr(s,32+i*64) == 58 and getw(o,at+16) != 0:
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
def object_command(s: readbuf, o: mutbuf, w: mutbuf, cp: u64, at: u64) -> u64:
    code = getr(s, cp)
    a = resolve_handle(w, getr(s, cp + 8))
    b = getr(s, cp + 16)
    c = getr(s, cp + 24)
    result = 0
    payload = 0
    first = getw(w, 8)
    if code < 41 or code > 59:
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
        result = partial_command(w,o,at+32,a,b,c)
        payload = 32
    if code == 59:
        result = gradient_present(w,a)
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
            putw(w,pi+360,1)
            j = 0
            n = getw(w, pi + 8)
            dst = getw(w, pi + 24)
            src = getw(w, pi + 160)
            while j < n:
                put4(w, dst + j * 4, addf(get4(w, dst + j * 4), get4(w, src + j * 4)))
                j = j + 1
        i = i + 1
    if retain == 0:
        return release_backward_scratch(w, limit, epoch)
    return 1
def sgd(w: mutbuf, t: u64, lr: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if getw(w, p + 40) != 0 or getw(w, p + 32) == 0:
        return terror(w, 5)
    if lr > 4294967295 or ltf(lr, 0) != 0 or absf(lr) >= 2139095040:
        return terror(w, 7)
    if getw(w,p+360) == 0:
        return 1
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
