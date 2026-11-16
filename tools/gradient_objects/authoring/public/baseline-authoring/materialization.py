def gradient_parent_count(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 0:
        return 0
    if op == 37 or op == 38:
        return getw(w,getw(w,p+184))
    return 2

def gradient_parent_at(w:mutbuf, t:u64, index:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 37 or op == 38:
        return getw(w,getw(w,p+184)+8+index*16)
    return getw(w,p+48+index*8)

def gradient_direct(w:mutbuf, t:u64, parent:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op >= 1 and op <= 4:
        q = dptr(w,parent)
        rank = getw(w,p)
        if getw(w,q) != rank:
            return 0
        j = 0
        while j < rank:
            if getw(w,p+96+j*8) != getw(w,q+96+j*8):
                return 0
            j = j+1
        return 1
    if op >= 5 and op <= 21:
        return 1
    if op >= 37 and op <= 40:
        return 1
    return 0

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
                    q = dptr(w,parent)
                    if getw(w,q+32) != 0:
                        part = getw(w,q+8)*5
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
            if getw(w,p+32) != 0 and getw(w,p+368) == 0:
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

def gradient_finish(w:mutbuf, t:u64, commit:u64) -> u64:
    j = 0
    count = gradient_parent_count(w,t)
    while j < count:
        parent = gradient_parent_at(w,t,j)
        if parent != 0:
            p = dptr(w,parent)
            if getw(w,p+368) != 0:
                src = getw(w,p+160)
                dst = getw(w,p+376)
                putw(w,p+160,dst)
                putw(w,p+368,0)
                putw(w,p+376,0)
                if commit != 0:
                    present = getw(w,p+336)
                    i = 0
                    n = getw(w,p+8)
                    while i < n:
                        value = get4(w,src+i*4)
                        if present != 0:
                            value = addf(get4(w,dst+i*4),value)
                        put4(w,dst+i*4,value)
                        i = i+1
                    putw(w,p+336,1)
        j = j+1
    return 1

def gradient_apply(w:mutbuf, t:u64, scratch:u64) -> u64:
    if getw(w,dptr(w,t)+40) == 0:
        return partial_apply_node(w,t)
    gradient_prepare(w,t,scratch)
    partial_apply_node(w,t)
    gradient_finish(w,t,choose(getw(w,0) == 0,1,0))
    return choose(getw(w,0) == 0,1,0)

def gradient_clear_marks(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if getw(w,p+368) != 0 or getw(w,p+376) != 0:
                return terror(w,9)
            putw(w,p+336,0)
        i = i+1
    return 1

def gadd(w:mutbuf, t:u64, index:u64, value:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+32) == 0:
        return 0
    address = getw(w,p+160)+index*4
    flags = getw(w,p+368)
    if flags > 1:
        flag = object_address(w,flags+index)
        if byte(w,flag) == 0:
            store(w,flag,1)
            put4(w,address,value)
            return 1
    put4(w,address,addf(get4(w,address),value))
    return 1

def partial_apply_node(w: mutbuf, t: u64) -> u64:
    p = dptr(w, t)
    op = getw(w, p + 40)
    if op == 0:
        present = getw(w,p+360)
        j = 0
        n = getw(w, p + 8)
        dst = getw(w, p + 24)
        src = getw(w, p + 160)
        while j < n:
            value = get4(w,src+j*4)
            if present != 0:
                value = addf(get4(w,dst+j*4),value)
            put4(w,dst+j*4,value)
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
def partial_execute(w: mutbuf, o: mutbuf, info: u64, t: u64, seed: u64, retain: u64) -> u64:
    if partial_admit(w, t, seed) == 0:
        return 0
    epoch = getw(w, 24) + 1
    if epoch == 0:
        return terror(w, 8)
    putw(w, 24, epoch)
    partial_mark(w, t, epoch)
    queue_bytes = 24 + (t + 1) * 16
    stage_bytes = gradient_workspace_size(w,t,epoch)
    if getw(w,0) != 0:
        return 0
    queue = alloc(w,queue_bytes+stage_bytes)
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
            putw(w,p+336,0)
        i = i + 1
    p = dptr(w, t)
    if seed == 0:
        put4(w, getw(w, p + 160), 1065353216)
    else:
        i = 0
        while i < getw(w, p + 8):
            put4(w, getw(w, p + 160) + i * 4, tget(w, seed, i))
            i = i + 1
    putw(w,p+336,1)
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
            gradient_apply(w,current,queue+queue_bytes)
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
    if gradient_clear_marks(w,t,epoch) == 0:
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
    stage_bytes = gradient_workspace_size(w,limit,epoch)
    if getw(w,0) != 0:
        return 0
    scratch = alloc(w,stage_bytes+1)
    if scratch == 0:
        return 0
    scratch_owner = getw(w,80)
    i = 1
    while i <= limit:
        current = live_id(w, i)
        pi = dptr(w, current)
        if getw(w, pi + 168) == epoch:
            memory_fill(w, getw(w, pi + 160), getw(w, pi + 8) * 4, 0)
            putw(w,pi+336,0)
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
    putw(w,p+336,1)
    i = limit
    while i > 0:
        current = live_id(w, i)
        pi = dptr(w, current)
        if getw(w, pi + 168) == epoch:
            op = getw(w, pi + 40)
            if op != 0:
                gradient_apply(w,current,scratch)
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
            partial_apply_node(w,current)
        i = i + 1
    if gradient_clear_marks(w,limit,epoch) == 0:
        return 0
    if hold_release(w,scratch_owner) == 0:
        return 0
    if retain == 0:
        return release_backward_scratch(w, limit, epoch)
    return 1
