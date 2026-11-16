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
