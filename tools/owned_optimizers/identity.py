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
