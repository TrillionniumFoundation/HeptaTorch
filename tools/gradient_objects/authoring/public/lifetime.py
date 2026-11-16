def gradient_gc_enqueue(w:mutbuf, t:u64) -> u64:
    if t == 0:
        return 1
    p = dptr(w,t)
    if getw(w,p+304) == 0 or getw(w,p+200) != 0 or getw(w,p+432) != 0:
        return 1
    if getw(w,p+208) != 0:
        if getw(w,p+312) != 0 or getw(w,p+400) != 0 or getw(w,p+88) != 0 or getw(w,p+240) == 0:
            return 1
    putw(w,p+432,1)
    putw(w,p+424,0)
    tail = getw(w,3976)
    if tail == 0:
        putw(w,3968,t)
    else:
        putw(w,dptr(w,tail)+424,t)
    putw(w,3976,t)
    return 1

def node_release(w:mutbuf, t:u64) -> u64:
    if t == 0:
        return 1
    p = dptr(w,t)
    refs = getw(w,p+208)
    if refs == 0:
        return terror(w,9)
    putw(w,p+208,refs-1)
    return gradient_gc_enqueue(w,t)

def tensor_row_release(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+240) == 0:
        return 1
    if getw(w,p+400) != 0 or getw(w,p+312) != 0 or getw(w,p+88) != 0:
        return terror(w,9)
    if gradient_unbind(w,t) == 0:
        return 0
    tid = getw(w,p+192)
    tp = so_slot_lookup(w,1,tid)
    if tp == 0:
        return terror(w,9)
    if so_version_release(w,so_uget(w,tp+24)) != 0 or so_storage_release(w,so_uget(w,tp+16)) != 0:
        return terror(w,9)
    if so_slot_drop(w,1,tid) != 0:
        return terror(w,9)
    putw(w,p+240,0)
    return 1

def graph_collect_since(w:mutbuf, first:u64) -> u64:
    i = getw(w,8)-1
    while i >= first and i > 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+200) == 0:
            if getw(w,p+208) == 0:
                gradient_gc_enqueue(w,i)
            elif getw(w,p+312) == 0 and getw(w,p+88) == 0 and getw(w,p+400) == 0:
                tensor_row_release(w,i)
        i = i-1
    budget = 0
    while getw(w,3968) != 0 and getw(w,0) == 0:
        if budget >= 131070:
            return terror(w,9)
        budget = budget+1
        current = getw(w,3968)
        p = dptr(w,current)
        putw(w,3968,getw(w,p+424))
        if getw(w,3968) == 0:
            putw(w,3976,0)
        putw(w,p+424,0)
        putw(w,p+432,0)
        if getw(w,p+304) != 0 and getw(w,p+200) == 0 and getw(w,p+208) == 0:
            graph_node_release(w,current)
        elif getw(w,p+304) != 0 and getw(w,p+200) == 0 and getw(w,p+312) == 0 and getw(w,p+400) == 0 and getw(w,p+88) == 0:
            tensor_row_release(w,current)
    return choose(getw(w,0) == 0,1,0)

def gradient_audit(w:mutbuf) -> u64:
    status = 0
    if getw(w,3968) != 0 or getw(w,3976) != 0 or getw(w,3984) != 0:
        return 9
    i = 1
    while i < getw(w,8) and status == 0:
        p = dptr(w,i)
        if getw(w,p+416) != 0 or getw(w,p+424) != 0 or getw(w,p+432) != 0 or getw(w,p+440) != 0:
            status = 9
        i = i+1
    i = 1
    while i < getw(w,8) and status == 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0:
            if getw(w,p+24) != 0 or getw(w,p+248) != 0 or getw(w,p+360) > 1:
                status = 9
            g = getw(w,p+384)
            if g == 0:
                if getw(w,p+392) != 0 or getw(w,p+360) != 0:
                    status = 9
            else:
                q = private_tensor(w,g,getw(w,p+392))
                if q == 0 or getw(w,p+360) != 1 or gradient_shape_matches(w,i,g) == 0:
                    status = 9
                else:
                    putw(w,q+416,getw(w,q+416)+1)
        i = i+1
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if getw(w,p+304) != 0:
            if getw(w,p+400) != getw(w,p+416):
                status = 9
        elif getw(w,p+400) != 0 or getw(w,p+416) != 0:
            status = 9
        putw(w,p+416,0)
        i = i+1
    return status
