def saved_create(w:mutbuf, t:u64, layout:u64) -> u64:
    if storage_capacity(w,5,0) == 0:
        return 0
    p = dptr(w,t)
    sid = getw(w,p+216)
    vid = getw(w,p+224)
    if so_storage_retain(w,sid) != 0 or so_version_retain(w,vid) != 0:
        return terror(w,9)
    id = so_slot_new(w,5)
    sp = so_slot_lookup(w,5,id)
    so_uput(w,sp+16,sid)
    so_uput(w,sp+24,vid)
    so_uput(w,sp+32,tversion(w,t))
    so_uput(w,sp+40,getw(w,p+232))
    storage = so_storage_live(w,sid)
    so_uput(w,sp+48,so_uget(w,storage+24)-getw(w,p+232))
    so_uput(w,sp+56,layout)
    return id

def saved_release(w:mutbuf, id:u64) -> u64:
    if id == 0:
        return 1
    p = so_saved_live(w,id)
    if p == 0:
        return terror(w,9)
    sid = so_uget(w,p+16)
    vid = so_uget(w,p+24)
    layout = so_uget(w,p+56)
    if so_version_release(w,vid) != 0 or so_storage_release(w,sid) != 0:
        return terror(w,9)
    if so_slot_drop(w,5,id) != 0:
        return terror(w,9)
    return hold_release(w,layout)

def graph_saved_check(w:mutbuf, id:u64) -> u64:
    if id == 0:
        return 1
    status = so_saved_check(w,id)
    if status == 5:
        return terror(w,6)
    if status != 0:
        return terror(w,9)
    return 1

def node_retain(w:mutbuf, t:u64) -> u64:
    if t == 0:
        return 1
    p = dptr(w,t)
    refs = getw(w,p+208)
    if refs >= 4294967295:
        return terror(w,8)
    putw(w,p+208,refs+1)
    return 1

def node_release(w:mutbuf, t:u64) -> u64:
    if t == 0:
        return 1
    p = dptr(w,t)
    refs = getw(w,p+208)
    if refs == 0:
        return terror(w,9)
    putw(w,p+208,refs-1)
    return 1

def record(w:mutbuf, t:u64, op:u64, a:u64, b:u64) -> u64:
    if t == 0 or getw(w,0) != 0:
        return 0
    p = dptr(w,t)
    putw(w,p+40,op)
    putw(w,p+48,a)
    putw(w,p+56,b)
    if op == 10 or op == 16:
        if bind_alias(w,t,a) == 0:
            return 0
    if getw(w,p+32) == 0:
        return t
    if op == 37 or op == 38:
        saved = getw(w,p+184)
        count = getw(w,saved)
        j = 0
        while j < count:
            node_retain(w,getw(w,saved+8+j*16))
            j = j+1
    else:
        node_retain(w,a)
        if b != 0:
            node_retain(w,b)
    req_a = getw(w,dptr(w,a)+32)
    req_b = 0
    if b != 0:
        req_b = getw(w,dptr(w,b)+32)
    save_a = 0
    save_b = 0
    save_self = 0
    if op == 3:
        save_a = req_b
        save_b = req_a
    if op == 4:
        save_a = req_b
        save_b = req_a | req_b
    if op == 9 or op == 25:
        save_a = req_b
        save_b = req_a
    if op == 22:
        save_a = 1
        save_b = 1
    if op == 13 or op == 23 or op == 24 or op == 27:
        save_a = 1
    if op == 40:
        save_b = req_b
    if op == 5 or op == 11 or op == 12 or op == 14 or op == 15 or op == 20 or op == 21:
        save_self = 1
    if save_a != 0:
        copy = 0
        if op == 25:
            copy = matmul_copy_side(w,a,b,0)
        if copy != 0:
            putw(w,p+280,saved_copy(w,a))
            putw(w,p+344,1)
        else:
            putw(w,p+280,saved_create(w,a,0))
    if save_b != 0:
        copy = 0
        if op == 25:
            copy = matmul_copy_side(w,a,b,1)
        if copy != 0:
            putw(w,p+288,saved_copy(w,b))
            putw(w,p+352,1)
        else:
            putw(w,p+288,saved_create(w,b,0))
    if save_self != 0:
        putw(w,p+296,saved_create(w,t,0))
    if getw(w,0) != 0:
        return 0
    return t

def tensor_row_release(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+240) == 0:
        return 1
    tid = getw(w,p+192)
    tp = so_slot_lookup(w,1,tid)
    if tp == 0:
        return terror(w,9)
    if so_version_release(w,so_uget(w,tp+24)) != 0:
        return terror(w,9)
    if so_storage_release(w,so_uget(w,tp+16)) != 0:
        return terror(w,9)
    if so_slot_drop(w,1,tid) != 0:
        return terror(w,9)
    putw(w,p+240,0)
    return 1

def graph_node_release(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+32) != 0 and getw(w,p+40) != 0:
        op = getw(w,p+40)
        if op == 37 or op == 38:
            saved = getw(w,p+184)
            j = 0
            while j < getw(w,saved):
                node_release(w,getw(w,saved+8+j*16))
                j = j+1
        else:
            node_release(w,getw(w,p+48))
            node_release(w,getw(w,p+56))
    saved_release(w,getw(w,p+280))
    saved_release(w,getw(w,p+288))
    saved_release(w,getw(w,p+296))
    tensor_row_release(w,t)
    hold_release(w,getw(w,p+248))
    hold_release(w,getw(w,p+256))
    hold_release(w,getw(w,p+264))
    hold_release(w,getw(w,p+272))
    hold_release(w,getw(w,p+320))
    putw(w,p+304,0)
    return 1

def graph_collect_since(w:mutbuf, first:u64) -> u64:
    i = getw(w,8)-1
    while i >= first and i > 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+200) == 0:
            if getw(w,p+208) == 0:
                graph_node_release(w,i)
            else:
                if getw(w,p+40) != 0 or getw(w,p+32) == 0:
                    tensor_row_release(w,i)
        i = i-1
    return choose(getw(w,0)==0,1,0)

def graph_collect(w:mutbuf) -> u64:
    return graph_collect_since(w,1)

def publish_tensor(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    putw(w,p+200,1)
    return getw(w,p+192)

def finish_command(w:mutbuf, first:u64) -> u64:
    i = first
    while i < getw(w,8):
        p = dptr(w,i)
        if getw(w,p+200) == 2:
            putw(w,p+200,0)
        i = i+1
    return graph_collect_since(w,first)

def drop_tensor(w:mutbuf, id:u64) -> u64:
    t = public_node(w,id)
    if t == 0:
        return 0
    putw(w,dptr(w,t)+200,0)
    return graph_collect(w)

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
    i = keep+1
    while i < old_next:
        p = dptr(w,i)
        if getw(w,p+304) != 0:
            return terror(w,9)
        memory_fill(w,p,384,0)
        i = i+1
    page = (keep >> 8)+1
    while page < 256:
        hold_release(w,getw(w,6144+page*8))
        putw(w,4096+page*8,0)
        putw(w,6144+page*8,0)
        page = page+1
    putw(w,8,keep+1)
    putw(w,40,keep)
    putw(w,32,getw(w,32)+1)
    return 1

def mark_concat(w:mutbuf, t:u64, epoch:u64) -> u64:
    saved = getw(w,dptr(w,t)+184)
    j = 0
    while j < getw(w,saved):
        a = getw(w,saved+8+j*16)
        p = dptr(w,a)
        if getw(w,p+32) != 0:
            putw(w,p+168,epoch)
        j = j+1
    return 1

def mark_graph(w:mutbuf, t:u64, epoch:u64) -> u64:
    putw(w,dptr(w,t)+168,epoch)
    i = t
    while i > 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch and getw(w,p+40) != 0:
            if graph_saved_check(w,getw(w,p+280)) == 0:
                return 0
            if graph_saved_check(w,getw(w,p+288)) == 0:
                return 0
            if graph_saved_check(w,getw(w,p+296)) == 0:
                return 0
            if indexed_add_backward_check(w,i) == 0:
                return 0
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

def back_binary(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    a = getw(w,p+48)
    b = getw(w,p+56)
    req_a = getw(w,dptr(w,a)+32)
    req_b = getw(w,dptr(w,b)+32)
    op = getw(w,p+40)
    n = getw(w,p+8)
    i = 0
    while i < n:
        ai = bindex(w,t,a,i)
        bi = bindex(w,t,b,i)
        g = get4(w,getw(w,p+160)+i*4)
        if req_a != 0:
            da = g
            if op == 3:
                da = mulf(g,tget(w,b,bi))
            if op == 4:
                da = divf(g,tget(w,b,bi))
            gadd(w,a,ai,da)
        if req_b != 0:
            db = g
            if op == 2:
                db = negf(g)
            if op == 3:
                db = mulf(g,tget(w,a,ai))
            if op == 4:
                x = tget(w,a,ai)
                y = tget(w,b,bi)
                db = negf(mulf(g,divf(divf(x,y),y)))
            gadd(w,b,bi,db)
        i = i+1
    return 1
