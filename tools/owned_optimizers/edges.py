def graph_parent_at(w:mutbuf, t:u64, index:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 37 or op == 38:
        return getw(w,getw(w,p+184)+8+index*16)
    return getw(w,p+48+index*8)

def graph_edge_enabled(w:mutbuf, t:u64, index:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 0:
        return 0
    if op == 37 or op == 38:
        return getw(w,getw(w,p+184)+16+index*16)
    return getw(w,p+72+index*8)

def record(w:mutbuf, t:u64, op:u64, a:u64, b:u64) -> u64:
    result = record_body(w,t,op,a,b)
    if result == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+32) == 0:
        putw(w,p+40,0)
        putw(w,p+48,0)
        putw(w,p+56,0)
        return result
    count = gradient_parent_count(w,t)
    i = 0
    while i < count:
        parent = graph_parent_at(w,t,i)
        flag = 0
        if parent != 0:
            q = dptr(w,parent)
            flag = getw(w,q+32)
            if flag != 0 and getw(w,q+40) == 0:
                if getw(w,q+88) >= 4294967295:
                    return terror(w,8)
                putw(w,q+88,getw(w,q+88)+1)
        if op == 37 or op == 38:
            putw(w,getw(w,p+184)+16+i*16,flag)
        else:
            putw(w,p+72+i*8,flag)
        i = i+1
    return result

def graph_release_edge(w:mutbuf, parent:u64, captured:u64) -> u64:
    if parent == 0:
        return 1
    q = dptr(w,parent)
    if captured != 0 and getw(w,q+40) == 0:
        refs = getw(w,q+88)
        if refs == 0:
            return terror(w,9)
        putw(w,q+88,refs-1)
    return node_release(w,parent)

def graph_node_release(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+312) != 0 or getw(w,p+88) != 0:
        return terror(w,9)
    if getw(w,p+40) != 0:
        j = 0
        while j < gradient_parent_count(w,t):
            if graph_release_edge(w,graph_parent_at(w,t,j),graph_edge_enabled(w,t,j)) == 0:
                return 0
            j = j+1
    saved_release(w,getw(w,p+280))
    saved_release(w,getw(w,p+288))
    saved_release(w,getw(w,p+296))
    tensor_row_release(w,t)
    hold_release(w,getw(w,p+248))
    hold_release(w,getw(w,p+256))
    hold_release(w,getw(w,p+264))
    hold_release(w,getw(w,p+272))
    putw(w,p+304,0)
    return choose(getw(w,0) == 0,1,0)

def gradient_parent_at(w:mutbuf, t:u64, index:u64) -> u64:
    if graph_edge_enabled(w,t,index) == 0:
        return 0
    return graph_parent_at(w,t,index)

def backward_getw(w:mutbuf, address:u64) -> u64:
    t = getw(w,112)
    if t != 0:
        i = 0
        count = gradient_parent_count(w,t)
        while i < count:
            parent = graph_parent_at(w,t,i)
            if parent != 0 and address == dptr(w,parent)+32:
                return graph_edge_enabled(w,t,i)
            i = i+1
    return getw(w,address)

def gadd(w:mutbuf, t:u64, index:u64, value:u64) -> u64:
    p = dptr(w,t)
    flags = getw(w,p+368)
    if flags == 0:
        return 0
    address = getw(w,p+160)+index*4
    if flags > 1:
        flag = object_address(w,flags+index)
        if byte(w,flag) == 0:
            store(w,flag,1)
            put4(w,address,value)
            return 1
    put4(w,address,addf(get4(w,address),value))
    return 1

def mark_concat(w:mutbuf, t:u64, epoch:u64) -> u64:
    j = 0
    while j < gradient_parent_count(w,t):
        parent = gradient_parent_at(w,t,j)
        if parent != 0:
            putw(w,dptr(w,parent)+168,epoch)
        j = j+1
    return 1

def mark_graph(w:mutbuf, t:u64, epoch:u64) -> u64:
    putw(w,dptr(w,t)+168,epoch)
    i = t
    while i > 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch and getw(w,p+40) != 0:
            if getw(w,p+328) != 0:
                return terror(w,10)
            if graph_saved_check(w,getw(w,p+280)) == 0:
                return 0
            if graph_saved_check(w,getw(w,p+288)) == 0:
                return 0
            if graph_saved_check(w,getw(w,p+296)) == 0:
                return 0
            if indexed_add_backward_check(w,i) == 0:
                return 0
            mark_concat(w,i,epoch)
        i = i-1
    return 1

def partial_mark(w:mutbuf, t:u64, epoch:u64) -> u64:
    putw(w,dptr(w,t)+168,epoch)
    i = t
    while i > 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch and getw(w,p+40) != 0:
            mark_concat(w,i,epoch)
        i = i-1
    return 1

def partial_parent(w:mutbuf, queue:u64, t:u64, apply:u64) -> u64:
    if t == 0:
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
    j = 0
    while j < gradient_parent_count(w,t):
        if partial_parent(w,queue,gradient_parent_at(w,t,j),apply) == 0:
            return 0
        j = j+1
    return 1

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
    cursor = scratch
    j = 0
    count = gradient_parent_count(w,t)
    while j < count:
        parent = gradient_parent_at(w,t,j)
        if parent != 0:
            p = dptr(w,parent)
            if getw(w,p+368) == 0:
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

def gradient_apply(w:mutbuf, t:u64, scratch:u64) -> u64:
    if getw(w,dptr(w,t)+40) == 0:
        return partial_apply_node(w,t)
    putw(w,112,t)
    gradient_prepare(w,t,scratch)
    partial_apply_node(w,t)
    gradient_finish(w,t,choose(getw(w,0) == 0,1,0))
    putw(w,112,0)
    return choose(getw(w,0) == 0,1,0)

def indexed_add_backward_check(w:mutbuf, t:u64) -> u64:
    putw(w,112,t)
    result = indexed_add_backward_check_body(w,t)
    putw(w,112,0)
    return result

def release_backward_saved(w:mutbuf, t:u64) -> u64:
    putw(w,112,t)
    result = release_backward_saved_body(w,t)
    putw(w,112,0)
    return result

def graph_audit(w:mutbuf) -> u64:
    status = 0
    budget = 0
    i = 1
    while i < getw(w,8) and status == 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0:
            count = gradient_parent_count(w,i)
            if count > 16777216-budget:
                status = 3
            else:
                budget = budget+count
                j = 0
                while j < count and status == 0:
                    parent = graph_parent_at(w,i,j)
                    flag = graph_edge_enabled(w,i,j)
                    if flag > 1 or parent >= i:
                        status = 9
                    elif parent != 0:
                        q = dptr(w,parent)
                        if getw(w,q+304) == 0:
                            status = 9
                        else:
                            putw(w,q+376,getw(w,q+376)+1)
                            if flag != 0 and getw(w,q+40) == 0:
                                putw(w,q+368,getw(w,q+368)+1)
                    elif flag != 0:
                        status = 9
                    j = j+1
        i = i+1
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if status == 0 and getw(w,p+304) != 0:
            if getw(w,p+88) != getw(w,p+368) or getw(w,p+208) != getw(w,p+376)+getw(w,p+312):
                status = 9
        putw(w,p+368,0)
        putw(w,p+376,0)
        i = i+1
    return status
