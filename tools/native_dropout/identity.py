def history_order(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    order = getw(w,p+632)
    if order == 0:
        return getw(w,p+320)
    return order

def history_accumulator_target(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+240) != 0:
        return 0
    target = getw(w,p+608)
    if target >= 65536:
        return 0
    if target == 0:
        return 0
    q = private_tensor(w,target,getw(w,p+624))
    if q == 0 or getw(w,q+600) == 0 or getw(w,p+40) != 0:
        return terror(w,9)
    return target

def history_accumulator_retain(w:mutbuf, t:u64, target:u64) -> u64:
    p = dptr(w,t)
    q = dptr(w,target)
    if getw(w,p+608) != 0 or getw(w,p+624) != 0:
        return terror(w,9)
    if getw(w,q+600) >= 4294967295:
        return terror(w,8)
    if node_retain(w,target) == 0:
        return 0
    putw(w,q+600,getw(w,q+600)+1)
    putw(w,p+608,target)
    putw(w,p+624,getw(w,q+320))
    return 1

def history_accumulator_release(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    target = history_accumulator_target(w,t)
    if getw(w,0) != 0:
        return 0
    if target != 0:
        q = dptr(w,target)
        if (getw(w,q+608) & 65535) == t and getw(w,q+624) == getw(w,p+320):
            putw(w,q+608,getw(w,q+608) & 131072)
            putw(w,q+624,0)
        putw(w,p+608,0)
        putw(w,p+624,0)
        putw(w,q+600,getw(w,q+600)-1)
        return node_release(w,target)
    if getw(w,p+624) != 0 and getw(w,p+608) == 0:
        return terror(w,9)
    return 1

def history_snapshot(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+32) == 0 or getw(w,p+240) != 1:
        return terror(w,9)
    slot = getw(w,8)
    if slot >= 65536:
        return terror(w,3)
    if getw(w,120) == 18446744073709551615:
        return terror(w,8)
    if descriptor_page(w,slot) == 0:
        return 0
    q = dptr(w,slot)
    memory_fill(w,q,640,0)
    putw(w,q,getw(w,p))
    putw(w,q+8,getw(w,p+8))
    putw(w,q+16,getw(w,p+16))
    putw(w,q+32,1)
    putw(w,q+216,getw(w,p+216))
    putw(w,q+224,getw(w,p+224))
    putw(w,q+232,getw(w,p+232))
    putw(w,q+304,1)
    putw(w,120,getw(w,120)+1)
    putw(w,q+320,getw(w,120))
    putw(w,q+632,history_order(w,t))
    field = 96
    while field < 160:
        putw(w,q+field,getw(w,p+field))
        field = field+8
    putw(w,8,slot+1)
    return slot

def history_edge_address(w:mutbuf, t:u64, index:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 37 or op == 38:
        return getw(w,p+184)+8+index*16
    return p+48+index*8

def history_rebind_captured_edges(w:mutbuf, t:u64, prior:u64) -> u64:
    p = dptr(w,t)
    q = dptr(w,prior)
    budget = 0
    i = 1
    while i < getw(w,8):
        ip = dptr(w,i)
        if i != t and i != prior and getw(w,ip+304) != 0:
            count = gradient_parent_count(w,i)
            if count > 16777216-budget:
                return terror(w,3)
            budget = budget+count
            j = 0
            while j < count:
                if graph_parent_at(w,i,j) == t and graph_edge_enabled(w,i,j) != 0:
                    if node_retain(w,prior) == 0:
                        return 0
                    if getw(w,p+40) == 0:
                        refs = getw(w,p+88)
                        if refs == 0 or getw(w,q+88) >= 4294967295:
                            return terror(w,9)
                        putw(w,p+88,refs-1)
                        putw(w,q+88,getw(w,q+88)+1)
                    putw(w,history_edge_address(w,i,j),prior)
                    if node_release(w,t) == 0:
                        return 0
                j = j+1
        i = i+1
    return 1

def history_transfer_role(w:mutbuf, p:u64, q:u64, field:u64) -> u64:
    putw(w,q+field,getw(w,p+field))
    putw(w,p+field,0)
    return 1

def history_capture(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    cached = getw(w,p+608) & 65535
    if getw(w,p+40) == 0 and cached != 0:
        q = dptr(w,cached)
        if getw(w,q+320) != getw(w,p+624) or history_accumulator_target(w,cached) != t:
            return terror(w,9)
        if getw(w,120) == 18446744073709551615:
            return terror(w,8)
        if history_rebind_captured_edges(w,t,cached) == 0:
            return 0
        if getw(w,q+256) == 0:
            history_transfer_role(w,p,q,160)
            history_transfer_role(w,p,q,256)
        else:
            if hold_release(w,getw(w,p+256)) == 0:
                return 0
            putw(w,p+160,0)
            putw(w,p+256,0)
        putw(w,120,getw(w,120)+1)
        putw(w,p+632,getw(w,120))
        return cached
    prior = history_snapshot(w,t)
    if prior == 0:
        return 0
    p = dptr(w,t)
    q = dptr(w,prior)
    if history_rebind_captured_edges(w,t,prior) == 0:
        return 0
    if getw(w,p+40) == 0:
        if history_accumulator_retain(w,prior,t) == 0:
            return 0
        putw(w,p+608,(getw(w,p+608) & 131072) | prior)
        putw(w,p+624,getw(w,q+320))
    elif history_retained_enabled(w,t) != 0:
        putw(w,q+608,t | 65536)
        putw(w,q+624,getw(w,p+320))
        putw(w,p+608,getw(w,p+608) | 131072)
    field = 40
    while field < 64:
        history_transfer_role(w,p,q,field)
        field = field+8
    history_transfer_role(w,p,q,72)
    history_transfer_role(w,p,q,80)
    history_transfer_role(w,p,q,160)
    history_transfer_role(w,p,q,176)
    history_transfer_role(w,p,q,184)
    field = 256
    while field <= 296:
        history_transfer_role(w,p,q,field)
        field = field+8
    history_transfer_role(w,p,q,328)
    history_transfer_role(w,p,q,344)
    history_transfer_role(w,p,q,352)
    history_transfer_role(w,p,q,584)
    history_transfer_role(w,p,q,592)
    putw(w,p+632,getw(w,q+320))
    return prior

def history_push(w:mutbuf, t:u64, epoch:u64) -> u64:
    if t == 0:
        return 1
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+168) != epoch:
        if getw(w,p+416) != 0 or getw(w,p+440) != 0:
            return terror(w,9)
        putw(w,p+168,epoch)
        putw(w,p+416,1)
        putw(w,p+440,getw(w,3984))
        putw(w,3984,t)
    return 1

def partial_mark(w:mutbuf, t:u64, epoch:u64) -> u64:
    if getw(w,3984) != 0:
        return terror(w,9)
    history_push(w,t,epoch)
    budget = 0
    while getw(w,3984) != 0 and getw(w,0) == 0:
        current = getw(w,3984)
        p = dptr(w,current)
        putw(w,3984,getw(w,p+440))
        putw(w,p+416,0)
        putw(w,p+440,0)
        count = gradient_parent_count(w,current)
        if count > 16777216-budget:
            terror(w,3)
        else:
            budget = budget+count
            j = 0
            while j < count and getw(w,0) == 0:
                parent = gradient_parent_at(w,current,j)
                if parent != 0:
                    if history_order(w,parent) >= history_order(w,current):
                        terror(w,9)
                    else:
                        history_push(w,parent,epoch)
                j = j+1
    while getw(w,3984) != 0:
        p = dptr(w,getw(w,3984))
        putw(w,3984,getw(w,p+440))
        putw(w,p+416,0)
        putw(w,p+440,0)
    return choose(getw(w,0) == 0,1,0)

def history_has_rebased_node(w:mutbuf, epoch:u64) -> u64:
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch and getw(w,p+632) != 0:
            return 1
        i = i+1
    return 0

def history_key_less(w:mutbuf, a:u64, b:u64) -> u64:
    leaf_a = a >> 63
    leaf_b = b >> 63
    if leaf_a != leaf_b:
        return choose(leaf_a < leaf_b,1,0)
    return choose(history_order(w,a & 9223372036854775807) < history_order(w,b & 9223372036854775807),1,0)

def history_accumulator_check(w:mutbuf, t:u64) -> u64:
    target = history_accumulator_target(w,t)
    if target == 0:
        return choose(getw(w,0) == 0,1,0)
    if getw(w,dptr(w,target)+40) != 0:
        return terror(w,11)
    return 1

def partial_check_node(w:mutbuf, t:u64) -> u64:
    if history_accumulator_check(w,t) == 0:
        return 0
    return history_original_partial_check_node(w,t)

def history_preflight_saved(w:mutbuf, epoch:u64) -> u64:
    i = getw(w,8)-1
    while i > 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if history_original_partial_check_node(w,i) == 0:
                return 0
        i = i-1
    return 1

def mark_graph(w:mutbuf, t:u64, epoch:u64) -> u64:
    if partial_mark(w,t,epoch) == 0:
        return 0
    return history_preflight_saved(w,epoch)

def history_audit(w:mutbuf) -> u64:
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if getw(w,p+304) != 0:
            if getw(w,p+632) > getw(w,120):
                return 9
            target = getw(w,p+608)
            if getw(w,p+240) != 0:
                if target >> 16 > 2 or (target & 65536) != 0:
                    return 9
                cached = target & 65535
                if cached != 0:
                    if cached >= getw(w,8):
                        return 9
                    q = dptr(w,cached)
                    if getw(w,q+304) == 0 or getw(w,q+320) != getw(w,p+624) or getw(w,q+608) != i or getw(w,q+624) != getw(w,p+320):
                        return 9
                elif getw(w,p+624) != 0:
                    return 9
            elif target >= 65536:
                if target >> 16 != 1 or (target & 65535) == 0 or getw(w,p+40) == 0 or getw(w,p+624) == 0:
                    return 9
            elif target != 0:
                q = private_tensor(w,target,getw(w,p+624))
                if q == 0 or getw(w,p+40) != 0 or getw(w,p+240) != 0 or getw(w,p+200) != 0 or getw(w,p+632) == 0:
                    return 9
                if getw(w,p+384) != 0 or getw(w,p+312) != 0 or getw(w,p+400) != 0:
                    return 9
                putw(w,q+616,getw(w,q+616)+1)
            elif getw(w,p+624) != 0:
                return 9
        elif getw(w,p+608) != 0 or getw(w,p+624) != 0 or getw(w,p+632) != 0:
            return 9
        i = i+1
    return 0

def history_retained_target(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+240) != 0:
        return t
    target = getw(w,p+608)
    if target >> 16 != 1:
        return 0
    target = target & 65535
    if private_tensor(w,target,getw(w,p+624)) == 0:
        return 0
    return target

def history_retained_enabled(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+240) != 0:
        if (getw(w,p+608) & 131072) != 0:
            return 0
        return retained_enabled(w,t)
    target = history_retained_target(w,t)
    if target == 0:
        return 0
    return retained_enabled(w,target)

def history_prepare_leaf(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+40) != 0:
        return 1
    target = history_accumulator_target(w,t)
    if getw(w,0) != 0:
        return 0
    if target != 0:
        p = dptr(w,target)
        if getw(w,p+40) != 0:
            return 1
    else:
        target = t
    if getw(w,p+32) != 0 and getw(w,p+384) == 0:
        if gradient_new(w,target) == 0:
            return 0
    return 1

def history_incoming_obeys_contract(w:mutbuf, t:u64, source:u64) -> u64:
    p = dptr(w,t)
    q = dptr(w,source)
    if is_dense(w,t) == 0:
        return incoming_contiguous(w,t,q+496)
    j = 0
    while j < getw(w,p):
        size = getw(w,p+96+j*8)
        stride = getw(w,q+496+j*8)
        if size != 1:
            if stride != getw(w,p+128+j*8):
                return 0
        elif stride == 0:
            return 0
        j = j+1
    return 1

def history_release_reservations(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            target = history_accumulator_target(w,i)
            if getw(w,0) != 0:
                return 0
            if target != 0 and getw(w,dptr(w,target)+360) == 0:
                if gradient_unbind(w,target) == 0:
                    return 0
        i = i+1
    return 1

def history_backward_root(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    cached = getw(w,p+608) & 65535
    if getw(w,p+40) == 0 and cached != 0:
        q = dptr(w,cached)
        if getw(w,q+304) == 0 or getw(w,q+320) != getw(w,p+624) or history_accumulator_target(w,cached) != t:
            return terror(w,9)
        return cached
    return t

def history_route_new_edges(w:mutbuf, t:u64) -> u64:
    i = 0
    while i < gradient_parent_count(w,t):
        parent = gradient_parent_at(w,t,i)
        if parent != 0:
            p = dptr(w,parent)
            cached = getw(w,p+608) & 65535
            if getw(w,p+240) != 0 and getw(w,p+40) == 0 and cached != 0:
                q = dptr(w,cached)
                if getw(w,q+320) != getw(w,p+624) or history_accumulator_target(w,cached) != parent:
                    return terror(w,9)
                if getw(w,p+88) == 0 or getw(w,q+88) >= 4294967295:
                    return terror(w,9)
                if node_retain(w,cached) == 0:
                    return 0
                putw(w,p+88,getw(w,p+88)-1)
                putw(w,q+88,getw(w,q+88)+1)
                putw(w,history_edge_address(w,t,i),cached)
                if node_release(w,parent) == 0:
                    return 0
        i = i+1
    return 1

def record(w:mutbuf, t:u64, op:u64, a:u64, b:u64) -> u64:
    result = history_original_record(w,t,op,a,b)
    if result == 0:
        return 0
    if history_route_new_edges(w,t) == 0:
        return 0
    return result

def release_backward_scratch(w:mutbuf, limit:u64, epoch:u64) -> u64:
    if history_original_release_backward_scratch(w,limit,epoch) == 0:
        return 0
    return history_release_reservations(w,limit,epoch)

def history_detach_inplace(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+408) != 0:
        return terror(w,5)
    if getw(w,p+40) != 0:
        if history_capture(w,t) == 0:
            return 0
    putw(w,p+32,0)
    return graph_collect_since(w,1)
