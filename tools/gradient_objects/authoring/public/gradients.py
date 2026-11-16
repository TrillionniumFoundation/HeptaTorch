def gradient_binding(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    g = getw(w,p+384)
    if g == 0:
        if getw(w,p+392) != 0 or getw(w,p+360) != 0:
            return terror(w,9)
        return 0
    q = private_tensor(w,g,getw(w,p+392))
    if q == 0 or getw(w,q+400) == 0:
        return terror(w,9)
    return g

def gradient_reference_retain(w:mutbuf, g:u64, incarnation:u64) -> u64:
    q = private_tensor(w,g,incarnation)
    if q == 0:
        return terror(w,9)
    if getw(w,q+400) >= 4294967295:
        return terror(w,8)
    if node_retain(w,g) == 0:
        return 0
    putw(w,q+400,getw(w,q+400)+1)
    return 1

def gradient_reference_release(w:mutbuf, g:u64, incarnation:u64) -> u64:
    q = private_tensor(w,g,incarnation)
    if q == 0 or getw(w,q+400) == 0:
        return terror(w,9)
    putw(w,q+400,getw(w,q+400)-1)
    return node_release(w,g)

def gradient_unbind(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    g = gradient_binding(w,t)
    if getw(w,0) != 0:
        return 0
    if g != 0:
        incarnation = getw(w,p+392)
        putw(w,p+384,0)
        putw(w,p+392,0)
        putw(w,p+360,0)
        if gradient_reference_release(w,g,incarnation) == 0:
            return 0
    return 1

def gradient_shape_matches(w:mutbuf, t:u64, g:u64) -> u64:
    p = dptr(w,t)
    q = dptr(w,g)
    rank = getw(w,p)
    if getw(w,q) != rank:
        return 0
    j = 0
    while j < rank:
        if getw(w,p+96+j*8) != getw(w,q+96+j*8):
            return 0
        j = j+1
    return 1

def gradient_cycle_push(w:mutbuf, t:u64, forbidden:u64) -> u64:
    if t == 0:
        return 1
    if t == forbidden:
        return terror(w,7)
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+416) == 0:
        putw(w,p+416,1)
        putw(w,p+440,getw(w,3984))
        putw(w,3984,t)
    return 1

def gradient_cycle_check(w:mutbuf, t:u64, g:u64) -> u64:
    if getw(w,3984) != 0:
        return terror(w,9)
    gradient_cycle_push(w,g,t)
    budget = 0
    while getw(w,3984) != 0 and getw(w,0) == 0:
        current = getw(w,3984)
        p = dptr(w,current)
        putw(w,3984,getw(w,p+440))
        putw(w,p+440,0)
        gradient_cycle_push(w,getw(w,p+384),t)
        count = gradient_parent_count(w,current)
        if count > 16777216-budget:
            terror(w,3)
        else:
            budget = budget+count
            j = 0
            while j < count and getw(w,0) == 0:
                gradient_cycle_push(w,graph_parent_at(w,current,j),t)
                j = j+1
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        putw(w,p+416,0)
        putw(w,p+440,0)
        i = i+1
    putw(w,3984,0)
    return choose(getw(w,0) == 0,1,0)

def gradient_assign(w:mutbuf, t:u64, g:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if g == 0:
        return gradient_unbind(w,t)
    if valid(w,g) == 0:
        return 0
    if t == g or gradient_shape_matches(w,t,g) == 0:
        return terror(w,2)
    if gradient_binding(w,t) == g:
        return 1
    if gradient_cycle_check(w,t,g) == 0:
        return 0
    incarnation = getw(w,dptr(w,g)+320)
    if gradient_reference_retain(w,g,incarnation) == 0:
        return 0
    if gradient_unbind(w,t) == 0:
        return 0
    putw(w,p+384,g)
    putw(w,p+392,incarnation)
    putw(w,p+360,1)
    return 1

def gradient_get_public(w:mutbuf, id:u64) -> u64:
    t = public_node(w,id)
    if t == 0:
        return 0
    if getw(w,dptr(w,t)+360) == 0:
        return 0
    g = gradient_binding(w,t)
    if g == 0:
        return 0
    return publish_tensor(w,g)

def gradient_present(w:mutbuf, id:u64) -> u64:
    t = public_node(w,id)
    if t == 0:
        return 0
    return getw(w,dptr(w,t)+360)

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
    g = gradient_binding(w,t)
    if g == 0:
        return 0
    return tget(w,g,index)

def gradient_new(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+384) != 0 or getw(w,p+360) != 0:
        return terror(w,9)
    g = descriptor(w,getw(w,p),p+96,0,1)
    if g == 0:
        return 0
    q = dptr(w,g)
    if is_dense(w,t) != 0:
        j = 0
        while j < getw(w,p):
            putw(w,q+128+j*8,getw(w,p+128+j*8))
            j = j+1
    incarnation = getw(w,q+320)
    if gradient_reference_retain(w,g,incarnation) == 0:
        return 0
    putw(w,p+384,g)
    putw(w,p+392,incarnation)
    putw(w,q+200,0)
    return g

def ensure_backward_work(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if getw(w,p+40) == 0 and getw(w,p+32) != 0 and getw(w,p+384) == 0:
                if gradient_new(w,i) == 0:
                    return 0
            if getw(w,p+256) == 0:
                address = alloc(w,getw(w,p+8)*4)
                if address == 0:
                    return 0
                putw(w,p+160,address)
                putw(w,p+256,getw(w,80))
        i = i+1
    return 1

def gradient_accumulate(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+32) == 0:
        return 1
    g = gradient_binding(w,t)
    if g == 0:
        return terror(w,9)
    present = getw(w,p+360)
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
            if hold_release(w,getw(w,p+256)) == 0:
                return 0
            putw(w,p+256,0)
            putw(w,p+160,0)
            if getw(w,p+360) == 0:
                if gradient_unbind(w,i) == 0:
                    return 0
        i = i+1
    return 1

def gradient_detach_inplace(w:mutbuf, g:u64) -> u64:
    p = dptr(w,g)
    if getw(w,p+40) != 0:
        if getw(w,p+408) != 0 or getw(w,p+208) != getw(w,p+312)+getw(w,p+400):
            return terror(w,5)
        j = 0
        while j < gradient_parent_count(w,g):
            if graph_release_edge(w,graph_parent_at(w,g,j),graph_edge_enabled(w,g,j)) == 0:
                return 0
            j = j+1
        saved_release(w,getw(w,p+280))
        saved_release(w,getw(w,p+288))
        saved_release(w,getw(w,p+296))
        hold_release(w,getw(w,p+256))
        hold_release(w,getw(w,p+264))
        hold_release(w,getw(w,p+272))
        putw(w,p+40,0)
        putw(w,p+48,0)
        putw(w,p+56,0)
        putw(w,p+72,0)
        putw(w,p+80,0)
        putw(w,p+160,0)
        putw(w,p+184,0)
        putw(w,p+256,0)
        putw(w,p+264,0)
        putw(w,p+272,0)
        putw(w,p+280,0)
        putw(w,p+288,0)
        putw(w,p+296,0)
        putw(w,p+328,0)
        putw(w,p+344,0)
        putw(w,p+352,0)
    putw(w,p+32,0)
    return choose(getw(w,0) == 0,1,0)

def zero_grad_none(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    if getw(w,dptr(w,t)+40) != 0:
        return terror(w,5)
    return gradient_unbind(w,t)

def zero_grad(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0:
        return terror(w,5)
    if getw(w,p+360) == 0:
        return 1
    g = gradient_binding(w,t)
    if g == 0 or version_can_bump(w,g,1) == 0:
        return 0
    if gradient_detach_inplace(w,g) == 0:
        return 0
    i = 0
    while i < getw(w,p+8):
        put4(w,taddr(w,g,i),0)
        i = i+1
    return bump_version(w,g)

def gradient_object_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64) -> u64:
    code = getr(s,cp)
    a = getr(s,cp+8)
    b = getr(s,cp+16)
    first = getw(w,8)
    result = 0
    field = 3
    while field < 8:
        if getr(s,cp+field*8) != 0:
            terror(w,7)
        field = field+1
    if code != 92 and b != 0:
        terror(w,7)
    if getw(w,0) == 0:
        if code == 91:
            result = gradient_get_public(w,a)
        elif code == 92:
            t = public_node(w,a)
            g = public_optional(w,b)
            if t != 0 and getw(w,0) == 0:
                result = gradient_assign(w,t,g)
        else:
            terror(w,4)
    if getw(w,0) == 0:
        finish_command(w,first)
    status = getw(w,0)
    if status != 0:
        result = 0
    putw(o,at,code)
    putw(o,at+8,status)
    putw(o,at+16,result)
    putw(o,at+24,0)
    return at+32
