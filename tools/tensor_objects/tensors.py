def descriptor_page(w:mutbuf, slot:u64) -> u64:
    address = 4096+(slot >> 8)*8
    if getw(w,address) == 0:
        page = alloc(w,98304)
        if getw(w,0) != 0:
            return 0
        putw(w,address,page)
        putw(w,6144+(slot >> 8)*8,getw(w,80))
    return 1

def descriptor(w:mutbuf, rank:u64, shape:u64, requires:u64, own:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    if rank > 4 or requires > 1:
        return terror(w,2)
    location = object_address(w,shape)
    if location > length(w) or 32 > length(w)-location:
        return terror(w,2)
    slot = getw(w,8)
    if slot >= 65536:
        return terror(w,3)
    n = 1
    axis = 0
    while axis < rank:
        d = getw(w,shape+axis*8)
        if d > 262144:
            return terror(w,3)
        if d == 0:
            n = 0
        axis = axis+1
    if n != 0:
        axis = 0
        while axis < rank:
            n = n*getw(w,shape+axis*8)
            if n > 16777216:
                return terror(w,3)
            axis = axis+1
    if descriptor_page(w,slot) == 0:
        return 0
    if storage_capacity(w,1,0) == 0:
        return 0
    if own != 0 and storage_capacity(w,3,0) == 0:
        return 0
    p = dptr(w,slot)
    memory_fill(w,p,384,0)
    putw(w,p,rank)
    putw(w,p+8,n)
    putw(w,p+32,requires)
    putw(w,p+64,choose(own!=0,slot,0))
    putw(w,p+200,2)
    putw(w,p+304,1)
    axis = rank
    stride = 1
    while axis > 0:
        axis = axis-1
        d = getw(w,shape+axis*8)
        putw(w,p+96+axis*8,d)
        putw(w,p+128+axis*8,stride)
        if d != 0:
            stride = stride*d
    tid = so_slot_new(w,1)
    tp = so_slot_lookup(w,1,tid)
    putw(w,p+192,tid)
    putw(w,p+240,1)
    so_uput(w,tp+48,slot)
    if own != 0:
        data = alloc(w,n*4)
        if getw(w,0) != 0:
            return 0
        hid = getw(w,80)
        hp = so_slot_lookup(w,6,hid)
        sid = so_uget(w,hp+16)
        vid = so_slot_new(w,3)
        vp = so_slot_lookup(w,3,vid)
        so_uput(w,vp+24,1)
        so_uput(w,tp+16,sid)
        so_uput(w,tp+24,vid)
        so_uput(w,tp+40,n*4)
        putw(w,p+16,data)
        putw(w,p+216,sid)
        putw(w,p+224,vid)
        so_slot_drop(w,6,hid)
    if requires != 0:
        grad = alloc(w,n*4)
        if getw(w,0) != 0:
            return 0
        putw(w,p+24,grad)
        putw(w,p+248,getw(w,80))
        work = alloc(w,n*4)
        if getw(w,0) != 0:
            return 0
        putw(w,p+160,work)
        putw(w,p+256,getw(w,80))
    putw(w,8,slot+1)
    return slot

def bind_alias(w:mutbuf, t:u64, source:u64) -> u64:
    p = dptr(w,t)
    q = dptr(w,source)
    sid = getw(w,q+216)
    vid = getw(w,q+224)
    if so_storage_retain(w,sid) != 0 or so_version_retain(w,vid) != 0:
        return terror(w,9)
    tp = so_slot_lookup(w,1,getw(w,p+192))
    so_uput(w,tp+16,sid)
    so_uput(w,tp+24,vid)
    so_uput(w,tp+32,getw(w,q+232))
    sp = so_storage_live(w,sid)
    so_uput(w,tp+40,so_uget(w,sp+24)-getw(w,q+232))
    putw(w,p+216,sid)
    putw(w,p+224,vid)
    putw(w,p+232,getw(w,q+232))
    return 1

def tget(w:mutbuf, t:u64, index:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if index >= getw(w,p+8):
        return terror(w,7)
    if so_storage_live(w,getw(w,p+216)) == 0:
        return terror(w,1)
    return get4(w,taddr(w,t,index))

def tversion(w:mutbuf, t:u64) -> u64:
    p = so_version_live(w,getw(w,dptr(w,t)+224))
    if p == 0:
        return terror(w,1)
    return so_uget(w,p+16)

def version_can_bump(w:mutbuf, t:u64, amount:u64) -> u64:
    p = so_version_live(w,getw(w,dptr(w,t)+224))
    if p == 0:
        return terror(w,1)
    if amount > 18446744073709551615-so_uget(w,p+16):
        return terror(w,8)
    return 1

def bump_version(w:mutbuf, t:u64) -> u64:
    if version_can_bump(w,t,1) == 0:
        return 0
    p = so_version_live(w,getw(w,dptr(w,t)+224))
    so_uput(w,p+16,so_uget(w,p+16)+1)
    return 1

def tset(w:mutbuf, t:u64, index:u64, bits:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if index >= getw(w,p+8) or bits > 4294967295:
        return terror(w,7)
    if version_can_bump(w,t,1) == 0:
        return 0
    put4(w,taddr(w,t,index),bits)
    return bump_version(w,t)

def zero_grad(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0 or getw(w,p+32) == 0:
        return terror(w,5)
    memory_fill(w,getw(w,p+24),getw(w,p+8)*4,0)
    return 1

def sgd(w:mutbuf, t:u64, lr:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0 or getw(w,p+32) == 0:
        return terror(w,5)
    if lr > 4294967295 or ltf(lr,0) != 0 or absf(lr) >= 2139095040:
        return terror(w,7)
    if version_can_bump(w,t,1) == 0:
        return 0
    n = getw(w,p+8)
    i = 0
    while i < n:
        address = taddr(w,t,i)
        g = get4(w,getw(w,p+24)+i*4)
        put4(w,address,subf(get4(w,address),mulf(lr,g)))
        i = i+1
    return bump_version(w,t)

def is_dense(w:mutbuf, a:u64) -> u64:
    p = dptr(w,a)
    if getw(w,p+8) == 0:
        return 1
    rank = getw(w,p)
    used = 0
    expected = 1
    done = 0
    while done < rank:
        choice = 4
        best = 18446744073709551615
        j = 0
        while j < rank:
            if (used & (1 << j)) == 0 and getw(w,p+96+j*8) > 1:
                stride = getw(w,p+128+j*8)
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
        done = done+1
    return 1

def clone(w:mutbuf, a:u64) -> u64:
    if valid(w,a) == 0:
        return 0
    p = dptr(w,a)
    t = tnew(w,getw(w,p),p+96,getw(w,p+32))
    if t == 0:
        return 0
    q = dptr(w,t)
    if is_dense(w,a) != 0:
        j = 0
        while j < getw(w,p):
            putw(w,q+128+j*8,getw(w,p+128+j*8))
            j = j+1
    i = 0
    while i < getw(w,p+8):
        put4(w,taddr(w,t,i),tget(w,a,i))
        i = i+1
    return record(w,t,17,a,0)

def detach(w:mutbuf, a:u64) -> u64:
    if valid(w,a) == 0:
        return 0
    pa = dptr(w,a)
    rank = getw(w,pa)
    t = descriptor(w,rank,pa+96,0,0)
    if t == 0:
        return 0
    pt = dptr(w,t)
    putw(w,pt+16,getw(w,pa+16))
    putw(w,pt+64,getw(w,pa+64))
    i = 0
    while i < rank:
        putw(w,pt+128+i*8,getw(w,pa+128+i*8))
        i = i+1
    if bind_alias(w,t,a) == 0:
        return 0
    return t
