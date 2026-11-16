def private_hold(w:mutbuf, id:u64, size:u64) -> u64:
    p = so_slot_lookup(w,6,id)
    if p == 0 or so_uget(w,p+24) != 1:
        return 0
    sp = so_storage_live(w,so_uget(w,p+16))
    if sp == 0 or so_uget(w,sp+24) != size:
        return 0
    return 9223372036854775808+so_uget(w,sp+16)

def public_saved(w:mutbuf, id:u64) -> u64:
    p = so_saved_live(w,id)
    if p == 0 or so_uget(w,p+56) == 0:
        return terror(w,1)
    if private_hold(w,so_uget(w,p+56),128) == 0:
        return terror(w,1)
    return p

def public_hold(w:mutbuf, id:u64) -> u64:
    p = so_slot_lookup(w,6,id)
    if p == 0 or so_uget(w,p+24) != 0:
        return terror(w,1)
    if so_storage_live(w,so_uget(w,p+16)) == 0:
        return terror(w,1)
    return p

def save_public(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    layout = alloc(w,128)
    if layout == 0:
        return 0
    hid = getw(w,80)
    putw(w,layout,getw(w,p))
    putw(w,layout+8,getw(w,p+8))
    j = 0
    while j < 4:
        putw(w,layout+32+j*8,getw(w,p+96+j*8))
        putw(w,layout+64+j*8,getw(w,p+128+j*8))
        j = j+1
    return saved_create(w,t,hid)

def saved_public_check(w:mutbuf, id:u64) -> u64:
    if public_saved(w,id) == 0:
        return 0
    return graph_saved_check(w,id)

def saved_public_release(w:mutbuf, id:u64) -> u64:
    if public_saved(w,id) == 0:
        return 0
    return saved_release(w,id)

def weak_create(w:mutbuf, t:u64) -> u64:
    sid = getw(w,dptr(w,t)+216)
    p = so_storage_live(w,sid)
    if p == 0:
        return terror(w,1)
    if so_uget(w,p+48) >= 4294967295:
        return terror(w,8)
    if storage_capacity(w,4,0) == 0:
        return 0
    id = so_slot_new(w,4)
    so_uput(w,so_slot_lookup(w,4,id)+16,sid)
    so_uput(w,p+48,so_uget(w,p+48)+1)
    return id

def weak_lock(w:mutbuf, id:u64) -> u64:
    wp = so_slot_lookup(w,4,id)
    if wp == 0:
        return terror(w,1)
    sid = so_uget(w,wp+16)
    if so_storage_live(w,sid) == 0:
        return terror(w,6)
    if storage_capacity(w,6,0) == 0:
        return 0
    status = so_storage_retain(w,sid)
    if status != 0:
        return terror(w,status)
    hid = so_slot_new(w,6)
    so_uput(w,so_slot_lookup(w,6,hid)+16,sid)
    return hid

def weak_release(w:mutbuf, id:u64) -> u64:
    p = so_slot_lookup(w,4,id)
    if p == 0:
        return terror(w,1)
    if so_weak_storage_release(w,so_uget(w,p+16)) != 0:
        return terror(w,9)
    if so_slot_drop(w,4,id) != 0:
        return terror(w,9)
    return 1

def public_hold_release(w:mutbuf, id:u64) -> u64:
    if public_hold(w,id) == 0:
        return 0
    return hold_release(w,id)

def saved_logical(w:mutbuf, id:u64, index:u64) -> u64:
    p = public_saved(w,id)
    if p == 0:
        return 0
    if graph_saved_check(w,id) == 0:
        return 0
    layout = private_hold(w,so_uget(w,p+56),128)
    if index >= getw(w,layout+8):
        return terror(w,2)
    axis = getw(w,layout)
    off = 0
    while axis > 0:
        axis = axis-1
        dim = getw(w,layout+32+axis*8)
        co = index % dim
        index = index // dim
        off = off+co*getw(w,layout+64+axis*8)
    sp = so_storage_live(w,so_uget(w,p+16))
    return get4(w,9223372036854775808+so_uget(w,sp+16)+so_uget(w,p+40)+off*4)

def hold_logical(w:mutbuf, id:u64, index:u64) -> u64:
    hp = public_hold(w,id)
    if hp == 0:
        return 0
    sp = so_storage_live(w,so_uget(w,hp+16))
    if index >= so_uget(w,sp+24)//4:
        return terror(w,2)
    return get4(w,9223372036854775808+so_uget(w,sp+16)+index*4)

def data_alias(w:mutbuf, t:u64) -> u64:
    if storage_capacity(w,3,0) == 0:
        return 0
    result = detach(w,t)
    if result == 0:
        return 0
    p = dptr(w,result)
    old = getw(w,p+224)
    vid = so_slot_new(w,3)
    if vid == 0:
        return terror(w,9)
    so_uput(w,so_slot_lookup(w,3,vid)+24,1)
    so_version_release(w,old)
    tp = so_slot_lookup(w,1,getw(w,p+192))
    so_uput(w,tp+24,vid)
    putw(w,p+224,vid)
    return result

def unpack_saved(w:mutbuf, id:u64) -> u64:
    p = public_saved(w,id)
    if p == 0:
        return 0
    if graph_saved_check(w,id) == 0:
        return 0
    layout = private_hold(w,so_uget(w,p+56),128)
    t = descriptor(w,getw(w,layout),layout+32,0,0)
    if t == 0:
        return 0
    tp = dptr(w,t)
    sid = so_uget(w,p+16)
    vid = so_uget(w,p+24)
    if so_storage_retain(w,sid) != 0 or so_version_retain(w,vid) != 0:
        return terror(w,9)
    record = so_slot_lookup(w,1,getw(w,tp+192))
    so_uput(w,record+16,sid)
    so_uput(w,record+24,vid)
    so_uput(w,record+32,so_uget(w,p+40))
    so_uput(w,record+40,so_uget(w,p+48))
    putw(w,tp+216,sid)
    putw(w,tp+224,vid)
    putw(w,tp+232,so_uget(w,p+40))
    sp = so_storage_live(w,sid)
    putw(w,tp+16,9223372036854775808+so_uget(w,sp+16)+so_uget(w,p+40))
    j = 0
    while j < getw(w,layout):
        putw(w,tp+128+j*8,getw(w,layout+64+j*8))
        j = j+1
    return t

def object_stats(w:mutbuf, o:mutbuf, at:u64) -> u64:
    putw(o,at,so_uget(w,16))
    putw(o,at+8,so_uget(w,24))
    putw(o,at+16,so_uget(w,40))
    putw(o,at+24,length(w))
    free = 128+so_uget(w,16)*384
    available = 0
    j = 0
    while j < so_uget(w,32):
        available = available+so_uget(w,free+j*16+8)
        j = j+1
    putw(o,at+32,available)
    kind = 1
    while kind <= 6:
        live = 0
        j = 1
        while j <= so_uget(w,16):
            if so_uget(w,so_slot_at(w,kind,j)+8) == 1:
                live = live+1
            j = j+1
        putw(o,at+32+kind*8,live)
        kind = kind+1
    putw(o,at+88,getw(w,8)-1)
    return 96
