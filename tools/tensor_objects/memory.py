def object_address(w:mutbuf, address:u64) -> u64:
    if address >= 9223372036854775808:
        return 139264 + so_data_base(so_uget(w,16)) + (address & 9223372036854775807)
    return address

def getw(w:mutbuf, p:u64) -> u64:
    p = object_address(w,p)
    x = 0
    j = 0
    while j < 8:
        x = x | (byte(w,p+j) << (j*8))
        j = j+1
    return x

def putw(w:mutbuf, p:u64, x:u64) -> u64:
    p = object_address(w,p)
    j = 0
    while j < 8:
        store(w,p+j,x >> (j*8))
        j = j+1
    return x

def get4(w:mutbuf, p:u64) -> u64:
    p = object_address(w,p)
    return byte(w,p) | (byte(w,p+1)<<8) | (byte(w,p+2)<<16) | (byte(w,p+3)<<24)

def put4(w:mutbuf, p:u64, x:u64) -> u64:
    p = object_address(w,p)
    store(w,p,x)
    store(w,p+1,x>>8)
    store(w,p+2,x>>16)
    store(w,p+3,x>>24)
    return x

def memory_fill(w:mutbuf, address:u64, size:u64, value:u64) -> u64:
    fill_range(w,object_address(w,address),size,value)
    return 0

def storage_capacity(w:mutbuf, kind:u64, amount:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    slots = so_uget(w,16)
    pool = so_uget(w,24)
    need_slots = slots
    need_pool = pool
    if kind != 0 and so_slot_available(w,kind) == 0:
        if slots == 65536:
            return terror(w,3)
        need_slots = slots*2
    if amount > 1073741824:
        return terror(w,3)
    if so_extent_room(w,amount) == 0:
        amount = so_align16(amount)
        if amount > 1073741824-pool:
            return terror(w,3)
        while need_pool < pool+amount:
            need_pool = need_pool*2
    if need_slots > slots or need_pool > pool:
        if need_slots > getw(w,48):
            putw(w,48,need_slots)
        if need_pool > getw(w,56):
            putw(w,56,need_pool)
        return terror(w,3)
    return 1

def alloc(w:mutbuf, amount:u64) -> u64:
    if storage_capacity(w,2,amount) == 0:
        return 0
    if storage_capacity(w,6,0) == 0:
        return 0
    extent = so_extent_allocate(w,amount)
    if extent == 0:
        return terror(w,9)
    sid = so_slot_new(w,2)
    hid = so_slot_new(w,6)
    if sid == 0 or hid == 0:
        return terror(w,9)
    sp = so_slot_lookup(w,2,sid)
    hp = so_slot_lookup(w,6,hid)
    so_uput(w,sp+16,extent-1)
    so_uput(w,sp+24,amount)
    so_uput(w,sp+32,so_align16(amount))
    so_uput(w,sp+40,1)
    so_uput(w,sp+48,1)
    so_uput(w,hp+16,sid)
    so_uput(w,hp+24,1)
    putw(w,80,hid)
    address = 9223372036854775808 + extent-1
    memory_fill(w,address,amount,0)
    return address

def hold_release(w:mutbuf, id:u64) -> u64:
    if id == 0:
        return 1
    hp = so_slot_lookup(w,6,id)
    if hp == 0:
        return terror(w,9)
    if so_storage_release(w,so_uget(w,hp+16)) != 0:
        return terror(w,9)
    if so_slot_drop(w,6,id) != 0:
        return terror(w,9)
    return 1

def init_state(w:mutbuf, slots:u64, pool:u64) -> u64:
    if length(w) != 139264+so_state_size(slots,pool):
        return 0
    fill_range(w,0,length(w),0)
    so_state_init(w,slots,pool)
    putw(w,8,1)
    putw(w,64,3694451708891714632)
    return 1

def clone_state(source:readbuf, target:mutbuf, slots:u64, pool:u64) -> u64:
    if so_rget(source,16) == slots and so_rget(source,24) == pool:
        copy_range(target,source,0,0,length(source))
        return 0
    copy_range(target,source,0,0,139264)
    return so_state_migrate(source,target,slots,pool)

def object_size(slots:u64, pool:u64) -> u64:
    return 139264+so_state_size(slots,pool)

def state_header(w:mutbuf) -> u64:
    if length(w) == 0:
        return 0
    if length(w) < 139264+1840+131072:
        return 7
    if so_state_header(w) != 0 or getw(w,64) != 3694451708891714632:
        return 7
    if getw(w,0) != 0 or getw(w,48) != 0 or getw(w,56) != 0:
        return 7
    if getw(w,88) != 0 or getw(w,96) != 0 or getw(w,104) != 0 or getw(w,112) != 0:
        return 7
    next = getw(w,8)
    if next == 0 or next > 65536 or getw(w,40) >= next:
        return 7
    if getw(w,32) == 0 and getw(w,40) != 0:
        return 7
    i = 0
    while i < 256:
        base = getw(w,4096+i*8)
        hid = getw(w,6144+i*8)
        if base != 0:
            hp = so_slot_lookup(w,6,hid)
            if hp == 0 or so_uget(w,hp+24) != 1:
                return 7
            sp = so_storage_live(w,so_uget(w,hp+16))
            if sp == 0 or so_uget(w,sp+24) != 98304:
                return 7
            if base != 9223372036854775808+so_uget(w,sp+16):
                return 7
        else:
            if hid != 0:
                return 7
        i = i+1
    return 0

def dptr(w:mutbuf, t:u64) -> u64:
    page = getw(w,4096+(t >> 8)*8)
    return page+(t & 255)*384

def live_id(w:mutbuf, slot:u64) -> u64:
    return slot

def valid(w:mutbuf, t:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    if t == 0 or t >= getw(w,8):
        return terror(w,1)
    if getw(w,dptr(w,t)+304) == 0:
        return terror(w,1)
    return 1

def public_node(w:mutbuf, id:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    id = resolve_handle(w,id)
    tp = so_tensor_live(w,id)
    if tp == 0:
        return terror(w,1)
    node = so_uget(w,tp+48)
    if node == 0 or node >= getw(w,8):
        return terror(w,1)
    p = dptr(w,node)
    if getw(w,p+304) == 0 or getw(w,p+192) != id or getw(w,p+200) != 1:
        return terror(w,1)
    return node

def public_optional(w:mutbuf, id:u64) -> u64:
    if id == 0:
        return 0
    return public_node(w,id)

def wire_node(s:readbuf, w:mutbuf, at:u64) -> u64:
    return public_node(w,getr(s,at))

def resolve_handle(w:mutbuf, id:u64) -> u64:
    if (id >> 24) == 254:
        ordinal = id & 16777215
        if ordinal == 0 or ordinal > getw(w,104) or getw(w,96) == 0:
            return terror(w,1)
        return getw(w,getw(w,96)+(ordinal-1)*8)
    return id
