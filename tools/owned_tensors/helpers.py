def init(w:mutbuf) -> u64:
    if length(w) < 65536:
        return 0
    fill_range(w,0,length(w),0)
    putw(w,8,1)
    putw(w,16,57344)
    putw(w,64,3622394114853786696)
    putw(w,4096,8192)
    return 1

def state_header(w:mutbuf) -> u64:
    size = length(w)
    if size == 0:
        return 0
    if size < 65536 or size > 1073741824 or (size & (size-1)) != 0:
        return 7
    if getw(w,64) != 3622394114853786696:
        return 7
    if getw(w,0) != 0 or getw(w,48) != 0 or getw(w,56) != 0:
        return 7
    next_slot = getw(w,8)
    arena = getw(w,16)
    generation = getw(w,32)
    keep = getw(w,40)
    if next_slot == 0 or next_slot > 65536 or arena < 57344 or arena > size:
        return 7
    if generation > 281474976710655 or keep >= next_slot:
        return 7
    if generation == 0 and keep != 0:
        return 7
    if getw(w,4096) != 8192:
        return 7
    page_end = 57344
    last_page = (next_slot-1) >> 8
    page = 1
    while page < 256:
        base = getw(w,4096+page*8)
        if page <= last_page:
            if base < page_end or base > arena or (base & 3) != 0:
                return 7
            if 49152 > arena-base:
                return 7
            page_end = base + 49152
        else:
            if base != 0:
                return 7
        page = page + 1
    return 0

def live_id(w:mutbuf, slot:u64) -> u64:
    if slot <= getw(w,40):
        return slot
    return (getw(w,32) << 16) | slot

def valid(w:mutbuf, t:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    slot = t & 65535
    if slot == 0 or slot >= getw(w,8):
        return terror(w,1)
    if t != live_id(w,slot):
        return terror(w,1)
    return 1

def dptr(w:mutbuf, t:u64) -> u64:
    slot = t & 65535
    page = getw(w,4096+(slot >> 8)*8)
    return page + (slot & 255)*192

def alloc(w:mutbuf, amount:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    at = getw(w,16)
    if at > 1073741824 or amount > 1073741824-at:
        return terror(w,3)
    needed = at + amount
    if needed > length(w):
        putw(w,48,needed)
        putw(w,56,1)
        return terror(w,3)
    fill_range(w,at,amount,0)
    putw(w,16,needed)
    return at

def descriptor_page(w:mutbuf, slot:u64) -> u64:
    address = 4096 + (slot >> 8)*8
    if getw(w,address) == 0:
        page = alloc(w,49152)
        if getw(w,0) != 0:
            return 0
        putw(w,address,page)
    return 1

def descriptor(w:mutbuf, rank:u64, shape:u64, requires:u64, own:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    if rank > 4 or requires > 1 or shape > length(w)-32:
        return terror(w,2)
    slot = getw(w,8)
    if slot >= 65536:
        return terror(w,3)
    t = live_id(w,slot)
    n = 1
    axis = 0
    while axis < rank:
        d = getw(w,shape+axis*8)
        if d > 262144:
            return terror(w,3)
        if d == 0:
            n = 0
        axis = axis + 1
    if n != 0:
        axis = 0
        while axis < rank:
            n = n * getw(w,shape+axis*8)
            if n > 16777216:
                return terror(w,3)
            axis = axis + 1
    if descriptor_page(w,slot) == 0:
        return 0
    p = dptr(w,t)
    fill_range(w,p,192,0)
    putw(w,p,rank)
    putw(w,p+8,n)
    putw(w,p+32,requires)
    putw(w,p+64,t)
    axis = rank
    stride = 1
    while axis > 0:
        axis = axis - 1
        d = getw(w,shape+axis*8)
        putw(w,p+96+axis*8,d)
        putw(w,p+128+axis*8,stride)
        if d != 0:
            stride = stride*d
    if own != 0:
        data = alloc(w,n*4)
        putw(w,p+16,data)
    if requires != 0:
        grad = alloc(w,n*4)
        work = alloc(w,n*4)
        putw(w,p+24,grad)
        putw(w,p+160,work)
    if getw(w,0) != 0:
        return 0
    putw(w,8,slot+1)
    return t

def tensor_initialize(w:mutbuf, t:u64, index:u64, bits:u64) -> u64:
    put4(w,taddr(w,t,index),bits)
    return 1

def graph_reset(w:mutbuf, keep:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    generation = getw(w,32)
    old_next = getw(w,8)
    if keep >= old_next or keep > 65535:
        return terror(w,7)
    if generation != 0 and keep != getw(w,40):
        return terror(w,7)
    if generation >= 281474976710655:
        return terror(w,3)
    arena = 57344
    page = 1
    while page <= (keep >> 8):
        page_end = getw(w,4096+page*8)+49152
        if page_end > arena:
            arena = page_end
        page = page + 1
    i = 1
    while i <= keep:
        p = dptr(w,i)
        if getw(w,p+40) != 0 or getw(w,p+64) != i:
            return terror(w,6)
        amount = getw(w,p+8)*4
        data_end = getw(w,p+16)+amount
        if data_end > arena:
            arena = data_end
        if getw(w,p+32) != 0:
            grad_end = getw(w,p+24)+amount
            if grad_end > arena:
                arena = grad_end
            work_end = getw(w,p+160)+amount
            if work_end > arena:
                arena = work_end
        i = i + 1
    i = keep + 1
    while i < old_next:
        fill_range(w,dptr(w,i),192,0)
        i = i + 1
    page = (keep >> 8) + 1
    while page < 256:
        putw(w,4096+page*8,0)
        page = page + 1
    old_arena = getw(w,16)
    fill_range(w,arena,old_arena-arena,0)
    putw(w,8,keep+1)
    putw(w,16,arena)
    putw(w,40,keep)
    putw(w,32,generation+1)
    return 1

def frame_valid(s:readbuf) -> u64:
    if length(s) < 32 or length(s) > 1048576:
        return 100
    if getr(s,0) != 3616469954759447624 or getr(s,16) != length(s) or getr(s,24) != 0:
        return 100
    if getr(s,8) > (length(s)-32)//64:
        return 100
    return 0

def reply_error(o:mutbuf, status:u64) -> u64:
    putw(o,0,3616478819571946568)
    putw(o,8,2)
    putw(o,16,0)
    putw(o,24,status)
    return 32

def growth_capacity(current:u64, needed:u64) -> u64:
    if needed <= current or needed > 1073741824:
        return 0
    result = current
    while result < needed:
        result = result * 2
    return result

def full_tensor(w:mutbuf, rank:u64, requires:u64, shape:u64, bits:u64) -> u64:
    if bits > 4294967295:
        return terror(w,7)
    t = tnew(w,rank,shape,requires)
    if t == 0:
        return 0
    p = dptr(w,t)
    n = getw(w,p+8)
    data = getw(w,p+16)
    i = 0
    while i < n:
        put4(w,data+i*4,bits)
        i = i + 1
    return t

def entry(s:readbuf, o:mutbuf, w:mutbuf) -> u32:
    if length(o) < 32:
        return 0
    reply_error(o,0)
    error = frame_valid(s)
    if error != 0:
        reply_error(o,error)
        return 32
    putw(w,0,0)
    putw(w,48,0)
    putw(w,56,0)
    count = getr(s,8)
    end = 32 + count*64
    i = 0
    at = 32
    while i < count:
        if at > length(o)-32:
            putw(o,24,3)
            return at
        at = command(s,o,w,32+i*64,at,end)
        i = i + 1
        putw(o,16,i)
        error = getw(w,0)
        if error != 0:
            putw(o,24,error)
            return at
    return at
