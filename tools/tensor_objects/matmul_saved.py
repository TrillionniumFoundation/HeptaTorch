def fold_view(w:mutbuf, t:u64, squeeze:u64, transpose:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+8) == 0:
        return 1
    rank = getw(w,p)-squeeze
    axis = rank-1
    expected = 0
    while axis > 0:
        axis = axis-1
        source = axis+squeeze
        if transpose != 0 and axis == rank-2:
            source = source+1
        dim = getw(w,p+96+source*8)
        stride = getw(w,p+128+source*8)
        if dim > 1:
            if expected != 0 and stride != expected:
                return 0
            expected = stride*dim
    return 1

def matmul_copy_side(w:mutbuf, a:u64, b:u64, side:u64) -> u64:
    pa = dptr(w,a)
    pb = dptr(w,b)
    ra = getw(w,pa)
    rb = getw(w,pb)
    if getw(w,pa+8) == 0 or getw(w,pb+8) == 0:
        return 0
    sa = 0
    sb = 0
    if ra == 3 and rb == 3 and getw(w,pa+96) != getw(w,pb+96):
        if getw(w,pa+96) == 1 and getw(w,pa+32) != 0:
            sa = 1
            ra = 2
        else:
            if getw(w,pb+96) == 1 and getw(w,pb+32) != 0:
                sb = 1
                rb = 2
    if ra <= 2 and rb <= 2:
        return 0
    large = a
    small = b
    squeeze = sa
    transpose = 0
    large_rank = ra
    small_rank = rb
    large_side = 0
    if rb > ra:
        large = b
        small = a
        squeeze = sb
        transpose = 1
        large_rank = rb
        small_rank = ra
        large_side = 1
    if large_rank >= 3 and small_rank <= 2:
        view = fold_view(w,large,squeeze,transpose)
        if getw(w,dptr(w,small)+32) != 0:
            if side == large_side and view == 0:
                return 1
            return 0
        if ra != 2 and view != 0:
            return 0
    if ra < 4 and rb < 4:
        return 0
    target0 = 1
    target1 = 1
    if ra == 4:
        target0 = getw(w,pa+96)
        target1 = getw(w,pa+104)
    else:
        if ra == 3:
            target1 = getw(w,pa+96)
    if rb == 4:
        dim = getw(w,pb+96)
        if dim > target0:
            target0 = dim
        dim = getw(w,pb+104)
        if dim > target1:
            target1 = dim
    else:
        if rb == 3:
            dim = getw(w,pb+96)
            if dim > target1:
                target1 = dim
    if target0 <= 1 or target1 <= 1:
        return 0
    source = pa
    rank = ra
    if side != 0:
        source = pb
        rank = rb
    stride0 = 0
    stride1 = 0
    if rank == 4:
        if getw(w,source+96) == target0:
            stride0 = getw(w,source+128)
        if getw(w,source+104) == target1:
            stride1 = getw(w,source+136)
    if rank == 3:
        if getw(w,source+96) == target1:
            stride1 = getw(w,source+128)
    return choose(stride0==stride1*target1,0,1)

def saved_copy(w:mutbuf, t:u64) -> u64:
    if storage_capacity(w,5,0) == 0 or storage_capacity(w,3,0) == 0:
        return 0
    n = getw(w,dptr(w,t)+8)
    data = alloc(w,n*4)
    if data == 0:
        return 0
    hid = getw(w,80)
    hp = so_slot_lookup(w,6,hid)
    sid = so_uget(w,hp+16)
    i = 0
    while i < n:
        put4(w,data+i*4,tget(w,t,i))
        i = i+1
    vid = so_slot_new(w,3)
    id = so_slot_new(w,5)
    if vid == 0 or id == 0:
        return terror(w,9)
    so_uput(w,so_slot_lookup(w,3,vid)+24,1)
    sp = so_slot_lookup(w,5,id)
    so_uput(w,sp+16,sid)
    so_uput(w,sp+24,vid)
    so_uput(w,sp+48,n*4)
    so_slot_drop(w,6,hid)
    return id

def matmul_saved_get(w:mutbuf, t:u64, side:u64, index:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+344+side*8) != 0:
        saved = getw(w,p+280+side*8)
        sp = so_saved_live(w,saved)
        storage = so_storage_live(w,so_uget(w,sp+16))
        return get4(w,9223372036854775808+so_uget(w,storage+16)+index*4)
    return tget(w,getw(w,p+48+side*8),index)
