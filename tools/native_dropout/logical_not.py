def bool_logical_not_byte(value:u64) -> u64:
    return choose((value & 255) == 0,1,0)

def bool_logical_not(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+576) != 1:
        return terror(w,7)
    if dist_no_internal_overlap(w,t) == 0:
        return 0
    if version_can_bump(w,t,1) == 0:
        return 0
    i = 0
    while i < getw(w,p+8):
        address = taddr(w,t,i)
        value = typed_load(w,address,1)
        typed_store(w,address,1,bool_logical_not_byte(value))
        i = i+1
    return bump_version(w,t)
