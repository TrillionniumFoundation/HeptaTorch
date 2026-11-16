def scalar_add_nograd(w:mutbuf, id:u64, value:u64) -> u64:
    if value > 4294967295:
        return terror(w,7)
    node = public_node(w,id)
    if node == 0:
        return 0
    p = dptr(w,node)
    if getw(w,p+8) != 0 and is_dense(w,node) == 0:
        axis = 0
        while axis < getw(w,p):
            if getw(w,p+96+axis*8) > 1 and getw(w,p+128+axis*8) == 0:
                return terror(w,7)
            axis = axis+1
    if version_can_bump(w,node,1) == 0:
        return 0
    i = 0
    while i < getw(w,p+8):
        at = taddr(w,node,i)
        put4(w,at,addf(get4(w,at),value))
        i = i+1
    if bump_version(w,node) == 0:
        return 0
    return getw(w,p+192)

def scalar_add_nograd_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64) -> u64:
    result = 0
    j = 3
    while j < 8:
        if getr(s,cp+j*8) != 0:
            terror(w,7)
        j = j+1
    if getw(w,0) == 0:
        result = scalar_add_nograd(w,getr(s,cp+8),getr(s,cp+16))
    status = getw(w,0)
    if status != 0:
        result = 0
    putw(o,at,95)
    putw(o,at+8,status)
    putw(o,at+16,result)
    putw(o,at+24,0)
    return at+32
