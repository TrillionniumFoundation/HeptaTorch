def integer_scalar_add(w:mutbuf, target:u64, scalar:u64) -> u64:
    if valid(w,target) == 0:
        return 0
    if tensor_dtype(w,target) != 2:
        return terror(w,7)
    if dist_no_internal_overlap(w,target) == 0:
        return 0
    if version_can_bump(w,target,1) == 0:
        return 0
    n = getw(w,dptr(w,target)+8)
    index = 0
    while index < n:
        value = tget(w,target,index)+scalar
        typed_store(w,taddr(w,target,index),8,value)
        index = index+1
    if bump_version(w,target) == 0:
        return 0
    return target

def integer_scalar_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64) -> u64:
    first = getw(w,8)
    result = 0
    if getr(s,cp) != 106 or getr(s,cp+24) != 0 or getr(s,cp+32) != 0 or getr(s,cp+40) != 0 or getr(s,cp+48) != 0 or getr(s,cp+56) != 0:
        terror(w,7)
    if getw(w,0) == 0:
        target = public_node(w,resolve_handle(w,getr(s,cp+8)))
        if target != 0:
            result = integer_scalar_add(w,target,getr(s,cp+16))
            if getw(w,0) == 0:
                result = publish_tensor(w,result)
    if getw(w,0) == 0:
        finish_command(w,first)
    status = getw(w,0)
    if status != 0:
        result = 0
    putw(o,at,106)
    putw(o,at+8,status)
    putw(o,at+16,result)
    putw(o,at+24,0)
    return at+32
