def integer_i64_to_f32(bits:u64) -> u64:
    sign = (bits >> 63) << 31
    magnitude = bits
    if sign != 0:
        magnitude = 0-bits
    if magnitude == 0:
        return 0
    exponent = 0
    scan = magnitude
    while scan > 1:
        scan = scan >> 1
        exponent = exponent+1
    if exponent <= 23:
        mantissa = magnitude << (23-exponent)
    else:
        shift = exponent-23
        mantissa = magnitude >> shift
        remainder = magnitude & ((1 << shift)-1)
        half = 1 << (shift-1)
        if remainder > half or (remainder == half and (mantissa & 1) != 0):
            mantissa = mantissa+1
        if mantissa == 16777216:
            mantissa = mantissa >> 1
            exponent = exponent+1
    return sign | ((exponent+127) << 23) | (mantissa & 8388607)

def integer_as_f32(bits:u64, dtype:u64) -> u64:
    if dtype == 0:
        return bits
    if dtype == 1:
        return choose(bits != 0,1065353216,0)
    return integer_i64_to_f32(bits)

def integer_result_dtype(w:mutbuf, op:u64, a:u64, b:u64) -> u64:
    ca = tensor_dtype(w,a)
    cb = tensor_dtype(w,b)
    if ca > 2 or cb > 2 or (ca != 2 and cb != 2):
        return terror(w,7)
    if op < 1 or op > 7:
        return terror(w,7)
    if op == 2 and (ca == 1 or cb == 1):
        return terror(w,7)
    if getw(w,dptr(w,a)+32) != 0 or getw(w,dptr(w,b)+32) != 0:
        return terror(w,5)
    if ca == 0 or cb == 0 or op == 7:
        return 0
    return 2

def integer_scalar_binary(op:u64, a:u64, b:u64, dtype:u64) -> u64:
    if dtype == 0:
        if op == 1:
            return addf(a,b)
        if op == 2:
            return subf(a,b)
        if op == 3:
            return mulf(a,b)
        return divf(a,b)
    if op == 1:
        return a+b
    if op == 2:
        return a-b
    if op == 3:
        return a*b
    if op == 4:
        return a & b
    if op == 5:
        return a | b
    return a ^ b

def integer_pointwise_compare_axes(w:mutbuf, p:u64, pa:u64, pb:u64, left:u64, right:u64) -> u64:
    which = 0
    while which < 2:
        q = choose(which == 1,pb,pa)
        width = dtype_width(getw(w,q+576))
        sa = dist_probability_stride(w,p,q,left)*width
        sb = dist_probability_stride(w,p,q,right)*width
        if sa != 0 and sb != 0:
            aa = sa ^ 9223372036854775808
            bb = sb ^ 9223372036854775808
            if aa < bb:
                return 1
            if aa > bb:
                return 2
            if getw(w,p+96+left*8) > getw(w,p+96+right*8):
                return 2
        which = which+1
    return 0

def integer_target_shape(w:mutbuf, target:u64, rank:u64) -> u64:
    p = dptr(w,target)
    if rank != getw(w,p):
        return terror(w,2)
    axis = 0
    while axis < rank:
        if getw(w,128+axis*8) != getw(w,p+96+axis*8):
            return terror(w,2)
        axis = axis+1
    return 1

def integer_view_end(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    extent = 0
    axis = 0
    while axis < getw(w,p):
        extent = extent+(getw(w,p+96+axis*8)-1)*getw(w,p+128+axis*8)
        axis = axis+1
    return getw(w,p+232)+(extent+1)*dtype_width(getw(w,p+576))

def integer_inplace_overlap(w:mutbuf, a:u64, b:u64) -> u64:
    pa = dptr(w,a)
    pb = dptr(w,b)
    if getw(w,pa+8) == 0 or getw(w,pb+8) == 0:
        return 1
    if dist_no_internal_overlap(w,a) == 0:
        return 0
    if is_dense(w,a) == 0 or is_dense(w,b) == 0:
        return 1
    if getw(w,pa+216) != getw(w,pb+216):
        return 1
    end_a = getw(w,pa+232)+getw(w,pa+8)*dtype_width(getw(w,pa+576))
    end_b = getw(w,pb+232)+getw(w,pb+8)*dtype_width(getw(w,pb+576))
    if getw(w,pa+232) >= end_b or getw(w,pb+232) >= end_a:
        return 1
    if getw(w,pa+232) != getw(w,pb+232) or end_a != end_b:
        return terror(w,2)
    if getw(w,pa) != getw(w,pb):
        return terror(w,2)
    axis = 0
    while axis < getw(w,pa):
        if getw(w,pa+128+axis*8) != getw(w,pb+128+axis*8):
            return terror(w,2)
        axis = axis+1
    return 1

def integer_binary_element(w:mutbuf, op:u64, target:u64, a:u64, b:u64, index:u64) -> u64:
    ca = tensor_dtype(w,a)
    cb = tensor_dtype(w,b)
    dtype = tensor_dtype(w,target)
    av = tget(w,a,bindex(w,target,a,index))
    bv = tget(w,b,bindex(w,target,b,index))
    if dtype == 0:
        av = integer_as_f32(av,ca)
        bv = integer_as_f32(bv,cb)
    else:
        if ca == 1:
            av = choose(av != 0,1,0)
        if cb == 1:
            bv = choose(bv != 0,1,0)
    value = integer_scalar_binary(op,av,bv,dtype)
    return typed_store(w,taddr(w,target,index),dtype_width(dtype),value)

def integer_promote_operand(w:mutbuf, source:u64, dtype:u64) -> u64:
    original = tensor_dtype(w,source)
    if original == dtype:
        return source
    if dtype != 0 and (dtype != 2 or original != 1):
        return terror(w,7)
    p = dptr(w,source)
    target = typed_descriptor(w,getw(w,p),p+96,0,1,dtype)
    if target == 0:
        return 0
    if dist_dense_layout(w,target,source,1) == 0:
        return 0
    n = getw(w,p+8)
    index = 0
    while index < n:
        value = tget(w,source,index)
        if dtype == 0:
            value = integer_as_f32(value,original)
        else:
            value = choose(value != 0,1,0)
        typed_store(w,taddr(w,target,index),dtype_width(dtype),value)
        index = index+1
    return target

def integer_binary(w:mutbuf, op:u64, a:u64, b:u64, inplace:u64) -> u64:
    if valid(w,a) == 0 or valid(w,b) == 0:
        return 0
    dtype = integer_result_dtype(w,op,a,b)
    if getw(w,0) != 0:
        return 0
    if inplace != 0:
        if integer_inplace_overlap(w,a,b) == 0:
            return 0
    rank = bshape(w,a,b)
    if getw(w,0) != 0:
        return 0
    target = a
    if inplace != 0:
        if integer_target_shape(w,a,rank) == 0:
            return 0
        if tensor_dtype(w,a) != 2 or dtype != 2:
            return terror(w,7)
        if version_can_bump(w,a,1) == 0:
            return 0
    if op >= 4 and op <= 6 and dtype == 0:
        return terror(w,7)
    left = integer_promote_operand(w,a,dtype)
    if left == 0:
        return 0
    right = integer_promote_operand(w,b,dtype)
    if right == 0:
        return 0
    if inplace == 0:
        rank = bshape(w,left,right)
        if getw(w,0) != 0:
            return 0
        target = typed_descriptor(w,rank,128,0,1,dtype)
        if target == 0:
            return 0
        if integer_pointwise_layout(w,target,left,right) == 0:
            return 0
    n = getw(w,dptr(w,target)+8)
    i = 0
    while i < n:
        integer_binary_element(w,op,target,left,right,i)
        i = i+1
    if inplace != 0:
        if bump_version(w,target) == 0:
            return 0
    return target

def integer_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64) -> u64:
    code = getr(s,cp)
    result = 0
    first = getw(w,8)
    if (code != 103 and code != 104) or getr(s,cp+32) != 0 or getr(s,cp+40) != 0 or getr(s,cp+48) != 0 or getr(s,cp+56) != 0:
        terror(w,7)
    if getw(w,0) == 0:
        a = public_node(w,resolve_handle(w,getr(s,cp+16)))
        b = public_node(w,resolve_handle(w,getr(s,cp+24)))
        if a != 0 and b != 0:
            result = integer_binary(w,getr(s,cp+8),a,b,choose(code == 104,1,0))
            if getw(w,0) == 0:
                result = publish_tensor(w,result)
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
