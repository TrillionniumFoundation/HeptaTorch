def retained_compare_axes(w:mutbuf, t:u64, old:u64, left:u64, right:u64) -> u64:
    p = dptr(w,t)
    q = dptr(w,old)
    operand = 0
    while operand < 2:
        layout = q+128
        if operand == 1:
            layout = p+496
        ls = getw(w,layout+left*8) << 2
        rs = getw(w,layout+right*8) << 2
        if ls != 0 and rs != 0:
            ordered_left = ls ^ 9223372036854775808
            ordered_right = rs ^ 9223372036854775808
            if ordered_left < ordered_right:
                return 1
            if ordered_left > ordered_right:
                return 2
            if getw(w,p+96+left*8) > getw(w,p+96+right*8):
                return 2
        operand = operand+1
    return 0

def retained_axis_order(w:mutbuf, t:u64, old:u64) -> u64:
    rank = getw(w,dptr(w,t))
    perm = 0
    i = 0
    while i < rank:
        perm = perm | ((rank-i-1) << (i*8))
        i = i+1
    i = 1
    while i < rank:
        anchor = i
        j = i
        done = 0
        while j > 0 and done == 0:
            j = j-1
            left = (perm >> (j*8)) & 255
            right = (perm >> (anchor*8)) & 255
            comparison = retained_compare_axes(w,t,old,left,right)
            if comparison == 2:
                difference = left ^ right
                perm = perm ^ (difference << (j*8)) ^ (difference << (anchor*8))
                anchor = j
            if comparison == 1:
                done = 1
        i = i+1
    return perm

def retained_incoming_channels_last(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p) != 4:
        return 0
    expected = 1
    i = 0
    while i < 4:
        axis = (131841 >> (i*8)) & 255
        size = getw(w,p+96+axis*8)
        if size > 1 and getw(w,p+496+axis*8) != expected:
            return 0
        expected = expected*size
        i = i+1
    return 1

def retained_sum_layout(w:mutbuf, g:u64, old:u64, t:u64) -> u64:
    out = dptr(w,g)
    p = dptr(w,t)
    q = dptr(w,old)
    rank = getw(w,p)
    if is_contiguous(w,old) != 0 and incoming_contiguous(w,t,p+496) != 0:
        return 1
    if pointwise_channels_last(w,old) != 0 and retained_incoming_channels_last(w,t) != 0:
        return pointwise_write_order(w,out,131841,0)
    if is_dense(w,old) != 0 and incoming_dense(w,t,p+496) != 0:
        same = 1
        i = 0
        while i < rank:
            if getw(w,q+128+i*8) != getw(w,p+496+i*8):
                same = 0
            i = i+1
        if same != 0:
            i = 0
            while i < rank:
                putw(w,out+128+i*8,getw(w,p+496+i*8))
                i = i+1
            return 1
    order = retained_axis_order(w,t,old)
    inverted = 1
    i = 0
    while i < rank:
        if ((order >> (i*8)) & 255) != rank-i-1:
            inverted = 0
        i = i+1
    if inverted != 0:
        return 1
    return pointwise_write_order(w,out,order,1)
