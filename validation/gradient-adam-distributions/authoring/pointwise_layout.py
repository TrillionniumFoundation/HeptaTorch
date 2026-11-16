def pointwise_compare_axes(w:mutbuf, out:u64, a:u64, b:u64, left:u64, right:u64) -> u64:
    operand = 0
    while operand < 2:
        source = a
        if operand == 1:
            source = b
        ls = dist_probability_stride(w,out,source,left) << 2
        rs = dist_probability_stride(w,out,source,right) << 2
        if ls != 0 and rs != 0:
            ordered_left = ls ^ 9223372036854775808
            ordered_right = rs ^ 9223372036854775808
            if ordered_left < ordered_right:
                return 1
            if ordered_left > ordered_right:
                return 2
            if getw(w,out+96+left*8) > getw(w,out+96+right*8):
                return 2
        operand = operand+1
    return 0

def pointwise_axis_order(w:mutbuf, out:u64, a:u64, b:u64) -> u64:
    rank = getw(w,out)
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
            comparison = pointwise_compare_axes(w,out,a,b,left,right)
            if comparison == 2:
                difference = left ^ right
                perm = perm ^ (difference << (j*8)) ^ (difference << (anchor*8))
                anchor = j
            if comparison == 1:
                done = 1
        i = i+1
    return perm

def pointwise_channels_last(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p) != 4:
        return 0
    expected = 1
    i = 0
    while i < 4:
        axis = (131841 >> (i*8)) & 255
        size = getw(w,p+96+axis*8)
        if size > 1 and getw(w,p+128+axis*8) != expected:
            return 0
        expected = expected*size
        i = i+1
    return 1

def pointwise_write_order(w:mutbuf, p:u64, order:u64, raw_zero:u64) -> u64:
    stride = 1
    i = 0
    while i < getw(w,p):
        axis = (order >> (i*8)) & 255
        putw(w,p+128+axis*8,stride)
        size = getw(w,p+96+axis*8)
        if size != 0 or raw_zero != 0:
            stride = stride*size
        i = i+1
    return 1

def pointwise_layout(w:mutbuf, t:u64, a:u64, b:u64) -> u64:
    p = dptr(w,t)
    pa = dptr(w,a)
    pb = dptr(w,b)
    rank = getw(w,p)
    if incoming_same_shape(w,a,b) != 0:
        if is_contiguous(w,a) != 0 and is_contiguous(w,b) != 0:
            return 1
        if pointwise_channels_last(w,a) != 0 and pointwise_channels_last(w,b) != 0:
            return pointwise_write_order(w,p,131841,0)
        if is_dense(w,a) != 0 and is_dense(w,b) != 0:
            same = 1
            i = 0
            while i < rank:
                if getw(w,pa+128+i*8) != getw(w,pb+128+i*8):
                    same = 0
                i = i+1
            if same != 0:
                i = 0
                while i < rank:
                    putw(w,p+128+i*8,getw(w,pb+128+i*8))
                    i = i+1
                return 1
    order = pointwise_axis_order(w,p,pa,pb)
    inverted = 1
    i = 0
    while i < rank:
        if ((order >> (i*8)) & 255) != rank-i-1:
            inverted = 0
        i = i+1
    if inverted != 0:
        return 1
    return pointwise_write_order(w,p,order,1)
