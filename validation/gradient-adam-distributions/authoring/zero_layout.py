def incoming_zero_axis_order(w:mutbuf, p:u64) -> u64:
    rank = getw(w,p)
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
            ls = getw(w,p+496+left*8)
            rs = getw(w,p+496+right*8)
            if ls != 0 and rs != 0:
                if ls > rs or (ls == rs and getw(w,p+96+left*8) > getw(w,p+96+right*8)):
                    difference = left ^ right
                    perm = perm ^ (difference << (j*8)) ^ (difference << (anchor*8))
                    anchor = j
                else:
                    if ls < rs:
                        done = 1
        i = i+1
    return perm

def incoming_zero_layout(w:mutbuf, t:u64, dest:u64) -> u64:
    p = dptr(w,t)
    if incoming_dense(w,t,p+496) != 0:
        j = 0
        while j < 4:
            putw(w,dest+j*8,getw(w,p+496+j*8))
            j = j+1
        return 1
    order = incoming_zero_axis_order(w,p)
    stride = 1
    j = 0
    while j < getw(w,p):
        axis = (order >> (j*8)) & 255
        putw(w,dest+axis*8,stride)
        size = getw(w,p+96+axis*8)
        if size > 1:
            stride = stride*size
        j = j+1
    return 1
