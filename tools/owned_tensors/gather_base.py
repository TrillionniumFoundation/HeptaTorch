def gather(s:readbuf, w:mutbuf, a:u64, axis:u64, config:u64, end:u64) -> u64:
    if valid(w,a) == 0:
        return 0
    if config < end or config > length(s) or config % 8 != 0:
        return terror(w,100)
    if length(s)-config < 8:
        return terror(w,100)
    rank = getr(s,config)
    ap = dptr(w,a)
    if rank > 4 or rank != getw(w,ap):
        return terror(w,2)
    arank = rank
    if arank == 0:
        arank = 1
    if axis >= 9223372036854775808:
        axis = axis + arank
    if axis >= arank:
        return terror(w,2)
    if length(s)-config < 8+rank*8:
        return terror(w,100)
    count = 1
    j = 0
    while j < rank:
        dim = getr(s,config+8+j*8)
        if dim > 262144:
            return terror(w,3)
        count = count * dim
        if count > 262144:
            return terror(w,3)
        putw(w,2048+j*8,dim)
        j = j + 1
    offset = config+8+rank*8
    if count > (length(s)-offset) // 8:
        return terror(w,100)
    if count != 0:
        j = 0
        while j < rank:
            if j != axis and getw(w,2048+j*8) > getw(w,ap+96+j*8):
                return terror(w,2)
            j = j + 1
    extent = 1
    if rank != 0:
        extent = getw(w,ap+96+axis*8)
    j = 0
    while j < count:
        if getr(s,offset+j*8) >= extent:
            return terror(w,7)
        j = j + 1
    indices = alloc(w,count*8)
    if indices == 0:
        return 0
    j = 0
    while j < count:
        putw(w,indices+j*8,getr(s,offset+j*8))
        j = j + 1
    t = tnew(w,rank,2048,getw(w,ap+32))
    if t == 0:
        return 0
    p = dptr(w,t)
    putw(w,p+176,axis)
    putw(w,p+184,indices)
    record(w,t,27,a,0)
    j = 0
    while j < count:
        source = gather_source(w,t,j)
        put4(w,getw(w,p+16)+j*4,tget(w,a,source))
        j = j + 1
    return t

