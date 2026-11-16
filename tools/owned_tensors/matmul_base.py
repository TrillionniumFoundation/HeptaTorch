def matmul(w:mutbuf, a:u64, b:u64) -> u64:
    if valid(w,a) == 0 or valid(w,b) == 0:
        return 0
    pa = dptr(w,a)
    pb = dptr(w,b)
    ra = getw(w,pa)
    rb = getw(w,pb)
    if ra == 0 or rb == 0:
        return terror(w,2)
    k = getw(w,pa+96+(ra-1)*8)
    kb = getw(w,pb+96+choose(rb==1,0,rb-2)*8)
    if k != kb:
        return terror(w,2)
    ba = choose(ra>2,ra-2,0)
    bb = choose(rb>2,rb-2,0)
    batches = choose(ba>bb,ba,bb)
    i = 0
    while i < batches:
        da = 1
        db = 1
        if i+ba >= batches:
            da = getw(w,pa+96+(i+ba-batches)*8)
        if i+bb >= batches:
            db = getw(w,pb+96+(i+bb-batches)*8)
        if da != db and da != 1 and db != 1:
            return terror(w,2)
        putw(w,128+i*8,choose(da==1,db,da))
        i = i+1
    if ra != 1:
        putw(w,128+i*8,getw(w,pa+96+(ra-2)*8))
        i = i+1
    if rb != 1:
        putw(w,128+i*8,getw(w,pb+96+(rb-1)*8))
        i = i+1
    n = 1
    axis = 0
    while axis < i:
        n = n*getw(w,128+axis*8)
        if n > 262144:
            return terror(w,3)
        axis = axis+1
    if k != 0:
        if n > 16777216//k:
            return terror(w,3)
    t = tnew(w,i,128,getw(w,pa+32)|getw(w,pb+32))
    if t == 0:
        return 0
    record(w,t,25,a,b)
    p = dptr(w,t)
    putw(w,p+176,batches)
    putw(w,p+184,k)
    i = 0
    while i < n:
        put4(w,getw(w,p+16)+i*4,matmul_point(w,t,i))
        i = i+1
    return t

