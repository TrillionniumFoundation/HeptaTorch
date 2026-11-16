def backward_seed(w:mutbuf, t:u64, seed:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+32) == 0:
        return terror(w,5)
    if seed == 0:
        if getw(w,p+8) != 1:
            return terror(w,5)
    else:
        if valid(w,seed) == 0:
            return 0
        q = dptr(w,seed)
        rank = getw(w,p)
        if getw(w,q) != rank:
            return terror(w,2)
        axis = 0
        while axis < rank:
            if getw(w,p+96+axis*8) != getw(w,q+96+axis*8):
                return terror(w,2)
            axis = axis + 1
    epoch = getw(w,24) + 1
    putw(w,24,epoch)
    if mark_graph(w,t,epoch) == 0:
        return 0
    limit = t & 255
    i = 1
    while i <= limit:
        current = live_id(w,i)
        pi = dptr(w,current)
        if getw(w,pi+168) == epoch:
            fill_range(w,getw(w,pi+160),getw(w,pi+8)*4,0)
        i = i + 1
    if seed == 0:
        put4(w,getw(w,p+160),1065353216)
    else:
        j = 0
        n = getw(w,p+8)
        dst = getw(w,p+160)
        while j < n:
            put4(w,dst+j*4,tget(w,seed,j))
            j = j + 1
    i = limit
    while i > 0:
        current = live_id(w,i)
        pi = dptr(w,current)
        if getw(w,pi+168) == epoch:
            op = getw(w,pi+40)
            if op >= 1 and op <= 4:
                back_binary(w,current)
            if op == 5 or op == 6 or op == 10 or (op >= 11 and op <= 17):
                back_unary(w,current)
            if op == 7 or op == 8:
                back_reduce(w,current)
            if op == 18 or op == 19:
                back_axis(w,current)
            if op == 20 or op == 21:
                back_softmax(w,current)
            if op >= 22 and op <= 24:
                back_spatial(w,current)
            if op == 25:
                back_matmul(w,current)
            if op == 26:
                back_index_select(w,current)
            if op == 27:
                back_gather(w,current)
            if op == 37 or op == 38:
                back_concat(w,current)
            if op == 39 or op == 40:
                back_indexed_add(w,current)
        i = i - 1
    i = 1
    while i <= limit:
        current = live_id(w,i)
        pi = dptr(w,current)
        if getw(w,pi+168) == epoch and getw(w,pi+40) == 0:
            j = 0
            n = getw(w,pi+8)
            dst = getw(w,pi+24)
            src = getw(w,pi+160)
            while j < n:
                put4(w,dst+j*4,addf(get4(w,dst+j*4),get4(w,src+j*4)))
                j = j + 1
        i = i + 1
    return 1
