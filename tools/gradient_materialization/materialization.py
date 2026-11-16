def gradient_parent_count(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 0:
        return 0
    if op == 37 or op == 38:
        return getw(w,getw(w,p+184))
    return 2

def gradient_parent_at(w:mutbuf, t:u64, index:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op == 37 or op == 38:
        return getw(w,getw(w,p+184)+8+index*16)
    return getw(w,p+48+index*8)

def gradient_direct(w:mutbuf, t:u64, parent:u64) -> u64:
    p = dptr(w,t)
    op = getw(w,p+40)
    if op >= 1 and op <= 4:
        q = dptr(w,parent)
        rank = getw(w,p)
        if getw(w,q) != rank:
            return 0
        j = 0
        while j < rank:
            if getw(w,p+96+j*8) != getw(w,q+96+j*8):
                return 0
            j = j+1
        return 1
    if op >= 5 and op <= 21:
        return 1
    if op >= 37 and op <= 40:
        return 1
    return 0

def gradient_workspace_size(w:mutbuf, limit:u64, epoch:u64) -> u64:
    maximum = 0
    t = 1
    while t <= limit:
        p = dptr(w,t)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            amount = 0
            j = 0
            count = gradient_parent_count(w,t)
            while j < count:
                parent = gradient_parent_at(w,t,j)
                if parent != 0:
                    q = dptr(w,parent)
                    if getw(w,q+32) != 0:
                        part = getw(w,q+8)*5
                        if part > 1073741824-amount:
                            return terror(w,3)
                        amount = amount+part
                j = j+1
            if amount > maximum:
                maximum = amount
        t = t+1
    return maximum

def gradient_prepare(w:mutbuf, t:u64, scratch:u64) -> u64:
    cursor = scratch
    j = 0
    count = gradient_parent_count(w,t)
    while j < count:
        parent = gradient_parent_at(w,t,j)
        if parent != 0:
            p = dptr(w,parent)
            if getw(w,p+32) != 0 and getw(w,p+368) == 0:
                n = getw(w,p+8)
                memory_fill(w,cursor,n*5,0)
                putw(w,p+376,getw(w,p+160))
                putw(w,p+160,cursor)
                putw(w,p+368,1)
                if gradient_direct(w,t,parent) != 0:
                    putw(w,p+368,cursor+n*4)
                cursor = cursor+n*5
        j = j+1
    return 1

def gradient_finish(w:mutbuf, t:u64, commit:u64) -> u64:
    j = 0
    count = gradient_parent_count(w,t)
    while j < count:
        parent = gradient_parent_at(w,t,j)
        if parent != 0:
            p = dptr(w,parent)
            if getw(w,p+368) != 0:
                src = getw(w,p+160)
                dst = getw(w,p+376)
                putw(w,p+160,dst)
                putw(w,p+368,0)
                putw(w,p+376,0)
                if commit != 0:
                    present = getw(w,p+336)
                    i = 0
                    n = getw(w,p+8)
                    while i < n:
                        value = get4(w,src+i*4)
                        if present != 0:
                            value = addf(get4(w,dst+i*4),value)
                        put4(w,dst+i*4,value)
                        i = i+1
                    putw(w,p+336,1)
        j = j+1
    return 1

def gradient_apply(w:mutbuf, t:u64, scratch:u64) -> u64:
    if getw(w,dptr(w,t)+40) == 0:
        return partial_apply_node(w,t)
    gradient_prepare(w,t,scratch)
    partial_apply_node(w,t)
    gradient_finish(w,t,choose(getw(w,0) == 0,1,0))
    return choose(getw(w,0) == 0,1,0)

def gradient_clear_marks(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if getw(w,p+368) != 0 or getw(w,p+376) != 0:
                return terror(w,9)
            putw(w,p+336,0)
        i = i+1
    return 1

def gadd(w:mutbuf, t:u64, index:u64, value:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+32) == 0:
        return 0
    address = getw(w,p+160)+index*4
    flags = getw(w,p+368)
    if flags > 1:
        flag = object_address(w,flags+index)
        if byte(w,flag) == 0:
            store(w,flag,1)
            put4(w,address,value)
            return 1
    put4(w,address,addf(get4(w,address),value))
    return 1
