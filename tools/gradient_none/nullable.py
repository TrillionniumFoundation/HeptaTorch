def zero_grad_none(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0 or getw(w,p+32) == 0:
        return terror(w,5)
    if hold_release(w,getw(w,p+248)) == 0:
        return 0
    putw(w,p+248,0)
    putw(w,p+24,0)
    putw(w,p+360,0)
    return 1

def zero_grad(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0 or getw(w,p+32) == 0:
        return terror(w,5)
    if getw(w,p+360) == 0:
        return 1
    if getw(w,p+248) == 0 or getw(w,p+24) == 0:
        return terror(w,9)
    memory_fill(w,getw(w,p+24),getw(w,p+8)*4,0)
    return 1

def tgrad(w:mutbuf, t:u64, index:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+40) != 0 or getw(w,p+32) == 0:
        return terror(w,5)
    if index >= getw(w,p+8):
        return terror(w,7)
    if getw(w,p+360) == 0:
        return 0
    if getw(w,p+248) == 0 or getw(w,p+24) == 0:
        return terror(w,9)
    return get4(w,getw(w,p+24)+index*4)

def ensure_backward_work(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if getw(w,p+40) == 0 and getw(w,p+248) == 0:
                if getw(w,p+360) != 0:
                    return terror(w,9)
                grad = alloc(w,getw(w,p+8)*4)
                if grad == 0:
                    return 0
                putw(w,p+24,grad)
                putw(w,p+248,getw(w,80))
            if getw(w,p+256) == 0:
                address = alloc(w,getw(w,p+8)*4)
                if address == 0:
                    return 0
                putw(w,p+160,address)
                putw(w,p+256,getw(w,80))
        i = i+1
    return 1

def release_backward_scratch(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if hold_release(w,getw(w,p+256)) == 0:
                return 0
            putw(w,p+256,0)
            putw(w,p+160,0)
            if getw(w,p+40) != 0 or getw(w,p+360) == 0:
                if hold_release(w,getw(w,p+248)) == 0:
                    return 0
                putw(w,p+248,0)
                putw(w,p+24,0)
        i = i+1
    return 1
