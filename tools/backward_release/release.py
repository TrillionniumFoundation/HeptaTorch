def backward_command(w:mutbuf, output:u64, seed:u64, flags:u64) -> u64:
    if flags > 1:
        return terror(w,7)
    t = public_node(w,output)
    v = public_optional(w,seed)
    if getw(w,0) != 0:
        return 0
    return backward_execute(w,t,v,flags)

def ensure_backward_work(w:mutbuf, limit:u64, epoch:u64) -> u64:
    i = 1
    while i <= limit:
        p = dptr(w,i)
        if getw(w,p+304) != 0 and getw(w,p+168) == epoch:
            if getw(w,p+256) == 0:
                address = alloc(w,getw(w,p+8)*4)
                if address == 0:
                    return 0
                putw(w,p+160,address)
                putw(w,p+256,getw(w,80))
        i = i+1
    return 1

def release_backward_saved(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    a = getw(w,p+280)
    b = getw(w,p+288)
    output = getw(w,p+296)
    released = choose(a != 0 or b != 0 or output != 0,1,0)
    if saved_release(w,a) == 0:
        return 0
    putw(w,p+280,0)
    if saved_release(w,b) == 0:
        return 0
    putw(w,p+288,0)
    if saved_release(w,output) == 0:
        return 0
    putw(w,p+296,0)
    op = getw(w,p+40)
    if op == 26 or op == 27 or op == 39 or op == 40:
        if op == 26 or op == 27:
            released = 1
        else:
            source = getw(w,p+56)
            if getw(w,dptr(w,source)+32) != 0:
                released = 1
        if hold_release(w,getw(w,p+264)) == 0:
            return 0
        putw(w,p+264,0)
        putw(w,p+184,0)
    if op == 24:
        if hold_release(w,getw(w,p+272)) == 0:
            return 0
        putw(w,p+272,0)
        putw(w,p+184,0)
    if released != 0:
        putw(w,p+328,1)
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
            if getw(w,p+40) != 0:
                if hold_release(w,getw(w,p+248)) == 0:
                    return 0
                putw(w,p+248,0)
                putw(w,p+24,0)
        i = i+1
    return 1

def backward_seed(w:mutbuf, t:u64, seed:u64) -> u64:
    return backward_execute(w,t,seed,1)
