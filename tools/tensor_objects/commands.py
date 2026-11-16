def command_argument(w:mutbuf, code:u64, field:u64, value:u64) -> u64:
    required = 0
    optional = 0
    if field == 1:
        required = 1654574268384
    if field == 2:
        required = 4713447484
        optional = 603979776
    if field == 3:
        required = 268435460
        optional = 67141632
    if code < 64:
        flag = 1 << code
        if (required & flag) != 0:
            return public_node(w,value)
        if (optional & flag) != 0:
            return public_optional(w,value)
    return value

def optimizer_command(s:readbuf, w:mutbuf, cp:u64, end:u64) -> u64:
    code = getr(s,cp)
    a = public_node(w,getr(s,cp+8))
    b = public_node(w,getr(s,cp+16))
    c = getr(s,cp+24)
    if code == 24:
        c = public_node(w,c)
    step = getr(s,cp+32)
    source = getr(s,cp+40)
    amount = 40
    if code == 24:
        amount = 48
    if source < end or source > length(s):
        return terror(w,100)
    if amount > length(s)-source or source % 8 != 0:
        return terror(w,100)
    i = 0
    while i < amount:
        putw(w,2080+i,getr(s,source+i))
        i = i+8
    if code == 24:
        return adam(w,a,b,c,step,2080)
    return sgd_momentum(w,a,b,step,2080)

def inspect_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64) -> u64:
    code = getr(s,cp)
    id = resolve_handle(w,getr(s,cp+8))
    a = public_node(w,id)
    result = 0
    payload = 0
    if a != 0:
        p = dptr(w,a)
        if code == 13:
            if length(o)-at-32 < 128:
                terror(w,3)
            else:
                putw(o,at+32,getw(w,p))
                putw(o,at+40,getw(w,p+8))
                putw(o,at+48,getw(w,p+32))
                j = 0
                while j < 4:
                    putw(o,at+56+j*8,getw(w,p+96+j*8))
                    putw(o,at+88+j*8,getw(w,p+128+j*8))
                    j = j+1
                putw(o,at+120,getw(w,p+216))
                putw(o,at+128,getw(w,p+224))
                putw(o,at+136,tversion(w,a))
                putw(o,at+144,getw(w,p+232))
                sp = so_storage_live(w,getw(w,p+216))
                putw(o,at+152,so_uget(w,sp+24))
                result = id
                payload = 128
        else:
            n = getw(w,p+8)
            first = 0
            count = n
            if code == 39 or code == 40:
                first = getr(s,cp+16)
                count = getr(s,cp+24)
            if code == 11 or code == 40:
                if getw(w,p+40) != 0 or getw(w,p+32) == 0:
                    terror(w,5)
            if first > n or count > n-first:
                terror(w,2)
            if count > (length(o)-at-32)//4:
                terror(w,3)
            if getw(w,0) == 0:
                j = 0
                while j < count:
                    value = 0
                    if code == 10 or code == 39:
                        value = tget(w,a,first+j)
                    else:
                        value = tgrad(w,a,first+j)
                    put4(o,at+32+j*4,value)
                    j = j+1
                result = count
                payload = count*4
    status = getw(w,0)
    if status != 0:
        result = 0
        payload = 0
    putw(o,at,code)
    putw(o,at+8,status)
    putw(o,at+16,result)
    putw(o,at+24,payload)
    return at+32+payload

def object_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64) -> u64:
    code = getr(s,cp)
    a = resolve_handle(w,getr(s,cp+8))
    b = getr(s,cp+16)
    c = getr(s,cp+24)
    result = 0
    payload = 0
    first = getw(w,8)
    if code < 41 or code > 56:
        terror(w,4)
    if code == 41:
        result = drop_tensor(w,a)
    if code == 42:
        t = public_node(w,a)
        if t != 0:
            result = weak_create(w,t)
    if code == 43:
        result = weak_lock(w,a)
    if code == 44:
        result = weak_release(w,a)
    if code == 45:
        result = public_hold_release(w,a)
    if code == 46:
        t = public_node(w,a)
        if t != 0:
            result = save_public(w,t)
    if code == 47:
        result = saved_public_check(w,a)
    if code == 48:
        result = saved_public_release(w,a)
    if code == 49 or code == 50:
        n = 0
        if code == 49:
            p = public_saved(w,a)
            if p != 0:
                if graph_saved_check(w,a) != 0:
                    layout = private_hold(w,so_uget(w,p+56),128)
                    n = getw(w,layout+8)
        else:
            p = public_hold(w,a)
            if p != 0:
                sp = so_storage_live(w,so_uget(w,p+16))
                n = so_uget(w,sp+24)//4
        if b > n or c > n-b:
            terror(w,2)
        if c > (length(o)-at-32)//4:
            terror(w,3)
        if getw(w,0) == 0:
            j = 0
            while j < c:
                value = 0
                if code == 49:
                    value = saved_logical(w,a,b+j)
                else:
                    value = hold_logical(w,a,b+j)
                put4(o,at+32+j*4,value)
                j = j+1
            result = c
            payload = c*4
    if code == 51 or code == 52:
        kind = code-49
        p = so_slot_lookup(w,kind,a)
        if p == 0:
            terror(w,1)
        else:
            if length(o)-at-32 < 64:
                terror(w,3)
            else:
                j = 0
                while j < 8:
                    putw(o,at+32+j*8,so_uget(w,p+j*8))
                    j = j+1
                result = a
                payload = 64
    if code == 53:
        status = so_audit(w)
        if status != 0:
            terror(w,status)
        else:
            result = 1
    if code == 54:
        if length(o)-at-32 < 96:
            terror(w,3)
        else:
            payload = object_stats(w,o,at+32)
            result = 1
    if code == 55:
        t = public_node(w,a)
        if t != 0:
            result = data_alias(w,t)
    if code == 56:
        result = unpack_saved(w,a)
    if getw(w,0) == 0:
        if code == 55 or code == 56:
            result = publish_tensor(w,result)
            finish_command(w,first)
    status = getw(w,0)
    if status != 0:
        result = 0
        payload = 0
    putw(o,at,code)
    putw(o,at+8,status)
    putw(o,at+16,result)
    putw(o,at+24,payload)
    return at+32+payload

def entry(s:readbuf, o:mutbuf, w:mutbuf) -> u32:
    if length(o) < 32:
        return 0
    reply_error(o,0)
    error = frame_valid(s)
    if error != 0:
        reply_error(o,error)
        return 32
    putw(w,0,0)
    putw(w,48,0)
    putw(w,56,0)
    count = getr(s,8)
    if count != 0:
        putw(w,96,8192)
    end = 32+count*64
    i = 0
    at = 32
    while i < count:
        if at > length(o)-32:
            putw(o,24,3)
            return at
        putw(w,104,i)
        next = command(s,o,w,32+i*64,at,end)
        error = getw(w,0)
        if error != 0:
            putw(o,24,error)
            return next
        putw(w,getw(w,96)+i*8,getw(o,at+16))
        at = next
        i = i+1
        putw(o,16,i)
    if count != 0:
        fill_range(w,8192,count*8,0)
        putw(w,96,0)
        putw(w,104,0)
    putw(w,80,0)
    if getw(w,0) != 0:
        reply_error(o,getw(w,0))
        return 32
    return at
