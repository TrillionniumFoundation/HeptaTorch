def binary_saved_base(w:mutbuf, p:u64) -> u64:
    id = getw(w,p+584)
    if id <= 65535:
        return 0
    hp = so_slot_lookup(w,6,id)
    if hp == 0 or so_uget(w,hp+24) != 1:
        return terror(w,9)
    sp = so_storage_live(w,so_uget(w,hp+16))
    if sp == 0 or so_uget(w,sp+24) != 32:
        return terror(w,9)
    address = 9223372036854775808+so_uget(w,sp+16)
    if address != getw(w,p+592):
        return terror(w,9)
    return address

def binary_saved_retain_one(w:mutbuf, address:u64, slot:u64, t:u64, saved:u64) -> u64:
    if saved == 0:
        return 1
    p = dptr(w,t)
    s = so_saved_live(w,saved)
    if s == 0 or getw(w,p+240) != 1 or getw(w,p+320) == 0:
        return terror(w,9)
    if so_uget(w,s+16) != getw(w,p+216) or so_uget(w,s+24) != getw(w,p+224):
        return terror(w,9)
    if getw(w,p+600) >= 4294967295:
        return terror(w,8)
    if node_retain(w,t) == 0:
        return 0
    putw(w,p+600,getw(w,p+600)+1)
    putw(w,address+slot*16,t)
    putw(w,address+slot*16+8,getw(w,p+320))
    return 1

def binary_saved_bind(w:mutbuf, t:u64, a:u64, b:u64) -> u64:
    p = dptr(w,t)
    sa = getw(w,p+280)
    sb = getw(w,p+288)
    if sa == 0 and sb == 0:
        return 1
    if getw(w,p+584) != 0 or getw(w,p+592) != 0:
        return terror(w,9)
    address = alloc(w,32)
    if getw(w,0) != 0:
        return 0
    putw(w,p+584,getw(w,80))
    putw(w,p+592,address)
    if binary_saved_retain_one(w,address,0,a,sa) == 0:
        return 0
    return binary_saved_retain_one(w,address,1,b,sb)

def record(w:mutbuf, t:u64, op:u64, a:u64, b:u64) -> u64:
    result = mixed_original_record(w,t,op,a,b)
    if result == 0:
        return 0
    if op == 3 or op == 4:
        if binary_saved_bind(w,t,a,b) == 0:
            return 0
    return result

def binary_saved_release(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    address = binary_saved_base(w,p)
    if address == 0:
        return terror(w,9)
    hold = getw(w,p+584)
    putw(w,p+584,0)
    putw(w,p+592,0)
    slot = 0
    while slot < 2:
        target = getw(w,address+slot*16)
        inc = getw(w,address+slot*16+8)
        if target != 0:
            q = private_tensor(w,target,inc)
            if q == 0 or getw(w,q+600) == 0:
                return terror(w,9)
            putw(w,address+slot*16,0)
            putw(w,address+slot*16+8,0)
            putw(w,q+600,getw(w,q+600)-1)
            if node_release(w,target) == 0:
                return 0
        elif inc != 0:
            return terror(w,9)
        slot = slot+1
    return hold_release(w,hold)

def dtype_mask_release(w:mutbuf, t:u64) -> u64:
    if getw(w,dptr(w,t)+584) > 65535:
        return binary_saved_release(w,t)
    return mixed_original_mask_release(w,t)

def binary_saved_audit(w:mutbuf, p:u64) -> u64:
    address = binary_saved_base(w,p)
    if address == 0 or getw(w,p+576) != 0 or getw(w,p+32) == 0:
        return 9
    op = getw(w,p+40)
    if (op != 3 and op != 4) or getw(w,p+328) != 0:
        return 9
    count = 0
    slot = 0
    while slot < 2:
        target = getw(w,address+slot*16)
        inc = getw(w,address+slot*16+8)
        saved = getw(w,p+280+slot*8)
        if target == 0:
            if saved != 0 or inc != 0:
                return 9
        else:
            q = private_tensor(w,target,inc)
            s = so_saved_live(w,saved)
            if q == 0 or s == 0 or inc >= getw(w,p+320):
                return 9
            if so_uget(w,s+16) != getw(w,q+216) or so_uget(w,s+24) != getw(w,q+224):
                return 9
            if so_uget(w,s+40) != getw(w,q+232) or so_uget(w,s+56) != 0:
                return 9
            putw(w,q+616,getw(w,q+616)+1)
            count = count+1
        slot = slot+1
    if count == 0:
        return 9
    return 0

def binary_saved_needed(w:mutbuf, p:u64, slot:u64) -> u64:
    op = getw(w,p+40)
    if op == 1 or op == 2:
        return 0
    if op != 3 and op != 4:
        return terror(w,7)
    if slot == 0:
        return getw(w,p+80)
    if op == 4:
        return getw(w,p+72) | getw(w,p+80)
    return getw(w,p+72)

def binary_saved_get(w:mutbuf, t:u64, slot:u64) -> u64:
    p = dptr(w,t)
    if slot > 1:
        return terror(w,7)
    needed = binary_saved_needed(w,p,slot)
    if getw(w,0) != 0 or needed == 0:
        return 0
    if getw(w,p+328) != 0:
        return terror(w,10)
    saved = getw(w,p+280+slot*8)
    if saved == 0:
        return terror(w,9)
    if graph_saved_check(w,saved) == 0:
        return 0
    address = binary_saved_base(w,p)
    if address == 0:
        return terror(w,9)
    target = getw(w,address+slot*16)
    if private_tensor(w,target,getw(w,address+slot*16+8)) == 0:
        return terror(w,9)
    return publish_tensor(w,target)

def binary_saved_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64) -> u64:
    result = 0
    if getr(s,cp+24) != 0 or getr(s,cp+32) != 0 or getr(s,cp+40) != 0 or getr(s,cp+48) != 0 or getr(s,cp+56) != 0:
        terror(w,7)
    if getw(w,0) == 0:
        t = public_node(w,resolve_handle(w,getr(s,cp+8)))
        if t != 0:
            result = binary_saved_get(w,t,getr(s,cp+16))
    status = getw(w,0)
    if status != 0:
        result = 0
    putw(o,at,107)
    putw(o,at+8,status)
    putw(o,at+16,result)
    putw(o,at+24,0)
    return at+32

def binary_backward_read(w:mutbuf, t:u64, index:u64) -> u64:
    value = tget(w,t,index)
    return integer_as_f32(value,tensor_dtype(w,t))
