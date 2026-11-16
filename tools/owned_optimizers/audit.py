def optimizer_audit_entry(w:mutbuf, blob:u64, i:u64, first:u64) -> u64:
    ep = optimizer_entry_at(w,blob,i)
    node = getw(w,ep)
    incarnation = getw(w,ep+8)
    p = private_tensor(w,node,incarnation)
    if p == 0 or getw(w,p+40) != 0:
        return 9
    if optimizer_audit_reference(w,node,incarnation) != 0:
        return 9
    key_index = getw(w,ep+16)
    if key_index < first or key_index > i:
        return 9
    key = optimizer_entry_at(w,blob,key_index)
    if getw(w,key) != node or getw(w,key+8) != incarnation or getw(w,key+16) != key_index:
        return 9
    if key_index != i:
        if getw(w,ep+24) != 0 or getw(w,ep+32) != 0 or getw(w,ep+40) != 0:
            return 9
    else:
        j = 0
        while j < i:
            other = optimizer_entry_at(w,blob,j)
            if getw(w,other) == node and getw(w,other+8) == incarnation:
                return 9
            j = j+1
        present = getw(w,key+24)
        if present > 1:
            return 9
        if present == 0:
            if getw(w,key+32) != 0 or getw(w,key+40) != 0:
                return 9
        else:
            buffer = getw(w,key+32)
            bp = private_tensor(w,buffer,getw(w,key+40))
            if bp == 0 or getw(w,bp+40) != 0 or getw(w,bp) != getw(w,p):
                return 9
            j = 0
            while j < getw(w,p):
                if getw(w,bp+96+j*8) != getw(w,p+96+j*8):
                    return 9
                j = j+1
            if optimizer_audit_reference(w,node,incarnation) != 0 or optimizer_audit_reference(w,buffer,getw(w,key+40)) != 0:
                return 9
    return choose(key_index == i,1,2)

def optimizer_audit_reference(w:mutbuf, node:u64, incarnation:u64) -> u64:
    p = private_tensor(w,node,incarnation)
    if p == 0 or getw(w,p+336) >= 4294967295:
        return 9
    putw(w,p+336,getw(w,p+336)+1)
    return 0

def optimizer_audit_record(w:mutbuf, row:u64) -> u64:
    groups = so_uget(w,row+24)
    count = so_uget(w,row+32)
    if groups == 0 or groups > 1024 or count > 4096 or so_uget(w,row+56) != 0:
        return 9
    if so_uget(w,row+48) > count or so_uget(w,row+40) != count-so_uget(w,row+48):
        return 9
    blob = private_hold(w,so_uget(w,row+16),64+groups*64+count*48)
    if blob == 0 or getw(w,blob) != groups or getw(w,blob+8) != count:
        return 9
    j = 0
    while j < 6:
        value = getw(w,blob+16+j*8)
        if (j < 4 and value > 4294967295) or (j >= 4 and value > 1):
            return 9
        j = j+1
    first = 0
    duplicate_count = 0
    group = 0
    while group < groups:
        gp = blob+64+group*64
        amount = getw(w,gp+56)
        if getw(w,gp+48) != first or first > count or amount > count-first:
            return 9
        j = 0
        while j < 6:
            value = getw(w,gp+j*8)
            if (j < 4 and value > 4294967295) or (j >= 4 and value > 1):
                return 9
            j = j+1
        i = first
        while i < first+amount:
            entry_status = optimizer_audit_entry(w,blob,i,first)
            if entry_status != 1 and entry_status != 2:
                return 9
            duplicate_count = duplicate_count+entry_status-1
            i = i+1
        first = first+amount
        group = group+1
    if first != count or duplicate_count != so_uget(w,row+48):
        return 9
    return 0

def optimizer_audit(w:mutbuf) -> u64:
    slots = so_uget(w,16)
    budget = 0
    j = 1
    while j <= slots:
        row = so_slot_at(w,8,j)
        if so_uget(w,row+8) == 1:
            count = so_uget(w,row+32)
            if count > 4096:
                return 9
            if count*count > 16777216-budget:
                return 3
            budget = budget+count*count
        j = j+1
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if getw(w,p+336) != 0 or getw(w,p+368) != 0 or getw(w,p+376) != 0:
            return 9
        if getw(w,p+304) != 0:
            if getw(w,p+320) == 0 or getw(w,p+320) > getw(w,120):
                return 9
        i = i+1
    status = graph_audit(w)
    j = 1
    while j <= slots and status == 0:
        row = so_slot_at(w,8,j)
        if so_uget(w,row+8) == 1:
            status = optimizer_audit_record(w,row)
            other_index = 1
            while other_index < j and status == 0:
                other = so_slot_at(w,8,other_index)
                if so_uget(w,other+8) == 1 and so_uget(w,other+16) == so_uget(w,row+16):
                    status = 9
                other_index = other_index+1
            other_index = 1
            while other_index <= slots and status == 0:
                other = so_slot_at(w,7,other_index)
                if so_uget(w,other+8) == 1 and so_uget(w,other+16) == so_uget(w,row+16):
                    status = 9
                other_index = other_index+1
        j = j+1
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if getw(w,p+304) != 0:
            if getw(w,p+336) != getw(w,p+312) or getw(w,p+208) < getw(w,p+312):
                status = 9
        elif getw(w,p+312) != 0 or getw(w,p+336) != 0:
            status = 9
        putw(w,p+336,0)
        i = i+1
    return status
