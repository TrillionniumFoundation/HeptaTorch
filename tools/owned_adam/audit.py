def owned_adam_audit_state(w:mutbuf, key:u64, parameter:u64) -> u64:
    present = getw(w,key+24)
    if present != 0 and present != 14:
        return 9
    j = 0
    while j < 3:
        node = getw(w,key+32+j*16)
        incarnation = getw(w,key+40+j*16)
        if present == 0:
            if node != 0 or incarnation != 0:
                return 9
        else:
            p = private_tensor(w,node,incarnation)
            if p == 0 or getw(w,p+40) != 0 or node == getw(w,key):
                return 9
            k = 0
            while k < j:
                if getw(w,key+32+k*16) == node:
                    return 9
                k = k+1
            if j == 0:
                if getw(w,p) != 0 or getw(w,p+8) != 1:
                    return 9
            else:
                if getw(w,p) != getw(w,parameter):
                    return 9
                k = 0
                while k < getw(w,parameter):
                    if getw(w,p+96+k*8) != getw(w,parameter+96+k*8):
                        return 9
                    k = k+1
            if optimizer_audit_reference(w,node,incarnation) != 0:
                return 9
        j = j+1
    if present != 0 and optimizer_audit_reference(w,getw(w,key),getw(w,key+8)) != 0:
        return 9
    return 0

def owned_adam_audit_entry(w:mutbuf, blob:u64, i:u64, first:u64) -> u64:
    ep = owned_adam_entry_at(w,blob,i)
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
    key = owned_adam_entry_at(w,blob,key_index)
    if getw(w,key) != node or getw(w,key+8) != incarnation or getw(w,key+16) != key_index:
        return 9
    if key_index != i:
        j = 24
        while j < 80:
            if getw(w,ep+j) != 0:
                return 9
            j = j+8
    else:
        j = 0
        while j < i:
            other = owned_adam_entry_at(w,blob,j)
            if getw(w,other) == node and getw(w,other+8) == incarnation:
                return 9
            j = j+1
        if owned_adam_audit_state(w,key,p) != 0:
            return 9
    return choose(key_index == i,1,2)

def owned_adam_audit_config(w:mutbuf, config:u64) -> u64:
    j = 0
    while j < 8:
        value = getw(w,config+j*8)
        if j < 5:
            if value > 4294967295:
                return 9
        elif j == 6:
            if value > 1:
                return 9
        elif value != 0:
            return 9
        j = j+1
    return 0

def owned_adam_audit_group(w:mutbuf, blob:u64, group:u64, first:u64) -> u64:
    gp = blob+96+group*80
    count = getw(w,blob+8)
    amount = getw(w,gp+72)
    if getw(w,gp+64) != first or first > count or amount > count-first:
        return 0
    if owned_adam_audit_config(w,gp) != 0:
        return 0
    duplicates = 0
    i = first
    while i < first+amount:
        status = owned_adam_audit_entry(w,blob,i,first)
        if status != 1 and status != 2:
            return 0
        duplicates = duplicates+status-1
        i = i+1
    return duplicates+1

def owned_adam_audit_record(w:mutbuf, row:u64) -> u64:
    groups = so_uget(w,row+24)
    count = so_uget(w,row+32)
    algorithm = so_uget(w,row+56)
    if groups == 0 or groups > 1024 or count > 4096 or algorithm < 1 or algorithm > 2:
        return 9
    if so_uget(w,row+48) > count or so_uget(w,row+40) != count-so_uget(w,row+48):
        return 9
    blob = private_hold(w,so_uget(w,row+16),96+groups*80+count*80)
    if blob == 0 or getw(w,blob) != groups or getw(w,blob+8) != count or getw(w,blob+16) != algorithm or getw(w,blob+24) != 0:
        return 9
    if owned_adam_audit_config(w,blob+32) != 0:
        return 9
    first = 0
    duplicate_count = 0
    group = 0
    while group < groups:
        result = owned_adam_audit_group(w,blob,group,first)
        if result == 0:
            return 9
        duplicate_count = duplicate_count+result-1
        first = first+getw(w,blob+96+group*80+72)
        group = group+1
    if first != count or duplicate_count != so_uget(w,row+48):
        return 9
    return 0

def optimizer_audit_record(w:mutbuf, row:u64) -> u64:
    algorithm = so_uget(w,row+56)
    if algorithm == 0:
        return sgd_optimizer_audit_record(w,row)
    if algorithm == 1 or algorithm == 2:
        return owned_adam_audit_record(w,row)
    return 9
