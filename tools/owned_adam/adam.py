def owned_adam_blob(w:mutbuf, id:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    row = so_slot_lookup(w,8,resolve_handle(w,id))
    if row == 0:
        return terror(w,1)
    groups = so_uget(w,row+24)
    count = so_uget(w,row+32)
    algorithm = so_uget(w,row+56)
    if groups == 0 or groups > 1024 or count > 4096 or algorithm < 1 or algorithm > 2:
        return terror(w,9)
    blob = private_hold(w,so_uget(w,row+16),96+groups*80+count*80)
    if blob == 0 or getw(w,blob) != groups or getw(w,blob+8) != count or getw(w,blob+16) != algorithm or getw(w,blob+24) != 0:
        return terror(w,9)
    return blob

def owned_adam_group_at(w:mutbuf, blob:u64, group:u64) -> u64:
    if group >= getw(w,blob):
        return terror(w,7)
    return blob+96+group*80

def owned_adam_entry_at(w:mutbuf, blob:u64, index:u64) -> u64:
    return blob+96+getw(w,blob)*80+index*80

def owned_adam_occurrence(w:mutbuf, blob:u64, group:u64, occurrence:u64) -> u64:
    gp = owned_adam_group_at(w,blob,group)
    if gp == 0:
        return 0
    if occurrence >= getw(w,gp+72):
        return terror(w,7)
    ep = owned_adam_entry_at(w,blob,getw(w,gp+64)+occurrence)
    if private_tensor(w,getw(w,ep),getw(w,ep+8)) == 0:
        return terror(w,9)
    return ep

def owned_adam_key(w:mutbuf, blob:u64, occurrence:u64) -> u64:
    index = getw(w,occurrence+16)
    if index >= getw(w,blob+8):
        return terror(w,9)
    key = owned_adam_entry_at(w,blob,index)
    if getw(w,key) != getw(w,occurrence) or getw(w,key+8) != getw(w,occurrence+8) or getw(w,key+16) != index:
        return terror(w,9)
    return key

def owned_adam_word(w:mutbuf, field:u64, value:u64) -> u64:
    if field < 5:
        if value > 4294967295:
            return terror(w,7)
    elif field == 6:
        if value > 1:
            return terror(w,7)
    elif value != 0:
        return terror(w,7)
    return 1

def owned_adam_defaults(s:readbuf, w:mutbuf, offset:u64, end:u64) -> u64:
    if optimizer_payload(s,w,offset,64,end) == 0:
        return 0
    j = 0
    while j < 8:
        if owned_adam_word(w,j,getr(s,offset+j*8)) == 0:
            return 0
        j = j+1
    lr = getr(s,offset)
    eps = getr(s,offset+24)
    decay = getr(s,offset+32)
    if (ltf(0,lr) == 0 and absf(lr) != 0) or (ltf(0,eps) == 0 and absf(eps) != 0) or (ltf(0,decay) == 0 and absf(decay) != 0):
        return terror(w,7)
    beta1 = getr(s,offset+8)
    beta2 = getr(s,offset+16)
    if ltf(beta1,0) != 0 or ltf(beta1,1065353216) == 0 or ltf(beta2,0) != 0 or ltf(beta2,1065353216) == 0:
        return terror(w,7)
    return 1

def owned_adam_group_payload(s:readbuf, w:mutbuf, offset:u64, end:u64) -> u64:
    if optimizer_payload(s,w,offset,80,end) == 0:
        return 0
    count = getr(s,offset)
    mask = getr(s,offset+8)
    if count > 4096 or mask > 127:
        return terror(w,7)
    if optimizer_payload(s,w,offset,80+count*8,end) == 0:
        return 0
    j = 0
    while j < 8:
        value = getr(s,offset+16+j*8)
        if (mask & (1 << j)) == 0:
            if value != 0:
                return terror(w,7)
        elif owned_adam_word(w,j,value) == 0:
            return 0
        j = j+1
    return offset+80+count*8

def owned_adam_fill_group(s:readbuf, w:mutbuf, blob:u64, group:u64, offset:u64, first:u64) -> u64:
    gp = blob+96+group*80
    mask = getr(s,offset+8)
    count = getr(s,offset)
    j = 0
    while j < 8:
        value = getw(w,blob+32+j*8)
        if (mask & (1 << j)) != 0:
            value = getr(s,offset+16+j*8)
        putw(w,gp+j*8,value)
        j = j+1
    putw(w,gp+64,first)
    putw(w,gp+72,count)
    i = 0
    duplicates = 0
    while i < count:
        t = public_node(w,getr(s,offset+80+i*8))
        if t == 0:
            return 0
        p = dptr(w,t)
        if getw(w,p+40) != 0:
            return terror(w,5)
        incarnation = getw(w,p+320)
        index = first+i
        key = index
        j = 0
        while j < index:
            previous = owned_adam_entry_at(w,blob,j)
            if getw(w,previous) == t and getw(w,previous+8) == incarnation:
                if j < first:
                    return terror(w,7)
                key = getw(w,previous+16)
                j = index
            else:
                j = j+1
        ep = owned_adam_entry_at(w,blob,index)
        putw(w,ep,t)
        putw(w,ep+8,incarnation)
        putw(w,ep+16,key)
        if key != index:
            duplicates = duplicates+1
        if private_tensor_retain(w,t,incarnation) == 0:
            return 0
        i = i+1
    return duplicates+1

def owned_adam_create(s:readbuf, w:mutbuf, algorithm:u64, defaults:u64, groups_at:u64, end:u64) -> u64:
    if algorithm < 1 or algorithm > 2:
        return terror(w,7)
    if owned_adam_defaults(s,w,defaults,end) == 0 or optimizer_payload(s,w,groups_at,8,end) == 0:
        return 0
    groups = getr(s,groups_at)
    if groups == 0 or groups > 1024:
        return terror(w,7)
    offset = groups_at+8
    count = 0
    i = 0
    while i < groups:
        next = owned_adam_group_payload(s,w,offset,end)
        if next == 0:
            return 0
        count = count+getr(s,offset)
        if count > 4096:
            return terror(w,3)
        offset = next
        i = i+1
    if storage_capacity(w,8,0) == 0:
        return 0
    blob = alloc(w,96+groups*80+count*80)
    if blob == 0:
        return 0
    hold = getw(w,80)
    putw(w,blob,groups)
    putw(w,blob+8,count)
    putw(w,blob+16,algorithm)
    j = 0
    while j < 8:
        putw(w,blob+32+j*8,getr(s,defaults+j*8))
        j = j+1
    offset = groups_at+8
    first = 0
    duplicates = 0
    i = 0
    while i < groups:
        result = owned_adam_fill_group(s,w,blob,i,offset,first)
        if result == 0:
            return 0
        duplicates = duplicates+result-1
        amount = getr(s,offset)
        first = first+amount
        offset = offset+80+amount*8
        i = i+1
    id = so_slot_new(w,8)
    p = so_slot_lookup(w,8,id)
    if p == 0:
        return terror(w,9)
    so_uput(w,p+16,hold)
    so_uput(w,p+24,groups)
    so_uput(w,p+32,count)
    so_uput(w,p+40,count-duplicates)
    so_uput(w,p+48,duplicates)
    so_uput(w,p+56,algorithm)
    return id

def owned_adam_add_group(s:readbuf, w:mutbuf, id:u64, offset:u64, end:u64) -> u64:
    old = owned_adam_blob(w,id)
    if old == 0 or owned_adam_group_payload(s,w,offset,end) == 0:
        return 0
    p = so_slot_lookup(w,8,resolve_handle(w,id))
    groups = getw(w,old)
    count = getw(w,old+8)
    extra = getr(s,offset)
    if groups >= 1024 or extra > 4096-count:
        return terror(w,3)
    blob = alloc(w,96+(groups+1)*80+(count+extra)*80)
    if blob == 0:
        return 0
    hold = getw(w,80)
    move_within(w,object_address(w,blob),object_address(w,old),96+groups*80)
    putw(w,blob,groups+1)
    putw(w,blob+8,count+extra)
    move_within(w,object_address(w,owned_adam_entry_at(w,blob,0)),object_address(w,owned_adam_entry_at(w,old,0)),count*80)
    result = owned_adam_fill_group(s,w,blob,groups,offset,count)
    if result == 0:
        return 0
    if hold_release(w,so_uget(w,p+16)) == 0:
        return 0
    so_uput(w,p+16,hold)
    so_uput(w,p+24,groups+1)
    so_uput(w,p+32,count+extra)
    duplicates = so_uget(w,p+48)+result-1
    so_uput(w,p+40,count+extra-duplicates)
    so_uput(w,p+48,duplicates)
    return 1

def owned_adam_drop(w:mutbuf, id:u64) -> u64:
    blob = owned_adam_blob(w,id)
    if blob == 0:
        return 0
    id = resolve_handle(w,id)
    p = so_slot_lookup(w,8,id)
    hold = so_uget(w,p+16)
    i = 0
    while i < getw(w,blob+8):
        ep = owned_adam_entry_at(w,blob,i)
        if private_tensor_release(w,getw(w,ep),getw(w,ep+8)) == 0:
            return 0
        if getw(w,ep+16) == i and getw(w,ep+24) != 0:
            if private_tensor_release(w,getw(w,ep),getw(w,ep+8)) == 0:
                return 0
            j = 0
            while j < 3:
                if private_tensor_release(w,getw(w,ep+32+j*16),getw(w,ep+40+j*16)) == 0:
                    return 0
                j = j+1
        i = i+1
    if hold_release(w,hold) == 0 or so_slot_drop(w,8,id) != 0:
        return terror(w,9)
    return graph_collect(w)

def owned_adam_zero(w:mutbuf, id:u64, none:u64) -> u64:
    if none > 1:
        return terror(w,7)
    blob = owned_adam_blob(w,id)
    if blob == 0:
        return 0
    i = 0
    while i < getw(w,blob+8):
        ep = owned_adam_entry_at(w,blob,i)
        node = getw(w,ep)
        if private_tensor(w,node,getw(w,ep+8)) == 0:
            return terror(w,9)
        if none != 0:
            zero_grad_none(w,node)
        else:
            zero_grad(w,node)
        if getw(w,0) != 0:
            return 0
        i = i+1
    return 1

def owned_adam_new_step(w:mutbuf) -> u64:
    node = descriptor(w,0,128,0,1)
    if node == 0:
        return 0
    p = dptr(w,node)
    if private_tensor_retain(w,node,getw(w,p+320)) == 0:
        return 0
    putw(w,p+200,0)
    return node

def owned_adam_initialize(w:mutbuf, key:u64) -> u64:
    parameter = getw(w,key)
    if getw(w,key+24) != 0:
        return 1
    if private_tensor_retain(w,parameter,getw(w,key+8)) == 0:
        return 0
    step = owned_adam_new_step(w)
    if step == 0:
        return 0
    average = owned_adam_new_moment(w,parameter)
    if average == 0:
        return 0
    square = owned_adam_new_moment(w,parameter)
    if square == 0:
        return 0
    putw(w,key+24,14)
    putw(w,key+32,step)
    putw(w,key+40,getw(w,dptr(w,step)+320))
    putw(w,key+48,average)
    putw(w,key+56,getw(w,dptr(w,average)+320))
    putw(w,key+64,square)
    putw(w,key+72,getw(w,dptr(w,square)+320))
    return 1

def owned_adam_plan_group(w:mutbuf, blob:u64, group:u64) -> u64:
    gp = blob+96+group*80
    first = getw(w,gp+64)
    count = getw(w,gp+72)
    i = 0
    while i < count:
        ep = owned_adam_entry_at(w,blob,first+i)
        p = private_tensor(w,getw(w,ep),getw(w,ep+8))
        if p == 0:
            return terror(w,9)
        if getw(w,p+360) != 0:
            key = owned_adam_key(w,blob,ep)
            if key == 0 or owned_adam_initialize(w,key) == 0:
                return 0
        i = i+1
    return 1

def owned_adam_step_exponent(w:mutbuf, step:u64) -> u64:
    value = addf(tget(w,step,0),1065353216)
    exponent = (value >> 23) & 255
    if (value & 2147483648) != 0 or exponent < 127 or exponent > 158:
        return terror(w,7)
    mantissa = (value & 8388607) | 8388608
    if exponent >= 150:
        return mantissa << (exponent-150)
    shift = 150-exponent
    if (mantissa & ((1 << shift)-1)) != 0:
        return terror(w,7)
    return mantissa >> shift

def owned_adam_config(w:mutbuf, group:u64, algorithm:u64, exponent:u64) -> u64:
    j = 0
    while j < 5:
        putw(w,2080+j*8,getw(w,group+j*8))
        j = j+1
    putw(w,2120,algorithm-1)
    putw(w,2128,subf(1065353216,powuf(getw(w,group+8),exponent)))
    putw(w,2136,subf(1065353216,powuf(getw(w,group+16),exponent)))
    putw(w,2144,getw(w,group+48))
    return 2080

def owned_optimizer_gradient_value(w:mutbuf, node:u64, index:u64) -> u64:
    return tgrad(w,node,index)

def owned_adam_gradient(w:mutbuf, parameter:u64, gradient:u64, cfg:u64) -> u64:
    if getw(w,cfg+64) != 0:
        gradient = negf(gradient)
    return adam_gradient(w,parameter,gradient,cfg)

def owned_adam_versions_ready(w:mutbuf, key:u64, group:u64, algorithm:u64) -> u64:
    count = 1
    if algorithm == 2 and absf(getw(w,group+32)) != 0:
        count = 2
    if version_can_bump(w,getw(w,key),count) == 0:
        return 0
    j = 0
    while j < 3:
        state = getw(w,key+32+j*16)
        if private_tensor(w,state,getw(w,key+40+j*16)) == 0:
            return terror(w,9)
        if version_can_bump(w,state,choose(j == 2,2,1)) == 0:
            return 0
        j = j+1
    return 1

def owned_adam_apply(w:mutbuf, blob:u64, group:u64, occurrence:u64) -> u64:
    parameter = getw(w,occurrence)
    p = private_tensor(w,parameter,getw(w,occurrence+8))
    if p == 0:
        return terror(w,9)
    if getw(w,p+360) == 0:
        return 1
    key = owned_adam_key(w,blob,occurrence)
    if key == 0 or getw(w,key+24) != 14:
        return terror(w,9)
    if owned_adam_versions_ready(w,key,group,getw(w,blob+16)) == 0:
        return 0
    step = getw(w,key+32)
    exponent = owned_adam_step_exponent(w,step)
    if exponent == 0:
        return 0
    cfg = owned_adam_config(w,group,getw(w,blob+16),exponent)
    put4(w,taddr(w,step,0),addf(tget(w,step,0),1065353216))
    bump_version(w,step)
    average = getw(w,key+48)
    square = getw(w,key+64)
    i = 0
    while i < getw(w,p+8):
        owned_adam_element(w,parameter,average,square,i,cfg)
        i = i+1
    if getw(w,0) != 0:
        return 0
    return adam_versions(w,parameter,average,square,cfg)

def owned_adam_step(w:mutbuf, id:u64) -> u64:
    blob = owned_adam_blob(w,id)
    if blob == 0:
        return 0
    group = 0
    while group < getw(w,blob):
        gp = blob+96+group*80
        if owned_adam_plan_group(w,blob,group) == 0:
            return 0
        first = getw(w,gp+64)
        i = 0
        while i < getw(w,gp+72):
            if owned_adam_apply(w,blob,gp,owned_adam_entry_at(w,blob,first+i)) == 0:
                return 0
            i = i+1
        group = group+1
    return graph_collect(w)

def owned_adam_group_inventory(w:mutbuf, o:mutbuf, at:u64, blob:u64, group:u64) -> u64:
    gp = owned_adam_group_at(w,blob,group)
    if gp == 0:
        return 0
    count = getw(w,gp+72)
    size = 80+count*16
    if size > length(o)-at:
        return terror(w,3)
    putw(o,at,group)
    putw(o,at+8,count)
    j = 0
    while j < 8:
        putw(o,at+16+j*8,getw(w,gp+j*8))
        j = j+1
    j = 0
    while j < count:
        ep = owned_adam_entry_at(w,blob,getw(w,gp+64)+j)
        p = private_tensor(w,getw(w,ep),getw(w,ep+8))
        if p == 0:
            return terror(w,9)
        putw(o,at+80+j*16,getw(w,ep+8))
        token = 0
        if getw(w,p+200) == 1:
            token = getw(w,p+192)
        putw(o,at+88+j*16,token)
        j = j+1
    return size

def owned_adam_state_inventory(w:mutbuf, o:mutbuf, at:u64, key:u64) -> u64:
    putw(o,at,getw(w,key+8))
    putw(o,at+8,getw(w,key+24))
    j = 0
    while j < 3:
        incarnation = getw(w,key+40+j*16)
        putw(o,at+16+j*16,incarnation)
        token = 0
        if getw(w,key+24) != 0:
            p = private_tensor(w,getw(w,key+32+j*16),incarnation)
            if p == 0:
                return terror(w,9)
            if getw(w,p+200) == 1:
                token = getw(w,p+192)
        putw(o,at+24+j*16,token)
        j = j+1
    return 64

def owned_adam_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64, end:u64) -> u64:
    code = getr(s,cp)
    a = getr(s,cp+8)
    b = getr(s,cp+16)
    c = getr(s,cp+24)
    d = getr(s,cp+32)
    result = 0
    payload = 0
    first = getw(w,8)
    if getr(s,cp+40) != 0 or getr(s,cp+48) != 0 or getr(s,cp+56) != 0:
        terror(w,7)
    if code == 77:
        if d < 1 or d > 3:
            terror(w,7)
    elif d != 0:
        terror(w,7)
    if code != 71 and code != 75 and code != 76 and code != 77 and c != 0:
        terror(w,7)
    if (code == 72 or code == 78) and b != 0:
        terror(w,7)
    if getw(w,0) == 0:
        if code == 71:
            if length(o)-at-32 < 32:
                terror(w,3)
            else:
                result = owned_adam_create(s,w,a,b,c,end)
                if result != 0:
                    payload = optimizer_inventory(w,o,at+32,result)
        elif code == 72:
            result = owned_adam_step(w,a)
        elif code == 73:
            result = owned_adam_zero(w,a,b)
        elif code == 78:
            result = owned_adam_drop(w,a)
        elif code == 79:
            if length(o)-at-32 < 32:
                terror(w,3)
            else:
                result = owned_adam_add_group(s,w,a,b,end)
                if result != 0:
                    payload = optimizer_inventory(w,o,at+32,a)
        else:
            blob = owned_adam_blob(w,a)
            if blob != 0:
                if code == 74:
                    payload = owned_adam_group_inventory(w,o,at+32,blob,b)
                    if payload != 0:
                        result = getw(o,at+40)
                elif code == 75 or code == 76 or code == 77:
                    ep = owned_adam_occurrence(w,blob,b,c)
                    if ep != 0:
                        if code == 75:
                            result = publish_tensor(w,getw(w,ep))
                        else:
                            key = owned_adam_key(w,blob,ep)
                            if key != 0:
                                if code == 76:
                                    if length(o)-at-32 < 64:
                                        terror(w,3)
                                    else:
                                        payload = owned_adam_state_inventory(w,o,at+32,key)
                                        result = choose(getw(w,key+24) != 0,1,0)
                                elif getw(w,key+24) != 0:
                                    result = publish_tensor(w,getw(w,key+16+d*16))
                else:
                    terror(w,4)
    if getw(w,0) == 0:
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

def owned_optimizer_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64, end:u64) -> u64:
    code = getr(s,cp)
    if code == 95:
        return scalar_add_nograd_command(s,o,w,cp,at)
    if code == 71:
        algorithm = getr(s,cp+8)
        if algorithm == 1 or algorithm == 2:
            return owned_adam_command(s,o,w,cp,at,end)
    elif code != 90:
        row = so_slot_lookup(w,8,resolve_handle(w,getr(s,cp+8)))
        if row != 0:
            algorithm = so_uget(w,row+56)
            if algorithm == 1 or algorithm == 2:
                return owned_adam_command(s,o,w,cp,at,end)
    return owned_sgd_command(s,o,w,cp,at,end)
