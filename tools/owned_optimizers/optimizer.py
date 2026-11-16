def optimizer_sgd_element(w:mutbuf, gp:u64, parameter:u64, buffer:u64, created:u64, index:u64) -> u64:
    p = dptr(w,parameter)
    grad = get4(w,getw(w,p+24)+index*4)
    if getw(w,gp+40) != 0:
        grad = negf(grad)
    address = taddr(w,parameter,index)
    parameter_value = get4(w,address)
    decay = getw(w,gp+24)
    if absf(decay) != 0:
        grad = addf(grad,mulf(decay,parameter_value))
    if buffer != 0:
        value = grad
        state_address = taddr(w,buffer,index)
        if created == 0:
            value = addf(mulf(get4(w,state_address),getw(w,gp+8)),mulf(subf(1065353216,getw(w,gp+16)),grad))
        put4(w,state_address,value)
        if getw(w,gp+32) != 0:
            grad = addf(grad,mulf(getw(w,gp+8),value))
        else:
            grad = value
    put4(w,address,addf(parameter_value,mulf(negf(getw(w,gp)),grad)))
    return 1

def optimizer_blob(w:mutbuf, id:u64) -> u64:
    if getw(w,0) != 0:
        return 0
    id = resolve_handle(w,id)
    p = so_slot_lookup(w,8,id)
    if p == 0:
        return terror(w,1)
    groups = so_uget(w,p+24)
    count = so_uget(w,p+32)
    if groups == 0 or groups > 1024 or count > 4096 or so_uget(w,p+56) != 0:
        return terror(w,9)
    address = private_hold(w,so_uget(w,p+16),64+groups*64+count*48)
    if address == 0 or getw(w,address) != groups or getw(w,address+8) != count:
        return terror(w,9)
    return address

def optimizer_group_at(w:mutbuf, blob:u64, group:u64) -> u64:
    if group >= getw(w,blob):
        return terror(w,7)
    return blob+64+group*64

def optimizer_entry_at(w:mutbuf, blob:u64, index:u64) -> u64:
    return blob+64+getw(w,blob)*64+index*48

def optimizer_occurrence(w:mutbuf, blob:u64, group:u64, occurrence:u64) -> u64:
    gp = optimizer_group_at(w,blob,group)
    if gp == 0:
        return 0
    if occurrence >= getw(w,gp+56):
        return terror(w,7)
    ep = optimizer_entry_at(w,blob,getw(w,gp+48)+occurrence)
    if private_tensor(w,getw(w,ep),getw(w,ep+8)) == 0:
        return terror(w,9)
    return ep

def optimizer_key(w:mutbuf, blob:u64, occurrence:u64) -> u64:
    index = getw(w,occurrence+16)
    if index >= getw(w,blob+8):
        return terror(w,9)
    key = optimizer_entry_at(w,blob,index)
    if getw(w,key) != getw(w,occurrence) or getw(w,key+8) != getw(w,occurrence+8) or getw(w,key+16) != index:
        return terror(w,9)
    return key

def optimizer_word(w:mutbuf, field:u64, value:u64) -> u64:
    if field < 4:
        if value > 4294967295:
            return terror(w,7)
    elif value > 1:
        return terror(w,7)
    return 1

def optimizer_payload(s:readbuf, w:mutbuf, offset:u64, size:u64, end:u64) -> u64:
    if offset < end or offset > length(s) or size > length(s)-offset or (offset & 7) != 0:
        return terror(w,100)
    return 1

def optimizer_defaults(s:readbuf, w:mutbuf, offset:u64, end:u64) -> u64:
    if optimizer_payload(s,w,offset,48,end) == 0:
        return 0
    j = 0
    while j < 6:
        if optimizer_word(w,j,getr(s,offset+j*8)) == 0:
            return 0
        j = j+1
    lr = getr(s,offset)
    momentum = getr(s,offset+8)
    dampening = getr(s,offset+16)
    decay = getr(s,offset+24)
    if ltf(lr,0) != 0 or ltf(momentum,0) != 0 or ltf(decay,0) != 0:
        return terror(w,7)
    if getr(s,offset+32) != 0:
        if ltf(momentum,0) != 0 or absf(momentum) == 0 or absf(dampening) != 0:
            return terror(w,7)
    return 1

def optimizer_group_payload(s:readbuf, w:mutbuf, offset:u64, end:u64) -> u64:
    if optimizer_payload(s,w,offset,64,end) == 0:
        return 0
    count = getr(s,offset)
    mask = getr(s,offset+8)
    if count > 4096 or mask > 63:
        return terror(w,7)
    if optimizer_payload(s,w,offset,64+count*8,end) == 0:
        return 0
    j = 0
    while j < 6:
        value = getr(s,offset+16+j*8)
        if (mask & (1 << j)) == 0:
            if value != 0:
                return terror(w,7)
        elif optimizer_word(w,j,value) == 0:
            return 0
        j = j+1
    return offset+64+count*8

def optimizer_fill_group(s:readbuf, w:mutbuf, blob:u64, group:u64, offset:u64, first:u64) -> u64:
    gp = blob+64+group*64
    mask = getr(s,offset+8)
    count = getr(s,offset)
    j = 0
    while j < 6:
        value = getw(w,blob+16+j*8)
        if (mask & (1 << j)) != 0:
            value = getr(s,offset+16+j*8)
        putw(w,gp+j*8,value)
        j = j+1
    putw(w,gp+48,first)
    putw(w,gp+56,count)
    i = 0
    duplicates = 0
    while i < count:
        t = public_node(w,getr(s,offset+64+i*8))
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
            previous = optimizer_entry_at(w,blob,j)
            if getw(w,previous) == t and getw(w,previous+8) == incarnation:
                if j < first:
                    return terror(w,7)
                key = getw(w,previous+16)
                j = index
            else:
                j = j+1
        ep = optimizer_entry_at(w,blob,index)
        putw(w,ep,t)
        putw(w,ep+8,incarnation)
        putw(w,ep+16,key)
        if key != index:
            duplicates = duplicates+1
        if private_tensor_retain(w,t,incarnation) == 0:
            return 0
        i = i+1
    return duplicates+1

def optimizer_create(s:readbuf, w:mutbuf, algorithm:u64, defaults:u64, groups_at:u64, end:u64) -> u64:
    if algorithm != 0:
        return terror(w,7)
    if optimizer_defaults(s,w,defaults,end) == 0 or optimizer_payload(s,w,groups_at,8,end) == 0:
        return 0
    groups = getr(s,groups_at)
    if groups == 0 or groups > 1024:
        return terror(w,7)
    offset = groups_at+8
    count = 0
    i = 0
    while i < groups:
        next = optimizer_group_payload(s,w,offset,end)
        if next == 0:
            return 0
        count = count+getr(s,offset)
        if count > 4096:
            return terror(w,3)
        offset = next
        i = i+1
    if storage_capacity(w,8,0) == 0:
        return 0
    blob = alloc(w,64+groups*64+count*48)
    if blob == 0:
        return 0
    hold = getw(w,80)
    putw(w,blob,groups)
    putw(w,blob+8,count)
    j = 0
    while j < 6:
        putw(w,blob+16+j*8,getr(s,defaults+j*8))
        j = j+1
    offset = groups_at+8
    first = 0
    duplicates = 0
    i = 0
    while i < groups:
        result = optimizer_fill_group(s,w,blob,i,offset,first)
        if result == 0:
            return 0
        duplicates = duplicates+result-1
        amount = getr(s,offset)
        first = first+amount
        offset = offset+64+amount*8
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
    return id

def optimizer_add_group(s:readbuf, w:mutbuf, id:u64, offset:u64, end:u64) -> u64:
    old = optimizer_blob(w,id)
    if old == 0 or optimizer_group_payload(s,w,offset,end) == 0:
        return 0
    id = resolve_handle(w,id)
    p = so_slot_lookup(w,8,id)
    groups = getw(w,old)
    count = getw(w,old+8)
    extra = getr(s,offset)
    if groups >= 1024 or extra > 4096-count:
        return terror(w,3)
    blob = alloc(w,64+(groups+1)*64+(count+extra)*48)
    if blob == 0:
        return 0
    hold = getw(w,80)
    move_within(w,object_address(w,blob),object_address(w,old),64+groups*64)
    putw(w,blob,groups+1)
    putw(w,blob+8,count+extra)
    move_within(w,object_address(w,optimizer_entry_at(w,blob,0)),object_address(w,optimizer_entry_at(w,old,0)),count*48)
    result = optimizer_fill_group(s,w,blob,groups,offset,count)
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

def optimizer_drop(w:mutbuf, id:u64) -> u64:
    blob = optimizer_blob(w,id)
    if blob == 0:
        return 0
    id = resolve_handle(w,id)
    p = so_slot_lookup(w,8,id)
    hold = so_uget(w,p+16)
    i = 0
    while i < getw(w,blob+8):
        ep = optimizer_entry_at(w,blob,i)
        if private_tensor_release(w,getw(w,ep),getw(w,ep+8)) == 0:
            return 0
        if getw(w,ep+16) == i and getw(w,ep+24) != 0:
            if private_tensor_release(w,getw(w,ep),getw(w,ep+8)) == 0:
                return 0
            if private_tensor_release(w,getw(w,ep+32),getw(w,ep+40)) == 0:
                return 0
        i = i+1
    if hold_release(w,hold) == 0 or so_slot_drop(w,8,id) != 0:
        return terror(w,9)
    return graph_collect(w)

def optimizer_zero(w:mutbuf, id:u64, none:u64) -> u64:
    if none > 1:
        return terror(w,7)
    blob = optimizer_blob(w,id)
    if blob == 0:
        return 0
    i = 0
    while i < getw(w,blob+8):
        ep = optimizer_entry_at(w,blob,i)
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

def optimizer_new_momentum(w:mutbuf, parameter:u64) -> u64:
    p = dptr(w,parameter)
    t = descriptor(w,getw(w,p),p+96,0,1)
    if t == 0:
        return 0
    q = dptr(w,t)
    if is_dense(w,parameter) != 0:
        j = 0
        while j < getw(w,p):
            putw(w,q+128+j*8,getw(w,p+128+j*8))
            j = j+1
    if private_tensor_retain(w,t,getw(w,q+320)) == 0:
        return 0
    putw(w,q+200,0)
    return t

def optimizer_plan_group(w:mutbuf, blob:u64, group:u64, plan:u64) -> u64:
    gp = blob+64+group*64
    first = getw(w,gp+48)
    count = getw(w,gp+56)
    momentum = getw(w,gp+8)
    i = 0
    while i < count:
        ep = optimizer_entry_at(w,blob,first+i)
        node = getw(w,ep)
        p = private_tensor(w,node,getw(w,ep+8))
        if p == 0:
            return terror(w,9)
        item = plan+i*32
        if getw(w,p+360) != 0:
            putw(w,item,node)
            if absf(momentum) != 0:
                key = optimizer_key(w,blob,ep)
                if key == 0:
                    return 0
                if getw(w,key+24) == 0:
                    if private_tensor_retain(w,node,getw(w,ep+8)) == 0:
                        return 0
                    putw(w,key+24,1)
                else:
                    buffer = getw(w,key+32)
                    if buffer != 0:
                        if private_tensor(w,buffer,getw(w,key+40)) == 0:
                            return terror(w,9)
                        putw(w,item+8,buffer)
                putw(w,item+16,key)
        i = i+1
    return 1

def optimizer_sgd_occurrence(w:mutbuf, gp:u64, item:u64) -> u64:
    parameter = getw(w,item)
    if parameter == 0:
        return 1
    p = dptr(w,parameter)
    n = getw(w,p+8)
    if getw(w,p+360) == 0 or getw(w,p+248) == 0 or getw(w,p+24) == 0:
        return terror(w,9)
    if version_can_bump(w,parameter,1) == 0:
        return 0
    momentum = getw(w,gp+8)
    buffer = getw(w,item+8)
    created = 0
    if absf(momentum) != 0:
        if buffer == 0:
            buffer = optimizer_new_momentum(w,parameter)
            if buffer == 0:
                return 0
            putw(w,item+8,buffer)
            putw(w,item+24,1)
            created = 1
        elif version_can_bump(w,buffer,2) == 0:
            return 0
    i = 0
    while i < n:
        optimizer_sgd_element(w,gp,parameter,buffer,created,i)
        i = i+1
    if absf(momentum) != 0 and created == 0:
        bump_version(w,buffer)
        bump_version(w,buffer)
    return bump_version(w,parameter)

def optimizer_commit_group(w:mutbuf, gp:u64, plan:u64) -> u64:
    i = 0
    while i < getw(w,gp+56):
        item = plan+i*32
        key = getw(w,item+16)
        if getw(w,item) != 0 and key != 0:
            buffer = getw(w,item+8)
            incarnation = getw(w,dptr(w,buffer)+320)
            old = getw(w,key+32)
            if buffer != old:
                if private_tensor_retain(w,buffer,incarnation) == 0:
                    return 0
                if old != 0 and private_tensor_release(w,old,getw(w,key+40)) == 0:
                    return 0
                putw(w,key+32,buffer)
                putw(w,key+40,incarnation)
            if getw(w,item+24) != 0:
                if private_tensor_release(w,buffer,incarnation) == 0:
                    return 0
        i = i+1
    return 1

def optimizer_step(w:mutbuf, id:u64) -> u64:
    blob = optimizer_blob(w,id)
    if blob == 0:
        return 0
    maximum = 0
    i = 0
    while i < getw(w,blob):
        count = getw(w,blob+64+i*64+56)
        if count > maximum:
            maximum = count
        i = i+1
    plan = 0
    hold = 0
    if maximum != 0:
        plan = alloc(w,maximum*32)
        if plan == 0:
            return 0
        hold = getw(w,80)
    i = 0
    while i < getw(w,blob):
        gp = blob+64+i*64
        count = getw(w,gp+56)
        if count != 0:
            memory_fill(w,plan,maximum*32,0)
            if optimizer_plan_group(w,blob,i,plan) == 0:
                return 0
            j = 0
            while j < count:
                if optimizer_sgd_occurrence(w,gp,plan+j*32) == 0:
                    return 0
                j = j+1
            if optimizer_commit_group(w,gp,plan) == 0:
                return 0
        i = i+1
    if hold_release(w,hold) == 0:
        return 0
    return graph_collect(w)

def optimizer_inventory(w:mutbuf, o:mutbuf, at:u64, id:u64) -> u64:
    p = so_slot_lookup(w,8,resolve_handle(w,id))
    if p == 0:
        return terror(w,1)
    putw(o,at,so_uget(w,p+24))
    putw(o,at+8,so_uget(w,p+32))
    putw(o,at+16,so_uget(w,p+40))
    putw(o,at+24,so_uget(w,p+48))
    return 32

def optimizer_group_inventory(w:mutbuf, o:mutbuf, at:u64, blob:u64, group:u64) -> u64:
    gp = optimizer_group_at(w,blob,group)
    if gp == 0:
        return 0
    count = getw(w,gp+56)
    size = 64+count*16
    if size > length(o)-at:
        return terror(w,3)
    putw(o,at,group)
    putw(o,at+8,count)
    j = 0
    while j < 6:
        putw(o,at+16+j*8,getw(w,gp+j*8))
        j = j+1
    j = 0
    while j < count:
        ep = optimizer_entry_at(w,blob,getw(w,gp+48)+j)
        p = private_tensor(w,getw(w,ep),getw(w,ep+8))
        if p == 0:
            return terror(w,9)
        putw(o,at+64+j*16,getw(w,ep+8))
        token = 0
        if getw(w,p+200) == 1:
            token = getw(w,p+192)
        putw(o,at+72+j*16,token)
        j = j+1
    return size

def optimizer_state_inventory(w:mutbuf, o:mutbuf, at:u64, key:u64) -> u64:
    putw(o,at,getw(w,key+8))
    putw(o,at+8,getw(w,key+24))
    putw(o,at+16,getw(w,key+40))
    token = 0
    if getw(w,key+24) != 0:
        p = private_tensor(w,getw(w,key+32),getw(w,key+40))
        if p == 0:
            return terror(w,9)
        if getw(w,p+200) == 1:
            token = getw(w,p+192)
    putw(o,at+24,token)
    return 32

def owned_optimizer_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64, end:u64) -> u64:
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
    if d != 0:
        terror(w,7)
    if code != 71 and code != 75 and code != 76 and code != 77 and c != 0:
        terror(w,7)
    if (code == 72 or code == 78) and b != 0:
        terror(w,7)
    if getw(w,0) == 0:
        if code == 90:
            result = leaf_requires_grad(w,a,b)
        elif code == 71:
            if length(o)-at-32 < 32:
                terror(w,3)
            else:
                result = optimizer_create(s,w,a,b,c,end)
                if result != 0:
                    payload = optimizer_inventory(w,o,at+32,result)
        elif code == 72:
            result = optimizer_step(w,a)
        elif code == 73:
            result = optimizer_zero(w,a,b)
        elif code == 78:
            result = optimizer_drop(w,a)
        elif code == 79:
            if length(o)-at-32 < 32:
                terror(w,3)
            else:
                result = optimizer_add_group(s,w,a,b,end)
                if result != 0:
                    payload = optimizer_inventory(w,o,at+32,a)
        else:
            blob = optimizer_blob(w,a)
            if blob != 0:
                if code == 74:
                    payload = optimizer_group_inventory(w,o,at+32,blob,b)
                    if payload != 0:
                        result = getw(o,at+40)
                elif code == 75 or code == 76 or code == 77:
                    ep = optimizer_occurrence(w,blob,b,c)
                    if ep != 0:
                        if code == 75:
                            result = publish_tensor(w,getw(w,ep))
                        else:
                            key = optimizer_key(w,blob,ep)
                            if key != 0:
                                if code == 76:
                                    if length(o)-at-32 < 32:
                                        terror(w,3)
                                    else:
                                        payload = optimizer_state_inventory(w,o,at+32,key)
                                        result = getw(w,key+24)
                                elif getw(w,key+24) != 0:
                                    result = publish_tensor(w,getw(w,key+32))
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
