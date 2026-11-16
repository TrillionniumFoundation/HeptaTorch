def tensor_dtype(w:mutbuf, t:u64) -> u64:
    return getw(w,dptr(w,t)+576)

def dtype_width(code:u64) -> u64:
    if code == 0:
        return 4
    if code == 1:
        return 1
    return 0

def valid_f32(w:mutbuf, t:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    if tensor_dtype(w,t) != 0:
        return terror(w,7)
    return 1

def public_f32(w:mutbuf, id:u64) -> u64:
    t = public_node(w,id)
    if t == 0:
        return 0
    if tensor_dtype(w,t) != 0:
        return terror(w,7)
    return t

def typed_load(w:mutbuf, address:u64, width:u64) -> u64:
    address = object_address(w,address)
    value = 0
    i = 0
    while i < width:
        value = value | (byte(w,address+i) << (i*8))
        i = i+1
    return value

def typed_store(w:mutbuf, address:u64, width:u64, value:u64) -> u64:
    address = object_address(w,address)
    i = 0
    while i < width:
        store(w,address+i,value >> (i*8))
        i = i+1
    return 1

def typed_copy_from_wire(w:mutbuf, s:readbuf, address:u64, data:u64, size:u64) -> u64:
    i = 0
    while i < size:
        store(w,address+i,byte(s,data+i))
        i = i+1
    return 1

def taddr(w:mutbuf, t:u64, index:u64) -> u64:
    p = dptr(w,t)
    axis = getw(w,p)
    offset = 0
    while axis > 0:
        axis = axis-1
        dim = getw(w,p+96+axis*8)
        coordinate = index % dim
        index = index // dim
        offset = offset+coordinate*getw(w,p+128+axis*8)
    return getw(w,p+16)+offset*dtype_width(getw(w,p+576))

def tget(w:mutbuf, t:u64, index:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    if index >= getw(w,p+8):
        return terror(w,7)
    sp = so_storage_live(w,getw(w,p+216))
    if sp == 0:
        return terror(w,1)
    width = dtype_width(getw(w,p+576))
    if width == 0:
        return terror(w,9)
    address = taddr(w,t,index)
    origin = 9223372036854775808+so_uget(w,sp+16)
    offset = address-origin
    if offset > so_uget(w,sp+24) or width > so_uget(w,sp+24)-offset:
        return terror(w,2)
    return typed_load(w,address,width)

def tset(w:mutbuf, t:u64, index:u64, bits:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    code = getw(w,p+576)
    width = dtype_width(code)
    if width == 0:
        return terror(w,9)
    if index >= getw(w,p+8) or bits > 4294967295:
        return terror(w,7)
    if code == 1 and bits > 1:
        return terror(w,7)
    if version_can_bump(w,t,1) == 0:
        return 0
    typed_store(w,taddr(w,t,index),width,bits)
    return bump_version(w,t)

def descriptor(w:mutbuf, rank:u64, shape:u64, requires:u64, own:u64) -> u64:
    return typed_descriptor(w,rank,shape,requires,own,0)

def typed_clone(w:mutbuf, a:u64) -> u64:
    if valid(w,a) == 0:
        return 0
    p = dptr(w,a)
    code = getw(w,p+576)
    t = typed_descriptor(w,getw(w,p),p+96,0,1,code)
    if t == 0:
        return 0
    if dist_dense_layout(w,t,a,1) == 0:
        return 0
    width = dtype_width(code)
    i = 0
    while i < getw(w,p+8):
        typed_store(w,taddr(w,t,i),width,tget(w,a,i))
        i = i+1
    return t

def clone(w:mutbuf, a:u64) -> u64:
    if valid(w,a) == 0:
        return 0
    if tensor_dtype(w,a) == 0:
        return dtype_original_clone(w,a)
    return typed_clone(w,a)

def inspect_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64) -> u64:
    if getr(s,cp) != 13:
        t = public_node(w,getr(s,cp+8))
        if t != 0 and tensor_dtype(w,t) != 0:
            terror(w,7)
    return dtype_original_inspect_command(s,o,w,cp,at)

def saved_logical(w:mutbuf, id:u64, index:u64) -> u64:
    p = public_saved(w,id)
    if p == 0:
        return 0
    layout = private_hold(w,so_uget(w,p+56),128)
    if getw(w,layout+16) != 0:
        return terror(w,7)
    return dtype_original_saved_logical(w,id,index)

def weak_create(w:mutbuf, t:u64) -> u64:
    id = dtype_original_weak_create(w,t)
    if id != 0:
        so_uput(w,so_slot_lookup(w,4,id)+24,tensor_dtype(w,t))
    return id

def weak_lock(w:mutbuf, id:u64) -> u64:
    hold = dtype_original_weak_lock(w,id)
    if hold != 0:
        weak = so_slot_lookup(w,4,id)
        so_uput(w,so_slot_lookup(w,6,hold)+32,so_uget(w,weak+24))
    return hold

def object_command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64) -> u64:
    code = getr(s,cp)
    id = resolve_handle(w,getr(s,cp+8))
    if code == 49:
        saved = public_saved(w,id)
        if saved != 0 and graph_saved_check(w,id) != 0:
            layout = private_hold(w,so_uget(w,saved+56),128)
            if getw(w,layout+16) != 0:
                terror(w,7)
    if code == 50:
        hold = public_hold(w,id)
        if hold != 0 and so_uget(w,hold+32) != 0:
            terror(w,7)
    if getw(w,0) != 0:
        putw(o,at,code)
        putw(o,at+8,getw(w,0))
        putw(o,at+16,0)
        putw(o,at+24,0)
        return at+32
    return dtype_original_object_command(s,o,w,cp,at)

def leaf_requires_grad(w:mutbuf, id:u64, requires:u64) -> u64:
    t = public_node(w,id)
    if t == 0:
        return 0
    if requires == 1 and tensor_dtype(w,t) != 0:
        return terror(w,5)
    return dtype_original_leaf_requires_grad(w,id,requires)

def dtype_mask_retain(w:mutbuf, t:u64, mask:u64) -> u64:
    p = dptr(w,t)
    q = dptr(w,mask)
    if getw(w,p+584) != 0 or getw(w,p+592) != 0:
        return terror(w,9)
    if getw(w,q+600) >= 4294967295:
        return terror(w,8)
    if node_retain(w,mask) == 0:
        return 0
    putw(w,q+600,getw(w,q+600)+1)
    putw(w,p+584,mask)
    putw(w,p+592,getw(w,q+320))
    return 1

def dtype_mask_node(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    mask = getw(w,p+584)
    q = private_tensor(w,mask,getw(w,p+592))
    if q == 0 or getw(w,q+600) == 0:
        return terror(w,9)
    return mask

def dtype_mask_release(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    mask = getw(w,p+584)
    if mask == 0:
        if getw(w,p+592) != 0:
            return terror(w,9)
        return 1
    q = private_tensor(w,mask,getw(w,p+592))
    if q == 0 or getw(w,q+600) == 0:
        return terror(w,9)
    putw(w,p+584,0)
    putw(w,p+592,0)
    putw(w,q+600,getw(w,q+600)-1)
    return node_release(w,mask)

def gradient_gc_enqueue(w:mutbuf, t:u64) -> u64:
    if t != 0 and getw(w,dptr(w,t)+600) != 0:
        return 1
    return dtype_original_gradient_gc_enqueue(w,t)

def tensor_row_release(w:mutbuf, t:u64) -> u64:
    if getw(w,dptr(w,t)+600) != 0:
        return 1
    return dtype_original_tensor_row_release(w,t)

def graph_node_release(w:mutbuf, t:u64) -> u64:
    if getw(w,dptr(w,t)+600) != 0:
        return terror(w,9)
    if dtype_mask_release(w,t) == 0:
        return 0
    return dtype_original_graph_node_release(w,t)

def release_backward_saved(w:mutbuf, t:u64) -> u64:
    if dtype_original_release_backward_saved(w,t) == 0:
        return 0
    return dtype_mask_release(w,t)

def dtype_audit(w:mutbuf) -> u64:
    status = 0
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if getw(w,p+616) != 0 or getw(w,p+624) != 0 or getw(w,p+632) != 0:
            status = 9
        i = i+1
    i = 1
    while i < getw(w,8) and status == 0:
        p = dptr(w,i)
        if getw(w,p+304) != 0:
            code = getw(w,p+576)
            if code > 1 or getw(w,p+608) != 0:
                status = 9
            if code == 1 and (getw(w,p+32) != 0 or getw(w,p+384) != 0):
                status = 9
            mask = getw(w,p+584)
            if mask != 0:
                q = private_tensor(w,mask,getw(w,p+592))
                if q == 0 or getw(w,p+40) != 42 or getw(w,p+288) == 0:
                    status = 9
                else:
                    saved = so_saved_live(w,getw(w,p+288))
                    if saved == 0:
                        status = 9
                    else:
                        if so_uget(w,saved+16) != getw(w,q+216) or so_uget(w,saved+24) != getw(w,q+224):
                            status = 9
                    putw(w,q+616,getw(w,q+616)+1)
            elif getw(w,p+592) != 0:
                status = 9
        elif getw(w,p+584) != 0 or getw(w,p+592) != 0 or getw(w,p+600) != 0:
            status = 9
        i = i+1
    i = 1
    while i < getw(w,8):
        p = dptr(w,i)
        if getw(w,p+600) != getw(w,p+616):
            status = 9
        putw(w,p+616,0)
        i = i+1
    return status

def gradient_audit(w:mutbuf) -> u64:
    if dtype_audit(w) != 0:
        return 9
    return dtype_original_gradient_audit(w)

def typed_fill(w:mutbuf, t:u64, bits:u64, no_grad:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    p = dptr(w,t)
    code = getw(w,p+576)
    width = dtype_width(code)
    if width == 0 or no_grad > 1 or bits > 4294967295:
        return terror(w,7)
    if code == 1 and bits > 1:
        return terror(w,7)
    if no_grad == 0 and getw(w,p+32) != 0:
        return terror(w,5)
    if version_can_bump(w,t,1) == 0:
        return 0
    i = 0
    while i < getw(w,p+8):
        typed_store(w,taddr(w,t,i),width,bits)
        i = i+1
    return bump_version(w,t)

def typed_from_wire(s:readbuf, w:mutbuf, dtype:u64, config:u64, end:u64) -> u64:
    width = dtype_width(dtype)
    if width == 0:
        return terror(w,7)
    if config < end or config > length(s) or 64 > length(s)-config or (config & 7) != 0:
        return terror(w,100)
    rank = getr(s,config)
    requires = getr(s,config+8)
    if rank > 4 or requires > 1:
        return terror(w,2)
    if dtype == 1 and requires != 0:
        return terror(w,5)
    axis = 0
    n = 1
    while axis < 4:
        dim = getr(s,config+16+axis*8)
        if axis >= rank:
            if dim != 0:
                return terror(w,2)
        else:
            if dim > 262144:
                return terror(w,3)
            if dim == 0:
                n = 0
        putw(w,2048+axis*8,dim)
        axis = axis+1
    if n != 0:
        axis = 0
        while axis < rank:
            n = n*getw(w,2048+axis*8)
            if n > 16777216:
                return terror(w,3)
            axis = axis+1
    data = getr(s,config+48)
    size = getr(s,config+56)
    if size != n*width or data < end or data > length(s) or size > length(s)-data:
        return terror(w,100)
    if dtype == 1:
        i = 0
        while i < size:
            if byte(s,data+i) > 1:
                return terror(w,7)
            i = i+1
    t = typed_descriptor(w,rank,2048,requires,1,dtype)
    if t == 0:
        return 0
    address = object_address(w,getw(w,dptr(w,t)+16))
    typed_copy_from_wire(w,s,address,data,size)
    return t

def typed_command(s:readbuf, o:mutbuf, w:mutbuf, scratch:mutbuf, cp:u64, at:u64, end:u64) -> u64:
    code = getr(s,cp)
    a = getr(s,cp+8)
    b = getr(s,cp+16)
    c = getr(s,cp+24)
    d = getr(s,cp+32)
    result = 0
    payload = 0
    first = getw(w,8)
    if code < 96 or code > 100:
        terror(w,4)
    if getr(s,cp+40) != 0 or getr(s,cp+48) != 0 or getr(s,cp+56) != 0:
        terror(w,7)
    if code != 99 and d != 0:
        terror(w,7)
    if (code == 96 or code == 97) and c != 0:
        terror(w,7)
    if getw(w,0) == 0:
        if code == 97:
            result = typed_from_wire(s,w,a,b,end)
            if getw(w,0) == 0:
                result = publish_tensor(w,result)
        else:
            t = public_node(w,a)
            if t != 0:
                if code == 96:
                    if b == 0:
                        if at+32 > length(o) or 32 > length(o)-at-32:
                            terror(w,3)
                        else:
                            p = dptr(w,t)
                            result = getw(w,p+576)
                            putw(o,at+32,result)
                            putw(o,at+40,dtype_width(result))
                            putw(o,at+48,getw(w,p+320))
                            putw(o,at+56,choose(getw(w,p+40) == 0,1,0))
                            payload = 32
                    elif b == 1:
                        result = native_dropout_saved_mask(w,t)
                        if getw(w,0) == 0:
                            result = publish_tensor(w,result)
                    else:
                        terror(w,7)
                if code == 98:
                    p = dptr(w,t)
                    n = getw(w,p+8)
                    width = dtype_width(getw(w,p+576))
                    if b > n or c > n-b:
                        terror(w,2)
                    elif c > (length(o)-at-32)//width:
                        terror(w,3)
                    else:
                        j = 0
                        while j < c:
                            typed_store(o,at+32+j*width,width,tget(w,t,b+j))
                            j = j+1
                        result = c
                        payload = c*width
                if code == 99:
                    if length(o)-at-32 < 8:
                        terror(w,3)
                    else:
                        result = native_dropout(w,scratch,t,b,c,d,at+32,o)
                        if getw(w,0) == 0:
                            result = publish_tensor(w,result)
                            payload = 8
                if code == 100:
                    result = typed_fill(w,t,b,c)
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
