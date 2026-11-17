def shape_clone_contiguous(w: mutbuf, source: u64) -> u64:
    p=dptr(w,source)
    dtype=getw(w,p+576)
    width=dtype_width(dtype)
    result=typed_descriptor(w,getw(w,p),p+96,getw(w,p+32),1,dtype)
    if result==0:
        return 0
    index=0
    while index<getw(w,p+8):
        value=to_convert_value(tget(w,source,index),dtype,dtype)
        if typed_store(w,taddr(w,result,index),width,value)==0:
            return 0
        index=index+1
    return record(w,result,17,source,0)

def shape_fresh_alias_version(w: mutbuf, result: u64) -> u64:
    if storage_capacity(w,3,0)==0:
        return 0
    p=dptr(w,result)
    old=getw(w,p+224)
    fresh=so_slot_new(w,3)
    if fresh==0:
        return terror(w,9)
    row=so_slot_lookup(w,3,fresh)
    so_uput(w,row+24,1)
    if so_version_release(w,old)!=0:
        return terror(w,9)
    tensor=so_slot_lookup(w,1,getw(w,p+192))
    so_uput(w,tensor+24,fresh)
    putw(w,p+224,fresh)
    return 1

def shape_alias(w: mutbuf, source: u64, rank: u64, sizes: u64, strides: u64, operation: u64) -> u64:
    p=dptr(w,source)
    result=typed_descriptor(w,rank,sizes,getw(w,p+32),0,getw(w,p+576))
    if result==0:
        return 0
    q=dptr(w,result)
    putw(w,q+16,getw(w,p+16))
    putw(w,q+64,getw(w,p+64))
    axis=0
    while axis<4:
        putw(w,q+128+8*axis,getw(w,strides+8*axis))
        axis=axis+1
    if operation!=16:
        if bind_alias(w,result,source)==0:
            return 0
    if operation==46:
        if shape_fresh_alias_version(w,result)==0:
            return 0
    if record(w,result,operation,source,0)==0:
        return 0
    if operation!=46 and getw(w,q+640)==0:
        if to_view_root_retain(w,result,source)==0:
            return 0
    return result

def shape_transform(w: mutbuf, source: u64, rank: u64, sizes: u64, reshape: u64) -> u64:
    source=to_view_public(w,source)
    if source==0:
        return 0
    p=dptr(w,source)
    if getw(w,p+408)!=0 and (getw(w,p+664)&1)==0:
        return terror(w,5)
    if shape_infer_dimensions(w,source,rank,sizes)==0:
        return 0
    strides=2080
    compatible=shape_compute_strides(w,source,rank,sizes,strides)
    if compatible!=0:
        operation=16
        if reshape!=0 and is_contiguous(w,source)==0:
            operation=45
        return shape_alias(w,source,rank,sizes,strides,operation)
    if reshape==0:
        return terror(w,2)
    temporary=shape_clone_contiguous(w,source)
    if temporary==0:
        return 0
    if shape_compute_strides(w,temporary,rank,sizes,strides)==0:
        return terror(w,9)
    return shape_alias(w,temporary,rank,sizes,strides,46)

def shape_command_finish(w: mutbuf, out: mutbuf, response: u64, operation: u64, first: u64, result: u64) -> u64:
    if getw(w,0)==0:
        result=publish_tensor(w,result)
    if getw(w,0)==0:
        finish_command(w,first)
    status=getw(w,0)
    if status!=0:
        result=0
    putw(out,response,operation)
    putw(out,response+8,status)
    putw(out,response+16,result)
    putw(out,response+24,0)
    return response+32

def to_command(r: readbuf, out: mutbuf, w: mutbuf, command: u64, response: u64, payload_start: u64) -> u64:
    operation=getr(r,command)
    if operation!=112 and operation!=113:
        return shape_previous_to_command(r,out,w,command,response,payload_start)
    first=getw(w,8)
    result=0
    if getr(r,command+24)!=40 or getr(r,command+32)!=0 or getr(r,command+40)!=0 or getr(r,command+48)!=0 or getr(r,command+56)!=0:
        terror(w,7)
    payload=getr(r,command+16)
    if getw(w,0)==0:
        if payload<payload_start or payload>length(r) or 40>length(r)-payload or (payload&7)!=0:
            terror(w,100)
    if getw(w,0)==0:
        rank=getr(r,payload)
        if rank>4:
            terror(w,2)
        else:
            axis=0
            while axis<4:
                value=getr(r,payload+8+8*axis)
                if axis>=rank and value!=0:
                    terror(w,7)
                putw(w,2048+8*axis,value)
                axis=axis+1
            if getw(w,0)==0:
                source=resolve_handle(w,getr(r,command+8))
                if getw(w,0)==0:
                    reshape=0
                    if operation==113:
                        reshape=1
                    result=shape_transform(w,source,rank,2048,reshape)
    return shape_command_finish(w,out,response,operation,first,result)

def partial_apply_node(w: mutbuf, ident: u64) -> u64:
    operation=getw(w,dptr(w,ident)+40)
    if operation==45 or operation==46:
        return to_backward(w,ident)
    return shape_previous_partial_apply_node(w,ident)

def gradient_direct(w: mutbuf, child: u64, parent: u64) -> u64:
    operation=getw(w,dptr(w,child)+40)
    if operation==45 or operation==46:
        return 1
    return shape_previous_gradient_direct(w,child,parent)

def incoming_describe(w: mutbuf, child: u64, parent: u64, edge: u64) -> u64:
    operation=getw(w,dptr(w,child)+40)
    if operation!=45 and operation!=46:
        return shape_previous_incoming_describe(w,child,parent,edge)
    memory_fill(w,3808,88,0)
    putw(w,3840,parent)
    putw(w,3848,2)
    incoming_contiguous_layout(w,parent,3856)
    if incoming_describe_alias(w,child,2,parent)==0:
        return 0
    if incoming_reshape_layout(w,child,parent,3856)!=0:
        return 1
    putw(w,3808,0)
    putw(w,3816,0)
    putw(w,3824,0)
    putw(w,3832,0)
    return incoming_contiguous_layout(w,parent,3856)
