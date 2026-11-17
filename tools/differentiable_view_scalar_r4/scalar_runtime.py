def scalar_peek_resolved(w: mutbuf, token: u64) -> u64:
    if (token>>24)!=254:
        return token
    index=token&16777215
    table=getw(w,96)
    if index==0 or index>getw(w,104) or table==0:
        return 0
    return getw(w,table+8*(index-1))

def scalar_peek_public(w: mutbuf, token: u64) -> u64:
    if getw(w,0)!=0:
        return 0
    token=scalar_peek_resolved(w,token)
    row=so_tensor_live(w,token)
    if row==0:
        return 0
    node=so_uget(w,row+48)
    if node==0 or node>=getw(w,8):
        return 0
    p=dptr(w,node)
    if getw(w,p+304)==0 or getw(w,p+192)!=token or getw(w,p+200)!=1:
        return 0
    return node

def dtype_width(dtype: u64) -> u64:
    if dtype==0:
        return 4
    if dtype==1:
        return 1
    if dtype==2 or dtype==3:
        return 8
    return 0

def typed_descriptor(w: mutbuf, rank: u64, sizes: u64, requires: u64, allocate: u64, dtype: u64) -> u64:
    if dtype==3 and (rank!=0 or requires!=0):
        return terror(w,7)
    return scalar_previous_typed_descriptor(w,rank,sizes,requires,allocate,dtype)

def typed_from_wire(r: readbuf, w: mutbuf, dtype: u64, payload: u64, payload_start: u64) -> u64:
    if dtype>2:
        return terror(w,7)
    return scalar_previous_typed_from_wire(r,w,dtype,payload,payload_start)

def binary_backward_read(w: mutbuf, source: u64, index: u64) -> u64:
    value=tget(w,source,index)
    dtype=tensor_dtype(w,source)
    if dtype==3:
        return wrapped_f64_to_f32(value)
    return integer_as_f32(value,dtype)

def wrapped_scalar_mul(w: mutbuf, source: u64, scalar_bits: u64) -> u64:
    source=to_view_public(w,source)
    if valid_f32(w,source)==0:
        return 0
    p=dptr(w,source)
    original=typed_descriptor(w,0,2048,0,1,3)
    if original==0:
        return 0
    if typed_store(w,taddr(w,original,0),8,scalar_bits)==0:
        return 0
    promoted=typed_descriptor(w,0,2048,0,1,0)
    if promoted==0:
        return 0
    if typed_store(w,taddr(w,promoted,0),4,wrapped_f64_to_f32(tget(w,original,0)))==0:
        return 0
    result=typed_descriptor(w,getw(w,p),p+96,getw(w,p+32),1,0)
    if result==0:
        return 0
    if integer_pointwise_layout(w,result,source,promoted)==0:
        return 0
    index=0
    while index<getw(w,p+8):
        value=wrapped_mul_f32(tget(w,source,index),tget(w,promoted,0))
        put4(w,taddr(w,result,index),value)
        index=index+1
    return record(w,result,3,source,original)

def dtype_audit(w: mutbuf) -> u64:
    result=scalar_previous_dtype_audit(w)
    if result!=0:
        return result
    index=1
    while index<getw(w,8):
        p=dptr(w,index)
        if getw(w,p+304)!=0 and getw(w,p+576)==3:
            if getw(w,p)!=0 or getw(w,p+8)!=1 or getw(w,p+32)!=0 or getw(w,p+40)!=0 or getw(w,p+408)!=0 or getw(w,p+384)!=0 or getw(w,p+640)!=0:
                return 9
        index=index+1
    return 0

def to_copy_tensor(w: mutbuf, source: u64, target_dtype: u64, memory_format: u64) -> u64:
    if valid(w,source)==0:
        return 0
    if tensor_dtype(w,source)==3 or target_dtype>2:
        return terror(w,7)
    return scalar_previous_to_copy_tensor(w,source,target_dtype,memory_format)

def transpose(w: mutbuf, source: u64, first_axis: u64, second_axis: u64) -> u64:
    if valid(w,source)==0:
        return 0
    if to_view_refresh(w,source)==0:
        return 0
    if tensor_dtype(w,source)==3:
        return terror(w,7)
    result=scalar_previous_transpose(w,source,first_axis,second_axis)
    if result==0:
        return 0
    q=dptr(w,result)
    if getw(w,q+640)==0:
        if to_view_root_retain(w,result,source)==0:
            return 0
    return result

def command_argument(w: mutbuf, operation: u64, position: u64, argument: u64) -> u64:
    value=scalar_previous_command_argument(w,operation,position,argument)
    mask=0
    if position==1:
        mask=1654574268384
    if position==2:
        mask=5317427260
    if position==3:
        mask=335577092
    if operation<64 and (mask&(1<<operation))!=0 and value!=0 and getw(w,0)==0:
        if tensor_dtype(w,value)==3 and operation!=13:
            return terror(w,7)
    return value

def typed_command(r: readbuf, out: mutbuf, w: mutbuf, journal: mutbuf, command: u64, response: u64, payload_start: u64) -> u64:
    operation=getr(r,command)
    if operation>=99 and operation<=102:
        first=getw(w,8)
        source=scalar_peek_public(w,getr(r,command+8))
        if source!=0 and tensor_dtype(w,source)==3:
            terror(w,7)
            return shape_command_finish(w,out,response,operation,first,0)
    return scalar_previous_typed_command(r,out,w,journal,command,response,payload_start)

def to_command(r: readbuf, out: mutbuf, w: mutbuf, command: u64, response: u64, payload_start: u64) -> u64:
    operation=getr(r,command)
    first=getw(w,8)
    if operation==114:
        result=0
        if getr(r,command+24)!=0 or getr(r,command+32)!=0 or getr(r,command+40)!=0 or getr(r,command+48)!=0 or getr(r,command+56)!=0:
            terror(w,7)
        if getw(w,0)==0:
            source=resolve_handle(w,getr(r,command+8))
            if getw(w,0)==0:
                result=wrapped_scalar_mul(w,source,getr(r,command+16))
        return shape_command_finish(w,out,response,operation,first,result)
    if operation==108 or operation==109 or operation==111 or operation==112 or operation==113:
        token=scalar_peek_resolved(w,getr(r,command+8))
        source=scalar_peek_public(w,token)
        if source!=0 and tensor_dtype(w,source)==3:
            terror(w,7)
            return shape_command_finish(w,out,response,operation,first,0)
    if operation==110 and getr(r,command+8)>2:
        terror(w,7)
        return shape_command_finish(w,out,response,operation,first,0)
    if operation==108 or operation==109:
        if getr(r,command+16)>2:
            terror(w,7)
            return shape_command_finish(w,out,response,operation,first,0)
    return scalar_previous_to_command(r,out,w,command,response,payload_start)

def data_alias(w: mutbuf, source: u64) -> u64:
    if valid(w,source)==0:
        return 0
    if tensor_dtype(w,source)==3:
        return terror(w,7)
    return scalar_previous_data_alias(w,source)

def save_public(w: mutbuf, source: u64) -> u64:
    if valid(w,source)==0:
        return 0
    if tensor_dtype(w,source)==3:
        return terror(w,7)
    return scalar_previous_save_public(w,source)

def rng_strided_view(r: readbuf, w: mutbuf, token: u64, payload: u64, payload_start: u64) -> u64:
    source=public_node(w,token)
    if source==0:
        return 0
    if tensor_dtype(w,source)==3:
        return terror(w,7)
    return scalar_previous_rng_strided_view(r,w,token,payload,payload_start)

def back_binary(w: mutbuf, source: u64) -> u64:
    p=dptr(w,source)
    if backward_getw(w,p+40)==3:
        other=backward_getw(w,p+56)
        if other!=0 and tensor_dtype(w,other)==3:
            return wrapped_scalar_back_binary(w,source)
    return scalar_previous_back_binary(w,source)

def incoming_describe(w: mutbuf, source: u64, parent: u64, side: u64) -> u64:
    described=scalar_previous_incoming_describe(w,source,parent,side)
    if described==0 or side!=0:
        return described
    p=dptr(w,source)
    other=getw(w,p+56)
    if getw(w,p+40)!=3 or other==0:
        return described
    if tensor_dtype(w,other)!=3:
        return described
    if incoming_same_shape(w,source,parent)==0:
        return terror(w,9)
    memory_fill(w,3808,88,0)
    putw(w,3840,parent)
    putw(w,3848,2)
    incoming_contiguous_layout(w,parent,3856)
    order=retained_axis_order(w,source,other)
    inverted=1
    axis_position=0
    while axis_position<getw(w,p):
        axis=(order>>(8*axis_position))&255
        if axis!=getw(w,p)-1-axis_position:
            inverted=0
        axis_position=axis_position+1
    if inverted!=0:
        return 1
    stride=1
    axis_position=0
    while axis_position<getw(w,p):
        axis=(order>>(8*axis_position))&255
        putw(w,3856+8*axis,stride)
        stride=stride*getw(w,p+96+8*axis)
        axis_position=axis_position+1
    return 1

def incoming_mark_preserve_targets(w: mutbuf, root: u64, epoch: u64) -> u64:
    if scalar_previous_incoming_mark_preserve_targets(w,root,epoch)==0:
        return 0
    source=1
    while source<=root:
        p=dptr(w,source)
        if getw(w,p+304)!=0 and getw(w,p+168)==epoch and getw(w,p+40)==3 and getw(w,p+8)!=0:
            other=getw(w,p+56)
            if other!=0 and tensor_dtype(w,other)==3:
                parent=gradient_parent_at(w,source,0)
                if parent!=0:
                    q=dptr(w,parent)
                    if getw(w,q+304)==0 or getw(w,q+168)!=epoch:
                        return terror(w,9)
                    putw(w,q+544,1)
        source=source+1
    return 1
