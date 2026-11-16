def integer_dropout_nonempty(w:mutbuf, scratch:mutbuf, input:u64, generator:u64, probability:u64, training:u64, o:mutbuf, at:u64) -> u64:
    p = dptr(w,input)
    n = getw(w,p+8)
    if n > 1000000:
        return terror(w,3)
    data = rng_object(w,generator)
    if data == 0:
        return 0
    complement = 0
    if training != 0:
        if dist_prob64_valid(probability) == 0:
            return terror(w,7)
        complement = dist_complement64(probability)
    mask = typed_descriptor(w,getw(w,p),p+96,0,1,1)
    if mask == 0 or dist_dense_layout(w,mask,input,1) == 0:
        return 0
    output = typed_descriptor(w,getw(w,p),p+96,0,1,2)
    if output == 0:
        return 0
    if training == 0:
        if dist_dense_layout(w,output,input,1) == 0:
            return 0
    else:
        if integer_pointwise_layout(w,output,input,mask) == 0:
            return 0
    if native_dropout_fill_mask(w,scratch,mask,data,complement,training) == 0:
        return 0
    i = 0
    while i < n:
        if training == 0:
            typed_store(w,taddr(w,output,i),8,tget(w,input,i))
        else:
            integer_binary_element(w,3,output,input,mask,i)
        i = i+1
    if training != 0:
        return 12
    putw(o,at+16,publish_tensor(w,output))
    putw(o,at+40,publish_tensor(w,mask))
    return 0

def integer_native_dropout(w:mutbuf, scratch:mutbuf, input:u64, generator:u64, probability:u64, training:u64, o:mutbuf, at:u64) -> u64:
    if valid(w,input) == 0:
        return 0
    if tensor_dtype(w,input) != 2 or training > 2:
        return terror(w,7)
    if rng_object(w,generator) == 0:
        return 0
    p = dptr(w,input)
    if getw(w,p+8) != 0:
        return integer_dropout_nonempty(w,scratch,input,generator,probability,training,o,at)
    mask = typed_descriptor(w,getw(w,p),p+96,0,1,2)
    if mask == 0 or dist_dense_layout(w,mask,input,1) == 0:
        return 0
    putw(o,at+16,publish_tensor(w,input))
    putw(o,at+40,publish_tensor(w,mask))
    return 0

def integer_dropout_command(s:readbuf, o:mutbuf, w:mutbuf, scratch:mutbuf, cp:u64, at:u64) -> u64:
    first = getw(w,8)
    error = 0
    payload = 0
    putw(o,at,105)
    putw(o,at+8,0)
    putw(o,at+16,0)
    putw(o,at+24,0)
    if getr(s,cp+40) != 0 or getr(s,cp+48) != 0 or getr(s,cp+56) != 0:
        terror(w,7)
    if at+32 > length(o) or 16 > length(o)-at-32:
        terror(w,3)
    if getw(w,0) == 0:
        putw(o,at+32,0)
        putw(o,at+40,0)
        input = public_node(w,resolve_handle(w,getr(s,cp+8)))
        if input != 0:
            error = integer_native_dropout(w,scratch,input,resolve_handle(w,getr(s,cp+16)),getr(s,cp+24),getr(s,cp+32),o,at)
        if getw(w,0) == 0:
            if finish_command(w,first) != 0:
                putw(o,at+32,error)
                payload = 16
    status = getw(w,0)
    if status != 0:
        putw(o,at+16,0)
        payload = 0
    putw(o,at+8,status)
    putw(o,at+24,payload)
    return at+32+payload
