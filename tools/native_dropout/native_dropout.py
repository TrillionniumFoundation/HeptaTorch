def native_dropout_scale(complement:u64) -> u64:
    if complement == 0:
        return 0
    significand = (complement & 4503599627370495) | 4503599627370496
    exponent = complement >> 52
    remainder = 0
    quotient = 0
    bit_index = 106
    while bit_index > 0:
        bit_index = bit_index-1
        remainder = remainder << 1
        if bit_index == 105:
            remainder = remainder | 1
        quotient = quotient << 1
        if remainder >= significand:
            remainder = remainder-significand
            quotient = quotient | 1
    doubled = remainder << 1
    if doubled > significand or (doubled == significand and (quotient & 1) != 0):
        quotient = quotient+1
    result_exponent = 2045-exponent
    if quotient == 9007199254740992:
        quotient = quotient >> 1
        result_exponent = result_exponent+1
    reciprocal = (result_exponent << 52) | (quotient & 4503599627370495)
    return dist_probability_to32(reciprocal)

def typed_iteration_address(w:mutbuf, p:u64, order:u64, index:u64) -> u64:
    offset = 0
    i = 0
    while i < getw(w,p):
        axis = (order >> (i*8)) & 255
        size = getw(w,p+96+axis*8)
        coordinate = index % size
        index = index // size
        offset = offset+coordinate*getw(w,p+128+axis*8)
        i = i+1
    return getw(w,p+16)+offset*dtype_width(getw(w,p+576))

def native_dropout_fill_mask(w:mutbuf, scratch:mutbuf, mask:u64, data:u64, probability:u64, training:u64) -> u64:
    p = dptr(w,mask)
    if training == 0:
        memory_fill(w,getw(w,p+16),getw(w,p+8),1)
        return 1
    if rng_load_scratch(w,scratch,data) == 0:
        return 0
    order = rng_axis_order(w,p)
    i = 0
    while i < getw(w,p+8):
        unit = rng_unit64(rng_next64(scratch))
        value = choose(unit < probability,1,0)
        typed_store(w,typed_iteration_address(w,p,order,i),1,value)
        i = i+1
    return rng_save_scratch(w,scratch,data)

def native_dropout_write(w:mutbuf, output:u64, input:u64, mask:u64, scale:u64, training:u64) -> u64:
    p = dptr(w,output)
    i = 0
    while i < getw(w,p+8):
        value = tget(w,input,i)
        if training != 0:
            coefficient = choose(tget(w,mask,i) != 0,1065353216,0)
            value = mulf(mulf(value,coefficient),scale)
        put4(w,taddr(w,output,i),value)
        i = i+1
    return 1

def native_dropout_record(w:mutbuf, output:u64, input:u64, mask:u64, scale:u64) -> u64:
    if record(w,output,42,input,0) == 0:
        return 0
    p = dptr(w,output)
    if getw(w,p+32) != 0:
        saved = saved_create(w,mask,0)
        if saved == 0:
            return 0
        putw(w,p+288,saved)
        putw(w,p+176,scale)
        if dtype_mask_retain(w,output,mask) == 0:
            return 0
    return output

def native_dropout(w:mutbuf, scratch:mutbuf, input:u64, generator:u64, probability:u64, training:u64, mask_out:u64, o:mutbuf) -> u64:
    if valid_f32(w,input) == 0:
        return 0
    data = rng_object(w,generator)
    if data == 0:
        return 0
    if training > 2:
        return terror(w,7)
    p = dptr(w,input)
    n = getw(w,p+8)
    if n == 0:
        if getw(w,p+32) != 0:
            return terror(w,5)
        mask = typed_descriptor(w,getw(w,p),p+96,0,1,0)
        if mask == 0:
            return 0
        dist_dense_layout(w,mask,input,1)
        putw(o,mask_out,publish_tensor(w,mask))
        return input
    if n > 1000000:
        return terror(w,3)
    complement = 0
    scale = 1065353216
    if training != 0:
        if dist_prob64_valid(probability) == 0:
            return terror(w,7)
        complement = dist_complement64(probability)
        scale = native_dropout_scale(complement)
    mask = typed_descriptor(w,getw(w,p),p+96,0,1,1)
    if mask == 0:
        return 0
    if dist_dense_layout(w,mask,input,1) == 0:
        return 0
    output = descriptor(w,getw(w,p),p+96,getw(w,p+32),1)
    if output == 0:
        return 0
    if dist_dense_layout(w,output,input,1) == 0:
        return 0
    if native_dropout_fill_mask(w,scratch,mask,data,complement,training) == 0:
        return 0
    native_dropout_write(w,output,input,mask,scale,training)
    backward_scale = scale
    if training != 1:
        backward_scale = 1065353216
    if native_dropout_record(w,output,input,mask,backward_scale) == 0:
        return 0
    putw(o,mask_out,publish_tensor(w,mask))
    return output

def native_dropout_saved_mask(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    if getw(w,p+40) != 42:
        return terror(w,5)
    if getw(w,p+328) != 0:
        return terror(w,10)
    if graph_saved_check(w,getw(w,p+288)) == 0:
        return 0
    return dtype_mask_node(w,t)

def native_dropout_backward(w:mutbuf, t:u64) -> u64:
    p = dptr(w,t)
    parent = gradient_parent_at(w,t,0)
    if parent == 0:
        return 1
    mask = dtype_mask_node(w,t)
    if mask == 0:
        return 0
    scale = getw(w,p+176)
    i = 0
    while i < getw(w,p+8):
        grad = get4(w,getw(w,p+160)+i*4)
        coefficient = choose(tget(w,mask,i) != 0,1065353216,0)
        value = mulf(mulf(grad,coefficient),scale)
        gadd(w,parent,i,value)
        i = i+1
    return 1

def partial_apply_node(w:mutbuf, t:u64) -> u64:
    if getw(w,dptr(w,t)+40) == 42:
        return native_dropout_backward(w,t)
    return dtype_original_partial_apply_node(w,t)

def native_dropout_compare_axes(w:mutbuf, p:u64, q:u64, left:u64, right:u64) -> u64:
    operand = 0
    while operand < 2:
        layout = p+496
        if operand == 1:
            layout = q+128
        ls = getw(w,layout+left*8) << 2
        rs = getw(w,layout+right*8) << 2
        if ls != 0 and rs != 0:
            ordered_left = ls ^ 9223372036854775808
            ordered_right = rs ^ 9223372036854775808
            if ordered_left < ordered_right:
                return 1
            if ordered_left > ordered_right:
                return 2
            if getw(w,p+96+left*8) > getw(w,p+96+right*8):
                return 2
        operand = operand+1
    return 0

def native_dropout_axis_order(w:mutbuf, p:u64, q:u64) -> u64:
    rank = getw(w,p)
    perm = 0
    i = 0
    while i < rank:
        perm = perm | ((rank-i-1) << (i*8))
        i = i+1
    i = 1
    while i < rank:
        anchor = i
        j = i
        done = 0
        while j > 0 and done == 0:
            j = j-1
            left = (perm >> (j*8)) & 255
            right = (perm >> (anchor*8)) & 255
            comparison = native_dropout_compare_axes(w,p,q,left,right)
            if comparison == 2:
                difference = left ^ right
                perm = perm ^ (difference << (j*8)) ^ (difference << (anchor*8))
                anchor = j
            if comparison == 1:
                done = 1
        i = i+1
    return perm

def native_dropout_grad_layout(w:mutbuf, t:u64, dest:u64) -> u64:
    p = dptr(w,t)
    mask = dtype_mask_node(w,t)
    if mask == 0:
        return 0
    q = dptr(w,mask)
    if incoming_dense(w,t,p+496) != 0 and is_dense(w,mask) != 0:
        same = 1
        j = 0
        while j < getw(w,p):
            if getw(w,p+496+j*8) != getw(w,q+128+j*8):
                same = 0
            j = j+1
        if same != 0:
            j = 0
            while j < 4:
                putw(w,dest+j*8,getw(w,p+496+j*8))
                j = j+1
            return 1
    if incoming_contiguous(w,t,p+496) != 0 and is_contiguous(w,mask) != 0:
        return incoming_contiguous_layout(w,t,dest)
    order = native_dropout_axis_order(w,p,q)
    stride = 1
    j = 0
    while j < getw(w,p):
        axis = (order >> (j*8)) & 255
        putw(w,dest+axis*8,stride)
        stride = stride*getw(w,p+96+axis*8)
        j = j+1
    return 1
