def dview_geometry_end(w: mutbuf, p: u64) -> u64:
    if getw(w,p+8)==0:
        return getw(w,p+232)
    end=getw(w,p+232)+4
    axis=0
    while axis<getw(w,p):
        size=getw(w,p+96+8*axis)
        if size>1:
            end=end+4*(size-1)*getw(w,p+128+8*axis)
        axis=axis+1
    return end

def dview_shared_offset(w: mutbuf, p: u64, q: u64) -> u64:
    left=getw(w,p+232)
    right=getw(w,q+232)
    if left<right:
        return left
    return right

def dview_span_bytes(w: mutbuf, p: u64, q: u64) -> u64:
    end=dview_geometry_end(w,p)
    other=dview_geometry_end(w,q)
    if other>end:
        end=other
    shared=dview_shared_offset(w,p,q)
    if end<shared or end-shared>so_uget(w,24):
        return terror(w,9)
    return end-shared

def dview_input_maybe_overlap(w: mutbuf, ident: u64) -> u64:
    p=dptr(w,ident)
    used=0
    extent=0
    stage=0
    while stage<getw(w,p):
        chosen=4
        smallest=18446744073709551615
        axis=0
        while axis<getw(w,p):
            if (used & (1<<axis))==0 and getw(w,p+96+8*axis)>1:
                stride=getw(w,p+128+8*axis)
                if chosen==4 or stride<smallest:
                    chosen=axis
                    smallest=stride
            axis=axis+1
        if chosen==4:
            return 0
        if smallest<=extent:
            return 1
        extent=extent+smallest*(getw(w,p+96+8*chosen)-1)
        used=used | (1<<chosen)
        stage=stage+1
    return 0

def dview_input_coverage(w: mutbuf, ident: u64, target: u64) -> u64:
    p=dptr(w,ident)
    count=0
    index=0
    while index<getw(w,p+8):
        if to_view_logical_offset(w,ident,index)==target:
            count=count+1
        index=index+1
    return count

def dview_output_value(w: mutbuf, child: u64, target: u64) -> u64:
    p=dptr(w,child)
    overlap=to_as_reduced_overlap(w,child)
    result=0
    index=0
    while index<getw(w,p+8):
        if to_as_reduction_representative(w,child,index)!=0:
            if to_view_logical_offset(w,child,index)==target:
                value=to_sum_value0(w,child,index)
                if overlap!=0:
                    result=scalar_add_preserving_bits(result,value)
                else:
                    result=value
        index=index+1
    return result

def to_as_strided_backward(w: mutbuf, child: u64) -> u64:
    parent=gradient_parent_at(w,child,0)
    if parent==0:
        return 1
    p=dptr(w,parent)
    normalize=dview_input_maybe_overlap(w,parent)
    index=0
    while index<getw(w,p+8):
        target=to_view_logical_offset(w,parent,index)
        value=dview_output_value(w,child,target)
        if normalize!=0:
            value=dview_coverage_divide(value,dview_input_coverage(w,parent,target))
        if gadd(w,parent,index,value)==0:
            return 0
        index=index+1
    if getw(w,0)!=0:
        return 0
    return 1

def incoming_mark_preserve_targets(w: mutbuf, highest: u64, epoch: u64) -> u64:
    if dview_previous_mark_preserve_targets(w,highest,epoch)==0:
        return 0
    ident=1
    while ident<=highest:
        p=dptr(w,ident)
        if getw(w,p+304)!=0 and getw(w,p+168)==epoch:
            if getw(w,p+40)==44 and getw(w,p+8)!=0:
                parent=gradient_parent_at(w,ident,0)
                if parent!=0:
                    q=dptr(w,parent)
                    if getw(w,q+304)==0 or getw(w,q+168)!=epoch:
                        return terror(w,9)
                    if getw(w,q+8)!=0:
                        putw(w,q+544,1)
        ident=ident+1
    return 1

def dview_preserve_capacity(w: mutbuf, p: u64) -> u64:
    capacity=4*getw(w,p+8)
    epoch=getw(w,p+168)
    ident=1
    while ident<getw(w,8):
        q=dptr(w,ident)
        if getw(w,q+304)!=0 and getw(w,q+168)==epoch:
            if getw(w,q+40)==44 and getw(w,q+8)!=0:
                parent=gradient_parent_at(w,ident,0)
                if parent!=0:
                    if dptr(w,parent)==p and getw(w,p+8)!=0:
                        required=dview_span_bytes(w,p,q)
                        if required>capacity:
                            capacity=required
        ident=ident+1
    return capacity

def dview_trim_preserve(w: mutbuf, ident: u64, requested: u64) -> u64:
    p=dptr(w,ident)
    hold=so_slot_lookup(w,6,getw(w,p+552))
    storage=so_slot_lookup(w,2,getw(w,p+560))
    version=so_slot_lookup(w,3,getw(w,p+568))
    if hold==0 or storage==0 or version==0:
        return terror(w,9)
    if so_uget(w,hold+16)!=getw(w,p+560) or so_uget(w,hold+24)!=1:
        return terror(w,9)
    if so_uget(w,hold+32)!=0:
        return terror(w,9)
    if so_uget(w,storage+40)!=1 or so_uget(w,storage+48)!=1:
        return terror(w,9)
    if so_uget(w,version+16)!=0 or so_uget(w,version+24)!=1:
        return terror(w,9)
    old=so_uget(w,storage+24)
    oldrounded=so_uget(w,storage+32)
    rounded=so_align16(requested)
    if requested>old or oldrounded!=so_align16(old) or rounded>oldrounded:
        return terror(w,9)
    if rounded<oldrounded:
        if so_extent_release(w,so_uget(w,storage+16)+rounded,oldrounded-rounded)!=0:
            return terror(w,9)
    so_uput(w,storage+24,requested)
    so_uput(w,storage+32,rounded)
    return 1

def dview_construct_incoming(w: mutbuf, parent: u64, child: u64) -> u64:
    p=dptr(w,parent)
    q=dptr(w,child)
    if getw(w,p+8)==0 or getw(w,q+8)==0:
        return terror(w,9)
    if getw(w,p+576)!=0 or getw(w,q+576)!=0:
        return terror(w,9)
    size=dview_span_bytes(w,p,q)
    if dview_trim_preserve(w,parent,size)==0:
        return 0
    pointer=getw(w,p+544)
    storage=getw(w,p+560)
    version=getw(w,p+568)
    shared=dview_shared_offset(w,p,q)
    offset=getw(w,p+232)-shared
    position=0
    while position<size:
        put4(w,pointer+position,0)
        position=position+4
    overlap=to_as_reduced_overlap(w,child)
    index=0
    while index<getw(w,q+8):
        if to_as_reduction_representative(w,child,index)!=0:
            absolute=4*to_view_logical_offset(w,child,index)
            if absolute<shared or absolute-shared>size-4:
                return terror(w,9)
            target=pointer+absolute-shared
            value=to_sum_value0(w,child,index)
            if overlap!=0:
                value=scalar_add_preserving_bits(get4(w,target),value)
            put4(w,target,value)
        index=index+1
    if getw(w,0)!=0 or incoming_version_bump(w,version)==0:
        return 0
    if dview_input_maybe_overlap(w,parent)!=0:
        position=0
        while position<size:
            count=dview_input_coverage(w,parent,(shared+position)>>2)
            value=dview_coverage_divide(get4(w,pointer+position),count)
            put4(w,pointer+position,value)
            position=position+4
        if getw(w,0)!=0 or incoming_version_bump(w,version)==0:
            return 0
    if incoming_bind(w,parent,storage,version,offset,2,parent)==0:
        return 0
    axis=0
    while axis<4:
        putw(w,p+496+8*axis,getw(w,p+128+8*axis))
        axis=axis+1
    if getw(w,0)!=0:
        return 0
    return 1

def incoming_describe(w: mutbuf, child: u64, parent: u64, edge: u64) -> u64:
    p=dptr(w,child)
    q=dptr(w,parent)
    if getw(w,p+40)==44 and getw(w,p+8)!=0 and getw(w,q+8)!=0:
        memory_fill(w,3808,88,0)
        putw(w,3832,getw(w,q+232)-dview_shared_offset(w,q,p))
        putw(w,3840,child)
        putw(w,3848,2)
        axis=0
        while axis<4:
            putw(w,3856+8*axis,getw(w,q+128+8*axis))
            axis=axis+1
        putw(w,3888,3)
        return 1
    return dview_previous_incoming_describe(w,child,parent,edge)

def incoming_construct(w: mutbuf, parent: u64) -> u64:
    if getw(w,3808)==0 and getw(w,3888)==3:
        return dview_construct_incoming(w,parent,getw(w,3840))
    if getw(w,3808)==0 and incoming_contiguous(w,parent,3856)==0:
        p=dptr(w,parent)
        if dview_trim_preserve(w,parent,4*getw(w,p+8))==0:
            return 0
    return dview_previous_incoming_construct(w,parent)
