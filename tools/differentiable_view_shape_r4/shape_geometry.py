def shape_infer_dimensions(w: mutbuf, source: u64, rank: u64, sizes: u64) -> u64:
    if rank>4:
        return terror(w,2)
    p=dptr(w,source)
    inferred=4
    has_zero=0
    axis=0
    while axis<rank:
        size=getw(w,sizes+8*axis)
        if size==18446744073709551615:
            if inferred!=4:
                return terror(w,2)
            inferred=axis
        else:
            if size>262144:
                return terror(w,2)
            if size==0:
                has_zero=1
        axis=axis+1
    if has_zero!=0:
        if inferred!=4 or getw(w,p+8)!=0:
            return terror(w,2)
        return 1
    known=1
    axis=0
    while axis<rank:
        if axis!=inferred:
            size=getw(w,sizes+8*axis)
            if size==0 or known>9223372036854775807//size:
                return terror(w,2)
            known=known*size
        axis=axis+1
    count=getw(w,p+8)
    if inferred!=4:
        if known==0 or count%known!=0:
            return terror(w,2)
        putw(w,sizes+8*inferred,count//known)
    else:
        if count!=known:
            return terror(w,2)
    return 1

def shape_compute_strides(w: mutbuf, source: u64, rank: u64, sizes: u64, strides: u64) -> u64:
    p=dptr(w,source)
    oldrank=getw(w,p)
    axis=0
    while axis<4:
        putw(w,strides+8*axis,0)
        axis=axis+1
    if oldrank==0:
        axis=0
        while axis<rank:
            putw(w,strides+8*axis,1)
            axis=axis+1
        return 1
    if getw(w,p+8)==0:
        same=0
        if oldrank==rank:
            same=1
            axis=0
            while axis<rank:
                if getw(w,p+96+8*axis)!=getw(w,sizes+8*axis):
                    same=0
                axis=axis+1
        if same!=0:
            axis=0
            while axis<rank:
                putw(w,strides+8*axis,getw(w,p+128+8*axis))
                axis=axis+1
            return 1
        axis=rank
        product=1
        while axis>0:
            axis=axis-1
            putw(w,strides+8*axis,product)
            size=getw(w,sizes+8*axis)
            if size>1:
                product=product*size
        return 1
    remaining=rank
    chunk_stride=getw(w,p+128+8*(oldrank-1))
    tensor_count=1
    view_count=1
    old_axis=oldrank
    while old_axis>0:
        old_axis=old_axis-1
        tensor_count=tensor_count*getw(w,p+96+8*old_axis)
        boundary=0
        if old_axis==0:
            boundary=1
        else:
            if getw(w,p+96+8*(old_axis-1))!=1:
                if getw(w,p+128+8*(old_axis-1))!=tensor_count*chunk_stride:
                    boundary=1
        if boundary!=0:
            consuming=1
            while remaining>0 and consuming!=0:
                size=getw(w,sizes+8*(remaining-1))
                if view_count<tensor_count or size==1:
                    putw(w,strides+8*(remaining-1),view_count*chunk_stride)
                    view_count=view_count*size
                    remaining=remaining-1
                else:
                    consuming=0
            if view_count!=tensor_count:
                return 0
            if old_axis>0:
                chunk_stride=getw(w,p+128+8*(old_axis-1))
                tensor_count=1
                view_count=1
    if remaining!=0:
        return 0
    return 1
