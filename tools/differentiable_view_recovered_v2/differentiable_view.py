def to_strided_view(s:readbuf, w:mutbuf, id:u64, config:u64, end:u64) -> u64:
    source = to_view_public(w,id)
    if source == 0:
        return 0
    p = dptr(w,source)
    if getw(w,p+576) != 0 and (getw(w,p+32) != 0 or getw(w,p+40) != 0):
        return terror(w,5)
    if getw(w,p+408) != 0 and getw(w,p+664) == 0:
        return terror(w,5)
    result = rng_strided_view(s,w,id,config,end)
    if result == 0:
        return 0
    if tensor_dtype(w,result) != 2:
        if to_view_root_retain(w,result,source) == 0:
            return 0
    # The Tensor ViewInfo keeps the ultimate base at 640/648. Its initial
    # derivative separately captures the immediate self edge and geometry.
    # Own requires remains false on a newly generated nonleaf view.
    if getw(w,p+32) != 0:
        q = dptr(w,result)
        putw(w,q+32,1)
        if record(w,result,44,source,0) == 0:
            return 0
    return result
