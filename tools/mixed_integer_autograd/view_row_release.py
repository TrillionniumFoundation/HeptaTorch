def tensor_row_release(w:mutbuf, t:u64) -> u64:
    result = mixed_original_tensor_row_release(w,t)
    if result == 0:
        return 0
    p = dptr(w,t)
    if getw(w,p+240) == 0 and getw(w,p+576) == 2 and getw(w,p+408) != 0:
        return dtype_mask_release(w,t)
    return result
