def integer_view_root_retain(w:mutbuf, view:u64, source:u64) -> u64:
    p = dptr(w,view)
    q = dptr(w,source)
    if getw(w,p+576) != 2 or getw(w,q+576) != 2:
        return terror(w,9)
    root = source
    if getw(w,q+408) != 0:
        root = dtype_mask_node(w,source)
        if root == 0:
            return 0
    q = dptr(w,root)
    if getw(w,q+408) != 0 or getw(w,q+584) != 0 or getw(w,q+40) != 0:
        return terror(w,9)
    if getw(w,p+216) != getw(w,q+216) or getw(w,p+224) != getw(w,q+224):
        return terror(w,9)
    if getw(w,q+320) >= getw(w,p+320):
        return terror(w,9)
    if dtype_mask_retain(w,view,root) == 0:
        return 0
    putw(w,p+408,1)
    return 1

def integer_view_root_valid(w:mutbuf, p:u64, q:u64) -> u64:
    if q == 0 or getw(w,p+576) != 2 or getw(w,p+408) != 1:
        return 0
    if getw(w,p+40) != 0 or getw(w,p+32) != 0 or getw(w,p+288) != 0:
        return 0
    if getw(w,q+576) != 2 or getw(w,q+408) != 0 or getw(w,q+584) != 0 or getw(w,q+40) != 0:
        return 0
    if getw(w,p+216) != getw(w,q+216) or getw(w,p+224) != getw(w,q+224):
        return 0
    if getw(w,q+320) >= getw(w,p+320):
        return 0
    return 1
