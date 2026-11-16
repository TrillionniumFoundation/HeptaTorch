def optimizer_new_effective_momentum(w:mutbuf, parameter:u64, group:u64) -> u64:
    t = optimizer_new_momentum(w,parameter)
    if t == 0:
        return 0
    source = gradient_binding(w,parameter)
    if source == 0:
        return terror(w,9)
    if getw(w,group+40) != 0:
        if optimizer_effective_pointwise_layout(w,t,source,source) == 0:
            return 0
        source = t
    if absf(getw(w,group+24)) != 0:
        if optimizer_effective_pointwise_layout(w,t,source,parameter) == 0:
            return 0
    return t
