def optimizer_audit_entry(w: mutbuf, blob: u64, i: u64, first: u64) -> u64:
    ep = optimizer_entry_at(w, blob, i)
    node = getw(w, ep)
    incarnation = getw(w, ep + 8)
    p = private_tensor(w, node, incarnation)
    if p == 0 or retained_parameter_blocker(w, node) != 0:
        return 9
    if optimizer_audit_reference(w, node, incarnation) != 0:
        return 9
    key_index = getw(w, ep + 16)
    if key_index < first or key_index > i:
        return 9
    key = optimizer_entry_at(w, blob, key_index)
    if getw(w, key) != node or getw(w, key + 8) != incarnation or getw(w, key + 16) != key_index:
        return 9
    if key_index != i:
        if getw(w, ep + 24) != 0 or getw(w, ep + 32) != 0 or getw(w, ep + 40) != 0:
            return 9
    else:
        j = 0
        while j < i:
            other = optimizer_entry_at(w, blob, j)
            if getw(w, other) == node and getw(w, other + 8) == incarnation:
                return 9
            j = j + 1
        present = getw(w, key + 24)
        if present > 1:
            return 9
        if present == 0:
            if getw(w, key + 32) != 0 or getw(w, key + 40) != 0:
                return 9
        else:
            buffer = getw(w, key + 32)
            bp = private_tensor(w, buffer, getw(w, key + 40))
            if bp == 0 or getw(w, bp + 40) != 0 or getw(w, bp) != getw(w, p):
                return 9
            j = 0
            while j < getw(w, p):
                if getw(w, bp + 96 + j * 8) != getw(w, p + 96 + j * 8):
                    return 9
                j = j + 1
            if optimizer_audit_reference(w, node, incarnation) != 0 or optimizer_audit_reference(w, buffer, getw(w, key + 40)) != 0:
                return 9
    return choose(key_index == i, 1, 2)

def owned_adam_audit_entry(w: mutbuf, blob: u64, i: u64, first: u64) -> u64:
    ep = owned_adam_entry_at(w, blob, i)
    node = getw(w, ep)
    incarnation = getw(w, ep + 8)
    p = private_tensor(w, node, incarnation)
    if p == 0 or retained_parameter_blocker(w, node) != 0:
        return 9
    if optimizer_audit_reference(w, node, incarnation) != 0:
        return 9
    key_index = getw(w, ep + 16)
    if key_index < first or key_index > i:
        return 9
    key = owned_adam_entry_at(w, blob, key_index)
    if getw(w, key) != node or getw(w, key + 8) != incarnation or getw(w, key + 16) != key_index:
        return 9
    if key_index != i:
        j = 24
        while j < 80:
            if getw(w, ep + j) != 0:
                return 9
            j = j + 8
    else:
        j = 0
        while j < i:
            other = owned_adam_entry_at(w, blob, j)
            if getw(w, other) == node and getw(w, other + 8) == incarnation:
                return 9
            j = j + 1
        if owned_adam_audit_state(w, key, p) != 0:
            return 9
    return choose(key_index == i, 1, 2)

def zero_grad(w: mutbuf, t: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    p = dptr(w, t)
    if retained_parameter_blocker(w, t) != 0:
        return terror(w, 5)
    if getw(w, p + 360) == 0:
        return 1
    g = gradient_binding(w, t)
    if g == 0 or version_can_bump(w, g, 1) == 0:
        return 0
    if gradient_detach_inplace(w, g) == 0:
        return 0
    i = 0
    while i < getw(w, p + 8):
        put4(w, taddr(w, g, i), 0)
        i = i + 1
    return bump_version(w, g)

def zero_grad_none(w: mutbuf, t: u64) -> u64:
    if valid(w, t) == 0:
        return 0
    if retained_parameter_blocker(w, t) != 0:
        return terror(w, 5)
    return gradient_unbind(w, t)

def optimizer_zero(w: mutbuf, id: u64, none: u64) -> u64:
    if none > 1:
        return terror(w, 7)
    blob = optimizer_blob(w, id)
    if blob == 0:
        return 0
    i = 0
    while i < getw(w, blob + 8):
        ep = optimizer_entry_at(w, blob, i)
        node = getw(w, ep)
        if private_tensor(w, node, getw(w, ep + 8)) == 0:
            return terror(w, 9)
        if none != 0:
            zero_grad_none(w, node)
        else:
            zero_grad(w, node)
        if getw(w, 0) != 0:
            return 0
        i = i + 1
    return 1

def owned_adam_zero(w: mutbuf, id: u64, none: u64) -> u64:
    if none > 1:
        return terror(w, 7)
    blob = owned_adam_blob(w, id)
    if blob == 0:
        return 0
    i = 0
    while i < getw(w, blob + 8):
        ep = owned_adam_entry_at(w, blob, i)
        node = getw(w, ep)
        if private_tensor(w, node, getw(w, ep + 8)) == 0:
            return terror(w, 9)
        if none != 0:
            zero_grad_none(w, node)
        else:
            zero_grad(w, node)
        if getw(w, 0) != 0:
            return 0
        i = i + 1
    return 1
