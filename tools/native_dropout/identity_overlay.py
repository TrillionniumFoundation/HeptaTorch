"""Deterministic authoring edits for separate current Tensor / captured AD history.

This module edits checked HIA authoring text only. No tensor operation runs here.
"""
from authoring_support import once, sha


def compose_identity(original, patches, own, combined, emitter, aliases):
    for name in ('partial_check_node', 'record', 'release_backward_scratch'):
        alias = 'history_original_' + name
        body = once(combined[name], ('fn ' + name + '(').encode(), ('fn ' + alias + '(').encode())
        assert once(body, ('fn ' + alias + '(').encode(), ('fn ' + name + '(').encode()) == combined[name]
        combined[alias] = body
        emitter.add_context(alias, body)
        aliases[name + ':history'] = {'original_sha256': sha(combined[name]), 'alias_sha256': sha(body), 'inverse_exact': True}

    # Old Tensor ordinal order is no longer AD creation order after set_history.
    queue = original['partial_execute']
    queue = once(queue, '    epoch = getw(w, 24) + 1\n',
        '    t = history_backward_root(w,t)\n    if t == 0:\n        return 0\n'
        '    limit = getw(w,8)-1\n    epoch = getw(w, 24) + 1\n')
    queue = once(queue, '    partial_mark(w, t, epoch)\n',
        '    if partial_mark(w,t,epoch) == 0:\n        return 0\n')
    for old, new in (
        ('24 + (t + 1) * 16', '24 + (limit + 1) * 16'),
        ('gradient_workspace_size(w, t, epoch)', 'gradient_workspace_size(w, limit, epoch)'),
        ('(t + 1) * 8', '(limit + 1) * 8'),
        ('putw(w, queue + 8, t)', 'putw(w, queue + 8, limit)'),
        ('ensure_backward_work(w, t, epoch)', 'ensure_backward_work(w, limit, epoch)'),
        ('retained_reserve(w, retained_table, t, epoch)', 'retained_reserve(w, retained_table, limit, epoch)'),
        ('while i <= t:', 'while i <= limit:'),
        ('gradient_clear_marks(w, t, epoch)', 'gradient_clear_marks(w, limit, epoch)'),
        ('release_backward_scratch(w, t, epoch)', 'release_backward_scratch(w, limit, epoch)'),
        ('retained_release_reservations(w, retained_table, t)', 'retained_release_reservations(w, retained_table, limit)'),
    ):
        queue = once(queue, old, new)
    patches['partial_execute'] = queue
    atomic = once(queue, 'def partial_execute(w: mutbuf, o: mutbuf, info: u64, t: u64, seed: u64, retain: u64)',
        'def history_atomic_execute(w: mutbuf, t: u64, seed: u64, retain: u64)')
    atomic = once(atomic,
        '    putw(o, info, error)\n    putw(o, info + 8, completed)\n    putw(o, info + 16, leaves)\n    putw(o, info + 24, failed_op)\n    return error',
        '    if error != 0:\n        return terror(w,error)\n    return 1')
    patches['history_atomic_execute'] = atomic
    patches['partial_push'] = once(original['partial_push'], 'if previous < key:',
        'if history_key_less(w,previous,key) != 0:')
    pop = once(original['partial_pop'], 'if getw(w, heap + (child + 1) * 8) > getw(w, heap + child * 8):',
        'if history_key_less(w,getw(w,heap+child*8),getw(w,heap+(child+1)*8)) != 0:')
    patches['partial_pop'] = once(pop, 'if key < value:', 'if history_key_less(w,key,value) != 0:')
    ensure = once(original['ensure_backward_work'],
        '            if getw(w, p + 40) == 0 and getw(w, p + 32) != 0 and (getw(w, p + 384) == 0):\n                if gradient_new(w, i) == 0:\n                    return 0\n',
        '            if history_prepare_leaf(w,i) == 0:\n                return 0\n')
    patches['ensure_backward_work'] = ensure
    patches['graph_audit'] = once(patches['graph_audit'], 'flag > 1 or parent >= i',
        'flag > 1 or (parent != 0 and flag != 0 and history_order(w,parent) >= history_order(w,i))')
    retained = original['retained_apply']
    retained = once(retained, 'if retained_enabled(w, t) == 0:', 'if history_retained_enabled(w,t) == 0:')
    retained = once(retained, '    p = dptr(w, t)\n',
        '    target = history_retained_target(w,t)\n    if target == 0:\n        return terror(w,9)\n    p = dptr(w, t)\n')
    retained = once(retained, 'gradient_binding(w, t)', 'gradient_binding(w,target)')
    patches['retained_apply'] = once(retained, 'gradient_assign(w, t, g)', 'gradient_assign(w,target,g)')
    patches['retained_reserve'] = once(original['retained_reserve'], 'retained_enabled(w, i)', 'history_retained_enabled(w,i)')
    patches['retained_enable'] = once(original['retained_enable'], '    so_uput(w, row + 56, 1)\n',
        '    if so_uget(w,row+56) == 0:\n        putw(w,p+608,getw(w,p+608) & 65535)\n    so_uput(w, row + 56, 1)\n')

    # The actual parameter owns its .grad; a captured accumulator has no fake G.
    acc = once(original['gradient_accumulate'], 'def gradient_accumulate(w: mutbuf, t: u64)',
        'def history_accumulate_source(w: mutbuf, t: u64, source: u64)')
    acc = once(acc, '    p = dptr(w, t)\n', '    p = dptr(w, t)\n    q = dptr(w,source)\n')
    for off in (448,472,488,160):
        old = f'getw(w, p + {off})'
        assert old in acc
        acc = acc.replace(old, f'getw(w, q + {off})')
    acc = once(acc, 'incoming_obeys_contract(w, t)', 'history_incoming_obeys_contract(w,t,source)')
    acc = once(acc, 'incoming_adopt_gradient(w, t, g)', 'incoming_adopt_gradient(w,source,g)')
    patches['history_accumulate_source'] = acc
    cycle = once(original['gradient_cycle_check'], '        gradient_cycle_push(w, getw(w, p + 384), t)\n',
        '        gradient_cycle_push(w, getw(w, p + 384), t)\n'
        '        gradient_cycle_push(w,history_accumulator_target(w,current),t)\n')
    patches['gradient_cycle_check'] = cycle

    typed = own['typed.py'].decode()
    typed = once(typed, '    if dtype_mask_release(w,t) == 0:\n        return 0\n    return dtype_original_graph_node_release(w,t)',
        '    if dtype_mask_release(w,t) == 0:\n        return 0\n'
        '    if history_accumulator_release(w,t) == 0:\n        return 0\n'
        '    p = dptr(w,t)\n    putw(w,p+608,0)\n    putw(w,p+624,0)\n    putw(w,p+632,0)\n'
        '    return dtype_original_graph_node_release(w,t)')
    typed = once(typed, 'if getw(w,p+616) != 0 or getw(w,p+624) != 0 or getw(w,p+632) != 0:',
        'if getw(w,p+616) != 0:')
    typed = once(typed, '    i = 1\n    while i < getw(w,8) and status == 0:\n',
        '    if status == 0:\n        status = history_audit(w)\n    i = 1\n    while i < getw(w,8) and status == 0:\n')
    typed = once(typed, 'if code > 1 or getw(w,p+608) != 0:', 'if code > 1:')
    typed = once(typed, '    return dtype_original_leaf_requires_grad(w,id,requires)\n',
        '    if requires == 1 and getw(w,dptr(w,t)+40) != 0:\n        return getw(w,dptr(w,t)+192)\n'
        '    return dtype_original_leaf_requires_grad(w,id,requires)\n')
    typed = once(typed, 'if code < 96 or code > 100:', 'if code < 96 or code > 101:')
    typed = once(typed, '    if (code == 96 or code == 97) and c != 0:\n',
        '    if code == 101 and (b != 0 or c != 0):\n        terror(w,7)\n'
        '    if (code == 96 or code == 97) and c != 0:\n')
    typed = once(typed, '                if code == 100:\n',
        '                if code == 101:\n                    if history_detach_inplace(w,t) != 0:\n                        result = publish_tensor(w,t)\n'
        '                if code == 100:\n')
    patches['entry'] = once(patches['entry'], 'code >= 96 and code <= 100', 'code >= 96 and code <= 101')

    nd = own['native_dropout.py'].decode()
    nd = once(nd,
        '        if getw(w,p+32) != 0:\n            return terror(w,5)\n', '')
    nd = once(nd, '        dist_dense_layout(w,mask,input,1)\n        putw(o,mask_out,publish_tensor(w,mask))\n',
        '        dist_dense_layout(w,mask,input,1)\n'
        '        if getw(w,p+32) != 0:\n'
        '            prior = history_capture(w,input)\n            if prior == 0:\n                return 0\n'
        '            empty_scale = 1065353216\n'
        '            if training == 1 and dist_prob64_valid(probability) != 0:\n'
        '                empty_scale = native_dropout_scale(dist_complement64(probability))\n'
        '            if native_dropout_record(w,input,prior,mask,empty_scale) == 0:\n                return 0\n'
        '        putw(o,mask_out,publish_tensor(w,mask))\n')
    nd = once(nd, 'def partial_apply_node(w:mutbuf, t:u64) -> u64:\n',
        'def partial_apply_node(w:mutbuf, t:u64) -> u64:\n'
        '    target = history_accumulator_target(w,t)\n'
        '    if getw(w,0) != 0:\n        return 0\n'
        '    if target != 0:\n'
        '        if history_accumulator_check(w,t) == 0:\n            return 0\n'
        '        return history_accumulate_source(w,target,t)\n')
    return typed, nd
