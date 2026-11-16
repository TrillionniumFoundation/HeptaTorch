"""Exact authoring changes to the frozen typed-Storage implementation.

These strings are emitted as HIA; this file performs no Tensor computation.
"""


def patches(original, once):
    result = {}
    result['dtype_width'] = once(original['dtype_width'],
        '    return 0', '    if code == 2:\n        return 8\n    return 0')
    for name in ('typed_descriptor', 'typed_from_wire'):
        result[name] = once(original[name], 'if dtype == 1 and requires',
                            'if dtype != 0 and requires')
    result['tset'] = once(original['tset'], 'bits > 4294967295',
                         '(code != 2 and bits > 4294967295)')
    result['typed_fill'] = once(original['typed_fill'], 'bits > 4294967295',
                              '(code != 2 and bits > 4294967295)')
    audit = once(original['dtype_audit'], 'if code > 1:', 'if code > 2:')
    result['dtype_audit'] = once(audit, 'if code == 1 and', 'if code != 0 and')
    records = once(original['so_audit_records'],
                   'kind == 4 and so_uget(w, p + 24) > 1',
                   'kind == 4 and so_uget(w, p + 24) > 2')
    result['so_audit_records'] = once(records, 'so_uget(w, p + 32) > 1',
                                     'so_uget(w, p + 32) > 2')
    result['rng_strided_view'] = once(original['rng_strided_view'],
        '    return result',
        '    if tensor_dtype(w, result) == 2:\n        if integer_view_root_retain(w, result, t) == 0:\n            return 0\n    return result')
    result['record'] = once(original['record'], '    return result',
        '    if tensor_dtype(w, t) == 2 and (op == 10 or op == 16):\n        if integer_view_root_retain(w, t, a) == 0:\n            return 0\n    return result')
    audit = result['dtype_audit']
    audit = once(audit,
        '                if q == 0 or getw(w, p + 40) != 42 or getw(w, p + 288) == 0:',
        '                if code == 2 and getw(w, p + 408) != 0:\n                    if integer_view_root_valid(w, p, q) == 0:\n                        status = 9\n                    else:\n                        putw(w, q + 616, getw(w, q + 616) + 1)\n                elif q == 0 or getw(w, p + 40) != 42 or getw(w, p + 288) == 0:')
    result['dtype_audit'] = once(audit,
        '            elif getw(w, p + 592) != 0:',
        '            elif getw(w, p + 592) != 0 or (code == 2 and getw(w, p + 408) != 0):')
    return result
