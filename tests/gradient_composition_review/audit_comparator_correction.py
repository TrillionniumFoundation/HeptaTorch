#!/usr/bin/env python3
"""Independently bind the one-function signed byte-stride correction.

This is source custody and a narrow arithmetic review, not a numerical test.
The separate fixed eight-case report establishes actual backend observations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


SHA = lambda raw: hashlib.sha256(raw).hexdigest()
OLD_SOURCE = '3e6e879444e2795d824ec074b2bc8ded4ff058c68382c6752e1af13c5a432401'
NEW_SOURCE = '1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae'
OLD_IMAGE = '4ed8d68613fcec6b9b236ecdad47ea79b651988a8848b360da1d39fa2f5723be'
NEW_IMAGE = '6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe'


def definitions(raw):
    matches = re.findall(rb'(^fn ([a-zA-Z0-9_]+)\([^\n]*\) effects\([^\n]*\) entry\([^\n]*\)\n.*?^end\n)',
                         raw, re.M | re.S)
    result = {name.decode(): body for body, name in matches}
    assert len(result) == len(matches)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-build', 'new-build', 'iterator-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    custody = {}

    def read(path, expected=None):
        path = path.resolve()
        raw = path.read_bytes()
        if expected:
            assert SHA(raw) == expected
        custody[path] = raw
        return raw

    old = read(args.old_build / 'runtime.hepta', OLD_SOURCE)
    new = read(args.new_build / 'runtime.hepta', NEW_SOURCE)
    image_name = 'tensor-gradient-adam-distributions-session'
    read(args.old_build / image_name, OLD_IMAGE)
    read(args.new_build / image_name, NEW_IMAGE)
    old_functions, new_functions = definitions(old), definitions(new)
    assert set(old_functions) == set(new_functions) and len(new_functions) == 428
    changed = sorted(name for name in new_functions if old_functions[name] != new_functions[name])
    assert changed == ['pointwise_compare_axes']
    old_authoring = read(args.old_build / 'pointwise-layout-authoring.py',
                         '86f26ca4667cf6110b4595182c7f7d27ec0c3131b7bad2ace5ffbe6140833057')
    new_authoring = read(args.new_build / 'pointwise-layout-authoring.py',
                         '89703d7dcdd6b7623b9c8261053110a1849754c81195e1718ea652872ae3e2ec')
    inverse = new_authoring
    edits = [
        (b'ls = dist_probability_stride(w,out,source,left) << 2',
         b'ls = dist_probability_stride(w,out,source,left)'),
        (b'rs = dist_probability_stride(w,out,source,right) << 2',
         b'rs = dist_probability_stride(w,out,source,right)'),
        (b'            ordered_left = ls ^ 9223372036854775808\n', b''),
        (b'            ordered_right = rs ^ 9223372036854775808\n', b''),
        (b'if ordered_left < ordered_right:', b'if ls < rs:'),
        (b'if ordered_left > ordered_right:', b'if ls > rs:'),
    ]
    for before, after in edits:
        assert inverse.count(before) == 1
        inverse = inverse.replace(before, after)
    assert inverse == old_authoring
    iterator = read(args.iterator_source,
                    'a71a7ec9ba0532bf1380aa8ebc92975e4acb31414cd4b8eeeba650afe9e4490c')
    assert b'op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;' in iterator
    assert b'int64_t stride0 = operands_[arg].stride_bytes[dim0];' in iterator
    shared_units = {}
    for name in ('rng03.hepta', 'owners.hepta'):
        a, b = read(args.old_build / name), read(args.new_build / name)
        assert a == b
        shared_units[name] = SHA(a)
    read(Path(__file__))
    for path, raw in custody.items():
        assert path.read_bytes() == raw
    report = {
        'schema': 'heptatorch.gradient-composition.comparator-source-correction.v1',
        'status': 'PASS_SOURCE_CUSTODY_AND_NARROW_REVIEW',
        'source_only': True, 'native_executed_here': False, 'torch_executed_here': False,
        'old_source_sha256': OLD_SOURCE, 'new_source_sha256': NEW_SOURCE,
        'old_executable_sha256': OLD_IMAGE, 'new_executable_sha256': NEW_IMAGE,
        'functions': 428, 'unchanged_function_bodies': 427, 'changed_functions': changed,
        'function_body_sha256': {name: {'before': SHA(old_functions[name]),
                                       'after': SHA(new_functions[name]),
                                       'identical': old_functions[name] == new_functions[name]}
                                 for name in sorted(new_functions)},
        'authoring_exact_inverse_sha256': SHA(inverse),
        'authoring_inverse_edits': [{'new': a.decode(), 'old': b.decode(), 'count': 1}
                                    for a, b in edits],
        'shared_owned_and_rng_sources': shared_units,
        'scope_reasoning': [
            'The actual upstream comparator operates on signed int64 byte strides, after float32 element strides have been scaled by four. The revised helper forms the u64 shifted word first.',
            'Testing wrapped byte stride zero before ordering reproduces the upstream ambiguous/broadcast comparison rule. This covers a nonzero element stride whose scaled word becomes zero.',
            'For every 64-bit word, toggling bit 63 maps signed two-complement order to unsigned order. Comparing the transformed words therefore implements the signed byte-stride ordering without narrowing the public element-stride admission.',
            'Equal-byte-stride shape tie handling, input order, insertion sort, fast paths, output allocation, tensor address calculation, owner/state and all numerical kernels are unchanged byte-for-byte.',
            'The unchanged caller, descriptor and view-admission functions are included in the 427-body identity check. No case-specific stride lookup table or rejection was inserted.',
        ],
        'official_source': {'sha256': SHA(iterator), 'lines': [[249, 281], [1245, 1261]],
                            'url': 'https://raw.githubusercontent.com/pytorch/pytorch/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/TensorIterator.cpp'},
        'input_sha256': {str(path): SHA(raw) for path, raw in custody.items()},
        'input_custody_after': True, 'full_pytorch_rewrite_complete': False,
        'limits': ['Source review does not replace actual native or fixed-wheel execution.',
                   'Existing float32, rank-at-most-four and admitted-view domain remains in force.',
                   'This is not a claim about all dtypes, all architectures or the entire TensorIterator API.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'changed_functions': changed,
                      'unchanged_function_bodies': 427, 'report_sha256': SHA(args.output.read_bytes())}))


if __name__ == '__main__':
    main()
