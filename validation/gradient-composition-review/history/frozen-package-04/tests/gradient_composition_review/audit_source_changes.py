#!/usr/bin/env python3
"""Bind a narrow, source-only review to the final unified runtime.

This is not a native or live-PyTorch test. It records the reviewed source rules,
checks the emitted helper/caller identities, and reads the fixed actual backend
observations. Native execution remains in the separate three-gate review.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


SHA = lambda raw: hashlib.sha256(raw).hexdigest()
SOURCE = '3e6e879444e2795d824ec074b2bc8ded4ff058c68382c6752e1af13c5a432401'
IMAGE = '4ed8d68613fcec6b9b236ecdad47ea79b651988a8848b360da1d39fa2f5723be'
POINTWISE = '86f26ca4667cf6110b4595182c7f7d27ec0c3131b7bad2ace5ffbe6140833057'
ITERATOR = 'a71a7ec9ba0532bf1380aa8ebc92975e4acb31414cd4b8eeeba650afe9e4490c'
REFERENCE = 'e664c9ff717a104b8c7317dfc55677e0248e58f981cb8315eacb3975049048ff'
HELPER_AUTHORING = '01058054297bceabf42d1c34d91cdb15a042cc1560e89dd50387f71bc98fcc0c'
HELPER_HIA = 'b9d272f526920879e66970641f60cfcc0fc1d71aaf3c285e2781a5a909c6a436'
CALLER_HIA = '230b6c97281c694f0d4dbc19c251291d26604716266e6f05b061d486feb070de'
OLD_CALLER_HIA = '0753da5f592b0a53e114437a4c468cde53d5295f8ea63fb1e0ba21b55f564218'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'iterator-source', 'scalar-observations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    inputs = {}

    def read(path, expected=None):
        path = path.resolve()
        raw = path.read_bytes()
        if expected:
            assert SHA(raw) == expected, path
        inputs[path] = raw
        return raw

    runtime = read(build / 'runtime.hepta', SOURCE)
    read(build / 'tensor-gradient-adam-distributions-session', IMAGE)
    pointwise = read(build / 'pointwise-layout-authoring.py', POINTWISE)
    scalar = read(build / 'scalar-add-bits-authoring.py', HELPER_AUTHORING)
    iterator = read(args.iterator_source, ITERATOR)
    reference = json.loads(read(args.scalar_observations, REFERENCE))
    read(Path(__file__).resolve())
    assert reference['status'] == 'OBSERVED' and reference['torch_executed'] is True
    assert reference['cases'] == 12 and reference['variants'] == 48
    assert reference['probe_errors'] == 0
    checks = reference['oracle_provenance']['checks']
    assert len(checks) == 11 and all(value is True for value in checks.values())
    assert all(reference['source_files_unchanged'].values())

    def body(name):
        matches = re.findall(rb'^fn ' + name.encode() + rb'\([^\n]*\n.*?^end\n',
                             runtime, re.M | re.S)
        assert len(matches) == 1, name
        return matches[0]

    names = ['pointwise_compare_axes', 'pointwise_axis_order', 'pointwise_channels_last',
             'pointwise_write_order', 'pointwise_layout']
    assert [node.name for node in ast.parse(pointwise).body] == names
    assert [node.name for node in ast.parse(scalar).body] == ['scalar_add_preserving_bits']
    assert SHA(body('scalar_add_preserving_bits')) == HELPER_HIA
    caller = body('scalar_add_nograd')
    assert SHA(caller) == CALLER_HIA
    old_caller = caller.replace(b'{"function":"scalar_add_preserving_bits"}',
                                b'{"function":"addf"}')
    assert caller.count(b'{"function":"scalar_add_preserving_bits"}') == 1
    assert SHA(old_caller) == OLD_CALLER_HIA
    binary = body('binary')
    assert SHA(binary) == '7d9f510af9f83d7f6beef26f7b18c941741dc5ac62c2d6a131ac9097460f2452'
    assert binary.count(b'{"function":"pointwise_layout"}') == 1
    assert binary.count(b'{"function":"taddr"}') == 1
    assert b'v134:u64 = call(v110,v116,v119) {"function":"taddr"}' in binary
    assert b'v136:u64 = call(v110,v134,v135) {"function":"put4"}' in binary
    assert SHA(body('descriptor')) == '3ea0866cdc43a6eda40d576863b84268f1e05d69a3efbd60bac682a94da524df'
    assert SHA(body('dist_probability_stride')) == '5eb5964eb7601b311fe55f8b8f8b98bbef099aad36095bf7a2f73fec2e2b52bd'
    assert b'"value":9223372036854775808' in body('rng_strided_view')
    assert 131841 == sum(axis << (8 * index) for index, axis in enumerate([1, 3, 2, 0]))

    # These are observation-to-rule checks of the reviewed special-value branches,
    # not a second execution of the native target or a host numerical runtime.
    observation_rows = []
    for row in reference['rows']:
        for variant in row['variants']:
            a = variant['input_word']
            b = variant['scalar_admitted_binary32_word']
            assert 0 <= a <= 0xffffffff and 0 <= b <= 0xffffffff
            aa, bb = a & 0x7fffffff, b & 0x7fffffff
            expected = None
            rule = 'unchanged addf; finite control is outside this special-value check'
            if bb > 0x7f800000:
                rule, expected = 'other NaN first; preserve sign/payload, set quiet bit', b | 0x00400000
            elif aa > 0x7f800000:
                rule, expected = 'self NaN second; preserve sign/payload, set quiet bit', a | 0x00400000
            elif aa == bb == 0x7f800000 and ((a ^ b) & 0x80000000):
                rule, expected = 'opposite infinities give negative canonical quiet NaN', 0xffc00000
            words = variant['after']['binary32_words']
            if expected is not None:
                assert words and all(word == expected for word in words)
            observation_rows.append({'case': row['case'], 'variant': variant['variant'],
                                     'rule': rule, 'special_branch_checked': expected is not None,
                                     'admitted_scalar_word': b,
                                     'observed_unique_output_words': sorted(set(words))})
    assert len(observation_rows) == 48

    inspected_names = names + ['binary', 'dist_probability_stride', 'incoming_same_shape',
                              'is_contiguous', 'is_dense', 'descriptor', 'rng_strided_view',
                              'scalar_add_preserving_bits', 'scalar_add_nograd']
    findings = [
        {'id': 'fast-path-priority', 'official_lines': [1322, 1460],
         'conclusion': 'Same-shape gate then all-contiguous, all-channels-last, then all-dense with exact stride equality, including size-one axes. Dense result copies the final defined input strides.'},
        {'id': 'broadcast-strides', 'official_lines': [1203, 1263],
         'conclusion': 'Missing leading input axes and an input size-one axis expanded to a different output size use stride zero. Size-one to size-one retains its original stride for ordering.'},
        {'id': 'two-input-axis-order', 'official_lines': [195, 301],
         'conclusion': 'Reverse initial permutation; input a precedes b. Either zero stride skips to the next input. Equal nonzero strides swap only when the output left size exceeds right size; otherwise comparison proceeds to the next input. Ambiguous insertion comparisons continue toward earlier axes.'},
        {'id': 'empty-and-dense-output-strides', 'official_lines': [540, 602],
         'conclusion': 'Non-reverse fallback multiplies the raw sizes, including zero. Exact reverse permutation retains the new canonical contiguous layout. Channels-last writes axis order C,W,H,N. Empty same-shape inputs take the earlier contiguous branch.'},
        {'id': 'actual-emitted-binary-store',
         'conclusion': 'The final HIA binary calls pointwise_layout after successful tnew and stores each logical result using taddr. Input bindex/tget, scalar_binary and record remain the numerical/graph path; source review does not substitute for their separate regressions.'},
        {'id': 'bounded-arithmetic',
         'conclusion': 'Existing admitted rank is at most four and each dimension at most 2^18. Nonempty numel is at most 2^24. A zero-numel rank-four shape has at most three nonzero dimensions, so its maximum canonical/permuted element stride is at most 2^54 (byte stride at most 2^56). No new stride-product u64 overflow occurs within that existing domain. Accepted explicit input strides are below 2^63; arbitrary fabricated descriptor bytes are outside this review.'},
        {'id': 'scalar-add-special-values',
         'conclusion': 'Both inputs are masked to u32 first. Abs bits above infinity identify NaN. Other is checked before self; OR 0x00400000 preserves sign and payload while quieting. Opposite infinities yield 0xffc00000. All remaining arithmetic delegates unchanged addf. Actual backend evidence uses the admitted f32 scalar after Python Scalar conversion.'},
        {'id': 'scalar-caller-exact-inverse',
         'conclusion': 'Replacing the single new helper callee with addf recovers the exact historical caller hash; owner, layout, iteration and Version logic are not replaced by this patch.'},
    ]
    for path, raw in inputs.items():
        assert path.read_bytes() == raw, path
    report = {
        'schema': 'heptatorch.gradient-composition.independent-source-review.v1',
        'status': 'NO_BLOCKER_FOUND_IN_REVIEWED_SOURCE', 'source_sha256': SOURCE,
        'executable_sha256': IMAGE, 'source_review_only': True,
        'native_tests_executed_here': 0, 'live_torch_tests_executed_here': 0,
        'actual_backend_reference_sha256': REFERENCE,
        'backend_provenance_checks': checks,
        'official_source': {'sha256': ITERATOR, 'bytes': len(iterator),
                            'commit': '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457',
                            'url': 'https://raw.githubusercontent.com/pytorch/pytorch/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/TensorIterator.cpp'},
        'findings': findings,
        'emitted_function_sha256': {name: SHA(body(name)) for name in inspected_names},
        'caller_exact_inverse_sha256': SHA(old_caller),
        'backend_special_value_observation_checks': observation_rows,
        'input_sha256': {str(path): SHA(raw) for path, raw in inputs.items()},
        'input_custody_after': True,
        'scope_limits': ['Existing valid float32 Tensor state, rank at most four.',
                         'Fresh functional binary output; not arbitrary out/inplace TensorIterator behavior.',
                         'No new native execution or live Torch run in this source review.',
                         'No all-ISA, all-compiler, dtype or whole-PyTorch closure claim.'],
        'whole_pytorch_rewrite_complete': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'report_sha256': SHA(args.output.read_bytes()),
                      'native_tests_executed_here': 0, 'live_torch_tests_executed_here': 0}))


if __name__ == '__main__':
    main()
