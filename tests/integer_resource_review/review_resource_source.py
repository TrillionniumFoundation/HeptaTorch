#!/usr/bin/env python3
"""Narrow source/delta custody; no native or reference execution."""
import argparse
import ast
import json
from pathlib import Path
import re

from resource_support import Custody, sha, write_json

OLD_SOURCE = '46cba6e9951de45374f4ba0a7567811cd9e03a16db0e5b9e67ba7512559b9bb3'
NEW_SOURCE = 'adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62'
NEW_IMAGE = 'b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08'


def functions(raw):
    matches = list(re.finditer(rb'^fn ([^(\n]+)\(', raw, re.M))
    result = {}
    for i, match in enumerate(matches):
        name = match.group(1).decode()
        assert name not in result
        end = matches[i+1].start() if i+1 < len(matches) else len(raw)
        result[name] = raw[match.start():end]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'prior-build', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    custody = Custody(out)
    custody.bind(Path(__file__), copy='executed-review.py')
    custody.bind(Path(__file__).with_name('resource_support.py'), copy='resource_support.py')
    old = functions(custody.bind(a.prior_build/'runtime.hepta', OLD_SOURCE, 'candidate06-runtime.hepta'))
    new = functions(custody.bind(a.build/'runtime.hepta', NEW_SOURCE, 'candidate07-runtime.hepta'))
    custody.bind(a.build/'tensor-integer-session', NEW_IMAGE, 'candidate07-native')
    assert set(old) == set(new) and len(new) == 544
    changed = [name for name in old if old[name] != new[name]]
    expected = {'integer_inplace_overlap', 'integer_result_dtype', 'integer_binary'}
    assert set(changed) == expected, changed
    author_raw = custody.bind(a.build/'overlay-authoring.py', copy='overlay-authoring.py')
    text = author_raw.decode()
    author = {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    selected = ['typed_descriptor', 'typed_load', 'typed_store', 'typed_from_wire',
        'saved_create', 'saved_logical', 'unpack_saved', 'weak_create', 'weak_lock',
        'so_storage_release', 'so_weak_storage_release', 'so_extent_allocate', 'so_extent_release',
        'so_state_migrate', 'integer_promote_operand', 'integer_binary', 'integer_command',
        'integer_dropout_nonempty', 'integer_native_dropout', 'integer_dropout_command',
        'native_dropout_fill_mask', 'finish_command', 'graph_collect_since',
        'integer_view_root_retain', 'integer_view_root_valid', 'dtype_audit', 'history_audit', 'entry']
    assert all(name in new for name in selected)
    rows = []
    for name in selected:
        path = out/'functions'/(name+'.hepta')
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(new[name])
        row = {'name': name, 'hia_bytes': len(new[name]), 'hia_sha256': sha(new[name]),
               'unchanged_from_candidate06': new[name] == old[name]}
        if name in author:
            row['current_authoring_sha256'] = sha(author[name].encode())
            row['current_normalized_AST_sha256'] = sha(ast.dump(ast.parse(author[name]), include_attributes=False).encode())
        rows.append(row)
    command = author['integer_dropout_command']
    assert command.index('16 > length(o)-at-32') < command.index('integer_native_dropout(')
    assert command.index('finish_command(') < command.index('putw(o,at+32,error)')
    drop = author['integer_dropout_nonempty']
    assert drop.index('mask = typed_descriptor(') < drop.index('output = typed_descriptor(') < drop.index('native_dropout_fill_mask(')
    promotion = author['integer_promote_operand']
    assert promotion.index('typed_descriptor(') < promotion.index('dist_dense_layout(') < promotion.index('while index < n:')
    binary = author['integer_binary']
    assert binary.index('integer_promote_operand(w,a,dtype)') < binary.index('integer_pointwise_layout(')
    assert binary.index('integer_promote_operand(w,b,dtype)') < binary.index('integer_pointwise_layout(')
    report = {'schema': 'heptatorch.i64-resource-source-review.v1', 'status': 'PASS',
        'review_kind': 'source custody, exact function delta and narrow resource ordering; not new runtime qualification',
        'native_executed': False, 'Torch_executed': False,
        'source_sha256': NEW_SOURCE, 'native_sha256': NEW_IMAGE, 'parent_source_sha256': OLD_SOURCE,
        'functions_compared': 544, 'unchanged_functions': 541, 'changed_functions': changed,
        'selected_functions': rows,
        'findings': [
            'Both real dropout allocations precede native mask/RNG generation; capacity checks precede the operation.',
            'CPU promotion allocates a real typed descriptor and Preserve-layout byte storage before output layout and arithmetic.',
            'Promotion results remain unpublished; integer_command publishes only the output and calls finish_command.',
            'Explicit105 runtime-cast error12 runs finish_command and normal header commit; entry stops the suffix on its nonzero payload.',
            'Storage release retains logical requested bytes in a weak tombstone; tests require physical offset/rounded extent to become zero.',
            'Resource tests inspect actual producer-owned offsets and tables and compare the complete owner on admission error; arbitrary forged owner admission is outside scope.',
            'The 07 overlap/error-order patch changes three arithmetic functions only; 06 runtime results are retained as historical observations, not relabelled as07.'
        ],
        'limitations': ['This is a limited source review and producer-state observer, not whole machine-code certification.',
                        'The unchanged full PyTorch gate remains HOLD with 2590 API denominator and zero full-rewrite closure.'],
        'inputs': custody.finish(), 'source_custody_after': True,
        'whole_pytorch_gate': 'HOLD', 'pytorch_api_denominator': 2590,
        'full_pytorch_rewrite_complete': False}
    write_json(out/'report.json', report)
    print(json.dumps({'status': 'PASS', 'functions_compared': 544, 'unchanged': 541,
                      'source_sha256': NEW_SOURCE, 'report_sha256': sha((out/'report.json').read_bytes())}))


if __name__ == '__main__':
    main()
