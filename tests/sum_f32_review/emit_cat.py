#!/usr/bin/env python3
"""Emit one source-only HIA correction; never compile or execute any runtime."""
import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path

BASE_SHA256 = '706673d47858a47fafeefe9f40aed6c52e5181caa4183ec12f72151385ca84e3'
FUNCTION_SHA256 = '6fec0a6aec5be686824fb0e78b4ebb93e5584c1962732dc8039784e2fdee1383'
FUNCTION = 'incoming_describe_cat'
BEFORE = b'''block b3(v35:mutbuf)
  v51:u64 = const() {"value":3848}
  v52:u64 = const() {"value":1}
  v53:u64 = call(v35,v51,v52) {"function":"putw"}
  ret(v52)
'''
AFTER = b'''block b3(v35:mutbuf)
  v51:u64 = const() {"value":3848}
  v52:u64 = const() {"value":1}
  v251:u64 = const() {"value":2}
  v53:u64 = call(v35,v51,v251) {"function":"putw"}
  ret(v52)
'''


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def functions(raw):
    found = list(re.finditer(rb'(?ms)^fn ([A-Za-z0-9_]+)\([^\n]*\n.*?^end\n', raw))
    names = [item.group(1).decode('ascii') for item in found]
    assert len(names) == len(set(names)) == 701
    return names, {name: item.group(0) for name, item in zip(names, found)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'fresh source-only output required'
    original = args.source.read_bytes()
    assert digest(original) == BASE_SHA256, 'selected current701 r2 source differs'
    names, old = functions(original)
    body = old[FUNCTION]
    assert digest(body) == FUNCTION_SHA256
    assert body.count(BEFORE) == 1 and b'v251:' not in body
    candidate_body = body.replace(BEFORE, AFTER, 1)
    assert candidate_body.count(AFTER) == 1
    candidate = original.replace(body, candidate_body, 1)
    new_names, new = functions(candidate)
    assert new_names == names
    changed = [name for name in names if old[name] != new[name]]
    assert changed == [FUNCTION]
    assert candidate_body.replace(AFTER, BEFORE, 1) == body
    assert candidate.replace(candidate_body, body, 1) == original
    assert original.splitlines()[0] == candidate.splitlines()[0]
    assert re.findall(rb'\{"function":"([^"]+)"\}', body) == re.findall(rb'\{"function":"([^"]+)"\}', candidate_body)
    args.output.mkdir(parents=True)
    payloads = {'runtime.hepta': candidate, 'before-function.hepta': body,
                'after-function.hepta': candidate_body,
                'candidate.diff': ''.join(difflib.unified_diff(
                    original.decode().splitlines(keepends=True),
                    candidate.decode().splitlines(keepends=True),
                    fromfile='current701-avx2/runtime.hepta',
                    tofile='current701-cat-empty-candidate/runtime.hepta')).encode()}
    records = {}
    for name, raw in payloads.items():
        path = args.output / name
        path.write_bytes(raw)
        path.chmod(0o644)
        records[name] = {'bytes': len(raw), 'sha256': digest(raw), 'mode': 0o644}
    report = {
        'schema': 'heptatorch.current701.cat-skipped-empty.source-only-candidate.v1',
        'status': 'SOURCE_ONLY_NOT_COMPILED_NOT_RUN',
        'base_source_sha256': BASE_SHA256,
        'candidate_source_sha256': digest(candidate),
        'original_functions': len(old), 'candidate_functions': len(new),
        'changed_functions': changed, 'unchanged_function_count': 700,
        'unchanged_functions': [{ 'name': name, 'sha256': digest(old[name]) }
                                for name in names if name != FUNCTION],
        'changed_function': {'name': FUNCTION, 'before_sha256': digest(body),
                             'after_sha256': digest(candidate_body)},
        'whole_source_inverse_exact': True, 'whole_function_inverse_exact': True,
        'function_names_and_order_unchanged': True,
        'function_call_targets_and_order_unchanged': True,
        'header_unchanged': True,
        'change': 'Only the skipped rank-one empty cat parent branch assigns incoming mode 2 instead of 1; return value stays exactly 1.',
        'reason': 'The branch creates a local cache-owned empty incoming gradient. A duplicate parent edge must use the ordinary dense accumulation path; mode 1 incorrectly forces the shared-cache refusal.',
        'unchanged_guards': ['incoming_can_accumulate', 'incoming_merge_edge',
                             'incoming_bind', 'concat_shape', 'gradient_parent_at'],
        'first_failure': {'case': 'cat-skipped-only-grad',
            'original_report_sha256': '381e5d1198512aa5f53877d6620dfb1f90e3697b72547bce6dbc2529eecc5839',
            'request_sha256': 'd3102ab3fe7e058363b37cc731e2e36cc8466a6ac9e02eb92b706258dbad3ef3',
            'response_sha256': '0f2e6e04d5e08316f6bd9b10b23c98c014edfff939a4771316f54387302030c2',
            'response_status': 9, 'response_records': 0,
            'original_actual_status': 'FAIL', 'failure_reclassified': False},
        'original_oracle_or_expectations_modified': False,
        'compiled': False, 'native_executed': False, 'Torch_executed': False,
        'files': records,
    }
    (args.output / 'transition.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    assert args.source.read_bytes() == original
    print(json.dumps({key: report[key] for key in ('status', 'base_source_sha256',
          'candidate_source_sha256', 'changed_functions', 'unchanged_function_count')}))


if __name__ == '__main__':
    main()
