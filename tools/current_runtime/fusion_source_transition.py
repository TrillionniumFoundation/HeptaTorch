"""Reverse the complete fixed729 source to exact cat700 source04.

Only source bytes are transformed. The actual executable/profile remains729.
This invokes no native tool or runtime and transfers no historical PASS.
"""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SOURCE = '04ae430e3e851831fcb97a3fab700b16b6318361d52e90fcea90d2183d1d564b'
PARENT_SOURCE = 'ba96abc05717643eaf4f2056325089d9a51f5a2e9838f89473e9a4e76e46fd99'
DELTA_SHA = '85a619dc25cdd8107ba53aa0646f8eab945d39588917de88fd35c33d574f4d6b'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def functions(raw, expected):
    matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', raw, re.M|re.S))
    bodies = {m[1].decode():m[0] for m in matches}
    require(len(bodies) == len(matches) == expected, 'exact complete function census')
    require(raw == raw.split(b'\n',1)[0]+b'\n'+b''.join(bodies.values()),
            'complete canonical source ordering and bytes')
    return bodies


def restore_source(source, bind):
    """Return the exact700 parent for old source proofs, plus the outer proof."""
    bind(Path(__file__).resolve())
    require(sha(source) == SOURCE, 'only the exact admitted729 source')
    raw = bind(HERE/'fusion_source_transition.json',DELTA_SHA)
    delta = json.loads(raw)
    require(delta['schema'] == 'heptatorch.fusion729-to-cat700.complete-source-delta.v1'
        and delta['source_sha256'] == SOURCE and delta['parent_source_sha256'] == PARENT_SOURCE
        and delta['functions'] == 729 and delta['parent_functions'] == 700
        and delta['unchanged_parent_functions'] == 682
        and len(delta['changed_functions']) == 18 and len(delta['added_functions']) == 29,
        'exact729/700 function delta')
    actual = functions(source,729)
    require(source.split(b'\n',1)[0].decode() == delta['header']
        and len(source) == delta['source_bytes'], 'exact canonical source header and extent')
    parent_order, added = delta['parent_function_order'],delta['added_functions']
    changed = delta['changed_functions']
    require(len(parent_order) == len(set(parent_order)) == 700
        and len(added) == len(set(added)) == 29 and not set(parent_order)&set(added)
        and list(actual) == parent_order+added
        and len(changed) == len(set(changed)) == 18 and set(changed) <= set(parent_order)
        and set(delta['bodies']) == set(changed) and set(delta['added_bodies']) == set(added),
        'exact declaration order and finite change sets')
    suffix = b''.join(actual[name] for name in added)
    require(source.endswith(suffix) and len(suffix) == delta['added_suffix_bytes']
        and sha(suffix) == delta['added_suffix_sha256'], 'complete appended29 suffix')
    for name in added:
        row,body = delta['added_bodies'][name],actual[name]
        require(len(body) == row['bytes'] and sha(body) == row['sha256'],
                'exact appended complete body: '+name)
    restored = source[:-len(suffix)]
    originals = {}
    for name in changed:
        row,before,after = delta['bodies'][name],delta['bodies'][name]['before_body'].encode(),actual[name]
        require(before.startswith(('fn '+name+'(').encode()) and before.endswith(b'end\n')
            and len(before) == row['before_bytes'] and sha(before) == row['before_sha256']
            and len(after) == row['after_bytes'] and sha(after) == row['after_sha256']
            and before != after and restored.count(after) == 1,
            'exact original/current complete body: '+name)
        originals[name] = before
        restored = restored.replace(after,before,1)
    previous = functions(restored,700)
    require(list(previous) == parent_order
        and [name for name in previous if actual[name] != previous[name]] == changed
        and sum(actual[name] == previous[name] for name in previous) == 682,
        'only the exact18 parent bodies differ')
    require(len(restored) == delta['parent_bytes'] and sha(restored) == PARENT_SOURCE,
            'complete original700 source hash constrains every remaining byte')
    forward = restored
    for name in changed:
        require(forward.count(originals[name]) == 1, 'unique full-body forward reconstruction')
        forward = forward.replace(originals[name],actual[name],1)
    require(forward+suffix == source, 'complete729 source round trip')
    proof = dict(schema='heptatorch.fusion729-to-cat700.complete-source-inverse.v1',
        source_sha256=SOURCE,parent_source_sha256=PARENT_SOURCE,functions=729,parent_functions=700,
        changed_functions=changed,unchanged_parent_functions=682,added_functions=added,
        delta_sha256=DELTA_SHA,whole_source_inverse_exact=True,whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        actual_executable_replaced=False,source_equivalence_only=True,
        original_cases_or_expected_values_modified=False,runtime_qualification_transferred=False)
    return restored,proof
