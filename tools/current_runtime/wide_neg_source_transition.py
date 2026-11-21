"""Recover the complete765 source before public I64/F64 unary negation.

All other native bodies are unchanged. The inverse checks source identity;
the current767 executable and every affected request need actual execution.
"""
import hashlib
import json
from pathlib import Path
import re

SOURCE = 'e0c8e0a3c3ee16d11cad8af6be1b52bd56295581e74bdcc65131f71a57540025'
PARENT_SOURCE = 'b4b2996978dce56c9375ce97b2d2638472cbb154a4c0a7c5de822234e31a9c68'
FUNCTIONS, PARENT_FUNCTIONS = 767, 765
CHANGED = ('unary', 'command_argument')
ADDED = ('wide_neg_previous_unary', 'wide_neg_tensor')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def functions(raw, count):
    matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', raw, re.M | re.S))
    bodies = {m[1].decode('ascii'): m[0] for m in matches}
    require(len(bodies) == len(matches) == count, 'complete unique wide-neg function census')
    require(raw == raw.split(b'\n', 1)[0] + b'\n' + b''.join(bodies.values()),
            'complete canonical source order and envelope')
    return bodies


def restore_source(source, bind):
    bind(Path(__file__).resolve())
    require(isinstance(source, bytes) and sha(source) == SOURCE, 'exact current767 source')
    delta_raw = bind(Path(__file__).with_name('wide_neg_source_delta.json'))
    delta = json.loads(delta_raw)
    require(delta['schema'] == 'heptatorch.wide-neg-source-delta.v1'
            and delta['source_sha256'] == SOURCE and delta['parent_source_sha256'] == PARENT_SOURCE
            and delta['functions'] == FUNCTIONS and delta['parent_functions'] == PARENT_FUNCTIONS
            and set(delta['changed_functions']) == set(CHANGED)
            and tuple(delta['appended_functions']) == ADDED,
            'exact two changed and two appended wide-neg functions')
    actual = functions(source, FUNCTIONS)
    require(tuple(actual)[-2:] == ADDED, 'only the exact two helpers are appended')
    suffix = b''.join(actual[name] for name in ADDED)
    require(source.endswith(suffix) and sha(suffix) == delta['helper_sha256'],
            'complete appended helper bytes')
    restored, originals = source[:-len(suffix)], {}
    for name in CHANGED:
        row, after = delta['changed_functions'][name], actual[name]
        before = row['before'].encode('utf-8')
        require(before.startswith(('fn ' + name + '(').encode()) and before.endswith(b'end\n')
                and sha(before) == row['before_sha256'] and sha(after) == row['after_sha256']
                and before != after and restored.count(after) == 1,
                'exact complete original/current body: ' + name)
        originals[name] = before
        restored = restored.replace(after, before, 1)
    parent = functions(restored, PARENT_FUNCTIONS)
    require(sha(restored) == PARENT_SOURCE and list(actual)[:PARENT_FUNCTIONS] == list(parent)
            and {name for name in parent if actual[name] != parent[name]} == set(CHANGED)
            and sum(actual[name] == parent[name] for name in parent) == 763,
            'every parent byte and all763 unchanged functions are constrained')
    require(actual['wide_neg_previous_unary'] == parent['unary'].replace(
        b'fn unary(', b'fn wide_neg_previous_unary(', 1), 'complete previous unary path retained')
    old = b'  v132:bool = bool.and(v123,v131)\n'
    new = (b'  wide_unary_opcode:u64 = const() {"value":3}\n'
           b'  wide_not_unary:bool = compare.ne(v73,wide_unary_opcode)\n'
           b'  wide_not_admitted:bool = bool.and(v131,wide_not_unary)\n'
           b'  v132:bool = bool.and(v123,wide_not_admitted)\n')
    require(parent['command_argument'].count(old) == 1
            and actual['command_argument'] == parent['command_argument'].replace(old, new, 1),
            'only public F64 unary command admission changes')
    forward = restored
    for name in CHANGED:
        require(forward.count(originals[name]) == 1, 'unique complete forward body')
        forward = forward.replace(originals[name], actual[name], 1)
    require(forward + suffix == source, 'complete current767 round trip')
    return restored, dict(schema='heptatorch.wide-neg767-to-addsub765.complete-source-inverse.v1',
        source_sha256=SOURCE, parent_source_sha256=PARENT_SOURCE,
        functions=FUNCTIONS, parent_functions=PARENT_FUNCTIONS,
        changed_functions=list(CHANGED), added_functions=list(ADDED), removed_functions=[],
        unchanged_parent_functions=763, delta_sha256=sha(delta_raw),
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        previous_unary_body_retained=True, only_public_F64_unary_admission_extended=True,
        newly_supported_request_outcomes_require_separate_successor_evidence=True,
        actual_executable_replaced=False, historical_cases_or_expected_values_modified=False,
        runtime_qualification_transferred=False)
