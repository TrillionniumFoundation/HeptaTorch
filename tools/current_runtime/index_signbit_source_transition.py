"""Recover the complete775 predecessor before native indexing and signbit.

Only source bytes are reconstructed. Every runtime test still executes the new
778-function image. This relation grants no numerical or whole-project result.
"""

import hashlib
import json
from pathlib import Path
import re

SOURCE = 'db88eaad9fb69010ae4f6925f4e8c54edce815b9f46507bc0c7c2536a4cb2c0b'
PARENT_SOURCE = '0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43'
FUNCTIONS, PARENT_FUNCTIONS = 778, 775
CHANGED = ('entry', 'index_select', 'predicate_bits', 'predicate_command')
ADDED = ('typed_index_select_previous', 'typed_index_select', 'typed_index_select_offset')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def functions(raw, count):
    matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', raw, re.M | re.S))
    bodies = {match[1].decode('ascii'): match[0] for match in matches}
    require(len(bodies) == len(matches) == count, 'complete unique function census')
    require(raw == raw.split(b'\n', 1)[0] + b'\n' + b''.join(bodies.values()),
            'complete canonical source envelope and definition order')
    return bodies


def restore_source(source, bind):
    bind(Path(__file__).resolve())
    delta_raw = bind(Path(__file__).with_name('index_signbit_source_delta.json'))
    delta = json.loads(delta_raw)
    require(isinstance(source, bytes) and sha(source) == SOURCE
            and delta['schema'] == 'heptatorch.index-select-signbit.source-delta.v1'
            and delta['source_sha256'] == SOURCE and delta['parent_source_sha256'] == PARENT_SOURCE
            and delta['functions'] == FUNCTIONS and delta['parent_functions'] == PARENT_FUNCTIONS
            and tuple(delta['changed_functions']) == CHANGED
            and tuple(delta['added_functions']) == ADDED
            and delta['unchanged_parent_functions'] == 771
            and set(delta['bodies']) == set(CHANGED)
            and set(delta['added_bodies']) == set(ADDED),
            'exact778 successor and exact775 parent with four changed and three appended functions')
    actual = functions(source, FUNCTIONS)
    require(tuple(actual)[-len(ADDED):] == ADDED, 'exact appended native index-select helpers')
    suffix = b''.join(actual[name] for name in ADDED)
    require(source.endswith(suffix), 'complete appended helper suffix')
    for name in ADDED:
        require(delta['added_bodies'][name] == dict(bytes=len(actual[name]), sha256=sha(actual[name])),
                'complete native added function identity: ' + name)
    restored = source[:-len(suffix)]
    for name in CHANGED:
        row, after = delta['bodies'][name], actual[name]
        before = row['before'].encode('utf-8')
        require(before.startswith(('fn ' + name + '(').encode()) and before.endswith(b'end\n')
                and sha(before) == row['before_sha256'] and sha(after) == row['after_sha256']
                and len(after) == row['after_bytes'] and restored.count(after) == 1,
                'complete changed native function identity: ' + name)
        restored = restored.replace(after, before, 1)
    parent = functions(restored, PARENT_FUNCTIONS)
    require(sha(restored) == PARENT_SOURCE and len(restored) == delta['parent_bytes']
            and len(source) == delta['source_bytes']
            and tuple(actual)[:PARENT_FUNCTIONS] == tuple(parent)
            and tuple(name for name in parent if actual[name] != parent[name]) == CHANGED
            and sum(actual[name] == parent[name] for name in parent) == 771,
            'every original source byte, definition order and all771 unchanged functions')
    require(actual['typed_index_select_previous'].replace(
                b'fn typed_index_select_previous(', b'fn index_select(', 1) == parent['index_select'],
            'the entire previous F32 index_select body remains in the actual new image')
    require(b'{"function":"typed_index_select_previous"}' in actual['index_select']
            and b'{"function":"typed_index_select"}' in actual['index_select'],
            'actual index_select dispatcher retains the old function and calls the new native path')
    forward = restored
    for name in CHANGED:
        require(forward.count(parent[name]) == 1, 'unique complete forward body')
        forward = forward.replace(parent[name], actual[name], 1)
    require(forward + suffix == source, 'complete778 forward round trip')
    retained = ('binary', 'unary', 'predicate_tensor', 'to_command', 'command',
                'typed_descriptor', 'integer_pointwise_layout', 'tget', 'taddr',
                'typed_store', 'index_select_source', 'backward')
    require(all(actual[name] == parent[name] for name in retained),
            'complete arithmetic, predicate allocation, owner, indexing AD and backward bodies retained')
    return restored, dict(
        schema='heptatorch.index-signbit778-to-relu775.complete-source-inverse.v1',
        source_sha256=SOURCE, parent_source_sha256=PARENT_SOURCE,
        functions=FUNCTIONS, parent_functions=PARENT_FUNCTIONS,
        changed_functions=list(CHANGED), appended_functions=list(ADDED),
        unchanged_parent_functions=771, delta_sha256=sha(delta_raw),
        original_f32_index_select_function_retained=True,
        original_arithmetic_and_backward_functions_retained=True,
        admitted_new_opcode=126, opcode125_remains_rejected=True,
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        newly_accepted_original_requests_require_explicit_numerical_successors=True,
        actual_executable_replaced=False, original_cases_or_expected_values_modified=False,
        native_executed=False, Torch_executed=False, runtime_qualification_transferred=False)
