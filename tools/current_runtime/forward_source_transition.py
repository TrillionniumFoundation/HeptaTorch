"""Recover every original781 source byte from the exact native forward repair.

This is a source identity proof. It executes no runtime or reference and does
not transfer qualification from an earlier source or numerical observation.
"""
import hashlib
import json
from pathlib import Path
import re

SOURCE = '0061acfaa8e886885053e8f539cf08187be1c1e10265c19e52df07435a71594b'
PARENT_SOURCE = 'ee214f05b56401ea17a5f55e792aaccf73447974ba0b7e0416990439bff6a8ff'
FUNCTIONS = 797
PARENT_FUNCTIONS = 781
SOURCE_BYTES = 2592583
PARENT_BYTES = 2543906
CHANGED = ('linear', 'reduce_axis', 'softmax', 'layernorm', 'batchnorm_eval', 'conv2d', 'pool2d', 'matmul', 'tensor_join')
ADDED = ('matmul_forward_index', 'matmul_forward_point', 'softmax_forward_group', 'axis_source_forward', 'conv_term_forward', 'conv_no_grad_forward', 'pool_point_forward', 'pool_no_grad_forward', 'matmul_output_version', 'concat_forward_target', 'concat_forward_part', 'concat_forward_copy', 'concat_forward_layout', 'concat_forward', 'linear_bias_output_version', 'normalization_output_layout')
DELTA_SHA256 = '43ec26417cbb9d68bc2692b7ded2a540bd8576b6d98c0bb15648b7ec1d58ea99'
HEADER = b'hia-asm.v1 hepta.hia.program.v8 heptatorch_runtime\n'
PROTECTED = ('record', 'gather', 'indexed_add', 'gather_source', 'indexed_add_forward',
             'indexed_add_location', 'matmul_index', 'matmul_point', 'softmax_group',
             'axis_source', 'conv_term', 'pool_point', 'backward')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def functions(raw, count):
    matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\([^\n]*\n.*?^end\n', raw, re.M | re.S))
    bodies = {match[1].decode('ascii'): match[0] for match in matches}
    require(len(matches) == len(bodies) == count, 'complete unique native function inventory')
    require(raw == HEADER + b''.join(bodies.values()), 'complete source envelope and ordered function bytes')
    return bodies


def restore_source(source, bind):
    bind(Path(__file__).resolve())
    delta_raw = bind(Path(__file__).with_name('forward_source_delta.json'), DELTA_SHA256)
    require(sha(delta_raw) == DELTA_SHA256, 'exact complete forward delta bytes')
    delta = json.loads(delta_raw)
    require(type(source) is bytes and sha(source) == SOURCE and len(source) == SOURCE_BYTES,
            'exact complete current native source bytes')
    require(delta['schema'] == 'heptatorch.native-forward.source-delta.v1'
            and delta['source_sha256'] == SOURCE and delta['parent_source_sha256'] == PARENT_SOURCE
            and type(delta['functions']) is int and delta['functions'] == FUNCTIONS
            and type(delta['parent_functions']) is int and delta['parent_functions'] == PARENT_FUNCTIONS
            and delta['source_bytes'] == SOURCE_BYTES and delta['parent_bytes'] == PARENT_BYTES
            and tuple(delta['changed_functions']) == CHANGED and tuple(delta['added_functions']) == ADDED
            and delta['unchanged_parent_functions'] == PARENT_FUNCTIONS-len(CHANGED)
            and set(delta['bodies']) == set(CHANGED) and set(delta['added_bodies']) == set(ADDED),
            'fixed complete source identities and finite declared change sets')
    actual = functions(source, FUNCTIONS)
    order = delta['parent_function_order']
    require(len(order) == len(set(order)) == PARENT_FUNCTIONS
            and list(actual) == order + list(ADDED)
            and len(CHANGED) == len(set(CHANGED)) and set(CHANGED) <= set(order)
            and len(ADDED) == len(set(ADDED)) and not set(ADDED) & set(order),
            'every original definition remains in its exact original position')
    suffix = b''.join(actual[name] for name in ADDED)
    require(source.endswith(suffix) and len(suffix) == delta['added_suffix_bytes']
            and sha(suffix) == delta['added_suffix_sha256'], 'complete exact appended helper suffix')
    for name in ADDED:
        require(delta['added_bodies'][name] == dict(bytes=len(actual[name]),sha256=sha(actual[name])),
                'complete exact appended native body: '+name)
    restored = source[:-len(suffix)]
    originals = {}
    for name in CHANGED:
        row = delta['bodies'][name]
        before, after = row['before_body'].encode('utf8'), actual[name]
        require(before.startswith(('fn '+name+'(').encode('ascii')) and before.endswith(b'end\n')
                and len(before) == row['before_bytes'] and sha(before) == row['before_sha256']
                and len(after) == row['after_bytes'] and sha(after) == row['after_sha256']
                and before != after and restored.count(after) == 1,
                'complete original and current native body: '+name)
        originals[name] = before
        restored = restored.replace(after,before,1)
    parent = functions(restored,PARENT_FUNCTIONS)
    require(sha(restored) == PARENT_SOURCE and len(restored) == PARENT_BYTES
            and list(parent) == order
            and tuple(name for name in parent if actual[name] != parent[name]) == CHANGED
            and sum(actual[name] == parent[name] for name in parent) == PARENT_FUNCTIONS-len(CHANGED),
            'every original byte and all undeclared parent bodies are constrained by the whole parent SHA')
    require(delta['unchanged_parent_body_sha256'] ==
            {name:sha(parent[name]) for name in parent if name not in CHANGED},
            'all unchanged complete original function identities')
    require(all(name in parent and actual[name] == parent[name] for name in PROTECTED)
            and all(actual[name] == parent[name] for name in parent if 'backward' in name),
            'original indexed dispatch, AD numeric helpers, record and every named backward body retained')
    forward = restored
    for name in CHANGED:
        require(forward.count(originals[name]) == 1, 'unique whole-body forward reconstruction')
        forward = forward.replace(originals[name],actual[name],1)
    require(forward+suffix == source, 'complete byte-for-byte forward reconstruction')
    return restored,dict(schema='heptatorch.native-forward-to-indexed781.complete-source-inverse.v1',
        source_sha256=SOURCE,parent_source_sha256=PARENT_SOURCE,functions=FUNCTIONS,parent_functions=PARENT_FUNCTIONS,
        source_bytes=SOURCE_BYTES,parent_bytes=PARENT_BYTES,changed_functions=list(CHANGED),appended_functions=list(ADDED),
        unchanged_parent_functions=PARENT_FUNCTIONS-len(CHANGED),delta_sha256=DELTA_SHA256,
        whole_source_inverse_exact=True,whole_round_trip_exact=True,complete_source_definition_order_preserved=True,
        source_envelope_retained=True,every_other_byte_constrained_by_complete_parent_sha256=True,
        protected_original_function_bodies=list(PROTECTED),original_indexed_dispatch_unchanged=True,
        original_AD_numeric_helpers_unchanged=True,original_backward_functions_unchanged=True,
        actual_executable_replaced=False,original_cases_or_expected_values_modified=False,
        native_executed=False,Torch_executed=False,runtime_qualification_transferred=False)
