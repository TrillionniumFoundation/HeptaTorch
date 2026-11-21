"""Recover the exact775 predecessor of the current781 native indexing image.

The two complete source inverses preserve their own identities and scopes.
Only source bytes are reconstructed; no old execution qualification transfers.
"""

import hashlib
import importlib.util
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


def _restore_index_signbit_source(source, bind):
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


INDEXED_SOURCE = 'ee214f05b56401ea17a5f55e792aaccf73447974ba0b7e0416990439bff6a8ff'
INDEXED_FUNCTIONS = 781
INDEXED_CHANGED = ('entry', 'index_select', 'gather', 'indexed_add',
                   'predicate_bits', 'predicate_command')
INDEXED_ADDED = ADDED + ('indexed_gather_forward_source', 'indexed_add_forward_inputs',
                         'indexed_add_forward_location')


def _restore_indexed_source(source, bind):
    """Bind both inverse steps and prove the complete actual source relation."""
    bind(Path(__file__).resolve())
    # The new indexed group explicitly verifies the first inverse itself before
    # supplying the exact778 intermediate to this immutable second inverse.
    if sha(source) == SOURCE:
        return _restore_index_signbit_source(source, bind)
    require(sha(source) == INDEXED_SOURCE, 'exact current781 source only')
    path = Path(__file__).with_name('indexed_forward_source_transition.py')
    bind(path)
    spec = importlib.util.spec_from_file_location('complete_indexed_forward_inverse', path)
    indexed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(indexed)
    intermediate, indexed_proof = indexed.restore_source(source, bind)
    require(sha(intermediate) == SOURCE and indexed_proof['source_sha256'] == INDEXED_SOURCE
            and indexed_proof['parent_source_sha256'] == SOURCE
            and indexed_proof['functions'] == INDEXED_FUNCTIONS
            and indexed_proof['parent_functions'] == FUNCTIONS
            and indexed_proof['unchanged_parent_functions'] == 776
            and indexed_proof['changed_functions'] == ['gather', 'indexed_add']
            and indexed_proof['appended_functions'] == list(INDEXED_ADDED[len(ADDED):])
            and indexed_proof['whole_source_inverse_exact'] is True
            and indexed_proof['original_requires_grad_dispatch_preserved'] is True
            and indexed_proof['original_backward_functions_unchanged'] is True,
            'complete781-to778 source inverse and unchanged autograd dispatch')
    parent, predecessor = _restore_index_signbit_source(intermediate, bind)
    actual, previous = functions(source, INDEXED_FUNCTIONS), functions(parent, PARENT_FUNCTIONS)
    require(tuple(actual)[:PARENT_FUNCTIONS] == tuple(previous)
            and tuple(actual)[PARENT_FUNCTIONS:] == INDEXED_ADDED
            and tuple(name for name in previous if actual[name] != previous[name]) == INDEXED_CHANGED
            and sum(actual[name] == previous[name] for name in previous) == 769,
            'all781 definitions, six changed parent bodies and all769 unchanged parent bodies')
    suffix = b''.join(actual[name] for name in INDEXED_ADDED)
    forward = parent
    for name in INDEXED_CHANGED:
        require(forward.count(previous[name]) == 1, 'one complete parent body: ' + name)
        forward = forward.replace(previous[name], actual[name], 1)
    require(forward + suffix == source, 'complete781-to775 inverse and exact forward round trip')
    retained = ('gather_source', 'indexed_add_forward', 'indexed_add_location', 'backward',
                'index_select_source', 'binary', 'unary', 'predicate_tensor', 'to_command',
                'command', 'typed_descriptor', 'integer_pointwise_layout', 'tget', 'taddr',
                'typed_store')
    require(all(actual[name] == previous[name] for name in retained),
            'complete original arithmetic, owner, requires-grad indexing and backward bodies retained')
    return parent, dict(predecessor,
        schema='heptatorch.indexed781-to-relu775.complete-source-inverse.v1',
        source_sha256=INDEXED_SOURCE, functions=INDEXED_FUNCTIONS,
        changed_functions=list(INDEXED_CHANGED), appended_functions=list(INDEXED_ADDED),
        unchanged_parent_functions=769,
        indexed_forward_source_inverse=indexed_proof,
        index_signbit_predecessor_source_inverse=predecessor,
        original_requires_grad_dispatch_preserved=True,
        original_backward_functions_unchanged=True,
        complete_source_definition_order_preserved=True,
        whole_source_inverse_exact=True, whole_round_trip_exact=True)


CURRENT_SOURCE = 'c73a7cb474981056404ac96bc214979d57aaa9a86e6474b1c140c3d98eb05779'
CURRENT_FUNCTIONS = 797
CURRENT_CHANGED = ('entry', 'linear', 'reduce_axis', 'softmax', 'layernorm', 'batchnorm_eval', 'conv2d', 'pool2d', 'matmul', 'index_select', 'gather', 'tensor_join', 'indexed_add', 'predicate_bits', 'predicate_command')
CURRENT_ADDED = ('typed_index_select_previous', 'typed_index_select', 'typed_index_select_offset', 'indexed_gather_forward_source', 'indexed_add_forward_inputs', 'indexed_add_forward_location', 'matmul_forward_index', 'matmul_forward_point', 'softmax_forward_group', 'axis_source_forward', 'conv_term_forward', 'conv_no_grad_forward', 'pool_point_forward', 'pool_no_grad_forward', 'matmul_output_version', 'concat_forward_target', 'concat_forward_part', 'concat_forward_copy', 'concat_forward_layout', 'concat_forward', 'linear_bias_output_version', 'normalization_output_layout')
FORWARD_INVERSE_SHA256 = 'c8fca08e2c2756873c331cef92003f94296e36cbe2e52014bfd8bc6942d7a1b6'


def restore_source(source, bind):
    """Compose the forward repair with both complete original indexing inverses."""
    bind(Path(__file__).resolve())
    if sha(source) in (SOURCE, INDEXED_SOURCE):
        return _restore_indexed_source(source, bind)
    require(sha(source) == CURRENT_SOURCE, 'exact current797 forward source only')
    path = Path(__file__).with_name('forward_source_transition.py')
    raw = bind(path, FORWARD_INVERSE_SHA256)
    require(sha(raw) == FORWARD_INVERSE_SHA256, 'complete forward inverse helper identity')
    # Compile exactly the bytes just bound; never reuse a stale timestamp pyc.
    import types
    extension = types.ModuleType('current_complete_forward_inverse')
    extension.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), extension.__dict__)
    intermediate, forward_proof = extension.restore_source(source, bind)
    require(sha(intermediate) == INDEXED_SOURCE
            and forward_proof['source_sha256'] == CURRENT_SOURCE
            and forward_proof['parent_source_sha256'] == INDEXED_SOURCE
            and forward_proof['functions'] == CURRENT_FUNCTIONS
            and forward_proof['parent_functions'] == INDEXED_FUNCTIONS
            and forward_proof['unchanged_parent_functions'] == 772
            and forward_proof['changed_functions'] == ['linear', 'reduce_axis', 'softmax', 'layernorm', 'batchnorm_eval', 'conv2d', 'pool2d', 'matmul', 'tensor_join']
            and forward_proof['appended_functions'] == ['matmul_forward_index', 'matmul_forward_point', 'softmax_forward_group', 'axis_source_forward', 'conv_term_forward', 'conv_no_grad_forward', 'pool_point_forward', 'pool_no_grad_forward', 'matmul_output_version', 'concat_forward_target', 'concat_forward_part', 'concat_forward_copy', 'concat_forward_layout', 'concat_forward', 'linear_bias_output_version', 'normalization_output_layout']
            and forward_proof['whole_source_inverse_exact'] is True
            and forward_proof['whole_round_trip_exact'] is True
            and forward_proof['original_indexed_dispatch_unchanged'] is True
            and forward_proof['original_AD_numeric_helpers_unchanged'] is True
            and forward_proof['original_backward_functions_unchanged'] is True,
            'complete797 to781 source inverse before all original indexing proofs')
    parent, predecessor = _restore_indexed_source(intermediate, bind)
    actual, previous = functions(source, CURRENT_FUNCTIONS), functions(parent, PARENT_FUNCTIONS)
    require(tuple(actual)[:PARENT_FUNCTIONS] == tuple(previous)
            and tuple(actual)[PARENT_FUNCTIONS:] == CURRENT_ADDED
            and tuple(name for name in previous if actual[name] != previous[name]) == CURRENT_CHANGED
            and sum(actual[name] == previous[name] for name in previous) == 760,
            'all797 definitions and every one of the760 unchanged775 parent bodies')
    suffix = b''.join(actual[name] for name in CURRENT_ADDED)
    rebuilt = parent
    for name in CURRENT_CHANGED:
        require(rebuilt.count(previous[name]) == 1, 'unique complete predecessor body: '+name)
        rebuilt = rebuilt.replace(previous[name], actual[name], 1)
    require(rebuilt + suffix == source, 'complete797 to775 source inverse and exact forward round trip')
    return parent, dict(predecessor,
        schema='heptatorch.forward797-to-relu775.complete-source-inverse.v1',
        source_sha256=CURRENT_SOURCE, functions=CURRENT_FUNCTIONS,
        changed_functions=list(CURRENT_CHANGED), appended_functions=list(CURRENT_ADDED),
        unchanged_parent_functions=760, forward_source_inverse=forward_proof,
        indexed_predecessor_source_inverse=predecessor,
        original_indexed_dispatch_unchanged=True, original_AD_numeric_helpers_unchanged=True,
        original_backward_functions_unchanged=True, complete_source_definition_order_preserved=True,
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        native_executed=False, Torch_executed=False, runtime_qualification_transferred=False)
