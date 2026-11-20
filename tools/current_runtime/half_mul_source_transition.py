"""Restore exact742 source before native Half tensor multiplication.

Only source bytes are reconstructed; runtime qualification is never inherited.
Historical recipes and evidence remain unchanged. The22 formerly refused
multiplications require new actual-Torch successor evidence; all other original
predicates execute against the current image.
"""
import hashlib
import json
from pathlib import Path
import re

SOURCE = '027d56d8d8103a48a158d77d6ac750b6eb5d1992a49dd6479e961fe085c51334'
IMAGE = '21c062d0f87f1a6eeeb132207930669b0fa0619955beccf685a4ffbf34b1f467'
MANIFEST = '2d387831207047472596d98710b936e4a774da4d897c5c70dadd9dc5a9b5655d'
PARENT_SOURCE = 'abbc84b8eca16670e72b8872d1d7ab479f8a1d44f7b2b65deea1649f7682b2c7'
FUNCTIONS, PARENT_FUNCTIONS = 758, 742
CHANGED = ('binary', 'command_argument')
ADDED = ('half_mul_previous_binary', 'half_mul_f32', 'half_mul_f64', 'half_mul_to_f32', 'half_mul_cast', 'half_mul_result_dtype', 'half_mul_promote', 'half_mul_is_scalar', 'half_mul_element', 'half_mul_tensor', 'half_mul_inner_extent', 'half_mul_nan_word', 'half_mul_copy_value', 'half_mul_inner_stride', 'half_mul_float_priority', 'half_mul_priority')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def functions(source, count):
    matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', source, re.M | re.S))
    result = {match[1].decode('ascii'): match[0] for match in matches}
    require(len(result) == len(matches) == count, 'complete unique Half multiplication source census')
    require(source == source.split(b'\n', 1)[0] + b'\n' + b''.join(result.values()),
            'complete canonical source envelope and declaration order')
    return result


def restore_source(source, bind):
    bind(Path(__file__).resolve())
    require(isinstance(source, bytes) and sha(source) == SOURCE,
            'exact admitted758 Half multiplication source')
    delta_raw = bind(Path(__file__).with_name('half_mul_source_delta.json'))
    delta = json.loads(delta_raw)
    require(delta['schema'] == 'heptatorch.half-mul-source-delta.v1'
            and delta['source_sha256'] == SOURCE and delta['parent_source_sha256'] == PARENT_SOURCE
            and delta['functions'] == FUNCTIONS and delta['parent_functions'] == PARENT_FUNCTIONS
            and tuple(delta['changed_functions']) == CHANGED
            and tuple(delta['appended_functions']) == ADDED,
            'exact two changed bodies and sixteen appended helpers')
    actual = functions(source, FUNCTIONS)
    require(tuple(actual)[-len(ADDED):] == ADDED, 'only the exact sixteen native helpers are appended')
    suffix = b''.join(actual[name] for name in ADDED)
    require(source.endswith(suffix) and sha(suffix) == delta['helper_sha256'],
            'complete Half multiplication helper suffix identity')
    restored = source[:-len(suffix)]
    originals = {}
    for name in CHANGED:
        row, after = delta['changed_functions'][name], actual[name]
        before = row['before'].encode('utf-8')
        require(before.startswith(('fn ' + name + '(').encode()) and before.endswith(b'end\n')
                and sha(before) == row['before_sha256'] and sha(after) == row['after_sha256']
                and before != after and restored.count(after) == 1,
                'exact complete old and current Half multiplication body: ' + name)
        originals[name] = before
        restored = restored.replace(after, before, 1)
    parent = functions(restored, PARENT_FUNCTIONS)
    require(sha(restored) == PARENT_SOURCE and list(actual)[:PARENT_FUNCTIONS] == list(parent)
            and tuple(name for name in parent if actual[name] != parent[name]) == CHANGED,
            'every reconstructed742 byte and all740 unchanged bodies match the predecessor')
    require(actual['half_mul_previous_binary'] == parent['binary'].replace(
                b'fn binary(', b'fn half_mul_previous_binary(', 1),
            'the complete predecessor binary remains byte identical except its function name')
    before_argument = parent['command_argument']
    old_guard = b'  v140:bool = bool.and(v138,v139)'
    new_guard = (b'  half_operation:bool = bool.and(v139,v125)\n'
                 b'  v140:bool = bool.and(v138,half_operation)')
    require(before_argument.count(old_guard) == 1 and actual['command_argument'] ==
            before_argument.replace(old_guard,new_guard,1),
            'only command2 is added to the public Half opcode admission guard')
    forward = restored
    for name in CHANGED:
        require(forward.count(originals[name]) == 1, 'unique complete forward body')
        forward = forward.replace(originals[name], actual[name], 1)
    require(forward + suffix == source, 'complete source758 forward round trip')
    return restored, dict(
        schema='heptatorch.half-mul758-to-bf16-copy742.complete-source-inverse.v1',
        source_sha256=SOURCE, parent_source_sha256=PARENT_SOURCE,
        functions=FUNCTIONS, parent_functions=PARENT_FUNCTIONS,
        changed_functions=list(CHANGED), added_functions=list(ADDED), removed_functions=[],
        unchanged_parent_functions=740, delta_sha256=sha(delta_raw),
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        actual_executable_replaced=False, historical_cases_or_expected_values_modified=False,
        newly_supported_request_outcomes_require_separate_successor_evidence=True,
        predecessor_binary_body_preserved_exactly=True,
        runtime_qualification_transferred=False,
        changed_bodies={name: dict(before_sha256=sha(parent[name]), after_sha256=sha(actual[name]))
                        for name in CHANGED},
        added_bodies={name: dict(bytes=len(actual[name]), sha256=sha(actual[name])) for name in ADDED})
