"""Restore exact740 source before the native BF16 copy correction.

Only source bytes are reconstructed. The current image still executes every
original test; the predecessor's outputs and test expectations are unchanged.
"""
import hashlib
import json
from pathlib import Path
import re

SOURCE = 'abbc84b8eca16670e72b8872d1d7ab479f8a1d44f7b2b65deea1649f7682b2c7'
IMAGE = '606f69aaecedbedc7ed6ad8f033275e0b64697f70632036d180106d315243c32'
MANIFEST = 'a750f6e388c4339c29ff299aca0dabda9fa8b0ff7cc69d5bfa45f69cdd86e009'
PARENT_SOURCE = '24011e3c861e3b0e77c6a25c07116375cded2225c1f05f42cd84aa1e4a139215'
FUNCTIONS, PARENT_FUNCTIONS = 742, 740
CHANGED = ('half_convert_value', 'half_copy_tensor')
ADDED = ('half_bf16_nan_word', 'half_narrow_bf16')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def functions(source, count):
    matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', source, re.M | re.S))
    result = {match[1].decode('ascii'): match[0] for match in matches}
    require(len(result) == len(matches) == count, 'complete unique BF16 source census')
    require(source == source.split(b'\n', 1)[0] + b'\n' + b''.join(result.values()),
            'complete canonical source envelope and declaration order')
    return result


def restore_source(source, bind):
    bind(Path(__file__).resolve())
    require(isinstance(source, bytes) and sha(source) == SOURCE,
            'exact admitted742 BF16 copy source')
    delta_raw = bind(Path(__file__).with_name('bf16_source_delta.json'))
    delta = json.loads(delta_raw)
    require(delta['schema'] == 'heptatorch.bf16-copy-source-delta.v1'
            and delta['source_sha256'] == SOURCE and delta['parent_source_sha256'] == PARENT_SOURCE
            and delta['functions'] == FUNCTIONS and delta['parent_functions'] == PARENT_FUNCTIONS
            and tuple(delta['changed_functions']) == CHANGED
            and tuple(delta['appended_functions']) == ADDED,
            'exact two changed bodies and two appended helpers')
    actual = functions(source, FUNCTIONS)
    require(tuple(actual)[-len(ADDED):] == ADDED, 'only the exact two native helpers are appended')
    suffix = b''.join(actual[name] for name in ADDED)
    require(source.endswith(suffix) and sha(suffix) == delta['helper_sha256'],
            'complete BF16 helper suffix identity')
    restored = source[:-len(suffix)]
    originals = {}
    for name in CHANGED:
        row, after = delta['changed_functions'][name], actual[name]
        before = row['before'].encode('utf-8')
        require(before.startswith(('fn ' + name + '(').encode()) and before.endswith(b'end\n')
                and sha(before) == row['before_sha256'] and sha(after) == row['after_sha256']
                and before != after and restored.count(after) == 1,
                'exact complete old and current BF16 body: ' + name)
        originals[name] = before
        restored = restored.replace(after, before, 1)
    parent = functions(restored, PARENT_FUNCTIONS)
    require(sha(restored) == PARENT_SOURCE and list(actual)[:PARENT_FUNCTIONS] == list(parent)
            and tuple(name for name in parent if actual[name] != parent[name]) == CHANGED,
            'every reconstructed740 byte and all738 unchanged bodies match the predecessor')
    forward = restored
    for name in CHANGED:
        require(forward.count(originals[name]) == 1, 'unique complete forward body')
        forward = forward.replace(originals[name], actual[name], 1)
    require(forward + suffix == source, 'complete source742 forward round trip')
    return restored, dict(
        schema='heptatorch.bf16-copy742-to-predicates740.complete-source-inverse.v1',
        source_sha256=SOURCE, parent_source_sha256=PARENT_SOURCE,
        functions=FUNCTIONS, parent_functions=PARENT_FUNCTIONS,
        changed_functions=list(CHANGED), added_functions=list(ADDED), removed_functions=[],
        unchanged_parent_functions=738, delta_sha256=sha(delta_raw),
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        actual_executable_replaced=False, original_cases_or_expected_values_modified=False,
        runtime_qualification_transferred=False,
        changed_bodies={name: dict(before_sha256=sha(parent[name]), after_sha256=sha(actual[name]))
                        for name in CHANGED},
        added_bodies={name: dict(bytes=len(actual[name]), sha256=sha(actual[name])) for name in ADDED})
