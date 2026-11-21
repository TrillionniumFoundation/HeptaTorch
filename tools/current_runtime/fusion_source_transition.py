"""Reverse the complete current source through the Half and storage parents.

Only source bytes are transformed. Tests execute the actual current image.
This invokes no native tool or runtime and transfers no historical PASS.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SOURCE = '51820849f7becec549be15babace00f00687601ae7f53bcee0e636ca02f721f9'
HALF_SOURCE = 'bf3641b1a9061c6f92b8cbeef75adfac89e82ee16d32a9f650cb938d9e918386'
CURRENT_SOURCE = '4722c53b248780534a846a472784a565c4454ab738db75d655a7801b6623f07f'
PREDICATE_INVERSE = '71244cd75fb41ce07876d7f5699b66ae43c73adc5dd5d09fb2da0dec9c8c7439'
PARENT_SOURCE = 'ba96abc05717643eaf4f2056325089d9a51f5a2e9838f89473e9a4e76e46fd99'
DELTA_SHA = '1d47ee1f02e33cc29b4277afa0019dc55c7519a39b23022b0e5e97217bc9de53'


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


def restore_storage_source(source, bind):
    """Return the exact700 parent for old source proofs, plus the outer proof."""
    bind(Path(__file__).resolve())
    require(sha(source) == SOURCE, 'only the exact admitted736 source')
    raw = bind(HERE/'fusion_source_transition.json',DELTA_SHA)
    delta = json.loads(raw)
    require(delta['schema'] == 'heptatorch.fusion736-to-cat700.complete-source-delta.v1'
        and delta['source_sha256'] == SOURCE and delta['parent_source_sha256'] == PARENT_SOURCE
        and delta['functions'] == 736 and delta['parent_functions'] == 700
        and delta['unchanged_parent_functions'] == 682
        and len(delta['changed_functions']) == 18 and len(delta['added_functions']) == 36,
        'exact736/700 function delta')
    actual = functions(source,736)
    require(source.split(b'\n',1)[0].decode() == delta['header']
        and len(source) == delta['source_bytes'], 'exact canonical source header and extent')
    parent_order, added = delta['parent_function_order'],delta['added_functions']
    changed = delta['changed_functions']
    require(len(parent_order) == len(set(parent_order)) == 700
        and len(added) == len(set(added)) == 36 and not set(parent_order)&set(added)
        and list(actual) == parent_order+added
        and len(changed) == len(set(changed)) == 18 and set(changed) <= set(parent_order)
        and set(delta['bodies']) == set(changed) and set(delta['added_bodies']) == set(added),
        'exact declaration order and finite change sets')
    suffix = b''.join(actual[name] for name in added)
    require(source.endswith(suffix) and len(suffix) == delta['added_suffix_bytes']
        and sha(suffix) == delta['added_suffix_sha256'], 'complete appended36 suffix')
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
    require(forward+suffix == source, 'complete736 source round trip')
    proof = dict(schema='heptatorch.fusion736-to-cat700.complete-source-inverse.v1',
        source_sha256=SOURCE,parent_source_sha256=PARENT_SOURCE,functions=736,parent_functions=700,
        changed_functions=changed,unchanged_parent_functions=682,added_functions=added,
        delta_sha256=DELTA_SHA,whole_source_inverse_exact=True,whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        actual_executable_replaced=False,source_equivalence_only=True,
        original_cases_or_expected_values_modified=False,runtime_qualification_transferred=False)
    return restored,proof


def restore_half_source(source, bind):
    """Compose exact Half-logical and original736 inverses, without execution."""
    bind(Path(__file__).resolve())
    require(sha(source) == HALF_SOURCE, 'only the exact admitted Half-logical source')
    path = HERE / 'half_logical_source_transition.py'
    bind(path, 'e0a2eac589ddfe8b09794fb55c569a73d7d7802888130646f59d33adca266f66')
    spec = importlib.util.spec_from_file_location('half_logical_complete_outer_inverse', path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    storage_source, half_inverse = extension.restore_source(source, bind)
    require(extension.SOURCE == HALF_SOURCE and extension.PARENT_SOURCE == SOURCE
            and sha(storage_source) == SOURCE, 'exact Half-logical to storage736 link')
    restored, storage_inverse = restore_storage_source(storage_source, bind)
    proof = dict(
        schema='heptatorch.half-logical736-to-cat700.complete-source-inverse.v1',
        source_sha256=HALF_SOURCE, intermediate_source_sha256=SOURCE,
        parent_source_sha256=PARENT_SOURCE, functions=736, parent_functions=700,
        half_logical_inverse=half_inverse, storage_fusion_inverse=storage_inverse,
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        actual_executable_replaced=False, source_equivalence_only=True,
        original_cases_or_expected_values_modified=False,
        runtime_qualification_transferred=False,
    )
    return restored, proof


def predicate_parent(source, bind):
    """Check the new opcode routing and recover the exact Half parent bytes."""
    bind(Path(__file__).resolve())
    require(sha(source) == CURRENT_SOURCE, 'only the exact admitted predicate source')
    arithmetic_path = HERE / 'public_f64_arithmetic_source_transition.py'
    bind(arithmetic_path, 'bfab0304ac467af58a3590a55ffd376b1bcc8cac0aefd4d4c80d08f344d9e1ec')
    arithmetic_spec = importlib.util.spec_from_file_location('public_f64_arithmetic_complete_outer_inverse', arithmetic_path)
    arithmetic = importlib.util.module_from_spec(arithmetic_spec)
    arithmetic_spec.loader.exec_module(arithmetic)
    source, arithmetic_inverse = arithmetic.restore_source(source, bind)
    require(arithmetic.SOURCE == CURRENT_SOURCE and sha(source) == arithmetic.PARENT_SOURCE
            and arithmetic.FUNCTIONS == 773 and arithmetic.PARENT_FUNCTIONS == 771,
            'exact mixed public F64773 to Half771 complete source relation')
    div_path = HERE / 'half_div_source_transition.py'
    bind(div_path)
    div_spec = importlib.util.spec_from_file_location('half_div_complete_outer_inverse', div_path)
    division = importlib.util.module_from_spec(div_spec)
    div_spec.loader.exec_module(division)
    source, div_inverse = division.restore_source(source, bind)
    require(division.SOURCE == arithmetic.PARENT_SOURCE and sha(source) == division.PARENT_SOURCE,
            'exact Half771 division to wide767 complete source relation')
    wide_path = HERE / 'wide_neg_source_transition.py'
    bind(wide_path)
    wide_spec = importlib.util.spec_from_file_location('wide_neg_complete_outer_inverse', wide_path)
    wide = importlib.util.module_from_spec(wide_spec)
    wide_spec.loader.exec_module(wide)
    source, wide_inverse = wide.restore_source(source, bind)
    require(wide.SOURCE == division.PARENT_SOURCE and sha(source) == wide.PARENT_SOURCE,
            'exact wide767 negation to Half765 complete source relation')
    addsub_path = HERE / 'half_addsub_source_transition.py'
    bind(addsub_path)
    addsub_spec = importlib.util.spec_from_file_location('half_addsub_complete_outer_inverse', addsub_path)
    addsub = importlib.util.module_from_spec(addsub_spec)
    addsub_spec.loader.exec_module(addsub)
    source, addsub_inverse = addsub.restore_source(source, bind)
    require(addsub.SOURCE == wide.PARENT_SOURCE and sha(source) == addsub.PARENT_SOURCE,
            'exact Half765 add/sub to Half761 complete source relation')
    neg_path = HERE / 'half_neg_source_transition.py'
    bind(neg_path)
    neg_spec = importlib.util.spec_from_file_location('half_neg_complete_outer_inverse', neg_path)
    neg = importlib.util.module_from_spec(neg_spec)
    neg_spec.loader.exec_module(neg)
    source, neg_inverse = neg.restore_source(source, bind)
    require(neg.SOURCE == addsub.PARENT_SOURCE and sha(source) == neg.PARENT_SOURCE,
            'exact Half761 negation to Half758 complete source relation')
    mul_path = HERE / 'half_mul_source_transition.py'
    bind(mul_path)
    mul_spec = importlib.util.spec_from_file_location('half_mul_complete_outer_inverse', mul_path)
    mul = importlib.util.module_from_spec(mul_spec)
    mul_spec.loader.exec_module(mul)
    source, mul_inverse = mul.restore_source(source, bind)
    require(mul.SOURCE == neg.PARENT_SOURCE and sha(source) == mul.PARENT_SOURCE,
            'exact Half758 to BF16742 complete source relation')
    bf16_path = HERE / 'bf16_source_transition.py'
    bind(bf16_path)
    bf16_spec = importlib.util.spec_from_file_location('bf16_copy_complete_outer_inverse', bf16_path)
    bf16 = importlib.util.module_from_spec(bf16_spec)
    bf16_spec.loader.exec_module(bf16)
    source, bf16_inverse = bf16.restore_source(source, bind)
    require(bf16.SOURCE == mul.PARENT_SOURCE and sha(source) == bf16.PARENT_SOURCE,
            'exact BF16742 to predicate740 source relation')
    path = HERE / 'predicate_source_transition.py'
    bind(path, PREDICATE_INVERSE)
    spec = importlib.util.spec_from_file_location('current_predicate_source_inverse', path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    previous, proof = extension.restore_source(source, bind)
    require(extension.SOURCE == bf16.PARENT_SOURCE and extension.PARENT_SOURCE == HALF_SOURCE
            and sha(previous) == HALF_SOURCE,
            'complete predicate-to-Half source relation')
    return previous, proof, bf16_inverse, mul_inverse, neg_inverse, addsub_inverse, wide_inverse, div_inverse, arithmetic_inverse


def restore_source(source, bind):
    """Compose all exact source inverses; preserve the actual current runtime."""
    previous, predicate_inverse, bf16_inverse, mul_inverse, neg_inverse, addsub_inverse, wide_inverse, div_inverse, arithmetic_inverse = predicate_parent(source, bind)
    restored, half_inverse = restore_half_source(previous, bind)
    return restored, dict(
        schema='heptatorch.current773-to-cat700.complete-source-inverse.v1',
        source_sha256=CURRENT_SOURCE, intermediate_source_sha256=HALF_SOURCE,
        parent_source_sha256=PARENT_SOURCE, functions=773, parent_functions=700,
        predicate_source_inverse=predicate_inverse,
        bf16_source_inverse=bf16_inverse,
        half_mul_source_inverse=mul_inverse,
        half_neg_source_inverse=neg_inverse,
        half_addsub_source_inverse=addsub_inverse,
        wide_neg_source_inverse=wide_inverse,
        half_div_source_inverse=div_inverse,
        public_f64_arithmetic_source_inverse=arithmetic_inverse,
        half_and_storage_source_inverse=half_inverse,
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        actual_executable_replaced=False, source_equivalence_only=True,
        original_cases_or_expected_values_modified=False,
        runtime_qualification_transferred=False,
    )
