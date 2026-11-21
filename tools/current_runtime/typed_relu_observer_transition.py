"""Recover both complete negation observers before four typed ReLU controls.

The existing arithmetic admission keeps its original Half-negation SHA pin;
its small adapter is independently inverted as a third complete source file.
These source proofs do not execute either runtime or transfer a previous PASS.
"""
import hashlib
import json
from pathlib import Path


PARENTS = {
    'tests/half_neg/run.py': '02f0ebfd4478179bf3e805ecde751fae087d76738385b9a2ec1cc6f8f28bc51d',
    'tests/wide_neg/run.py': 'd78951f3a5e5e445943fa9b4dd303b8097472732db3d28ff3dedb95de3058822',
}
BINDINGS = {
    'tools/current_runtime/half_addsub_observer_transition.py':
        '0aa60a3db378b5ee2c2a7532e565b18086e09d24d4b6bc8b959e961fdebedc1b',
}
INPUT_PINS = {
    'tests/half_neg/__init__.py': 'f02584b81106f7440ca54bf8f59beb9a3e05feb8c1123964325d5e946b839163',
    'tests/half_neg/cases.py': '643da7cd50f0b9364444bac11a091b99b1fd6670b91979dca40e3227ee07cd69',
    'tests/half_neg/reference.py': 'b6f7ee4cd1ec107800abb650ce0aabbd63fd7edbbbb642a2290c5ae4d178a2bf',
    'tests/wide_neg/__init__.py': '01ea777820ede90061e80091e3f765f6eac9844d2c43d60dab95dcb3c48a2ba6',
    'tests/wide_neg/cases.py': '5796a0561637004f3cd5a5ac16cc0612e0ed51bd0c381c0440c2ced666ec0b84',
    'tests/wide_neg/reference.py': 'b40d811d0d3a030f45a8d086b6cf7df1e549cb1be778a6f7ea430346ec342ba3',
}
ORIGINAL_CASES = {'half_neg': 371, 'wide_neg': 405}
SUCCESSORS = {'half_neg': 2, 'wide_neg': 2}
CONTROLS = {'half_neg': 46, 'wide_neg': 46}
ALLOCATIONS = {'half_neg': 2, 'wide_neg': 2}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _restore_file(raw, path, bind):
    bind(Path(__file__).resolve())
    delta_raw = bind(Path(__file__).with_name('typed_relu_observer_delta.json'))
    delta = json.loads(delta_raw)
    parents = dict(PARENTS, **BINDINGS)
    require(delta['schema'] == 'heptatorch.typed-relu-four-negation-controls.observer-delta.v1'
            and set(delta['files']) == set(parents) and path in parents
            and delta['original_cases'] == ORIGINAL_CASES
            and delta['replaced_refusals'] == SUCCESSORS
            and delta['original_protocol_control_slots'] == CONTROLS
            and delta['actual_allocation_controls'] == ALLOCATIONS,
            'exact two original groups, four unary controls and original protocol/OOM denominators')
    record = delta['files'][path]
    require(type(raw) is bytes and len(raw) == record['source_bytes']
            and sha(raw) == record['source_sha256'] and record['parent_sha256'] == parents[path],
            'complete current observer and immutable original source identities')
    text = raw.decode('utf-8')
    for edit in reversed(record['edits']):
        require(set(edit) == {'before', 'after', 'occurrences'} and edit['occurrences'] in (1, 2)
                and edit['before'] != edit['after'] and text.count(edit['after']) == edit['occurrences'],
                'exact ordered complete observer edit inverse')
        text = text.replace(edit['after'], edit['before'])
    parent = text.encode('utf-8')
    require(len(parent) == record['parent_bytes'] and sha(parent) == parents[path],
            'every complete immutable predecessor observer byte recovered')
    for edit in record['edits']:
        require(text.count(edit['before']) == edit['occurrences'], 'exact forward observer edit count')
        text = text.replace(edit['before'], edit['after'])
    require(text.encode('utf-8') == raw, 'complete observer forward round trip')
    return parent, dict(schema='heptatorch.typed-relu-observer-file-inverse.v1', path=path,
        source_bytes=len(raw), source_sha256=sha(raw), parent_bytes=len(parent), parent_sha256=sha(parent),
        delta_sha256=sha(delta_raw), ordered_edits=len(record['edits']),
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        runtime_qualification_transferred=False)


def restore_runner(raw, path, bind):
    require(path in PARENTS, 'only one of the two original negation runners')
    return _restore_file(raw, path, bind)


def restore_observers(bind):
    """Return both whole original runners and the deterministic admission proof."""
    root = Path(__file__).resolve().parents[2]
    parents, proofs, binding_proofs = {}, {}, {}
    for path in PARENTS:
        parents[path], proofs[path] = restore_runner(bind(root/path), path, bind)
    for path in BINDINGS:
        _, binding_proofs[path] = _restore_file(bind(root/path), path, bind)
    for path, expected in INPUT_PINS.items():
        require(sha(bind(root/path)) == expected, 'every original negation recipe/reference byte: '+path)
    proof = dict(schema='heptatorch.typed-relu-four-negation-controls.observer-inverse.v1',
        paths=list(PARENTS), files=proofs, existing_pin_consumer_inverses=binding_proofs,
        helper_sha256=sha(bind(Path(__file__).resolve())),
        delta_sha256=sha(bind(Path(__file__).with_name('typed_relu_observer_delta.json'))),
        original_inputs=INPUT_PINS, original_cases=ORIGINAL_CASES,
        explicit_numerical_successors=SUCCESSORS, original_protocol_control_slots=CONTROLS,
        actual_allocation_controls=ALLOCATIONS, changed_opcode=3, changed_operation=5,
        changed_native_dtypes=[2, 4, 5, 6],
        all_original_cases_and_negation_numerical_predicates_retained=True,
        only_four_original_unary_op5_refusals_receive_numerical_successors=True,
        all_other_original_refusals_and_owner_rng_lifetime_checks_retained=True,
        original_FE_and_failed_suffix_owner_mapping_checks_retained=True,
        original_actual_OOM_identical_request_recovery_and_EOF_checks_retained=True,
        complete_actual_shared776_reference_required=True,
        full_literal_inputs_and_original_request_bytes_required=True,
        replacement_controls_require_actual_torch_bits_geometry_and_fresh_owners=True,
        historical_half239_and_f6487_partitions_unchanged=True,
        existing_old_observer_SHA_pins_preserved=True,
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        native_executed=False, Torch_executed=False, runtime_qualification_transferred=False)
    return parents, proof
