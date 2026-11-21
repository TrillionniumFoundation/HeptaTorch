"""Restore the exact312-case multiplication observer before four add/sub controls.

All312 original input recipes and numerical outcomes remain unchanged. Four
previous unsupported-operation checks now observe genuine add/sub results,
their complete ownership, and their separately retained fixed-Torch evidence.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE='051bf9fb08ac721079eecb339ac8462206b376fb5b634cfd9df3330750c0f4d1'
PARENT='49a1d9c4f82ca4c2c44f17866013d670e62f2a0397273cd5977e8d36f8826cf5'
PATH='tests/half_mul/run.py'
LABELS=tuple(f'promotion-{dtype}-{dtype}-3:operation-{operation}' for dtype in (5,6) for operation in (1,2))
NEGATION_INPUTS={
    'tests/half_neg/__init__.py':'f02584b81106f7440ca54bf8f59beb9a3e05feb8c1123964325d5e946b839163',
    'tests/half_neg/cases.py':'643da7cd50f0b9364444bac11a091b99b1fd6670b91979dca40e3227ee07cd69',
    'tests/half_neg/reference.py':'b6f7ee4cd1ec107800abb650ce0aabbd63fd7edbbbb642a2290c5ae4d178a2bf',
    'tests/half_neg/run.py':'02f0ebfd4478179bf3e805ecde751fae087d76738385b9a2ec1cc6f8f28bc51d',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def restore_mul_runner(raw,bind):
    bind(Path(__file__).resolve())
    inverse_path=Path(__file__).with_name('half_div_observer_transition.py');bind(inverse_path)
    spec=importlib.util.spec_from_file_location('half_mul_division_observer_inverse',inverse_path)
    inverse=importlib.util.module_from_spec(spec);spec.loader.exec_module(inverse)
    raw,division_proof=inverse.restore_runner(raw,PATH,bind)
    root=Path(__file__).resolve().parents[2]
    for path,expected in NEGATION_INPUTS.items():
        require(sha(bind(root/path))==expected,'unchanged complete371-case negation observer: '+path)
    delta_raw=bind(Path(__file__).with_name('half_addsub_mul_observer_delta.json'));delta=json.loads(delta_raw)
    require(sha(raw)==SOURCE and delta['source_sha256']==SOURCE and delta['parent_sha256']==PARENT
        and delta['schema']=='heptatorch.half-addsub-mul-observer-delta.v1' and delta['path']==PATH,
        'exact current and immutable predecessor multiplication observers')
    text=raw.decode('utf-8')
    for edit in reversed(delta['edits']):
        require(set(edit)=={'before','after','occurrences'} and edit['occurrences'] in (1,2)
            and edit['before']!=edit['after'] and text.count(edit['after'])==edit['occurrences'],
            'exact unique complete observer edit')
        text=text.replace(edit['after'],edit['before'])
    parent=text.encode('utf-8')
    require(sha(parent)==PARENT,'every predecessor observer byte reconstructed')
    for edit in delta['edits']:
        require(text.count(edit['before'])==edit['occurrences'],'exact forward observer edit')
        text=text.replace(edit['before'],edit['after'])
    require(text.encode('utf-8')==raw,'complete observer forward round trip')
    return parent,dict(schema='heptatorch.half-addsub-four-mul-controls.observer-inverse.v1',
        path=PATH,source_sha256=SOURCE,parent_sha256=PARENT,delta_sha256=sha(delta_raw),
        replaced_refusal_labels=list(LABELS),all312_original_cases_unchanged=True,
        every_original_multiplication_numerical_predicate_retained=True,
        other_refusals_and_original_owner_rng_lifetime_checks_retained=True,
        replacement_controls_require_actual_torch_bits_and_geometry=True,
        replacement_controls_require_fresh_ownership_and_unchanged_inputs=True,
        all371_negation_cases_and_observer_bytes_unchanged=NEGATION_INPUTS,
        native_division_observer_inverse=division_proof,
        actual_runner_sha256=division_proof['source_sha256'],
        whole_source_inverse_exact=True,whole_round_trip_exact=True,runtime_qualification_transferred=False)
