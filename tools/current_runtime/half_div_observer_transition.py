"""Restore both complete original arithmetic runners before six div controls."""
import hashlib
import json
from pathlib import Path

PARENTS={
    'tests/half_mul/run.py':'051bf9fb08ac721079eecb339ac8462206b376fb5b634cfd9df3330750c0f4d1',
    'tests/half_addsub/run.py':'cac61c1022008be413d626d0a7456a35ce47ea2ecd390c2ea0a6c82c19e5777a',
}
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def restore_runner(raw,path,bind):
    bind(Path(__file__).resolve())
    delta_raw=bind(Path(__file__).with_name('half_div_observer_delta.json'));delta=json.loads(delta_raw)
    require(delta['schema']=='heptatorch.half-div-six-arithmetic-controls.observer-delta.v1'
        and set(delta['files'])==set(PARENTS) and path in PARENTS
        and delta['replaced_refusals']=={'half_mul':2,'half_addsub':4}
        and delta['original_cases']=={'half_mul':312,'half_addsub':1006},'exact six-control successor domain')
    record=delta['files'][path]
    require(len(raw)==record['source_bytes'] and sha(raw)==record['source_sha256']
        and record['parent_sha256']==PARENTS[path],'complete observer successor identity')
    text=raw.decode()
    for edit in reversed(record['edits']):
        require(set(edit)=={'before','after','occurrences'} and edit['occurrences'] in (1,3)
            and edit['before']!=edit['after'] and text.count(edit['after'])==edit['occurrences'],
            'exact ordered original observer edit inverse')
        text=text.replace(edit['after'],edit['before'])
    parent=text.encode()
    require(sha(parent)==PARENTS[path] and len(parent)==record['parent_bytes'],
        'every complete predecessor arithmetic observer byte restored')
    for edit in record['edits']:
        require(text.count(edit['before'])==edit['occurrences'],'exact forward observer edit count')
        text=text.replace(edit['before'],edit['after'])
    require(text.encode()==raw,'complete arithmetic observer forward round trip')
    return parent,dict(schema='heptatorch.half-div-six-arithmetic-controls.observer-inverse.v1',
        path=path,source_sha256=sha(raw),parent_sha256=sha(parent),delta_sha256=sha(delta_raw),
        original_cases=delta['original_cases'],replaced_refusals=delta['replaced_refusals'],
        all_original_cases_and_numerical_predicates_retained=True,
        other_refusals_and_owner_rng_lifetime_checks_retained=True,
        complete_actual_shared1196_reference_required=True,
        replacement_controls_require_actual_torch_bits_geometry_and_fresh_owners=True,
        whole_source_inverse_exact=True,whole_round_trip_exact=True,
        runtime_qualification_transferred=False)
