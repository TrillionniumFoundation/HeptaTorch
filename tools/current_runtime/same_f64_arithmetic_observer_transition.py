"""Restore all original mixed773 observer bytes before the same-F64 successor.

Only source admission and the now-false predecessor route marker change.
The complete 2541 original cases and all numerical, error, lifetime, rollback
and actual allocation predicates are retained. No execution PASS transfers.
"""
import ast
import hashlib
import json
from pathlib import Path

sha=lambda raw:hashlib.sha256(raw).hexdigest()
DELTA='f4eb2011b3aaf98877cf8994b7dd011b9d8c7146bb8bc1e281210aafc509cebb'
CASES='68aba41765e05a925bf29dc53f28503bc180a211d094ccaf8c923596aee04d87'
PATHS=('tests/public_f64_arithmetic/run.py','tests/public_f64_arithmetic/reference.py')
PARENTS=('efff9a4c847762641d119e9cc1f50a5c047e7481e096e547f931538655f0bb07',
         'cd49631824c9b292de0a9729a87f821e48a4ea21775d49cb003cff887a2ba24e')

def require(value,message):
    if not value:raise ValueError(message)

def definitions(raw):
    lines=raw.splitlines(keepends=True)
    return {node.name:b''.join(lines[node.lineno-1:node.end_lineno]) for node in ast.parse(raw).body
        if isinstance(node,(ast.FunctionDef,ast.ClassDef))}

def restore_observers(bind):
    root=Path(__file__).resolve().parents[2];bind(Path(__file__).resolve())
    delta_raw=bind(Path(__file__).with_name('same_f64_arithmetic_observer_delta.json'))
    require(sha(delta_raw)==DELTA,'immutable complete mixed observer delta')
    delta=json.loads(delta_raw)
    require(delta['schema']=='heptatorch.same-f64-arithmetic.observer-delta.v1'
        and [row['path'] for row in delta['changes']]==list(PATHS)
        and [row['parent_sha256'] for row in delta['changes']]==list(PARENTS),
        'exact two frozen mixed773 predecessor observers')
    restored={};identities={}
    for record in delta['changes']:
        path=record['path'];raw=bind(root/path);parent=raw
        require(sha(raw)==record['source_sha256'],'complete successor observer identity')
        for edit in reversed(record['replacements']):
            old,new=edit['original'].encode(),edit['current'].encode()
            require(parent.count(new)==1 and old!=new,'one exact observer admission or scope edit')
            parent=parent.replace(new,old,1)
        require(sha(parent)==record['parent_sha256'],'complete immutable mixed773 observer recovered')
        forward=parent
        for edit in record['replacements']:
            old,new=edit['original'].encode(),edit['current'].encode()
            require(forward.count(old)==1,'one exact original observer edit location')
            forward=forward.replace(old,new,1)
        require(forward==raw,'complete observer forward round trip')
        before,after=definitions(parent),definitions(raw)
        require(before.keys()==after.keys(),'no observer definition added or removed')
        changed=[name for name in before if before[name]!=after[name]]
        require(changed==(['source_inverse','qualify'] if path==PATHS[0] else ['main']),
            'only source admission and truthful scope marker differ')
        if path==PATHS[0]:
            require(before['qualify'].replace(b'legacy_same_f64_route_unchanged=True',
                b'legacy_same_f64_route_unchanged=False')==after['qualify'],
                'all2541 original native comparisons and replay predicates retained byte-for-byte')
        else:
            require(before['main'].replace(b'legacy_same_f64_route_unchanged=True',
                b'legacy_same_f64_route_unchanged=False')==after['main'],
                'actual Torch inputs, operations and retained result bytes are unchanged')
        restored[path]=parent
        identities[path]=dict(source_sha256=sha(raw),parent_sha256=sha(parent),changed_definitions=changed)
    cases=bind(root/'tests/public_f64_arithmetic/cases.py')
    # The predecessor runner independently binds every original case and value.
    require(sha(cases)==CASES and b'CASE_COUNT=len(CASES)' in cases and b'CASES=list(recipes())' in cases,
        'the original fixed mixed arithmetic case construction remains explicit')
    return restored,dict(schema='heptatorch.same-f64-arithmetic.observer-inverse.v1',files=identities,
        original_mixed_reference_cases=2541,original_mixed_case_file_sha256=sha(cases),
        original_mixed_numerical_error_lifetime_and_allocation_predicates_retained=True,
        original_fixed_Torch_arithmetic_observation_code_unchanged=True,
        same_f64_predecessor_route_marker_now_false=True,
        whole_observer_inverse_exact=True,whole_observer_forward_round_trip_exact=True,
        runtime_qualification_transferred=False)
