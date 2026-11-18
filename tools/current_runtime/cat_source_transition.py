"""Reverse exactly the three reviewed cat bodies before old source proofs.

This is a source-identity proof only. It starts no process and transfers no
historical runtime result. Every remaining body is constrained by the complete
restored 0c1c source hash, not by a relaxed per-function exception list.
"""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
SOURCE='ba96abc05717643eaf4f2056325089d9a51f5a2e9838f89473e9a4e76e46fd99'
PARENT_SOURCE='0c1c0dd3acf0285a887a2766c1308dfd992acf3fe51849dcd98749bb64a00f3b'
DELTA_SHA='5ce5962e35ec9eb26bc10c39b751e412aa20379123758e2e3f17fc9063772297'
CHANGED=('concat_copy','incoming_describe_cat','tensor_join')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def functions(raw):
    matches=list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',raw,re.M|re.S))
    bodies={m[1].decode():m[0] for m in matches}
    require(len(bodies)==len(matches)==700,'exact complete cat profile function census')
    require(raw==raw.split(b'\n',1)[0]+b'\n'+b''.join(bodies.values()),'complete canonical source ordering')
    return bodies


def restore_source(source,bind):
    """Bind actual current bytes and return the exact complete previous source."""
    bind(Path(__file__).resolve())
    require(sha(source)==SOURCE,'only the actual reviewed cat source is admitted')
    raw=bind(HERE/'cat_source_transition.json',DELTA_SHA)
    delta=json.loads(raw)
    require(delta['schema']=='heptatorch.current-cat.three-function-delta.v1'
        and delta['source_sha256']==SOURCE and delta['parent_source_sha256']==PARENT_SOURCE
        and delta['functions']==700 and delta['changed_functions']==list(CHANGED)
        and delta['unchanged_functions']==697,'exact three-function source delta')
    actual=functions(source)
    require(source.split(b'\n',1)[0].decode()==delta['header'],'unchanged canonical source header')
    restored=source
    original={}
    for name in CHANGED:
        row=delta['bodies'][name]
        before=row['before_body'].encode()
        after=actual[name]
        require(len(before)==row['before_bytes'] and sha(before)==row['before_sha256']
            and before.startswith(('fn '+name+'(').encode()) and before.endswith(b'end\n'),
            'exact original complete body: '+name)
        require(len(after)==row['after_bytes'] and sha(after)==row['after_sha256']
            and before!=after and restored.count(after)==1,'exact current complete body: '+name)
        original[name]=before
        restored=restored.replace(after,before,1)
    previous=functions(restored)
    require(list(previous)==list(actual)
        and sorted(name for name in actual if actual[name]!=previous[name])==list(CHANGED)
        and sum(actual[name]==previous[name] for name in actual)==697,
        'only the three complete reviewed bodies may change')
    require(sha(restored)==PARENT_SOURCE,'complete previous source hash; all 697 other bodies constrained')
    again=restored
    for name in CHANGED:
        require(again.count(original[name])==1,'unique forward source reconstruction')
        again=again.replace(original[name],actual[name],1)
    require(again==source,'complete source round trip')
    proof=dict(schema='heptatorch.current-cat.complete-source-inverse.v1',
        source_sha256=SOURCE,parent_source_sha256=PARENT_SOURCE,functions=700,
        changed_functions=list(CHANGED),unchanged_functions=697,
        delta_sha256=DELTA_SHA,whole_source_inverse_exact=True,
        all_other_bodies_constrained_by_complete_parent_sha256=True,
        original_header_preserved=True,whole_round_trip_exact=True,
        original_reference_values_modified=False,old_source_proofs_modified=False,
        runtime_qualification_transferred=False)
    return restored,proof
