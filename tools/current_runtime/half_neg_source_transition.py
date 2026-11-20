"""Restore every758 source byte before native Half tensor negation.

This inverse never transfers runtime qualification. Two historical negation
refusals require separate actual-Torch successor evidence; the original inputs
and all remaining checks stay intact.
"""
import hashlib
import json
from pathlib import Path
import re

SOURCE='ab3ed0e0898d2006fbdf145357c79cafd599c2be344fcb3cb65fd6ad55efb231'
IMAGE='756de9418c05f4ecfc12319c7262cf1ff22e335e4ec32b140721b6c77ee226d7'
MANIFEST='ef76a93ddee90e2fc546b2a6f95a5133b7663bd535e1f7693bae3b519345060c'
PARENT_SOURCE='027d56d8d8103a48a158d77d6ac750b6eb5d1992a49dd6479e961fe085c51334'
FUNCTIONS,PARENT_FUNCTIONS=761,758
CHANGED=('unary','command_argument')
ADDED=('half_neg_previous_unary','half_neg_value','half_neg_tensor')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def functions(source,count):
    matches=list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',source,re.M|re.S))
    result={match[1].decode('ascii'):match[0] for match in matches}
    require(len(result)==len(matches)==count,'complete unique Half negation source census')
    require(source==source.split(b'\n',1)[0]+b'\n'+b''.join(result.values()),
        'complete source envelope and declaration order')
    return result


def restore_source(source,bind):
    bind(Path(__file__).resolve())
    require(isinstance(source,bytes) and sha(source)==SOURCE,'exact admitted761 Half negation source')
    delta_raw=bind(Path(__file__).with_name('half_neg_source_delta.json'));delta=json.loads(delta_raw)
    require(delta['schema']=='heptatorch.half-neg-source-delta.v1'
        and delta['source_sha256']==SOURCE and delta['parent_source_sha256']==PARENT_SOURCE
        and delta['functions']==FUNCTIONS and delta['parent_functions']==PARENT_FUNCTIONS
        and set(delta['changed_functions'])==set(CHANGED) and tuple(delta['appended_functions'])==ADDED,
        'exact two changed bodies and three appended helpers')
    actual=functions(source,FUNCTIONS)
    require(tuple(actual)[-len(ADDED):]==ADDED,'only the exact three negation helpers are appended')
    suffix=b''.join(actual[name] for name in ADDED)
    require(source.endswith(suffix) and sha(suffix)==delta['helper_sha256'],'complete native helper suffix identity')
    restored=source[:-len(suffix)];originals={}
    for name in CHANGED:
        row,after=delta['changed_functions'][name],actual[name];before=row['before'].encode('utf-8')
        require(before.startswith(('fn '+name+'(').encode()) and before.endswith(b'end\n')
            and sha(before)==row['before_sha256'] and sha(after)==row['after_sha256']
            and before!=after and restored.count(after)==1,'exact complete old/current negation body: '+name)
        originals[name]=before;restored=restored.replace(after,before,1)
    parent=functions(restored,PARENT_FUNCTIONS)
    require(sha(restored)==PARENT_SOURCE and list(actual)[:PARENT_FUNCTIONS]==list(parent)
        and tuple(name for name in parent if actual[name]!=parent[name])==CHANGED,
        'every reconstructed758 byte and all756 unchanged bodies match the predecessor')
    require(actual['half_neg_previous_unary']==parent['unary'].replace(
        b'fn unary(',b'fn half_neg_previous_unary(',1),'the complete predecessor unary body is retained exactly')
    before=parent['command_argument'];old=b'  v140:bool = bool.and(v138,half_operation)'
    new=(b'  half_unary_opcode:u64 = const() {"value":3}\n'
         b'  half_not_unary:bool = compare.ne(v73,half_unary_opcode)\n'
         b'  half_not_arithmetic:bool = bool.and(half_operation,half_not_unary)\n'
         b'  v140:bool = bool.and(v138,half_not_arithmetic)')
    require(before.count(old)==1 and actual['command_argument']==before.replace(old,new,1),
        'only command3 is added to the public Half opcode admission guard')
    mul_names=tuple(name for name in parent if name.startswith('half_mul_'))
    require(len(mul_names)==16 and all(actual[name]==parent[name] for name in mul_names),
        'every758 multiplication helper remains byte identical')
    forward=restored
    for name in CHANGED:
        require(forward.count(originals[name])==1,'unique complete forward body')
        forward=forward.replace(originals[name],actual[name],1)
    require(forward+suffix==source,'complete source761 forward round trip')
    return restored,dict(schema='heptatorch.half-neg761-to-half-mul758.complete-source-inverse.v1',
        source_sha256=SOURCE,parent_source_sha256=PARENT_SOURCE,functions=FUNCTIONS,parent_functions=PARENT_FUNCTIONS,
        changed_functions=list(CHANGED),added_functions=list(ADDED),removed_functions=[],unchanged_parent_functions=756,
        unchanged_multiplication_helpers=list(mul_names),delta_sha256=sha(delta_raw),whole_source_inverse_exact=True,
        whole_round_trip_exact=True,every_other_byte_constrained_by_complete_parent_sha256=True,
        actual_executable_replaced=False,historical_cases_or_expected_values_modified=False,
        newly_supported_request_outcomes_require_separate_successor_evidence=True,
        predecessor_unary_body_preserved_exactly=True,runtime_qualification_transferred=False,
        changed_bodies={name:dict(before_sha256=sha(parent[name]),after_sha256=sha(actual[name])) for name in CHANGED},
        added_bodies={name:dict(bytes=len(actual[name]),sha256=sha(actual[name])) for name in ADDED})
