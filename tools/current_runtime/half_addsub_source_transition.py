"""Restore every761 source byte before Half add/sub fixed-alpha-one admission.

The inverse transfers no runtime qualification. Every changed request needs
fresh actual-Torch evidence, including the four old multiplication controls.
"""
import hashlib
import json
from pathlib import Path
import re

SOURCE='b4b2996978dce56c9375ce97b2d2638472cbb154a4c0a7c5de822234e31a9c68'
IMAGE='62f74e9232ae30027a64ca61881bb581320b1787529c6ee9e96818a0f8447afe'
MANIFEST='100f60ce946ac809949d79faf4eca8e8e16d5f4aeacfd907d228edc1ea0c1fd2'
PARENT_SOURCE='ab3ed0e0898d2006fbdf145357c79cafd599c2be344fcb3cb65fd6ad55efb231'
FUNCTIONS,PARENT_FUNCTIONS=765,761
CHANGED=('binary',)
ADDED=('half_addsub_f32','half_addsub_f64','half_addsub_element','half_addsub_tensor')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def functions(source,count):
    matches=list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',source,re.M|re.S))
    result={match[1].decode('ascii'):match[0] for match in matches}
    require(len(result)==len(matches)==count,'complete unique Half add/sub source census')
    require(source==source.split(b'\n',1)[0]+b'\n'+b''.join(result.values()),
        'complete source envelope and declaration order')
    return result


def restore_source(source,bind):
    bind(Path(__file__).resolve())
    require(isinstance(source,bytes) and sha(source)==SOURCE,'exact admitted765 Half add/sub source')
    delta_raw=bind(Path(__file__).with_name('half_addsub_source_delta.json'));delta=json.loads(delta_raw)
    require(delta['schema']=='heptatorch.half-addsub-source-delta.v1'
        and delta['source_sha256']==SOURCE and delta['parent_source_sha256']==PARENT_SOURCE
        and delta['functions']==FUNCTIONS and delta['parent_functions']==PARENT_FUNCTIONS
        and set(delta['changed_functions'])==set(CHANGED) and tuple(delta['appended_functions'])==ADDED,
        'exact single changed body and four appended arithmetic helpers')
    actual=functions(source,FUNCTIONS)
    require(tuple(actual)[-len(ADDED):]==ADDED,'only the exact four add/sub helpers are appended')
    suffix=b''.join(actual[name] for name in ADDED)
    require(source.endswith(suffix) and sha(suffix)==delta['helper_sha256'],'complete native helper suffix identity')
    restored=source[:-len(suffix)];originals={}
    for name in CHANGED:
        row,after=delta['changed_functions'][name],actual[name];before=row['before'].encode('utf-8')
        require(before.startswith(('fn '+name+'(').encode()) and before.endswith(b'end\n')
            and sha(before)==row['before_sha256'] and sha(after)==row['after_sha256']
            and before!=after and restored.count(after)==1,'exact complete old/current arithmetic body: '+name)
        originals[name]=before;restored=restored.replace(after,before,1)
    parent=functions(restored,PARENT_FUNCTIONS)
    require(sha(restored)==PARENT_SOURCE and list(actual)[:PARENT_FUNCTIONS]==list(parent)
        and tuple(name for name in parent if actual[name]!=parent[name])==CHANGED,
        'every reconstructed761 byte and all760 unchanged bodies match the predecessor')
    old=b'  branch half3 multiply(s3,o3,l3,r3) previous(s3,o3,l3,r3)\n'
    new=b'''  branch half3 arithmetic(s3,o3,l3,r3) previous(s3,o3,l3,r3)
block arithmetic(sa:mutbuf,oa:u64,la:u64,ra:u64)
  onea:u64 = const() {"value":1}
  twoa:u64 = const() {"value":2}
  adda:bool = compare.eq(oa,onea)
  suba:bool = compare.eq(oa,twoa)
  addsuba:bool = bool.or(adda,suba)
  branch addsuba addsub(sa,oa,la,ra) multiply(sa,oa,la,ra)
block addsub(ss:mutbuf,os:u64,ls:u64,rs:u64)
  results:u64 = call(ss,os,ls,rs) {"function":"half_addsub_tensor"}
  ret(results)
'''
    require(parent['binary'].count(old)==1 and actual['binary']==parent['binary'].replace(old,new,1),
        'only Half operations1/2 change routing; every other predecessor branch remains exact')
    arithmetic=tuple(n for n in parent if n.startswith(('half_mul_','half_neg_')))
    require(len(arithmetic)==19 and all(actual[n]==parent[n] for n in arithmetic)
        and actual['command_argument']==parent['command_argument'],
        'all19 existing Half mul/neg helpers and command admission remain byte identical')
    forward=restored
    for name in CHANGED:
        require(forward.count(originals[name])==1,'unique complete forward body')
        forward=forward.replace(originals[name],actual[name],1)
    require(forward+suffix==source,'complete source765 forward round trip')
    return restored,dict(schema='heptatorch.half-addsub765-to-half-neg761.complete-source-inverse.v1',
        source_sha256=SOURCE,parent_source_sha256=PARENT_SOURCE,functions=FUNCTIONS,parent_functions=PARENT_FUNCTIONS,
        changed_functions=list(CHANGED),added_functions=list(ADDED),removed_functions=[],unchanged_parent_functions=760,
        unchanged_existing_half_arithmetic_helpers=list(arithmetic),delta_sha256=sha(delta_raw),whole_source_inverse_exact=True,
        whole_round_trip_exact=True,every_other_byte_constrained_by_complete_parent_sha256=True,
        actual_executable_replaced=False,historical_cases_or_expected_values_modified=False,
        newly_supported_request_outcomes_require_separate_successor_evidence=True,
        non_half_and_predecessor_half_other_operation_routes_preserved=True,runtime_qualification_transferred=False,
        changed_bodies={name:dict(before_sha256=sha(parent[name]),after_sha256=sha(actual[name])) for name in CHANGED},
        added_bodies={name:dict(bytes=len(actual[name]),sha256=sha(actual[name])) for name in ADDED})
