#!/usr/bin/env python3
"""Read-only independent source preservation checks for owned Adam02.

No authoring module or numerical implementation is imported or executed.
This source audit complements, rather than replaces, actual native execution.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

SHA=lambda raw:hashlib.sha256(raw).hexdigest()
FUNCTION=re.compile(rb'^(fn (\w+)\([^\n]* entry\(\w+\)\n.*?^end\n)',re.M|re.S)
BASE='7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf'
OLD='bd38b67c9f39859ddb81f304d9043af625bdcf3305d60e98ba988a6208e42885'
NEW='253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6'
PIN='31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
PRIMARY={
 'adam.py':('5ecb37579ebbec6cdcf4aa69b394136b98d425144a549d679c6369f9c76146b8','torch/optim/adam.py'),
 'adamw.py':('b9cbfb1e2112b61d7aa88b0f8e2a777ddae9cca5e348e231256f1e39f06f1d92','torch/optim/adamw.py'),
 'optimizer.py':('4b0ecf48fa034f60e6ac1e5055846bbc907ec63c2f174b1bb81472354593075b','torch/optim/optimizer.py'),
 'MemoryOverlap.cpp':('eaa96f7e06abe6274adeb5d2bec717bb2844fb99ea524c0e2ad19913f8ed04d6','aten/src/ATen/MemoryOverlap.cpp')}


def functions(raw):
    rows=FUNCTION.findall(raw);result={name.decode():body for body,name in rows}
    assert len(result)==len(rows)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','previous','build','upstream-sources','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();inputs={}
    def read(path,pin=None):
        path=path.resolve();raw=path.read_bytes();inputs[path]=raw
        if pin:assert SHA(raw)==pin,path
        return raw
    base=functions(read(a.baseline/'runtime.hepta',BASE));old=functions(read(a.previous/'runtime.hepta',OLD));new=functions(read(a.build/'runtime.hepta',NEW))
    assert set(old)-set(new)==set() and set(new)-set(old)=={'owned_adam_new_moment'}
    delta=[name for name in old if old[name]!=new[name]]
    assert delta==['owned_adam_initialize'],delta
    converted=new['owned_adam_initialize']
    needle=b'{"function":"owned_adam_new_moment"}'
    assert converted.count(needle)==2
    converted=converted.replace(needle,b'{"function":"optimizer_new_momentum"}')
    assert converted==old['owned_adam_initialize']
    allocator=new['owned_adam_new_moment'].replace(b'fn owned_adam_new_moment(',b'fn optimizer_new_momentum(',1)
    assert allocator==base['optimizer_new_momentum']==new['optimizer_new_momentum']
    inverses=[
        (b'  v15:u64 = call(v0,v1,v4) {"function":"owned_optimizer_gradient_value"}',b'  v15:u64 = call(v0,v14) {"function":"get4"}'),
        (b'  v16:u64 = call(v0,v7,v15,v5) {"function":"owned_adam_gradient"}',b'  v16:u64 = call(v0,v7,v15,v5) {"function":"adam_gradient"}'),
        (b'  v21:u64 = call(v0,v2,v4) {"function":"taddr"}',b'  v21:u64 = int.add.wrap(v20,v13)'),
        (b'  v25:u64 = call(v0,v3,v4) {"function":"taddr"}',b'  v25:u64 = int.add.wrap(v24,v13)')]
    numeric=new['owned_adam_element'];restored=numeric.replace(b'fn owned_adam_element(',b'fn adam_element(',1)
    for current,previous in inverses:
        assert restored.count(current)==1;restored=restored.replace(current,previous,1)
    assert restored==base['adam_element']==new['adam_element']
    assert SHA(restored)=='c4c0b566b978a57deca386ba8452d19dac7fa5ce9cab08f060ca6b9721fd1f4a'
    baseline_changes=[name for name in base if base[name]!=new[name]]
    assert set(baseline_changes)=={'entry','state_header','init_state','so_state_init','so_state_header','owned_optimizer_command','optimizer_audit_record'}
    for original,renamed in [('owned_optimizer_command','owned_sgd_command'),('optimizer_audit_record','sgd_optimizer_audit_record')]:
        assert new[renamed].replace(('fn '+renamed+'(').encode(),('fn '+original+'(').encode(),1)==base[original]
    immutable_modules={}
    for name in ('rng03.hepta','owners.hepta'):
        before=read(a.baseline/name);after=read(a.build/name);assert before==after
        immutable_modules[name]=SHA(after)
    primary={};custody=json.loads(read(a.upstream_sources/'source-custody.json'));assert custody['commit']==PIN
    for name,(digest,path) in PRIMARY.items():
        raw=read(a.upstream_sources/name,digest)
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==custody['files'][name]['git_blob_sha1']
        primary[name]={'sha256':digest,'bytes':len(raw),'git_blob_sha1':blob,
                       'url':'https://github.com/pytorch/pytorch/blob/'+PIN+'/'+path}
    authoring=read(a.build/'adam-authoring.py')
    nodes={n.name:n for n in ast.parse(authoring).body if isinstance(n,ast.FunctionDef)}
    source_locations={name:{'first_line':node.lineno,'last_line':node.end_lineno,
                              'normalized_ast_sha256':SHA(ast.dump(node,include_attributes=False).encode())}
                      for name,node in nodes.items()}
    for path,raw in inputs.items():assert path.read_bytes()==raw,path
    report={'schema':'heptatorch.independent-owned-adam-source-audit.v1','status':'PASS',
            'baseline_source_sha256':BASE,'previous_source_sha256':OLD,'current_source_sha256':NEW,
            'candidate01_to02_changed_functions':delta,'two_initializer_call_names_only':True,
            'new_allocator_body_exact_renaming_of_frozen_zeros_like_parameter_helper':True,
            'allocator_sha256':SHA(new['owned_adam_new_moment']),
            'frozen_scalar_numerical_function_sha256':SHA(restored),'new_scalar_function_sha256':SHA(numeric),
            'four_access_changes_and_name_inverse_byte_identical':True,'floating_operations_unchanged':True,
            'baseline_functions':len(base),'baseline_functions_unchanged':len(base)-len(baseline_changes),
            'baseline_changes':baseline_changes,'original_SGD_command_and_audit_bodies_retained_by_exact_rename':True,
            'immutable_modules':immutable_modules,'primary_sources':primary,
            'read_only_review':[
                {'functions':['owned_adam_plan_group','owned_adam_initialize','owned_adam_key'],
                 'finding':'Only defined gradients realize state; the ordered group planning pass initializes one state per repeated Tensor key before numeric updates.'},
                {'functions':['owned_adam_step_exponent','owned_adam_apply'],
                 'finding':'Increment is actual binary32 Tensor arithmetic; admitted exponent is derived from the incremented value. Positive integral values below 2^32 pass; no independent integer step counter exists.'},
                {'functions':['owned_adam_versions_ready','owned_adam_apply'],
                 'finding':'Version capacity is checked for parameter, step and both moment Tensors before writes. The existing actual-version helper remains byte-identical.'},
                {'functions':['owned_adam_fill_group','owned_adam_initialize','owned_adam_drop'],
                 'finding':'Membership retains each occurrence, realized first keys retain one further parameter root and three real state Tensors; drop removes matching obligations before graph collection.'},
                {'functions':['owned_adam_create','owned_adam_group_payload','owned_adam_add_group'],
                 'finding':'Counts are bounded before size arithmetic: 1024 groups, 4096 total occurrences. New group blob copies scalar IDs/offsets and carries existing logical roots; no native owner descriptor is copied into bytes.'},
                {'functions':['owned_adam_command','owned_adam_group_inventory'],
                 'finding':'Payload output capacity is checked before inventory writes; entry protects its record header. Unknown fields, widths, flags and offsets retain ordinary frame-atomic failure.'}],
            'source_locations':source_locations,'general_real_exponent_not_implemented':True,
            'arbitrary_forged_owned_state_authentication_claimed':False,'arbitrary_python_optimizer_exception_effects_claimed':False,
            'native_execution_established_by_separate_replay_not_this_source_audit':True,
            'source_custody_after':True,'inputs':{str(path):SHA(raw) for path,raw in inputs.items()}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    assert not a.output.exists();a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':report['status'],'source':NEW,'baseline_functions_unchanged':report['baseline_functions_unchanged'],
                      'new_allocator_exact':True,'numerical_inverse_exact':True,'report_sha256':SHA(a.output.read_bytes())}))


if __name__=='__main__':main()
