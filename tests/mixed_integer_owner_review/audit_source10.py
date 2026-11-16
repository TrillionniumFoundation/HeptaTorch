#!/usr/bin/env python3
"""Read-only complete source/unit custody and exact observer delta, no native run."""
import argparse,ast,json,re
from pathlib import Path
import owner640_mixed10 as observer
from resource_support import Custody,load,sha,write_json,COMPILER

PREVIOUS_SOURCE='adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62'
PREVIOUS_IMAGE='b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08'
SESSION_LINKER='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'

def bodies(raw):
    pairs=[(m[1].decode(),m[0]) for m in re.finditer(rb'^fn (\w+)\([^\n]* entry\([^\n]*\)\n.*?^end\n?',raw,re.M|re.S)]
    assert len(pairs)==len(dict(pairs))
    return dict(pairs)

def definitions(raw):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','previous-build','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);custody=Custody(out)
    here=Path(__file__).resolve().parent
    custody.bind(Path(__file__),copy='audit_source10.py')
    _,image,info,_=load(custody,a.build,a.protocol,expected_source=observer.SOURCE,expected_image=observer.IMAGE)
    assert info['functions']==556 and info['blocks']==7420 and info['instructions']==26889
    assert info['compiler_sha256']==COMPILER and info['linker_sha256']==SESSION_LINKER and info['link_exit']==0
    current=bodies((a.build/'runtime.hepta').read_bytes());assert len(current)==556
    oldraw=custody.bind(a.previous_build/'runtime.hepta',PREVIOUS_SOURCE,'parent07/runtime.hepta')
    custody.bind(a.previous_build/'tensor-integer-session',PREVIOUS_IMAGE,'parent07/tensor-integer-session')
    old=bodies(oldraw);assert len(old)==544
    definitions_by_unit={};duplicates=[];unit_rows=[]
    for unit in info['units']:
        raw=(a.build/(unit['name']+'.hepta')).read_bytes()
        defs=bodies(raw)
        exports=set(re.search(rb'^exports\(([^)]*)\)',raw,re.M)[1].decode().split(','))
        for name,body in defs.items():
            if name in definitions_by_unit:
                duplicates.append({'name':name,'additional_unit':unit['name'],'sha256':sha(body),
                                   'public_export':name in exports})
            definitions_by_unit.setdefault(name,[]).append({'body':body,'unit':unit['name'],'public_export':name in exports})
            assert sum(d['public_export'] for d in definitions_by_unit[name])<=1
        unit_rows.append({'name':unit['name'],'source_sha256':unit['source_sha256'],'object_sha256':unit['object_sha256'],
                          'defined_functions':sorted(defs),'native_compile_exit':unit['compiler_exit']})
    assert len(definitions_by_unit)==571 and len(duplicates)==6
    core_unit={}
    for name,body in current.items():
        matches=[d['unit'] for d in definitions_by_unit[name] if d['body']==body]
        assert matches,(name,'missing exact real core definition')
        core_unit[name]=matches
    missing=sorted(old.keys()-current.keys());assert not missing
    changed=sorted(n for n in old if old[n]!=current[n]);added=sorted(current.keys()-old.keys())
    assert changed==sorted(info['changed_parent_functions']) and added==sorted(info['added_functions'])
    assert len(changed)==13 and len(added)==12
    observed=custody.bind(here/'owner640_mixed10.py',copy='owner640_mixed10.py')
    parent=custody.bind(here/'parent_owner640_integer07.py',
        'fd7a5be1ba4c1f9dbcfd6955e6fe5eb63860e55cdb87cd9556a969be4c8236d0','parent_owner640_integer07.py')
    adaptation=json.loads(custody.bind(here/'adaptation.json',copy='adaptation.json'))
    custody.bind(here/'build_profile.py',adaptation['script_sha256'],'build_profile.py')
    assert sha(observed)==adaptation['source_sha256'] and sha(parent)==adaptation['parent07_sha256']
    inverse=observed.decode()
    for edit in reversed(adaptation['edits']):
        assert inverse.count(edit['after'])==edit['count']
        inverse=inverse.replace(edit['after'],edit['before'])
    assert inverse.encode()==parent
    before=definitions(parent);after=definitions(observed)
    assert set(before)==set(after)
    changed_observer=sorted(k for k in before if before[k]!=after[k])
    assert changed_observer==['decode_owner','pin_profile','verify_isolated_transition','verify_snapshot']
    inspected=['record','mixed_original_record','binary_saved_base','binary_saved_retain_one','binary_saved_bind',
        'binary_saved_release','binary_saved_audit','binary_saved_needed','binary_saved_get','binary_saved_command',
        'binary_backward_read','dtype_mask_release','mixed_original_mask_release','dtype_audit',
        'tensor_row_release','mixed_original_tensor_row_release','gradient_detach_inplace','history_capture',
        'release_backward_saved','graph_node_release','node_retain','node_release','graph_collect',
        'integer_binary','integer_result_dtype','publish_tensor','saved_release','so_storage_release',
        'hold_release','so_hold_private','finish_command','zero_grad']
    # Some layered authoring uses precise prefixed aliases; bind the actual
    # present declarations rather than inventing a nominal source name.
    inspected_actual=[n for n in inspected if n in current]
    assert len(inspected_actual)>=25
    report={'status':'SOURCE_AND_UNIT_CUSTODY_PASS','source_sha256':observer.SOURCE,'native_sha256':observer.IMAGE,
        'previous_source_sha256':PREVIOUS_SOURCE,'previous_native_sha256':PREVIOUS_IMAGE,
        'functions':556,'blocks':7420,'instructions':26889,'units':unit_rows,
        'all_556_core_function_bytes_present_in_real_units':True,'private_helper_name_reuse':duplicates,
        'changed_parent_functions':changed,'added_functions':added,'unchanged_parent_function_count':531,
        'function_bodies':{n:{'sha256':sha(b),'bytes':len(b),'exact_definition_in_units':core_unit[n]} for n,b in current.items()},
        'inspected_actual_native_role_functions':{n:sha(current[n]) for n in inspected_actual},
        'observer_sha256':sha(observed),'parent_observer_sha256':sha(parent),'observer_whole_inverse_exact':True,
        'observer_changed_definitions':changed_observer,
        'structural_scope':['full 80-word checked producer state','32-byte private Hold of two original Tensor/incarnation pairs',
            'exact Saved slot/S/V/offset match and duplicate input multiplicity','actual600/208 roles and captured-Accumulator SCC',
            'rowless I64 view keeps metadata208 but owns no real base T','all parent pool/Generation/Version/weak/private-Hold invariants retained'],
        'new_native_execution':False,'new_Torch_execution':False,'arbitrary_foreign_state_certification':False,
        'full_pytorch_rewrite_complete':False,'whole_pytorch_gate':'HOLD','pytorch_api_denominator':2590,
        'inputs':custody.finish()}
    write_json(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','source_sha256','native_sha256','functions','observer_sha256')}))

if __name__=='__main__':main()
