#!/usr/bin/env python3
"""Compose the complete pinned partial-backward core with frozen native RNG.

This creates a new profile. It neither edits nor relabels either frozen parent.
Only the enumerated layout/entry changes may alter parent HIA function bodies.
"""
import argparse,ast,copy,hashlib,json,re,subprocess,sys
from pathlib import Path

SHA=lambda b:hashlib.sha256(b).hexdigest()
RN_MANIFEST='5f0833b00e8f3068da6b60308f9941b7319c77793d10f90df5d779f520e9437b'
BASE09='4e2cc7f7612c826ec35280432be527cf98075af7d0f2c35a560df67e35c3904e'
AUTHOR09='4f3fa887e6372f19777ecde955d39716ae633a1bb9be7408c065d6e94e0fffbd'
OWNER='993651c88e69be93dea49ef473b8409b9f609e5b226c6fcec10de230ef4af99e'
FUNCTION=re.compile(rb'^(fn (\w+)\([^\n]* entry\(\w+\)\n.*?^end\n)',re.M|re.S)
def functions(raw):return {name.decode():body for body,name in FUNCTION.findall(raw)}
def nodes(raw):return {x.name:x for x in ast.parse(raw).body}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tensor09','partial-build','base-pin','rng-package','rng-source','upstream','stack-build','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--entry-authoring',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    old=a.tensor09.resolve();parent=a.partial_build.resolve();rn=a.rng_package.resolve()
    base=json.loads(a.base_pin.read_text());assert base['status']=='PINNED_FOR_COMPOSITION'
    assert base['state_magic']=='HTSTATE5' and base['owner_source_sha256']==OWNER
    manifest=(rn/'source-pins.sha256').read_bytes();assert SHA(manifest)==RN_MANIFEST
    inputs={a.base_pin.resolve():a.base_pin.read_bytes(),Path(__file__).resolve():Path(__file__).read_bytes()}
    for line in manifest.decode().splitlines():
        pin,name=line.split('  ',1);path=(rn/name).resolve();assert path.is_relative_to(rn)
        raw=path.read_bytes();assert SHA(raw)==pin;inputs[path]=raw
    expected={old/'runtime.hepta':BASE09,old/'overlay-authoring.py':AUTHOR09,
        parent/'runtime.hepta':base['runtime_source_sha256'],
        parent/'overlay-authoring.py':base['authoring_source_sha256'],
        parent/base['owner_filename']:OWNER}
    for path,pin in expected.items():
        raw=path.read_bytes();assert SHA(raw)==pin,path;inputs[path]=raw
    old_fns=functions(inputs[old/'runtime.hepta']);parent_fns=functions(inputs[parent/'runtime.hepta'])
    rn_map=json.loads(inputs[rn/'source-map.json']);changed=set(rn_map['changed_baseline_functions'])
    old_magic=str(int.from_bytes(b'HTSTATE3','little')).encode()
    parent_magic=str(int.from_bytes(base['state_magic'].encode(),'little')).encode()
    inherited=[]
    for name in sorted(changed-{'entry'}):
        before=old_fns[name];actual=parent_fns[name]
        if name in ('init_state','state_header'):
            assert old_magic in before;before=before.replace(old_magic,parent_magic)
        assert before==actual,('modified-parent-function-needs-explicit-review',name)
        inherited.append({'name':name,'partial_sha256':SHA(actual),'old09_sha256':SHA(old_fns[name]),
            'identical_except_declared_state_magic':True})
    authored=nodes(inputs[old/'overlay-authoring.py'].decode())
    entry_path=a.entry_authoring.resolve() if a.entry_authoring else parent/'overlay-authoring.py'
    if entry_path not in inputs:inputs[entry_path]=entry_path.read_bytes()
    if a.entry_authoring:assert SHA(inputs[entry_path])==base['entry_authoring_source_sha256']
    parent_nodes=nodes(inputs[entry_path].decode());assert 'entry' in parent_nodes
    original_entry=copy.deepcopy(parent_nodes['entry']);authored['entry']=original_entry
    entry_text=ast.unparse(original_entry)
    assert entry_text.count('next = command(s, o, w, 32 + i * 64, at, end)')==1
    assert 'halted = 1' in entry_text and 'getr(s, 32 + i * 64) == 58' in entry_text
    authoring='\n\n'.join(ast.unparse(x) for x in authored.values())+'\n'
    adapter=out/'adapter';adapter.mkdir()
    (adapter/'runtime.hepta').write_bytes(inputs[parent/'runtime.hepta'])
    (adapter/'overlay-authoring.py').write_text(authoring)
    (adapter/'objects_owners.hepta').write_bytes(inputs[parent/base['owner_filename']])
    rng_authoring=inputs[rn/'rng.authoring'].decode()
    exact_interval='''        if larger > 2130706431:
            if smaller > subf(2139095039,larger):
                return 0'''
    binary64_interval='''        if larger == 2139095039:
            if smaller > 1686110208:
                return 0
        else:
            if larger > 2130706431:
                if smaller > subf(2139095039,larger):
                    return 0'''
    assert rng_authoring.count(exact_interval)==1
    rng_authoring=rng_authoring.replace(exact_interval,binary64_interval)
    before_rng=nodes(inputs[rn/'rng.authoring'].decode());after_rng=nodes(rng_authoring)
    assert {n for n in before_rng if ast.dump(before_rng[n])!=ast.dump(after_rng[n])}=={'rng_interval'}
    (adapter/'rng.authoring').write_text(rng_authoring)
    script=inputs[rn/'build.py'].decode()
    substitutions=[("BASE_SHA='"+BASE09+"'","BASE_SHA='"+base['runtime_source_sha256']+"'"),
        ("AUTHOR_SHA='"+AUTHOR09+"'","AUTHOR_SHA='"+SHA(authoring.encode())+"'"),
        ("new_backing=int.from_bytes(b'HTORNGS1','little');new_main=int.from_bytes(b'HTSTRNG1','little')",
         "new_backing=int.from_bytes(b'HTORNGP1','little');new_main=int.from_bytes(b'HTRNGP01','little')"),
        ("'main_magic':'HTSTRNG1','backing_magic':'HTORNGS1'","'main_magic':'HTRNGP01','backing_magic':'HTORNGP1'")]
    for before,after in substitutions:
        assert script.count(before)==1,before;script=script.replace(before,after)
    (adapter/'build.py').write_text(script)
    report={'schema':'heptatorch.tensor-rng-partial-composition.v1','status':'PREPARED',
        'profile':'heptatorch.tensor-rng.partial.f32.seven-kind.abi3.v1',
        'complete_parent_runtime_sha256':base['runtime_source_sha256'],'parent_state_magic':base['state_magic'],
        'frozen_rn_source_manifest_sha256':RN_MANIFEST,'new_state_magic':'HTRNGP01','new_backing_magic':'HTORNGP1',
        'new_persistent_node_fields':[],'parent_temporaries_reserved':[336,368,376],
        'preserved_parent_presence_field':360,'layout_functions_checked_against_both_parents':inherited,
        'native_completed':False,'all_pytorch_complete':False,
        'range_admission_change':{'from':'exact real difference <= binary32 maximum',
            'to':'binary64 RNE(to-from) <= binary32 maximum',
            'equivalent_exact_bound':'binary32 maximum + 2^74; equality accepted (even significand)',
            'changed_authoring_function':'rng_interval','frozen_rn03_modified':False,
            'before':exact_interval,'after':binary64_interval},
        'input_sha256':{str(path):SHA(raw) for path,raw in inputs.items()},
        'adapter_authoring_sha256':SHA(authoring.encode()),'adapter_script_sha256':SHA(script.encode()),
        'exact_text_adaptations':[{'before':before,'after':after,'expected_occurrences':1} for before,after in substitutions]}
    def save():(out/'composition.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    if a.prepare_only:
        for path,raw in inputs.items():assert path.read_bytes()==raw
        print(json.dumps({'status':'PREPARED','parent':base['runtime_source_sha256']}));return
    result=subprocess.run([sys.executable,'-I','-S','-B',str(adapter/'build.py'),
        '--baseline',str(adapter),'--rng-source',str(a.rng_source.resolve()),'--upstream',str(a.upstream.resolve()),
        '--stack-build',str(a.stack_build.resolve()),'--output',str(out/'native')],capture_output=True)
    (out/'build.stdout').write_bytes(result.stdout);(out/'build.stderr').write_bytes(result.stderr)
    report['build_exit']=result.returncode;save();assert result.returncode==0
    built=json.loads((out/'native/build.json').read_text());combined=functions((out/'native/runtime.hepta').read_bytes())
    actual_changed={name for name,body in parent_fns.items() if combined[name]!=body}
    assert actual_changed==changed,('unexpected-parent-body-changes',actual_changed,changed)
    for name in ('partial_backward_run','object_command'):
        if name in parent_fns:assert combined[name]==parent_fns[name]
    generated_nodes=nodes((out/'native/overlay-authoring.py').read_text())
    restored=copy.deepcopy(generated_nodes['entry']);assert restored.args.args[-1].arg=='scratch';restored.args.args.pop()
    loop=next(x for x in restored.body if isinstance(x,ast.While))
    index=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='code')
    dispatch=loop.body[index+1];assert isinstance(dispatch,ast.If)
    assert ast.unparse(dispatch.test)=='code >= 60 and code <= 69'
    assert ast.unparse(dispatch.orelse[0])=='next = command(s, o, w, 32 + i * 64, at, end)'
    loop.body[index:index+2]=dispatch.orelse
    assert ast.dump(restored,include_attributes=False)==ast.dump(original_entry,include_attributes=False)
    for path,raw in inputs.items():assert path.read_bytes()==raw,path
    report.update(status='NATIVE_BUILD_PASS',native_completed=True,
        compiler_sha256=built['compiler_sha256'],linker_sha256=built['linker_sha256'],
        executable_sha256=built['executable_sha256'],runtime_source_sha256=built['runtime_source_sha256'],
        complete_parent_functions=len(parent_fns),unchanged_parent_functions=len(parent_fns)-len(changed),
        changed_parent_functions=sorted(changed),entry_route_inverse_ast_identical=True,
        all_parent_function_bodies={name:{'before':SHA(body),'after':SHA(combined[name]),'unchanged':body==combined[name]} for name,body in parent_fns.items()})
    save();print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','all_parent_function_bodies','layout_functions_checked_against_both_parents','exact_text_adaptations')}))

if __name__=='__main__':main()
