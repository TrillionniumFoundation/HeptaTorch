#!/usr/bin/env python3
"""Prepare a separate full84-word fixed35 profile; never execute native or Torch."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import sys

PARENT_OWNER='8361f28ab6a6548ec0b3d4b97fb0f5b7c96788bcbfc6ba96fe644f5c2834bb3f'
PARENT_SOURCE='5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49'
PARENT_IMAGE='69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6'
PARENT_PROFILE='mixed-integer-556-fixed35-F32-Saved'
PARENT_PROFILE_SHA='1ac4bdf9bc28df618d2fa959acf2732916c196d10143a69a7e3967b37e4ecbea'
PARENT_GENERAL='aa8a9c2faf0fccabe0acb42d3f21e316c10c5c8b8626baa323bd8f186bead7ea'
PARENT_STRUCTURAL='58c99c9d43b7af30f35374ea5e529328bee782f50c44596e180df908b8b1aebf'
LEGACY='281f1ea437046a9cd884de2c8da34f38649a3b6482ade23123ac4aa992faed64'
REFUSAL='6ea9e8b33164089034a2a71a03f4850e1ec18ed1a09155d7841f700ef311c833'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def bodies(raw):
    found=re.findall(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',raw,re.M|re.S)
    result={name.decode():body for body,name in found}
    assert found and len(result)==len(found)
    return result


def assignment(raw,name):
    tree=ast.parse(raw)
    values=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
            and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
    assert len(values)==1
    return values[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent-directory','general-owner','legacy-refusal','native-build','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-source-sha256',required=True)
    p.add_argument('--expected-native-sha256',required=True)
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=out/'tests/tensor_to_frozen35_accounting';root.mkdir(parents=True)
    original=out/'parent-files';original.mkdir()
    inputs={}
    def read(path,pin=None):
        path=path.resolve();raw=path.read_bytes()
        assert pin is None or sha(raw)==pin,(str(path),sha(raw),pin)
        inputs[str(path)]={'bytes':len(raw),'sha256':sha(raw),'mode':path.stat().st_mode&511}
        return raw
    read(Path(__file__))
    source=read(a.native_build/'runtime.hepta',a.expected_source_sha256)
    read(a.native_build/'tensor-to-copy-session',a.expected_native_sha256)
    build=json.loads(read(a.native_build/'build.json'))
    assert build['status']=='PASS' and build['source_sha256']==a.expected_source_sha256
    assert build['native_sha256']==a.expected_native_sha256 and build['descriptor_bytes']==672
    assert len(build['units'])==12 and build['link_exit']==0
    fs=bodies(source);assert len(fs)==build['functions']
    profile=f'tensor-to-copy-{len(fs)}-fixed35-F32-Saved'
    general=read(a.general_owner)
    assert assignment(general,'SOURCE')==a.expected_source_sha256
    assert assignment(general,'IMAGE')==a.expected_native_sha256
    assert assignment(general,'DESCRIPTOR')==672
    general_profile=assignment(general,'PROFILE')
    before=f'PROFILE = "{general_profile}"'.encode();after=f'PROFILE = "{profile}"'.encode()
    assert general.count(before)==1
    structural=general.replace(before,after)
    assert structural.replace(after,before)==general
    (root/'owner672_general.py').write_bytes(general)
    (root/'owner672_fixed35.py').write_bytes(structural)
    legacy=read(a.parent_directory/'legacy_owner_rules.py',LEGACY)
    (root/'legacy_owner_rules.py').write_bytes(legacy)
    (root/'legacy_owner640_refusal.py').write_bytes(read(a.legacy_refusal,REFUSAL))
    oldprofile=json.loads(read(a.parent_directory/'profile.json',PARENT_PROFILE_SHA))
    (original/'profile.json').write_bytes((a.parent_directory/'profile.json').read_bytes())
    correction=read(a.parent_directory/'storage-tombstone-correction.json',
        '59ece06f7fef4736dd835f78bd6159c055a30f4d91f5ebd579bf109aae3c3577')
    (root/'storage-tombstone-correction.json').write_bytes(correction)
    structural_proof={'parent_general_source_sha256':sha(general),
        'new_profile_source_sha256':sha(structural),'only_change':'PROFILE name literal',
        'whole_inverse_exact':True,'all_checker_function_bodies_unchanged':True,
        'descriptor_bytes':672,'descriptor_words':84}
    (root/'structural-profile-adaptation.json').write_text(json.dumps(structural_proof,indent=2)+'\n')
    changes={}
    def derive(src,dest,edits):
        raw=read(a.parent_directory/src);(original/src).write_bytes(raw)
        text=raw.decode();rows=[]
        for label,before,after,count in edits:
            count=text.count(before) if count is None else count
            assert count and text.count(before)==count,(src,label,text.count(before),count)
            text=text.replace(before,after)
            rows.append({'label':label,'before':before,'after':after,'count':count})
        inverse=text
        for row in reversed(rows):
            assert row['after'] and inverse.count(row['after'])==row['count'],(dest,row['label'])
            inverse=inverse.replace(row['after'],row['before'])
        assert inverse.encode()==raw
        ast.parse(text)
        (root/dest).write_text(text)
        changes[dest]={'parent_file':src,'before_sha256':sha(raw),'after_sha256':sha(text.encode()),
                       'whole_file_inverse_exact':True,'edits':rows}
        return text
    oldraw=read(a.parent_directory/'owner_accounting_mixed.py',PARENT_OWNER)
    owner_edits=[
        ('new-header','640-byte','672-byte',None),
        ('whole84-doc','full 80-word','full 84-word',None),
        ('whole84-oldcomment','Frozen640 reads','Frozen672 reads',1),
        ('whole84-dual-comment','complete 640 structural','complete 672 structural',1),
        ('general-local-dependency','HERE.parent / "mixed_integer_owner_review" / "owner640_mixed10.py"',
            'HERE / "owner672_general.py"',1),
        ('structural-local-dependency','"owner640_saved_fixed35.py"','"owner672_fixed35.py"',1),
        ('general-exact-pin',PARENT_GENERAL,sha(general),1),
        ('structural-exact-pin',PARENT_STRUCTURAL,sha(structural),1),
        ('structural-module-name','"frozen35_structural_owner640"','"frozen35_structural_owner672"',1),
        ('width-data','structural.DESCRIPTOR == 640','structural.DESCRIPTOR == 672',1),
        ('width-constant','structural.BASE, 640, structural.TAG','structural.BASE, 672, structural.TAG',1),
        ('proof-width-key','full_640_snapshot_no_projection','full_672_snapshot_no_projection',1),
        ('full84-width-predicate','"len(node) == 80"','"len(node) == 84"',1),
        ('proof-code-name','::<explicit-640-Saved-binding>','::<explicit-672-Saved-binding>',1),
        ('fixed-domain-full-tail-zero','(608,616,624,632)','(608,616,624,632,640,648,656,664)',1),
        ('structural-custody-key','structural_owner640','structural_owner672',None),
        ('all-source-functions','len(bodies) == 556',f'len(bodies) == {len(fs)}',1),
        ('exact-source-function-set','    for name, expected in CONFIG["function_pins"].items():\n',
            '    need(set(bodies) == set(CONFIG["function_pins"]), "native-function-pin-coverage")\n'
            '    for name, expected in CONFIG["function_pins"].items():\n',1),
    ]
    owner=derive('owner_accounting_mixed.py','owner_accounting_to.py',owner_edits)
    constructors=('linear','mse','layernorm','cross_entropy_probs','batchnorm_eval',
                  'binary','matmul','cat','stack','indexed_add')
    for name in constructors:
        assert sha(fs[name])==oldprofile['function_pins'][name],('constructor changed',name)
    targets=list(oldprofile['capture_callers'])+[
        'to_view_root_retain','to_view_base_release','to_view_refresh','to_view_public',
        'to_view_edge_flag_at','to_zero_factory_incoming']
    calls={target:sorted(name for name,body in fs.items()
                        if ('"function":"'+target+'"').encode() in body) for target in targets}
    cfg={**oldprofile,'schema':'heptatorch.frozen35.to672-F32-Saved-profile.v1',
        'source_sha256':a.expected_source_sha256,'native_sha256':a.expected_native_sha256,
        'descriptor_bytes':672,'structural_observer_sha256':sha(structural),
        'original_structural_observer_sha256':sha(general),
        'parent_profile_sha256':PARENT_PROFILE_SHA,'parent_source_sha256':PARENT_SOURCE,
        'parent_native_sha256':PARENT_IMAGE,'parent_observer_sha256':PARENT_OWNER,
        'function_pins':{name:sha(body) for name,body in fs.items()},'capture_callers':calls,
        'structural_claim':'Complete actual84word nodes, all general672 structural rules, unchanged original-T Saved32B/600/208 equations. No snapshot projection or residual allocation count.',
        'fixed35_domain':'Only original35 commands1..56 and F32. Ordinary mul/div Saved32B roots are real and counted. New ViewInfo640..664, dtype/history608/616/624/632 and NativeDropout mask roles must all be absent in this fixed old command domain; they are checked, not ignored.',
        'program_descriptor_counts_unchanged':True,
        'constructor_functions_exact_parent':{name:sha(fs[name]) for name in constructors},
        'native_and_Torch_executed_during_preparation':False}
    (root/'profile.json').write_text(json.dumps(cfg,indent=2)+'\n')
    profile_sha=sha((root/'profile.json').read_bytes())
    launcher=derive('run_frozen35_accounted.py','run_frozen35_accounted.py',[
        ('new-import','owner_accounting_mixed','owner_accounting_to',1),
        ('full-width-description','full80-word owner under640','full84-word owner under672',1),
        ('full-width-scope','556/640 owner, complete actual80-word snapshots',
            f'{len(fs)}/672 owner, complete actual84-word snapshots',1),
    ])
    derive('check_owner_snapshots.py','check_owner_snapshots.py',[
        ('new-accounting-pin',PARENT_OWNER,sha(owner.encode()),1),
        ('new-profile-name',PARENT_PROFILE,profile,1),
    ])
    derive('check_frozen35_accounted.py','check_frozen35_accounted.py',[
        ('current-image',PARENT_IMAGE,a.expected_native_sha256,1),
        ('current-source',PARENT_SOURCE,a.expected_source_sha256,1),
        ('current-name',PARENT_PROFILE,profile,1),
        ('accounting-pin',PARENT_OWNER,sha(owner.encode()),1),
        ('launcher-pin','c8396130466038036cd383afa46a7cb51fae5bea2032eb10d122842754c2a773',sha(launcher.encode()),1),
        ('structural-pin',PARENT_STRUCTURAL,sha(structural),1),
        ('general-pin',PARENT_GENERAL,sha(general),1),
        ('profile-pin',PARENT_PROFILE_SHA,profile_sha,1),
        ('structural-custody-key','structural_owner640','structural_owner672',None),
        ('width-data','["descriptor_bytes"]==640','["descriptor_bytes"]==672',1),
        ('proof-width-key','full_640_snapshot_no_projection','full_672_snapshot_no_projection',1),
    ])
    width_old='''        # Pure old-predicate replay under the new snapshot identity. The old
        # file and every function body remain byte-identical; only these four
        # module data bindings identify this profile's copied JSON snapshot.
'''
    width_new='''        # Replay the old refusal on the complete84-word copied snapshot.
        # The frozen6ea file is unchanged. Only its descriptor-width literal
        # is adapted in memory; no node is projected, and refusal rules stay.
        legacy_raw = LEGACY_REFUSAL.read_bytes()
        legacy_tree = ast.parse(legacy_raw)
        legacy_fn = next(n for n in legacy_tree.body
                         if isinstance(n, ast.FunctionDef) and n.name == "verify_snapshot")
        legacy_text = ast.get_source_segment(legacy_raw.decode(), legacy_fn)
        old_width, new_width = "len(node) == 80", "len(node) == 84"
        assert legacy_text.count(old_width) == 1
        adapted_text = legacy_text.replace(old_width, new_width)
        assert adapted_text.replace(new_width, old_width) == legacy_text
        adapted_fn = ast.parse(adapted_text).body[0]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[adapted_fn], type_ignores=[])),
                     str(LEGACY_REFUSAL)+"::<only-width84>", "exec"), original.__dict__)
        legacy_descriptor_bytes = original.DESCRIPTOR
        original.DESCRIPTOR = observer.structural.DESCRIPTOR
        report["historical_predicate_width_adapter"] = {
            "before_ast_sha256": observer.sha(ast.dump(legacy_fn, include_attributes=False).encode()),
            "after_ast_sha256": observer.sha(ast.dump(adapted_fn, include_attributes=False).encode()),
            "only_function_literal": [old_width, new_width],
            "original_descriptor_bytes": legacy_descriptor_bytes,
            "descriptor_bytes": original.DESCRIPTOR, "snapshot_projection": False,
            "whole_function_inverse_exact": True, "historical_file_unchanged": True}
'''
    derive('test_storage_tombstone_legacy_refusal.py','test_storage_tombstone_legacy_refusal.py',[
        ('new-import','owner_accounting_mixed','owner_accounting_to',1),
        ('add-AST-import','import argparse\n','import argparse\nimport ast\n',1),
        ('local-exact-historical-refusal','HERE.parent / "dtype_owner_accounting" / "owner640.py"',
            'HERE / "legacy_owner640_refusal.py"',1),
        ('full84-historical-predicate-width',width_old,width_new,1),
        ('retain-old-width-data-identity',
            'for name in ("SCHEMA", "SOURCE", "IMAGE", "PROFILE")}\n',
            'for name in ("SCHEMA", "SOURCE", "IMAGE", "PROFILE")}\n'
            '        historical_identity["DESCRIPTOR"] = legacy_descriptor_bytes\n',1),
        ('declare-fifth-data-binding',
            '"changed_module_data_names": ["SCHEMA", "SOURCE", "IMAGE", "PROFILE"]',
            '"changed_module_data_names": ["SCHEMA", "SOURCE", "IMAGE", "PROFILE", "DESCRIPTOR"]',1),
        ('report-one-explicit-code-adapter','"function_code_changed": False',
            '"function_code_changed": True, "only_function_code_change": "descriptor width80 to84"',1),
    ])
    # Import only the prepared pure checker; this executes AST/inverse checks,
    # never target, compiler, /proc, original oracle entry, or Torch.
    path=root/'owner_accounting_to.py'
    spec=importlib.util.spec_from_file_location('prepared672_fixed35_owner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bound=module.pin_profile(a.native_build/'tensor-to-copy-session',a.native_build/'runtime.hepta')
    proof=module.inverse_proof()
    assert proof['full_672_snapshot_no_projection'] is True
    assert len(proof['functions'])==10
    for path,row in inputs.items():
        p=Path(path);assert p.stat().st_size==row['bytes'] and sha(p.read_bytes())==row['sha256']
        assert p.stat().st_mode&511==row['mode']
    report={'schema':'heptatorch.to672-fixed35-preparation.v1','status':'PREPARED_NOT_RUN',
        'profile':bound,'source_sha256':a.expected_source_sha256,
        'native_sha256':a.expected_native_sha256,'functions':len(fs),'changes':changes,
        'structural_adaptation':structural_proof,'inputs':inputs,'inputs_unchanged':True,
        'pure_import_and_inverse_check_executed':True,'native_executed':False,'Torch_executed':False,
        'original35_numeric_Version_EOF_tolerances_unchanged':True,
        'original18_named_negative_rules_unchanged':True,
        'new84word_transition_qualified':False,'whole_pytorch_gate':'HOLD',
        'pytorch_api_denominator':2590,'closed_schema_count':0,'full_pytorch_rewrite_complete':False}
    (out/'source-adaptation.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'executed-builder.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':'PREPARED_NOT_RUN','profile':profile,'functions':len(fs),
        'observer_sha256':sha(owner.encode()),'profile_sha256':profile_sha,
        'source_adaptation_sha256':sha((out/'source-adaptation.json').read_bytes())}))


if __name__=='__main__':main()
