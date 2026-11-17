#!/usr/bin/env python3
"""Verify final R4 integration syntax, unchanged inherited workflow and dry-run routes."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys,traceback
import yaml
R=Path('/home/qian-qi/heptatorch-continuation-20261116/runs')
REPO=R/'view-reshape-r4-integration-01/repo'
SOURCE=R/'root-r4-integration-static-source-03'
OUT=R/'root-r4-integration-static-03'
PINS={'Makefile':'1ea6a94d61840ccf111678034750217c91219bd87ed2fa18d3f936d22cddcebb',
'.github/workflows/native.yml':'c5866f930b0f3b97ece4260f9c3b704b09477a54fe212f823f14fe346644c81d',
'tools/seal_r4_ci_evidence.py':'a6ba4158f74c301c0103002a1a7c2a4f800cff85f913df767ffe5130fdfb0eca'}
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def command(label,args):
    child=subprocess.run(args,cwd=REPO,stdin=subprocess.DEVNULL,capture_output=True,timeout=60)
    (OUT/(label+'.stdout')).write_bytes(child.stdout);(OUT/(label+'.stderr')).write_bytes(child.stderr)
    result={'argv':args,'exit_code':child.returncode,'stdout_bytes':len(child.stdout),'stdout_sha256':sha(child.stdout),'stderr_bytes':len(child.stderr),'stderr_sha256':sha(child.stderr)}
    report['commands'][label]=result
    assert child.returncode==0 and not child.stderr,(label,result)
    return child.stdout.decode()
OUT.mkdir(exist_ok=False)
report={'schema':'heptatorch.root.r4-final-integration-static.v1','status':'RUNNING','native_executed':False,'compiler_executed':False,'Torch_executed':False,'emitter_executed':False,'commands':{},'pins':PINS,'closed_schema_count':0,'whole_pytorch_gate':'HOLD'}
code=1
try:
    for name,pin in PINS.items():assert sha((REPO/name).read_bytes())==pin,name
    workflow=yaml.safe_load((REPO/'.github/workflows/native.yml').read_bytes())
    previous=yaml.safe_load((SOURCE/'base-native.yml').read_bytes())
    old_steps=previous['jobs']['native']['steps'];steps=workflow['jobs']['native']['steps']
    assert len(old_steps)==116 and len(steps)==126,(len(old_steps),len(steps))
    assert workflow['jobs']['native']['timeout-minutes']==previous['jobs']['native']['timeout-minutes']==90
    byname={s['name']:s for s in steps};assert len(byname)==len(steps)
    old_names=[s['name'] for s in old_steps]
    assert [s['name'] for s in steps if s['name'] in old_names]==old_names
    for s in old_steps:assert byname[s['name']]==s,('inherited workflow changed',s['name'])
    seal=byname['Seal complete R4 file bytes and original modes even after failure']
    upload=byname['Export complete R4 builds references and all 76 native sessions']
    assert seal['if']=='always()' and upload['if']=='always()'
    assert steps.index(seal)<steps.index(byname['Index complete original CI reports and checkout identity'])<steps.index(upload)
    assert 'build/view-reshape-r4-custody.make.' not in seal['run']
    assert '--source-commit "$GITHUB_SHA"' in seal['run']
    for name in ['resource3','backward-resource','custody']:
        assert 'build/view-reshape-r4-'+name+'/' in upload['with']['path']
        assert 'build/view-reshape-r4-'+name+'.json' in '\n'.join(s.get('run','') for s in steps)
    for name in ['build/r4-custody-seal.stdout','build/r4-custody-seal.stderr']:assert name in upload['with']['path']
    parsed={}
    roots=['tools/differentiable_view_r4_build_review','tools/differentiable_view_shape_r4','tools/differentiable_view_scalar_r4',
           'tests/differentiable_view_shape_review','tests/differentiable_view_shape_reference_review',
           'tests/wrapped_scalar_resource_review','tests/backward_private_cache_resource_review']
    paths={REPO/'tools/seal_r4_ci_evidence.py'}
    for root in roots:paths.update((REPO/root).rglob('*.py'))
    for path in sorted(paths):
        raw=path.read_bytes();ast.parse(raw,filename=str(path));parsed[str(path.relative_to(REPO))]={'bytes':len(raw),'sha256':sha(raw)}
    text=command('resources-make-dry-run',['make','-n',
        '-o','build/differentiable-view-r4-scalar07b/success.json',
        '-o','build/view-reshape-r4-reference32/.complete',
        'test-differentiable-view-r4-resource3','test-differentiable-view-r4-backward-resource'])
    for wrapper in ['tests/wrapped_scalar_resource_review/run_resources.py','tests/backward_private_cache_resource_review/run_resource.py']:
        assert 'python3 -I -S -B '+wrapper in text
    assert text.count('--expected-build-sha256 "$(sha256sum ')==2
    assert '--prepare-only' not in text and 'hia-native-compiler' not in text
    text76=command('native76-make-dry-run',['make','-n',
        '-o','build/differentiable-view-r4-scalar07b/success.json',
        '-o','build/view-reshape-r4-reference26/.complete','-o','build/view-reshape-r4-reference32/.complete',
        '-o','build/differentiable-view-reference14/report.json','-o','build/differentiable-view-input4-reference/report.json',
        'test-differentiable-view-r4-native76'])
    for name in ['replay_shape26_epoch_r1.py','replay_wrapped_scalar32_r1.py','replay_direct_view14_profile.py','replay_input_boundaries4.py']:
        assert 'python3 -B tests/' in text76 and name in text76
    assert text76.count('--expected-functions "658"')==4 and text76.count('--expected-native-units "13"')==4
    command('git-diff-check',['git','diff','--check'])
    for name,pin in PINS.items():assert sha((REPO/name).read_bytes())==pin
    report.update(status='PASS_SOURCE_SYNTAX_WORKFLOW_AND_MAKE_DRY_RUN',inherited_steps_unchanged=116,total_workflow_steps=126,
                  added_steps=10,python_sources=parsed,source_count=len(parsed),required_r4_reports=11,all_inputs_unchanged=True)
    code=0
except BaseException as e:report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
dump(OUT/'report.json',report)
print(json.dumps({k:report.get(k) for k in ['status','error','source_count','total_workflow_steps']}))
raise SystemExit(code)
