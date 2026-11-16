#!/usr/bin/env python3
"""Replay frozen numerical tensor suites on the guarded ABI3 image."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--application-build',type=Path,required=True)
p.add_argument('--canonical-binary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();R=a.repo.resolve();B=a.application_build.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
profile=json.loads((Path(__file__).resolve().parent/'profile.json').read_bytes())
build=json.loads((B/'build.json').read_bytes());app=B/'owned-tensor-session'
assert build['pass'] and sha(app.read_bytes())==build['executable_sha256']
assert build['executable_sha256']==profile['current_tensor']['executable_sha256']
assert build['compiler_sha256']==profile['compiler']['executable_sha256']
assert build['linker_sha256']==profile['linkers']['session']['executable_sha256']
assert build['runtime_source_sha256']=='af130eab846b1ae79a11d5227720abe9d75058205acf37c98be3366568ba69cf'
manifest=json.loads((R/'validation/owned-tensor-package.json').read_bytes())
frozen=R/'tools/owned_tensors';peer_base=R/'tests/owned_tensor_review/candidate-03-review/executed-sources'
inputs=[frozen/'test_dynamic.py',frozen/'protocol.py',frozen/'evidence/portable-native-report.json',
        R/'tools/owned_sessions/client.py',peer_base/'review_dynamic.py',peer_base/'protocol.py',peer_base/'client.py',
        peer_base.parent/'report.json',B/'build.json',app,a.canonical_binary.resolve()]
snapshots={path:path.read_bytes() for path in inputs}
for relative,expected in profile['tensor_test_sources_sha256'].items():
    path=R/relative
    assert sha(snapshots[path])==expected,relative
assert sha(snapshots[a.canonical_binary.resolve()])==profile['current_tensor']['canonical_executable_sha256']
for path in (frozen/'test_dynamic.py',frozen/'protocol.py',frozen/'evidence/portable-native-report.json'):
    identity=manifest['files'][path.relative_to(R).as_posix()]
    assert identity['sha256']==sha(snapshots[path]) and identity['bytes']==len(snapshots[path])
sources=O/'executed-sources/owned_tensors';sources.mkdir(parents=True)
transport=sources.parent/'owned_sessions';transport.mkdir()
(transport/'client.py').write_bytes(snapshots[R/'tools/owned_sessions/client.py'])
(sources/'protocol.py').write_bytes(snapshots[frozen/'protocol.py'])
original=snapshots[frozen/'test_dynamic.py'].decode();changed=original
edits=[("assert p.meta(metadata)['version'] == step","assert p.meta(metadata)['version'] == step * (2 if kind == 'AdamW' else 1)"),
       ("assert all(p.meta(record)['version'] == 32 for record in records)","assert [p.meta(record)['version'] for record in records] == [32, 64]")]
for old,new in edits:
    assert changed.count(old)==1;changed=changed.replace(old,new,1)
restored=changed
for old,new in reversed(edits):
    assert restored.count(new)==1;restored=restored.replace(new,old,1)
assert restored==original
(sources/'test_dynamic.py').write_text(changed)
patch=''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='frozen-candidate03/test_dynamic.py',tofile='version-corrected-guarded/test_dynamic.py'))
(O/'version-expectations.patch').write_text(patch)
peer=O/'executed-sources/independent';peer.mkdir()
for name in ('protocol.py','client.py'):(peer/name).write_bytes(snapshots[peer_base/name])
peer_original=snapshots[peer_base/'review_dynamic.py'].decode()
old="TARGET_SHA='05c5544da2b7b34a8db9bd019a3ad0fa930f9ed5c18fd4b7582e7150f21eaeea'"
new="TARGET_SHA='"+build['executable_sha256']+"'"
assert peer_original.count(old)==1
peer_changed=peer_original.replace(old,new,1);assert peer_changed.replace(new,old,1)==peer_original
(peer/'review_dynamic.py').write_text(peer_changed)
(O/'peer-image-pin.patch').write_text(''.join(difflib.unified_diff(peer_original.splitlines(True),peer_changed.splitlines(True),fromfile='frozen-peer/review_dynamic.py',tofile='guarded-peer/review_dynamic.py')))
executed={path:path.read_bytes() for path in (O/'executed-sources').rglob('*') if path.is_file()}
commands=[('native',sources/'test_dynamic.py',['--service',app,'--build-report',B/'build.json','--output',O/'native']),
          ('independent',peer/'review_dynamic.py',['--binary',app,'--canonical-binary',a.canonical_binary.resolve(),'--transport-observer',transport/'client.py','--output',O/'independent'])]
for name,script,arguments in commands:
    process=subprocess.run([sys.executable,'-I','-S','-B',str(script),*map(str,arguments)],capture_output=True,timeout=240)
    (O/(name+'.stdout')).write_bytes(process.stdout);(O/(name+'.stderr')).write_bytes(process.stderr)
    assert process.returncode==0 and not process.stderr,(name,process.returncode,process.stderr[-2000:])
    print(name+': PASS',flush=True)
native=json.loads((O/'native/report.json').read_bytes());peer_report=json.loads((O/'independent/report.json').read_bytes())
assert native['pass'] and native['cases']==11 and native['response_frames']==161
assert native['compared_binary32_values']==1078028 and native['rollback_compared_owned_bytes']==59310080
normalize=lambda rows:[{k:v for k,v in row.items() if k!='elapsed_seconds'} for row in rows]
assert normalize(native['rows'])==normalize(json.loads(snapshots[frozen/'evidence/portable-native-report.json'])['rows'])
assert peer_report['status']=='PASS' and peer_report['interactive_frames']==16
assert peer_report['exact_scalar_or_f32_fields_compared']==612
assert peer_report['entire_owner_rollback_bytes_compared']==41943040
assert peer_report['numerical_checks']==json.loads(snapshots[peer_base.parent/'report.json'])['numerical_checks']
for path,raw in {**snapshots,**executed}.items():assert path.read_bytes()==raw,path
report={'schema':'heptatorch.stack-budget-current-tensor-qualification.v1','status':'PASS',
        'executable_sha256':build['executable_sha256'],'build_report_sha256':sha(snapshots[B/'build.json']),
        'native_cases':11,'native_response_frames':161,'compared_binary32_values':1078028,
        'native_rollback_bytes':59310080,'independent_response_frames':16,
        'independent_scalar_fields':612,'independent_rollback_bytes':41943040,
        'reports':{'native':native,'independent':peer_report},'numerical_assertions_and_tolerances_unchanged':True,
        'prior_version_expectation_edits':edits,'peer_edited_only_by_image_identity':True,
        'source_and_image_rechecked_after':True,'real_pytorch_oracle_modified':False,
        'executed_source_sha256':{path.relative_to(O).as_posix():sha(raw) for path,raw in executed.items()},
        'reviewer_sha256':sha(Path(__file__).read_bytes())}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('reports','executed_source_sha256')},indent=2))
