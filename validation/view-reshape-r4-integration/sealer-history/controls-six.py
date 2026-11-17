#!/usr/bin/env python3
"""Concrete source-only controls for R4 evidence failure preservation and read races."""
import hashlib, importlib.util, io, json, os, stat, sys, tarfile, traceback
from pathlib import Path
R=Path('/home/qian-qi/heptatorch-continuation-20261116/runs')
SOURCE=R/'view-reshape-r4-integration-01/repo/tools/seal_r4_ci_evidence.py'
PIN='af3179c567a4b7d34a7ef7da15cee8d777963a37fbbdb8dae5ec348b8128583e'
OUT=R/'root-r4-ci-sealer-controls-01'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def meta(path):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':sha(raw),'mode':stat.S_IMODE(path.stat().st_mode)}
def load(name):
    spec=importlib.util.spec_from_file_location(name,SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def fixture(name):
    m=load('sealer_control_'+name);root=OUT/name;repo=root/'repo';stack=root/'stack'
    repo.mkdir(parents=True,exist_ok=False);stack.mkdir()
    for directory in m.DIRECTORIES:
        p=repo/directory;p.mkdir(parents=True,exist_ok=True)
        (p/'fixture-preserved.bin').write_bytes(('fixture '+directory).encode())
    for filename in m.REPORTS:
        dump(repo/filename,{'fixture_only':True,'status':'NOT_RUN','native_executed':False})
    for filename in m.SOURCE_FILES:
        p=repo/filename;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(SOURCE.read_bytes() if filename.endswith('seal_r4_ci_evidence.py') else b'fixture-only source\n')
    targets=[]
    for i,filename in enumerate(m.INPUT_LISTS):
        t=repo/('build/bound-fixture-%d.bin'%i);t.write_bytes(('original-fixture-%d'%i).encode());t.chmod(0o644)
        targets.append(t);dump(repo/filename,{str(t):meta(t)})
    mode_file=repo/'build/differentiable-view-r4-scalar07b/mode500-fixture.bin'
    mode_file.write_bytes(b'fixture-only-mode500');mode_file.chmod(0o500)
    (repo/'build/view-reshape-r4-fixture.make.stdout').write_bytes(b'fixture Make stdout\n')
    return m,repo,stack,targets,mode_file
def run_case(name,kind):
    m,repo,stack,targets,mode_file=fixture(name)
    output=repo/'build/view-reshape-r4-custody';trigger={'done':False}
    original_info=m.info;original_rglob=Path.rglob;original_glob=Path.glob;argv=sys.argv
    if kind=='malformed':
        dump(repo/m.INPUT_LISTS[0],{str(repo/'build/missing-input.bin'):{'bytes':1,'sha256':'0'*64,'mode':0o644}})
        (repo/m.INPUT_LISTS[1]).write_bytes(b'{broken JSON\n')
        wrong=meta(targets[2]);wrong.update(sha256='0'*64,mode=0o600)
        dump(repo/m.INPUT_LISTS[2],{str(targets[2]):wrong})
    if kind in ('payload-race','list-race','archive-read-failure'):
        selected=repo/m.INPUT_LISTS[0] if kind=='list-race' else targets[0]
        def racing_info(path):
            result=original_info(path)
            if Path(path)==selected and not trigger['done']:
                trigger['done']=True
                if kind=='archive-read-failure':selected.unlink()
                elif kind=='list-race':dump(selected,{str(targets[0]):meta(targets[0]),str(targets[1]):meta(targets[1])})
                else:selected.write_bytes(b'actual-file-after-preflight-race')
            return result
        m.info=racing_info
    if kind=='traversal-failure':
        selected=repo/m.DIRECTORIES[0]
        def limited_walk(self,pattern,*args,**kwargs):
            if self==selected:
                trigger['done']=True
                yield self/'fixture-preserved.bin'
                raise OSError('fixture traversal error after one readable member')
            yield from original_rglob(self,pattern,*args,**kwargs)
        Path.rglob=limited_walk
        def limited_glob(self,pattern,*args,**kwargs):
            if self==repo/'build' and pattern.endswith('.make.stdout'):
                yield self/'view-reshape-r4-fixture.make.stdout'
                raise OSError('fixture Make glob error after one readable stream')
            yield from original_glob(self,pattern,*args,**kwargs)
        Path.glob=limited_glob
    try:
        sys.argv=[str(SOURCE),'--repo',str(repo),'--stack-build',str(stack),
                  '--source-commit','0'*40,'--output',str(output)]
        code=m.main()
    finally:
        sys.argv=argv;Path.rglob=original_rglob;Path.glob=original_glob;m.info=original_info
    report=json.loads((output/'report.json').read_bytes())
    manifest=json.loads((output/'file-manifest.json').read_bytes())
    with tarfile.open(output/'complete.tar.xz','r:xz') as tf:
        members={member.name:member for member in tf.getmembers()}
        assert set(members)==set(manifest['members'])
        for name,row in manifest['members'].items():
            raw=tf.extractfile(members[name]).read()
            assert sha(raw)==row['sha256'] and len(raw)==row['bytes'] and members[name].mode==row['mode']
        expected=0 if kind=='valid' else 1
        assert code==expected and report['status']==('PASS' if expected==0 else 'FAIL')
        assert all(name in members for name in m.REPORTS)
        assert members[mode_file.relative_to(repo).as_posix()].mode==0o500
        assert tf.extractfile(mode_file.relative_to(repo).as_posix()).read()==b'fixture-only-mode500'
        stages={row['stage'] for row in report['errors']}
        if kind=='valid':
            assert stages==set() and report['bound_input_count']==4
            assert report['declared_input_count']==4 and len(manifest['input_list_bindings'])==4
        elif kind=='malformed':
            assert {'input-list','declared-input','input-at-archive'}<=stages
            assert tf.extractfile(m.INPUT_LISTS[1]).read()==b'{broken JSON\n'
            assert tf.extractfile(targets[2].relative_to(repo).as_posix()).read()==targets[2].read_bytes()
            assert report['bound_input_count']==1
        elif kind=='payload-race':
            assert trigger['done'] and 'input-at-archive' in stages and report['bound_input_count']==3
            name=targets[0].relative_to(repo).as_posix()
            assert tf.extractfile(name).read()==b'actual-file-after-preflight-race'
            assert manifest['members'][name]['sha256']!=manifest['required_input_bindings'][str(targets[0])]['sha256']
        elif kind=='list-race':
            assert trigger['done'] and 'input-list-at-archive' in stages
            assert manifest['members'][m.INPUT_LISTS[0]]['sha256']!=manifest['input_list_bindings'][str(repo/m.INPUT_LISTS[0])]['sha256']
            assert tf.extractfile(m.INPUT_LISTS[0]).read()==(repo/m.INPUT_LISTS[0]).read_bytes()
        elif kind=='traversal-failure':
            assert trigger['done'] and {'directory-traversal','make-stream-traversal'}<=stages
            assert 'build/view-reshape-r4-fixture.make.stdout' in members
            assert m.DIRECTORIES[0]+'/fixture-preserved.bin' in members
            assert m.DIRECTORIES[-1]+'/fixture-preserved.bin' in members
        elif kind=='archive-read-failure':
            assert trigger['done'] and 'archive-read' in stages and report['bound_input_count']==3
            assert targets[0].relative_to(repo).as_posix() not in members
            assert targets[0].relative_to(repo).as_posix() in report['unarchived_selected_members']
    return {'name':kind,'status':'PASS_EXPECTED_BYTE_CUSTODY_RESULT','sealer_exit_code':code,
            'sealer_status':report['status'],'error_stages':sorted(stages),
            'readable_evidence_archived':True,'all_member_bytes_sizes_modes_verified':True,
            'manifest':meta(output/'file-manifest.json'),'report':meta(output/'report.json'),
            'native_executed':False,'Torch_executed':False,'compiler_executed':False}
def main():
    OUT.mkdir(exist_ok=False)
    result={'schema':'heptatorch.r4.ci-sealer-concrete-controls.v1','status':'RUNNING',
            'source_sha256':PIN,'fixture_only':True,'source_commit_is_synthetic_zero_sha':True,
            'original_production_reports_modified':False,'native_executed':False,
            'Torch_executed':False,'compiler_executed':False,'cases':[]}
    code=1
    try:
        assert sha(SOURCE.read_bytes())==PIN
        for i,kind in enumerate(('valid','malformed','payload-race','list-race','traversal-failure','archive-read-failure')):
            result['cases'].append(run_case('%02d-%s'%(i,kind),kind))
        assert sha(SOURCE.read_bytes())==PIN
        result.update(status='PASS_SIX_CONCRETE_FILE_CUSTODY_CONTROLS',case_count=6)
        code=0
    except BaseException as error:result.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
    dump(OUT/'report.json',result)
    print(json.dumps({'status':result['status'],'cases':len(result['cases']),'error':result.get('error')}))
    return code
if __name__=='__main__':raise SystemExit(main())
