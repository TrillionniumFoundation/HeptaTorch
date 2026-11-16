#!/usr/bin/env python3
"""Cold-build523 and integrate immutable, separately counted native gates.

This orchestrates existing producers and independent reviewers. It contains no
Tensor implementation and no numerical oracle. There is deliberately no mode
that substitutes an existing523 executable for the cold producer invocation.
Live32 and frozen35 remain separate mandatory CI executions.
"""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
import traceback

SOURCE='6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3'
IMAGE='830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51'
PARENT_SOURCE='666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b'
PARENT_IMAGE='7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5'
COMPILER='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FLAT='c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6'
MANIFESTS=(
 ('validation/native-dropout/production-handoff-manifest.json','5210dfd212d9c81bdbc00780dab0c7eef27fa5ff717223825f7004466f9e3d33',123),
 ('tools/native_dropout/package-manifest.json','3e087c9ddeae0e34f29120ee9e05632e3048aaf5d17b449330faaaed15633708',99),
 ('validation/native-dropout-identity9/package-manifest.json','1d15826630aba1913033bbd77099244a08ce9993966e69d5b37997b7d45c9e28',17),
 ('validation/dtype-owner-accounting/package-manifest.json','fcc157b500430f3b118d1ca1c8fa775024a85a44ed525fba9ff7e5382bf45c86',31),
)
PRODUCER_STEPS=(
 'cold-native-build','typed-slice','typed-lifetimes','bool-logical-not','dtype-admission101',
 'owned-allocation','history-mul-controls8','actual-native-dropout28','original-native-dropout8',
 'actual-existing-optimizer6','actual-sgd-effective-layout6','actual-retained12','inherited16')
PROVENANCE={'python_matches','platform_matches','version_backend_matches','git_matches_oracle_build_pin',
 'distribution_version_matches','direct_url_matches','direct_url_archive_sha256_matches',
 'pip_report_url_matches','pip_report_archive_sha256_matches','pip_report_version_matches','pip_report_requested'}


def require(ok, why):
    if not ok: raise AssertionError(why)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def unique(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,('duplicate JSON key',k));out[k]=v
    return out


def decode(path):return json.loads(Path(path).read_bytes(),object_pairs_hook=unique)


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def relative(root,name):
    p=PurePosixPath(name)
    require(name and not p.is_absolute() and '..' not in p.parts and str(p)==name,('unsafe relative path',name))
    return root/name


class Custody:
    def __init__(self):self.files={}
    def bind(self,path,expected=None,size=None,mode=None):
        p=Path(path).resolve();require(p.is_file(),('missing input',str(p)))
        now={'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode),'sha256':digest(p)}
        require(expected is None or now['sha256']==expected,('SHA',str(p),now['sha256'],expected))
        require(size is None or now['bytes']==size,('length',str(p)))
        require(mode is None or now['mode']==mode,('mode',str(p)))
        require(p not in self.files or self.files[p]==now,('input changed',str(p)))
        self.files[p]=now;return now
    def read(self,path,expected=None):self.bind(path,expected);return decode(path)
    def finish(self):
        for p,v in list(self.files.items()):self.bind(p,v['sha256'],v['bytes'],v['mode'])


def manifest(repo,name,pin,count,custody):
    m=custody.read(relative(repo,name),pin);rows=m['files']
    rows=[dict(v,path=k) for k,v in rows.items()] if isinstance(rows,dict) else rows
    require(len(rows)==count and len({x['path'] for x in rows})==count,('manifest count',name))
    relocations={}
    for row in rows:
        target=row['path']
        if name=='validation/native-dropout/production-handoff-manifest.json' and target=='handoff.json':
            relocations[target]='validation/native-dropout/production-handoff.json';target=relocations[target]
        custody.bind(relative(repo,target),row['sha256'],row.get('bytes',row.get('size')),row.get('mode'))
    return {'path':name,'sha256':pin,'members':count,'explicit_metadata_relocations':relocations,'all_bytes_modes_lengths_verified':True}


def runtime_identity(report):
    sources=[report[x] for x in ('source_sha256','native_source_sha256','runtime_source_sha256') if x in report]
    images=[report[x] for x in ('native_sha256','binary_sha256','executable_sha256','native_executable_sha256') if x in report]
    require(sources and all(x==SOURCE for x in sources),'report exact source')
    require(images and all(x==IMAGE for x in images),'report exact executable')


def build_identity(directory,profile,custody,parent=False):
    r=custody.read(directory/'build.json')
    require(r['status']=='PASS' and r['link_exit']==0,'actual native build success')
    for name in ('source_sha256','native_sha256','functions','blocks','instructions','descriptor_bytes','state_format','backing_format','units'):
        require(r[name]==profile[name],('build profile',name))
    require(r['compiler_sha256']==COMPILER and r['linker_sha256']==LINKER,'build tool pins')
    require(len(r['units'])==(9 if parent else 11),'actual native module count')
    if parent:require(r['module_count']==9,'parent declared native module count')
    custody.bind(directory/'runtime.hepta',PARENT_SOURCE if parent else SOURCE)
    filename='tensor-retained-session' if parent else 'tensor-dtype-dropout-session'
    custody.bind(directory/filename,PARENT_IMAGE if parent else IMAGE)
    require(len({x['name'] for x in r['units']})==len(r['units']),'unique actual module names')
    for row in r['units']:
        require(Path(row['name']).name==row['name'] and row['compiler_exit']==0,'actual compiled module')
        custody.bind(directory/(row['name']+'.hepta'),row['source_sha256'])
        custody.bind(directory/(row['name']+'.o'),row['object_sha256'])
    return {'build_report_sha256':digest(directory/'build.json'),'source_sha256':r['source_sha256'],
            'native_sha256':r['native_sha256'],'actual_units_verified':len(r['units'])}


def producer_identity(root,custody):
    r=custody.read(root/'report.json');runtime_identity(r)
    require(r['status']=='PASS' and r['step_count']==len(r['steps'])==13,'producer13 completion')
    require(tuple(x['name'] for x in r['steps'])==PRODUCER_STEPS,'all exact producer13 steps in order')
    require(r['package_manifest_sha256']==MANIFESTS[1][1],'producer execution-input manifest')
    require(r['all_package_members_unchanged'] is True and r['all_expected_statuses_and_counts_matched'] is True,'producer custody')
    for row in r['steps']:
        expected='QUALIFIED_EXPOSED_SCOPE' if row['name']=='actual-retained12' else 'PASS'
        require(row['exit_code']==0 and row['status']==row['expected_status']==expected,('producer child status',row['name']))
        custody.bind(root/(row['name']+'.stdout'),row['stdout_sha256'])
        custody.bind(root/(row['name']+'.stderr'),row['stderr_sha256'])
        child=custody.read(relative(root,row['report']),row['report_sha256']);runtime_identity(child)
        require(child['status']==expected,'producer child original report')
        if row['name']=='actual-retained12':
            require(child['variants_executed']==child['variants_qualified']==12 and child['unexpected_failures']==0,'all retained12')
            require(child['inputs']['observations']=='ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3','retained actual pin')
            require(set(child['reference_provenance_checks'])==PROVENANCE and all(x is True for x in child['reference_provenance_checks'].values()),'retained eleven provenance')
            require(child['all_inputs_unchanged'] is True,'retained reference/source unchanged')
        elif row['name']=='inherited16':
            require(child['step_count']==16 and child['all_sources_unchanged'] is True,'all inherited16 without relabeling')
    return {'report_sha256':digest(root/'report.json'),'steps':13,'all_exact_labels_admitted':True,'original_inherited_partial_reports_preserved':True}


def archive_evidence(out):
    excluded={'report.json','evidence-manifest.json','raw-evidence.tar.xz'}
    files={}
    for p in sorted(out.rglob('*')):
        require(not p.is_symlink(),('evidence symlink',str(p)))
        if not p.is_file():continue
        name=p.relative_to(out).as_posix()
        if name in excluded:continue
        files[name]={'bytes':p.stat().st_size,'sha256':digest(p),'mode':stat.S_IMODE(p.stat().st_mode)}
    archive=out/'raw-evidence.tar.xz'
    with lzma.open(archive,'wb',preset=6) as stream,tarfile.open(fileobj=stream,mode='w|',format=tarfile.PAX_FORMAT) as tar:
        for name,pin in files.items():
            p=relative(out,name);require(digest(p)==pin['sha256'],'archive source unchanged')
            info=tarfile.TarInfo(name);info.size=pin['bytes'];info.mode=pin['mode'];info.mtime=0
            with p.open('rb') as f:tar.addfile(info,f)
    with tarfile.open(archive,'r:xz') as tar:
        members=tar.getmembers();require(len(members)==len(files) and {x.name for x in members}==set(files),'archive membership')
        for member in members:
            pin=files[member.name];require(member.size==pin['bytes'] and member.mode&0o777==pin['mode'],'archive metadata')
            h=hashlib.sha256()
            with tar.extractfile(member) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            require(h.hexdigest()==pin['sha256'] and digest(relative(out,member.name))==pin['sha256'],'archive exact bytes')
    result={'schema':'heptatorch.native-dropout-fresh-raw-evidence.v1','status':'VERIFIED','source_sha256':SOURCE,'native_sha256':IMAGE,
      'members':len(files),'uncompressed_bytes':sum(x['bytes'] for x in files.values()),'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,
      'all_member_bytes_modes_sizes_verified':True,'scope':'Byte custody; this status is not suite admission.','files':files}
    dump(out/'evidence-manifest.json',result)
    return {'archive':'raw-evidence.tar.xz','archive_sha256':result['archive_sha256'],'archive_bytes':result['archive_bytes'],
      'manifest':'evidence-manifest.json','manifest_sha256':digest(out/'evidence-manifest.json'),'members':result['members'],'all_members_verified':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('repo','parent-build','upstream','compiler','linker','flat-compiler','stack-build','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();repo=a.repo.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=a.parent_build.resolve();stack=a.stack_build.resolve();custody=Custody()
    report={'schema':'heptatorch.native-dropout-native-integration.v1','status':'RUNNING','source_sha256':SOURCE,'native_sha256':IMAGE,
      'producer_mode':'FRESH_COLD_REQUIRED','producer_executed_this_invocation':False,'Torch_executed_here':False,'commands':[],
      'counting_boundary':'Named suites overlap and remain separately counted, not unique upstream tests or APIs.',
      'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590,'closed_schema_count':0,'whole_pytorch_gate':'HOLD'}
    def save():dump(out/'report.json',report)
    def execute(name,argv,receipt):
        require(not receipt.exists(),('fresh child output required',name))
        custody.bind(argv[3] if argv[1:3]==['-I','-B'] else argv[4])
        row={'name':name,'argv':[str(x) for x in argv],'status':'RUNNING'};report['commands'].append(row)
        dump(out/(name+'.argv.json'),row['argv']);save()
        with (out/(name+'.stdout')).open('wb') as so,(out/(name+'.stderr')).open('wb') as se:
            proc=subprocess.run(row['argv'],cwd=repo,stdout=so,stderr=se,timeout=1200)
        row['exit_code']=proc.returncode
        for suffix in ('stdout','stderr'):
            f=out/(name+'.'+suffix);row[suffix]={'bytes':f.stat().st_size,'sha256':digest(f)}
        if receipt.is_file():
            child=decode(receipt);row.update(report=str(receipt.relative_to(out)),report_sha256=digest(receipt),status=child.get('status'))
        else:child={};row['status']='MISSING_REPORT'
        save();require(proc.returncode==0 and child.get('status')=='PASS',('child failed',name,proc.returncode,child.get('status')))
        custody.bind(receipt,row['report_sha256']);runtime_identity(child);return child
    try:
        custody.bind(Path(__file__));report['manifests']=[manifest(repo,*row,custody) for row in MANIFESTS]
        profile=custody.read(repo/'tools/native_dropout/profile.json');parent_profile=custody.read(repo/'tools/retained_gradients/profile.json')
        require(profile['source_sha256']==SOURCE and profile['native_sha256']==IMAGE,'frozen profile')
        custody.bind(a.compiler,COMPILER);custody.bind(a.linker,LINKER);custody.bind(a.flat_compiler,FLAT)
        require(a.compiler.resolve()==(stack/'native/stack-module-compiler').resolve() and a.linker.resolve()==(stack/'native/stack-linker-session').resolve(),'same explicit stack tools')
        report['parent_build']=build_identity(parent,parent_profile,custody,True)
        producer=out/'producer'
        execute('producer13',[sys.executable,'-I','-S','-B',repo/'tools/native_dropout/qualify.py',
          '--parent',parent,'--repository',repo,'--upstream',a.upstream.resolve(),'--compiler',a.compiler.resolve(),
          '--linker',a.linker.resolve(),'--flat-compiler',a.flat_compiler.resolve(),'--output',producer],producer/'report.json')
        report['producer_executed_this_invocation']=True
        report['producer_admission']=producer_identity(producer,custody)
        native=producer/'native';report['new_build']=build_identity(native,profile,custody)
        identity=out/'actual-identity9'
        child=execute('actual-identity9',[sys.executable,'-I','-S','-B',repo/'tests/native_dropout_identity9/replay_actual_identity9.py',
          '--binary',native/'tensor-dtype-dropout-session','--source',native/'runtime.hepta','--build-report',native/'build.json',
          '--protocol',repo/'tests/retained_nonleaf_review/protocol.py','--reference',repo/'validation/native-dropout/reference/torch-empty-dropout-identity.json','--output',identity],identity/'report.json')
        require(child['recipes']==child['passing_recipes']==9 and child['native_sessions']==child['passed_sessions']==13,'all actual9 recipes/sessions')
        require(child['frames']==963 and child['assertions']==6888 and child['inputs_unchanged'] is True and child['reference_unchanged'] is True,'actual9 fixed scope and unchanged reference')
        require(child['reference_sha256']=='de7db2f985873be797bd7e167fb374f82b5690021e0a0541d2fb4f7a36b7ad83' and child['runner_sha256']=='fd09055646e407f32bc62f08b131d01fa45bcf020b3f22e4c89eb0dfd83da2bb','actual9 exact reference/runner')
        require(all(x['status']=='PASS' and x['native_exit_code']==0 for x in child['rows']),'actual9 all EOF')
        specs=custody.read(repo/'validation/dtype-resource-review/cli.json')
        require(specs['source_sha256']==SOURCE and specs['native_sha256']==IMAGE,'resource CLI exact profile')
        require([x['name'] for x in specs['commands']]==['actual-detach26','typed-handoff2','owned-entry5'],'three exact resource gates')
        values={'python':sys.executable,'repo':str(repo),'native_build':str(native),'parent_build':str(parent),'stack_build':str(stack),'output':str(out)}
        for spec in specs['commands']:
            argv=[x.format_map(values) for x in spec['argv']];name=spec['name']
            child=execute(name,argv,out/name/'report.json')
            for key,value in spec['required'].items():require(type(child[key]) is type(value) and child[key]==value,('resource required',name,key))
        owner=out/'owner640'
        child=execute('owner640',[sys.executable,'-I','-S','-B',repo/'tests/dtype_owner_accounting/qualify.py','--repo',repo,'--build',native,'--output',owner],owner/'report.json')
        require(child['native_case_count']==4 and child['native_frames']==68 and child['copied_negative_controls']==24,'independent640 fixed controls')
        require(child['inputs_unchanged'] is True and child['native_executed_this_invocation'] is True and child['frozen_355_428_446_observer_unchanged'] is True,'independent640 custody')
        report['separate_required_CI_gates']={'live32':'NOT_EXECUTED_BY_NATIVE_COORDINATOR','frozen35_with_640_accounting':'NOT_EXECUTED_BY_NATIVE_COORDINATOR','no_inference_from_inherited16':True}
        report['counts']={'producer_top_level_steps':13,'inherited_named_suites_with_original_OPENs':16,'actual_identity_recipes':9,'actual_identity_sessions':13,'actual_detach_cases':26,'native_typed_directions':2,'native_entry_controls':5,'independent_owner_native_controls':4,'copied_snapshot_negative_controls':24}
        custody.finish();report['complete_current_execution_evidence']=archive_evidence(out);custody.finish()
        report.update(status='PASS',all_inputs_unchanged=True,checked_inputs={str(k):v for k,v in custody.files.items()})
        save();print(json.dumps({'status':'PASS','report_sha256':digest(out/'report.json'),'commands_executed':len(report['commands']),'source_sha256':SOURCE,'native_sha256':IMAGE}));return 0
    except BaseException as e:
        report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc(),successful_suite_completion_claimed=False)
        save()
        try:report['partial_failed_execution_evidence']=archive_evidence(out)
        except BaseException as ae:report['archive_error']=repr(ae)
        report['checked_inputs']={str(k):v for k,v in custody.files.items()};save()
        print(json.dumps({'status':'FAIL','error':repr(e),'report':str(out/'report.json')}));return 1


if __name__=='__main__':raise SystemExit(main())
