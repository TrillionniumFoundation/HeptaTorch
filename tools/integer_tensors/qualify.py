"""Cold native int64 construction, seven fixed-reference gates and legacy gates.

The independent fixed inplace40 runner also executes on the cold image.
Ownership/resource and live Torch35/32 reports are separate mandatory
integration evidence, never inferred by this producer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from archive_evidence import archive

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','repository','upstream','compiler','linker','flat-compiler','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    profile_raw=(HERE/'profile.json').read_bytes();profile=json.loads(profile_raw)
    manifest_path=HERE/'package-manifest.json';manifest_raw=manifest_path.read_bytes();manifest=json.loads(manifest_raw)
    tests=PACKAGE/'tests/integer_tensor_review';refs=PACKAGE/'validation/integer-tensors/reference'
    independent=profile['independent_inplace40'];repo=a.repository.resolve()
    external_manifest=repo/independent['manifest_path'];external_raw=external_manifest.read_bytes()
    assert sha(external_raw)==independent['manifest_sha256']
    external_inputs={external_manifest:external_raw}
    for pin in json.loads(external_raw)['files']:
        f=repo/pin['path'];raw=f.read_bytes()
        assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256'] and f.stat().st_mode&511==pin['mode']
        external_inputs[f]=raw

    def custody():
        assert manifest_path.read_bytes()==manifest_raw and (HERE/'profile.json').read_bytes()==profile_raw
        for name,pin in manifest['files'].items():
            f=PACKAGE/name;raw=f.read_bytes()
            assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256'] and f.stat().st_mode&511==pin['mode'],name
        assert all(f.read_bytes()==raw for f,raw in external_inputs.items())

    custody()
    assert sha(a.flat_compiler.read_bytes())==profile['flat_compiler_sha256']
    for name,pin in profile['fixed_references'].items():
        raw=(refs/name).read_bytes();ref=json.loads(raw)
        assert sha(raw)==pin['sha256'] and len(raw)==pin['bytes']
        assert ref['status']=='OBSERVED' and ref['cases']==pin['cases'] and ref['probe_error_count']==0
        assert ref['probe_sha256']==pin['probe_sha256'] and ref['source_files_unchanged'] is True
        assert len(ref['oracle_provenance']['checks'])==11 and all(v is True for v in ref['oracle_provenance']['checks'].values())
    report={'schema':'heptatorch.integer-tensor-portable-qualification.v1','status':'RUNNING',
            'source_sha256':profile['source_sha256'],'native_sha256':profile['native_sha256'],
            'profile_sha256':sha(profile_raw),'package_manifest_sha256':sha(manifest_raw),'steps':[],
            'fixed_references':profile['fixed_references'],'pinned_Torch_executed_here':False,
            'counting_boundary':'Seven separately named reference gates and twelve legacy gates overlap. They are not added into a unique upstream API total.',
            'independent_inplace40_package_sha256':sha(external_raw),
            'independent_integration_reports_required':['peer I64 resources and owned entry','typed previous-profile handoff','live32','strict accounted35 and negative controls'],
            'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590}

    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

    def run(name,script,args,receipt,expected_status,counts,compact=True,unchanged_field='all_inputs_unchanged'):
        cmd=[sys.executable,'-B',str(script),*map(str,args)]
        row={'name':name,'command':cmd,'status':'STARTED','process_started':False};report['steps'].append(row);save()
        with (out/(name+'.stdout')).open('wb') as so,(out/(name+'.stderr')).open('wb') as se:
            process=subprocess.Popen(cmd,stdout=so,stderr=se);row['process_started']=True;save()
            try:code=process.wait(timeout=1800)
            except BaseException:
                process.kill();process.wait();raise
        row.update(exit_code=code,stdout_sha256=sha((out/(name+'.stdout')).read_bytes()),stderr_sha256=sha((out/(name+'.stderr')).read_bytes()))
        if receipt.exists():
            raw=receipt.read_bytes();value=json.loads(raw)
            row.update(status=value['status'],report=str(receipt.relative_to(out)),report_sha256=sha(raw))
        else:value=None;row['status']='NO_REPORT'
        save();assert code==0,(name,code);assert value['status']==expected_status,(name,value['status'])
        for key,want in counts.items():assert value[key]==want,(name,key,value[key],want)
        if name!='cold-native-build':
            assert value['source_sha256']==profile['source_sha256'] and value['native_sha256']==profile['native_sha256']
            assert value[unchanged_field] is True
        if compact:row['archive']=archive(receipt.parent,out/'evidence',name)
        save();return value

    try:
        native=out/'native'
        run('cold-native-build',HERE/'rebuild.py',[
            '--parent',a.parent.resolve(),'--repository',a.repository.resolve(),'--upstream',a.upstream.resolve(),
            '--compiler',a.compiler.resolve(),'--linker',a.linker.resolve(),'--output',native],
            native/'profile-rebuild.json','PASS',{'units_reproduced_exactly':11},False)
        common=['--binary',native/'tensor-integer-session','--source',native/'runtime.hepta','--protocol',tests/'wire/protocol.py']
        all54=[]
        for name,script,reference,counts in (
            ('original-storage12','replay_storage12_v2.py','torch-int64-tensor-reference.json',{'cases':12,'passed':12,'failed':0,'frames':235,'reference_indices':list(range(12))}),
            ('original-arithmetic30','replay_arithmetic30.py','torch-int64-tensor-reference.json',{'cases':30,'passed':30,'failed':0,'frames':1122,'reference_indices':list(range(12,42))}),
            ('original-dropout12','replay_dropout12.py','torch-int64-tensor-reference.json',{'cases':12,'passed':12,'failed':0,'frames':333,'reference_indices':list(range(42,54))}),
            ('alias-mutation8','replay_alias8_v2.py','torch-int64-alias-reference.json',{'cases':8,'passed':8,'native_frames':516,'exact_checks':2792}),
            ('integer-RNE1237','replay_rne_v2.py','torch-int64-f32-rne.json',{'cases':5,'integers':1237,'native_frames':60,'exact_checks':1282}),
            ('CPU-promotion20','replay_promotion20.py','torch-int64-promotion-reference.json',{'cases':20,'passed':20,'original_mixed_operations':40,'native_frames':2280,'exact_checks':10580}),
            ('view-base-lifetime6','replay_view_base6.py','torch-int64-view-base-lifetime.json',{'cases':6,'passed':6,'native_frames':450,'exact_checks':2046,'native_role_checks':168})):
            value=run(name,tests/script,common+['--reference',refs/reference,'--output',out/name],out/name/'report.json','PASS',counts)
            assert value['reference_sha256']==profile['fixed_references'][reference]['sha256']
            if reference=='torch-int64-tensor-reference.json':all54.extend(value['reference_indices'])
        assert all54==list(range(54))
        value=run('independent-inplace40',repo/independent['runner_path'],common+[
            '--observer',tests/'native_observer.py','--reference',repo/independent['reference_path'],
            '--expected-source-sha256',profile['source_sha256'],'--expected-native-sha256',profile['native_sha256'],
            '--output',out/'independent-inplace40'],out/'independent-inplace40/report.json','PASS',
            {'cases':40,'passed':40,'native_frames':1747,'exact_checks':7178,'raw_words':2958,
             'rng_bytes':808960,'rollback_bytes':13980480},unchanged_field='source_files_unchanged')
        assert value['reference_sha256']==independent['reference_sha256']
        run('legacy523-behavior12',tests/'run_legacy523_07.py',[
            '--native-build',native,'--repository',a.repository.resolve(),'--adapters',tests/'inherited-adapters',
            '--flat-compiler',a.flat_compiler.resolve(),'--output',out/'legacy523-behavior12'],
            out/'legacy523-behavior12/report.json','PASS',{'step_count':12,'existing_native_units_verified':11,'all_expected_statuses_and_counts_matched':True})
        custody()
        report.update(status='PASS',step_count=len(report['steps']),all_package_members_unchanged=True,
                      original54_reference_indices_exactly_once=all54,
                      native_cold_rebuild_executed=True,all_named_gate_statuses_and_counts_matched=True)
        save();print(json.dumps({k:v for k,v in report.items() if k not in ('steps','fixed_references')}));return 0
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));save();raise


if __name__=='__main__':raise SystemExit(main())
