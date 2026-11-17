#!/usr/bin/env python3
"""Verify recovered four-suite original bytes and describe exact replay CLIs."""
import ast
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import tarfile
import traceback

ROOT=Path('/dev/shm/heptatorch-d26699e25fea/hia-frozen-source-recovery-01')
DEST=Path('/workspace/scratch/d26699e25fea/hia-recovery-review-20261116')
sha=lambda x:hashlib.sha256(x).hexdigest()

def metadata(p):
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':sha(b)}

def verify_archive(arc, manifest, reference=False):
    compressed=arc.read_bytes()
    digest=manifest['archive']['sha256'] if reference else manifest['archive_sha256']
    size=manifest['archive']['bytes'] if reference else manifest['archive_bytes']
    assert len(compressed)==size and sha(compressed)==digest
    decoder=lzma.LZMADecompressor();raw=decoder.decompress(compressed)
    assert decoder.eof and not decoder.unused_data
    if reference:
        assert len(raw)==manifest['tar']['bytes'] and sha(raw)==manifest['tar']['sha256']
    expected=manifest['files'] if reference else manifest['members']
    found={};total=0;contents={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tf:
        for m in tf:
            p=PurePosixPath(m.name)
            assert m.isfile() and not p.is_absolute() and '..' not in p.parts
            assert str(p)==m.name and m.name not in found and m.name in expected
            b=tf.extractfile(m).read();item=expected[m.name]
            actual={'bytes':len(b),'sha256':sha(b),'mode':m.mode}
            assert m.size==len(b) and all(actual[k]==item[k] for k in actual)
            found[m.name]=actual;total+=len(b)
            if m.name.endswith(('transport.json','report.json','.py')):contents[m.name]=b
    assert set(found)==set(expected)
    assert len(found)==manifest['file_count' if reference else 'member_count']
    assert total==manifest['expanded_bytes' if reference else 'original_bytes']
    return {'archive':metadata(arc),'members':len(found),'expanded_bytes':total,
            'tar_bytes':len(raw),'tar_sha256':sha(raw),'single_xz_eof':True,
            'complete_bytes_sizes_modes_verified':True},found,contents

def main():
    report={'schema':'heptatorch.recovered-to09b-four-regressions.v1','status':'RUNNING',
            'native_executions':0,'torch_executions':0,'compiler_executions':0,
            'old_outer_cold_result_remains_FAIL':True,'suites':[]}
    try:
        producer=ROOT/'producer';cold=ROOT/'cold01/original/producer'
        pm=json.loads((producer/'tools/tensor_to_copy/package-manifest.json').read_bytes())
        assert sha((producer/'tools/tensor_to_copy/package-manifest.json').read_bytes())=='5179e869868633ff38c0a357039b8f223f5d9672a6453af6187d4d2bd025e3a1'
        needed={}
        for stem,n in [('copy4',4),('reduction4',4),('lanes2',2),('branches6',6)]:
            reader_rel='tests/tensor_to_copy_review/replay_as_strided_'+stem+'.py'
            reference_rel='validation/tensor-to-copy/reference/as-strided-'+stem+'.json'
            reader=producer/reader_rel;reference=producer/reference_rel
            tree=ast.parse(reader.read_bytes());constants={}
            for node in tree.body:
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                    try:constants[node.targets[0].id]=ast.literal_eval(node.value)
                    except (ValueError,TypeError):pass
            actual=json.loads(reference.read_bytes());assert sha(reference.read_bytes())==constants['REFERENCE']
            assert actual['status']=='OBSERVED' and actual['cases']==n and len(actual['rows'])==n
            assert actual['probe_error_count']==0 and actual['source_files_unchanged'] is True
            assert actual['probe_sha256']==constants['PROBE']
            checks=actual['oracle_provenance']['checks'];assert set(checks)==constants['CHECKS'] and len(checks)==11 and all(v is True for v in checks.values())
            refpack=producer/'validation/tensor-to-copy/reference-packets'/('as-strided-'+stem)
            rm=json.loads((refpack/'manifest.json').read_bytes())
            refproof,rfound,_=verify_archive(refpack/'reference.tar.xz',rm,True)
            assert constants['PROBE'] in {x['sha256'] for x in rfound.values()}
            assert constants['REFERENCE'] in {x['sha256'] for x in rfound.values()}
            native=cold/('fixed-as-strided-'+stem)
            original=json.loads((native/'report.json').read_bytes())
            assert original['status']=='PASS' and original['cases']==original['passed']==n and original['failed']==0
            assert original['reader_sha256']==sha(reader.read_bytes())
            assert original['reference_sha256']==constants['REFERENCE'] and original['probe_sha256']==constants['PROBE']
            assert original['source_sha256']=='4f8f73f463f681b5eb29423d71a35a0b42d807fd92bf976bbf76c1970fbcee33'
            assert original['native_sha256']=='53fb43b869858f020c0b936a65f00f3b14b97d9e37038a74c357e1da471fe083'
            assert original['observer_sha256']=='42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc'
            assert original['all_inputs_unchanged'] is True
            assert set(original['provenance_checks'])==constants['CHECKS'] and all(v is True for v in original['provenance_checks'].values())
            nm=json.loads((cold/'evidence'/('fixed-as-strided-'+stem+'.manifest.json')).read_bytes())
            nativeproof,nfound,_=verify_archive(cold/'evidence'/('fixed-as-strided-'+stem+'.tar.xz'),nm)
            transports=[]
            for i in range(n):
                tp=native/('case-%02d'%i)/'raw/transport.json';tr=json.loads(tp.read_bytes())
                assert tr['exit_code']==0,(stem,i,tr)
                transports.append(metadata(tp))
            fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
            main=ast.get_source_segment(reader.read_text(),fn)
            assert "'binary','source','build-report','protocol','observer','reference','output'" in main
            assert "len(build['units']) == 12" in main and "build['descriptor_bytes'] == 672" in main
            argv=['python3','-I','-S','-B',reader_rel,
              '--binary','build/tensor-to-recovered/tensor-to-copy-session',
              '--source','build/tensor-to-recovered/runtime.hepta',
              '--build-report','build/tensor-to-recovered/build.json',
              '--protocol','tests/tensor_to_copy_review/protocol.py',
              '--observer','tests/tensor_to_copy_review/native_observer09.py',
              '--reference',reference_rel,
              '--expected-source-sha256',original['source_sha256'],
              '--expected-native-sha256',original['native_sha256'],
              '--expected-observer-sha256',original['observer_sha256'],
              '--output','build/tensor-to-recovered-bits/'+stem]
            local=[reader_rel,reference_rel,'tests/tensor_to_copy_review/protocol.py','tests/tensor_to_copy_review/owner_client.py','tests/tensor_to_copy_review/native_observer09.py']
            for rel in local:
                p=producer/rel;item=pm['files'][rel];assert sha(p.read_bytes())==item['sha256']
                needed[rel]=metadata(p)
            report['suites'].append({'name':stem,'case_count':n,'reader':metadata(reader),'reference':metadata(reference),
                'probe_sha256':constants['PROBE'],'original_native_report':metadata(native/'report.json'),
                'original_native_counts':{k:original[k] for k in ('cases','frames','checks','exact_words','rng_bytes','rollback_bytes')},
                'original_all_case_eof_zero':transports,'reference_packet':refproof,'original_native_raw_archive':nativeproof,
                'proposed_CI_argv_not_executed':argv})
        report['minimum_execution_files']=needed
        report['proposed_artifact_path']='build/tensor-to-recovered-bits/'
        report['coordinator_recovery']={'a6f91_whole_source_found_in_these_two_archives':False,
            'full_pin_from_original_receipt':'a6f91b1d6134255652a7ac2fea5538dc084e96f521866063abafe88a9ac35c62',
            'draft_b230_source_present_but_not_equivalent_to_a6f91':True,
            'original_source_review_receipt':str(ROOT/'cold01/root/coordinator-final09b-root-source-review.json')}
        report['status']='PASS_RECOVERED_ORIGINAL16_BYTES_ONLY'
    except BaseException:
        report['status']='FAIL_RECOVERY_REVIEW';report['error']=traceback.format_exc()
    path=DEST/'four-regressions-review.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'path':str(path),'sha256':sha(path.read_bytes()),
                      'suites':len(report['suites']),'error':report.get('error')}))
    return 0 if report['status'].startswith('PASS_') else 1

if __name__=='__main__':raise SystemExit(main())
