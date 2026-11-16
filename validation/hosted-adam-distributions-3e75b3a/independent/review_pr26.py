#!/usr/bin/env python3
"""Read-only review of the exact PR26 original report artifact.

The artifact is never rewritten. JSON decoding is used only for checks and a
new, separately identified review receipt. No native or PyTorch test is rerun.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import re
import subprocess
import zipfile
from review_pr26_details import review as review_combined

HEAD = '3e75b3aa67481d8377059584e91b5885716750db'
ZIP_SHA = '55ac621bd927887980e3f151b9ca1dafa5572c0da143031062deecfe97696fa2'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested'}
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--repository', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.artifact.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reads = {}
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def raw(name):
        data = (root / name).read_bytes()
        reads[name] = {'bytes': len(data), 'sha256': sha(data)}
        return data

    def git(relative):
        return subprocess.check_output(['git', '-C', str(args.repository),
                                        'show', HEAD + ':' + relative])

    def provenance(value, label):
        check(value['status'] == 'PASS' and value['metadata_error'] is None,
              label + ': provenance status')
        check(set(value['checks']) == CHECKS and
              all(v is True for v in value['checks'].values()), label + ': all 11 provenance checks')
        check(value['lock_sha256'] == sha(git('compatibility/pytorch-oracle-lock.json')),
              label + ': exact committed oracle lock')
        check(value['observed']['version'] == '2.15.0+cpu' and
              value['observed']['git_commit'] == 'd0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
              label + ': actual wheel build identity')
        eq = value['release_source_equivalence']
        check(eq['inventory_commit'] == '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457' and
              eq['torch_git_matches_upstream_pin'] is False and
              eq['changed_paths'] == ['pyproject.toml'], label + ': limited release/source equivalence retained')

    def unit_custody(report, profile, build_name):
        custody = report['source_custody_before']
        check(custody == report['source_custody_after'], build_name + ': before/after custody unchanged')
        for unit in profile['units']:
            for suffix, key in [('.hepta', 'source_sha256'), ('.o', 'object_sha256')]:
                ending = '/build/' + build_name + '/' + unit['name'] + suffix
                matched = [v for k, v in custody.items() if k.endswith(ending)]
                check(matched == [unit[key]], build_name + ': exact ' + unit['name'] + suffix)

    def brief(value):
        fields = ['status', 'cases', 'case_count', 'expected_cases', 'reference_cases',
                  'executed_cases', 'qualified_exposed_scope', 'partial_cases',
                  'open_not_executed_cases', 'unexpected_failures', 'failures', 'failed',
                  'failed_cases', 'passed', 'frames', 'live_frames', 'frames_before_EOF',
                  'compared_values', 'exact_comparisons', 'exact_binary32_words',
                  'serialized_state_bytes', 'exact_float32_words']
        return {k: (len(value[k]) if isinstance(value[k], (list, dict)) else value[k])
                for k in fields if k in value}

    manifest_bytes = raw('ci-report-manifest.json')
    manifest = json.loads(manifest_bytes)
    check(sha(raw('reports.zip')) == ZIP_SHA, 'exact original ZIP SHA-256')
    check(manifest['status'] == 'COMPLETE' and manifest['job_status_before_export'] == 'success',
          'CI manifest complete with successful pre-export job')
    check(manifest['checkout_commit'] == 'f429231bbc94cce040144fc2fa87b5256d99abf2' and
          manifest['checkout_tree'] == 'e9fc404b5c5b314fc850505bf4a736a9848ffc3a',
          'exact PR26 merge checkout and tree')
    check(manifest['github']['GITHUB_RUN_ID'] == '43836034765' and
          manifest['github']['GITHUB_REF'] == 'refs/pull/26/merge' and
          manifest['github']['HEPTATORCH_SOURCE_HEAD_SHA'] == HEAD,
          'PR26 run and source head, not an earlier-run attribution')
    check(manifest['tracked_worktree_dirty'] is False and
          manifest['original_report_bytes_reserialized'] is False, 'clean checkout and original-byte export declaration')
    workflow = git('.github/workflows/native.yml')
    check(sha(workflow) == manifest['workflow_sha256'], 'workflow equals source-head committed bytes')
    check(sha(git('tools/collect_ci_reports.py')) == manifest['collector_sha256'],
          'collector equals source-head committed bytes')
    expected_envelopes = sorted(set(re.findall(rb'tools/emit_ci_report\.py build/([A-Za-z0-9_.-]+\.json)', workflow)))
    check([v.decode() for v in expected_envelopes] == manifest['required_envelope_reports'] and
          len(expected_envelopes) == 24 and manifest['missing_envelope_reports'] == [],
          'all 24 exact workflow envelope paths present')
    reports = {}
    with zipfile.ZipFile(root / 'reports.zip') as archive:
        names = archive.namelist()
        expected_names = [row['path'] for row in manifest['reports']] + ['ci-report-manifest.json']
        check(len(names) == len(set(names)) == 51 and set(names) == set(expected_names),
              '51 unique ZIP members: manifest plus 50 original reports')
        check(archive.read('ci-report-manifest.json') == manifest_bytes, 'ZIP manifest exact extracted bytes')
        for row in manifest['reports']:
            data = raw(row['path'])
            check(len(data) == row['bytes'] and sha(data) == row['sha256'], row['path'] + ': manifest size/hash')
            check(archive.read(row['path']) == data, row['path'] + ': ZIP member byte identical')
            reports[row['path']] = json.loads(data)

    adam = reports['hosted-owned-adam-native.json']
    dist = reports['hosted-cpu-distributions-native.json']
    adam_profile = json.loads(git('tools/owned_adam/profile.json'))
    dist_profile = json.loads(git('tools/cpu_distributions_partial/profile.json'))
    check(adam['native_source_sha256'] == adam_profile['source_sha256'] and
          adam['native_executable_sha256'] == adam_profile['native_sha256'], 'Adam report matches committed source/image profile')
    check(dist['source_sha256'] == dist_profile['runtime_source_sha256'] and
          dist['executable_sha256'] == dist_profile['executable_sha256'], 'distribution report matches committed source/image profile')
    unit_custody(adam, adam_profile, 'owned-adam')
    unit_custody(dist, dist_profile, 'cpu-distributions-partial')
    check(len(adam['reports']) == len(adam['original_report_sha256']) == 19, 'Adam 19 distinct subreports')
    check(len(dist['reports']) == len(dist['original_report_sha256']) == 11, 'distribution 11 distinct subreports')
    check(adam['cold_reproduction']['status'] == 'PASS' and
          adam['cold_reproduction']['native_modules'] == 8 and
          adam['cold_reproduction']['all_source_object_image_bytes_identical'] is True,
          'Adam cold eight-unit reproduction recorded')
    for kind in ['atomic', 'partial']:
        cold = dist['cold_reproduction'][kind]
        check(cold['status'] == 'PASS' and cold['native_units'] == 7 and
              cold['all_source_object_image_bytes_identical'] is True, 'distribution ' + kind + ' cold seven-unit reproduction recorded')
    check(adam['status'] == 'QUALIFIED_EXPOSED_ADAM_WITH_OPEN_SCOPE' and
          (adam['reference_cases'], adam['native_executed_cases'], adam['qualified_exposed_scope'],
           adam['partial_cases'], adam['open_not_executed_cases']) == (24,23,22,1,1), 'Adam exact 24-case exposed ledger')
    check(Counter(row['status'] for row in adam['case_ledger']) == {
        'QUALIFIED_EXPOSED_SCOPE':22, 'PARTIAL_WITH_RETAINED_NONLEAF_OPEN':1, 'OPEN_NOT_EXECUTED':1},
        'Adam ledger rows preserve 22 qualified, 1 partial, 1 open')
    check(adam['original_first20_replay_and_assertions_unchanged'] is True and
          adam['original_old24_report_reclassified'] is False, 'old optimizer report not relabeled')
    old = adam['reports']['optimizer-original']
    new = adam['reports']['adam-21-24']
    check((old['executed_cases'], old['qualified_exposed_scope'],old['partial_cases'],old['open_not_executed_cases'],
           old['unexpected_failures'],old['frames'],old['exact_comparisons']) == (19,18,1,5,0,2332,9535), 'Adam inherited old19 original counts')
    check((new['qualified_exposed_scope'],new['failures'],new['frames'],new['exact_comparisons']) == (4,0,908,3469),
          'Adam fixed actual cases21-24 exact counts')
    for label, reference in [('Adam old19',old),('Adam new4',new),('distribution',dist)]:
        provenance(reference['oracle_provenance'],label)
    for label, expected in [('atomic/actual-reference',{'PASS':161,'OPEN_TRANSACTION_DIFFERENCE':4,'OPEN_UNREPRESENTED':18}),
                            ('partial/atomic-reference',{'PASS':161,'OPEN_TRANSACTION_DIFFERENCE':4,'OPEN_UNREPRESENTED':18}),
                            ('partial/explicit-reference',{'PASS':165,'OPEN_UNREPRESENTED':18})]:
        sub = dist['reports'][label]
        check(sub['reference_cases'] == len(sub['rows']) == 183 and sub['counts'] == expected and
              dict(Counter(row['status'] for row in sub['rows'])) == expected, label + ': original 183 ledger unchanged')
    check(dist['status'] == 'PARTIAL' and dist['native_scoped_qualification_passed'] is True,
          'distribution scoped success does not erase PARTIAL status')
    check(adam['full_pytorch_rewrite_complete'] is False and dist['full_pytorch_rewrite_complete'] is False and
          adam['pytorch_api_denominator'] == dist['pytorch_api_denominator'] == 2590,
          'full rewrite remains open with unchanged 2590 denominator')

    combined_details = review_combined(args, reports, check, git, provenance, workflow, manifest)
    common_profile = combined_details['profile']

    source_paths = {'frozen_28_oracle':'tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py',
                    'frozen_21_session':'tests/pytorch_session_oracle.py',
                    'frozen_reference_support':'tests/pytorch_oracle.py',
                    'oracle_lock':'compatibility/pytorch-oracle-lock.json',
                    'v3_entry':'tests/tensor_object_oracle/pytorch_tensor_objects_oracle.py',
                    'v3_bridge':'tests/tensor_object_oracle/bridge.py',
                    'v3_lifecycle':'tests/tensor_object_oracle/lifecycle.py'}
    targets = [('owned-adam', adam_profile['source_sha256'], adam_profile['native_sha256']),
               ('cpu-distributions',dist_profile['runtime_source_sha256'],dist_profile['executable_sha256']),
               ('adam-distributions',common_profile['runtime_source_sha256'],common_profile['executable_sha256'])]
    focus = []
    older_response_digest_differences = {}
    for prefix, source_hash, executable_hash in targets:
        backward_name = 'pytorch-' + prefix + '-backward.json'
        frozen_name = 'pytorch-' + prefix + '-frozen35.json'
        joint = reports[backward_name]
        frozen = reports[frozen_name]
        check(joint['native_source_sha256'] == source_hash and joint['native_executable_sha256'] == executable_hash and
              frozen['native_executable_sha256'] == executable_hash, prefix + ': native/live32/frozen35 same exact image')
        for name, value, count in [(backward_name,joint,32),(frozen_name,frozen,35)]:
            check(value['status'] == 'PASS' and value['cases'] == value['expected_cases'] == len(value['rows']) == count and
                  value['failed_cases'] == 0 and all(row['status'] == 'PASS' for row in value['rows']), name + ': all cases actual PASS')
            provenance(value['oracle_provenance'],name)
            check(all(v is True for v in value['source_files_unchanged_during_run'].values()), name + ': execution custody unchanged')
        check(joint['pytorch_executed'] is True and joint['one_native_process_per_trajectory'] is True,
              prefix + ': live PyTorch and interactive native execution declared')
        check((joint['frames'],joint['exact_binary32_words'],joint['serialized_state_bytes'],
               joint['absolute_version_comparisons'],joint['gradient_presence_comparisons']) == (2132,17455,1137600,238,166),
              prefix + ': original32 comparison counts')
        check(joint['source_custody']['oracle'] == sha(git('tests/pytorch_tensor_rng_backward_oracle.py')) and
              joint['source_custody']['reference_support'] == sha(git('tests/pytorch_oracle.py')),
              prefix + ': unchanged live32 committed source bytes')
        check(joint['case_plan'] == reports['pytorch-tensor-rng-backward.json']['case_plan'],prefix + ': original32 plan unchanged')
        check(joint['rows'] == reports['pytorch-owned-optimizer-backward.json']['rows'] and
              joint['rows'] == reports['pytorch-nullable-rng-backward.json']['rows'],
              prefix + ': exact32 per-case rows equal both immediate inherited profile witnesses')
        legacy_rows = reports['pytorch-tensor-rng-backward.json']['rows']
        differences = []
        for old_row, current_row in zip(legacy_rows, joint['rows']):
            check({k:v for k,v in old_row.items() if k!='response_stream_sha256'} ==
                  {k:v for k,v in current_row.items() if k!='response_stream_sha256'},
                  prefix + ': original32 request/reference/metric fields ' + old_row['case'])
            if old_row['response_stream_sha256'] != current_row['response_stream_sha256']:
                differences.append({'case':old_row['case'],
                    'older_common05_response_stream_sha256':old_row['response_stream_sha256'],
                    'current_and_nullable_and_SGD_response_stream_sha256':current_row['response_stream_sha256']})
        check([d['case'] for d in differences] == ['version-error-prefix','spent-saved-prefix',
              'ready-leaf-before-error','two-generators-prefix'],prefix + ': exact four previously inherited response-digest differences retained')
        older_response_digest_differences[prefix] = differences
        check((frozen['frames'],frozen['compared_values'],frozen['direct_parameter_absolute_version_mismatches'],
               frozen['explicit_state_absolute_version_differences']) == (6344,2590274,0,283),
              prefix + ': frozen35 numeric counts and 283 legacy differences retained')
        check(frozen['case_plan'] == reports['pytorch-tensor-objects-abi3.json']['case_plan'] and
              frozen['overlay_audit'] == reports['pytorch-tensor-objects-abi3.json']['overlay_audit'],
              prefix + ': frozen35 plan and inverse-AST audit unchanged')
        for label, path in source_paths.items():
            check(frozen['source_custody'][label]['sha256'] == sha(git(path)),prefix + ': exact committed ' + label)
        check(frozen['native_unchanged_during_run'] is True and
              frozen['known_open_contracts'] == reports['pytorch-tensor-objects-abi3.json']['known_open_contracts'] and
              frozen['observed_open_policy_differences'] == reports['pytorch-tensor-objects-abi3.json']['observed_open_policy_differences'],
              prefix + ': frozen legacy retention and other open contracts preserved')
        focus += [{'report':backward_name,**brief(joint),'native_source_sha256':source_hash,
                   'native_executable_sha256':executable_hash,'provenance_checks':11},
                  {'report':frozen_name,**brief(frozen),'native_source_sha256':source_hash,
                   'native_executable_sha256':executable_hash,'provenance_checks':11,
                   'explicit_state_absolute_version_differences':283}]

    old_counts = {'pytorch-native-linked.json':(270,13076), 'pytorch-monolithic.json':(270,13076),
                  'pytorch-session-modular.json':(21,491847),'pytorch-session-monolithic.json':(21,491847),
                  'pytorch-owned-tensor.json':(28,2590207),'pytorch-stack-tensor.json':(28,2590207)}
    for name,(count,values) in old_counts.items():
        value=reports[name]
        check(value['status']=='PASS' and value['failed_cases']==0 and
              value['cases']==value['expected_cases']==len(value['rows'])==count and
              value['compared_values']==values,name+': original legacy case/value counts')
        provenance(value['oracle_provenance'],name)
    for name in ['session-independent-modular.json','session-independent-monolithic.json']:
        check(reports[name]['status']=='PASS' and reports[name]['cases']==477,name+': original477 native cases')
    for name,value in reads.items():
        check(sha((root/name).read_bytes())==value['sha256'] and (root/name).stat().st_size==value['bytes'],
              name+': original bytes unchanged after review')
    result={'schema':'heptatorch.independent-pr26-original-artifact-review.v1','status':'PASS',
            'reviewer_script_sha256':sha(Path(__file__).read_bytes()),'artifact_directory':str(root),
            'zip_sha256':ZIP_SHA,'run_id':43836034765,'job_id_supplied_by_parent':133192768480,
            'source_head':HEAD,'checkout_commit':manifest['checkout_commit'],'checkout_tree':manifest['checkout_tree'],
            'independent_successful_step_count_verified':True,
            'step_count_scope':'Independent GitHub run, artifact and job responses bind this exact source/run; all 68 actual job steps are successful.',
            'manifest_sha256':sha(manifest_bytes),'original_reports':50,'required_envelopes':24,
            'checks_passed':len(checks),'checks':checks,'original_inputs':reads,
            'focus_live_reports':focus,'adam_subreports':{k:brief(v) for k,v in adam['reports'].items()},
            'older_common05_response_digest_differences':older_response_digest_differences,
            'reviewer_predecessor':{'source':'review_pr25.py','sha256':'0c582e4d948e1a3f7afa73501bbb16dde624b2abba974b0dea128ca0f8c6a68e','report_sha256':'8bb3ecdbeff91a51731b89b0214aebbf32df797fbcd71f8e1afa97202d58a6f6','original_history_preserved':True},
            'combined_native_review':combined_details,
            'detail_reviewer_sha256':sha(Path(__file__).with_name('review_pr26_details.py').read_bytes()),
            'distribution_subreports':{k:brief(v) for k,v in dist['reports'].items()},
            'adam_24_case_ledger':adam['case_ledger'],
            'distribution_original_183_counts':{k:dist['reports'][k]['counts'] for k in
                ['atomic/actual-reference','partial/atomic-reference','partial/explicit-reference']},
            'all_50_report_statuses_and_counts':{k:brief(v) for k,v in reports.items()},
            'original_artifact_modified':False,'original_JSON_reserialized_or_replaced':False,
            'native_or_Torch_reexecuted_by_this_review':False,'attribution':'PR26 run43836034765 only, not predecessor runs',
            'whole_PyTorch_rewrite_complete':False,'pytorch_api_denominator':2590}
    (out/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    rows=['# PR26 original artifact review','',
          'PASS. Independent read-only review of exact source head `'+HEAD+'`, GitHub run **43836034765**, job **133192768480**. The official artifact digest matches all original ZIP bytes. Independently fetched job metadata reports all **68 steps successful**.','',
          '| Runtime | Native/report scope | Actual live32 | Frozen35 |',
          '| --- | --- | --- | --- |',
          '| Frozen Adam02 | Original 19 reports; 22 qualified / 1 partial / 1 open of24 | PASS: 2,132 frames / 17,455 exact words | PASS: 6,344 frames / 2,590,274 values |',
          '| Frozen distributions partial01 | Original 11 reports; explicit165 /18open; atomic161 /4transaction differences /18open | Same exact32 thresholds | Same exact35 thresholds |',
          '| Unified Adam + distributions378 | 13 producer suites plus2 independent gates; old atomic86/87 negative still actual FAIL/exit1 | Same exact32 thresholds, source5d3a/image3fc59 | Same exact35 thresholds, image3fc59 |','',
          'All50 original reports and24 workflow-designated envelope report paths are present. The51 unique ZIP members are the50 reports and their manifest. Every original report matches the ZIP and manifest length/SHA. Embedded native subreports recover their declared original hashes. No original JSON was overwritten.','',
          'Every encountered oracle provenance block has exactly11 successful checks. The fixed183 distribution index/name/family/status rows and original24 optimizer names are preserved; cases14 and20 remain partial/open. The32-case plans, all request/reference/count fields and prior nullable/SGD response digests match. The35-case plans, unchanged oracle sources and inverse-AST receipts match, retaining283 old explicit-state Version observations and zero direct parameter Version mismatches.','',
          'All predecessor src/tools/tests/lock Git bytes and modes are unchanged from mainf46. The old workflow step byte blocks remain, with exactly3 new steps. Both committed raw-history archives were independently decoded member by member, including preserved original negative failures. Full17-gate acceptance remains HOLD with denominator2590.','',
          'This review did not rerun native or PyTorch tests. It does not qualify the later576B public-gradient candidate or general cmd95 NaN propagation. Those have separate retained failures and ongoing qualification.','']
    (out/'README.md').write_text('\n'.join(rows))
    print(json.dumps({'status':'PASS','checks':len(checks),'report_sha256':sha((out/'report.json').read_bytes()),
                      'report':str(out/'report.json'),'table':str(out/'README.md')}))


if __name__=='__main__':
    main()
