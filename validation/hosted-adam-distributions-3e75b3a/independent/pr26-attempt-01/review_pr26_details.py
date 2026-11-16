"""Additional read-only checks for the source-bound PR26 original artifact."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

HEAD = '3e75b3aa67481d8377059584e91b5885716750db'
BASE = 'f46ae3434b8aa91e898a2a70469e4ca423aa8034'
SOURCE = '5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873'
IMAGE = '3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22'
REFERENCE = 'c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467'
OPTIMIZER_REFERENCE = '7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160'
OPEN = [4, 9, 14, 19, 24, 29, 34, 39, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174]
LATE = [101, 103, 105, 179]
NAMES = ['atomic_reference', 'explicit_reference', 'optimizer_reference', 'adam_reference',
         'legacy_atomic_controls', 'new_command_admission', 'atomic_controls_extended',
         'partial_controls', 'adam_controls', 'adam_guards', 'adam_step_counter',
         'owned_entry', 'typed_handoffs']
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def review(args, reports, check, git, provenance, workflow, manifest):
    here = Path(__file__).resolve().parent
    connector_inputs = {}
    for name in ('pr26-job-summary.connector.json', 'pr26-workflow-run.github.json',
                 'pr26-artifacts.connector.json'):
        raw = (here / name).read_bytes()
        connector_inputs[name] = {'sha256': sha(raw), 'bytes': len(raw)}
    job = json.loads((here / 'pr26-job-summary.connector.json').read_bytes())['jobs']
    check(len(job) == 1, 'independent GitHub response: exactly one native job in this run')
    job = job[0]
    check((job['id'], job['run_id'], job['status'], job['conclusion']) ==
          (133192768480, 43836034765, 'completed', 'success'), 'independent GitHub job identity/status')
    check(len(job['steps']) == 68 and len({s['number'] for s in job['steps']}) == 68 and
          all(s['status'] == 'completed' and s['conclusion'] == 'success' for s in job['steps']),
          'independent actual job listing: all 68 steps successful')
    run = json.loads((here / 'pr26-workflow-run.github.json').read_bytes())
    check((run['id'], run['head_sha'], run['event'], run['status'], run['conclusion']) ==
          (43836034765, HEAD, 'pull_request', 'completed', 'success'), 'independent GitHub run pins exact PR26 head')
    check(len(run['pull_requests']) == 1 and run['pull_requests'][0]['number'] == 26 and
          run['pull_requests'][0]['head']['sha'] == HEAD and
          run['pull_requests'][0]['base']['sha'] == BASE, 'independent GitHub run binds main f46 predecessor')
    artifacts = json.loads((here / 'pr26-artifacts.connector.json').read_bytes())['artifacts']
    check(len(artifacts) == 1, 'independent GitHub artifact inventory')
    artifact = artifacts[0]
    check(artifact['id'] == 14210734282 and artifact['size_in_bytes'] == 2869752 and
          artifact['digest'] == 'sha256:55ac621bd927887980e3f151b9ca1dafa5572c0da143031062deecfe97696fa2' and
          artifact['workflow_run']['head_sha'] == HEAD and artifact['workflow_run']['id'] == run['id'],
          'downloaded original ZIP matches official artifact digest/length/head')

    def old_git(path):
        return subprocess.check_output(['git', '-C', str(args.repository), 'show', BASE + ':' + path])

    def tree(commit):
        data = subprocess.check_output(['git', '-C', str(args.repository), 'ls-tree', '-r', '-z', commit])
        return {row.split(b'\t', 1)[1].decode(): row.split(b'\t', 1)[0].decode()
                for row in data.split(b'\0') if row}

    base_tree, current_tree = tree(BASE), tree(HEAD)
    inherited = {path: entry for path, entry in base_tree.items()
                 if path.startswith(('src/', 'tools/', 'tests/')) or path == 'compatibility/pytorch-oracle-lock.json'}
    check(all(current_tree.get(path) == entry for path, entry in inherited.items()),
          'all predecessor src/tools/tests/lock Git bytes and modes unchanged')
    old_steps = Counter(old_git('.github/workflows/native.yml').split(b'\n      - ')[1:])
    new_steps = Counter(workflow.split(b'\n      - ')[1:])
    check(all(new_steps[s] >= count for s, count in old_steps.items()),
          'all predecessor workflow step byte blocks retained')
    check(sum(new_steps.values()) - sum(old_steps.values()) == 3,
          'exactly three new CI steps: native plus original32 plus original35')
    admission = json.loads(git('validation/hosted-adam-distributions-predecessors-abc2046/main-admission.json'))
    check(admission['status'] == 'PASS_SCOPED_MAIN_ADMISSION' and
          admission['distributions']['merge_commit'] == BASE and
          admission['whole_project_decision'] == 'HOLD' and admission['open_full_subsystem_gates'] == 17,
          'committed predecessor admission remains scoped main f46')
    acceptance = json.loads(git('compatibility/full-acceptance.json'))
    old_acceptance = json.loads(old_git('compatibility/full-acceptance.json'))
    check(acceptance['whole_pytorch_rewrite_complete'] is False and acceptance['status'] == 'HOLD' and
          acceptance['native_schema_count'] == 2590 and len(acceptance['gates']) == 17 and
          [(g['id'], g['status']) for g in acceptance['gates']] ==
          [(g['id'], g['status']) for g in old_acceptance['gates']],
          'all 17 full-scope gates and 2590 denominator remain unchanged/open')

    common = reports['hosted-adam-distributions-native.json']
    profile = json.loads(git('tools/adam_distributions/profile.json'))
    check(common['status'] == 'QUALIFIED_COMPONENT_WITH_OPEN_SCOPE' and
          common['native_scoped_qualification_passed'] is True and
          common['runtime_source_sha256'] == profile['runtime_source_sha256'] == SOURCE and
          common['executable_sha256'] == profile['executable_sha256'] == IMAGE,
          '378 combined native report bound to exact committed source and image')
    build = common['build']
    check(build['status'] == 'PASS' and build['runtime_source_sha256'] == SOURCE and
          build['executable_sha256'] == IMAGE and build['units'] == profile['units'] and
          len(build['units']) == 8 and (build['core_functions'], build['core_blocks'], build['core_instructions']) ==
          (378, 4971, 18539), '378 exact native build structure and eight unit pins')
    check(build['aggregate_core_host_check'] is False and build['host_production_numerical_runtime'] is False,
          'per-function host authoring is not relabeled aggregate Core admission or host target runtime')
    check(build['compiler_sha256'] == profile['compiler_sha256'] ==
          'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f' and
          build['linker_sha256'] == profile['linker_sha256'] ==
          'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16',
          '378 actual ABI3 compiler/linker identities')
    for unit in profile['units']:
        for suffix, key in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
            ending = '/build/adam-distributions/' + unit['name'] + suffix
            matches = [v for k, v in common['input_sha256'].items() if k.endswith(ending)]
            check(matches == [unit[key]], '378 input custody binds ' + unit['name'] + suffix)
    reproduction = common['reproduction']
    check(reproduction['status'] == 'PASS' and reproduction['native_units'] == 8 and
          reproduction['runtime_source_sha256'] == SOURCE and reproduction['executable_sha256'] == IMAGE and
          reproduction['complete_native_compilation_and_linking_executed'] is True and
          reproduction['all_unit_source_object_and_image_bytes_identical'] is True,
          '378 cold actual compiler/linker byte reproduction')
    check([v for k, v in common['input_sha256'].items() if k.endswith('/build/adam-distributions/build.json')] ==
          [reproduction['build_report_sha256']], '378 cold receipt binds actual build report bytes')
    check(common['source_custody_after'] is True and
          [v for k, v in common['input_sha256'].items() if k.endswith('/tests/qualify_adam_distributions.py')] ==
          [sha(git('tests/qualify_adam_distributions.py'))], '378 executed root qualifier source bound to committed bytes')
    check(common['formal_report_count'] == 15 and common['producer_suite_count'] == 13 and
          len(common['reports']) == len(common['report_sha256']) == 16,
          '378 producer summary plus 13 suites and two independent reports')
    child_formats = {}
    for name, value in common['reports'].items():
        matching = []
        for sorted_keys in (False, True):
            for ascii_only in (False, True):
                raw = (json.dumps(value, indent=2, sort_keys=sorted_keys, ensure_ascii=ascii_only) + '\n').encode()
                if sha(raw) == common['report_sha256'][name]:
                    matching.append({'indent': 2, 'sort_keys': sorted_keys, 'ensure_ascii': ascii_only})
        check(bool(matching), name + ': embedded content recovers declared original report SHA')
        child_formats[name] = matching
    producer = common['reports']['producer']
    check(producer['status'] == 'QUALIFIED_COMPONENT_WITH_OPEN_SCOPE' and
          producer['suite_count'] == 13 and [s['name'] for s in producer['suites']] == NAMES and
          producer['unexpected_failures'] == 0 and producer['expected_old_negative_failure_count'] == 1,
          '378 exact 13 producer suites with one explicit expected original failure')
    check(producer['runtime_source_sha256'] == SOURCE and producer['executable_sha256'] == IMAGE and
          producer['input_custody_after'] is True and producer['fixed_reference_rows_and_tolerances_unchanged'] is True,
          '378 producer summary source/image/custody/frozen expectation scope')

    bindings = json.loads(git('validation/adam-distributions/reports/test-bindings.json'))
    paths = {
        'atomic_controls.py': 'tools/cpu_distributions/test_controls.py',
        'partial_controls.py': 'tools/cpu_distributions_partial/test_controls.py',
        'owned_entry.py': 'tools/cpu_distributions_partial/test_entry.py',
        'typed_handoffs.py': 'tools/cpu_distributions_partial/test_handoffs.py',
        'reference.py': 'tools/cpu_distributions_partial/reference.py',
        'adam_controls.py': 'tools/owned_adam/test_native.py',
        'adam_guards.py': 'tools/owned_adam/test_guards.py',
        'adam_step_counter.py': 'tools/owned_adam/test_step_counter.py',
    }
    adapted = {}
    for row in bindings['scripts']:
        original = git(paths[row['name']])
        check(sha(original) == row['original_sha256'], row['name'] + ': original frozen test bytes')
        value = original.decode()
        for change in row['exact_edits']:
            check(value.count(change['before']) == change['count'], row['name'] + ': exact primary edit count')
            value = value.replace(change['before'], change['after'])
        check(sha(value.encode()) == row['adapted_sha256'], row['name'] + ': exact primary adapted bytes')
        inverse = value
        for change in reversed(row['exact_edits']):
            inverse = inverse.replace(change['after'], change['before'])
        check(inverse.encode() == original, row['name'] + ': independently reconstructed exact inverse')
        adapted[row['name']] = value.encode()
    derivative = bindings['explicit_new_command_admission_derivative']
    value = adapted['atomic_controls.py'].decode()
    for change in derivative['exact_edits']:
        check(value.count(change['before']) == change['count'], '86/87 derivative exact occurrence count')
        value = value.replace(change['before'], change['after'])
    check(sha(value.encode()) == derivative['adapted_sha256'] and
          derivative['changed_error_expectations'] == [
              {'arguments': [], 'new': 1, 'old': 4, 'opcode': 86},
              {'arguments': [], 'new': 1, 'old': 4, 'opcode': 87}], 'only 86/87 explicit new-command errors changed')
    inverse = value
    for change in reversed(derivative['exact_edits']):
        inverse = inverse.replace(change['after'], change['before'])
    check(inverse.encode() == adapted['atomic_controls.py'], '86/87 derivative exact inverse preserves original failure test')
    adapted['atomic_controls_extended.py'] = value.encode()
    for row in producer['suites']:
        name = row['name']
        detail = common['reports'][name]
        check(row['report_sha256'] == common['report_sha256'][name] and row['reported_status'] == detail['status'],
              name + ': suite links exact actual report')
        expected = 1 if name == 'legacy_atomic_controls' else 0
        check(row['exit'] == row['expected_exit'] == expected, name + ': original actual exit retained')
        script = row['command'][1]
        filename = Path(script).name
        if '/adapted-tests/' in script:
            expected_script = sha(adapted[filename])
        else:
            expected_script = sha(git(script.split('/HeptaTorch/HeptaTorch/', 1)[1]))
        check(row['script_sha256'] == expected_script, name + ': actual executed test source bound')
        if name != 'new_command_admission':
            check(SOURCE in [detail.get(k) for k in ('runtime_source_sha256','native_source_sha256','source_sha256','current_source_sha256')] and
                  IMAGE in [detail.get(k) for k in ('executable_sha256','native_executable_sha256','native_sha256','current_executable_sha256')],
                  name + ': complete combined source/image identity')
    check(common['reports']['legacy_atomic_controls']['status'] == 'FAIL' and
          common['expected_legacy_unknown_opcode_failure_count'] == 1 and
          common['old_atomic_test_reclassified_as_pass'] is False and
          producer['original_atomic_unknown_opcode_control_passed'] is False and
          producer['old_atomic_control_known_failure_preserved'] is True,
          'original atomic 86/87 negative remains FAIL, not reclassified PASS')

    distribution_raw = git('tools/cpu_distributions/evidence/torch-bernoulli-dropout.json')
    optimizer_raw = git('validation/adam-object-review/torch-optimizer-objects.json')
    check(sha(distribution_raw) == REFERENCE and sha(optimizer_raw) == OPTIMIZER_REFERENCE,
          'both original actual 183/24 reference byte identities')
    distribution = json.loads(distribution_raw)
    optimizer = json.loads(optimizer_raw)
    provenance(distribution['oracle_provenance'], '378 fixed original183')
    provenance(optimizer['oracle_provenance'], '378 fixed original24')
    for name, late, expected_counts in (
            ('atomic_reference', LATE, {'PASS': 161, 'OPEN_TRANSACTION_DIFFERENCE': 4, 'OPEN_UNREPRESENTED': 18}),
            ('explicit_reference', [], {'PASS': 165, 'OPEN_UNREPRESENTED': 18})):
        reference = common['reports'][name]
        check(reference['reference_sha256'] == REFERENCE and reference['reference_cases'] ==
              len(reference['rows']) == len(distribution['rows']) == 183, name + ': exact original183 reference')
        for index, (row, original) in enumerate(zip(reference['rows'], distribution['rows'], strict=True)):
            expected = 'OPEN_UNREPRESENTED' if index in OPEN else 'OPEN_TRANSACTION_DIFFERENCE' if index in late else 'PASS'
            check((row['index'], row['name'], row['family'], row['status']) ==
                  (index, original['name'], original['family'], expected), name + ': exact index/name/family/status ' + str(index))
        check(reference['counts'] == dict(Counter(row['status'] for row in reference['rows'])) == expected_counts,
              name + ': computed exact status counter')
        check(reference['actual_reference_provenance_checks'] == distribution['oracle_provenance']['checks'],
              name + ': all 11 original provenance checks retained')
    old, new = common['reports']['optimizer_reference'], common['reports']['adam_reference']
    check(old['observations_sha256'] == new['actual_reference_sha256'] == OPTIMIZER_REFERENCE and
          len(old['rows']) == len(optimizer['rows']) == 24 and [r['index'] for r in new['rows']] == [21,22,23,24],
          '378 exact original24 indices and four separate Adam additions')
    additions = {r['index']: r for r in new['rows']}
    ledger = []
    for index, (row, original) in enumerate(zip(old['rows'], optimizer['rows'], strict=True), 1):
        chosen = additions.get(index, row)
        check(chosen['case'] == row['case'] == original['case'], 'original24 exact case name ' + str(index))
        if index in additions:
            check(row['status'] == 'OPEN_NOT_EXECUTED' and chosen['status'] == 'QUALIFIED_EXPOSED_SCOPE',
                  'original old24 row remains OPEN alongside separate actual Adam replay ' + str(index))
        ledger.append({'index': index, 'case': chosen['case'], 'status': chosen['status'],
                       'report': 'adam_reference' if index in additions else 'optimizer_reference'})
    check(ledger == common['optimizer_original_24_ledger'] and
          Counter(r['status'] for r in ledger) == {'QUALIFIED_EXPOSED_SCOPE':22,'PARTIAL_WITH_RETAINED_NONLEAF_OPEN':1,'OPEN_NOT_EXECUTED':1} and
          ledger[13]['status'] == 'PARTIAL_WITH_RETAINED_NONLEAF_OPEN' and ledger[19]['status'] == 'OPEN_NOT_EXECUTED',
          '378 exact recomputed 22 qualified, case14 partial, case20 open ledger')
    check((old['frames'], old['exact_comparisons'], old['unexpected_failures'], new['frames'], new['exact_comparisons'], new['failures']) ==
          (2332,9535,0,908,3469,0), '378 original19 and Adam4 exact execution counts')
    check(old['numeric_policy'] == reports['hosted-owned-adam-native.json']['reports']['optimizer-original']['numeric_policy'] and
          new['numeric_policy'] == reports['hosted-owned-adam-native.json']['reports']['adam-21-24']['numeric_policy'],
          '378 numeric tolerances exactly equal qualified Adam predecessor policies')
    check(common['new_independent_cross_contracts'] == 6 and common['cross_contracts_added_to_existing_case_denominators'] is False,
          'six new cross contracts do not inflate frozen183/24 denominator')
    cross = common['reports']['independent-joint']
    check(cross['status'] == 'PASS' and cross['runtime_source_sha256'] == SOURCE and cross['executable_sha256'] == IMAGE and
          (cross['case_count'],cross['frames_before_EOF'],cross['exact_assertions'],cross['exact_float32_words'],
           cross['complete_MT_state_comparisons'],cross['rollback_bytes_compared']) == (6,462,334,249,22,4892928),
          '378 independently executed six cross contracts exact observations')
    check(all(common[k] is False for k in ('all_rng_complete','all_optimizer_complete','public_gradient_successor_composed','full_pytorch_rewrite_complete')) and
          common['whole_project_decision'] == profile['whole_project_decision'] == 'HOLD',
          '378 does not claim full RNG/optimizer/public-grad successor/PyTorch closure')

    transport = json.loads(git('validation/adam-distributions/evidence/archive-parts.json'))
    parts = []
    for part in transport['parts']:
        data = git(part['path'])
        check(len(data) == part['bytes'] and sha(data) == part['sha256'], 'frozen archive transport part exact bytes')
        parts.append(data)
    producer_archive = b''.join(parts)
    archive_results = {}
    for label, raw, path, expected_sha, expected_count, expected_size in (
            ('producer', producer_archive, 'validation/adam-distributions/evidence/archive-manifest.json',
             '0ec89d5c56f5bd3a39db09bf3a9ed56a2c13e759648327f8388488ec0f25a907',3190,139822732),
            ('independent', git('validation/adam-distributions-review/all-raw-observations.tar.xz'),
             'validation/adam-distributions-review/all-raw-observations.json',
             '7b04428fa1098fa8e57c6a6006a0a01d3f4eb67faee7b98f7088c1e3217e5b74',118,18928809)):
        catalog = json.loads(git(path))
        check(sha(raw) == catalog['archive_sha256'] == expected_sha and len(raw) == catalog['archive_bytes'],
              label + ': committed history archive exact original identity')
        expected = {row['path']: row for row in catalog['members']}
        check(len(expected) == len(catalog['members']) == expected_count, label + ': complete unique archive member inventory')
        seen, size, negatives = set(), 0, []
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r|*') as stream:
            for member in stream:
                check(member.isfile() and member.name in expected and member.name not in seen, label + ': member ' + member.name)
                record = expected[member.name]
                digest, length = hashlib.sha256(), 0
                negative = [] if member.name.endswith('legacy_atomic_controls/report.json') else None
                with stream.extractfile(member) as payload:
                    while chunk := payload.read(1 << 20):
                        digest.update(chunk)
                        length += len(chunk)
                        if negative is not None:
                            negative.append(chunk)
                check((length,digest.hexdigest(),member.mode) == (record['bytes'],record['sha256'],record['mode']),
                      label + ': decoded bytes/hash/mode ' + member.name)
                if negative is not None:
                    data = json.loads(b''.join(negative))
                    check(data['status'] == 'FAIL' and data['runtime_source_sha256'] == SOURCE and data['executable_sha256'] == IMAGE,
                          'archived original unknown-opcode failure retains actual combined identity')
                    negatives.append({'path': member.name, 'sha256': record['sha256']})
                seen.add(member.name)
                size += length
        check(seen == set(expected) and size == expected_size, label + ': all history bytes verified')
        if label == 'producer':
            check(len(negatives) >= 1, 'original legacy negative reports remain in frozen raw archive')
        archive_results[label] = {'sha256':expected_sha,'verified_members':len(seen),'verified_expanded_bytes':size,
                                  'retained_negative_reports':negatives}
        check({k:v for k,v in archive_results[label].items() if k!='retained_negative_reports'} == common['raw_history_archives'][label],
              label + ': independent archive verification agrees with hosted custody')

    provenance_paths = []
    def walk(value, path):
        if isinstance(value, dict):
            if 'oracle_provenance' in value:
                provenance(value['oracle_provenance'], path + '/oracle_provenance')
                provenance_paths.append(path + '/oracle_provenance')
            for key, child in value.items():
                if key != 'oracle_provenance':
                    walk(child, path + '/' + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + '/' + str(index))
    for name, value in reports.items():
        walk(value, name)
    check(len(provenance_paths) >= 24, 'all encountered actual oracle provenance blocks independently checked')
    return {'profile':profile,'schema':'heptatorch.independent-pr26-combined-details.v1','status':'PASS',
            'base_commit':BASE,'inherited_source_files_checked':len(inherited),
            'inherited_blob_map_sha256':sha(json.dumps(inherited,sort_keys=True).encode()),
            'GitHub_connector_receipts':connector_inputs,'actual_independently_verified_successful_job_steps':68,
            'artifact_id':artifact['id'],'embedded_report_original_hash_recovery_formats':child_formats,
            'source':SOURCE,'image':IMAGE,'formal_reports':15,'producer_suites':13,
            'expected_original_negative_failures':1,'unexpected_failures':0,
            'distribution_open_indices':OPEN,'old_atomic_late_error_difference_indices':LATE,
            'optimizer_original24_ledger':ledger,'independently_decoded_history_archives':archive_results,
            'all_checked_oracle_provenance_paths':provenance_paths,
            'provenance_blocks_checked':len(provenance_paths),'full_subsystem_gates_still_open':17,
            'general_cmd95_NaN_compatibility_claimed_closed':False,
            'new_public_gradient_successor_qualified_by_this_PR':False,
            'whole_pytorch_rewrite_complete':False,'whole_project_decision':'HOLD'}
