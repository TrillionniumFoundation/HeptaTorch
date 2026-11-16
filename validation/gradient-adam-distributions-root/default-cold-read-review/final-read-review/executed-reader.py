#!/usr/bin/env python3
"""Read-only custody review of the actual candidate05 default cold execution.

Written by the integration author; this is not an additional independent
semantic qualification. It neither imports the aggregator nor executes native
code. All source, binary, input, transcript and JSON observations are read again.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import stat
import struct


SOURCE = '1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae'
IMAGE = '6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe'
AGGREGATE = 'a753d00c5053f0da3d5a146ee3e98587fcb132548c5b36ec1a7872ffba631df8'
SCRIPT = '42295619cdd761b9bc16837aef612285c8defc6a788df06e8d4f4e3cc125baa9'
LOCK = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'report', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    repo, report_file, out = args.repo.resolve(), args.report.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checks = []
    result = {'schema': 'heptatorch.root-cold-integration-read-review.v1', 'status': 'RUNNING',
              'role': 'Read-only receipt/custody review by integration author; no additional independent semantic qualification.',
              'aggregate_path': str(report_file), 'aggregate_sha256': AGGREGATE,
              'native_executed_here': False, 'Torch_executed_here': False,
              'aggregator_imported_or_executed_here': False, 'full_rewrite_complete': False,
              'pytorch_api_denominator': 2590}

    def check(ok, label):
        checks.append({'check': label, 'passed': bool(ok)})
        if not ok:
            raise AssertionError(label)

    def read(path, expected=None):
        path = Path(path)
        raw = path.read_bytes()
        if expected:
            check(hashlib.sha256(raw).hexdigest() == expected, 'raw SHA ' + str(path))
        return json.loads(raw)

    def runtime(report, label):
        sources = [report[k] for k in ('source_sha256', 'native_source_sha256', 'runtime_source_sha256', 'current_source_sha256') if k in report]
        images = [report[k] for k in ('native_sha256', 'image_sha256', 'executable_sha256', 'native_executable_sha256', 'current_executable_sha256') if k in report]
        check(bool(sources) and set(sources) == {SOURCE} and bool(images) and set(images) == {IMAGE}, 'runtime binding ' + label)

    def argument(command, option):
        check(command.count(option) == 1, 'unique argument ' + option)
        return command[command.index(option) + 1]

    def embedded(record, label):
        raw = read(record['path'], record['sha256'])
        check(raw == record['report'], 'complete embedded JSON ' + label)
        if 'bytes' in record:
            check(Path(record['path']).stat().st_size == record['bytes'], 'raw JSON length ' + label)
        runtime(raw, label)
        return raw

    def functions(raw):
        starts = list(re.finditer(rb'^fn ([A-Za-z0-9_]+)\(', raw, re.M))
        definitions = {}
        for i, match in enumerate(starts):
            header = raw[match.start():raw.index(b'\n', match.start())]
            if header.endswith(b' external()'):
                continue
            check(b' entry(' in header, 'definition has a real entry block')
            definitions[match.group(1).decode()] = raw[match.start():starts[i + 1].start() if i + 1 < len(starts) else len(raw)]
        return definitions

    try:
        aggregate = read(report_file, AGGREGATE)
        root = report_file.parent
        check(report_file.stat().st_size == 10419557, 'original aggregate exact byte length')
        check(aggregate['integration_script_sha256'] == SCRIPT and
              digest(repo / 'tests/qualify_gradient_adam_distributions.py') == SCRIPT and
              digest(root / 'executed-integration.py') == SCRIPT, 'installed and actually captured integration script bytes')
        runtime(aggregate, 'aggregate')
        check(aggregate['status'] == 'QUALIFIED_COMPONENT_WITH_OPEN_SCOPE' and
              aggregate['producer_executed_this_invocation'] is True and
              aggregate['producer']['mode'] == 'executed_here', 'default cold branch actually reported')
        check(aggregate['readiness'] == 'HOLD' and aggregate['pytorch_api_denominator'] == 2590 and
              aggregate['full_rewrite_complete'] is False and aggregate['all_rng_complete'] is False and
              aggregate['full_optimizer_complete'] is False and aggregate['new_Torch_executed_here'] is False and
              aggregate['full_32_live_Torch_executed_here'] is False and aggregate['full_35_live_Torch_executed_here'] is False,
              'whole-project and separate live-oracle boundaries')
        check(aggregate['input_file_count'] == len(aggregate['checked_inputs']) == 427 and
              aggregate['input_custody_after'] is True, '427 bound inputs')
        input_bytes = 0
        for name, info in aggregate['checked_inputs'].items():
            p = Path(name)
            check(p.is_file() and p.stat().st_size == info['bytes'] and
                  stat.S_IMODE(p.stat().st_mode) == info['mode'] and digest(p) == info['sha256'],
                  'input bytes/hash/mode ' + name)
            input_bytes += info['bytes']
        producer = root / 'producer'
        native = producer / 'native'
        check(Path(aggregate['producer']['directory']) == producer and
              Path(aggregate['producer']['native_directory']) == native,
              'fresh producer/native path under this invocation')
        expected_outer = ['producer-17', 'native-review-3', 'zero-merge-mathematical', 'actual-zero-merge',
                          'actual-scalar-add-nan', 'native-cross', 'extreme-stride-observe', 'extreme-stride-compare']
        check([s['name'] for s in aggregate['execution_steps']] == expected_outer, 'eight actual outer command records')
        outer_json = {}
        for step in aggregate['execution_steps']:
            command = step['command']
            check(command[1:4] == ['-I', '-S', '-B'] and step['exit_code'] == 0, 'outer command exit ' + step['name'])
            script = Path(command[4])
            check(script.is_file() and str(script.resolve()) in aggregate['checked_inputs'], 'bound outer script ' + step['name'])
            for stream in ('stdout', 'stderr'):
                check(digest(root / step[stream + '_path']) == step[stream + '_sha256'], 'outer original stream ' + step['name'] + '.' + stream)
            raw = read(step['report_path'], step['report_sha256'])
            check(Path(argument(command, '--output')) == Path(step['report_path']).parent,
                  'outer report belongs to output ' + step['name'])
            outer_json[step['name']] = raw
            if step['name'] == 'producer-17':
                check('--producer-review' not in command and Path(argument(command, '--output')) == producer,
                      'actual fresh producer command, no reuse flag')
            if step['name'] not in ('producer-17', 'native-review-3', 'extreme-stride-compare'):
                if '--build' in command:
                    check(Path(argument(command, '--build')) == native, 'current native build argv ' + step['name'])
                else:
                    check(Path(argument(command, '--binary')) == native / 'tensor-gradient-adam-distributions-session' and
                          Path(argument(command, '--source')) == native / 'runtime.hepta', 'current source/image argv ' + step['name'])
        prod = read(producer / 'report.json', aggregate['producer']['original_report_sha256'])
        check(prod == aggregate['producer']['original_report'] == outer_json['producer-17'], 'whole producer report exact embed/outer binding')
        check(prod['status'] == 'PASS' and prod['step_count'] == len(prod['steps']) == 17 and
              prod['all_package_sources_unchanged'] is True, '17 original producer stages')
        for step in prod['steps']:
            name = step['name']; child = aggregate['producer']['children'][name]
            check(step['exit_code'] == 0 and child['sha256'] == step['report_sha256'] and
                  Path(child['path']) == producer / step['report_path'], 'producer child identity ' + name)
            data = embedded(child, name)
            check(data['status'] == step['report_status'], 'producer child original status ' + name)
            for stream in ('stdout', 'stderr'):
                check(digest(producer / (name + '.' + stream)) == step[stream + '_sha256'], 'producer original stream ' + name + '.' + stream)
            command = step['command']
            if name != 'cold-native-build':
                option = '--binary' if '--binary' in command else '--executable'
                check(Path(argument(command, option)) == native / 'tensor-gradient-adam-distributions-session',
                      '17-stage current image argv ' + name)
            else:
                check(Path(argument(command, '--output')) == native, 'native cold-build output argv')
        build = read(native / 'build.json')
        profile = read(repo / 'tools/gradient_adam_distributions/profile.json')
        check(build['units'] == profile['units'] and len(build['units']) == 9, 'all nine pinned module receipts')
        check((build['functions'], build['blocks'], build['instructions']) == (428, 5738, 21348), 'current core exact statistics')
        check(digest(native / 'runtime.hepta') == SOURCE and digest(native / 'tensor-gradient-adam-distributions-session') == IMAGE,
              'actual cold source and image bytes')
        for row in build['units']:
            check(digest(native / (row['name'] + '.hepta')) == row['source_sha256'] and
                  digest(native / (row['name'] + '.o')) == row['object_sha256'], 'actual nine-unit source/object ' + row['name'])
        executable = (native / 'tensor-gradient-adam-distributions-session').read_bytes()
        check(executable[:7] == b'\x7fELF\x02\x01\x01' and struct.unpack_from('<HH', executable, 16) == (2, 62), 'actual Linux x86-64 ET_EXEC')
        phoff = struct.unpack_from('<Q', executable, 32)[0]
        phsize, phcount = struct.unpack_from('<HH', executable, 54)
        types = [struct.unpack_from('<I', executable, phoff + i * phsize)[0] for i in range(phcount)]
        check(2 not in types and 3 not in types, 'no dynamic segment or ELF interpreter')
        ntop = read(root / 'native-review/report.json', aggregate['native_review']['original_report_sha256'])
        check(ntop == aggregate['native_review']['original_report'] == outer_json['native-review-3'] and
              ntop['status'] == 'PASS' and ntop['suite_count'] == len(ntop['suites']) == 3, 'three native gates original wrapper')
        for step in ntop['suites']:
            child = aggregate['native_review']['children'][step['name']]
            check(child['sha256'] == step['report_sha256'] and step['exit'] == 0, 'native child original execution ' + step['name'])
            embedded(child, step['name'])
        compiled = aggregate['native_review']['children']['recompile']['report']
        bodies = functions((native / 'runtime.hepta').read_bytes())
        check(len(bodies) == len(compiled['core_body_bindings']) == 428 and set(bodies) == set(compiled['core_body_bindings']), '428 complete source bodies')
        units = {u['name']: functions((native / (u['name'] + '.hepta')).read_bytes()) for u in build['units']}
        core_units = {binding['unit'] for binding in compiled['core_body_bindings'].values()}
        check(core_units == set(units) - {'rng03', 'owners'}, 'exactly seven current-runtime partition units')
        for name, binding in compiled['core_body_bindings'].items():
            body = bodies[name]
            check(len(body) == binding['bytes'] and hashlib.sha256(body).hexdigest() == binding['sha256'] and
                  units[binding['unit']][name] == body and sum(name in units[unit] for unit in core_units) == 1,
                  'source/body/unit exact once ' + name)
        check(compiled['actual_EOF_cleanup_exit'] == 0 and compiled['all_object_and_image_bytes_identical'] is True,
              'actual image reproduction and EOF execution receipt')
        for name, child in aggregate['peer_review'].items():
            data = embedded(child, name)
            check(data == outer_json[name] and data['status'] == 'PASS', 'peer actual command result ' + name)
        observation = embedded(aggregate['extreme_stride']['observation'], 'extreme-observer')
        strict = embedded(aggregate['extreme_stride']['comparison'], 'extreme-strict')
        check(observation == outer_json['extreme-stride-observe'] and strict == outer_json['extreme-stride-compare'], 'extreme raw outer result binding')
        check(strict['native_observations_sha256'] == aggregate['extreme_stride']['observation']['sha256'] and
              (strict['cases'], strict['passing_cases'], strict['failing_cases'], strict['field_comparisons'], strict['different_fields']) == (8, 8, 0, 344, 0),
              'strict8 actually consumed fresh80-frame observation and all344fields')
        check(observation['frames'] == 80 and observation['actual_torch_compared'] is False, 'observe before compare, no relabel')
        reference_paths = sorted((repo / 'validation/gradient-adam-distributions/reference').glob('*.json'))
        reference_paths += [repo / 'validation/adam-distributions-gradient-review/actual/torch-zero-merge-layout.json',
                            repo / 'validation/adam-distributions-gradient-review/actual/torch-scalar-add-nan.json',
                            repo / 'validation/gradient-composition-review/extreme-stride/torch-extreme-stride.json']
        for path in reference_paths:
            value = read(path); provenance = value['oracle_provenance']
            check(value['status'] == 'OBSERVED' and provenance['status'] == 'PASS' and provenance['lock_sha256'] == LOCK and
                  set(provenance['checks']) == CHECKS and all(v is True for v in provenance['checks'].values()),
                  'all11 original provenance ' + str(path.relative_to(repo)))
        dist_ref = read(repo / 'validation/gradient-adam-distributions/reference/torch-bernoulli-dropout.json')
        open_indices = [4, 9, 14, 19, 24, 29, 34, 39, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174]
        for label, counts in [('distribution-atomic-183', {'PASS': 161, 'OPEN_TRANSACTION_DIFFERENCE': 4, 'OPEN_UNREPRESENTED': 18}),
                              ('distribution-explicit-183', {'PASS': 165, 'OPEN_UNREPRESENTED': 18})]:
            value = aggregate['producer']['children'][label]['report']
            check(len(value['rows']) == 183 and value['counts'] == counts and dict(Counter(x['status'] for x in value['rows'])) == counts,
                  'original183 classifications ' + label)
            check([x['index'] for x in value['rows'] if x['status'] == 'OPEN_UNREPRESENTED'] == open_indices,
                  'same18 exact open indices ' + label)
            check([(x['index'], x['name'], x['family']) for x in value['rows']] ==
                  [(i, x['name'], x['family']) for i, x in enumerate(dist_ref['rows'])], 'all183 index/name/family ' + label)
        union = aggregate['optimizer_reference_union']
        check(union['original_reference_cases'] == 24 and union['qualified_exposed_scope'] == 23 and union['partial_cases'] == 1 and
              union['qualified_one_based_indices'] == sorted(set(range(1, 25)) - {14}) and union['partial_one_based_indices'] == [14],
              'unchanged24 reference union23 plus1 partial')
        check(aggregate['legacy_regression_boundaries']['presence8'] == {'state_version_differences': 8, 'state_creation_differences': 65} and
              aggregate['legacy_regression_boundaries']['gradient_clear10'] == {'state_version_differences': 11, 'state_creation_differences': 9, 'missing_step_Tensor_observations': 30, 'Adam_different_bits_with_original_tolerance': 6},
              'legacy differences remain exact')
        history = aggregate['retained_history']['candidate04_extreme_stride_failure']
        old = read(repo / history['path'], history['sha256'])
        check(old == history['report'] and old['status'] == 'FAIL' and (old['passing_cases'], old['failing_cases']) == (4, 4), 'original04 actual failure retained verbatim')
        check(digest(report_file) == AGGREGATE, 'aggregate unchanged after read review')
        result.update(status='PASS_READ_ONLY_REVIEW', aggregate_bytes=report_file.stat().st_size,
                      source_sha256=SOURCE, native_sha256=IMAGE, integration_script_sha256=SCRIPT,
                      input_files_rehashed=427, input_bytes_rehashed=input_bytes,
                      actual_outer_execution_records=8, fresh_producer_stages=17,
                      native_review_stages=3, independent_peer_runners=4, strict_extreme_cases=8,
                      complete_body_bindings_rechecked=428, actual_module_pairs_rehashed=9,
                      actual_reference_reports_with_all11_provenance=len(reference_paths),
                      default_producer_native_output_verified=str(native),
                      readiness='HOLD')
    except Exception as error:
        result.update(status='FAIL_READ_ONLY_REVIEW', error=repr(error))
    result['check_count'] = len(checks)
    result['checks'] = checks
    result['reader_sha256'] = digest(Path(__file__).resolve())
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'executed-reader.py').write_bytes(Path(__file__).resolve().read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))
    return 0 if result['status'] == 'PASS_READ_ONLY_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
