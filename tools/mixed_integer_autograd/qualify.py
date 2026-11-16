"""Cold-build mixed autograd and execute fixed numerical and inherited gates.

The general owner/resource, owned-entry, typed handoff, and live Torch gates
remain separately required integration evidence. The independent public Saved
lifetime reader is executed unchanged as the fifth cold-build gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from archive_evidence import archive

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
SHA = lambda raw: hashlib.sha256(raw).hexdigest()
CHECKS = ('python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'repository', 'upstream', 'compiler', 'linker',
                'flat-compiler', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = a.repository.resolve()
    tests = PACKAGE / 'tests/mixed_integer_autograd_review'
    profile_path = HERE / 'profile.json'
    manifest_path = HERE / 'package-manifest.json'
    profile_raw = profile_path.read_bytes()
    manifest_raw = manifest_path.read_bytes()
    profile = json.loads(profile_raw)
    manifest = json.loads(manifest_raw)
    external = {}
    external_manifests = {}
    report = {
        'schema': 'heptatorch.mixed-integer-autograd-portable-qualification.v1',
        'status': 'RUNNING', 'source_sha256': profile['source_sha256'],
        'native_sha256': profile['native_sha256'],
        'profile_sha256': SHA(profile_raw),
        'package_manifest_sha256': SHA(manifest_raw), 'steps': [],
        'native_cold_rebuild_executed': False, 'pinned_Torch_executed_here': False,
        'counting_boundary': 'The 36 mixed cases, fourteen public Saved lifetime cases, eight I64 gates and twelve legacy gates overlap. These are not a sum of upstream APIs.',
        'independent_integration_reports_required': [
            'general I64 owner and mixed Saved strong-root accounting',
            'resource and output capacity controls',
            'typed previous-profile handoff',
            'live32 and correctly adapted accounted35 with negative controls'],
        'full_pytorch_rewrite_complete': False, 'pytorch_api_denominator': 2590}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def verify_member(base, relative, pin):
        relative = Path(relative)
        assert not relative.is_absolute() and '..' not in relative.parts
        path = base / relative
        raw = path.read_bytes()
        assert len(raw) == pin['bytes'] and SHA(raw) == pin['sha256'], str(path)
        assert path.stat().st_mode & 511 == pin['mode'], str(path)
        return path, raw

    def custody():
        assert manifest_path.read_bytes() == manifest_raw
        assert profile_path.read_bytes() == profile_raw
        for relative, pin in manifest['files'].items():
            verify_member(PACKAGE, relative, pin)
        for path, pin in external.items():
            raw = path.read_bytes()
            assert len(raw) == pin['bytes'] and SHA(raw) == pin['sha256']
            assert path.stat().st_mode & 511 == pin['mode']
        for path, raw in external_manifests.items():
            assert path.read_bytes() == raw

    def execute(name, script, arguments, receipt, expected_status, counts,
                unchanged_field='all_inputs_unchanged', compact=True,
                python_options=('-B',)):
        command = [sys.executable, *python_options, str(script), *map(str, arguments)]
        row = {'name': name, 'command': command, 'status': 'STARTED',
               'process_started': False, 'process_completed': False}
        report['steps'].append(row)
        save()
        with (out / (name + '.stdout')).open('wb') as so, (out / (name + '.stderr')).open('wb') as se:
            process = subprocess.Popen(command, stdout=so, stderr=se)
            row['process_started'] = True
            save()
            try:
                code = process.wait(timeout=1800)
            except BaseException:
                process.kill()
                process.wait()
                raise
        row.update(process_completed=True, exit_code=code,
                   stdout_sha256=SHA((out / (name + '.stdout')).read_bytes()),
                   stderr_sha256=SHA((out / (name + '.stderr')).read_bytes()))
        if receipt.exists():
            raw = receipt.read_bytes()
            value = json.loads(raw)
            row.update(status=value['status'], report=str(receipt.relative_to(out)),
                       report_sha256=SHA(raw))
        else:
            value = None
            row['status'] = 'NO_REPORT'
        save()
        assert code == 0, (name, code)
        assert value['status'] == expected_status, (name, value['status'])
        for key, wanted in counts.items():
            assert value[key] == wanted, (name, key, value[key], wanted)
        assert value['source_sha256'] == profile['source_sha256']
        assert value['native_sha256'] == profile['native_sha256']
        assert value[unchanged_field] is True
        if compact:
            row['archive'] = archive(receipt.parent, out / 'evidence', name)
        save()
        return value

    save()
    try:
        for item in profile['external_manifests']:
            path = repo / item['path']
            raw = path.read_bytes()
            assert SHA(raw) == item['sha256']
            external_manifests[path] = raw
            files = json.loads(raw)['files']
            iterable = files.items() if isinstance(files, dict) else ((r['path'], r) for r in files)
            for relative, pin in iterable:
                member, _ = verify_member(repo, relative, pin)
                if member in external:
                    assert all(external[member][k] == pin[k] for k in ('bytes', 'sha256', 'mode'))
                external[member] = {k: pin[k] for k in ('bytes', 'sha256', 'mode')}
        custody()
        assert SHA(a.flat_compiler.read_bytes()) == profile['flat_compiler_sha256']
        ref = profile['fixed_reference']
        reference = PACKAGE / ref['path']
        raw = reference.read_bytes()
        actual = json.loads(raw)
        assert len(raw) == ref['bytes'] and SHA(raw) == ref['sha256']
        assert SHA((PACKAGE / ref['probe_path']).read_bytes()) == ref['probe_sha256']
        assert actual['status'] == 'OBSERVED' and actual['cases'] == 36
        assert actual['probe_error_count'] == 0 and actual['source_files_unchanged'] is True
        assert actual['probe_sha256'] == ref['probe_sha256']
        provenance = actual['oracle_provenance']
        assert provenance['status'] == 'PASS' and set(provenance['checks']) == set(CHECKS)
        assert all(provenance['checks'][key] is True for key in CHECKS)
        saved = profile['independent_saved_lifetime14']
        saved_manifest_raw = (PACKAGE / saved['manifest_path']).read_bytes()
        assert SHA(saved_manifest_raw) == saved['manifest_sha256']
        saved_manifest = json.loads(saved_manifest_raw)
        assert saved_manifest['file_count'] == 17
        for relative, pin in saved_manifest['files'].items():
            verify_member(PACKAGE, relative, pin)
        for role in ('reference', 'probe', 'reader', 'observer'):
            raw = (PACKAGE / saved[role + '_path']).read_bytes()
            assert SHA(raw) == saved[role + '_sha256'], role
        saved_actual = json.loads((PACKAGE / saved['reference_path']).read_bytes())
        assert saved_actual['status'] == 'OBSERVED' and saved_actual['cases'] == 14
        assert saved_actual['probe_error_count'] == 0
        assert saved_actual['source_files_unchanged'] is True
        assert saved_actual['probe_sha256'] == saved['probe_sha256']
        saved_provenance = saved_actual['oracle_provenance']
        assert saved_provenance['status'] == 'PASS'
        assert set(saved_provenance['checks']) == set(CHECKS)
        assert all(saved_provenance['checks'][key] is True for key in CHECKS)
        report['fixed_reference'] = ref
        report['independent_saved_lifetime14'] = saved
        report['external_manifest_pins'] = profile['external_manifests']
        report['external_package_file_count'] = len(external)
        save()
        native = out / 'native'
        execute('cold-native-build', HERE / 'rebuild.py', [
            '--parent', a.parent.resolve(), '--repository', repo,
            '--upstream', a.upstream.resolve(), '--compiler', a.compiler.resolve(),
            '--linker', a.linker.resolve(), '--output', native],
            native / 'profile-rebuild.json', 'PASS', {'units_reproduced_exactly': 11},
            unchanged_field='all_package_members_unchanged', compact=False)
        report['native_cold_rebuild_executed'] = True
        save()
        value = execute('fixed-mixed-actual36', tests / 'replay_actual36.py', [
            '--binary', native / 'tensor-mixed-autograd-session',
            '--source', native / 'runtime.hepta', '--build-report', native / 'build.json',
            '--protocol', repo / 'tests/retained_nonleaf_review/protocol.py',
            '--observer', tests / 'native_observer.py', '--reference', reference,
            '--expected-source-sha256', profile['source_sha256'],
            '--expected-native-sha256', profile['native_sha256'],
            '--output', out / 'fixed-mixed-actual36'],
            out / 'fixed-mixed-actual36/report.json', 'PASS', {
                'cases': 36, 'passed': 36, 'failed': 0, 'frames': 4728,
                'checks': 21477, 'exact_words': 4670, 'rollback_bytes': 58018992,
                'rng_bytes': 940416, 'reference_indices': list(range(36))})
        assert value['reference_sha256'] == ref['sha256']
        assert value['probe_sha256'] == ref['probe_sha256']
        assert set(value['provenance_checks']) == set(CHECKS)
        assert all(value['provenance_checks'][key] is True for key in CHECKS)
        value = execute('inherited-I64-eight', tests / 'run_inherited_integer.py', [
            '--build', native, '--repo', repo, '--output', out / 'inherited-I64-eight'],
            out / 'inherited-I64-eight/report.json', 'PASS', {})
        assert len(value['steps']) == 8 and all(r['status'] == 'PASS' for r in value['steps'])
        execute('legacy523-twelve', tests / 'run_legacy523.py', [
            '--native-build', native, '--repository', repo,
            '--adapters', tests / 'legacy-adapters',
            '--flat-compiler', a.flat_compiler.resolve(), '--output', out / 'legacy523-twelve'],
            out / 'legacy523-twelve/report.json', 'PASS', {
                'step_count': 12, 'existing_native_units_verified': 11,
                'all_expected_statuses_and_counts_matched': True})
        value = execute('fixed-saved-public14', PACKAGE / saved['reader_path'], [
            '--binary', native / 'tensor-mixed-autograd-session',
            '--source', native / 'runtime.hepta', '--build-report', native / 'build.json',
            '--protocol', repo / 'tests/retained_nonleaf_review/protocol.py',
            '--observer', PACKAGE / saved['observer_path'],
            '--reference', PACKAGE / saved['reference_path'],
            '--expected-source-sha256', profile['source_sha256'],
            '--expected-native-sha256', profile['native_sha256'],
            '--output', out / 'fixed-saved-public14'],
            out / 'fixed-saved-public14/report.json', 'PASS', {
                'cases': 14, 'passed': 14, 'failed': 0, 'frames': 2058,
                'checks': 8842, 'exact_words': 2166, 'rollback_bytes': 27261936,
                'rng_bytes': 480320, 'reference_indices': list(range(14))},
            python_options=('-I', '-S', '-B'))
        assert value['reference_sha256'] == saved['reference_sha256']
        assert value['probe_sha256'] == saved['probe_sha256']
        assert set(value['provenance_checks']) == set(CHECKS)
        assert all(value['provenance_checks'][key] is True for key in CHECKS)
        custody()
        report.update(status='PASS', step_count=5, all_package_members_unchanged=True,
                      all_external_package_members_unchanged=True,
                      all_named_gate_statuses_and_counts_matched=True)
        save()
        print(json.dumps({k: v for k, v in report.items() if k not in ('steps', 'fixed_reference')}))
        return 0
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
