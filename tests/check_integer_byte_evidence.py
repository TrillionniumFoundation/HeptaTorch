#!/usr/bin/env python3
"""Verify complete original integer execution bytes; never run native or Torch code."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import stat
import struct
import traceback

PINS = {'i64': {'kernel': '251705542f4e117e3c9cdaec36cead8ab38252b24ebfef1104d765c546fd0716', 'elf': 'fbb6cf1648385be200c94fc14ed73d1539cba3b0239aeea4f7866058c8395ebc', 'cases': 33, 'values': 351174, 'oracle': '1896142e1c25abe9478e84ffc7ffe735fd82ef3ba129504bd40e6dbdcc777f74'}, 'i8': {'kernel': '7ca1765e2cede95ec19b64742f69816581733619b5b1b844f57aa72b743ed8d3', 'elf': 'eba8c05bfe514c38ffa1a5e1ccbc9aacd513531f74becd18f4e3dca73db0a0b1', 'cases': 28, 'values': 1093216, 'oracle': '3eb05c82a7951be3bb2b21349f8af903bd0f92f3bb00243f39c25be250f7fac9'}}
OBSERVER = '6a5ff8a8a6f920f8734f9c5a26533ba26d71965d8de4fbbf1e7f7aa1c631f453'
COMPILER = '87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54'
SUPPORT = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
PROVENANCE_NAMES = ['direct_url_archive_sha256_matches', 'direct_url_matches', 'distribution_version_matches', 'git_matches_oracle_build_pin', 'pip_report_archive_sha256_matches', 'pip_report_requested', 'pip_report_url_matches', 'pip_report_version_matches', 'platform_matches', 'python_matches', 'version_backend_matches']
PROVENANCE_NAMES = set(PROVENANCE_NAMES)

def sha(data):
    return hashlib.sha256(data).hexdigest()



def corpus64():
    mask = (1 << 64) - 1
    values = [0, 1, 2, 3, 7, 255, 256, 1024, (1 << 16) - 1, 1 << 16,
              (1 << 31) - 1, 1 << 31, (1 << 32) - 1, 1 << 32,
              (1 << 53) - 1, 1 << 53, (1 << 63) - 1, 1 << 63, mask, mask - 1, mask - 2,
              0x5555555555555555, 0xaaaaaaaaaaaaaaaa]
    edges = [(a, b) for a in values for b in values]
    for mode in range(6):
        yield 'empty-mode-' + str(mode), mode, []
        yield 'edge-product-mode-' + str(mode), mode, edges
        rng = random.Random(192830 + mode * 19)
        yield '10k-random-mode-' + str(mode), mode, [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(10000)]
        yield '48k-boundary-mode-' + str(mode), mode, [
            (rng.getrandbits(64), rng.getrandbits(64)) if i % 13 else edges[i % len(edges)] for i in range(48000)]



def corpus8():
    exhaustive = [(a, b) for a in range(256) for b in range(256)]
    for mode in range(6):
        yield 'empty-' + str(mode), mode, []
        yield 'exhaustive-' + str(mode), mode, exhaustive
        rng = random.Random(20261116 + mode * 1009)
        yield '100k-random-' + str(mode), mode, [(rng.randrange(256), rng.randrange(256)) for _ in range(100000)]
    rng = random.Random(836485)
    yield '100k-admission-boundary', 2, [(rng.randrange(256), rng.randrange(256)) for _ in range(100000)]



def packet(profile, mode, pairs):
    payload = b''.join(struct.pack('<QQ', a, b) for a, b in pairs) if profile == 'i64' else bytes(v for pair in pairs for v in pair)
    return struct.pack('<II', mode, len(pairs)) + payload



def expected(profile):
    mask = (1 << (64 if profile == 'i64' else 8)) - 1
    output = []
    for name, mode, pairs in (corpus64() if profile == 'i64' else corpus8()):
        values = [((a + b, a - b, a * b, a & b, a | b, a ^ b)[mode] & mask) for a, b in pairs]
        payload = b''.join(struct.pack('<Q', value) for value in values) if profile == 'i64' else bytes(values)
        output.append({'case': name, 'mode': mode, 'count': len(pairs), 'compared_values': len(pairs),
                       'input_sha256': sha(packet(profile, mode, pairs)),
                       'response_sha256': sha(bytes(4) + payload), 'reference_sha256': sha(payload)})
    if profile == 'i64':
        refused = [('short-header', bytes(7)), ('unknown-mode', struct.pack('<II', 6, 0)),
                   ('u32-mode', struct.pack('<II', 0xffffffff, 0)),
                   ('truncated-pair', struct.pack('<II', 0, 1) + bytes(15)),
                   ('count-too-large', struct.pack('<II', 1, 48001)),
                   ('count-max', struct.pack('<II', 0, 0xffffffff)),
                   ('payload-after-empty', struct.pack('<II', 2, 0) + b'X'),
                   ('payload-after-pair', packet(profile, 3, [(1, 2)]) + b'X'),
                   ('wrong-count', struct.pack('<II', 4, 2) + struct.pack('<QQ', 3, 4))]
    else:
        refused = [('short-header', bytes(7)), ('unknown-mode', struct.pack('<II', 6, 0)),
                   ('mode-fmax', struct.pack('<II', 0xffffffff, 0)),
                   ('count-over', struct.pack('<II', 0, 100001)), ('count-u32max', struct.pack('<II', 0, 0xffffffff)),
                   ('truncated-pair', struct.pack('<II', 0, 1) + bytes(1)),
                   ('trailing-empty', struct.pack('<II', 1, 0) + b'x'),
                   ('trailing-pair', packet(profile, 2, [(0, 255)]) + b'x'),
                   ('count-mismatch', struct.pack('<II', 0, 2) + bytes(2))]
    for name, raw in refused:
        output.append({'case': 'refuse-' + name, 'input_sha256': sha(raw), 'response_sha256': sha(b'\x01\0\0\0')})
    return output



class Checks:
    def __init__(self):
        self.rows = []

    def check(self, name, condition):
        self.rows.append({'name': name, 'passed': condition is True})
        if condition is not True:
            raise AssertionError(name)

    def equal(self, name, actual, expected):
        self.check(name, type(actual) is type(expected) and actual == expected)


def verify_provenance_fields(provenance, lock, c):
    """Check real recorded metadata, independently of the eleven boolean flags.

    The lock's complete wheel URL and both archive hash representations are
    required. Original CI paths remain observations; they are never rewritten
    to this offline reader's local extraction paths.
    """
    c.equal('provenance-fields/top-level-keys', set(provenance),
            {'status', 'lock_path', 'lock_sha256', 'expected', 'observed',
             'checks', 'metadata_error', 'release_source_equivalence'})
    c.equal('provenance-fields/status', provenance['status'], 'PASS')
    c.equal('provenance-fields/lock-sha', provenance['lock_sha256'], LOCK)
    c.equal('provenance-fields/lock-basename', Path(provenance['lock_path']).name,
            'pytorch-oracle-lock.json')
    c.equal('provenance-fields/expected-entire-object', provenance['expected'], lock['oracle'])
    c.equal('provenance-fields/source-equivalence-entire-object',
            provenance['release_source_equivalence'], lock['release_source_equivalence'])
    c.equal('provenance-fields/eleven-check-names', set(provenance['checks']), PROVENANCE_NAMES)
    for name, value in provenance['checks'].items():
        c.check('provenance-fields/executed-check/' + name, value is True)
    c.equal('provenance-fields/metadata-error', provenance['metadata_error'], None)
    observed = provenance['observed']
    expected = lock['oracle']
    c.equal('provenance-fields/observed-keys', set(observed),
            {'python', 'platform', 'version', 'git_commit', 'backend_versions',
             'distribution_version', 'direct_url', 'install_report'})
    for name in ('version', 'git_commit', 'backend_versions', 'platform'):
        c.equal('provenance-fields/actual-' + name, observed[name], expected[name])
    c.equal('provenance-fields/python-keys', set(observed['python']),
            {'implementation', 'version', 'full_version'})
    for name in ('implementation', 'version'):
        c.equal('provenance-fields/python-' + name, observed['python'][name], expected['python'][name])
    c.check('provenance-fields/python-full-version-observed',
            type(observed['python']['full_version']) is str and bool(observed['python']['full_version']))
    c.equal('provenance-fields/distribution-version', observed['distribution_version'], expected['version'])
    wheel = expected['wheel']
    expected_archive = {'hash': 'sha256=' + wheel['sha256'], 'hashes': {'sha256': wheel['sha256']}}
    expected_download = {'url': wheel['url'], 'archive_info': expected_archive}
    c.equal('provenance-fields/direct-url-entire-object', observed['direct_url'], expected_download)
    recorded = observed['install_report']
    c.equal('provenance-fields/install-report-keys', set(recorded),
            {'path', 'pip_version', 'download_info', 'metadata_name', 'metadata_version', 'requested'})
    c.equal('provenance-fields/install-report-path', recorded['path'], expected['install_report'])
    c.equal('provenance-fields/install-download-entire-object', recorded['download_info'], expected_download)
    c.equal('provenance-fields/install-metadata-name', recorded['metadata_name'], expected['distribution'])
    c.equal('provenance-fields/install-metadata-version', recorded['metadata_version'], expected['version'])
    c.check('provenance-fields/install-requested', recorded['requested'] is True)
    c.check('provenance-fields/pip-version-observed',
            type(recorded['pip_version']) is str and bool(recorded['pip_version']))


def verify(directory, profile, reference, c):
    spec = PINS[profile]
    manifest_path = directory / 'byte-evidence.json'
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    c.equal('manifest-schema', manifest['schema'], 'heptatorch.integer-original-byte-evidence.v1')
    c.equal('complete-recorded-execution', manifest['status'], 'COMPLETE_RECORDED_EXECUTION')
    c.equal('profile', manifest['profile'], profile)
    c.equal('reference', manifest['reference'], reference)
    c.equal('only-manifest-self-exclusion', manifest['manifest_self_exclusion'], 'byte-evidence.json')
    c.check('no-runtime-modification', manifest['runtime_behavior_modified'] is False)
    c.check('empty-Torch-call-not-invented', manifest['empty_framework_operation_claimed'] is False)
    roles = {'kernel.hepta', 'compiler.elf', 'oracle.py', 'observer.py', 'reader.py', 'pytorch_oracle.py', 'pytorch-oracle-lock.json'}
    if reference == 'pytorch':
        roles.add('pytorch-install.json')
    c.equal('all-input-roles', set(manifest['inputs']), roles)
    c.equal('all-input-unchanged-roles', set(manifest['inputs_unchanged']), roles)
    for role in roles:
        c.check('unchanged-input/' + role, manifest['inputs_unchanged'][role] is True)
    wanted = expected(profile)
    case_records = manifest['cases']
    c.equal('observed-case-count', len(case_records), spec['cases'])
    c.equal('observed-complete-ordered-plan', [r['case'] for r in case_records], [r['case'] for r in wanted])
    c.equal('observed-unique-plan', len({r['case'] for r in case_records}), spec['cases'])
    expected_files = {'report.json', profile + '-native', 'compiler-0.stderr', 'compiler-1.stderr'}
    expected_files.update('evidence/sources/' + role for role in roles)
    for i in (0, 1):
        expected_files.update('evidence/compiler-' + str(i) + suffix for suffix in ('.stdout.bin', '.stderr.bin', '.json'))
    for record, fixture in zip(case_records, wanted):
        prefix = 'evidence/cases/' + fixture['case'] + '/'
        expected_files.update(prefix + name for name in ('request.bin', 'response.bin', 'stderr.bin', 'native.json', 'expected.bin'))
        if not fixture['case'].startswith('refuse-'):
            expected_files.add(prefix + 'reference-cpython.bin')
            if reference == 'pytorch' and fixture['count']:
                expected_files.add(prefix + 'reference-pytorch.bin')
    c.equal('nonempty-explicit-complete-file-inventory', set(manifest['files']), expected_files)
    actual_files = set()
    for path in directory.rglob('*'):
        c.check('no-symlink/' + str(path.relative_to(directory)), not path.is_symlink())
        if path.is_file():
            actual_files.add(path.relative_to(directory).as_posix())
    c.equal('no-missing-or-undeclared-actual-files', actual_files, expected_files | {'byte-evidence.json'})
    data = {}
    for name in sorted(expected_files):
        path = directory / name
        mode = path.lstat().st_mode
        c.check('regular-file/' + name, stat.S_ISREG(mode))
        raw = path.read_bytes(); data[name] = raw
        record = manifest['files'][name]
        c.equal('size/' + name, len(raw), record['bytes'])
        c.equal('sha/' + name, sha(raw), record['sha256'])
        c.equal('mode/' + name, stat.S_IMODE(mode), record['mode'])
    pins = {'kernel.hepta': spec['kernel'], 'compiler.elf': COMPILER,
            'oracle.py': spec['oracle'], 'observer.py': OBSERVER,
            'pytorch_oracle.py': SUPPORT, 'pytorch-oracle-lock.json': LOCK}
    for role in roles:
        r = manifest['inputs'][role]; expected_path = 'evidence/sources/' + role
        c.equal('snapshot-role-path/' + role, r['snapshot'], expected_path)
        c.equal('input-size/' + role, r['bytes'], len(data[expected_path]))
        c.equal('input-sha/' + role, r['sha256'], sha(data[expected_path]))
        c.check('original-path-is-observed-absolute/' + role, Path(r['original_path']).is_absolute())
        if role in pins:
            c.equal('fixed-source-pin/' + role, sha(data[expected_path]), pins[role])
    c.equal('same-offline-reader-source', data['evidence/sources/reader.py'], Path(__file__).read_bytes())
    c.check('actual-compiler-ELF', data['evidence/sources/compiler.elf'].startswith(b'\x7fELF'))
    image = data[profile + '-native']
    c.check('actual-application-ELF', image.startswith(b'\x7fELF'))
    c.equal('actual-application-pin', sha(image), spec['elf'])
    compilers = manifest['compiler_invocations']
    c.equal('two-real-compiler-invocations', len(compilers), 2)
    for i, record in enumerate(compilers):
        prefix = 'evidence/compiler-' + str(i)
        wanted_record = {'generation': i, 'returncode': 0, 'stdout': prefix + '.stdout.bin', 'stderr': prefix + '.stderr.bin'}
        c.equal('compiler-record/' + str(i), record, wanted_record)
        c.equal('compiler-json/' + str(i), json.loads(data[prefix + '.json']), wanted_record)
        c.equal('compiler-produced-same-full-ELF/' + str(i), data[record['stdout']], image)
        c.equal('compiler-original-stderr/' + str(i), data['compiler-' + str(i) + '.stderr'], b'')
        c.equal('compiler-captured-stderr/' + str(i), data[record['stderr']], b'')
    c.equal('original-report-path', manifest['report']['path'], 'report.json')
    c.equal('original-report-size', manifest['report']['bytes'], len(data['report.json']))
    c.equal('original-report-sha', manifest['report']['sha256'], sha(data['report.json']))
    report = json.loads(data['report.json'])
    c.equal('original-report-status', report['status'], 'PASS')
    c.equal('original-report-reference', report['reference'], reference)
    c.equal('original-report-compiler-pin', report['compiler_sha256'], COMPILER)
    c.equal('original-report-HIA-pin', report['source_sha256'], spec['kernel'])
    c.equal('original-report-oracle-pin', report['oracle_script_sha256' if profile == 'i64' else 'oracle_source_sha256'], spec['oracle'])
    c.equal('original-report-ELF-pin', report['native_elf_sha256' if profile == 'i64' else 'executable_sha256'], spec['elf'])
    c.equal('original-report-count', report['tests' if profile == 'i64' else 'cases'], spec['cases'])
    c.equal('original-report-failures', report['failed_tests' if profile == 'i64' else 'failed_cases'], 0)
    c.equal('original-report-total-values', report['compared_values'], spec['values'])
    c.equal('original-report-rows', len(report['rows']), spec['cases'])
    c.equal('original-report-plan', [r['case'] for r in report['rows']], [r['case'] for r in wanted])
    total = 0
    for observed, fixture, row in zip(case_records, wanted, report['rows']):
        name = fixture['case']; prefix = 'evidence/cases/' + name + '/'
        c.equal(name + '/original-PASS', row['status'], 'PASS')
        c.check(name + '/no-old-error', not ({'error', 'traceback', 'reference_disagreements', 'reference_crosscheck_errors'} & set(row)))
        if 'mismatches' in row:
            c.equal(name + '/zero-mismatches', row['mismatches'], [])
        c.equal(name + '/request-path', observed['request'], prefix + 'request.bin')
        native = {'returncode': 0, 'stdout': prefix + 'response.bin', 'stderr': prefix + 'stderr.bin'}
        c.equal(name + '/native-record', observed['native'], native)
        c.equal(name + '/native-json', json.loads(data[prefix + 'native.json']), native)
        c.equal(name + '/native-stderr-empty', data[native['stderr']], b'')
        c.equal(name + '/complete-request-sha', sha(data[observed['request']]), fixture['input_sha256'])
        c.equal(name + '/complete-response-sha', sha(data[native['stdout']]), fixture['response_sha256'])
        for key, value in fixture.items():
            mapped = {'response_sha256': 'output_sha256', 'reference_sha256': 'expected_sha256'}.get(key, key) if profile == 'i8' else key
            c.equal(name + '/original-row/' + mapped, row[mapped], value)
        selection = observed['expected']
        c.equal(name + '/expected-path', selection['path'], prefix + 'expected.bin')
        payload = data[selection['path']]
        if name.startswith('refuse-'):
            c.equal(name + '/no-fabricated-numeric-reference', observed['references'], {})
            c.equal(name + '/refusal-selection', selection, {'kind': 'full-response', 'origin': 'protocol-literal', 'path': prefix + 'expected.bin'})
            c.equal(name + '/exact-refusal-literal', payload, b'\x01\0\0\0')
            c.equal(name + '/actual-refusal', data[native['stdout']], payload)
        else:
            total += fixture['count']
            references = {'cpython': prefix + 'reference-cpython.bin'}
            if reference == 'pytorch' and fixture['count']:
                references['pytorch'] = prefix + 'reference-pytorch.bin'
            c.equal(name + '/actual-reference-set', observed['references'], references)
            origin = 'cpython' if reference == 'cpython' else ('pytorch' if fixture['count'] else 'empty-no-framework-operation')
            c.equal(name + '/selection-origin', selection, {'kind': 'payload', 'origin': origin, 'path': prefix + 'expected.bin'})
            for source, filename in references.items():
                c.equal(name + '/complete-' + source + '-reference', data[filename], payload)
            c.equal(name + '/full-expected-payload-sha', sha(payload), fixture['reference_sha256'])
            c.equal(name + '/full-expected-payload-size', len(payload), fixture['count'] * (8 if profile == 'i64' else 1))
            c.equal(name + '/full-actual-vs-recorded-reference', data[native['stdout']], bytes(4) + payload)
    c.equal('complete-value-count', total, spec['values'])
    provenance = report['oracle_provenance' if profile == 'i64' else 'provenance']
    lock = json.loads(data['evidence/sources/pytorch-oracle-lock.json'])
    c.equal('original-upstream-inventory', report['upstream_commit'], lock['inventory_commit'])
    if reference == 'cpython':
        c.equal('CPython-not-labelled-Torch', provenance, None)
    else:
        c.equal('real-Torch-provenance-status', provenance['status'], 'PASS')
        c.equal('real-Torch-exact-lock', provenance['lock_sha256'], LOCK)
        c.equal('real-Torch-exact-expected', provenance['expected'], lock['oracle'])
        c.equal('real-Torch-source-equivalence', provenance['release_source_equivalence'], lock['release_source_equivalence'])
        c.equal('all-eleven-provenance-names', set(provenance['checks']), PROVENANCE_NAMES)
        for name, value in provenance['checks'].items():
            c.check('real-Torch-provenance/' + name, value is True)
        c.equal('no-provenance-metadata-error', provenance['metadata_error'], None)
        verify_provenance_fields(provenance, lock, c)
        observed = provenance['observed']; actual_oracle = lock['oracle']
        for key in ('version', 'git_commit', 'backend_versions', 'platform'):
            c.equal('actual-Torch/' + key, observed[key], actual_oracle[key])
        c.equal('actual-Torch/Python-version', observed['python']['version'], actual_oracle['python']['version'])
        c.equal('actual-Torch/Python-implementation', observed['python']['implementation'], actual_oracle['python']['implementation'])
        pip = json.loads(data['evidence/sources/pytorch-install.json'])
        torch_entries = [r for r in pip['install'] if r['metadata']['name'].casefold() == 'torch']
        c.equal('full-original-pip-has-one-Torch', len(torch_entries), 1)
        installed = torch_entries[0]
        c.equal('full-original-pip-metadata-version', installed['metadata']['version'], actual_oracle['version'])
        c.check('full-original-pip-requested', installed['requested'] is True)
        c.equal('full-original-pip-wheel-url', installed['download_info']['url'], actual_oracle['wheel']['url'])
        c.equal('full-original-pip-wheel-sha', installed['download_info']['archive_info']['hashes']['sha256'], actual_oracle['wheel']['sha256'])
        recorded = observed['install_report']
        c.equal('full-pip-vs-embedded-download', installed['download_info'], recorded['download_info'])
        c.equal('full-pip-vs-embedded-version', installed['metadata']['version'], recorded['metadata_version'])
        c.equal('full-pip-vs-embedded-name', installed['metadata']['name'], recorded['metadata_name'])
        c.equal('full-pip-vs-embedded-pip-version', pip['pip_version'], recorded['pip_version'])
    for name, original in data.items():
        c.equal('source-byte-inventory-unchanged-after/' + name, (directory / name).read_bytes(), original)
    c.equal('manifest-unchanged-after', manifest_path.read_bytes(), manifest_raw)
    return {'profile': profile, 'reference': reference, 'cases': spec['cases'], 'compared_values': spec['values'],
            'elf_sha256': spec['elf'], 'compiler_sha256': COMPILER, 'source_sha256': spec['kernel'],
            'oracle_sha256': spec['oracle'], 'observer_sha256': OBSERVER,
            'original_report_sha256': sha(data['report.json']), 'manifest_sha256': sha(manifest_raw),
            'complete_files': len(expected_files) + 1, 'complete_file_bytes': sum(len(r) for r in data.values()) + len(manifest_raw),
            'direct_original_binary_bytes_verified': True, 'complete_native_request_response_bytes_verified': True,
            'actual_framework_expected_bytes_verified': reference == 'pytorch',
            'empty_Torch_arithmetic_claimed': False, 'full_rewrite_complete': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--profile', choices=('i64', 'i8'), required=True)
    p.add_argument('--reference', choices=('cpython', 'pytorch'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    directory = args.directory.resolve(); output = args.output.resolve()
    assert not output.exists() and directory not in output.parents
    c = Checks()
    report = {'schema': 'heptatorch.integer-byte-evidence-offline-review.v1', 'status': 'FAIL',
              'native_or_Torch_executions_by_reader': 0, 'full_rewrite_complete': False}
    try:
        report.update(verify(directory, args.profile, args.reference, c))
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = repr(error); report['traceback'] = traceback.format_exc()
    report['checks'] = c.rows
    report['checks_passed'] = sum(x['passed'] for x in c.rows)
    report['checks_failed'] = sum(not x['passed'] for x in c.rows)
    report['reader_sha256'] = sha(Path(__file__).read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('checks', 'traceback')}, sort_keys=True))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
