#!/usr/bin/env python3
"""Verify the fixed ROG evidence packet without executing native code or Torch."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import re
import tarfile

SOURCE = '5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49'
IMAGE = '69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6'
ARCHIVE = '6499e621172b98fefea2f49e8f0178c5d7047e55438aa61819dd25a3395292d7'
MANIFEST = 'c92af08e0e486c8fefeba2801ec2f7a6e6c6d7fb7e5d6b6013bfb53c9eefa952'
RECEIPT = '375318f85971d3e8754421e4cf53f3027d00e3e5c538d7d5d6f4caa4b31981d0'
SOURCE_MANIFEST = '347a9078f627a2844625f57a8f82aa06ee8c9cd47b37a3f8d44b934cf263eae1'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
 'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
 'direct_url_archive_sha256_matches', 'pip_report_url_matches',
 'pip_report_archive_sha256_matches', 'pip_report_version_matches', 'pip_report_requested'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--extract', type=Path)
    p.add_argument('--source-package', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    root = a.evidence.resolve()
    archive = root/'raw-evidence.tar.xz'
    assert archive.stat().st_size == 11375728 and digest(archive) == ARCHIVE
    raw_manifest = (root/'evidence-manifest.json').read_bytes()
    raw_receipt = (root/'evidence-receipt.json').read_bytes()
    assert sha(raw_manifest) == MANIFEST and sha(raw_receipt) == RECEIPT
    assert digest(root/'source-manifest.original.json') == SOURCE_MANIFEST
    manifest, receipt = json.loads(raw_manifest), json.loads(raw_receipt)
    assert manifest['native_source_sha256'] == SOURCE and manifest['native_executable_sha256'] == IMAGE
    assert receipt['status'] == 'COMPLETE_RAW_CUSTODY'
    assert receipt['archive']['sha256'] == ARCHIVE
    wanted = {e['path']: e for e in manifest['files']}
    assert len(wanted) == len(manifest['files']) == manifest['file_count'] == 271
    assert sum(e['bytes'] for e in wanted.values()) == manifest['expanded_bytes'] == 49233889
    dec = lzma.LZMADecompressor()
    tar_hash, tar_bytes = hashlib.sha256(), 0
    with archive.open('rb') as f:
        while not dec.eof:
            data = f.read(65536) if dec.needs_input else b''
            assert data or not dec.needs_input, 'truncated XZ'
            block = dec.decompress(data, max_length=1 << 20)
            tar_hash.update(block)
            tar_bytes += len(block)
        assert not dec.unused_data and f.read(1) == b'', 'trailing XZ stream'
    assert tar_hash.hexdigest() == receipt['uncompressed_tar_sha256']
    assert tar_bytes == receipt['uncompressed_tar_bytes']
    selected_names = {
        'bundle/bundle-manifest.json', 'bundle/native/build.json', 'bundle/native/runtime.hepta',
        'bundle/profile/tests/mixed_frozen35_accounting/profile.json',
        'run-live32-01/original.json', 'run-live32-01/transport-report.json',
        'run-frozen35-01/frozen35/original.json', 'run-frozen35-01/frozen35/accounted.json',
        'run-frozen35-01/copied18/report.json', 'run-frozen35-01/transport-report.json'}
    assert selected_names <= set(wanted)
    selected, seen = {}, set()
    destination = a.extract.resolve() if a.extract else None
    if destination:
        destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r|xz') as tf:
        for member in tf:
            name = member.name
            relative = Path(name)
            assert member.isfile() and name in wanted and name not in seen
            assert not relative.is_absolute() and '..' not in relative.parts and str(relative) == name
            e = wanted[name]
            assert member.size == e['bytes'] and member.mode == e['mode']
            h, count = hashlib.sha256(), 0
            blocks = [] if name in selected_names else None
            target = destination/relative if destination else None
            writer = None
            if target:
                target.parent.mkdir(parents=True, exist_ok=True)
                writer = target.open('xb')
            try:
                with tf.extractfile(member) as f:
                    while block := f.read(1 << 20):
                        h.update(block)
                        count += len(block)
                        if blocks is not None:
                            blocks.append(block)
                        if writer:
                            writer.write(block)
            finally:
                if writer:
                    writer.close()
            assert count == e['bytes'] and h.hexdigest() == e['sha256']
            if target:
                target.chmod(e['mode'])
                assert target.stat().st_mode & 0o777 == e['mode'] and digest(target) == e['sha256']
            if blocks is not None:
                selected[name] = b''.join(blocks)
            seen.add(name)
    assert seen == set(wanted)
    bundle = json.loads(selected['bundle/bundle-manifest.json'])
    assert sha(selected['bundle/bundle-manifest.json']) == 'f1b7be0a83e3309182422ab4ccd05192e36d2f3ea7500ccb7fb0f5e610eac46e'
    for e in bundle['files']:
        actual = wanted['bundle/'+e['path']]
        assert all(actual[k] == e[k] for k in ('bytes', 'sha256', 'mode'))
    build = json.loads(selected['bundle/native/build.json'])
    assert build['source_sha256'] == SOURCE and build['native_sha256'] == IMAGE
    assert build['link_exit'] == 0 and len(build['units']) == 11
    for u in build['units']:
        assert u['compiler_exit'] == 0
        assert wanted['bundle/native/'+u['name']+'.hepta']['sha256'] == u['source_sha256']
        assert wanted['bundle/native/'+u['name']+'.o']['sha256'] == u['object_sha256']
    profile_raw = selected['bundle/profile/tests/mixed_frozen35_accounting/profile.json']
    assert sha(profile_raw) == '1ac4bdf9bc28df618d2fa959acf2732916c196d10143a69a7e3967b37e4ecbea'
    profile = json.loads(profile_raw)
    functions = {name.decode(): sha(body) for body, name in re.findall(
        rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',
        selected['bundle/native/runtime.hepta'], re.M | re.S)}
    assert len(functions) == 556 and functions == profile['function_pins']
    live = json.loads(selected['run-live32-01/original.json'])
    original = json.loads(selected['run-frozen35-01/frozen35/original.json'])
    outer = json.loads(selected['run-frozen35-01/frozen35/accounted.json'])
    controls = json.loads(selected['run-frozen35-01/copied18/report.json'])
    for d in (live, original):
        assert d['status'] == 'PASS' and d['failed_cases'] == 0
        assert d['oracle_provenance']['status'] == 'PASS' and set(d['oracle_provenance']['checks']) == CHECKS
        assert all(v is True for v in d['oracle_provenance']['checks'].values())
        assert all(r['status'] == 'PASS' for r in d['rows'])
        assert d['native_executable_sha256'] == IMAGE
    assert live['native_source_sha256'] == SOURCE and live['cases'] == live['expected_cases'] == len(live['rows']) == 32
    assert (live['frames'], live['exact_binary32_words'], live['serialized_state_bytes']) == (2132, 17455, 1137600)
    assert original['cases'] == original['expected_cases'] == len(original['rows']) == 35
    assert (original['frames'], original['compared_values']) == (6344, 2590274)
    assert original['explicit_state_absolute_version_differences'] == 283
    assert original['direct_parameter_absolute_version_mismatches'] == 0
    assert outer['status'] == 'PASS' and outer['original_entry_return_code'] == 0 and outer['exception'] is None
    bound = outer['original_oracle_report']
    assert bound['sha256'] == sha(selected['run-frozen35-01/frozen35/original.json'])
    assert bound['bytes'] == len(selected['run-frozen35-01/frozen35/original.json'])
    assert bound['path'] == manifest['source_root']+'/run-frozen35-01/frozen35/original.json'
    assert outer['profile']['source_sha256'] == SOURCE and outer['profile']['native_sha256'] == IMAGE
    assert outer['profile']['name'] == 'mixed-integer-556-fixed35-F32-Saved'
    assert outer['all_original_EOF_and_post_close_Version_gates_executed'] is True
    assert outer['numeric_oracle_rows_reclassified_by_wrapper'] is False
    assert len(outer['observer_cases']) == 35
    for row, case in zip(original['rows'], outer['observer_cases']):
        o = row['private_node_observer']
        assert case == dict(case=row['case'], **o)
        assert o['status'] == 'PASS' and o['eof_exit_assertions_executed'] is True
        assert o['post_close_parameter_version_assertions_executed'] is True
    assert controls['status'] == 'PASS_OBSERVER_CONTROLS_ONLY' and controls['failures'] == 0
    assert controls['positive_cases'] == len(controls['positive_rows']) == 4
    assert controls['negative_cases'] == len(controls['negative_rows']) == 18
    for suite in ('live32', 'frozen35'):
        d = json.loads(selected['run-'+suite+'-01/transport-report.json'])
        assert d['status'] == 'PASS' and d['bundle_unchanged'] is True and d['original_inputs_unchanged'] is True
        assert all(child['return_code'] == 0 for child in d['children'])
    source_manifest = a.source_package / 'validation/mixed-frozen35-accounting/source-package-manifest.json'
    assert digest(source_manifest) == SOURCE_MANIFEST
    for e in json.loads(source_manifest.read_bytes())['files']:
        q = a.source_package / e['path']
        assert q.is_file() and not q.is_symlink()
        assert q.stat().st_size == e['bytes'] and q.stat().st_mode & 511 == e['mode']
        assert digest(q) == e['sha256']
    from recheck_transitions import recheck
    transitions = recheck(archive, manifest, original,
        a.source_package/'tests/mixed_frozen35_accounting/owner_accounting_mixed.py')
    report = {'status': 'PASS_EXACT_FIXED_REMOTE_EVIDENCE',
        'native_source_sha256': SOURCE, 'native_sha256': IMAGE,
        'archive_sha256': ARCHIVE, 'manifest_sha256': MANIFEST, 'receipt_sha256': RECEIPT,
        'source_manifest_sha256': SOURCE_MANIFEST, 'members': len(seen),
        'expanded_bytes': manifest['expanded_bytes'], 'single_XZ_EOF_verified': True,
        'all_members_bytes_sizes_modes_verified': True, 'native_units_verified': 11,
        'HIA_functions_verified': 556, 'actual_live32': 32, 'actual_frozen35': 35,
        'live32_frames': 2132, 'live32_exact_words': 17455, 'live32_RNG_bytes': 1137600,
        'frozen35_frames': 6344, 'frozen35_values': 2590274,
        'state_absolute_version_differences_preserved': 283,
        'direct_parameter_version_mismatches': 0, 'original_EOF_and_postclose_Version_gates_all_executed': True,
        'copied_positive': 4, 'copied_negative': 18, 'provenance_per_reference': 11,
        'original_reports': {name: wanted[name] for name in ('run-live32-01/original.json',
            'run-frozen35-01/frozen35/original.json', 'run-frozen35-01/frozen35/accounted.json',
            'run-frozen35-01/copied18/report.json')},
        'full_archived_transition_recomputation': transitions,
        'new_native_or_Torch_executed_by_this_reader': False,
        'whole_pytorch_gate': 'HOLD', 'pytorch_api_denominator': 2590,
        'full_pytorch_rewrite_complete': False}
    assert not a.report.exists()
    a.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'PASS_EXACT_FIXED_REMOTE_EVIDENCE', 'archive_sha256': ARCHIVE,
        'members': len(seen), 'native_units': 11, 'HIA_functions': 556,
        'live32': 32, 'fixed35': 35, 'copied_positive': 4, 'copied_negative': 18,
        'new_native_execution': False, 'new_Torch_execution': False,
        'whole_pytorch_gate': 'HOLD', 'pytorch_api_denominator': 2590,
        'full_pytorch_rewrite_complete': False}))


if __name__ == '__main__':
    main()
