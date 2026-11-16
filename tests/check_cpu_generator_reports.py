#!/usr/bin/env python3
"""Bind the complete native Generator qualification to exact frozen sources."""
from pathlib import Path
import argparse
import hashlib
import json

IMAGE = 'ba344748d42f0def1cd5d88e5e27232a6dfa08e33ef4c42e6f00486f3ca08572'
LINKER = '761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
MANIFEST = 'd5d529a697056625a19c06abf4ddc25326dad8854ae87b0406ec86f5a66fd085'
ORACLE = '3b9e1c9058776c67c00c4b07610a2a0d377dda16ca4a956d7e66c442bdfd4157'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / 'validation/cpu-generator-package.sha256'
    assert sha(manifest) == MANIFEST
    custody = {}
    for line in manifest.read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert relative not in custody and sha(repo / relative) == expected, relative
        custody[relative] = expected
    assert len(custody) == 26
    assert sha(repo / 'tests/pytorch_cpu_generator_oracle.py') == ORACLE
    paths = {'build': args.build / 'build.json', 'rebuild': args.build / 'rebuild.json',
             'native': args.review / 'native/report.json', 'entry': args.review / 'entry/report.json'}
    reports = {name: json.loads(path.read_bytes()) for name, path in paths.items()}
    assert all(value['status'] == 'PASS' for value in reports.values())
    build, rebuild, native, entry = (reports[name] for name in ('build', 'rebuild', 'native', 'entry'))
    assert sha(args.build / 'generator-session') == build['executable_sha256'] == IMAGE
    assert build['generator_linker_sha256'] == rebuild['generator_linker_sha256'] == LINKER
    assert build['compiler_sha256'] == entry['compiler_sha256'] == COMPILER
    assert build['rng_all_17_nonentry_bodies_unchanged'] is True
    assert build['host_production_rng_or_owner_runtime'] is False
    assert build['frozen_profiles_modified'] is False
    assert rebuild['native_compile_and_link'] is True and rebuild['executable_sha256'] == IMAGE
    assert rebuild['build_receipt_sha256'] == native['build_receipt_sha256'] == sha(paths['build'])
    assert native['image_sha256'] == entry['production_image_sha256'] == IMAGE
    assert entry['linker_sha256'] == LINKER
    assert len(native['cases']) == 11 and len(entry['cases']) == entry['case_count'] == 59
    assert all(row['status'] == 'PASS' for row in native['cases'] + entry['cases'])
    assert len({row['name'] for row in native['cases']}) == 11
    assert len({row['case'] for row in entry['cases']}) == 59
    for field, expected in [('frames', 1715), ('exact_output_values', 121611),
                            ('full_rng_state_checks', 471), ('full_pool_failure_rollbacks', 72)]:
        assert native[field] == expected, (field, native[field], expected)
    for relative, expected in rebuild['package_source_sha256'].items():
        assert sha(repo / 'tools/cpu_generators' / relative) == expected
    for relative, expected in native['test_sources_sha256'].items():
        assert sha(repo / 'tools/cpu_generators' / relative) == expected
    assert entry['test_source_sha256'] == sha(repo / 'tools/cpu_generators/test_entry.py')
    assert entry['caller_created_owners_in_qualification_only'] is True
    assert entry['production_accepts_native_pointers_from_packets'] is False
    report = {'schema': 'heptatorch.cpu-generator-integrated-native-qualification.v1',
              'status': 'PASS', 'manifest_sha256': MANIFEST, 'package_custody': custody,
              'oracle_source_sha256': ORACLE, 'native_image_sha256': IMAGE,
              'report_sha256': {name: sha(path) for name, path in paths.items()},
              'reports': reports, 'real_pytorch_run': False,
              'internal_state_boundary': 'Owned state comes from this service; internal diagnostics do not admit arbitrary foreign owner bytes.',
              'full_pytorch_generator_complete': False, 'whole_pytorch_rewrite_complete': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'native_cases': 11, 'entry_cases': 59,
                      'frames': 1715, 'report_sha256': sha(args.output)}, indent=2))


if __name__ == '__main__':
    main()
