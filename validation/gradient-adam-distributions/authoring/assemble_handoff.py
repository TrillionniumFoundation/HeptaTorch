"""Freeze source-bound native evidence; this tool does no target computation."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'development-root', 'qualification', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root, package, qual = (args.development_root.resolve(), args.package.resolve(),
                           args.qualification.resolve())
    author = root / 'gradient-adam-distributions-extension'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    copied = []

    def copy(source, target):
        source, target = Path(source), out / target
        assert source.is_file() and not source.is_symlink(), source
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), target
        raw = source.read_bytes()
        shutil.copy2(source, target)
        assert target.read_bytes() == raw
        copied.append({'source_development_reference': str(source),
                       'path': target.relative_to(out).as_posix(),
                       'bytes': len(raw), 'sha256': sha(raw),
                       'mode': source.stat().st_mode & 0o777})

    for source in sorted(package.rglob('*')):
        if source.is_file():
            copy(source, source.relative_to(package))
    tool = Path('tools/gradient_adam_distributions')
    validation = Path('validation/gradient-adam-distributions')
    for name in ('README.md', 'PROTOCOL.md'):
        copy(author / name, tool / name)
    for name in ('build.py', 'zero_layout.py', 'preserve_cache.py', 'pointwise_layout.py',
                 'scalar_add_bits.py', 'binary-original.py', 'scalar_add_preserving_bits.qualified.hepta',
                 'scalar_add_nograd.qualified.hepta', 'assemble_portable_sources.py',
                 'assemble_qualification_package.py', 'portable_build.py', 'qualify.py',
                 'assemble_handoff.py'):
        copy(author / name, validation / 'authoring' / name)
    for name in ('three-way-functions.json', 'binary-author-source-custody.json',
                 'scalar-helper-custody.json', 'signed-byte-stride-delta-05.json',
                 'signed-byte-stride-05.patch'):
        copy(author / name, validation / 'source-custody' / name)

    top = json.loads((qual / 'report.json').read_bytes())
    assert top['status'] == 'PASS' and top['step_count'] == 17
    assert top['source_sha256'] == '1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae'
    assert top['native_sha256'] == '6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe'
    copy(qual / 'report.json', validation / 'reports/candidate-05/portable-qualification.json')
    suites = {}
    for name, item in top['suite_reports'].items():
        raw = (qual / item['path']).read_bytes()
        assert sha(raw) == item['sha256']
        target = validation / 'reports/candidate-05' / (name + '.json')
        copy(qual / item['path'], target)
        suites[name] = {'path': target.as_posix(), 'sha256': sha(raw),
                        'status': json.loads(raw)['status']}
    copy(qual / 'distribution-build-field-bridge.json',
         validation / 'source-custody/distribution-build-field-bridge-05.json')
    source_manifest = json.loads((package / tool / 'package-manifest.json').read_bytes())
    for path, record in source_manifest['files'].items():
        data = (out / path).read_bytes()
        assert sha(data) == record['sha256'] and len(data) == record['bytes']

    archives = []
    for source in sorted(author.glob('*.tar.gz')):
        manifest = source.with_suffix(source.suffix + '.manifest.json')
        receipt = json.loads(manifest.read_bytes())
        assert receipt['status'] == 'PASS'
        assert sha(source.read_bytes()) == receipt['archive_sha256']
        assert source.stat().st_size == receipt['archive_bytes']
        target = validation / 'evidence' / source.name
        copy(source, target)
        copy(manifest, validation / 'evidence' / manifest.name)
        archives.append({'path': target.as_posix(), 'sha256': receipt['archive_sha256'],
                         'bytes': receipt['archive_bytes'],
                         'members_verified': receipt['members_verified'],
                         'uncompressed_bytes': receipt['uncompressed_bytes']})
    mandatory = {'candidate-01.tar.gz', 'candidate-02.tar.gz', 'candidate-03.tar.gz',
                 'candidate-04.tar.gz', 'candidate-05.tar.gz', 'portable-qualification-first-04.tar.gz',
                 'gradient-clear-tagfix-04.tar.gz', 'portable-qualification-05.tar.gz',
                 'portable-source-host-refusal-04.tar.gz', 'business-regressions-04.tar.gz'}
    assert mandatory <= {Path(item['path']).name for item in archives}

    peer_info = []
    for directory, manifest_sha in (
        ('adam-distributions-gradient-review/handoff-05',
         '455a64492cfaf7a75b24724542e3b5fb81a0e9c92b06dccb27211ed64a12a6aa'),
        ('adam-distributions-gradient-independent-review/handoff-04',
         'ae10fbd928c28ff626b497b9b18697aa75682004e1d196442ebef0c5dfecb0d2'),
        ('adam-distributions-gradient-independent-review/handoff-05-additive',
         'a483343a288d68eb0d17e545c0ff489376c3c214d58b3b6164665626bf50f944'),
    ):
        source = root / directory / 'manifest.json'
        assert sha(source.read_bytes()) == manifest_sha
        name = Path(directory).parent.name + '-' + Path(directory).name + '-manifest.json'
        copy(source, validation / 'independent-source-custody' / name)
        peer_info.append({'development_handoff_reference': directory,
                          'manifest_sha256': manifest_sha,
                          'copied_manifest': (validation / 'independent-source-custody' / name).as_posix(),
                          'separate_package_required': True})

    summary = {
        'schema': 'heptatorch.gradient-adam-distributions-handoff-qualification.v1',
        'status': 'QUALIFIED_EXPOSED_SCOPE',
        'native_source_sha256': top['source_sha256'], 'native_executable_sha256': top['native_sha256'],
        'native_functions': 428, 'native_blocks': 5738, 'native_instructions': 21348,
        'native_modules': 9, 'descriptor_bytes': 576,
        'state_format': 'HTGAD002', 'backing_format': 'HTOGAD02',
        'cold_portable_qualification_steps': 17,
        'cold_portable_report': (validation / 'reports/candidate-05/portable-qualification.json').as_posix(),
        'cold_portable_report_sha256': sha((qual / 'report.json').read_bytes()),
        'input_manifest_sha256': sha((package / tool / 'package-manifest.json').read_bytes()),
        'suites': suites, 'evidence_archives': archives, 'independent_review_packages': peer_info,
        'optimizer_reference_union': {'original_reference_cases': 24, 'qualified': 23, 'partial': 1,
                                      'case20_is_an_existing_case': True},
        'counting': 'Separate overlapping suites; no sum is a count of unique upstream tests.',
        'reference_execution': 'Immutable hosted observations with fixed provenance; replay here executes native HIA only.',
        'all_original_expected_values_and_tolerances_unchanged': True,
        'legacy_8_state_differences': {'Version': 8, 'creation': 65},
        'legacy_10_state_differences': {'Version': 11, 'creation': 9, 'missing_step_Tensor': 30,
                                       'Adam_words_different_within_original_tolerance': 6},
        'prior_failure_qualification_relabelled': False,
        'new_live_Torch_35_run': 'NOT_RUN_BY_THIS_PACKAGE',
        'full_rewrite_complete': False, 'pytorch_api_denominator': 2590,
    }
    (out / validation / 'qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / validation / 'source-custody/handoff-copied-files.json').write_text(
        json.dumps({'schema': 'heptatorch.byte-preserving-handoff-copies.v1', 'files': copied,
                    'development_paths_are_not_required_for_portable_execution': True}, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'copied_files': len(copied),
                      'qualification_sha256': sha((out / validation / 'qualification.json').read_bytes()),
                      'status': 'ASSEMBLED_NOT_YET_MANIFEST_FROZEN'}))


if __name__ == '__main__':
    main()
