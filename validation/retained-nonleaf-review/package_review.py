#!/usr/bin/env python3
"""Copy the executed independent review, verifying every frozen raw member."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--executed', type=Path, required=True)
    p.add_argument('--additive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    own = root / 'retained-nonleaf-native-review'
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    test = output / 'tests/retained_nonleaf_review'
    validation = output / 'validation/retained-nonleaf-review'
    reference = validation / 'reference'
    test.mkdir(parents=True)
    reference.mkdir(parents=True)
    copied = []
    def copy(source, target, expected=None):
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected:
            assert digest == expected, source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        assert source.read_bytes() == target.read_bytes() == raw
        copied.append({'source': str(source), 'destination': str(target.relative_to(output)), 'sha256': digest, 'bytes': len(raw)})
    codec = root / 'adam-distributions-gradient-independent-review/handoff-04/tests/adam_distributions_gradient_review'
    for filename, digest in (
        ('replay_retained_nonleaf.py', 'c925e33872dc87ead29ecc2125b1799763a87b13c8ae2eba85e966b4202f1d8e'),
        ('replay_optimizer_retained_admission.py', 'afd73c315b0083c5dc42fddf43bea79708d084da381412b48f55fce4f6facf72')):
        copy(own / filename, test / filename, digest)
    copy(codec / 'protocol.py', test / 'protocol.py', '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9')
    copy(codec / 'owner_client.py', test / 'owner_client.py', 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b')
    copy(own / 'freeze_actual.py', test / 'freeze_actual.py')
    copy(own / 'package_review.py', validation / 'package_review.py')
    copy(own / 'README.md', output / 'README.md')
    copy(own / 'SEMANTICS.md', validation / 'SEMANTICS.md')
    for filename, target, digest in (
        ('torch-retained-nonleaf.json', 'torch-retained-nonleaf.json', 'ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3'),
        ('pytorch_retained_nonleaf_probe.py', 'pytorch_retained_nonleaf_probe.py', '7cdb53f7a95c48acd1b85d93d90c693d4caa97ada72a54a4021bb616d098fcba'),
        ('semantic-observations.json', 'semantic-observations.json', None),
        ('manifest.json', 'actual-custody-manifest.json', '73147d71b61120ba58621b4716f8ebb9d6e16bc7c0fe27a7d1aef8a62e8fdb52')):
        copy(own / 'actual-01' / filename, reference / target, digest)
    old = root / 'optimizer-object-review'
    for filename, target, digest in (
        ('hosted-optimizer-object-observations.json', 'original-optimizer-observations.json', '7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160'),
        ('pytorch_optimizer_object_probe.py', 'pytorch_optimizer_object_probe.py', 'f69340170dc2e55886bca73a8d37265ba2c95a8c793c7633c7e8a22234cfa386'),
        ('replay_optimizer_objects_v2.py', 'replay_optimizer_objects_v2.py', 'ea748d4100b539e0144f032ca301b4886d0f9520c36676c980279e0bbc9bc034')):
        copy(old / filename, reference / target, digest)
    copy(own / 'prepared-01/report.json', validation / 'prepared-01.json')
    copy(a.executed / 'report.json', validation / 'actual12-native-01.json', '6540097fd11f9f1155813309e1237101567a4555636af4678a2c2858a3156e08')
    copy(a.additive / 'report.json', validation / 'original-optimizer-additive-01.json', 'c97959194c8628a4abb6925fb6a136cef123e58da79d8eee054710e6df0fd42f')
    report = json.loads((a.executed / 'report.json').read_bytes())
    assert report['status'] == 'QUALIFIED_EXPOSED_SCOPE' and report['variants_qualified'] == 12
    assert report['unexpected_failures'] == 0 and report['all_inputs_unchanged'] is True
    assert all(row['responses_before_stdin_eof'] is True and row['exit_code'] == 0 for row in report['rows'])
    assert report['binary_sha256'] == '7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5'
    assert report['source_sha256'] == '666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b'
    inputs = []
    for prefix, directory in (('actual12', a.executed), ('original-optimizer-additive', a.additive)):
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                inputs.append((prefix + '/' + str(path.relative_to(directory)), path))
    members = [{'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in inputs]
    archive = validation / 'native-01-raw.tar.gz'
    with archive.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for name, path in inputs:
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                    with path.open('rb') as stream:
                        tar.addfile(info, stream)
    with tarfile.open(archive, 'r:gz') as tar:
        assert [x.name for x in tar.getmembers()] == [m['path'] for m in members]
        for member, (_, original) in zip(members, inputs, strict=True):
            stream = tar.extractfile(member['path'])
            with original.open('rb') as source:
                hashed = hashlib.sha256()
                count = 0
                while True:
                    chunk, expected = stream.read(1 << 20), source.read(1 << 20)
                    assert chunk == expected, member['path']
                    if not chunk:
                        break
                    hashed.update(chunk)
                    count += len(chunk)
                assert count == member['bytes'] and hashed.hexdigest() == member['sha256']
    emit(validation / 'native-01-raw-members.json', {
        'archive': archive.name, 'sha256': sha(archive), 'bytes': archive.stat().st_size,
        'all_members_byte_compared_with_executed_original': True, 'members': members})
    qualification = {
        'schema': 'heptatorch.retained-nonleaf-independent-qualification.v1',
        'status': 'QUALIFIED_EXPOSED_SCOPE',
        'source_sha256': report['source_sha256'], 'binary_sha256': report['binary_sha256'],
        'actual_reference_sha256': report['inputs']['observations'],
        'actual12_report_sha256': sha(validation / 'actual12-native-01.json'),
        'actual12': {'families': 8, 'variants': 12, 'unexpected_failures': 0, **report['totals']},
        'original_optimizer_retained_subcase': {
            'status': 'ORIGINAL_RETAINED_NONLEAF_SUBOBSERVATION_QUALIFIED', 'frames': 15,
            'reference_outcomes_matched': 2,
            'report_sha256': sha(validation / 'original-optimizer-additive-01.json'),
            'original_24_replay_and_reports_unchanged': True,
            'other_original_24_observations_reexecuted_here': False},
        'archive_members': len(members), 'archive_member_bytes': sum(m['bytes'] for m in members),
        'archive_members_exactly_verified': True,
        'full_rewrite_complete': False, 'pytorch_api_denominator': 2590,
        'not_claimed': report['diagnostic_only'] + ['producer source review', 'independent OOM/resource/typed-owner gates',
            'higher-order or custom-hook behavior', 'all PyTorch operators or optimizer modes'],
    }
    emit(validation / 'qualification.json', qualification)
    emit(validation / 'copy-custody.json', {'files': copied, 'all_copied_source_bytes_unchanged': True})
    for row in copied:
        assert sha(row['source']) == row['sha256']
    files = [{'path': str(path.relative_to(output)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(output.rglob('*')) if path.is_file()]
    emit(output / 'manifest.json', {'schema': 'heptatorch.retained-nonleaf-independent-handoff.v1', 'files': files})
    print(json.dumps({'manifest_sha256': sha(output / 'manifest.json'), 'files': len(files),
                      'bytes': sum(x['bytes'] for x in files), 'raw_members': len(members)}))


if __name__ == '__main__':
    main()
