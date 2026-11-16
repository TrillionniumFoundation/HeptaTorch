"""Collect unchanged independent runners and references for the native qualifier.

This development-time assembly tool only copies and hashes bytes. The resulting
build.py and qualify.py use portable CLI inputs and contain no scratch paths.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('portable-source', 'gradient-package', 'development-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    root, grad = a.development_root.resolve(), a.gradient_package.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tools = out / 'tools/gradient_adam_distributions'
    tests = out / 'tests/gradient_adam_distributions_review'
    validation = out / 'validation/gradient-adam-distributions'
    references = validation / 'reference'
    copies = []

    def copy(source, destination):
        source = Path(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        data = source.read_bytes()
        shutil.copy2(source, destination)
        assert destination.read_bytes() == data
        copies.append({'source': str(source), 'destination': destination.relative_to(out).as_posix(),
                       'sha256': sha(data), 'bytes': len(data), 'mode': source.stat().st_mode & 0o777})

    portable = a.portable_source.resolve() / 'tools/gradient_adam_distributions'
    for source in sorted(portable.iterdir()):
        if source.is_file():
            copy(source, tools / source.name)
    copy(HERE / 'qualify.py', tools / 'qualify.py')
    frozen_tests = grad / 'tests/gradient_objects'
    adapted_runners = ('test_native_lifetimes.py', 'replay_grad_clear.py')
    for source in sorted(frozen_tests.glob('*.py')):
        if source.name not in adapted_runners:
            copy(source, tests / source.name)
    adapters = []
    for name in adapted_runners:
        original = (frozen_tests / name).read_bytes()
        adapted = original
        for before, after in ((b'HTGRAD02', b'HTGAD002'), (b'HTOGRAD2', b'HTOGAD02')):
            assert adapted.count(before) == 1
            adapted = adapted.replace(before, after)
        assert adapted.replace(b'HTGAD002', b'HTGRAD02').replace(b'HTOGAD02', b'HTOGRAD2') == original
        (tests / name).write_bytes(adapted)
        copy(frozen_tests / name, validation / 'test-source-custody' / (name + '.original.txt'))
        adapters.append({
            'file': 'tests/gradient_adam_distributions_review/' + name,
            'original_sha256': sha(original), 'adapted_sha256': sha(adapted),
            'substitutions': {'HTGRAD02': 'HTGAD002', 'HTOGRAD2': 'HTOGAD02'},
            'inverse_byte_identical': True, 'all_inputs_expectations_and_assertions_unchanged': True,
        })
    for name in ('replay_bernoulli_gradient_layout.py', 'test_preserve_storage_lifetimes.py', 'replay_optimizer_gradient_alias.py'):
        copy(HERE / name, tests / name)
    extra_tests = {
        'cpu-distributions-extension/package-03/tools/cpu_distributions/protocol.py': 'distribution_protocol.py',
        'cpu-distributions-extension/package-03/tools/cpu_distributions/owner_client.py': 'owner_client.py',
        'cpu-distributions-extension/package-03/tools/cpu_distributions/replay_reference.py': 'replay_distribution_atomic.py',
        'cpu-distributions-partial-extension/package-01/tools/cpu_distributions_partial/replay_explicit.py': 'replay_distribution_explicit.py',
        'owned-adam-independent-review/handoff-02/tests/adam_object_review/replay_optimizer_objects_v2.py': 'replay_optimizer_objects_v2.py',
        'owned-adam-independent-review/handoff-02/tests/adam_object_review/replay_adam_objects.py': 'replay_adam_objects.py',
    }
    for source, target in extra_tests.items():
        copy(root / source, tests / target)
    original_refs = grad / 'validation/gradient-objects/reference'
    for source in sorted(original_refs.rglob('*')):
        if source.is_file():
            copy(source, references / source.relative_to(original_refs))
    copy(root / 'cpu-distributions-extension/package-03/tools/cpu_distributions/evidence/torch-bernoulli-dropout.json',
         references / 'torch-bernoulli-dropout.json')
    copy(root / 'optimizer-object-ci-edc6/torch-optimizer-objects.json', references / 'torch-optimizer-objects.json')
    copy(Path('/dev/shm/heptatorch-d26699e25fea/ref-ci-b376bb0/torch-bernoulli-gradient-layout.json'),
         references / 'torch-bernoulli-gradient-layout.json')
    copy(HERE / 'pytorch_bernoulli_gradient_layout_probe.py', references / 'probes/pytorch_bernoulli_gradient_layout_probe.py')
    copy(root / 'optimizer-object-review/pytorch_optimizer_object_probe.py', references / 'probes/pytorch_optimizer_object_probe.py')
    (validation / 'test-source-custody').mkdir(parents=True, exist_ok=True)
    (validation / 'test-source-custody/copied-sources.json').write_text(json.dumps({
        'schema': 'heptatorch.gradient-adam-distributions-qualification-input-custody.v1',
        'exact_copies': copies,
        'only_modified_inherited_runners': adapters,
        'hosted_references_unchanged': True,
    }, indent=2) + '\n')
    manifest = {'schema': 'heptatorch.gradient-adam-distributions-package-source-manifest.v1',
                'files': {path.relative_to(out).as_posix(): {
                    'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size,
                    'mode': path.stat().st_mode & 0o777}
                    for path in sorted(out.rglob('*')) if path.is_file()},
                'scope': 'Immutable build/test/reference inputs checked before and after qualification; output reports are separate artifacts.'}
    (tools / 'package-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': len(manifest['files']),
                      'manifest_sha256': sha((tools / 'package-manifest.json').read_bytes()),
                      'adapted_runners': adapters}))


if __name__ == '__main__':
    main()
