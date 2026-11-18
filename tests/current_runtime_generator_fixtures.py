"""Restore complete fixed Generator sources without executing a runtime."""
import ast
import json
from pathlib import Path
import stat

from current_runtime_fixtures import read, require, restore, safe_name, sha

ORIGINAL26_ARCHIVE = '30249ad9a07bdf4aa63051199476e82167ad56feecac988137fb381556f48644'
ORIGINAL26_MANIFEST = '0da4084676c8b864a7a0c6cffc1c28d589b3a06efb9550d4f3ff0899226f9ef5'
ORIGINAL26_PIN = 'd5d529a697056625a19c06abf4ddc25326dad8854ae87b0406ec86f5a66fd085'
CANDIDATE_ARCHIVE = '9fcbeccffeaa4aad12e5662cb35739baf94f880ba54944ceea62b052e1ba47c2'
CANDIDATE_MANIFEST = 'b1b1c5d41d7cda5976fdeb0d534de738b1177de522f76f7751bffc8b6598cf1e'
CANDIDATE_PACKAGE = '8ac67a00bddbd3d709b4191e154a671aa302ddc4ae1a7347e4aa39443a571f82'
QUALIFIER_ARCHIVE = '57e9cc80121d922efaeb046c28a7c37b48f22d07144be0387abb7d9c77b6cd88'
QUALIFIER_MANIFEST = '35648c1cc57fb7eaf152441f12a8ef8f2551a7b216bcd0d829c9503051fc098e'
SOURCE04_MANIFEST = '8b0a5189f9146af5d9280266e71b76cfd75b5d7a584b6063e3d615fbf8acd484'
SOURCE03_MANIFEST = '7131330d802ffc8fa7cae681bbc656ae06eff8b8845723b933b4544013b5b91e'
CONTROLS03 = 'b6d306bcaf2d59ab93ef547d90066a6dbcbdf7019ddfa6c100771278292fed8c'


def matches(path, pin, inputs=None):
    raw = read(path, pin['sha256'], inputs)
    require(len(raw) == pin['bytes'] and stat.S_IMODE(path.stat().st_mode) == pin['mode'],
            'complete Generator source bytes/mode: ' + str(path))
    return raw


def restore_original26(repository, destination, inputs=None):
    destination = restore(repository, destination, 'validation/current-generator-original26',
                          ORIGINAL26_ARCHIVE, ORIGINAL26_MANIFEST,
                          'heptatorch.current-runtime.original-generator26.v1', inputs)
    raw = read(destination / 'validation/cpu-generator-package.sha256', ORIGINAL26_PIN, inputs)
    names = set()
    for line in raw.decode('ascii').splitlines():
        pin, name = line.split('  ', 1)
        safe_name(name)
        require(name not in names, 'duplicate original26 member')
        names.add(name)
        read(destination / name, pin, inputs)
    require(len(names) == 26, 'all original26 members are mandatory')
    return destination


def restore_candidate(repository, destination, inputs=None):
    repository = Path(repository)
    destination = restore(repository, destination, 'validation/current-generator-candidate',
                          CANDIDATE_ARCHIVE, CANDIDATE_MANIFEST,
                          'heptatorch.current-runtime.generator712-candidate-metadata.v1', inputs)
    metadata = json.loads(read(repository / 'validation/current-generator-candidate/manifest.json',
                               CANDIDATE_MANIFEST, inputs))
    original = json.loads(read(destination / 'package-manifest.json', CANDIDATE_PACKAGE, inputs))
    require(original['schema'] == 'heptatorch.generator-native712.source-package.v1'
            and original['payload_files'] == len(original['files']) == 31
            and original['payload_bytes'] == 4997255, 'complete original candidate31 package')
    pins = {row['path']: {k: row[k] for k in ('bytes', 'sha256', 'mode')}
            for row in original['files']}
    require(len(pins) == 31 and len(metadata['canonical_mapping']) == 22,
            'unique candidate31 and current22 source mappings')
    from current_runtime_generator_binding import parent_source_files
    reconstructed, source_inverse = parent_source_files(repository, inputs)
    require(set(reconstructed) == {n for n in metadata['canonical_mapping'] if n.startswith('source/')},
            'only labelled original712 source members reconstructed')
    for name, current in metadata['canonical_mapping'].items():
        safe_name(name); safe_name(current)
        require(name in pins and not (destination / name).exists(), 'unique canonical candidate member')
        raw = reconstructed[name] if name in reconstructed else matches(repository / current, pins[name], inputs)
        require(len(raw) == pins[name]['bytes'] and sha(raw) == pins[name]['sha256'],
                'complete original candidate member identity')
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        path.chmod(pins[name]['mode'])
    actual = {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}
    require(actual == set(pins) | {'package-manifest.json'}, 'exact reconstructed candidate inventory')
    for name, pin in pins.items():
        safe_name(name)
        matches(destination / name, pin, inputs)
    return destination


def verify_source04_qualifier(package, inputs=None):
    package = Path(package)
    manifest = json.loads(read(package / 'package-manifest.json', SOURCE04_MANIFEST, inputs))
    actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'package-manifest.json'}, 'complete portable qualifier inventory')
    for name, pin in manifest['files'].items():
        safe_name(name)
        matches(package / name, pin, inputs)
    parent = json.loads(read(package / 'parent-source03/package-manifest.json', SOURCE03_MANIFEST, inputs))
    proof = json.loads(read(package / 'physical-root-adaptation.json',
                            manifest['physical_root_adaptation_sha256'], inputs))
    require(proof['schema'] == 'heptatorch.generator712-qualification.physical-root-inverse.v1'
            and proof['original_source03_manifest_sha256'] == SOURCE03_MANIFEST
            and proof['original_controls_sha256'] == CONTROLS03
            and set(proof['changes']) == {'run.py', 'admission.py'}, 'exact physical-root adaptation')
    for name, pin in parent['files'].items():
        old_path = package / ('parent-source03/' + name if name in proof['changes'] else name)
        old = matches(old_path, pin, inputs)
        if name not in proof['changes']:
            continue
        change = proof['changes'][name]
        current = matches(package / name, change['derived'], inputs)
        require(change['original'] == pin and change['whole_file_inverse_exact'] is True,
                'original source03 file identity')
        inverse = current.decode('utf-8')
        for edit in reversed(change['edits']):
            require(edit['count'] == 1 and inverse.count(edit['after']) == 1,
                    'one exact physical-root edit')
            inverse = inverse.replace(edit['after'], edit['before'])
        require(inverse.encode() == old, 'whole source03 inverse: ' + name)
        def bodies(raw):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        before, after = bodies(old), bodies(current)
        require(before.keys() == after.keys()
                and {n for n in before if before[n] != after[n]} == ({'main'} if name == 'run.py' else {'admit'}),
                'only command-line/physical-custody function changes')
    require(read(package / 'controls.py', CONTROLS03, inputs)
            and all(proof[k] is True for k in ('all_original_comparisons_and_controls_unchanged',
                'original26_manifest_and_member_pins_unchanged', 'original24_actual_repository_provenance_unchanged'))
            and proof['native_executed'] is False and proof['Torch_executed'] is False,
            'preserved source03 controls and no qualification transfer')
    return proof


def verify_qualifier(package, inputs=None):
    """Read every derived file and reverse only the explicit729 profile edits."""
    package = Path(package)
    manifest = json.loads(read(package / 'package-manifest.json', QUALIFIER_MANIFEST, inputs))
    actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'package-manifest.json'},
            'complete physical729 portable qualifier inventory')
    for name, pin in manifest['files'].items():
        safe_name(name)
        matches(package / name, pin, inputs)
    previous = verify_source04_qualifier(package / 'parent-source04', inputs)
    parent = json.loads(read(package / 'parent-source04/package-manifest.json', SOURCE04_MANIFEST, inputs))
    proof = json.loads(read(package / 'profile-adaptation.json', manifest['profile_adaptation_sha256'], inputs))
    require(proof['schema'] == 'heptatorch.generator131.physical729-source-profile-inverse.v1'
            and proof['parent_source04_manifest_sha256'] == SOURCE04_MANIFEST
            and proof['source_sha256'] == '04ae430e3e851831fcb97a3fab700b16b6318361d52e90fcea90d2183d1d564b'
            and proof['native_sha256'] == '4749413ea952bfd9d1fe41770f94ce1b9b9e3efabb407aa90f725cf33d59c4db'
            and proof['current_functions'] == 729 and proof['current_units'] == 14
            and set(proof['changes']) == {'run.py', 'admission.py', 'processes.py'},
            'exact729 source/profile adaptation inventory')
    for name, original_pin in parent['files'].items():
        original = matches(package / 'parent-source04' / name, original_pin, inputs)
        current = read(package / name, manifest['files'][name]['sha256'], inputs)
        if name not in proof['changes']:
            require(current == original, 'unchanged complete original qualifier file: ' + name)
            continue
        change = proof['changes'][name]
        require(change['original'] == {k: original_pin[k] for k in ('bytes', 'sha256')}
                and len(current) == change['derived']['bytes']
                and sha(current) == change['derived']['sha256']
                and change['whole_file_inverse_exact'] is True,
                'explicit whole original/derived file identities')
        reverse = current.decode()
        for edit in reversed(change['edits']):
            require(reverse.count(edit['after']) == edit['count'], 'exact729 identity inverse count')
            reverse = reverse.replace(edit['after'], edit['before'])
        require(reverse.encode() == original, 'complete original source04 inverse: ' + name)
    require(all(proof[k] is True for k in ('original24_11_59_case_sources_byte_identical',
                'original37_controls_byte_identical',
                'actual_current_objects_required_for_new_probe_and_reordered_images'))
            and proof['native_Torch_compiler_executed'] is False
            and proof['old_runtime_PASS_transferred'] is False,
            'original cases and separate actual probe execution remain mandatory')
    return {'source04_to03': previous, 'physical729_to_source04': proof}


def restore_qualifier(repository, destination, inputs=None):
    destination = restore(repository, destination, 'validation/current-generator131-source',
                          QUALIFIER_ARCHIVE, QUALIFIER_MANIFEST,
                          'heptatorch.generator729-original-qualification-source.v1', inputs,
                          embedded_manifest=True)
    verify_qualifier(destination, inputs)
    return destination
