"""Restore complete fixed Generator sources without executing a runtime."""
import ast
import json
from pathlib import Path
import re
import stat

from current_runtime_fixtures import read, require, restore, safe_name, sha

ORIGINAL26_ARCHIVE = '30249ad9a07bdf4aa63051199476e82167ad56feecac988137fb381556f48644'
ORIGINAL26_MANIFEST = '0da4084676c8b864a7a0c6cffc1c28d589b3a06efb9550d4f3ff0899226f9ef5'
ORIGINAL26_PIN = 'd5d529a697056625a19c06abf4ddc25326dad8854ae87b0406ec86f5a66fd085'
CANDIDATE_ARCHIVE = '9fcbeccffeaa4aad12e5662cb35739baf94f880ba54944ceea62b052e1ba47c2'
CANDIDATE_MANIFEST = 'b1b1c5d41d7cda5976fdeb0d534de738b1177de522f76f7751bffc8b6598cf1e'
CANDIDATE_PACKAGE = '8ac67a00bddbd3d709b4191e154a671aa302ddc4ae1a7347e4aa39443a571f82'
QUALIFIER_ARCHIVE = '08a61e25450142adcdcd9741ea298d2bcbfd9063fa195c2d824574217e394e88'
QUALIFIER_MANIFEST = 'e0a9f338adfafe2aac4e0fafa9786616640d5a5d24f3c187eba333af3c7ed70f'
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


def verify_qualifier740(package, inputs=None):
    """Read every derived file and reverse only the explicit736 profile edits."""
    package = Path(package)
    manifest = json.loads(read(package / 'package-manifest.json', QUALIFIER_MANIFEST, inputs))
    actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'package-manifest.json'},
            'complete physical740 portable qualifier inventory')
    for name, pin in manifest['files'].items():
        safe_name(name)
        matches(package / name, pin, inputs)
    previous = verify_source04_qualifier(package / 'parent-source04', inputs)
    parent = json.loads(read(package / 'parent-source04/package-manifest.json', SOURCE04_MANIFEST, inputs))
    proof = json.loads(read(package / 'profile-adaptation.json', manifest['profile_adaptation_sha256'], inputs))
    require(proof['schema'] == 'heptatorch.generator131.physical740-source-profile-inverse.v1'
            and proof['parent_source04_manifest_sha256'] == SOURCE04_MANIFEST
            and proof['source_sha256'] == '24011e3c861e3b0e77c6a25c07116375cded2225c1f05f42cd84aa1e4a139215'
            and proof['native_sha256'] == '52e89b1cb6f4b859d3c9c2f5b528c8c22ddd983e21990a4a4dd5357625c3aba2'
            and proof['current_functions'] == 740 and proof['current_units'] == 14
            and set(proof['changes']) == {'run.py', 'admission.py', 'processes.py'},
            'exact736 source/profile adaptation inventory')
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
            require(reverse.count(edit['after']) == edit['count'], 'exact740 identity inverse count')
            reverse = reverse.replace(edit['after'], edit['before'])
        require(reverse.encode() == original, 'complete original source04 inverse: ' + name)
    require(all(proof[k] is True for k in ('original24_11_59_case_sources_byte_identical',
                'original37_controls_byte_identical',
                'actual_current_objects_required_for_new_probe_and_reordered_images'))
            and proof['native_Torch_compiler_executed'] is False
            and proof['old_runtime_PASS_transferred'] is False,
            'original cases and separate actual probe execution remain mandatory')
    return {'source04_to03': previous, 'physical740_to_source04': proof}


def restore_qualifier740(repository, destination, inputs=None):
    destination = restore(repository, destination, 'validation/current-generator131-source',
                          QUALIFIER_ARCHIVE, QUALIFIER_MANIFEST,
                          'heptatorch.generator740-original-qualification-source.v1', inputs,
                          embedded_manifest=True)
    verify_qualifier740(destination, inputs)
    return destination


CURRENT_SOURCE = '0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43'
CURRENT_IMAGE = 'ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e'
CURRENT_MANIFEST = '348ed8e949256f985cdca0d9c612dc646ae738beee6013c27ab405d2df3e67c8'
CURRENT_BINDING = 'a04f4fd3979b5750f8342bdd63604dc56aa3dc2f6b0f77fcb96542c22903a6b9'


def successor_qualifier(parent, inputs=None):
    """Derive only the current profile in a copy of the complete frozen reader."""
    parent = Path(parent)
    original_proof = verify_qualifier740(parent, inputs)
    original_manifest = json.loads(read(parent / 'package-manifest.json', QUALIFIER_MANIFEST, inputs))
    members = {name: matches(parent / name, pin, inputs)
               for name, pin in original_manifest['files'].items()}
    members['package-manifest.json'] = read(parent / 'package-manifest.json', QUALIFIER_MANIFEST, inputs)
    output = {'parent740/' + name: raw for name, raw in members.items()}
    output.update({name: raw for name, raw in members.items() if name != 'package-manifest.json'})
    changes = {}
    identities = (
        ('24011e3c861e3b0e77c6a25c07116375cded2225c1f05f42cd84aa1e4a139215', CURRENT_SOURCE),
        ('52e89b1cb6f4b859d3c9c2f5b528c8c22ddd983e21990a4a4dd5357625c3aba2', CURRENT_IMAGE),
        ('e6eca58702d42be6dedae69a402d829c60444f2f4d4283b33cf4fc0ccfe34961', CURRENT_MANIFEST),
        ('ecaf51bfb23dcd69ae03b62a2f3976fe9f5ac8b526a42ced14b566e46b2b021f', CURRENT_BINDING),
    )
    for name in ('run.py', 'admission.py', 'processes.py'):
        original = members[name]
        current = original.decode()
        edits = []
        # No hash is changed by the function-count substitution.
        require(not any('740' in value for value in re.findall(r'\b[0-9a-f]{64}\b', current)),
                'profile-count replacement cannot alter a frozen source pin')
        count = {'run.py': 7, 'admission.py': 8, 'processes.py': 0}[name]
        profile_chunks = re.split(r'([0-9a-f]{64})', current)
        require(current.count('740') == count
                and sum(chunk.count('775') for chunk in profile_chunks[::2]) == 0,
                'exact original current-profile labels and count guards')
        if count:
            current = current.replace('740', '775')
            edits.append(dict(before='740', after='775', count=count))
        if name in ('run.py', 'processes.py'):
            require(current.count('current14') == 2 and 'current15' not in current,
                    'exact current object-custody field labels')
            current = current.replace('current14', 'current15')
            edits.append(dict(before='current14', after='current15', count=2))
        if name == 'processes.py':
            before = "need(len(self.objects)==14,'full14 actual objects required by original entry qualification')"
            after = "need(len(self.objects)==15,'full15 actual objects required by original entry qualification')"
            require(current.count(before) == 1 and after not in current,
                    'one exact complete current object inventory guard')
            current = current.replace(before, after, 1)
            edits.append(dict(before=before, after=after, count=1))
        if name == 'admission.py':
            for before, after in identities:
                require(current.count(before) == 1 and after not in current,
                        'one exact source, image, manifest or live binding identity')
                current = current.replace(before, after, 1)
                edits.append(dict(before=before, after=after, count=1))
            # Only the actual current object inventory increases. The frozen
            # 712 source reconstruction retains its own fifteen-unit layout.
            for before, after in (
                ("NAMES = ['rng03'] + [f'logical_half_{i:02d}' for i in range(12)] + ['owners']",
                 '# Current unit order is admitted from the exact canonical manifest.'),
                ("[r['name'] for r in processes]==NAMES+['link']",
                 "[r['name'] for r in processes]==[entry['name'] for entry,_,_ in units]+['link']"),
                ("receipt['units_planned']==14", "receipt['units_planned']==15"),
                ("len(receipt['units'])==14", "len(receipt['units'])==15"),
                ("struct.pack('<3Q',14,4,0)", "struct.pack('<3Q',15,4,0)"),
                ("'core_functions':775,'units':14,'actual_cold_processes':15",
                 "'core_functions':775,'units':15,'actual_cold_processes':16"),
            ):
                require(current.count(before) == 1 and after not in current,
                        'one exact current cold-build inventory guard: ' + before)
                current = current.replace(before, after, 1)
                edits.append(dict(before=before, after=after, count=1))
        restored = current
        for edit in reversed(edits):
            if edit['before'] == '740':
                # The immutable linker hash begins761a. Count labels may not
                # modify any complete source/tool digest during the inverse.
                pieces = re.split(r'([0-9a-f]{64})', restored)
                require(sum(piece.count(edit['after']) for piece in pieces[::2]) == edit['count'],
                        'complete successor profile-label inverse edit count')
                for index in range(0, len(pieces), 2):
                    pieces[index] = pieces[index].replace(edit['after'], edit['before'])
                restored = ''.join(pieces)
            else:
                require(restored.count(edit['after']) == edit['count'], 'complete successor inverse edit count')
                restored = restored.replace(edit['after'], edit['before'])
        require(restored.encode() == original, 'complete frozen740 qualifier source inverse: ' + name)
        def bodies(raw):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        old_bodies, new_bodies = bodies(original), bodies(current)
        expected = {'run.py': {'package', 'tensor_owner', 'main', 'entry59'},
                    'admission.py': {'admit'}, 'processes.py': {'ToolAudit'}}[name]
        require(old_bodies.keys() == new_bodies.keys()
                and {key for key in old_bodies if old_bodies[key] != new_bodies[key]} == expected,
                'only package/profile admission, owner binding and result identity change')
        output[name] = current.encode()
        changes[name] = dict(original_sha256=sha(original), derived_sha256=sha(output[name]),
                            whole_file_inverse_exact=True, edits=edits)
    modes = {name: original_manifest['files'][name]['mode'] for name in original_manifest['files']}
    modes['package-manifest.json'] = stat.S_IMODE((parent / 'package-manifest.json').stat().st_mode)
    file_modes = {'parent740/' + name: mode for name, mode in modes.items()}
    file_modes.update({name: mode for name, mode in modes.items() if name != 'package-manifest.json'})
    manifest = dict(schema='heptatorch.generator775-original-qualification-source.v1',
        parent740_manifest_sha256=QUALIFIER_MANIFEST,
        source_sha256=CURRENT_SOURCE, native_sha256=CURRENT_IMAGE,
        current_source_manifest_sha256=CURRENT_MANIFEST, current_binding_sha256=CURRENT_BINDING,
        files={name: dict(bytes=len(raw), sha256=sha(raw), mode=file_modes[name])
               for name, raw in sorted(output.items())},
        source_profile_inverse=changes, original_cases_and_controls_unchanged=True)
    return output, file_modes, manifest, original_proof


def verify_qualifier(package, inputs=None):
    package = Path(package)
    files, modes, manifest, original_proof = successor_qualifier(package / 'parent740', inputs)
    raw = read(package / 'package-manifest.json', inputs=inputs)
    require(json.loads(raw) == manifest, 'exact derived775 manifest and full frozen740 inverse')
    actual = {str(path.relative_to(package)) for path in package.rglob('*') if path.is_file()}
    require(actual == set(files) | {'package-manifest.json'}, 'complete current775 qualifier inventory')
    for name, expected in files.items():
        require(matches(package / name, manifest['files'][name], inputs) == expected,
                'every generated qualifier file equals its exact profile-only derivation')
    return dict(parent740=original_proof, successor_profile_inverse=manifest['source_profile_inverse'],
                manifest_sha256=sha(raw), original_cases_and_controls_unchanged=True)


def restore_qualifier(repository, destination, inputs=None):
    destination = Path(destination)
    require(not destination.exists(), 'fresh successor Generator qualifier directory')
    destination.mkdir(parents=True)
    parent = restore_qualifier740(repository, destination / 'parent740', inputs)
    files, modes, manifest, _ = successor_qualifier(parent, inputs)
    for name, raw in files.items():
        if name.startswith('parent740/'):
            continue
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        path.chmod(modes[name])
    with (destination / 'package-manifest.json').open('x') as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    verify_qualifier(destination, inputs)
    return destination
