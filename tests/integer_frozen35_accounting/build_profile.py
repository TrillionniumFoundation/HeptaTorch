#!/usr/bin/env python3
"""Create an isolated F32 fixed35 observer profile for the actual I64 image.

This adapts pins, tags and exact source identities. It does not widen the fixed
command domain, alter an ownership equation, or claim general I64 accounting.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil

PARENT_MANIFEST = '934aa2916e6cd00d634839b9b3676b73b62dfe70815a8a44b043dfb65b91d3b3'
OLD_SOURCE = '6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3'
OLD_IMAGE = '830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51'
NEW_SOURCE = 'adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62'
NEW_IMAGE = 'b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08'
ORIGINAL = '6ea9e8b33164089034a2a71a03f4850e1ec18ed1a09155d7841f700ef311c833'
OLD_STRUCTURAL = 'f503df2d1dbce8242591e77211b96a7787201e991db8186c9f1cca6576244409'
OLD_OBSERVER = '4cd9c3d1fafd1eee04ac1b2b8bb720a099d53dc821b25b9cf4380e6c67b36a71'
OLD_LAUNCHER = '1dca385159d332ed75adb57c1173b6a0b010597adabf5fe38001d97d1ea769d2'
OLD_PROFILE = '83f673c3f1ff4fe3fa7a2f35749326ff8eeeaa60df570da080bc82d8712e683b'
NEW_NAME = 'integer-tensor-544-fixed35-F32'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2)+'\n').encode()


def exact(text, before, after, count=1):
    assert text.count(before) == count, (before, text.count(before), count)
    return text.replace(before, after)


def inverse(old, new):
    a, b = old.decode(), new.decode()
    edits = []
    for tag, a0, a1, b0, b1 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != 'equal':
            edits.append({'old_start': a0, 'old_end': a1, 'new_start': b0, 'new_end': b1,
                          'before': a[a0:a1], 'after': b[b0:b1]})
    restored = b
    for e in reversed(edits):
        assert restored[e['new_start']:e['new_end']] == e['after']
        restored = restored[:e['new_start']]+e['before']+restored[e['new_end']:]
    assert restored.encode() == old
    return {'original_sha256': sha(old), 'derived_sha256': sha(new), 'edits': edits,
            'inverse_exact_original_bytes': True}


def funcs(raw):
    return {name.decode(): body for body, name in re.findall(
        rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)', raw, re.M|re.S)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-package', 'parent-build', 'build', 'original-structural', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent = a.parent_package.resolve()
    manifest = parent/'validation/dtype-frozen35-accounting/source-package-manifest.json'
    assert sha(manifest.read_bytes()) == PARENT_MANIFEST
    m = json.loads(manifest.read_bytes())
    inputs = {}

    def bind(path, expected=None):
        path = path.resolve()
        raw = path.read_bytes()
        assert expected is None or sha(raw) == expected, (path, sha(raw), expected)
        inputs[str(path)] = {'bytes': len(raw), 'sha256': sha(raw), 'mode': path.stat().st_mode & 0o777}
        return raw

    bind(Path(__file__))
    bind(manifest, PARENT_MANIFEST)
    for entry in m['files']:
        raw = bind(parent/entry['path'], entry['sha256'])
        assert len(raw) == entry['bytes'] and (parent/entry['path']).stat().st_mode & 0o777 == entry['mode']
        dst = out/entry['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)
        dst.chmod(entry['mode'])
    original = bind(a.original_structural, ORIGINAL)
    dep = out/'tests/dtype_owner_accounting/owner640.py'
    dep.parent.mkdir(parents=True, exist_ok=True)
    dep.write_bytes(original)
    old_native = bind(a.parent_build/'runtime.hepta', OLD_SOURCE)
    bind(a.parent_build/'tensor-dtype-dropout-session', OLD_IMAGE)
    current_native = bind(a.build/'runtime.hepta', NEW_SOURCE)
    bind(a.build/'tensor-integer-session', NEW_IMAGE)
    old_functions, current_functions = funcs(old_native), funcs(current_native)
    assert len(old_functions) == 523 and len(current_functions) == 544
    base = parent/'tests/dtype_frozen35_accounting'
    dest = out/'tests/integer_frozen35_accounting'
    dest.mkdir(parents=True)
    all_inverse = {}

    def publish(name, content, old=None):
        raw = content.encode() if isinstance(content, str) else content
        (dest/name).write_bytes(raw)
        if old is not None:
            all_inverse[name] = inverse(old, raw)
        return sha(raw)

    structural_old = bind(base/'owner640_storage_tombstone.py', OLD_STRUCTURAL)
    text = structural_old.decode()
    for before, after, count in ((OLD_SOURCE, NEW_SOURCE, 1), (OLD_IMAGE, NEW_IMAGE, 1),
                                 ('dtype-native-dropout-523', NEW_NAME, 1),
                                 ('HTDTYP03', 'HTINT002', 2), ('HTODTY03', 'HTOINT02', 2)):
        text = exact(text, before, after, count)
    structural_sha = publish('owner640_fixed35.py', text, structural_old)
    structural_proof = all_inverse['owner640_fixed35.py']
    structural_proof.update(scope='Only pins, profile name and two exact state tags changed; all equations and rejects retained.',
                            general_I64_owner_verifier=False,
                            admission_requires_all_actual_node_words_576_to_639_zero=True)
    publish('structural-profile-adaptation.json', json_bytes(structural_proof))
    publish('legacy_owner_rules.py', bind(base/'legacy_owner_rules.py'))
    publish('storage-tombstone-correction.json', bind(base/'storage-tombstone-correction.json'))

    config = json.loads(bind(base/'profile.json', OLD_PROFILE))
    config.update(schema='heptatorch.frozen35.integer544-F32-domain-profile.v1',
        source_sha256=NEW_SOURCE, native_sha256=NEW_IMAGE,
        structural_observer_sha256=structural_sha,
        function_pins={name: sha(body) for name, body in current_functions.items()},
        fixed35_domain='Original35 commands1..56 only. Every actual full80-word node has zero dtype/history words576..639. No I64 Tensor, I64 view-base edge, CPU-promotion temporary,105 dropout or106 scalar operation is admitted by this observer. General I64 ownership has separate source-bound resource controls.',
        structural_claim='Complete actual80-word nodes; all previous640 equations and original18 refusals retained. Only source/image/profile and state tags change. No descriptor projection or allocation remainder.',
        parent_profile_sha256=OLD_PROFILE, parent_source_sha256=OLD_SOURCE,
        parent_native_sha256=OLD_IMAGE, parent_observer_sha256=OLD_OBSERVER,
        original_structural_observer_sha256=ORIGINAL)
    config['capture_callers'].update(integer_view_root_retain=['record', 'rng_strided_view'],
                                    integer_promote_operand=['integer_binary'],
                                    integer_native_dropout=['integer_dropout_command'],
                                    integer_scalar_add=['integer_scalar_command'])
    for target, expected in config['capture_callers'].items():
        got = [name for name, body in current_functions.items() if ('"function":"'+target+'"').encode() in body]
        assert sorted(got) == sorted(expected), (target, got, expected)
    profile_sha = publish('profile.json', json_bytes(config))

    observer_old = bind(base/'owner_accounting523.py', OLD_OBSERVER)
    text = observer_old.decode()
    text = exact(text, 'STRUCTURAL = HERE / "owner640_storage_tombstone.py"', 'STRUCTURAL = HERE / "owner640_fixed35.py"')
    text = exact(text, OLD_STRUCTURAL, structural_sha)
    text = exact(text, 'len(bodies) == 523', 'len(bodies) == 544')
    text = exact(text, '"accounting_profile": PROFILE_FILE}',
        '"accounting_profile": PROFILE_FILE,\n            "structural_profile_adaptation": HERE / "structural-profile-adaptation.json"}')
    observer_sha = publish('owner_accounting_int64.py', text, observer_old)
    launcher_old = bind(base/'run_frozen35_accounted.py', OLD_LAUNCHER)
    text = exact(launcher_old.decode(), 'import owner_accounting523 as observer', 'import owner_accounting_int64 as observer')
    launcher_sha = publish('run_frozen35_accounted.py', text, launcher_old)

    checker_old = bind(base/'check_frozen35_accounted.py')
    text = checker_old.decode()
    for before, after, count in ((OLD_SOURCE, NEW_SOURCE, 1), (OLD_IMAGE, NEW_IMAGE, 1),
                                 ('dtype-native-dropout-523', NEW_NAME, 1),
                                 (OLD_STRUCTURAL, structural_sha, 1), (OLD_OBSERVER, observer_sha, 1),
                                 (OLD_LAUNCHER, launcher_sha, 1), (OLD_PROFILE, profile_sha, 1)):
        text = exact(text, before, after, count)
    publish('check_frozen35_accounted.py', text, checker_old)
    controls_old = bind(base/'check_owner_snapshots.py')
    text = exact(controls_old.decode(), OLD_OBSERVER, observer_sha)
    text = exact(text, 'dtype-native-dropout-523', NEW_NAME)
    publish('check_owner_snapshots.py', text, controls_old)

    tombstone_old = bind(base/'test_storage_tombstone.py')
    text = exact(tombstone_old.decode(), 'import owner_accounting523 as observer', 'import owner_accounting_int64 as observer')
    marker = '        original = load("tombstone_original640", observer.ORIGINAL_STRUCTURAL)\n'
    inserted = marker+'''        # Pure old-predicate replay under the new snapshot identity. The old
        # file and every function body remain byte-identical; only these three
        # module data bindings identify this profile's copied JSON snapshot.
        original.SOURCE = observer.structural.SOURCE
        original.IMAGE = observer.structural.IMAGE
        original.PROFILE = observer.structural.PROFILE
        report["original_predicate_profile_binding"] = {
            "source_sha256": original.SOURCE, "native_sha256": original.IMAGE,
            "profile": original.PROFILE, "function_code_changed": False,
            "purpose": "copied JSON predicate replay only; old native profile not relabelled"}
'''
    text = exact(text, marker, inserted)
    publish('test_storage_tombstone.py', text, tombstone_old)
    changed = [name for name in old_functions if old_functions[name] != current_functions[name]]
    assert len(changed) == 14
    construction_names = ['binary', 'matmul', 'linear', 'mse', 'layernorm', 'batchnorm_eval',
                          'cross_entropy_probs', 'descriptor', 'descriptor_page', 'finish_command']
    assert all(old_functions[name] == current_functions[name] for name in construction_names)
    release = {'schema': 'heptatorch.integer544-fixed35-profile-adaptation.v1', 'status': 'SOURCE_PREPARED_NOT_EXECUTED',
        'parent_manifest_sha256': PARENT_MANIFEST, 'source_sha256': NEW_SOURCE, 'native_sha256': NEW_IMAGE,
        'old_functions': 523, 'new_functions': 544, 'changed_existing_functions': changed,
        'added_functions': [name for name in current_functions if name not in old_functions],
        'unchanged_constructor_functions': construction_names,
        'new_record_guard': 'The only new record root is conditioned on dtype2 and view op10/16; every full35 node is additionally required to have all words576..639 zero.',
        'no_native_or_Torch_execution_by_adapter': True, 'all_legacy_numeric_and_18_refusal_assertions_retained': True,
        'inverse_proofs': all_inverse, 'inputs': inputs,
        'whole_pytorch_gate': 'HOLD', 'pytorch_api_denominator': 2590, 'full_pytorch_rewrite_complete': False}
    (out/'profile-adaptation.json').write_bytes(json_bytes(release))
    for path, e in inputs.items():
        f = Path(path)
        assert f.stat().st_size == e['bytes'] and sha(f.read_bytes()) == e['sha256'] and f.stat().st_mode & 0o777 == e['mode']
    print(json.dumps({'status': 'SOURCE_PREPARED_NOT_EXECUTED', 'observer_sha256': observer_sha,
                      'structural_sha256': structural_sha, 'profile_sha256': profile_sha,
                      'launcher_sha256': launcher_sha, 'adaptation_sha256': sha((out/'profile-adaptation.json').read_bytes())}))


if __name__ == '__main__':
    main()
