"""Explicit current-source admission around the unchanged PR38 F32 reader.

The original twelve reader files are never changed. This layer independently
admits the current build, reverses the exact logical-reduction source delta,
executes all three original source inverses, and extends the owner metadata
predicate to the newly accepted Bool source types.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PR38_SOURCE = '82360ce87dae6f9a004ef50be4f991b6f83905a83f4b17ca8d6634827b7bd313'
DELTA_SHA = '4dc0bcae639b1934674a8e786fef04b55b4dd9bf138fa4953ec9824b51a27683'
READER_MANIFEST_SHA = 'a97c262047f17c50a9ad77815754644e1df19350ec2511f274997b4711dda974'
SUPPORT_SHA = '5e0a66b1a45c469f6c80ec611ff49d26288230b736b6939f957cba8a7ad21ec1'
OWNER_SHA = '4ef9545bb42dbc1ef5da603814a157163bed18418992d9b2489fd39378c3a4aa'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(condition, rule):
    if not condition:
        raise AssertionError(rule)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def current_profile():
    return load(ROOT / 'tools/current_runtime/qualification.py', 'fusion_current_profile')


def bind_frozen_reader(reader, inputs):
    reader = Path(reader)
    manifest = json.loads(inputs.bind(HERE / 'source-manifest.json', READER_MANIFEST_SHA))
    require(manifest['file_count'] == len(manifest['files']) == 12, 'all twelve original reader files')
    for name, pin in manifest['files'].items():
        original = inputs.bind(HERE / name, pin['sha256'])
        copied = inputs.bind(reader / name, pin['sha256'])
        require(original == copied and len(copied) == pin['bytes'], 'exact original reader member: ' + name)
    inputs.bind(Path(__file__).resolve())
    inputs.bind(HERE / 'fusion-source.json', DELTA_SHA)
    return manifest


def _logical_source_inverse(source, inputs, support):
    """Reconstruct the complete PR38 source using the observed current bodies."""
    profile = current_profile()
    require(sha(source) == '0c1c0dd3acf0285a887a2766c1308dfd992acf3fe51849dcd98749bb64a00f3b', 'current source pin before fusion inverse')
    delta = json.loads(inputs.bind(HERE / 'fusion-source.json', DELTA_SHA))
    require(delta['parent_source_sha256'] == PR38_SOURCE and delta['current_functions'] == 700,
            'exact logical fusion source scope')
    actual = support.source_functions(source, 700)
    baseline = inputs.bind(Path(support.__file__).parent / 'baseline/runtime.hepta', support.PARENT_SOURCE)
    old = support.source_functions(baseline, 700)
    old_call = b'  v23:u64 = call(v9,v10,v11,v12,v13,v14,v15) {"function":"sd_sum_tensor"}\n'
    new_call = old_call.replace(b'sd_sum_tensor', b'sf_sum_f32')
    original_dispatch = old['si_dispatch_sum']
    require(original_dispatch.count(old_call) == 1, 'exact original F32 dispatch position')
    parent = dict(old)
    parent['si_dispatch_sum'] = original_dispatch.replace(old_call, new_call)
    for name, pin in delta['preserved_PR38_fixes'].items():
        require(sha(actual[name]) == pin, 'preserved complete PR38 fix: ' + name)
        parent[name] = actual[name]
    parent_header = baseline.split(b'\n', 1)[0].replace(b'sum_integral_bool_runtime', b'sum_integral_f32_runtime')
    original = parent_header + b'\n' + b''.join(parent.values())
    require(sha(original) == PR38_SOURCE and len(parent) == 701, 'complete original PR38 source reconstruction')
    modified, added, removed = (delta[key] for key in ('modified_functions', 'added_functions', 'removed_functions'))
    require(set(actual) == (set(parent) - set(removed)) | set(added), 'exact current function inventory delta')
    require(set(modified) <= set(actual) & set(parent), 'all modified functions existed')
    for name, pin in {**modified, **added}.items():
        require(sha(actual[name]) == pin, 'exact current authored body: ' + name)
    for name, pin in removed.items():
        require(name not in actual and sha(parent[name]) == pin, 'exact unreachable body removed: ' + name)
    unchanged = sorted(set(actual) & set(parent) - set(modified))
    require(len(unchanged) == 689 and all(actual[n] == parent[n] for n in unchanged),
            'all 689 other complete bodies unchanged')
    # Reconstruct from the current bodies rather than accepting the baseline as
    # a substitute for them. Only the five changed and seven removed bodies may
    # be supplied by the separately bound historical source.
    restored = parent_header + b'\n' + b''.join(
        parent[n] if n in modified or n in removed else actual[n] for n in parent)
    require(restored == original and sha(restored) == PR38_SOURCE, 'whole current-to-PR38 inverse exact')
    original_proof = support.source_inverse(restored, inputs)
    require(original_proof['source_sha256'] == PR38_SOURCE and original_proof['whole_inverse_exact'] is True,
            'execute complete original three-layer source proof')
    return dict(schema='heptatorch.current-logical-fusion.source-proof.v1', whole_inverse_exact=True,
        source_sha256=sha(source), parent_source_sha256=PR38_SOURCE,
        current_functions=700, original_functions=701, unchanged_functions=689,
        modified_functions=sorted(modified), added_functions=sorted(added), removed_functions=sorted(removed),
        preserved_PR38_fixes=delta['preserved_PR38_fixes'], current_header=source.split(b'\n', 1)[0].decode(),
        pr38_source_reconstructed_sha256=sha(restored), pr38_to_baseline=original_proof,
        original_FAIL_reports_modified=False, runtime_qualification_transferred=False)


def source_inverse(source, inputs, support):
    """Add exact729 and cat inverses before the original historical proofs."""
    profile = current_profile()
    require(sha(source) == profile.SOURCE and profile.FUNCTIONS == 729 and profile.UNITS == 14,
            'actual complete current729 profile before historical source proofs')
    outer_path = ROOT / 'tools/current_runtime/fusion_source_transition.py'
    inputs.bind(outer_path, 'ca68c251be10292f0e2e47c4f010a19ec8542cb26b30c92a2d27d37dddba1ae8')
    outer = load(outer_path, 'fusion_current729_source_transition')
    historical700, outer_proof = outer.restore_source(source, inputs.bind)
    path = ROOT / 'tools/current_runtime/cat_source_transition.py'
    inputs.bind(path)
    cat = load(path, 'fusion_current_cat_source_transition')
    require(sha(historical700) == cat.SOURCE == outer.PARENT_SOURCE,
            'exact historical700 subject before the unchanged cat inverse')
    restored, proof = cat.restore_source(historical700, inputs.bind)
    previous = _logical_source_inverse(restored, inputs, support)
    require(previous['source_sha256'] == cat.PARENT_SOURCE
            and previous['whole_inverse_exact'] is True,
            'the complete original logical/PR38/AVX2 proof must execute')
    return dict(schema='heptatorch.current729-cat.source-proof.v1',
        source_sha256=sha(source), parent_source_sha256=cat.PARENT_SOURCE,
        whole_inverse_exact=True, outer_source_inverse=outer_proof,
        cat_transition=proof, logical_fusion=previous,
        actual_functions=729, historical_functions=700, historical_count_binding_separated=True,
        original_source_proofs_modified=False, runtime_qualification_transferred=False)

def adapted_parent(repository, filename, inputs, *, owner=False):
    """Use the original full observer with only exact identity substitutions."""
    support = sys.modules['native_support']
    profile = current_profile()
    raw = inputs.bind(Path(repository) / support.INTENT, support.INTENT_SHA)
    row = json.loads(raw)['source_payloads'][filename]
    original = row['text'].encode()
    require(sha(original) == row['sha256'] == support.PARENT_SHA[filename]
            and len(original) == row['size'] and row['eof'], 'frozen complete owner payload')
    edits = [(b'HTSINT01', b'HTSINT02', 2 if owner else 1),
             (b'HTOSIN01', b'HTOSIN02', 2 if owner else 1)]
    if owner:
        count = str(profile.FUNCTIONS).encode()
        edits += [(b'functions == 699', b'functions == ' + count, 1),
                  (b'sum-integral-source01-UNBOUND-699', b'current-fusion-UNBOUND-' + count, 1),
                  (b'sum-integral-source01-%d-preserved47-48', b'current-fusion-%d-preserved47-48', 1),
                  (b'heptatorch.sum-integral-owner-source01.v1', b'heptatorch.current-fusion-owner.v1', 1)]
    derived = original
    for before, after, count in edits:
        require(derived.count(before) == count and after not in derived, 'exact owner identity substitution')
        derived = derived.replace(before, after)
    inverse = derived
    for before, after, count in reversed(edits):
        require(inverse.count(after) == count, 'owner identity inverse count')
        inverse = inverse.replace(after, before)
    require(inverse == original, 'whole original owner inverse')
    name = 'current_fusion_' + filename.replace('.py', '')
    module = types.ModuleType(name)
    module.__file__ = str(Path(repository) / support.INTENT) + '#' + filename
    sys.modules[name] = module
    exec(compile(derived, module.__file__, 'exec'), module.__dict__)
    proof = dict(parent_name=filename, original_sha256=sha(original), derived_sha256=sha(derived),
                 original_bytes=len(original), derived_bytes=len(derived), whole_inverse_exact=True,
                 edits=[dict(before=a.decode(), after=b.decode(), count=n) for a, b, n in edits])
    return module, proof


def owner_adapter(reader):
    """Derive only the explicit function-count and Bool-source-type changes."""
    reader = Path(reader)
    profile = current_profile()
    raw = (reader / 'owner_adapter.py').read_bytes()
    require(sha(raw) == OWNER_SHA, 'fixed original F32 owner adapter')
    edits = [
        (b'observer.configure(owner, source_sha, image_sha, 701)',
         ('observer.configure(owner, source_sha, image_sha, %d)' % profile.FUNCTIONS).encode()),
        (b'w(sn, 576) in (1, 2) and w(rn, 576) == 1',
         b'w(sn, 576) in (0, 1, 2, 3) and w(rn, 576) == 1'),
    ]
    derived = raw
    for before, after in edits:
        require(derived.count(before) == 1 and after not in derived, 'exact explicit owner extension')
        derived = derived.replace(before, after)
    inverse = derived
    for before, after in reversed(edits):
        require(inverse.count(after) == 1, 'owner extension inverse count')
        inverse = inverse.replace(after, before)
    require(inverse == raw, 'whole fixed owner adapter inverse')
    module = types.ModuleType('fusion_owner_adapter')
    module.__file__ = str(reader / 'owner_adapter.py') + '#explicit-fusion-derivation'
    sys.modules[module.__name__] = module
    exec(compile(derived, module.__file__, 'exec'), module.__dict__)
    original_make = module.make_observer

    def make(repository, inputs, source_sha, image_sha):
        inputs.bind(reader / 'owner_adapter.py', OWNER_SHA)
        inputs.bind(Path(__file__).resolve())
        observer, owner, proof = original_make(repository, inputs, source_sha, image_sha)
        proof['explicit_fusion_owner'] = dict(original_sha256=sha(raw), derived_sha256=sha(derived),
            whole_inverse_exact=True, edits=[dict(before=a.decode(), after=b.decode(), count=1) for a,b in edits],
            Bool_source_types=[0,1,2,3], all_original_F32_cast_and_lifetime_predicates_unchanged=True)
        return observer, owner, proof

    module.make_observer = make
    return module


def load_reader(reader):
    reader = Path(reader).resolve()
    sys.path.insert(0, str(reader))
    support = load(reader / 'native_support.py', 'native_support')
    support.adapted_parent = adapted_parent
    adapter = owner_adapter(reader)
    sys.modules['owner_adapter'] = adapter
    native = load(reader / 'native.py', 'fusion_original_f32_reader')
    profile = current_profile()
    native.SOURCE, native.IMAGE = profile.SOURCE, profile.IMAGE

    def admit_build(args, inputs):
        bind_frozen_reader(reader, inputs)
        inputs.bind(ROOT / 'tools/current_runtime/qualification.py')
        require(args.expected_source_sha256 == profile.SOURCE and args.expected_native_sha256 == profile.IMAGE,
                'explicit current requested identities')
        build = profile.admit_build(args.build_report.parent, inputs.bind, args.expected_build_sha256)
        require(args.source == args.build_report.parent / 'runtime.hepta'
                and args.binary == args.build_report.parent / 'heptatorch', 'exact current build product locations')
        build['admitted_source_transition'] = source_inverse(inputs.bind(args.source, profile.SOURCE), inputs, support)
        return build

    native.admit_build = admit_build
    return native, support, adapter
