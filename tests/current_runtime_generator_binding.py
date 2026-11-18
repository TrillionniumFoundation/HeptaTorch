"""Bind Generator131 to the physical729 profile and reconstruct source-only712.

Reconstructed predecessor files are qualification source fixtures. No historical
application executable, object, build receipt, or runtime PASS is substituted.
"""
import hashlib
import json
from pathlib import Path
import re
import types

SOURCE = '04ae430e3e851831fcb97a3fab700b16b6318361d52e90fcea90d2183d1d564b'
IMAGE = '4749413ea952bfd9d1fe41770f94ce1b9b9e3efabb407aa90f725cf33d59c4db'
MANIFEST = 'd952fc722c411a8a5ea45a1c0b06786b533eed514e8f4ad3c8df2d621cdbb4b4'
PARENT_SOURCE = 'fac3d413c6c0f6874d224cbea312108ee2c8baa027102d41890d85094a5faa38'
PARENT_MANIFEST = '708e4b642f630604cedb1d2cf566fefcfb3d9afd74a835f5a265d7e51f1ff4e5'
BUILDER = '9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
QUALIFICATION = '9a45cee45a289dae0a35b0a256f66f6a1ea2d3c8becd30bb2174da1ffb51e9bb'
WITNESS = 'ab13bd5c5a93adf322077cf3145d6ca011b8a01fbc42c687b51a66b7edb36846'
NAMES = ['rng03'] + ['logical_f64_%02d' % i for i in range(12)] + ['owners']
PARENT_NAMES = ['rng03'] + ['cat_f32_%02d' % i for i in range(12)] + ['cg_model', 'owners']
FUNCTION = re.compile(rb'^fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n', re.M | re.S)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, inputs, expected=None):
    path = Path(path)
    need(not path.is_symlink() and path.is_file(), 'regular physical source input required')
    raw = inputs.bind(path, expected) if inputs is not None else path.read_bytes()
    need(expected is None or sha(raw) == expected, 'exact source input identity: ' + str(path))
    return raw


def loaded(path, raw, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def definitions(raw, count, complete=False):
    matches = list(FUNCTION.finditer(raw))
    result = {m[1].decode(): m[0] for m in matches}
    need(len(matches) == len(result) == count, 'complete unique function census')
    prefix = raw[:matches[0].start()]
    if complete:
        need(prefix + b''.join(result.values()) == raw, 'all complete source bytes covered')
    return prefix, result


def current_source(repository, inputs):
    repository = Path(repository).resolve(strict=True)
    source = repository / 'tools/current_runtime/source'
    manifest_raw = read(source / 'manifest.json', inputs, MANIFEST)
    manifest = json.loads(manifest_raw)
    need(manifest['runtime']['sha256'] == SOURCE and manifest['runtime']['functions'] == 729
         and manifest['expected_native_sha256'] == IMAGE
         and [row['name'] for row in manifest['units']] == NAMES,
         'exact physical729/14 source and executable profile')
    path = repository / 'tools/current_runtime/build.py'
    builder = loaded(path, read(path, inputs, BUILDER), 'generator131_actual729_builder')
    bind = (lambda path, expected=None: read(path, inputs, expected))
    runtime, units, summary = builder.check_sources(manifest, source, bind)
    need(summary['source_sha256'] == SOURCE and summary['functions'] == 729
         and summary['complete_core_bodies_matched'] == 729 and summary['units_planned'] == 14,
         'every physical729 body and all fourteen source partitions')
    return runtime, units, summary, manifest_raw


def parent_source_files(repository, inputs=None):
    """Return labelled predecessor source bytes, never predecessor products."""
    read(Path(__file__).resolve(), inputs)
    witness_raw = read(Path(__file__).with_suffix('.json'), inputs, WITNESS)
    witness = json.loads(witness_raw)
    need(witness['schema'] == 'heptatorch.generator131.physical729-parent712-source.v1'
         and witness['source_sha256'] == SOURCE and witness['parent_source_sha256'] == PARENT_SOURCE
         and witness['current_functions'] == 729 and witness['parent_functions'] == 712
         and witness['unchanged_functions'] == 683,
         'exact independently bound source-only predecessor witness')
    runtime, units, summary, manifest_raw = current_source(repository, inputs)
    header, current = definitions(runtime, 729, complete=True)
    need(header.decode() == witness['current_header'] and list(current) == witness['current_order'],
         'exact physical declaration order and aggregate header')
    order = witness['parent_order']
    changed, removed, added = (witness[k] for k in ('changed', 'removed', 'added'))
    need(len(order) == len(set(order)) == 712
         and len(changed) == 22 and len(removed) == 7 and len(added) == 24
         and set(current) - set(order) == set(added)
         and set(order) - set(current) == set(removed)
         and set(witness['before_bodies']) == set(changed) | set(removed),
         'complete finite source delta without filtering')
    parent = {}
    for name in order:
        if name in witness['before_bodies']:
            row = witness['before_bodies'][name]
            body = row['body'].encode()
            need(sha(body) == row['sha256'] and len(body) == row['bytes']
                 and list(definitions(body, 1, complete=True)[1]) == [name],
                 'complete exact original function: ' + name)
            if name in changed:
                need(sha(current[name]) == row['current_sha256']
                     and len(current[name]) == row['current_bytes'] and body != current[name],
                     'complete exact modified physical function: ' + name)
            parent[name] = body
        else:
            parent[name] = current[name]
    for name in added:
        need(sha(current[name]) == witness['added_bodies'][name]['sha256']
             and len(current[name]) == witness['added_bodies'][name]['bytes'],
             'complete exact new physical function: ' + name)
    need(sum(current.get(name) == body for name, body in parent.items()) == 683
         and [name for name in order if name in current and current[name] != parent[name]] == changed,
         'all remaining683 ancestor bodies unchanged')
    ancestor = witness['parent_header'].encode() + b''.join(parent.values())
    need(sha(ancestor) == PARENT_SOURCE and len(ancestor) == witness['parent_bytes'],
         'complete712 inverse constrains every predecessor byte')
    forward = {}
    for name in witness['current_order']:
        forward[name] = current[name] if name in set(changed) | set(added) else parent[name]
    need(header + b''.join(forward.values()) == runtime, 'complete729 source round trip')
    physical = {entry['name']: raw for entry, raw, parsed in units}
    originals = {'source/runtime.hepta': ancestor}
    old_manifest = witness['parent_manifest'].encode()
    need(sha(old_manifest) == PARENT_MANIFEST, 'complete original712 manifest bytes')
    originals['source/manifest.json'] = old_manifest
    old_profile = json.loads(old_manifest)
    need([row['name'] for row in old_profile['units']] == PARENT_NAMES, 'original15 source unit order')
    for row in old_profile['units']:
        name = row['name']
        detail = witness['parent_units'][name]
        if name in ('rng03', 'owners'):
            raw = physical[name]
        else:
            raw = detail['prefix'].encode() + b''.join(parent[fn] for fn in detail['functions'])
        need(len(raw) == detail['bytes'] and sha(raw) == detail['sha256'] == row['sha256'],
             'entire reconstructed predecessor source unit: ' + name)
        originals['source/' + row['path']] = raw
    cg = witness['generator_functions']
    need(len(cg) == 11 and len(set(cg)) == 11
         and all(parent[name] == current[name] for name in cg),
         'all eleven actual Generator core functions remain byte-identical')
    need(len(originals) == 17 and set(originals) == set(witness['source_members']),
         'only the original aggregate/manifest/fifteen source units reconstructed')
    return originals, dict(
        schema='heptatorch.generator131.physical729-parent712-inverse.v1',
        physical_source_sha256=SOURCE, physical_native_sha256=IMAGE,
        physical_functions=729, physical_units=14, parent_source_sha256=PARENT_SOURCE,
        parent_functions=712, parent_units=15, unchanged_parent_bodies=683,
        changed_parent_bodies=changed, removed_parent_bodies=removed, added_physical_bodies=added,
        all11_generator_bodies_byte_exact=True, owners_and_rng_support_byte_exact=True,
        complete_source_inverse_exact=True, complete_round_trip_exact=True,
        reconstructed_source_members={name: {'bytes': len(raw), 'sha256': sha(raw)}
                                      for name, raw in sorted(originals.items())},
        predecessor_source_bytes_labelled=True, historical_ELF_or_objects_substituted=False,
        predecessor_build_receipt_substituted=False, historical_runtime_PASS_transferred=False)


def bind_current(repository, candidate, build, build_sha, inputs):
    repository, candidate = Path(repository), Path(candidate)
    parent, proof = parent_source_files(repository, inputs)
    for name, raw in parent.items():
        need(read(candidate / name, inputs, sha(raw)) == raw,
             'physical predecessor-source fixture equals exact inverse: ' + name)
    path = repository / 'tools/current_runtime/qualification.py'
    qualification = loaded(path, read(path, inputs, QUALIFICATION), 'generator131_current729_qualification')
    need((qualification.SOURCE, qualification.IMAGE, qualification.FUNCTIONS, qualification.UNITS)
         == (SOURCE, IMAGE, 729, 14), 'shared admission has the exact physical729 profile')
    built = qualification.admit_build(build, inputs.bind, expected=build_sha)
    runtime, units, summary, manifest_raw = current_source(repository, inputs)
    return runtime, units, summary, manifest_raw, built, proof
