"""Bind Generator131 to the current profile and reconstruct source-only712.

Reconstructed predecessor files are qualification source fixtures. No historical
application executable, object, build receipt, or runtime PASS is substituted.
"""
import hashlib
import json
from pathlib import Path
import re
import types

SOURCE = 'db88eaad9fb69010ae4f6925f4e8c54edce815b9f46507bc0c7c2536a4cb2c0b'
IMAGE = '929529e7c86ea3ce12e522f9b01b1a7c4180feb00bde812bc26268091d5032e4'
MANIFEST = '361609f574cdde8854d176f22c810aea9a72ef80adcfd0d8beeb1c67ae4b6f52'
HALF_SOURCE = 'bf3641b1a9061c6f92b8cbeef75adfac89e82ee16d32a9f650cb938d9e918386'
PREDICATE_INVERSE = '71244cd75fb41ce07876d7f5699b66ae43c73adc5dd5d09fb2da0dec9c8c7439'
STORAGE_SOURCE = '51820849f7becec549be15babace00f00687601ae7f53bcee0e636ca02f721f9'
HALF_EXTENSION = 'e0a2eac589ddfe8b09794fb55c569a73d7d7802888130646f59d33adca266f66'
PARENT_SOURCE = 'fac3d413c6c0f6874d224cbea312108ee2c8baa027102d41890d85094a5faa38'
PARENT_MANIFEST = '708e4b642f630604cedb1d2cf566fefcfb3d9afd74a835f5a265d7e51f1ff4e5'
BUILDER = '9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
QUALIFICATION = '9a45cee45a289dae0a35b0a256f66f6a1ea2d3c8becd30bb2174da1ffb51e9bb'
WITNESS = 'b4422af9ec3512eb3eb1676345e35515d4ff94dbb79e202d370b0a0a12a7c83d'
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
    names = [row['name'] for row in manifest['units']]
    need(manifest['runtime']['sha256'] == SOURCE and manifest['runtime']['functions'] == 778
         and manifest['expected_native_sha256'] == IMAGE
         and len(names) == len(set(names)) == 15
         and names[0] == 'rng03' and names[-1] == 'owners',
         'exact physical775/15 source and executable profile')
    path = repository / 'tools/current_runtime/build.py'
    builder = loaded(path, read(path, inputs, BUILDER), 'generator131_actual736_builder')
    bind = (lambda path, expected=None: read(path, inputs, expected))
    runtime, units, summary = builder.check_sources(manifest, source, bind)
    need(summary['source_sha256'] == SOURCE and summary['functions'] == 778
         and summary['complete_core_bodies_matched'] == 778 and summary['units_planned'] == 15,
         'every physical775 body and all fifteen source partitions')
    return runtime, units, summary, manifest_raw


def parent_source_files(repository, inputs=None):
    """Return labelled predecessor source bytes, never predecessor products."""
    read(Path(__file__).resolve(), inputs)
    witness_raw = read(Path(__file__).with_suffix('.json'), inputs, WITNESS)
    witness = json.loads(witness_raw)
    need(witness['schema'] == 'heptatorch.generator131.physical736-parent712-source.v1'
         and witness['source_sha256'] == STORAGE_SOURCE and witness['parent_source_sha256'] == PARENT_SOURCE
         and witness['current_functions'] == 736 and witness['parent_functions'] == 712
         and witness['unchanged_functions'] == 683,
         'exact independently bound source-only predecessor witness')
    runtime, units, summary, manifest_raw = current_source(repository, inputs)
    index_path = Path(repository) / 'tools/current_runtime/index_signbit_source_transition.py'
    indexing = loaded(index_path, read(index_path, inputs), 'generator131_index_signbit_source_inverse')
    runtime, index_inverse = indexing.restore_source(runtime, lambda path, expected=None: read(path, inputs, expected))
    need(index_inverse['source_sha256'] == SOURCE and index_inverse['functions'] == 778
         and index_inverse['parent_functions'] == 775
         and index_inverse['whole_source_inverse_exact'] is True
         and index_inverse['whole_round_trip_exact'] is True,
         'complete778 source before the original775 Generator source obligations')
    relu_path = Path(repository) / 'tools/current_runtime/typed_relu_source_transition.py'
    relu = loaded(relu_path, read(relu_path, inputs), 'generator131_typed_relu_source_inverse')
    runtime, relu_inverse = relu.restore_source(runtime, lambda path, expected=None: read(path, inputs, expected))
    need(relu.SOURCE == indexing.PARENT_SOURCE and sha(runtime) == relu.PARENT_SOURCE
         and relu_inverse['functions'] == 775 and relu_inverse['parent_functions'] == 773
         and relu_inverse['changed_functions'] == ['unary']
         and relu_inverse['whole_source_inverse_exact'] is True
         and relu_inverse['whole_round_trip_exact'] is True,
         'complete typed ReLU775 inverse before the frozen same-F64 source')
    same_path = Path(repository) / 'tools/current_runtime/same_f64_arithmetic_source_transition.py'
    same = loaded(same_path, read(same_path, inputs), 'generator131_same_f64_arithmetic_source_inverse')
    runtime, same_inverse = same.restore_source(runtime, lambda path, expected=None: read(path, inputs, expected))
    need(same.SOURCE == relu.PARENT_SOURCE and sha(runtime) == same.PARENT_SOURCE
         and same_inverse['functions'] == same_inverse['parent_functions'] == 773
         and same_inverse['changed_functions'] == ['half_mul_priority','public_f64_binary']
         and same_inverse['unchanged_parent_functions'] == 771
         and same_inverse['whole_source_inverse_exact'] is True
         and same_inverse['whole_round_trip_exact'] is True,
         'exact same-F64 arithmetic inverse before the frozen mixed773 source')
    arithmetic_path = Path(repository) / 'tools/current_runtime/public_f64_arithmetic_source_transition.py'
    arithmetic = loaded(arithmetic_path, read(arithmetic_path, inputs),
                        'generator131_public_f64_arithmetic_source_inverse')
    runtime, arithmetic_inverse = arithmetic.restore_source(
        runtime, lambda path, expected=None: read(path, inputs, expected))
    need(arithmetic.SOURCE == same.PARENT_SOURCE and sha(runtime) == arithmetic.PARENT_SOURCE
         and arithmetic_inverse['functions'] == 773 and arithmetic_inverse['parent_functions'] == 771
         and arithmetic_inverse['changed_functions'] == ['binary','half_mul_cast']
         and arithmetic_inverse['unchanged_parent_functions'] == 769
         and arithmetic_inverse['whole_source_inverse_exact'] is True
         and arithmetic_inverse['whole_round_trip_exact'] is True,
         'exact mixed public F64 extension inverted before Half771')
    div_path = Path(repository) / 'tools/current_runtime/half_div_source_transition.py'
    division = loaded(div_path, read(div_path, inputs), 'generator131_half_div_source_inverse')
    runtime, div_inverse = division.restore_source(
        runtime, lambda path, expected=None: read(path, inputs, expected))
    need(division.SOURCE == arithmetic.PARENT_SOURCE and sha(runtime) == division.PARENT_SOURCE
         and div_inverse['functions'] == 771 and div_inverse['parent_functions'] == 767
         and div_inverse['changed_functions'] == ['binary']
         and div_inverse['unchanged_parent_functions'] == 766
         and div_inverse['whole_source_inverse_exact'] is True
         and div_inverse['whole_round_trip_exact'] is True,
         'exact Half771 division extension inverted before wide767')
    wide_path = Path(repository) / 'tools/current_runtime/wide_neg_source_transition.py'
    wide = loaded(wide_path, read(wide_path, inputs), 'generator131_wide_neg_source_inverse')
    runtime, wide_inverse = wide.restore_source(
        runtime, lambda path, expected=None: read(path, inputs, expected))
    need(wide.SOURCE == division.PARENT_SOURCE and sha(runtime) == wide.PARENT_SOURCE,
         'exact wide767 negation extension inverted before Half765')
    addsub_path = Path(repository) / 'tools/current_runtime/half_addsub_source_transition.py'
    addsub = loaded(addsub_path, read(addsub_path, inputs), 'generator131_half_addsub_source_inverse')
    runtime, addsub_inverse = addsub.restore_source(
        runtime, lambda path, expected=None: read(path, inputs, expected))
    need(addsub.SOURCE == wide.PARENT_SOURCE and sha(runtime) == addsub.PARENT_SOURCE,
         'exact Half765 add/sub extension inverted before Half761')
    neg_path = Path(repository) / 'tools/current_runtime/half_neg_source_transition.py'
    neg = loaded(neg_path, read(neg_path, inputs), 'generator131_half_neg_source_inverse')
    runtime, neg_inverse = neg.restore_source(
        runtime, lambda path, expected=None: read(path, inputs, expected))
    need(neg.SOURCE == addsub.PARENT_SOURCE and sha(runtime) == neg.PARENT_SOURCE,
         'exact Half761 negation extension inverted before Half758')
    mul_path = Path(repository) / 'tools/current_runtime/half_mul_source_transition.py'
    mul = loaded(mul_path, read(mul_path, inputs), 'generator131_half_mul_source_inverse')
    runtime, mul_inverse = mul.restore_source(
        runtime, lambda path, expected=None: read(path, inputs, expected))
    need(mul.SOURCE == neg.PARENT_SOURCE and sha(runtime) == mul.PARENT_SOURCE,
         'exact Half758 multiplication extension inverted before BF16742')
    bf16_path = Path(repository) / 'tools/current_runtime/bf16_source_transition.py'
    bf16 = loaded(bf16_path, read(bf16_path, inputs), 'generator131_bf16_source_inverse')
    runtime, bf16_inverse = bf16.restore_source(
        runtime, lambda path, expected=None: read(path, inputs, expected))
    need(bf16.SOURCE == mul.PARENT_SOURCE and sha(runtime) == bf16.PARENT_SOURCE,
         'exact BF16 copy correction inverted before the original740 source witness')
    predicate_path = Path(repository) / 'tools/current_runtime/predicate_source_transition.py'
    predicate = loaded(predicate_path, read(predicate_path, inputs, PREDICATE_INVERSE),
                       'generator131_predicate_source_inverse')
    runtime, predicate_inverse = predicate.restore_source(
        runtime, lambda path, expected=None: read(path, inputs, expected))
    need(predicate.SOURCE == bf16.PARENT_SOURCE and predicate.PARENT_SOURCE == HALF_SOURCE
         and sha(runtime) == HALF_SOURCE,
         'the exact predicate routing and four helpers precede the unchanged Half witness')
    extension_path = Path(repository) / 'tools/current_runtime/half_logical_source_transition.py'
    extension = loaded(extension_path, read(extension_path, inputs, HALF_EXTENSION),
                       'generator131_half_logical_source_inverse')
    runtime, half_inverse = extension.restore_source(
        runtime, lambda path, expected=None: read(path, inputs, expected))
    need(extension.SOURCE == HALF_SOURCE and extension.PARENT_SOURCE == STORAGE_SOURCE
         and sha(runtime) == STORAGE_SOURCE,
         'the exact three Half edits are inverted before the unchanged storage736 witness')
    header, current = definitions(runtime, 736, complete=True)
    need(header.decode() == witness['current_header'] and list(current) == witness['current_order'],
         'exact physical declaration order and aggregate header')
    order = witness['parent_order']
    changed, removed, added = (witness[k] for k in ('changed', 'removed', 'added'))
    need(len(order) == len(set(order)) == 712
         and len(changed) == 22 and len(removed) == 7 and len(added) == 31
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
    need(header + b''.join(forward.values()) == runtime, 'complete736 source round trip')
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
        schema='heptatorch.generator131.physical775-parent712-inverse.v1',
        physical_source_sha256=SOURCE, physical_native_sha256=IMAGE,
        half_logical_source_inverse=half_inverse,
        predicate_source_inverse=predicate_inverse,
        bf16_source_inverse=bf16_inverse,
        half_mul_source_inverse=mul_inverse,
        half_neg_source_inverse=neg_inverse,
        half_addsub_source_inverse=addsub_inverse,
        wide_neg_source_inverse=wide_inverse,
        half_div_source_inverse=div_inverse,
        public_f64_arithmetic_source_inverse=arithmetic_inverse,
        typed_relu_source_inverse=relu_inverse,
        index_signbit_source_inverse=index_inverse,
        same_f64_arithmetic_source_inverse=same_inverse,
        physical_functions=778, physical_units=15, parent_source_sha256=PARENT_SOURCE,
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
    qualification = loaded(path, read(path, inputs, QUALIFICATION), 'generator131_current736_qualification')
    need((qualification.SOURCE, qualification.IMAGE, qualification.FUNCTIONS, qualification.UNITS)
         == (SOURCE, IMAGE, 778, 15), 'shared admission has the exact physical775 profile')
    built = qualification.admit_build(build, inputs.bind, expected=build_sha)
    runtime, units, summary, manifest_raw = current_source(repository, inputs)
    return runtime, units, summary, manifest_raw, built, proof
