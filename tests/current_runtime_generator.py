"""Pure exact 712-to-701 source/support proof for the new Generator image."""
import hashlib
import json
from pathlib import Path
import re

SOURCE = 'fac3d413c6c0f6874d224cbea312108ee2c8baa027102d41890d85094a5faa38'
IMAGE = '5b7aac203d2c913d43781f0da45d40e9bac2037c2bc074ec30f8dd663710a805'
PARENT_SOURCE = '82360ce87dae6f9a004ef50be4f991b6f83905a83f4b17ca8d6634827b7bd313'
TRANSITION = '5b2661f5f21c8d3d3cae708eb4f4be6592010dcb89b1bdb07c325a1873454180'
MANIFEST = '708e4b642f630604cedb1d2cf566fefcfb3d9afd74a835f5a265d7e51f1ff4e5'
BUILDER = '9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
OLD_BUILDER = 'b9d039c365a8a3988e3add120e8cffa46a4e1297878f4cced30103953b030bfb'
LINKER = '761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e'
OLD_LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
RNG = '7c32981c916bc277e16221cce1c3dd2381ce0018642e21b20836d861cf585865'
OLD_RNG = 'a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3'
OWNERS = '867a27b3a86634ea6a873909eb8bc16d5d8027c98c7cb9340376fba51312ec86'
OLD_OWNERS = '9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884'
MODEL = '4ca8c5a0ff81cd82cf2ae325085e1ae0450dc8d53b2da8668726cc002e28990f'
OLD_MODEL = 'dbae4d899b8a962716b23119980fe5da22e1c269c4b3efc8fb162a9166eee3ec'
CG_FUNCTIONS = ('cg_state_valid', 'cg_stat', 'cg_find', 'cg_mutates', 'cg_payload',
                'cg_validate', 'cg_size', 'cg_reply', 'cg_read', 'cg_initialize', 'cg_apply')
FUNCTION = re.compile(rb'^fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n', re.M | re.S)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def definitions(raw, count, complete=False):
    rows = list(FUNCTION.finditer(raw))
    result = {m[1].decode(): m[0] for m in rows}
    need(len(rows) == len(result) == count, 'exact complete function denominator')
    if complete:
        need(raw[:rows[0].start()] + b''.join(result.values()) == raw,
             'all complete-source bytes belong to the header or an exact function')
    return result


def replace_once(raw, before, after):
    need(raw.count(before) == 1, 'one exact inverse replacement')
    return raw.replace(before, after, 1)


def prove(source, build_directory, evidence_directory, inputs):
    """Bind physical inputs, then compute separately labelled inverse bytes."""
    build, evidence = Path(build_directory), Path(evidence_directory)
    inputs.bind(Path(__file__).resolve())
    transition_raw = inputs.bind(evidence / 'generator-transition.json', TRANSITION)
    transition = json.loads(transition_raw)
    need(transition['schema'] == 'heptatorch.tensor-generator-native.source-transition.v1'
         and transition['candidate_source_sha256'] == SOURCE
         and transition['original_source_sha256'] == PARENT_SOURCE
         and transition['candidate_core_functions'] == 712
         and transition['original_core_functions'] == 701
         and transition['units'] == 15
         and transition['whole701_source_inverse_exact'] is True,
         'the exact independently reviewed Generator source transition')
    need(sha(source) == SOURCE, 'exact new complete source')
    need(inputs.bind(build / 'runtime.hepta', SOURCE) == source, 'actual build source matches input')
    manifest = json.loads(inputs.bind(build / 'manifest.json', MANIFEST))
    need(manifest['runtime']['sha256'] == SOURCE and manifest['runtime']['functions'] == 712,
         'exact new source manifest')
    names = ['rng03'] + ['cat_f32_%02d' % i for i in range(12)] + ['cg_model', 'owners']
    need([row['name'] for row in manifest['units']] == names, 'all fifteen source units in fixed order')
    units = {}
    for row in manifest['units']:
        units[row['name']] = inputs.bind(build / (row['name'] + '.hepta'), row['sha256'])
    current = definitions(source, 712, complete=True)
    need(tuple(current)[-11:] == CG_FUNCTIONS, 'only the eleven original Generator core functions appended')
    model = units['cg_model']
    need(sha(model) == MODEL and re.sub(rb'\bcg_putw\b', b'putw', model).count(b'cg_putw') == 0,
         'exact Generator model unit')
    original_model = re.sub(rb'\bcg_putw\b', b'putw', model)
    need(len(re.findall(rb'\bcg_putw\b', model)) == 29 and sha(original_model) == OLD_MODEL,
         'whole original Generator model inverse is raw-writer identifier binding only')
    model_bodies = definitions(model, 11)
    need(tuple(model_bodies) == CG_FUNCTIONS
         and all(current[name] == model_bodies[name] for name in CG_FUNCTIONS),
         'all added complete core bodies equal the actual fixed model unit')
    appended = b''.join(model_bodies.values())
    new_header = b'hia-asm.v1 hepta.hia.program.v8 tensor_generator_complete01\n'
    old_header = b'hia-asm.v1 hepta.hia.program.v8 sum_integral_f32_runtime\n'
    need(source.startswith(new_header) and source.endswith(appended), 'exact new header and appended model suffix')
    restored = old_header + source[len(new_header):-len(appended)]
    need(sha(restored) == PARENT_SOURCE, 'whole 712-to-701 inverse equals original source bytes')
    original = definitions(restored, 701, complete=True)
    need(tuple(current)[:701] == tuple(original)
         and all(current[name] == body for name, body in original.items()),
         'every original Tensor function remains byte-identical')
    matched = {}
    old_units = {row['name']: row['sha256'] for row in transition['old_core_units_byte_identical']}
    need(list(old_units) == names[1:-2], 'exact twelve original Tensor unit identities')
    for name in names[1:-2]:
        need(sha(units[name]) == old_units[name],
             'every original Tensor partition remains byte-identical')
        for fn, body in definitions(units[name], len(FUNCTION.findall(units[name]))).items():
            need(fn in original and fn not in matched and original[fn] == body,
                 'exact original Tensor partition body coverage')
            matched[fn] = name
    need(set(matched) == set(original), 'all 701 original bodies still covered by twelve exact units')

    rng = units['rng03']
    need(sha(rng) == RNG, 'exact new RNG support')
    rng_bodies = definitions(rng, 18)
    alias = rng_bodies['cg_putw']
    need(sha(alias) == '3763f0acdfefec6ce8216f8e24d766ff4b129bdf56c63e495dd447cae751f3fe'
         and rng.endswith(alias), 'exact new raw writer alias')
    old_rng = replace_once(rng[:-len(alias)],
        b'exports(rng_seed,rng_restore,rng_next,rng_next64,rng_unit32,rng_unit64,round_f16,round_bf16,cg_putw)\n',
        b'exports(rng_seed,rng_restore,rng_next,rng_next64,rng_unit32,rng_unit64)\n')
    need(sha(old_rng) == OLD_RNG and len(definitions(old_rng, 17)) == 17,
         'whole RNG support inverse preserves all original bodies')
    need(original['getr'] == rng_bodies['getr'] and original['putw'] != rng_bodies['putw'],
         'original raw-reader identity and raw/tagged-writer separation')

    owners = units['owners']
    need(sha(owners) == OWNERS, 'exact new four-argument owner support')
    owner_bodies = definitions(owners, 10)
    need(list(owner_bodies) == ['make_trial','step_impl','close','tensor_step','cg_error',
                              'generator_step','mux_request_kind','mux_refuse_generator','mux_refuse_tensor','step'],
         'exact owner function inventory')
    tail = b''.join(list(owner_bodies.values())[4:])
    need(owners.endswith(tail), 'exact appended Generator and dispatcher suffix')
    inverse_owners = owners[:-len(tail)]
    for name in ('cg_apply','cg_mutates','cg_payload','cg_read','cg_reply','cg_size','cg_validate','getr'):
        matches = list(re.finditer(rb'^fn ' + name.encode() + rb'\([^\n]* external\(\)\n', inverse_owners, re.M))
        need(len(matches) == 1, 'one exact additional owner import')
        inverse_owners = inverse_owners.replace(matches[0][0], b'', 1)
    inverse_owners = replace_once(inverse_owners,
        b'hia-asm.v1 hepta.hia.program.v8 tensor_generator_owners01\n',
        b'hia-asm.v1 hepta.hia.program.v8 owned_tensor_objects\n')
    inverse_owners = replace_once(inverse_owners, b'fn tensor_step(', b'fn step(')
    need(sha(inverse_owners) == OLD_OWNERS, 'whole original Tensor owners inverse is exact')
    signature = owner_bodies['step'].split(b'\n',1)[0]
    arguments = signature.split(b'(',1)[1].split(b')',1)[0].split(b',')
    need([argument.split(b':',1)[1] for argument in arguments] == [b'readbuf',b'mutbuf',b'buffer',b'readbuf'],
         'actual read-only fourth entropy argument')
    builder = inputs.bind(evidence / 'generator-build.py', BUILDER)
    inverse_builder = builder
    for new, old in ((LINKER,OLD_LINKER),(RNG,OLD_RNG),(OWNERS,OLD_OWNERS),('HIACGS01','HIASTKS1')):
        inverse_builder = replace_once(inverse_builder, new.encode(), old.encode())
    need(sha(inverse_builder) == OLD_BUILDER, 'entire builder inverse has only four literal contract updates')
    return restored, dict(
        schema='heptatorch.generator712.original701-source-inverse.v1',
        source_sha256=SOURCE, inverse_source_sha256=PARENT_SOURCE,
        functions=712, original_functions=701, unchanged_functions=701,
        added_functions=list(CG_FUNCTIONS), units=15, unchanged_tensor_units=12,
        whole_inverse_exact=True, complete_original_core_bodies_byte_identical=True,
        generator_model_whole_inverse_exact=True, original_rng17_bodies_unchanged=True,
        whole_original_tensor_owner_inverse_exact=True,
        support_sha256=dict(rng03=RNG,owners=OWNERS,cg_model=MODEL),
        builder_whole_four_literal_inverse_exact=True, builder_sha256=BUILDER,
        linker_sha256=LINKER, link_request_magic='HIACGS01',
        actual_entry_arguments=['readbuf','mutbuf','buffer','readbuf'],
        transition_sha256=TRANSITION, manifest_sha256=MANIFEST,
        physical_input_records_modified=False,
        inverse_bytes_labelled_separately=True,
        native_Torch_compiler_executed=False,
        runtime_qualification_transferred=False)
