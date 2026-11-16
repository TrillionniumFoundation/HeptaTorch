"""Package deterministic HIA-only composition inputs for a frozen candidate.

This is source custody tooling, not a numerical implementation. Qualification
of the resulting native program remains separate and is never inferred here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
FUNCTION = re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)', re.M | re.S)
PARENTS = {
    'gradient': ('1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5',
                 'be18ab877f38a4027a527783db80a1418a137c7d6da91e5bc39e2af8c044cb6b', 355),
    'adam_distribution': ('5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873',
                          '3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22', 378),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def functions(raw):
    matches = FUNCTION.findall(raw)
    result = {name.decode(): body for body, name in matches}
    assert len(result) == len(matches)
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'gradient-build', 'adam-distribution-build', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    parents = {'gradient': a.gradient_build.resolve(), 'adam_distribution': a.adam_distribution_build.resolve()}
    candidate = a.candidate.resolve()
    build_raw = (candidate / 'build.json').read_bytes()
    build = json.loads(build_raw)
    runtime = (candidate / 'runtime.hepta').read_bytes()
    native = (candidate / 'tensor-gradient-adam-distributions-session').read_bytes()
    assert build['status'] == 'PASS'
    assert sha(runtime) == build['source_sha256'] and sha(native) == build['native_sha256']
    assert len(native) == build['native_bytes']
    assert build['descriptor_bytes'] == 576
    body_sources, parent_meta, input_pins = {}, {}, {}
    for name, directory in parents.items():
        source = (directory / 'runtime.hepta').read_bytes()
        receipt = json.loads((directory / 'build.json').read_bytes())
        source_pin, native_pin, count = PARENTS[name]
        assert sha(source) == source_pin
        source_field = 'source_sha256' if name == 'gradient' else 'runtime_source_sha256'
        native_field = 'native_sha256' if name == 'gradient' else 'executable_sha256'
        schemas = (['heptatorch.gradient-incoming-native-build.v1', 'heptatorch.gradient-objects-native-build.v1']
                   if name == 'gradient' else ['heptatorch.adam-distributions-composition-native-build.v1'])
        assert receipt['status'] == 'PASS' and receipt['schema'] in schemas
        assert receipt[source_field] == source_pin and receipt[native_field] == native_pin
        body_sources[name] = functions(source)
        assert len(body_sources[name]) == count
        parent_meta[name] = {'source_sha256': source_pin, 'native_sha256': native_pin, 'functions': count,
                             'receipt_source_field': source_field, 'receipt_native_field': native_field,
                             'accepted_receipt_schemas': schemas}
        for filename in ('runtime.hepta', 'rng03.hepta', 'owners.hepta', 'build.json'):
            input_pins[str(directory / filename)] = sha((directory / filename).read_bytes())
    exact = functions(runtime)
    assert len(exact) == build['functions']
    overlay, mapping = [], []
    for name, body in exact.items():
        origin = next((source for source in ('gradient', 'adam_distribution')
                       if body_sources[source].get(name) == body), 'overlay')
        if origin == 'overlay':
            overlay.append(body)
        item = {'name': name, 'origin': origin, 'sha256': sha(body)}
        for source in ('gradient', 'adam_distribution'):
            prior = body_sources[source].get(name)
            item[source + '_sha256'] = sha(prior) if prior is not None else None
        mapping.append(item)
    fragment = b'hia-asm.v1 hepta.hia.program.v8 gradient_adam_distributions_overlay\n' + b''.join(overlay)
    (out / 'overlay.hepta').write_bytes(fragment)
    write_json(out / 'functions.json', mapping)
    producer = (HERE / 'portable_build.py').read_bytes()
    (out / 'build.py').write_bytes(producer)
    profile = {
        'schema': 'heptatorch.gradient-adam-distributions-profile.v1',
        'status': 'NATIVE_BUILD_PINNED_EXECUTION_QUALIFICATION_SEPARATE',
        'parents': parent_meta, 'source_sha256': build['source_sha256'],
        'native_sha256': build['native_sha256'], 'native_bytes': build['native_bytes'],
        'functions': build['functions'], 'blocks': build['blocks'], 'instructions': build['instructions'],
        'state_format': build['state_format'], 'backing_format': build['backing_format'],
        'descriptor_bytes': 576, 'runtime_header': runtime.split(b'\n', 1)[0].decode() + '\n',
        'compiler_sha256': build['compiler_sha256'], 'linker_sha256': build['linker_sha256'],
        'rng_source_sha256': sha((candidate / 'rng03.hepta').read_bytes()),
        'owners_source_sha256': sha((candidate / 'owners.hepta').read_bytes()),
        'overlay_sha256': sha(fragment), 'overlay_functions': len(overlay),
        'functions_map_sha256': sha((out / 'functions.json').read_bytes()),
        'portable_builder_sha256': sha(producer),
        'unchanged_gradient_bodies': sum(body == body_sources['gradient'].get(name) for name, body in exact.items()),
        'unchanged_adam_distribution_bodies': sum(body == body_sources['adam_distribution'].get(name) for name, body in exact.items()),
        'all_distribution_bodies_unchanged': all(body == body_sources['adam_distribution'][name]
                                               for name, body in exact.items() if name.startswith('dist_')),
        'units': build['units'],
        'qualification_is_separate': True, 'full_rewrite_complete': False, 'pytorch_api_denominator': 2590,
    }
    assert profile['all_distribution_bodies_unchanged']
    for directory in parents.values():
        assert sha((directory / 'rng03.hepta').read_bytes()) == profile['rng_source_sha256']
        assert sha((directory / 'owners.hepta').read_bytes()) == profile['owners_source_sha256']
    for unit in build['units']:
        assert sha((candidate / (unit['name'] + '.hepta')).read_bytes()) == unit['source_sha256']
        assert sha((candidate / (unit['name'] + '.o')).read_bytes()) == unit['object_sha256']
        assert unit['compiler_exit'] == 0
    write_json(out / 'profile.json', profile)
    write_json(out / 'source-custody.json', {
        'schema': 'heptatorch.gradient-adam-distributions-packaging-source-custody.v1',
        'candidate_build_sha256': sha(build_raw), 'candidate_source_sha256': sha(runtime),
        'candidate_native_sha256': sha(native), 'parents': parent_meta,
        'developer_input_source_hashes': input_pins,
        'packaged_files': {name: sha((out / name).read_bytes())
                           for name in ('build.py', 'profile.json', 'functions.json', 'overlay.hepta')},
        'no_native_execution_during_packaging': True,
        'host_owns_no_Tensor_or_gradient_execution': True,
    })
    print(json.dumps({key: profile[key] for key in ('source_sha256', 'native_sha256', 'functions',
                                                  'overlay_functions', 'state_format', 'backing_format')}))


if __name__ == '__main__':
    main()
