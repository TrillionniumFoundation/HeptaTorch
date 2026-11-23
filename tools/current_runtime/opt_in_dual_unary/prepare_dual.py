"""Cold-build inputs for exact #61 plus the disjoint #59 and #62 opt-in edits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import types

PINS = {
    'base_manifest': 'a5cf09f191db4887b4452001e0b72e75fffb2147f2476544cc3be49d8ffa2c85',
    'base_runtime': 'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c',
    'logical_transform': '523a46f5a3cb055a408619eaca2737bf84cbff3c1ba3ecde20e81d638ece806e',
    'abs_transform': 'a495bc95fb84046a1f5ce6aeb43baa6f410f007c7d690cf84d39c428bef0ba1e',
    'logical_runtime': 'e3fe2e56a812dec618716ae14f7b95b821d4dc3195caf4dd19ee54c96fe0b4ba',
    'abs_runtime': 'e79a4015cbe8c0e0f2976c9ff6be650551cad8d892fa876ac15073cb5a9bf8c5',
}
CHANGES = {
    'wide_neg_tensor': ('runtime800_06', 'abs'),
    'predicate_bits': ('runtime800_10', 'logical'),
    'entry': ('runtime800_12', 'logical'),
    'predicate_command': ('runtime800_12', 'logical'),
}
UNIT_PINS = {
    'runtime800_06': ('37c19cd3b86eef66118d17d751d65c9fb3cdc1c776ddfd12e7c0aec88112c7fd',
                      '91a4cb5a1bd45c0c3747c684427ece796a8ab60eab763f87189d7cd3caede87a'),
    'runtime800_10': ('f0a185d5669ef20ed9e59833acb5b20ab79b67edf83363e3ab73c5823b213821',
                      '82d34309904925334f6d4168ae72f4dde69ef45487fdd942d2a8dae5f8d5e45e'),
    'runtime800_12': ('8e8ca44f2e03fa8dc3b3cc968abd0369abc38b2a8e499ea6c55e18ef77bb8a14',
                      '588beef6ef4573603a57dc30113fe81573d915796ff5863827a8329be919e833'),
}
CANDIDATE_SOURCE = 'f4c43f037a3fb046425f7634215ec382aa5f8d6b0e7abe126ef64236adb5753a'
CANDIDATE_NATIVE = '5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def insist(value, why):
    if not value:
        raise ValueError(why)

def bound(path, expected):
    insist(not path.is_symlink() and path.is_file(), f'not an ordinary file: {path}')
    raw = path.read_bytes()
    insist(sha(raw) == expected, f'identity mismatch: {path}')
    return raw

def module(path, pin):
    m = types.ModuleType(pin)
    raw = bound(path, PINS[pin])
    m.__file__ = str(path)
    exec(compile(raw, m.__file__, 'exec'), m.__dict__)
    return m

def body(raw, name):
    pattern = rb'^fn '+name.encode()+rb'\([^\n]*\n.*?^end\n'
    values = re.findall(pattern, raw, re.M | re.S)
    insist(len(values) == 1, f'expected unique function body: {name}')
    return values[0]

def overlay(raw, changes):
    for left, right in changes.values():
        insist(raw.count(left) == 1 and right not in raw, 'non-unique byte overlay')
        raw = raw.replace(left, right, 1)
    return raw

def inverse(changes):
    return {name:(right,left) for name,(left,right) in changes.items()}

def replace_exact(raw, old, new):
    insist(raw.count(old)==1, 'unit layout anchor not unique')
    return raw.replace(old,new,1)

def move_truth_before_predicate(raw, forward):
    truth = body(raw, 'logical_truth')
    pred = body(raw, 'predicate_bits')
    last = body(raw, 'logical_reduce')
    if forward:
        raw = replace_exact(raw, truth, b'')
        raw = replace_exact(raw, pred, truth+pred)
    else:
        raw = replace_exact(raw, truth, b'')
        raw = replace_exact(raw, last, truth+last)
    entries = re.search(rb'^exports\(([^\n]+)\)$',raw,re.M)
    names = entries[1].split(b',')
    names.remove(b'logical_truth')
    target = b'predicate_bits' if forward else b'logical_reduce'
    names.insert(names.index(target),b'logical_truth')
    return raw[:entries.start(1)] + b','.join(names) + raw[entries.end(1):]

def serialize(value):
    return (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()

def prepare(base, logical_path, abs_path, dest):
    insist(not (base/'units').is_symlink(), 'wrong units path')
    manifest = json.loads(bound(base/'manifest.json', PINS['base_manifest']))
    canonical = bound(base/'runtime.hepta', PINS['base_runtime'])
    logical = module(logical_path, 'logical_transform')
    abs_mod = module(abs_path, 'abs_transform')
    logical_only = logical.apply_native(canonical)
    abs_before = body(canonical, 'wide_neg_tensor')
    abs_after = abs_mod.modify(abs_before)
    abs_only = overlay(canonical, {'wide_neg_tensor': (abs_before, abs_after)})
    insist(sha(logical_only)==PINS['logical_runtime'] and sha(abs_only)==PINS['abs_runtime'],
           'standalone transform did not recover pinned candidate bytes')
    branches = {'abs': {}, 'logical': {}}
    variants = {'abs': abs_only, 'logical': logical_only}
    for name, (_, axis) in CHANGES.items():
        before, after = body(canonical,name), body(variants[axis],name)
        insist(before!=after and body(variants['logical' if axis=='abs' else 'abs'],name)==before,
               f'operators overlap or do not change {name}')
        branches[axis][name] = (before, after)
    for axis, paths in branches.items():
        insist(overlay(canonical, paths)==variants[axis] and
               overlay(variants[axis], inverse(paths))==canonical, 'standalone inverse/subset mismatch')
    both = {**branches['logical'], **branches['abs']}
    final = overlay(canonical,both)
    insist(final==overlay(abs_only,branches['logical'])==overlay(logical_only,branches['abs']) and
           overlay(final,inverse(both))==canonical and
           overlay(final,inverse(branches['logical']))==abs_only and
           overlay(final,inverse(branches['abs']))==logical_only, 'aggregate inverse/subset mismatch')
    insist(sha(final)==CANDIDATE_SOURCE, 'composed source pin differs')
    roster = ['rng03', *(f'runtime800_{i:02d}' for i in range(13)), 'owners']
    insist([entry['name'] for entry in manifest['units']]==roster, 'wrong exact unit roster')
    units = {}
    identities = []
    for entry in manifest['units']:
        name = entry['name']
        path = f'units/{name}.hepta'
        insist(entry['path']==path, 'wrong exact unit path')
        original = bound(base/path, entry['sha256'])
        changes = {n:v for n,v in both.items() if CHANGES[n][0]==name}
        result = overlay(original,changes)
        if name=='runtime800_10':
            result=move_truth_before_predicate(result,True)
        undone=move_truth_before_predicate(result,False) if name=='runtime800_10' else result
        insist(overlay(undone,inverse(changes))==original, 'partition inverse mismatch')
        if changes:
            new_source, new_object = UNIT_PINS[name]
            insist(sha(result)==new_source, 'partition did not reproduce standalone pinned input: '+name+' '+sha(result))
            entry.update(sha256=new_source, expected_object_sha256=new_object)
        units[path] = result
        identities.append({'name':name,'source_sha256':entry['sha256'],
                           'native_object_sha256':entry['expected_object_sha256'],
                           'changed_functions':sorted(changes)})
    manifest['name'] = 'current800_opt_in_abs_logical_not'
    manifest['expected_native_sha256'] = CANDIDATE_NATIVE
    manifest['runtime'].update(sha256=sha(final), instructions=final.count(b' = '),
                               blocks=len(re.findall(rb'^block ',final,re.M)))
    manifest_raw=serialize(manifest)
    dest.mkdir()
    (dest/'units').mkdir()
    for path, raw in units.items():
        (dest/path).write_bytes(raw)
    (dest/'manifest.json').write_bytes(manifest_raw)
    (dest/'runtime.hepta').write_bytes(final)
    return {'source_sha256':sha(final),'manifest_sha256':sha(manifest_raw),
            'all_exact_inverses':True,'composition_order_independent':True,'units':identities}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('base-source-root','logical-transform','abs-transform','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    root=a.base_source_root.resolve(strict=True)
    insist(not a.output.is_symlink() and not a.output.resolve().is_relative_to(root),
           'output must be outside the production source')
    print(serialize(prepare(root,a.logical_transform,a.abs_transform,a.output)).decode(),end='')
