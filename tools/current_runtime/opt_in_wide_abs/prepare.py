"""Prepare the isolated opt-in current800 I64/F64 Tensor abs source, not production."""
import argparse
import hashlib
import json
from pathlib import Path
import re
BASE_MANIFEST = 'a5cf09f191db4887b4452001e0b72e75fffb2147f2476544cc3be49d8ffa2c85'
BASE_SOURCE = 'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c'
CANDIDATE_SOURCE = 'e79a4015cbe8c0e0f2976c9ff6be650551cad8d892fa876ac15073cb5a9bf8c5'
CANDIDATE_OBJECT = '91a4cb5a1bd45c0c3747c684427ece796a8ab60eab763f87189d7cd3caede87a'
CANDIDATE_ELF = '2adc665f1b042145725f8f92ded85e3f7f051f293180d704287847b16a95e005'
BASE_ELF = '083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'not regular source: {path}')
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f'unexpected bound source: {path}')
    return raw

def one(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError(f'expected one anchor: {old!r}')
    return raw.replace(old, new, 1)

def modify(old):
    new = one(old,
        b'  branch negative admit(s,input) unsupported(s)\n',
        b'  abs_op:u64 = const() {"value":7}\n'
        b'  absolute:bool = compare.eq(operation,abs_op)\n'
        b'  admitted:bool = bool.or(negative,absolute)\n'
        b'  branch admitted admit(s,input,absolute) unsupported(s)\n')
    for block, name in (('admit','x2:u64'),('allocate','p4:u64'),('layout','y6:u64'),
                        ('prepare','y7:u64'),('loop','index8:u64'),('element','index9:u64')):
        m = re.search(rb'^block '+block.encode()+rb'\([^\n]*'+name.encode()+rb'\)',new,re.M)
        if not m:
            raise ValueError(f'missing block: {block}')
        new = one(new,m[0],m[0][:-1]+b',absolute_'+block.encode()+b':bool)')
    for before, after in (
        (b'allocate(s2,x2,p2)',b'allocate(s2,x2,p2,absolute_admit)'),
        (b'layout(s4,x4,p4,dtype4,result4)',b'layout(s4,x4,p4,dtype4,result4,absolute_allocate)'),
        (b'prepare(s6,x6,p6,d6,y6)',b'prepare(s6,x6,p6,d6,y6,absolute_layout)'),
        (b'loop(s7,x7,y7,count7,floating7,zero7)',b'loop(s7,x7,y7,count7,floating7,zero7,absolute_prepare)'),
        (b'element(s8,x8,y8,count8,floating8,index8)',b'element(s8,x8,y8,count8,floating8,index8,absolute_loop)'),
        (b'loop(s9,x9,y9,count9,floating9,next9)',b'loop(s9,x9,y9,count9,floating9,next9,absolute_element)'),
    ):
        new = one(new,before,after)
    return one(new,b'  value9:u64 = select(floating9,float9,integer9)\n',
               b'  old9:u64 = select(floating9,float9,integer9)\n'
               b'  abs_negative9:bool = compare.ge(word9,sign9)\n'
               b'  abs_i9:u64 = select(abs_negative9,integer9,word9)\n'
               b'  abs_mask9:u64 = const() {"value":9223372036854775807}\n'
               b'  abs_f9:u64 = bit.and(word9,abs_mask9)\n'
               b'  abs_word9:u64 = select(floating9,abs_f9,abs_i9)\n'
               b'  value9:u64 = select(absolute_element,abs_word9,old9)\n')

def prepare(base, out):
    manifest = json.loads(read(base / 'manifest.json', BASE_MANIFEST))
    if manifest['expected_native_sha256'] != BASE_ELF or manifest['runtime']['sha256'] != BASE_SOURCE:
        raise ValueError('baseline identity differs')
    runtime = read(base / 'runtime.hepta', BASE_SOURCE)
    original, = re.findall(rb'^fn wide_neg_tensor\([^\n]+\n.*?^end\n',runtime,re.M | re.S)
    replacement = modify(original)
    candidate = one(runtime,original,replacement)
    if sha(candidate) != CANDIDATE_SOURCE:
        raise ValueError('wrong opt-in source digest')
    expected_roster = ['rng03', *(f'runtime800_{i:02d}' for i in range(13)), 'owners']
    if [entry['name'] for entry in manifest['units']] != expected_roster or (base/'units').is_symlink():
        raise ValueError('unexpected unit roster')
    units = {}
    changed = []
    for entry in manifest['units']:
        relative = f'units/{entry["name"]}.hepta'
        if entry['path'] != relative:
            raise ValueError('unexpected unit path')
        raw = read(base / relative,entry['sha256'])
        if original in raw:
            raw = one(raw,original,replacement)
            entry['sha256'] = sha(raw)
            entry['expected_object_sha256'] = CANDIDATE_OBJECT
            changed.append(entry['name'])
        units[relative] = raw
    if changed != ['runtime800_06']:
        raise ValueError('expected exactly one changed partition')
    manifest['expected_native_sha256'] = CANDIDATE_ELF
    manifest['name'] = 'current800_opt_in_wide_abs'
    manifest['runtime']['sha256'] = sha(candidate)
    manifest['runtime']['instructions'] += sum(
        line.startswith(b'  ') for line in replacement.splitlines()) - sum(
        line.startswith(b'  ') for line in original.splitlines())
    out.mkdir()
    (out/'units').mkdir()
    for relative, raw in units.items():
        (out/relative).write_bytes(raw)
    (out/'runtime.hepta').write_bytes(candidate)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'changed':changed,'source_sha256':sha(candidate)}))

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base-source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); baseline=a.base_source_root.resolve(strict=True)
    if a.output.is_symlink() or a.output.resolve().is_relative_to(baseline):
        raise ValueError('output must be outside the production source')
    prepare(baseline,a.output)
