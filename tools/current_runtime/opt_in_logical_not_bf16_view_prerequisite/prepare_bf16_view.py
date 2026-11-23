"""Source-only reversible projection: admit public BF16 whole-root dense op111."""
from pathlib import Path
from hashlib import sha256
import argparse, importlib.util, json, re

PARENT = '29e3ef6794f1a670cabcc94de734a883738acf81c88adb0fd5f31b52782fdaea'
UNIT_PARENT = 'c0f41fb1fe357626da7ee734a3945f9895badc8c21b3d83cbb77366b21cf774b'
BUILDER_SHA = '9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
UNIT = 'runtime800_11'
UNRELATED_LAYOUT = 'logical_out_permuted_f32_bool_i64_f64_subspan'
OLD = b'public_f16_whole_dense_view'
NEW = b'public_half_whole_dense_view'
PATCH = {
 b'to_command': (
 (b'  rejected_half:bool = bool.or(rejected_f16,v83)\n',
  b'  rejected_bf16:bool = bool.and(v83,other_f64_command)\n'
  b'  rejected_half:bool = bool.or(rejected_f16,rejected_bf16)\n'),
 ),
 b'to_strided_view': (
 (b'  branch direct_f16 f16_constructor(hr,hs,hh,hp,he,hi,hd) guarded_constructor(hr,hs,hh,hp,he,hi,hd)\n',
  b'  bf16_tag:u64 = const() {"value":6}\n'
  b'  direct_bf16:bool = compare.eq(hdt,bf16_tag)\n'
  b'  direct_half:bool = bool.or(direct_f16,direct_bf16)\n'
  b'  branch direct_half public_half_constructor(hr,hs,hh,hp,he,hi,hd) guarded_constructor(hr,hs,hh,hp,he,hi,hd)\n'),
 (b'block f16_constructor(', b'block public_half_constructor('),
 (b'{"function":"'+OLD+b'"}', b'{"function":"'+NEW+b'"}'),
 ),
 OLD: (
 (b'fn '+OLD+b'(', b'fn '+NEW+b'('),
 (b'  f16:u64 = const() {"value":5}\n', b'  f16:u64 = const() {"value":5}\n  bf16:u64 = const() {"value":6}\n'),
 (b'  dt_ok:bool = bool.and(is_f16,is_two)\n',
  b'  is_bf16:bool = compare.eq(dt,bf16)\n'
  b'  f16_width_ok:bool = bool.and(is_f16,is_two)\n'
  b'  bf16_width_ok:bool = bool.and(is_bf16,is_two)\n'
  b'  dt_ok:bool = bool.or(f16_width_ok,bf16_width_ok)\n'),
 ),
}

def sha(b): return sha256(b).hexdigest()

def swap(raw,a,b):
    assert raw.count(a)==1, a
    return raw.replace(a,b,1)

def project(raw, unit=False):
    original=raw
    inv=[]
    for name, body_edits in PATCH.items():
        old=re.search(rb'^fn '+name+rb'\([^\n]+\n.*?^end\n',raw,re.M|re.S)[0]
        new=old
        for a,b in body_edits:new=swap(new,a,b)
        raw=swap(raw,old,new)
        inv.append((new,old))
    if unit:
        old=raw.splitlines(keepends=True)[1]
        assert old.startswith(b'exports(') and old.endswith(b')\n')
        new=swap(old,OLD,NEW)
        raw=swap(raw,old,new)
        inv.append((new,old))
    reverse=raw
    for a,b in reversed(inv):reverse=swap(reverse,a,b)
    assert reverse==original
    return raw

def checked(digest, data):
    assert sha(data)==digest
    return data

def controls(build,data):
    aggregate=build.parse_source(data)
    for name in (b'to_command',b'to_strided_view',NEW):
        fn=aggregate['definitions'][name.decode()]
        def args(b):return len(b.split(b',')) if b else 0
        blocks={m[1]:args(m[2]) for m in re.finditer(rb'^block (\w+)\(([^)]*)\)$',fn,re.M)}
        for line in fn.splitlines():
            if line.startswith((b'  jump ',b'  branch ')):
                for target,aa in re.findall(rb'(\w+)\(([^)]*)\)',line):
                    assert blocks[target]==args(aa),(name,line)
            c=re.search(rb' = call\(([^)]*)\) \{"function":"(\w+)"\}',line)
            if c:assert len(aggregate['signatures'][c[2].decode()][0])==args(c[1]),(name,line)
    return aggregate

def read_source(src,build):
    manifest=json.loads((src/'manifest.json').read_bytes())
    runtime=checked(manifest['runtime']['sha256'],(src/manifest['runtime']['path']).read_bytes())
    units={Path(r['path']):checked(r['sha256'],(src/r['path']).read_bytes()) for r in manifest['units']}
    return manifest, runtime, units, build.parse_source(runtime)

def project_source(build,exact_src,composed_src=None):
    _, exact_raw, exact_units, exact=read_source(exact_src,build)
    assert sha(exact_raw)==PARENT
    src=composed_src or exact_src
    manifest,runtime,units,orig=read_source(src,build)
    unit_path=Path('units')/(UNIT+'.hepta')
    assert sha(units[unit_path])==UNIT_PARENT
    if composed_src:
        assert orig['signatures']==exact['signatures']
        assert {n for n,body in orig['definitions'].items() if body!=exact['definitions'][n]}=={UNRELATED_LAYOUT}
        assert set(exact_units)==set(units)
        assert {p for p in units if units[p]!=exact_units[p]}=={Path('units/runtime800_12.hepta')}
    new=project(runtime)
    ag=controls(build,new)
    assert ag['signatures'][NEW.decode()]==orig['signatures'][OLD.decode()]
    assert set(ag['definitions'])-set(orig['definitions'])=={NEW.decode()}
    assert set(orig['definitions'])-set(ag['definitions'])=={OLD.decode()}
    assert {n for n in set(ag['definitions'])&set(orig['definitions']) if ag['definitions'][n]!=orig['definitions'][n]}=={'to_command','to_strided_view'}
    assert all(orig['signatures'][n]==ag['signatures'][n] for n in orig['definitions'] if n!=OLD.decode())
    manifest.pop('expected_native_sha256',None)
    manifest['name']='opt_in_public_f16_bf16_whole_dense_view'
    manifest['runtime'].update(sha256=sha(new),**{k:ag[k] for k in ('functions','blocks','instructions')})
    new_unit=project(units[unit_path],unit=True)
    units[unit_path]=new_unit
    for row in manifest['units']:
        if row['name']==UNIT:
            row['sha256']=sha(new_unit)
            row.pop('expected_object_sha256',None)
    units[Path(manifest['runtime']['path'])]=new
    summary=build.check_sources(manifest,src,lambda path,digest:checked(digest,units[path.relative_to(src)]))[2]
    return manifest,units,summary

def load_builder(tool):
    assert sha(tool.read_bytes())==BUILDER_SHA
    spec=importlib.util.spec_from_file_location('pinned_builder',tool)
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    return b

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('builder',type=Path)
    p.add_argument('--compose',type=Path)
    p.add_argument('--output',type=Path,help='authorize source materialization only for later sole native build lane')
    a=p.parse_args()
    m,units,check=project_source(load_builder(a.builder),a.source,a.compose)
    if a.output:
        a.output.mkdir()
        (a.output/'units').mkdir()
        for path,data in units.items():(a.output/path).write_bytes(data)
        (a.output/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
    print(json.dumps(check,sort_keys=True))

if __name__=='__main__':main()
