"""Derive only tag201's bit-preserving backward variant from the exact #64 scratch source."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

BASE_MANIFEST='11f5061fc49a6c9b1bd949553f5e57cd11dd98477e08e634eeb15cc655b7f747'
BASE_SOURCE='562e877ac9d3de25f5f819901f2ab3156d194703c62bf5b34c153bf20bbcbd30'
BASE_ELF='f9408443b278cd4c5d2131845ae31b3ad0e03469ca5e6e3c69c031e31694fe46'
PINS={'elf': '3e427044155ae8787621a926425a1ab0911611c69cd2f3da7f08417dfa9d7774', 'runtime800_02': '9125dd5ff69fdf3808d585217bebc0486fe61b57b2523d171415ae5886ca4781', 'runtime800_03': '05757e38db81cb99e649f8a7b2b91903c148cdec35db53a48d954581cca9c0e2'}
OUTPUT_MANIFEST='e4ab4fe1b39550dd22f45c82fb2a9f16ed2a97f904a7042b132d312ee9f28185'
UNITS={'gradient_direct':'runtime800_02','back_unary':'runtime800_03'}

def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(raw): return sha256(raw).hexdigest()
def bound(path,digest):
    need(path.is_file() and not path.is_symlink(),'wrong input type: '+path.name)
    raw=path.read_bytes()
    need(sha(raw)==digest,'wrong input identity: '+path.name)
    return raw
def body(raw,name):
    m=re.findall(rb'^fn '+name.encode()+rb'\([^\n]*\n.*?^end\n',raw,re.M|re.S)
    need(len(m)==1,'incorrect body count: '+name)
    return m[0]
def once(raw,old,new):
    need(raw.count(old)==1 and (not new or raw.count(new)==0),'nonunique overlay')
    return raw.replace(old,new,1)
def edit(name,raw):
    src=raw.decode('ascii')
    if name=='gradient_direct':
        src=once(src,'  branch v11 b1(v0) b2(v0,v1,v2)', '''  exact_abs_tag:u64 = const() {"value":201}
  uses_abs:bool = compare.eq(v6,exact_abs_tag)
  tracks_first:bool = bool.or(v11,uses_abs)
  branch tracks_first b1(v0) b2(v0,v1,v2)''')
    elif name=='back_unary':
        src=once(src,'''  derivative:u64 = call(ga,sign) {"function":"mulf"}
  jump b8(sa,ya,pa,xa,ta,na,ia,derivative)''','''  word_mask:u64 = const() {"value":4294967295}
  word:u64 = bit.and(ga,word_mask)
  sign_mask:u64 = const() {"value":2147483648}
  mag_mask:u64 = const() {"value":2147483647}
  inf:u64 = const() {"value":2139095040}
  quiet:u64 = const() {"value":4194304}
  neg_indefinite:u64 = const() {"value":4290772992}
  gmag:u64 = bit.and(word,mag_mask)
  gsign:u64 = bit.and(word,sign_mask)
  is_nan:bool = compare.gt(gmag,inf)
  is_inf:bool = compare.eq(gmag,inf)
  zero_sgn:bool = compare.eq(sign,zero)
  sign_flip:u64 = bit.and(sign,sign_mask)
  signed_word:u64 = bit.xor(word,sign_flip)
  invalid_or_zero:u64 = select(is_inf,neg_indefinite,gsign)
  finite_product:u64 = select(zero_sgn,invalid_or_zero,signed_word)
  quiet_g:u64 = bit.or(word,quiet)
  derivative:u64 = select(is_nan,quiet_g,finite_product)
  jump b8(sa,ya,pa,xa,ta,na,ia,derivative)''')
    else:
        raise ValueError(name)
    return src.encode()

def derive(base,dest):
    pins=PINS
    need(not (base/'units').is_symlink() and not dest.resolve().is_relative_to(base.resolve()),'wrong scratch paths')
    m=json.loads(bound(base/'manifest.json',BASE_MANIFEST))
    need(m['expected_native_sha256']==BASE_ELF and m['runtime']['sha256']==BASE_SOURCE,'wrong pinned lineage')
    rt=bound(base/'runtime.hepta',BASE_SOURCE)
    deltas={name:(body(rt,name),edit(name,body(rt,name))) for name in UNITS}
    def apply(raw,names):
        result=raw
        for name in names: result=once(result,*deltas[name])
        inverse=result
        for name in reversed(names): inverse=once(inverse,*deltas[name][::-1])
        need(inverse==raw,'overlay is not reversible')
        return result
    final=apply(rt,list(deltas))
    need(re.findall(rb'^fn (\w+)',rt,re.M)==re.findall(rb'^fn (\w+)',final,re.M),'runtime ABI roster changed')
    name_to_unit=[]
    units={}
    for u in m['units']:
        path=u['path']; name=u['name']
        raw=bound(base/path,u['sha256'])
        names=[n for n,p in UNITS.items() if p==name]
        name_to_unit.extend(names)
        units[path]=apply(raw,names)
        if names:
            u['sha256']=sha(units[path])
            u['expected_object_sha256']=pins[name]
    need(sorted(name_to_unit)==sorted(UNITS),'partition census mismatch')
    m['name']='current800_opt_in_f32_abs_bits'
    m['expected_native_sha256']=pins['elf']
    m['runtime'].update(sha256=sha(final),instructions=final.count(b' = '),blocks=len(re.findall(rb'^block ',final,re.M)))
    final_manifest=(json.dumps(m,indent=2,sort_keys=True)+'\n').encode()
    need(sha(final_manifest)==OUTPUT_MANIFEST,'not the physically verified output manifest')
    dest.mkdir(); (dest/'units').mkdir()
    for p,b in units.items(): (dest/p).write_bytes(b)
    (dest/'runtime.hepta').write_bytes(final)
    (dest/'manifest.json').write_bytes(final_manifest)
    return {n:{'prior_sha256':sha(a),'candidate_sha256':sha(b),'partition':UNITS[n]} for n,(a,b) in deltas.items()}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base',type=Path);p.add_argument('dest',type=Path)
    a=p.parse_args()
    print(json.dumps(derive(a.base,a.dest),indent=2,sort_keys=True))
