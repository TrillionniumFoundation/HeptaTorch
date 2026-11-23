"""Exact opt-in F32 rank-1 dense graph merge for a bounded incoming edge profile."""
import argparse,json,re
from hashlib import sha256
from pathlib import Path
BASE_MANIFEST='1fd4916426f8a0d9cbd8598bcdd333ac14ae4f1c5f7fb73ac0b934e812f2a2f6'
BASE_SOURCE='e5944de78270c709ba6152363cdc2d0b9b0ac09e03aa1a35851d17c084bde3c3'
BASE_ELF='9074399826154287c2a97bd7172d6d86562eb5dfe1f5ab641ac33d6a03fb59de'
PINS={'elf': '1f0c6efc64e96fe25106acee7bf77a2ef86dd3d5b129b55bb773f5e5ff46e768', 'runtime800_02': 'c2e05adb00cde735ca9cc17d635a4a1c57e8164a5f0df0d32de057b218dde261'}
OUTPUT_MANIFEST='67da40febbaa2e743ccb1ed95765c50dc1650178aaf0cd533e904ad86b5cf279'
NAME='incoming_merge_edge'
UNIT='runtime800_02'
def sha(b):return sha256(b).hexdigest()
def req(ok,msg):
    if not ok:raise ValueError(msg)
def bound(p,mark):
    req(p.is_file() and not p.is_symlink(),'non-ordinary source')
    r=p.read_bytes();req(sha(r)==mark,'wrong source pin '+p.name);return r
def splice(a,b,c):
    req(a.count(b)==1 and c not in a,'incorrect overlay')
    return a.replace(b,c,1)

def edit(raw):
    s=raw.decode()
    guard='''  source_desc:u64 = call(v0,v1) {"function":"dptr"}
  tag_offset:u64 = const() {"value":40}
  source_tag_addr:u64 = int.add.wrap(source_desc,tag_offset)
  target_tag_addr:u64 = int.add.wrap(v5,tag_offset)
  source_tag:u64 = call(v0,source_tag_addr) {"function":"getw"}
  target_tag:u64 = call(v0,target_tag_addr) {"function":"getw"}
  clone_tag:u64 = const() {"value":17}
  abs_tag:u64 = const() {"value":201}
  clone_edge:bool = compare.eq(source_tag,clone_tag)
  abs_edge:bool = compare.eq(source_tag,abs_tag)
  unary_edge:bool = bool.or(clone_edge,abs_edge)
  leaf:bool = compare.eq(target_tag,v7)
  type_from:u64 = call(v0,v1) {"function":"tensor_dtype"}
  type_to:u64 = call(v0,v2) {"function":"tensor_dtype"}
  real_from:bool = compare.eq(type_from,v7)
  real_to:bool = compare.eq(type_to,v7)
  one:u64 = const() {"value":1}
  source_rank:u64 = call(v0,source_desc) {"function":"getw"}
  target_rank:u64 = call(v0,v5) {"function":"getw"}
  source_rank_one:bool = compare.eq(source_rank,one)
  target_rank_one:bool = compare.eq(target_rank,one)
  stride0:u64 = const() {"value":128}
  source_stride_addr:u64 = int.add.wrap(source_desc,stride0)
  target_stride_addr:u64 = int.add.wrap(v5,stride0)
  source_stride:u64 = call(v0,source_stride_addr) {"function":"getw"}
  target_stride:u64 = call(v0,target_stride_addr) {"function":"getw"}
  dense_from:bool = compare.eq(source_stride,one)
  dense_to:bool = compare.eq(target_stride,one)
  extent_offset:u64 = const() {"value":8}
  source_extent_addr:u64 = int.add.wrap(source_desc,extent_offset)
  target_extent_addr:u64 = int.add.wrap(v5,extent_offset)
  source_extent:u64 = call(v0,source_extent_addr) {"function":"getw"}
  target_extent:u64 = call(v0,target_extent_addr) {"function":"getw"}
  shape_match:bool = compare.eq(source_extent,target_extent)
  existing_stride_offset:u64 = const() {"value":496}
  existing_stride_addr:u64 = int.add.wrap(v5,existing_stride_offset)
  existing_stride:u64 = call(v0,existing_stride_addr) {"function":"getw"}
  described_stride_addr:u64 = const() {"value":3856}
  described_stride:u64 = call(v0,described_stride_addr) {"function":"getw"}
  existing_contiguous:bool = compare.eq(existing_stride,one)
  described_contiguous:bool = compare.eq(described_stride,one)
  direct_parent:u64 = call(v0,v1,v7) {"function":"gradient_parent_at"}
  exact_edge:bool = compare.eq(direct_parent,v2)
  scope1:bool = bool.and(unary_edge,leaf)
  scope2:bool = bool.and(real_from,real_to)
  scope3:bool = bool.and(source_rank_one,target_rank_one)
  scope4:bool = bool.and(dense_from,dense_to)
  scope5:bool = bool.and(existing_contiguous,described_contiguous)
  scope6:bool = bool.and(shape_match,exact_edge)
  scope7:bool = bool.and(scope1,scope2)
  scope8:bool = bool.and(scope3,scope4)
  scope9:bool = bool.and(scope5,scope6)
  scope10:bool = bool.and(scope7,scope8)
  merge_scope:bool = bool.and(scope9,scope10)
'''
    rule='''  old_sum:u64 = call(v141,v126) {"function":"addf"}
  merge_mag:u64 = const() {"value":2147483647}
  merge_inf:u64 = const() {"value":2139095040}
  merge_quiet:u64 = const() {"value":4194304}
  merge_indef:u64 = const() {"value":4290772992}
  prev_mag:u64 = bit.and(v141,merge_mag)
  next_mag:u64 = bit.and(v126,merge_mag)
  prev_nan:bool = compare.gt(prev_mag,merge_inf)
  next_nan:bool = compare.gt(next_mag,merge_inf)
  sole:bool = compare.ne(prev_nan,next_nan)
  words_equal:bool = compare.eq(v141,v126)
  same_raw_nan:bool = bool.and(prev_nan,words_equal)
  preserve_nan:bool = bool.or(sole,same_raw_nan)
  prev_inf:bool = compare.eq(prev_mag,merge_inf)
  next_inf:bool = compare.eq(next_mag,merge_inf)
  both_inf:bool = bool.and(prev_inf,next_inf)
  different:bool = compare.ne(v141,v126)
  opposite_inf:bool = bool.and(both_inf,different)
  picked:u64 = select(prev_nan,v141,v126)
  quieted:u64 = bit.or(picked,merge_quiet)
  defined_here:bool = bool.or(preserve_nan,opposite_inf)
  allow:bool = bool.and(merge_b11,defined_here)
  profiled:u64 = select(preserve_nan,quieted,merge_indef)
  v142:u64 = select(allow,profiled,old_sum)
'''
    s=s.replace('  branch v8 b1(v0) b2(v0,v2,v4,v5)',guard+'  branch v8 b1(v0) b2(v0,v2,v4,v5)')
    s=s.replace('  v142:u64 = call(v141,v126) {"function":"addf"}\n',rule)
    carry={'b2','b3','b7','b8','b9','b11','b13','b14','b15','b16','b18'}
    lines=[];block=''
    for l in s.splitlines():
        if l.startswith('block '):
            block=l.split('(',1)[0].split()[1]
            if block in carry:l=l[:-1]+',merge_'+block+':bool)'
        elif re.match(r'  (branch|jump) ',l):
            here='merge_scope' if block=='b0' else 'merge_'+block
            for b in carry:
                if re.search(r'\b'+b+r'\(',l):
                    req(block=='b0' or block in carry,'unbound incoming guard')
                    l=re.sub(r'\b'+b+r'\(([^)]*)\)',lambda m:f'{b}({m[1]},{here})',l)
        lines.append(l)
    return ('\n'.join(lines)+'\n').encode()

def derive(base,dest):
    pins=PINS
    m=json.loads(bound(base/'manifest.json',BASE_MANIFEST));r=bound(base/'runtime.hepta',BASE_SOURCE)
    req(m['expected_native_sha256']==BASE_ELF and not dest.resolve().is_relative_to(base.resolve()),'bad pins/output')
    bs=re.findall(rb'^fn '+NAME.encode()+rb'\(.*?^end\n',r,re.M|re.S)
    req(len(bs)==1,'wrong body census'); old,=bs; new=edit(old)
    def apply(raw):
        n=splice(raw,old,new);req(splice(n,new,old)==raw,'non-exact inverse');return n
    final=apply(r);data={}
    for u in m['units']:
        path=u['path'];buf=bound(base/path,u['sha256'])
        if u['name']==UNIT:
            buf=apply(buf);u['sha256']=sha(buf)
            u['expected_object_sha256']=pins[UNIT]
        data[path]=buf
    m['name']='current800_opt_in_f32_clone_abs_leaf_merge_words'
    m['expected_native_sha256']=pins['elf']
    m['runtime'].update(sha256=sha(final),instructions=final.count(b' = '),blocks=len(re.findall(rb'^block ',final,re.M)))
    mr=(json.dumps(m,indent=2,sort_keys=True)+'\n').encode()
    req(sha(mr)==OUTPUT_MANIFEST,'candidate output manifest differs')
    dest.mkdir();(dest/'units').mkdir()
    for p,b in data.items():(dest/p).write_bytes(b)
    (dest/'runtime.hepta').write_bytes(final);(dest/'manifest.json').write_bytes(mr)
    return {'body':NAME,'partition':UNIT,'old_sha256':sha(old),'new_sha256':sha(new),
            'source_sha256':sha(final),'manifest_sha256':sha(mr)}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('dest',type=Path)
    a=p.parse_args()
    print(json.dumps(derive(a.base,a.dest),indent=2,sort_keys=True))
