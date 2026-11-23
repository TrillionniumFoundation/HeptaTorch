"""Exact static projection only; no compiler or runtime invocation."""
from pathlib import Path
from hashlib import sha256
import json,re,sys
OLD='2e29170811c8dc9dc4dfe9979da4ca6aadd638e7dfa1cd95fbb086bfdb77ed34'
NAME='logical_out_permuted_f32_bool_i64_f64'
NEW=NAME+'_subspan'
def digest(b):return sha256(b).hexdigest()
def project(inp):
 start=inp.index(f'fn {NAME}('.encode());stop=inp.index(b'\nend\n',start)+5
 old=inp[start:stop]; code=old.decode()
 def rep(a,b):
  nonlocal code
  assert code.count(a)==1,a
  code=code.replace(a,b,1)
 rep('  p_kill1:u64 = bit.or(p_edge,p_offset)','  p_kill1:u64 = bit.or(p_edge,p_z)')
 rep('  p_guard0:bool = bool.and(p_clean_view,p_type_ok)','''  p_zero_offset:bool = compare.eq(p_offset,p_z)
  p_aligned_rem:u64 = int.rem.checked(p_offset,p_four)
  p_aligned:bool = compare.eq(p_aligned_rem,p_z)
  p_f32_aligned:bool = bool.and(p_allow_f32,p_aligned)
  p_offset_admissible:bool = bool.or(p_zero_offset,p_f32_aligned)
  p_dtype_offset:bool = bool.and(p_type_ok,p_offset_admissible)
  p_guard0:bool = bool.and(p_clean_view,p_dtype_offset)''')
 rep('  v_ok:bool = bool.and(v_root_layout_ok,v_whole)\n  branch v_ok dimensions(v_s,v_node,v_desc,v_store,v_rank,v_count,v_z,v_z,v_one,v_width) bad(v_z)','''  branch v_root_layout_ok root_span(v_s,v_node,v_desc,v_store,v_rank,v_count,v_z,v_one,v_width,v_offset,v_root_count,v_whole) bad(v_z)
block root_span(k_s:mutbuf,k_node:u64,k_desc:u64,k_store:u64,k_rank:u64,k_count:u64,k_z:u64,k_one:u64,k_width:u64,k_offset:u64,k_root_count:u64,k_whole:bool)
  k_zero:bool = compare.eq(k_offset,k_z)
  branch k_zero whole_span(k_s,k_node,k_desc,k_store,k_rank,k_count,k_z,k_one,k_width,k_offset,k_whole) subspan(k_s,k_node,k_desc,k_store,k_rank,k_count,k_z,k_one,k_width,k_offset,k_root_count)
block whole_span(w_s:mutbuf,w_node:u64,w_desc:u64,w_store:u64,w_rank:u64,w_count:u64,w_z:u64,w_one:u64,w_width:u64,w_offset:u64,w_whole:bool)
  branch w_whole dimensions(w_s,w_node,w_desc,w_store,w_rank,w_count,w_z,w_z,w_one,w_width,w_offset) bad(w_z)
block subspan(j_s:mutbuf,j_node:u64,j_desc:u64,j_store:u64,j_rank:u64,j_count:u64,j_z:u64,j_one:u64,j_width:u64,j_offset:u64,j_root_count:u64)
  j_elem:u64 = int.div.checked(j_offset,j_width)
  j_offset_valid:bool = compare.le(j_elem,j_root_count)
  branch j_offset_valid subspan_room(j_s,j_node,j_desc,j_store,j_rank,j_count,j_z,j_one,j_width,j_offset,j_root_count,j_elem) bad(j_z)
block subspan_room(c_s:mutbuf,c_node:u64,c_desc:u64,c_store:u64,c_rank:u64,c_count:u64,c_z:u64,c_one:u64,c_width:u64,c_offset:u64,c_root_count:u64,c_elem:u64)
  c_room:u64 = int.sub.wrap(c_root_count,c_elem)
  c_fits:bool = compare.le(c_count,c_room)
  branch c_fits dimensions(c_s,c_node,c_desc,c_store,c_rank,c_count,c_z,c_z,c_one,c_width,c_offset) bad(c_z)''')
 for block,tag in [('root','r'),('root_layout','v'),('dimensions','d'),('axis','a'),('product','m'),('next','n'),('density','y'),('check_density','h'),('storage','s'),('extent','e')]:
  code,n=re.subn(fr'(?m)^(block {block}\([^\n]*?{tag}_width:u64)',fr'\1,{tag}_offset:u64',code);assert n==1
 # Each listed edge existed exactly once in the predecessor; root_span additions above already pass offsets.
 for dst,tag in [('root','p'),('root_layout','r'),('axis','d'),('density','d'),('product','a'),('next','m'),('dimensions','n'),('check_density','y'),('storage','h'),('extent','s')]:
  code,n=re.subn(fr'(?m)(\b{dst}\({tag}_s,[^()]*?{tag}_width)',fr'\1,{tag}_offset',code)
  assert n==1,(dst,tag,n)
 rep('branch e_base_ok bounded(e_base,e_size,e_start,e_count,e_z,e_width,e_tag) bad(e_z)','branch e_base_ok bounded(e_base,e_size,e_start,e_count,e_z,e_width,e_tag,e_offset) bad(e_z)')
 rep('b_z:u64,b_width:u64,b_tag:u64)','b_z:u64,b_width:u64,b_tag:u64,b_offset:u64)')
 rep('''  b_expected:u64 = int.add.wrap(b_tag,b_base)
  b_origin:bool = compare.eq(b_start,b_expected)
  b_tag_room:bool = compare.le(b_bytes,b_room)
  b_capacity:bool = compare.le(b_bytes,b_size)
  b_both:bool = bool.and(b_origin,b_tag_room)
  b_ok:bool = bool.and(b_both,b_capacity)
  branch b_ok good() bad(b_z)''','''  b_tag_base:u64 = int.add.wrap(b_tag,b_base)
  b_offset_tag:bool = compare.le(b_offset,b_room)
  b_offset_size:bool = compare.le(b_offset,b_size)
  b_offset_ok:bool = bool.and(b_offset_tag,b_offset_size)
  branch b_offset_ok remaining(b_tag_base,b_start,b_bytes,b_room,b_size,b_offset,b_z) bad(b_z)
block remaining(t_base:u64,t_start:u64,t_bytes:u64,t_tag_room:u64,t_size:u64,t_offset:u64,t_z:u64)
  t_tag_remaining:u64 = int.sub.wrap(t_tag_room,t_offset)
  t_storage_remaining:u64 = int.sub.wrap(t_size,t_offset)
  t_tag_fits:bool = compare.le(t_bytes,t_tag_remaining)
  t_storage_fits:bool = compare.le(t_bytes,t_storage_remaining)
  t_fits:bool = bool.and(t_tag_fits,t_storage_fits)
  branch t_fits exact_start(t_base,t_start,t_offset,t_z) bad(t_z)
block exact_start(x_base:u64,x_start:u64,x_offset:u64,x_z:u64)
  x_expected:u64 = int.add.wrap(x_base,x_offset)
  x_origin:bool = compare.eq(x_start,x_expected)
  branch x_origin good() bad(x_z)''')
 new=code.encode()
 output=(inp[:start]+new+inp[stop:]).replace(NAME.encode(),NEW.encode())
 assert output.replace(NEW.encode(),NAME.encode()).replace(new,old,1)==inp
 return output,old,new.replace(NAME.encode(),NEW.encode())
if __name__=='__main__':
 root,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
 mon=(root/'runtime.hepta').read_bytes(); assert digest(mon)==OLD
 unit=(root/'units/runtime800_12.hepta').read_bytes()
 manifest=json.loads((root/'manifest.json').read_bytes())
 pinned,=[r['sha256'] for r in manifest['units'] if r['name']=='runtime800_12']
 assert digest(unit)==pinned
 new,_,body=project(mon); new_unit,_,copy=project(unit);assert copy==body
 (out/'helper.hepta').write_bytes(body)
 (out/'fingerprint.json').write_text(json.dumps(dict(base_commit='c16f10cb76944a07a965a5cc146e86be22dbdb66', base_source_sha256=OLD,base_unit_sha256=pinned,projected_source_sha256=digest(new),projected_unit_sha256=digest(new_unit),projected_helper_sha256=digest(body),execution='source-only; no compile or native test'),sort_keys=True,indent=2)+'\n')
