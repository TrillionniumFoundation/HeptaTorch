"""Reversible projection over exact ctor-admitted F16 source. Source-only; no compiler."""
import hashlib,re
EXPECTED='29e3ef6794f1a670cabcc94de734a883738acf81c88adb0fd5f31b52782fdaea'
NAME='logical_out_permuted_f32_bool_i64_f64_subspan'
BEFORE=b'  p_dtype_offset:bool = bool.and(p_type_ok,p_offset_admissible)\n'
AFTER=b'  p_old_dtype_offset:bool = bool.and(p_type_ok,p_offset_admissible)\n  p_f16_tag:u64 = const() {"value":5}\n  p_f16:bool = compare.eq(p_dtype,p_f16_tag)\n  p_width2:bool = compare.eq(p_width,p_two)\n  p_f16_width2:bool = bool.and(p_f16,p_width2)\n  p_f16_zero_only:bool = bool.and(p_f16_width2,p_zero_offset)\n  p_dtype_offset:bool = bool.or(p_old_dtype_offset,p_f16_zero_only)\n'
def project(raw):
 p=rb'^fn '+NAME.encode()+rb'\([^\n]+\n.*?^end\n'
 hit=re.search(p,raw,re.M|re.S)
 assert hit and hit[0].count(BEFORE)==1 and AFTER not in raw
 old=hit[0];new=old.replace(BEFORE,AFTER)
 out=raw[:hit.start()]+new+raw[hit.end():]
 assert out.replace(new,old,1)==raw
 return out
