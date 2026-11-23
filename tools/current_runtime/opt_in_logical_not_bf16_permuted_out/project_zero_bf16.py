"""Reversible projection over exact constructor-admitted BF16 source. No compiler."""
import hashlib,re
EXPECTED='abb362f523a09c74f3e9de6deb0327f4d1df984882d7da766bae7068dda3653a'
NAME='logical_out_permuted_f32_bool_i64_f64_subspan'
BEFORE=b'  p_dtype_offset:bool = bool.or(p_old_dtype_offset,p_f16_zero_only)\n'
AFTER=b'  p_old_or_f16_zero:bool = bool.or(p_old_dtype_offset,p_f16_zero_only)\n  p_bf16_tag:u64 = const() {"value":6}\n  p_bf16:bool = compare.eq(p_dtype,p_bf16_tag)\n  p_bf16_width2:bool = bool.and(p_bf16,p_width2)\n  p_bf16_zero_only:bool = bool.and(p_bf16_width2,p_zero_offset)\n  p_dtype_offset:bool = bool.or(p_old_or_f16_zero,p_bf16_zero_only)\n'
def project(raw):
 p=rb'^fn '+NAME.encode()+rb'\([^\n]+\n.*?^end\n'
 hit=re.search(p,raw,re.M|re.S)
 assert hit and hit[0].count(BEFORE)==1 and AFTER not in raw
 old=hit[0];new=old.replace(BEFORE,AFTER)
 out=raw[:hit.start()]+new+raw[hit.end():]
 assert out.replace(new,old,1)==raw
 return out
