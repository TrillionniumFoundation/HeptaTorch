"""Literal public binary64 inputs and arguments, never expected outputs."""
import hashlib
import json
import math
import struct

MASK=(1<<64)-1
PUBLIC_DTYPE=4
OPERATIONS={'any':118,'all':119,'sum_bool':115}
Z=0
NZ=0x8000000000000000
TINY=1
NTINY=0x8000000000000001
ONE=0x3ff0000000000000
NEG_ONE=0xbff0000000000000
QNAN=0x7ff8000000000001
NQNAN=0xfff8000000001234
PINF=0x7ff0000000000000
NINF=0xfff0000000000000
SMALL=0x0170000000000000
NSMALL=0x8170000000000000
SPECIAL=(Z,NZ,TINY,NTINY,QNAN,NQNAN,PINF,NINF,SMALL,NSMALL)
MIXED=(Z,NZ,TINY,NTINY,QNAN,PINF,ONE,NEG_ONE)


def recipe(name,words,shape,axes,keepdim=False):
    words,shape=list(words),list(shape)
    assert len(shape)<=4 and all(type(d) is int and d>=0 for d in shape)
    assert len(words)==math.prod(shape) and all(type(w) is int and 0<=w<=MASK for w in words)
    assert axes is None or all(type(axis) is int for axis in axes)
    return dict(name=name,representation='public_float64',native_dtype=PUBLIC_DTYPE,
                words=words,shape=shape,axes=axes,keepdim=keepdim,requires_grad=False,
                construction='direct97_dense_target_shape',storage_offset=0)


RECIPES=(
 recipe('scalar_positive_zero',[Z],[],None),
 recipe('scalar_negative_zero',[NZ],[],None),
 recipe('scalar_minimum_subnormal',[TINY],[],None),
 recipe('scalar_negative_minimum_subnormal',[NTINY],[],None),
 recipe('scalar_nan_payload',[QNAN],[],None),
 recipe('scalar_negative_nan_payload',[NQNAN],[],None),
 recipe('scalar_positive_infinity',[PINF],[],None),
 recipe('scalar_negative_infinity',[NINF],[],None),
 recipe('scalar_binary64_normal_underflow32',[SMALL],[],None),
 recipe('scalar_negative_binary64_normal_underflow32',[NSMALL],[],None),
 recipe('scalar_zero_empty_axes_keep',[Z],[],[],True),
 recipe('scalar_negative_zero_axis_zero_keep',[NZ],[],[0],True),
 recipe('scalar_subnormal_axis_negative',[TINY],[],[-1]),
 recipe('scalar_one_empty_axes',[ONE],[],[]),
 recipe('scalar_positive_axis_out_of_range',[ONE],[],[1]),
 recipe('scalar_negative_axis_out_of_range',[ONE],[],[-2]),
 recipe('scalar_duplicate_normalized_axes',[ONE],[],[0,-1]),
 recipe('rank1_signed_zeros_all',[Z,NZ],[2],None),
 recipe('rank1_signed_zeros_empty_axes',[Z,NZ],[2],[]),
 recipe('rank1_special_all',SPECIAL,[10],None),
 recipe('rank1_special_axis_zero_keep',SPECIAL,[10],[0],True),
 recipe('rank1_special_negative_axis',SPECIAL,[10],[-1]),
 recipe('rank1_opposite_nonzero_values',[ONE,NEG_ONE],[2],None),
 recipe('rank1_opposite_binary64_subnormals',[TINY,NTINY],[2],None),
 recipe('rank1_nonzero_above_low32',[Z,0x0000000100000000],[2],None),
 recipe('rank1_all_nonzero',[TINY,NTINY,QNAN,PINF],[4],[0]),
 recipe('rank1_all_signed_zeros_keep',[Z,NZ,NZ,Z],[4],[0],True),
 recipe('rank2_all',MIXED,[2,4],None),
 recipe('rank2_empty_axes',MIXED,[2,4],[]),
 recipe('rank2_axis_zero',MIXED,[2,4],[0]),
 recipe('rank2_axis_one_keep',MIXED,[2,4],[1],True),
 recipe('rank2_negative_axis_keep',MIXED,[2,4],[-1],True),
 recipe('rank2_unsorted_axes',MIXED,[2,4],[1,0]),
 recipe('rank2_negative_axes_keep',MIXED,[2,4],[-1,-2],True),
 recipe('rank2_duplicate_axes',MIXED,[2,4],[0,0]),
 recipe('rank2_duplicate_normalized_axes',MIXED,[2,4],[0,-2]),
 recipe('rank2_positive_axis_out_of_range',MIXED,[2,4],[2]),
 recipe('rank2_negative_axis_out_of_range',MIXED,[2,4],[-3]),
 recipe('rank3_all_keep',MIXED,[2,2,2],None,True),
 recipe('rank3_empty_axes_keep',MIXED,[2,2,2],[],True),
 recipe('rank3_axis_zero',MIXED,[2,2,2],[0]),
 recipe('rank3_trailing_axes',MIXED,[2,2,2],[1,2]),
 recipe('rank3_unsorted_axes_keep',MIXED,[2,2,2],[2,0],True),
 recipe('rank3_negative_axes',MIXED,[2,2,2],[-3,-1]),
 recipe('rank4_all',MIXED,[2,1,2,2],None),
 recipe('rank4_empty_axes_keep',MIXED,[2,1,2,2],[],True),
 recipe('rank4_singleton_axis',MIXED,[2,1,2,2],[1]),
 recipe('rank4_unsorted_axes_keep',MIXED,[2,1,2,2],[3,0],True),
 recipe('rank4_all_explicit_axes',MIXED,[2,1,2,2],[0,1,2,3]),
 recipe('rank4_negative_axes_keep',MIXED,[2,1,2,2],[-1,-3],True),
 recipe('rank4_positive_axis_out_of_range',MIXED,[2,1,2,2],[4]),
 recipe('rank4_negative_axis_out_of_range',MIXED,[2,1,2,2],[-5]),
 recipe('empty_rank1_all',[],[0],None),
 recipe('empty_rank1_empty_axes_keep',[],[0],[],True),
 recipe('empty_rank1_axis_zero_keep',[],[0],[0],True),
 recipe('empty_rank2_reduce_zero_extent',[],[0,3],[0]),
 recipe('empty_rank2_empty_output',[],[0,3],[1]),
 recipe('empty_rank2_all_keep',[],[0,3],None,True),
 recipe('empty_rank2_empty_axes',[],[0,3],[]),
 recipe('empty_rank3_reduce_zero_extent',[],[2,0,3],[1]),
 recipe('empty_rank3_empty_output',[],[2,0,3],[0]),
 recipe('empty_rank3_all_keep',[],[2,0,3],None,True),
 recipe('empty_rank4_unsorted_axes_keep',[],[1,2,0,3],[2,0],True),
 recipe('empty_input_still_checks_invalid_axis',[],[0,3],[2]),
)
CASES=tuple(dict(spec,operation=op,name=op+'/'+spec['name']) for spec in RECIPES for op in OPERATIONS)
ALLOCATION_SPEC=recipe('allocation_empty_axes_copy',[NZ]*65536,[65536],[])
ALLOCATION_CASES=tuple(dict(ALLOCATION_SPEC,operation=op,name=op+'/'+ALLOCATION_SPEC['name'])
                       for op in ('any','all'))
BOUNDARY_OBSERVATIONS=tuple(dict(kind=kind,operation=op,name=kind+'/'+op)
    for kind in ('requires_grad_public_float64','as_strided_public_float64') for op in OPERATIONS)


def raw_input(spec):
    assert spec['representation']=='public_float64' and spec['native_dtype']==4
    return struct.pack('<'+'Q'*len(spec['words']),*spec['words'])


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


CASE_SHA256=hashlib.sha256(canonical(CASES)).hexdigest()
ALLOCATION_SHA256=hashlib.sha256(canonical(ALLOCATION_CASES)).hexdigest()
BOUNDARY_SHA256=hashlib.sha256(canonical(BOUNDARY_OBSERVATIONS)).hexdigest()
assert len(RECIPES)==64 and len(CASES)==192 and len({r['name'] for r in CASES})==192
