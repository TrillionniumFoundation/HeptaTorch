"""Recover exact mixed-F64773 and the retained failed same-F64 predecessor.

The successor admits same public F64 op1..4 and corrects multiplication's
single-element rank-broadcast strides after the existing axis permutation.
All 771 remaining function bodies,
arithmetic value kernels, ownership and protocol predicates are retained.
This module proves source identity only; passing needs actual execution.
"""
import hashlib
import json
from pathlib import Path
import re

SOURCE='846e592beaa7847383cee668bc33d25cd81e79052bd4612252e2f72a83d96d05'
FAILED_V1_SOURCE='7b49e0e54851e0e4f470e28226a7904e458f38dcd48390e418ddf5f5dd4b17f8'
PARENT_SOURCE='4722c53b248780534a846a472784a565c4454ab738db75d655a7801b6623f07f'
FUNCTIONS=PARENT_FUNCTIONS=773
CHANGED=['half_mul_priority','public_f64_binary']
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def require(ok,message):
    if not ok:raise ValueError(message)

def definitions(raw):
    matches=list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',raw,re.M|re.S))
    result={item[1].decode():item[0] for item in matches}
    require(len(matches)==len(result) and raw==raw.split(b'\n',1)[0]+b'\n'+b''.join(result.values()),
        'all complete canonical HIA definitions exactly once')
    return result

def restore_source(source,bind):
    bind(Path(__file__).resolve())
    delta=json.loads(bind(Path(__file__).with_name('same_f64_arithmetic_source_delta.json')))
    require(delta['schema']=='heptatorch.same-f64-arithmetic.source-delta.v2'
        and sha(source)==SOURCE and delta['source_sha256']==SOURCE
        and delta['parent_source_sha256']==PARENT_SOURCE and delta['failed_v1_source_sha256']==FAILED_V1_SOURCE
        and delta['functions']==FUNCTIONS and delta['parent_functions']==PARENT_FUNCTIONS,
        'exact actual same-F64 successor, original failed v1, and frozen mixed parent identities')
    require(delta['changed_functions']==CHANGED and delta['unchanged_parent_functions']==771
        and delta['appended_functions']==[],'two actual changed bodies and no new functions')
    old,new=delta['original_body'].encode(),delta['current_body'].encode()
    prior_priority,current_priority=delta['original_priority_body'].encode(),delta['current_priority_body'].encode()
    require(sha(old)==delta['original_body_sha256'] and sha(new)==delta['current_body_sha256']
        and sha(prior_priority)==delta['original_priority_body_sha256']
        and sha(current_priority)==delta['current_priority_body_sha256']
        and source.count(new)==source.count(current_priority)==1,
        'complete original and successor gate and multiplication-priority bodies')
    failed_v1=source.replace(current_priority,prior_priority,1)
    require(sha(failed_v1)==FAILED_V1_SOURCE and failed_v1.replace(prior_priority,current_priority,1)==source,
        'exact failed v1 source recovered without relabeling it as passing')
    previous=failed_v1.replace(new,old,1)
    require(sha(previous)==PARENT_SOURCE and previous.count(old)==previous.count(prior_priority)==1
        and previous.replace(old,new,1).replace(prior_priority,current_priority,1)==source,
        'complete original mixed773 source inverse and forward round trip')
    current,parent=definitions(source),definitions(previous)
    require(len(current)==len(parent)==773 and tuple(current)==tuple(parent)
        and [name for name in current if current[name]!=parent[name]]==CHANGED,
        'all771 other bodies and all773 definition positions unchanged')
    removed=delta['removed_guard'].encode();before=delta['prior_type_guard'].encode();after=delta['current_type_guard'].encode()
    require(old.count(removed)==old.count(before)==new.count(after)==1
        and old.replace(removed,b'',1).replace(before,after,1)==new,
        'only the different-dtype admission gate is removed')
    prior_prefix=delta['prior_priority_prefix'].encode();current_prefix=delta['current_priority_prefix'].encode()
    prior_coordinates=delta['prior_priority_coordinates'].encode();current_coordinates=delta['current_priority_coordinates'].encode()
    require(prior_priority.count(prior_prefix)==current_priority.count(current_prefix)==1
        and prior_priority.count(prior_coordinates)==current_priority.count(current_coordinates)==1
        and prior_priority.replace(prior_prefix,current_prefix,1).replace(prior_coordinates,current_coordinates,1)==current_priority,
        'all existing priority selection branches retained after the new geometry guard')
    renamed=prior_coordinates
    for old_name,new_name in (('scalar_right','src'),('result','yc'),('dtype','dc'),('inner','nc'),('ls','lsc'),('rs','rsc'),('s','sc')):
        renamed=re.sub(rb'\b'+old_name.encode()+rb'\b',new_name.encode(),renamed)
    require(renamed==current_coordinates,'existing coordinate operations only receive fresh SSA block parameter names')
    require(b'  types:bool = bool.and(anydouble,public)\n' in new
        and b'  lsmall:bool = compare.le(ld,two)\n' in new
        and b'  rsmall:bool = compare.le(rd,two)\n' in new
        and b'  lower:bool = compare.ge(operation,one)\n' in new
        and b'  upper:bool = compare.le(operation,four)\n' in new
        and b'  originalp:u64 = call(sp,op,lp,rp) {"function":"half_mul_previous_binary"}\n  ret(originalp)\n' in new,
        'same4/4 admits only op1..4; non-F64/private/unsupported requests retain exact fallback')
    require(b'  double:bool = compare.eq(dtype,four)\n' in current_prefix
        and b'  singleton:bool = compare.eq(count,one)\n' in current_prefix
        and b'  different_rank:bool = compare.ne(left_rank,right_rank)\n' in current_prefix
        and b'  broadcast_singleton:bool = bool.and(singleton,different_rank)\n' in current_prefix
        and b'  order_s:u64 = call(ss,yps,lps,rps) {"function":"integer_pointwise_axis_order"}\n' in current_prefix
        and b'  last_s:u64 = int.sub.wrap(rank_s,one_s)\n' in current_prefix
        and b'  shift_s:u64 = int.mul.wrap(last_s,eight_s)\n' in current_prefix
        and b'  packed_s:u64 = int.shr.checked(order_s,shift_s)\n' in current_prefix
        and b'  axis:u64 = bit.and(packed_s,mask_s)\n' in current_prefix
        and current_prefix.count(b'{"function":"dist_probability_stride"}')==2,
        'only F64 single-element unequal-rank multiplication uses the last reordered axis strides')
    require(all(current[name]==parent[name] for name in ('binary','half_addsub_tensor','half_mul_tensor',
        'half_div_tensor','half_mul_result_dtype','half_mul_cast','integer_i64_to_f64','half_mul_previous_binary',
        'half_mul_f64','half_mul_f32','half_mul_float_priority','half_mul_inner_stride','half_mul_inner_extent',
        'half_addsub_f64','half_div_f64','f64_binary','command','command_argument',
        'integer_pointwise_axis_order','integer_pointwise_compare_axes','dist_probability_stride')),
        'complete value arithmetic, promotion, AD, ownership and protocol bodies retained')
    return previous,dict(schema='heptatorch.same-f64-arithmetic.source-inverse.v2',source_sha256=SOURCE,
        parent_source_sha256=PARENT_SOURCE,failed_v1_source_sha256=FAILED_V1_SOURCE,
        functions=773,parent_functions=773,changed_functions=CHANGED,unchanged_parent_functions=771,appended_functions=[],
        actual_route_sha256=sha(new),parent_route_sha256=sha(old),
        actual_priority_sha256=sha(current_priority),parent_priority_sha256=sha(prior_priority),
        same_f64_op1_to4_route_changed=True,same_f64_old_fallback_preserved=False,
        same_f64_unsupported_operations_original_fallback=True,mixed_f64_arithmetic_route_unchanged=True,
        f64_singleton_unequal_rank_multiplication_priority_changed=True,
        singleton_multiplication_uses_last_reordered_axis=True,existing_integer_axis_order_helpers_unchanged=True,
        non_f64_routes_unchanged=True,Half_routes_unchanged=True,private_dtype3_original_fallback=True,
        complete_existing_noAD_guards_unchanged=True,all_arithmetic_value_kernel_bodies_unchanged=True,
        exact_failed_v1_source_recovered=True,whole_source_inverse_exact=True,whole_round_trip_exact=True,
        native_executed=False,Torch_executed=False,host_expected_numerical_values_computed=False)
