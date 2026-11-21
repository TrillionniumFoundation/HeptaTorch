"""Exact whole native773 to frozen native771 source inverse, with closed routing."""
import hashlib
import json
from pathlib import Path
import re

SOURCE='4722c53b248780534a846a472784a565c4454ab738db75d655a7801b6623f07f'
PARENT_SOURCE='f534c77069abc15ba15879d44492aacbc5f3709f6a861a0f0b9e0e365232fca1'
FUNCTIONS=773
PARENT_FUNCTIONS=771
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def require(value,message):
    if not value:raise ValueError(message)

def definitions(raw):
    rows=list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',raw,re.M|re.S))
    result={row[1].decode():row[0] for row in rows}
    require(len(rows)==len(result) and raw==raw.split(b'\n',1)[0]+b'\n'+b''.join(result.values()),
        'unique complete canonical HIA source bodies')
    return result

def restore_source(source,bind):
    bind(Path(__file__).resolve())
    require(sha(source)==SOURCE,'exact current773 mixed public F64 source')
    delta=json.loads(bind(Path(__file__).with_name('public_f64_arithmetic_source_delta.json')))
    require(delta['source_sha256']==SOURCE and delta['parent_source_sha256']==PARENT_SOURCE
        and delta['functions']==FUNCTIONS and delta['parent_functions']==PARENT_FUNCTIONS
        and delta['changed_functions']==['binary','half_mul_cast']
        and delta['unchanged_parent_functions']==769
        and delta['appended_functions']==['integer_i64_to_f64','public_f64_binary'],
        'exact two-body mixed F64 extension inventory')
    current=definitions(source);require(len(current)==FUNCTIONS,'all current773 definitions')
    suffix=b''.join(current[name] for name in delta['appended_functions'])
    require(source.endswith(suffix) and len(suffix)==delta['appended_bytes']
        and all(sha(current[name])==delta['appended_bodies'][name] for name in delta['appended_functions']),
        'only the two declared appended whole native functions')
    previous=source[:-len(suffix)]
    for name in delta['changed_functions']:
        old=delta['original_bodies'][name].encode();new=current[name]
        require(sha(old)==delta['original_body_sha256'][name] and sha(new)==delta['current_body_sha256'][name]
            and previous.count(new)==1,'exact whole original and current modified body')
        previous=previous.replace(new,old,1)
    require(sha(previous)==PARENT_SOURCE,'every byte of original771 source restored')
    parent=definitions(previous)
    require(len(parent)==PARENT_FUNCTIONS and tuple(current)[:PARENT_FUNCTIONS]==tuple(parent),
        'all original771 definitions and order retained')
    require([name for name in parent if current[name]!=parent[name]]==delta['changed_functions'],
        'all769 other complete original bodies unchanged')
    forward=previous
    for name in delta['changed_functions']:forward=forward.replace(parent[name],current[name],1)
    require(forward+suffix==source,'complete771-to773 source forward round trip')
    before=delta['binary_route_before'].encode();after=delta['binary_route_after'].encode()
    require(current['binary'].count(after)==1
        and current['binary'].replace(after,before,1)==parent['binary'],
        'only the non-Half binary fallback callee changes; all original Half route bytes retained')
    cast_before=delta['cast_route_before'].encode();cast_after=delta['cast_route_after'].encode()
    cast_guard=delta['cast_guard'].encode();cast=current['half_mul_cast']
    require(cast.count(cast_guard)==1 and cast.count(cast_after)==1 and cast.endswith(cast_guard+b'end\n')
        and cast.replace(cast_guard,b'',1).replace(cast_after,cast_before,1)==parent['half_mul_cast'],
        'only the direct I64-to-F64 guard is inserted before the exact old cast route')
    route=current['public_f64_binary']
    require(route.count(b'{"function":"half_mul_previous_binary"}')==1
        and b'  originalp:u64 = call(sp,op,lp,rp) {"function":"half_mul_previous_binary"}\n  ret(originalp)\n' in route
        and b'  different:bool = compare.ne(ld,rd)\n' in route
        and b'  lsmall:bool = compare.le(ld,two)\n' in route
        and b'  rsmall:bool = compare.le(rd,two)\n' in route
        and b'  lower:bool = compare.ge(operation,one)\n' in route
        and b'  upper:bool = compare.le(operation,four)\n' in route,
        'closed mixed public dtype and op1..4 route with unchanged fallback arguments')
    integer=current['integer_i64_to_f64']
    expected=parent['integer_i64_to_f32'].replace(b'fn integer_i64_to_f32(',b'fn integer_i64_to_f64(',1)
    for old,new,count in [
        (b'  v3:u64 = const() {"value":31}\n',b'  v3:u64 = const() {"value":63}\n',1),
        (b'{"value":23}',b'{"value":52}',4),
        (b'{"value":16777216}',b'{"value":9007199254740992}',1),
        (b'{"value":127}',b'{"value":1023}',1),
        (b'{"value":8388607}',b'{"value":4503599627370495}',1)]:
        require(expected.count(old)==count,'exact original unsigned integer conversion instruction')
        expected=expected.replace(old,new)
    require(integer==expected,'entire original nearest-even integer algorithm with only F64 format constants')
    return previous,dict(schema='heptatorch.public-f64-arithmetic.source-inverse.v1',
        source_sha256=SOURCE,parent_source_sha256=PARENT_SOURCE,functions=FUNCTIONS,parent_functions=PARENT_FUNCTIONS,
        changed_functions=delta['changed_functions'],unchanged_parent_functions=769,
        appended_functions=delta['appended_functions'],
        actual_binary_sha256=sha(current['binary']),parent_binary_sha256=sha(parent['binary']),
        actual_cast_sha256=sha(cast),parent_cast_sha256=sha(parent['half_mul_cast']),
        actual_mixed_route_sha256=sha(route),direct_cast_guard_sha256=sha(cast_guard),
        integer_i64_to_f64_sha256=sha(integer),integer_conversion_derived_from_complete_parent_algorithm=True,
        direct_i64_to_f64_round_to_nearest_ties_to_even=True,intermediate_f32_conversion=False,
        preexisting_half_binary_routes_unchanged=True,same_f64_route_preserved=True,
        non_f64_routes_preserved=True,private_dtype3_delegated=True,unsupported_operations_delegated=True,
        original_arithmetic_kernel_bodies_unchanged=True,
        whole_source_inverse_exact=True,whole_round_trip_exact=True,
        native_executed=False,Torch_executed=False,host_expected_values_computed=False)
