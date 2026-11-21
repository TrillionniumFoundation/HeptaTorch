"""Recover the exact same-F64 parent before native typed ReLU.

Only command3/operation5 for no-grad I64/F64/F16/BF16 gains a route. This
source proof neither executes the application nor transfers a prior PASS.
"""
import hashlib
import json
from pathlib import Path
import re

SOURCE = '0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43'
PARENT_SOURCE = '846e592beaa7847383cee668bc33d25cd81e79052bd4612252e2f72a83d96d05'
FUNCTIONS, PARENT_FUNCTIONS = 775, 773
CHANGED = ('unary',)
ADDED = ('typed_relu_value', 'typed_relu_tensor')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def functions(raw, count):
    matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', raw, re.M | re.S))
    bodies = {match[1].decode('ascii'): match[0] for match in matches}
    require(len(bodies) == len(matches) == count, 'complete unique typed-ReLU function census')
    require(raw == raw.split(b'\n', 1)[0] + b'\n' + b''.join(bodies.values()),
            'complete canonical source envelope and definition order')
    return bodies


def restore_source(source, bind):
    bind(Path(__file__).resolve())
    delta_raw = bind(Path(__file__).with_name('typed_relu_source_delta.json'))
    delta = json.loads(delta_raw)
    require(isinstance(source, bytes) and sha(source) == SOURCE
            and delta['schema'] == 'heptatorch.typed-relu.source-delta.v1'
            and delta['source_sha256'] == SOURCE and delta['parent_source_sha256'] == PARENT_SOURCE
            and delta['functions'] == FUNCTIONS and delta['parent_functions'] == PARENT_FUNCTIONS
            and tuple(delta['changed_functions']) == CHANGED and tuple(delta['appended_functions']) == ADDED
            and delta['unchanged_parent_functions'] == 772,
            'exact775 successor,773 parent, one changed and two appended functions')
    current = functions(source, FUNCTIONS)
    require(tuple(current)[-2:] == ADDED, 'exact final two typed-ReLU helpers')
    suffix = b''.join(current[name] for name in ADDED)
    require(source.endswith(suffix) and len(suffix) == delta['helper_bytes']
            and sha(suffix) == delta['helper_sha256'], 'complete appended helper bytes')
    for name in ADDED:
        row = delta['added_bodies'][name]
        require(row == dict(bytes=len(current[name]), sha256=sha(current[name])),
                'exact complete new helper: ' + name)
    before = delta['original_unary'].encode('utf-8')
    after = current['unary']
    require(sha(before) == delta['original_unary_sha256'] and sha(after) == delta['current_unary_sha256']
            and source.count(after) == 1 and before.startswith(b'fn unary(') and before.endswith(b'end\n'),
            'complete original and current unary bodies')
    restored = source[:-len(suffix)].replace(after, before, 1)
    parent = functions(restored, PARENT_FUNCTIONS)
    require(sha(restored) == PARENT_SOURCE and tuple(current)[:PARENT_FUNCTIONS] == tuple(parent)
            and [name for name in parent if current[name] != parent[name]] == list(CHANGED)
            and sum(current[name] == parent[name] for name in parent) == 772,
            'every complete parent byte and all772 unchanged function bodies')
    old_route = delta['original_route'].encode()
    new_route = delta['current_route'].encode()
    require(before.count(old_route) == after.count(new_route) == 1
            and before.replace(old_route, new_route, 1) == after,
            'only the typed-ReLU route precedes the complete previous unary path')
    require(re.findall(rb'\{"function":"([^"]+)"\}', after) ==
            [b'valid', b'tensor_dtype', b'typed_relu_tensor', b'tensor_dtype', b'wide_neg_tensor', b'wide_neg_previous_unary']
            and b'  i64_r:bool = compare.eq(dr,two_r)\n' in new_route
            and b'  f64_r:bool = compare.eq(dr,four_r)\n' in new_route
            and b'  f16_r:bool = compare.eq(dr,five_r)\n' in new_route
            and b'  bf16_r:bool = compare.eq(dr,six_r)\n' in new_route
            and b'  typed_r:bool = bool.or(wide_r,half_r)\n' in new_route
            and b'  relu_r:bool = compare.eq(opr,five_r)\n' in new_route
            and b'  admitted_r:bool = bool.and(typed_r,relu_r)\n' in new_route
            and b'  branch admitted_r native_relu(sr,opr,xr) previous_route(sr,opr,xr)\n' in new_route,
            'new branch requires exactly dtype2/4/5/6 and operation5; prior arguments are unchanged')
    require(all(b'  '+name+b':u64 = const() {"value":'+value+b'}\n' in new_route
                for name, value in ((b'two_r', b'2'), (b'four_r', b'4'), (b'five_r', b'5'), (b'six_r', b'6'))),
            'literal public dtype and operation constants')
    template = parent['half_neg_tensor'].replace(b'fn half_neg_tensor(', b'fn typed_relu_tensor(', 1)
    template = template.replace(b'  six:u64 = const() {"value":6}\n', b'  six:u64 = const() {"value":5}\n', 1)
    template = template.replace(b'  two9:u64 = const() {"value":2}\n',
                                b'  two9:u64 = call(d9) {"function":"dtype_width"}\n', 1)
    template = template.replace(b'{"function":"half_neg_value"}', b'{"function":"typed_relu_value"}', 1)
    require(template == current['typed_relu_tensor'],
            'original no-AD, allocation, typed layout, iteration and error guards retained in new helper')
    require(not re.findall(rb'\{"function":"([^"]+)"\}', current['typed_relu_value'])
            and b'  sign_i:u64 = const() {"value":9223372036854775808}\n' in current['typed_relu_value']
            and b'  chunk_r:u64 = const() {"value":32}\n' in current['typed_relu_value']
            and b'  bfloat_nan_v:u64 = const() {"value":65535}\n' in current['typed_relu_value'],
            'new value kernel uses native integer/bit instructions without calls or allocations')
    retained = ('binary', 'command', 'command_argument', 'typed_descriptor', 'integer_pointwise_layout',
                'half_mul_inner_extent', 'dtype_width', 'tget', 'taddr', 'typed_store',
                'wide_neg_tensor', 'half_neg_tensor', 'to_command', 'dtype_audit', 'f64_dtype_audit')
    require(all(current[name] == parent[name] for name in retained),
            'complete prior arithmetic, dispatch, allocation, ownership, AD and audit bodies unchanged')
    require(restored.count(before) == 1 and restored.replace(before, after, 1) + suffix == source,
            'complete775 forward round trip')
    return restored, dict(schema='heptatorch.typed-relu775-to-same773.complete-source-inverse.v1',
        source_sha256=SOURCE, parent_source_sha256=PARENT_SOURCE,
        functions=FUNCTIONS, parent_functions=PARENT_FUNCTIONS,
        changed_functions=list(CHANGED), appended_functions=list(ADDED), unchanged_parent_functions=772,
        actual_unary_sha256=sha(after), parent_unary_sha256=sha(before),
        new_branch_operation=5, new_branch_dtypes=[2, 4, 5, 6],
        previous_unary_path_retained=True, previous_arguments_preserved=True,
        original_untyped_and_other_operation_paths_preserved=True,
        complete_existing_noAD_allocation_and_layout_guards_retained=True,
        value_kernel_has_no_calls_or_allocations=True,
        complete_prior_dispatch_and_ownership_bodies_unchanged=True,
        newly_accepted_original_requests_require_explicit_numerical_successors=True,
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        delta_sha256=sha(delta_raw), actual_executable_replaced=False,
        original_cases_or_expected_values_modified=False,
        native_executed=False, Torch_executed=False, runtime_qualification_transferred=False)
