"""Exact source775 path obligations for the unchanged seven resource recipes.

This proves a conditional source relation on the original public0/1/2/private3
domain. The unchanged complete owner and resource recipes must still execute.
The original resource requests exclude unary opcode3; new I64/F64 negation
requires separate successor execution and is not covered by these recipes.
Typed ReLU operation5 also uses unary opcode3, explicitly absent from all seven
original resource request streams; its new outcomes require actual execution.
Half/BF16 division operation4 remains behind the dtype5/6 binary guard, outside
the original resource owner domain; its new numerical outcomes require their
own actual execution. Mixed public F64 arithmetic requires an operand dtype4
and is likewise outside the original public0/1/2/private3 resource domain.
No reconstructed historical body is presented as actual executable code.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = 'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c'
IMAGE = '083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'
OUTER = 'f6367a2fba18703e9d807f250e18d503c717969454369754166ee510640c7a40'
OBLIGATIONS = '4fa8028bec733a72ac2857dce326e25d6d378aafd1045c2504bb5828553cf2bb'
OWNER = 'e54509ceba0ef3bf171f5fc1018665dc97e56eeccec0cba1e8c14f9a7ed81501'
CHANGED = ('command_argument', 'dtype_audit', 'to_command', 'typed_command', 'typed_fill')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reverse(actual, after, before, original, label):
    require(actual.count(after) == 1 and after != before,
            'one finite complete source edit: ' + label)
    restored = actual.replace(after, before, 1)
    require(restored == original, 'every remaining body byte is original: ' + label)
    return dict(before_sha256=sha(original), after_sha256=sha(actual),
                before_bytes=len(original), after_bytes=len(actual),
                before_hex=before.hex(), after_hex=after.hex(), count=1,
                complete_body_inverse_exact=True)


def calls(body):
    return re.findall(rb'\{"function":"([^"]+)"\}', body)


def prove(source, bind):
    """Purely inspect complete actual source; return no runtime PASS."""
    bind(Path(__file__).resolve())
    require(sha(source) == SOURCE, 'exact source775 actual resource subject')
    path = HERE / 'fusion_source_transition.py'
    bind(path, OUTER)
    spec = importlib.util.spec_from_file_location('resource775_complete_outer_inverse', path)
    outer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(outer)
    historical, complete = outer.restore_source(source, bind)
    actual, parent = outer.functions(source, 800), outer.functions(historical, 700)
    view_path = HERE / 'public_f64_view_source_transition.py'
    view_raw = bind(view_path)
    import types
    view = types.ModuleType('resource_current_f64_view_original_domain')
    view.__file__ = str(view_path)
    exec(compile(view_raw, str(view_path), 'exec'), view.__dict__)
    view_route = view.prove_original_domain(source, bind)
    predicate_inverse = complete['predicate_source_inverse']
    guard = predicate_inverse['dispatcher_guard']
    delegation = predicate_inverse['old_path_delegation']
    mul_inverse = complete['half_mul_source_inverse']
    neg_inverse = complete['half_neg_source_inverse']
    addsub_inverse = complete['half_addsub_source_inverse']
    wide_inverse = complete['wide_neg_source_inverse']
    div_inverse = complete['half_div_source_inverse']
    arithmetic_inverse = complete['public_f64_arithmetic_source_inverse']
    same_inverse = complete['same_f64_arithmetic_source_inverse']
    relu_inverse = complete['typed_relu_source_inverse']
    index_inverse = complete['index_signbit_source_inverse']
    require(index_inverse['source_sha256'] == SOURCE and index_inverse['functions'] == 800
        and index_inverse['parent_functions'] == 775
        and index_inverse['whole_source_inverse_exact'] is True
        and index_inverse['whole_round_trip_exact'] is True
        and relu_inverse['source_sha256'] == index_inverse['parent_source_sha256']
        and relu_inverse['functions'] == 775 and relu_inverse['parent_functions'] == 773
        and relu_inverse['changed_functions'] == ['unary']
        and relu_inverse['appended_functions'] == ['typed_relu_value','typed_relu_tensor']
        and relu_inverse['unchanged_parent_functions'] == 772
        and relu_inverse['new_branch_operation'] == 5
        and relu_inverse['new_branch_dtypes'] == [2,4,5,6]
        and relu_inverse['previous_unary_path_retained'] is True
        and relu_inverse['previous_arguments_preserved'] is True
        and relu_inverse['whole_source_inverse_exact'] is True
        and relu_inverse['whole_round_trip_exact'] is True
        and same_inverse['source_sha256'] == relu_inverse['parent_source_sha256']
        and same_inverse['functions'] == same_inverse['parent_functions'] == 773
        and same_inverse['changed_functions'] == ['half_mul_priority','public_f64_binary']
        and same_inverse['unchanged_parent_functions'] == 771
        and same_inverse['non_f64_routes_unchanged'] is True
        and same_inverse['Half_routes_unchanged'] is True
        and same_inverse['whole_source_inverse_exact'] is True
        and same_inverse['whole_round_trip_exact'] is True
        and arithmetic_inverse['source_sha256'] == same_inverse['parent_source_sha256']
        and arithmetic_inverse['functions'] == 773 and arithmetic_inverse['parent_functions'] == 771
        and arithmetic_inverse['changed_functions'] == ['binary','half_mul_cast']
        and arithmetic_inverse['unchanged_parent_functions'] == 769
        and arithmetic_inverse['whole_source_inverse_exact'] is True
        and arithmetic_inverse['whole_round_trip_exact'] is True
        and arithmetic_inverse['preexisting_half_binary_routes_unchanged'] is True
        and arithmetic_inverse['same_f64_route_preserved'] is True
        and arithmetic_inverse['non_f64_routes_preserved'] is True
        and arithmetic_inverse['private_dtype3_delegated'] is True
        and arithmetic_inverse['unsupported_operations_delegated'] is True
        and arithmetic_inverse['original_arithmetic_kernel_bodies_unchanged'] is True
        and div_inverse['source_sha256'] == arithmetic_inverse['parent_source_sha256']
        and div_inverse['functions'] == 771 and div_inverse['parent_functions'] == 767
        and div_inverse['changed_functions'] == ['binary']
        and div_inverse['unchanged_parent_functions'] == 766
        and div_inverse['whole_source_inverse_exact'] is True
        and div_inverse['whole_round_trip_exact'] is True
        and div_inverse['only_new_half_binary_operation'] == 4
        and wide_inverse['source_sha256'] == div_inverse['parent_source_sha256']
        and wide_inverse['parent_source_sha256'] == addsub_inverse['source_sha256']
        and neg_inverse['source_sha256'] == addsub_inverse['parent_source_sha256']
        and mul_inverse['source_sha256'] == neg_inverse['parent_source_sha256']
        and complete['bf16_source_inverse']['source_sha256'] == mul_inverse['parent_source_sha256']
        and predicate_inverse['source_sha256'] == complete['bf16_source_inverse']['parent_source_sha256']
        and predicate_inverse['functions'] == 740
        and predicate_inverse['parent_functions'] == 736
        and predicate_inverse['unchanged_parent_functions'] == 735
        and predicate_inverse['changed_functions'] == ['entry']
        and predicate_inverse['entry_only_upper_bound_and_dispatch_target_changed'] is True
        and guard['opcode_min'] == 120 and guard['opcode_max'] == 124
        and guard['reads_before_guard'] == ['getr(request,at)']
        and guard['exact_closed_range_verified'] is True
        and delegation['target'] == 'to_command'
        and delegation['parameter_count'] == 6
        and delegation['argument_order_preserved'] is True
        and delegation['complete_parent_body_unchanged'] is True
        and delegation['parent_body_sha256'] == delegation['actual_body_sha256']
            == view_route['old_to_command_sha256']
        and view_route['source_sha256'] == SOURCE
        and view_route['original_domain_dispatcher_guard_unchanged'] is True
        and view_route['original_non_f64_constructor_and_all_arguments_retained'] is True,
        'physical775 entry delegates every original108..119 opcode through the exact six-argument original to_command body')
    predicate_route = dict(
        actual_entry_sha256=sha(actual['entry']),
        actual_predicate_command_sha256=sha(actual['predicate_command']),
        actual_to_command_sha256=sha(actual['to_command']),
        historical740_entry_only_upper_bound_and_dispatch_target_changed=True,
        current_signbit_opcode=126, current_source_extension=index_inverse,
        entry_original_delegation_opcodes=list(range(108, 120)),
        dispatcher_guard=guard, old_path_delegation=delegation,
        new_predicate_branch_reachable_for_original_opcodes=False,
        historical_body_substituted_for_actual_body=False)
    binary = actual['binary']
    previous_binary = actual['half_mul_previous_binary']
    require(addsub_inverse['changed_functions'] == ['binary']
        and addsub_inverse['non_half_and_predecessor_half_other_operation_routes_preserved'] is True
        and mul_inverse['changed_functions'] == ['binary', 'command_argument']
        and mul_inverse['predecessor_binary_body_preserved_exactly'] is True
        and sha(previous_binary) == mul_inverse['added_bodies']['half_mul_previous_binary']['sha256']
        and sha(binary) == arithmetic_inverse['actual_binary_sha256']
        and arithmetic_inverse['parent_binary_sha256'] == div_inverse['actual_binary_sha256']
        and sha(actual['half_mul_cast']) == arithmetic_inverse['actual_cast_sha256']
        and div_inverse['parent_binary_sha256'] == addsub_inverse['changed_bodies']['binary']['after_sha256']
        and div_inverse['non_half_routes_unchanged'] is True
        and div_inverse['non_division_routes_unchanged'] is True
        and addsub_inverse['changed_bodies']['binary']['before_sha256']
            == mul_inverse['changed_bodies']['binary']['after_sha256']
        and sha(previous_binary.replace(b'fn half_mul_previous_binary(', b'fn binary(', 1))
            == mul_inverse['changed_bodies']['binary']['before_sha256']
        and calls(binary) == [b'valid', b'valid', b'tensor_dtype', b'tensor_dtype',
                              b'half_addsub_tensor', b'half_div_tensor', b'half_mul_tensor',
                              b'public_f64_binary']
        and b'  branch half3 arithmetic(s3,o3,l3,r3) previous(s3,o3,l3,r3)\n' in binary
        and b'  onea:u64 = const() {"value":1}\n  twoa:u64 = const() {"value":2}\n'
            b'  adda:bool = compare.eq(oa,onea)\n  suba:bool = compare.eq(oa,twoa)\n'
            b'  addsuba:bool = bool.or(adda,suba)\n'
            b'  branch addsuba addsub(sa,oa,la,ra) multiply(sa,oa,la,ra)\n' in binary
        and b'block multiply(s4:mutbuf,o4:u64,l4:u64,r4:u64)\n'
            b'  four4:u64 = const() {"value":4}\n'
            b'  division4:bool = compare.eq(o4,four4)\n'
            b'  branch division4 divide(s4,o4,l4,r4) retained_multiply(s4,o4,l4,r4)\n'
            b'block divide(sd:mutbuf,od:u64,ld:u64,rd:u64)\n'
            b'  quotientd:u64 = call(sd,od,ld,rd) {"function":"half_div_tensor"}\n'
            b'  ret(quotientd)\n'
            b'block retained_multiply(sm:mutbuf,om:u64,lm:u64,rm:u64)\n'
            b'  productm:u64 = call(sm,om,lm,rm) {"function":"half_mul_tensor"}\n'
            b'  ret(productm)\n' in binary
        and b'  result5:u64 = call(s5,o5,l5,r5) {"function":"public_f64_binary"}\n  ret(result5)\n' in binary
        and b'  five3:u64 = const() {"value":5}\n  six3:u64 = const() {"value":6}\n' in binary
        and b'  lh3:bool = compare.eq(ld3,five3)\n  lb3:bool = compare.eq(ld3,six3)\n'
            b'  rh3:bool = compare.eq(rd3,five3)\n  rb3:bool = compare.eq(rd3,six3)\n'
            b'  left_half3:bool = bool.or(lh3,lb3)\n  right_half3:bool = bool.or(rh3,rb3)\n'
            b'  half3:bool = bool.or(left_half3,right_half3)\n' in binary,
        'complete current binary routes live original-domain operands to the exact retained predecessor body')
    mixed = actual['public_f64_binary']
    mixed_front = mixed.split(b'block previous(', 1)[0]
    require(sha(mixed) == same_inverse['actual_route_sha256']
        and same_inverse['parent_route_sha256'] == arithmetic_inverse['actual_mixed_route_sha256']
        and calls(mixed_front) == [b'tensor_dtype', b'tensor_dtype']
        and b'  four:u64 = const() {"value":4}\n  two:u64 = const() {"value":2}\n' in mixed_front
        and b'  ldouble:bool = compare.eq(ld,four)\n  rdouble:bool = compare.eq(rd,four)\n'
            b'  anydouble:bool = bool.or(ldouble,rdouble)\n' in mixed_front
        and b'  types:bool = bool.and(anydouble,public)\n' in mixed_front
        and b'  admitted:bool = bool.and(types,arithmetic)\n' in mixed_front
        and b'  branch admitted select_operation(s,operation,left,right) previous(s,operation,left,right)\n' in mixed_front
        and b'block previous(sp:mutbuf,op:u64,lp:u64,rp:u64)\n'
            b'  originalp:u64 = call(sp,op,lp,rp) {"function":"half_mul_previous_binary"}\n  ret(originalp)\n' in mixed,
        'actual mixed route requires public dtype4 and otherwise delegates the same four original arguments')
    binary_route = dict(actual_binary_sha256=sha(binary),
        actual_public_f64_binary_sha256=sha(mixed),
        public_f64_arithmetic_source_inverse=arithmetic_inverse,
        same_f64_arithmetic_source_inverse=same_inverse,
        new_public_f64_branch_requires_operand_dtype=4,
        new_public_f64_branch_other_operand_dtype=[0,1,2,4],
        new_public_f64_branch_operations=[1,2,3,4],
        new_public_f64_branch_reachable_for_original_domain=False,
        new_public_f64_outcomes_require_separate_successor_evidence=True,
        retained_previous_binary_sha256=sha(previous_binary),
        predecessor_binary_sha256=mul_inverse['changed_bodies']['binary']['before_sha256'],
        declaration_name_inverse_exact=True, delegated_argument_order_unchanged=True,
        extra_pre_route_calls=['valid(left)', 'valid(right)', 'tensor_dtype(left)', 'tensor_dtype(right)',
                              'tensor_dtype(left)', 'tensor_dtype(right)'],
        new_arithmetic_branch_requires_operand_dtype=[5,6],
        new_addsub_branch_operations=[1,2],
        new_division_branch_operations=[4],
        new_division_branch_requires_operand_dtype=[5,6],
        prior_half_operation_routes_preserved_except=[4],
        new_division_outcomes_require_separate_successor_evidence=True,
        original_domain_live_operands_delegate=True,
        original_error_and_resource_recipes_must_execute=True,
        historical_body_substituted_for_actual_body=False)
    unary = actual['unary']
    wide_previous_unary = actual['wide_neg_previous_unary']
    previous_unary = actual['half_neg_previous_unary']
    require(wide_inverse['changed_functions'] == ['unary', 'command_argument']
        and wide_inverse['previous_unary_body_retained'] is True
        and wide_inverse['only_public_F64_unary_admission_extended'] is True
        and wide_inverse['newly_supported_request_outcomes_require_separate_successor_evidence'] is True
        and neg_inverse['changed_functions'] == ['unary', 'command_argument']
        and neg_inverse['predecessor_unary_body_preserved_exactly'] is True
        and sha(previous_unary) == neg_inverse['added_bodies']['half_neg_previous_unary']['sha256']
        and sha(wide_previous_unary.replace(b'fn wide_neg_previous_unary(', b'fn unary(', 1))
            == neg_inverse['changed_bodies']['unary']['after_sha256']
        and sha(previous_unary.replace(b'fn half_neg_previous_unary(', b'fn unary(', 1))
            == neg_inverse['changed_bodies']['unary']['before_sha256']
        and sha(unary) == relu_inverse['actual_unary_sha256']
        and calls(unary) == [b'valid', b'tensor_dtype', b'typed_relu_tensor',
                             b'tensor_dtype', b'wide_neg_tensor', b'wide_neg_previous_unary']
        and b'  i64_dtype2:u64 = const() {"value":2}\n  f64_dtype2:u64 = const() {"value":4}\n' in unary
        and b'  is_i64_2:bool = compare.eq(dtype2,i64_dtype2)\n'
            b'  is_f64_2:bool = compare.eq(dtype2,f64_dtype2)\n'
            b'  wide2:bool = bool.or(is_i64_2,is_f64_2)\n' in unary
        and b'  branch wide2 negative(s2,o2,x2) previous(s2,o2,x2)\n' in unary
        and b'  y4:u64 = call(s4,o4,x4) {"function":"wide_neg_previous_unary"}\n  ret(y4)\n' in unary
        and calls(wide_previous_unary) == [b'valid', b'tensor_dtype', b'half_neg_tensor', b'half_neg_previous_unary']
        and b'  five2:u64 = const() {"value":5}\n  six2:u64 = const() {"value":6}\n' in wide_previous_unary
        and b'  half2:bool = compare.eq(dtype2,five2)\n  bf2:bool = compare.eq(dtype2,six2)\n'
            b'  reduced2:bool = bool.or(half2,bf2)\n' in wide_previous_unary
        and b'  branch reduced2 negative(s2,o2,x2) previous(s2,o2,x2)\n' in wide_previous_unary
        and b'  y4:u64 = call(s4,o4,x4) {"function":"half_neg_previous_unary"}\n  ret(y4)\n' in wide_previous_unary,
        'complete typed ReLU, wide and retained Half wrappers preserve the exact original unary body for dtype0/1/3')
    unary_route = dict(actual_unary_sha256=sha(unary),
        typed_relu_source_inverse=relu_inverse,
        new_relu_operation=5, new_relu_operand_dtypes=[2,4,5,6],
        new_relu_branch_reachable_for_original_resource_requests=False,
        new_relu_outcomes_require_separate_successor_evidence=True,
        retained_wide_previous_unary_sha256=sha(wide_previous_unary),
        retained_previous_unary_sha256=sha(previous_unary),
        predecessor_unary_sha256=neg_inverse['changed_bodies']['unary']['before_sha256'],
        declaration_name_inverse_exact=True, delegated_argument_order_unchanged=True,
        extra_pre_route_calls=['valid(input)', 'tensor_dtype(input)', 'tensor_dtype(input)',
                              'valid(input)', 'tensor_dtype(input)'],
        new_arithmetic_branch_requires_operand_dtype=[2,4,5,6],
        wide_branch_operand_dtype=[2,4], retained_half_branch_operand_dtype=[5,6],
        original_dtype_domain_live_operands_that_delegate=[0,1,3],
        original_domain_live_operands_delegate=False,
        new_wide_outcomes_require_separate_successor_evidence=True,
        original_resource_requests_must_exclude_unary_opcode=3,
        original_error_and_resource_recipes_must_execute=True,
        historical_body_substituted_for_actual_body=False)
    original = json.loads(bind(ROOT / 'tests/wrapped_scalar_resource_review/portable07b/source-obligations.json', OBLIGATIONS))
    pins = original['functions']
    require([n for n in pins if actual[n] != parent[n]] == list(CHANGED),
            'only the five reviewed R4 obligation bodies changed since cat700')
    rows = {}

    after = b'''  v122:u64 = const() {"value":4}
  v123:bool = compare.eq(v84,v122)
  v124:u64 = const() {"value":2}
  v125:bool = compare.ne(v73,v124)
  v126:u64 = const() {"value":12}
  v127:bool = compare.ne(v73,v126)
  v128:u64 = const() {"value":13}
  v129:bool = compare.ne(v73,v128)
  v130:bool = bool.and(v125,v127)
  v131:bool = bool.and(v130,v129)
  wide_unary_opcode:u64 = const() {"value":3}
  wide_not_unary:bool = compare.ne(v73,wide_unary_opcode)
  wide_not_admitted:bool = bool.and(v131,wide_not_unary)
  v132:bool = bool.and(v123,wide_not_admitted)
  v133:bool = compare.eq(v84,v85)
  v86:bool = bool.or(v133,v132)
  v134:u64 = const() {"value":5}
  v135:u64 = const() {"value":6}
  v136:bool = compare.eq(v84,v134)
  v137:bool = compare.eq(v84,v135)
  v138:bool = bool.or(v136,v137)
  v139:bool = bool.and(v127,v129)
  half_operation:bool = bool.and(v139,v125)
  half_unary_opcode:u64 = const() {"value":3}
  half_not_unary:bool = compare.ne(v73,half_unary_opcode)
  half_not_arithmetic:bool = bool.and(half_operation,half_not_unary)
  v140:bool = bool.and(v138,half_not_arithmetic)
  v141:bool = bool.or(v86,v140)
  branch v141 b9(v72,v73,v76) b10(v72,v76,v141)
'''
    rows['command_argument'] = dict(inverse=reverse(actual['command_argument'], after,
        b'  v86:bool = compare.eq(v84,v85)\n  branch v86 b9(v72,v73,v76) b10(v72,v76,v86)\n',
        parent['command_argument'], 'command_argument'),
        relation='old dtype3 predicate plus dtype4 and dtype5/6 restrictions; both added predicates are false for dtype0/1/2/3',
        new_branch_requires_dtype=[4,5,6], new_branch_opcode_exceptions={'4':[2,3,12,13],'5/6':[2,3,12,13]},
        additional_calls=[], additional_writes_or_allocations=False)

    after = b'''  v94:u64 = const() {"value":4}
  v95:bool = compare.eq(v58,v94)
  v96:u64 = const() {"value":100}
  v97:bool = compare.ne(v43,v96)
  v98:bool = bool.and(v95,v97)
  v99:bool = compare.eq(v58,v59)
  v60:bool = bool.or(v99,v98)
  v100:u64 = const() {"value":5}
  v101:u64 = const() {"value":6}
  v102:bool = compare.eq(v58,v100)
  v103:bool = compare.eq(v58,v101)
  v104:bool = bool.or(v102,v103)
  v105:bool = bool.and(v104,v97)
  v106:bool = bool.or(v60,v105)
  jump b4(v36,v37,v38,v39,v40,v41,v42,v43,v44,v106)
'''
    rows['typed_command'] = dict(inverse=reverse(actual['typed_command'], after,
        b'  v60:bool = compare.eq(v58,v59)\n  jump b4(v36,v37,v38,v39,v40,v41,v42,v43,v44,v60)\n',
        parent['typed_command'], 'typed_command'),
        relation='old dtype3 predicate plus dtype4/5/6 restriction except fill100; every added dtype comparison is false for dtype0/1/2/3',
        new_branch_requires_dtype=[4,5,6], new_branch_opcode_exceptions=[100],
        additional_calls=[], additional_writes_or_allocations=False)

    after = b'''  v160:u64 = const() {"value":4}
  v161:bool = compare.ne(v19,v160)
  v162:bool = bool.and(v27,v161)
  v30:bool = bool.and(v162,v29)
  v31:bool = bool.or(v25,v30)
  v163:u64 = const() {"value":5}
  v164:u64 = const() {"value":6}
  v165:bool = compare.eq(v19,v163)
  v166:bool = compare.eq(v19,v164)
  v167:bool = bool.or(v165,v166)
  v168:u64 = const() {"value":65535}
  v169:bool = compare.gt(v13,v168)
  v170:bool = bool.and(v167,v169)
  v171:bool = bool.or(v31,v170)
  branch v171 b3(v11) b4(v11,v12,v13,v14,v16,v19,v20)
'''
    rows['typed_fill'] = dict(inverse=reverse(actual['typed_fill'], after,
        b'  v30:bool = bool.and(v27,v29)\n  v31:bool = bool.or(v25,v30)\n  branch v31 b3(v11) b4(v11,v12,v13,v14,v16,v19,v20)\n',
        parent['typed_fill'], 'typed_fill'),
        relation='dtype4 upper32-bit exception and dtype5/6 upper16-bit guard; both dtype tests leave dtype0/1/2/3 unchanged',
        new_exception_requires_dtype=4, new_upper16_guard_requires_dtype=[5,6],
        additional_calls=[], additional_writes_or_allocations=False)

    audit_inverse = reverse(actual['dtype_audit'],
        b'  v1:u64 = call(v0) {"function":"half_dtype_audit"}\n',
        b'  v1:u64 = call(v0) {"function":"scalar_previous_dtype_audit"}\n',
        parent['dtype_audit'], 'dtype_audit')
    bound_inverse = reverse(actual['scalar_previous_dtype_audit'],
        b'  v89:u64 = const() {"value":6}\n', b'  v89:u64 = const() {"value":3}\n',
        parent['scalar_previous_dtype_audit'], 'scalar_previous_dtype_audit')
    audit = actual['f64_dtype_audit']
    require(view_route['actual_f64_dtype_audit_sha256'] == sha(audit)
        and view_route['original_dtype_audit_called_first_and_unchanged'] is True
        and view_route['new_audit_checks_only_live_public_dtype4'] is True
        and view_route['original_root_incarnation_version_release_audits_unchanged'] is True,
        'actual complete F64 audit preserves the original dtype0/1/2/3 resource domain')
    half_audit = actual['half_dtype_audit']
    require(set(calls(half_audit)) == {b'f64_dtype_audit',b'getw',b'dptr'}
        and half_audit.count(b'{"function":"f64_dtype_audit"}') == 1
        and b'  v12:u64 = call(v0) {"function":"f64_dtype_audit"}\n' in half_audit
        and b'  branch v14 b1(v12) b2(v0)\n' in half_audit
        and b'  v23:u64 = const() {"value":5}\n  v24:bool = compare.eq(v22,v23)\n  v25:u64 = const() {"value":6}\n  v26:bool = compare.eq(v22,v25)\n  v27:bool = bool.or(v24,v26)\n' in half_audit
        and b'  v32:bool = compare.ne(v30,v31)\n  v33:bool = bool.and(v32,v27)\n  branch v33 b5(v5,v6,v19) b6(v5,v6)\n' in half_audit,
        'original audits first, then read-only scan whose checks require live dtype5/6')
    rows['dtype_audit'] = dict(inverse=audit_inverse, original_audit_bound_inverse=bound_inverse,
        additional_audit_sha256=sha(audit), additional_audit_bytes=len(audit),
        additional_audit_complete_call_set=sorted(x.decode() for x in set(calls(audit))),
        half_audit_sha256=sha(half_audit), half_audit_bytes=len(half_audit),
        half_audit_complete_call_set=sorted(x.decode() for x in set(calls(half_audit))),
        relation='original audit bound is extended to6; readonly F64 and Half scans only inspect live dtype4/5/6 nodes, excluded by original owner domain',
        new_branch_requires_dtype=[4,5,6], additional_writes_or_allocations=False)

    delegated = actual['f64_previous_to_command']
    require(delegated.count(b'fn f64_previous_to_command(') == 1
        and delegated.replace(b'fn f64_previous_to_command(', b'fn to_command(', 1) == parent['to_command'],
        'complete retained cat700 dispatcher has only its declaration renamed')
    wrapper = actual['to_command']
    require(set(calls(wrapper)) == {b'getr',b'scalar_peek_public',b'tensor_dtype',b'getw',
        b'terror',b'shape_command_finish',b'f64_previous_to_command'}
        and wrapper.count(b'{"function":"f64_previous_to_command"}') == 1
        and b'  v52:u64 = call(v25,v26,v27,v28,v29,v30) {"function":"f64_previous_to_command"}\n  ret(v52)\n' in wrapper
        and b'  v44:u64 = const() {"value":4}\n  v45:bool = compare.eq(v43,v44)\n  v80:u64 = const() {"value":5}\n  v81:u64 = const() {"value":6}\n  v82:bool = compare.eq(v43,v80)\n  v83:bool = compare.eq(v43,v81)\n  v84:bool = bool.or(v82,v83)\n  as_strided_opcode:u64 = const() {"value":111}\n  other_f64_command:bool = compare.ne(v19,as_strided_opcode)\n  rejected_f64:bool = bool.and(v45,other_f64_command)\n  v85:bool = bool.or(rejected_f64,v84)\n  branch v85 b3(v15,v14,v17,v19) b4(v13,v14,v15,v16,v17,v18)\n' in wrapper
        and b'  branch v42 b4(v6,v7,v8,v9,v10,v11) b2(v6,v7,v8,v9,v10,v11,v12,v40)\n' in wrapper,
        'fixed complete F64/Half wrapper delegates the original domain and unresolved source with all six arguments')
    after = b'''  v427:u64 = const() {"value":4}
  v428:bool = compare.ne(v395,v427)
  v429:bool = bool.and(v397,v428)
  v430:u64 = const() {"value":5}
  v431:u64 = const() {"value":6}
  v432:bool = compare.ne(v395,v430)
  v433:bool = compare.ne(v395,v431)
  v434:bool = bool.and(v429,v432)
  v435:bool = bool.and(v434,v433)
  branch v435 b33(v377,v378,v379,v381,v383,v384) b35(v377,v378,v379,v380,v381,v382)
'''
    dispatch_inverse = reverse(actual['sd_previous_to_command'], after,
        b'  branch v397 b33(v377,v378,v379,v381,v383,v384) b35(v377,v378,v379,v380,v381,v382)\n',
        parent['sd_previous_to_command'], 'sd_previous_to_command')
    normalized = parent['sd_previous_to_command'].replace(b'fn sd_previous_to_command(', b'fn to_command(', 1)
    require(sha(normalized) == pins['to_command'], 'original full R4 dispatcher retained after finite target4/5/6 extension')
    front = parent['to_command'].split(b'block b6(', 1)[0]
    require(sha(front) == '56bbfa7109fe9774deb3ab7b3485fb970ed1433157f0690695197dc370ea74a6'
        and calls(front) == [b'getr',b'sd_permute_command',b'sd_accumulator_command',b'sd_previous_to_command'],
        'all original R4 opcode routes retain the existing700 dispatcher proof')
    rows['to_command'] = dict(relation='new dtype4/5/6 guard is false on original dtype0/1/2/3, exact renamed700 delegation, then exact original R4 delegation',
        predicate_entry_route=predicate_route,
        additional_wrapper_sha256=sha(wrapper), retained700_dispatcher_sha256=sha(delegated),
        declaration_name_inverse_exact=True, retained_R4_dispatcher_inverse=dispatch_inverse,
        reconstructed_original_R4_dispatcher_sha256=sha(normalized), original700_prefix_sha256=sha(front),
        immediate_delegation_opcodes=[108,109,118,119], immediate_delegation_sum115_target_dtype=1,
        new_refusal_requires_valid_public_dtype=[4,5,6], unresolved_source_delegates=True,
        delegated_argument_order_unchanged=True, original_invalid_token_precedence_preserved=True)

    for name in ('getr','getw','dptr','scalar_peek_public','scalar_peek_resolved','so_uget','tensor_dtype','valid'):
        require(actual[name] == parent[name], 'guard lookup helper is unchanged: ' + name)
    require(view_route['old_so_tensor_live_sha256'] == sha(parent['so_tensor_live'])
        and view_route['actual_so_tensor_live_sha256'] == sha(actual['so_tensor_live'])
        and view_route['original_byte_offset_domain'] == [0,1073741824]
        and view_route['ordinary_owner_byte_limit_span_and_successor_arguments_retained'] is True
        and view_route['added_ordinary_owner_guard_uses_only_unchanged_read_helpers'] is True
        and view_route['original_legal_owner_offsets_never_enter_new_tag_helper'] is True,
        'actual owner lookup retains original byte-offset behavior through explicitly proved tag guards')
    for name in CHANGED:
        if name != 'to_command':
            require(sha(parent[name]) == pins[name], 'original R4 complete obligation pin: ' + name)
        rows[name].update(name=name, original_sha256=pins[name], actual_sha256=sha(actual[name]),
                          actual_bytes=len(actual[name]), conditional_dtype_domain=[0,1,2,3])

    owner_path = ROOT / 'tests/wrapped_scalar_resource_review/portable07b/owner_view05.py'
    owner = bind(owner_path, OWNER)
    tree = ast.parse(owner)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_snapshot')
    body = ast.get_source_segment(owner.decode(), fn).encode()
    require(body.count(b'need(dtype in (0, 1, 2, 3), "dtype-code", node=i)') == 1,
            'unchanged full owner still rejects public4/5/6 in the original resource recipe domain')
    return dict(schema='heptatorch.fusion778.resource7-source-paths.v1', source_sha256=SOURCE,
        native_sha256=IMAGE, functions=800, units=15, outer_source_inverse=complete,
        changed_R4_obligations=list(CHANGED), obligations={n:rows[n] for n in CHANGED},
        public_f64_view_original_domain_route=view_route,
        binary_original_domain_route=binary_route,
        unary_original_domain_route=unary_route,
        original_owner_sha256=OWNER, original_verify_snapshot_sha256=sha(body),
        domain_must_still_be_checked_by_original_runtime_owner=True,
        original_dtype_domain=[0,1,2,3], original_numeric_or_resource_assertions_modified=False,
        historical_bodies_substituted_for_actual_bodies=False,
        native_executed=False, Torch_executed=False, runtime_qualification_transferred=False)
