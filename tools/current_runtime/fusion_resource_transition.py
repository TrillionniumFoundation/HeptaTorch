"""Exact source729 path obligations for the unchanged seven resource recipes.

This proves a conditional source relation on the original public0/1/2/private3
domain. The unchanged complete owner and resource recipes must still execute.
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
SOURCE = '04ae430e3e851831fcb97a3fab700b16b6318361d52e90fcea90d2183d1d564b'
IMAGE = '4749413ea952bfd9d1fe41770f94ce1b9b9e3efabb407aa90f725cf33d59c4db'
OUTER = 'ca68c251be10292f0e2e47c4f010a19ec8542cb26b30c92a2d27d37dddba1ae8'
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
    require(sha(source) == SOURCE, 'exact source729 actual resource subject')
    path = HERE / 'fusion_source_transition.py'
    bind(path, OUTER)
    spec = importlib.util.spec_from_file_location('resource729_complete_outer_inverse', path)
    outer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(outer)
    historical, complete = outer.restore_source(source, bind)
    actual, parent = outer.functions(source, 729), outer.functions(historical, 700)
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
  v132:bool = bool.and(v123,v131)
  v133:bool = compare.eq(v84,v85)
  v86:bool = bool.or(v133,v132)
'''
    rows['command_argument'] = dict(inverse=reverse(actual['command_argument'], after,
        b'  v86:bool = compare.eq(v84,v85)\n', parent['command_argument'], 'command_argument'),
        relation='old dtype3 branch plus dtype4-only restriction; identical predicate for dtype0/1/2/3',
        new_branch_requires_dtype=4, new_branch_opcode_exceptions=[2,12,13],
        additional_calls=[], additional_writes_or_allocations=False)

    after = b'''  v94:u64 = const() {"value":4}
  v95:bool = compare.eq(v58,v94)
  v96:u64 = const() {"value":100}
  v97:bool = compare.ne(v43,v96)
  v98:bool = bool.and(v95,v97)
  v99:bool = compare.eq(v58,v59)
  v60:bool = bool.or(v99,v98)
'''
    rows['typed_command'] = dict(inverse=reverse(actual['typed_command'], after,
        b'  v60:bool = compare.eq(v58,v59)\n', parent['typed_command'], 'typed_command'),
        relation='old dtype3 branch plus dtype4-only restriction except fill100; original domain unchanged',
        new_branch_requires_dtype=4, new_branch_opcode_exceptions=[100],
        additional_calls=[], additional_writes_or_allocations=False)

    after = b'''  v160:u64 = const() {"value":4}
  v161:bool = compare.ne(v19,v160)
  v162:bool = bool.and(v27,v161)
  v30:bool = bool.and(v162,v29)
'''
    rows['typed_fill'] = dict(inverse=reverse(actual['typed_fill'], after,
        b'  v30:bool = bool.and(v27,v29)\n', parent['typed_fill'], 'typed_fill'),
        relation='only dtype4 is exempted from original upper32-bit rejection; dtype0/1/2/3 unchanged',
        new_exception_requires_dtype=4, additional_calls=[], additional_writes_or_allocations=False)

    audit_inverse = reverse(actual['dtype_audit'],
        b'  v1:u64 = call(v0) {"function":"f64_dtype_audit"}\n',
        b'  v1:u64 = call(v0) {"function":"scalar_previous_dtype_audit"}\n',
        parent['dtype_audit'], 'dtype_audit')
    bound_inverse = reverse(actual['scalar_previous_dtype_audit'],
        b'  v89:u64 = const() {"value":4}\n', b'  v89:u64 = const() {"value":3}\n',
        parent['scalar_previous_dtype_audit'], 'scalar_previous_dtype_audit')
    audit = actual['f64_dtype_audit']
    require(set(calls(audit)) == {b'scalar_previous_dtype_audit',b'getw',b'dptr'}
        and audit.count(b'{"function":"scalar_previous_dtype_audit"}') == 1
        and b'  v12:u64 = call(v0) {"function":"scalar_previous_dtype_audit"}\n' in audit
        and b'  branch v14 b1(v12) b2(v0)\n' in audit
        and b'  v28:u64 = const() {"value":4}\n  v29:bool = compare.eq(v27,v28)\n  v30:bool = bool.and(v24,v29)\n  branch v30 b5(v5,v6,v19) b6(v5,v6)\n' in audit,
        'fixed original audit first, then read-only scan whose checks require live dtype4')
    rows['dtype_audit'] = dict(inverse=audit_inverse, original_audit_bound_inverse=bound_inverse,
        additional_audit_sha256=sha(audit), additional_audit_bytes=len(audit),
        additional_audit_complete_call_set=sorted(x.decode() for x in set(calls(audit))),
        relation='original audit plus live-dtype4-only checks; extra scan has no writes or allocations',
        new_branch_requires_dtype=4, additional_writes_or_allocations=False)

    delegated = actual['f64_previous_to_command']
    require(delegated.count(b'fn f64_previous_to_command(') == 1
        and delegated.replace(b'fn f64_previous_to_command(', b'fn to_command(', 1) == parent['to_command'],
        'complete retained cat700 dispatcher has only its declaration renamed')
    wrapper = actual['to_command']
    require(set(calls(wrapper)) == {b'getr',b'scalar_peek_public',b'tensor_dtype',b'getw',
        b'terror',b'shape_command_finish',b'f64_previous_to_command'}
        and wrapper.count(b'{"function":"f64_previous_to_command"}') == 1
        and b'  v52:u64 = call(v25,v26,v27,v28,v29,v30) {"function":"f64_previous_to_command"}\n  ret(v52)\n' in wrapper
        and b'  v44:u64 = const() {"value":4}\n  v45:bool = compare.eq(v43,v44)\n  branch v45 b3(v15,v14,v17,v19) b4(v13,v14,v15,v16,v17,v18)\n' in wrapper
        and b'  branch v42 b4(v6,v7,v8,v9,v10,v11) b2(v6,v7,v8,v9,v10,v11,v12,v40)\n' in wrapper,
        'fixed complete F64 wrapper delegates the original domain and unresolved source with all six arguments')
    after = b'''  v427:u64 = const() {"value":4}
  v428:bool = compare.ne(v395,v427)
  v429:bool = bool.and(v397,v428)
  branch v429 b33(v377,v378,v379,v381,v383,v384) b35(v377,v378,v379,v380,v381,v382)
'''
    dispatch_inverse = reverse(actual['sd_previous_to_command'], after,
        b'  branch v397 b33(v377,v378,v379,v381,v383,v384) b35(v377,v378,v379,v380,v381,v382)\n',
        parent['sd_previous_to_command'], 'sd_previous_to_command')
    normalized = parent['sd_previous_to_command'].replace(b'fn sd_previous_to_command(', b'fn to_command(', 1)
    require(sha(normalized) == pins['to_command'], 'original full R4 dispatcher retained after finite target4 extension')
    front = parent['to_command'].split(b'block b6(', 1)[0]
    require(sha(front) == '56bbfa7109fe9774deb3ab7b3485fb970ed1433157f0690695197dc370ea74a6'
        and calls(front) == [b'getr',b'sd_permute_command',b'sd_accumulator_command',b'sd_previous_to_command'],
        'all original R4 opcode routes retain the existing700 dispatcher proof')
    rows['to_command'] = dict(relation='new dtype4-only guard, exact renamed700 delegation, then exact original R4 delegation',
        additional_wrapper_sha256=sha(wrapper), retained700_dispatcher_sha256=sha(delegated),
        declaration_name_inverse_exact=True, retained_R4_dispatcher_inverse=dispatch_inverse,
        reconstructed_original_R4_dispatcher_sha256=sha(normalized), original700_prefix_sha256=sha(front),
        immediate_delegation_opcodes=[108,109,118,119], immediate_delegation_sum115_target_dtype=1,
        new_refusal_requires_valid_public_dtype=4, unresolved_source_delegates=True,
        delegated_argument_order_unchanged=True, original_invalid_token_precedence_preserved=True)

    for name in ('getr','getw','dptr','scalar_peek_public','scalar_peek_resolved','so_tensor_live','so_uget','tensor_dtype'):
        require(actual[name] == parent[name], 'guard lookup helper is unchanged: ' + name)
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
            'unchanged full owner still rejects public4 in the original resource recipe domain')
    return dict(schema='heptatorch.fusion729.resource7-source-paths.v1', source_sha256=SOURCE,
        native_sha256=IMAGE, functions=729, units=14, outer_source_inverse=complete,
        changed_R4_obligations=list(CHANGED), obligations={n:rows[n] for n in CHANGED},
        original_owner_sha256=OWNER, original_verify_snapshot_sha256=sha(body),
        domain_must_still_be_checked_by_original_runtime_owner=True,
        original_dtype_domain=[0,1,2,3], original_numeric_or_resource_assertions_modified=False,
        historical_bodies_substituted_for_actual_bodies=False,
        native_executed=False, Torch_executed=False, runtime_qualification_transferred=False)
