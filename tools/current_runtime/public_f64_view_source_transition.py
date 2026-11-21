"""Recover every published 797 byte from the native public F64 view extension.

This is a complete source proof. Runtime outcomes require actual same-image
execution and independent reference/physical readback; no old PASS transfers.
"""
import hashlib
import json
from pathlib import Path
import re

SOURCE = 'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c'
PARENT_SOURCE = 'c73a7cb474981056404ac96bc214979d57aaa9a86e6474b1c140c3d98eb05779'
FUNCTIONS = 800
PARENT_FUNCTIONS = 797
SOURCE_BYTES = 2664217
PARENT_BYTES = 2592988
CHANGED = ('bind_alias', 'so_tensor_live', 'to_command', 'to_strided_view', 'f64_dtype_audit')
ADDED = ('public_f64_empty_strided_view', 'public_f64_empty_descriptor', 'so_f64_empty_offset_live')
DELTA_SHA256 = '36f3da06db594870b93354feae99a65de8762cf0a90d078190b98ae2883606cf'
HEADER = b'hia-asm.v1 hepta.hia.program.v8 heptatorch_runtime\n'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def functions(raw, count):
    matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\([^\n]*\n.*?^end\n', raw, re.M | re.S))
    result = {match[1].decode('ascii'): match[0] for match in matches}
    require(len(result) == len(matches) == count, 'complete unique native function census')
    require(raw == HEADER + b''.join(result.values()), 'complete canonical source envelope and order')
    return result


def restore_source(source, bind):
    bind(Path(__file__).resolve())
    raw = bind(Path(__file__).with_name('public_f64_view_source_delta.json'), DELTA_SHA256)
    require(sha(raw) == DELTA_SHA256, 'complete independently fixed F64 source delta')
    delta = json.loads(raw)
    require(type(source) is bytes and sha(source) == SOURCE and len(source) == SOURCE_BYTES,
            'complete actual F64 view source identity')
    require(delta['schema'] == 'heptatorch.public-f64-view.source-delta.v1'
            and delta['source_sha256'] == SOURCE and delta['parent_source_sha256'] == PARENT_SOURCE
            and delta['functions'] == FUNCTIONS and delta['parent_functions'] == PARENT_FUNCTIONS
            and delta['source_bytes'] == SOURCE_BYTES and delta['parent_bytes'] == PARENT_BYTES
            and tuple(delta['changed_functions']) == CHANGED and tuple(delta['added_functions']) == ADDED
            and delta['unchanged_parent_functions'] == PARENT_FUNCTIONS - len(CHANGED)
            and set(delta['bodies']) == set(CHANGED) and set(delta['added_bodies']) == set(ADDED),
            'fixed complete source identities and finite changed/appended definition sets')
    actual = functions(source, FUNCTIONS)
    order = delta['parent_function_order']
    require(len(order) == len(set(order)) == PARENT_FUNCTIONS
            and list(actual) == order + list(ADDED) and set(CHANGED) <= set(order)
            and not set(ADDED) & set(order) and len(CHANGED) == len(set(CHANGED))
            and len(ADDED) == len(set(ADDED)), 'all previous definitions retain their exact position')
    suffix = b''.join(actual[name] for name in ADDED)
    require(source.endswith(suffix) and len(suffix) == delta['added_suffix_bytes']
            and sha(suffix) == delta['added_suffix_sha256'], 'complete fixed appended suffix')
    for name in ADDED:
        require(delta['added_bodies'][name] == dict(bytes=len(actual[name]), sha256=sha(actual[name])),
                'complete appended HIA definition: ' + name)
    restored = source[:-len(suffix)] if suffix else source
    for name in CHANGED:
        row = delta['bodies'][name]
        before, after = row['before_body'].encode('utf8'), actual[name]
        require(before.startswith(('fn ' + name + '(').encode()) and before.endswith(b'end\n')
                and sha(before) == row['before_sha256'] and len(before) == row['before_bytes']
                and sha(after) == row['after_sha256'] and len(after) == row['after_bytes']
                and before != after and restored.count(after) == 1,
                'complete changed definition and exact parent body: ' + name)
        restored = restored.replace(after, before, 1)
    previous = functions(restored, PARENT_FUNCTIONS)
    require(sha(restored) == PARENT_SOURCE and len(restored) == PARENT_BYTES
            and list(previous) == order
            and tuple(name for name in previous if previous[name] != actual[name]) == CHANGED
            and sum(previous[name] == actual[name] for name in previous) == PARENT_FUNCTIONS - len(CHANGED),
            'every original byte and every undeclared definition is constrained by the whole parent hash')
    require(delta['unchanged_parent_body_sha256'] ==
            {name: sha(previous[name]) for name in previous if name not in CHANGED},
            'complete unchanged definition identities')
    rebuilt = restored
    for name in CHANGED:
        require(rebuilt.count(previous[name]) == 1, 'unique original definition: ' + name)
        rebuilt = rebuilt.replace(previous[name], actual[name], 1)
    require(rebuilt + suffix == source, 'complete exact source forward round trip')
    require(all(actual[name] == previous[name] for name in previous if 'backward' in name),
            'every original named backward definition is retained')
    return restored, dict(schema='heptatorch.public-f64-view-to797.complete-source-inverse.v1',
        source_sha256=SOURCE, parent_source_sha256=PARENT_SOURCE,
        functions=FUNCTIONS, parent_functions=PARENT_FUNCTIONS,
        source_bytes=SOURCE_BYTES, parent_bytes=PARENT_BYTES,
        changed_functions=list(CHANGED), appended_functions=list(ADDED),
        unchanged_parent_functions=PARENT_FUNCTIONS - len(CHANGED), delta_sha256=DELTA_SHA256,
        whole_source_inverse_exact=True, whole_round_trip_exact=True,
        complete_source_definition_order_preserved=True,
        every_other_byte_constrained_by_complete_parent_sha256=True,
        original_backward_functions_unchanged=True,
        actual_executable_replaced=False, original_cases_or_expected_values_modified=False,
        native_executed=False, Torch_executed=False, runtime_qualification_transferred=False)



def prove_original_domain(source, bind):
    """Inspect actual guarded HIA bodies for the fixed resource dtype0/1/2/3 domain.

This proves guarded source routing only. All original live resource requests,
allocation failures, owner snapshots and process/EOF checks must still run.
"""
    parent, inverse = restore_source(source, bind)
    actual, previous = functions(source, FUNCTIONS), functions(parent, PARENT_FUNCTIONS)
    require(CHANGED == ('bind_alias', 'so_tensor_live', 'to_command', 'to_strided_view', 'f64_dtype_audit'),
            'every changed original body requires an explicit original-domain proof')
    alias_old = (b'  v73:u64 = call(v45,v68) {"function":"getw"}\n'
        b'  v74:u64 = int.add.wrap(v71,v62)\n'
        b'  v75:u64 = call(v45,v74) {"function":"so_uget"}\n'
        b'  v76:bool = compare.le(v73,v75)\n'
        b'  branch v76 b5(v45,v48,v49,v50,v51,v58,v71) b7(v45,v48,v49,v50,v51,v58,v72)\n')
    alias_new = (b'  alias_high:u64 = const() {"value":9223372036854775808}\n'
        b'  alias_encoded:bool = compare.ge(v69,alias_high)\n'
        b'  branch alias_encoded b7(v45,v48,v49,v50,v51,v58,v72) ordinary_span(v45,v48,v49,v50,v51,v58,v71,v68,v62,v72)\n'
        b'block ordinary_span(as_s:mutbuf,as_node:u64,as_source:u64,as_storage:u64,as_version:u64,as_row:u64,as_storage_row:u64,as_offset_at:u64,as_size_field:u64,as_zero:u64)\n'
        b'  v73:u64 = call(as_s,as_offset_at) {"function":"getw"}\n'
        b'  v74:u64 = int.add.wrap(as_storage_row,as_size_field)\n'
        b'  v75:u64 = call(as_s,v74) {"function":"so_uget"}\n'
        b'  v76:bool = compare.le(v73,v75)\n'
        b'  branch v76 b5(as_s,as_node,as_source,as_storage,as_version,as_row,as_storage_row) b7(as_s,as_node,as_source,as_storage,as_version,as_row,as_zero)\n')
    alias=actual['bind_alias']
    require(alias.count(alias_new)==1 and alias.replace(alias_new,alias_old,1)==previous['bind_alias'],
            'complete alias change is exactly the high-bit branch and parameter-preserving ordinary block')
    require(b'  v67:u64 = const() {"value":232}\n'
            b'  v68:u64 = int.add.wrap(v49,v67)\n'
            b'  v69:u64 = call(v45,v68) {"function":"getw"}\n' in alias,
            'alias tag is read from the exact original source descriptor offset')

    live_old = (b'  v50:u64 = const() {"value":1073741824}\n'
        b'  v51:bool = compare.gt(v46,v50)\n'
        b'  branch v51 b7(v39) b8(v39,v41,v42,v46,v49)\n')
    live_new = (b'  owner_high:u64 = const() {"value":9223372036854775808}\n'
        b'  owner_encoded:bool = compare.ge(v46,owner_high)\n'
        b'  branch owner_encoded encoded_empty(v39,v41,v42) ordinary_offset(v39,v41,v42,v46,v49)\n'
        b'block encoded_empty(ee_s:mutbuf,ee_row:u64,ee_storage:u64)\n'
        b'  ee_valid:u64 = call(ee_s,ee_row,ee_storage) {"function":"so_f64_empty_offset_live"}\n'
        b'  ret(ee_valid)\n'
        b'block ordinary_offset(oo_s:mutbuf,oo_row:u64,oo_storage:u64,oo_offset:u64,oo_span:u64)\n'
        b'  oo_index_field:u64 = const() {"value":48}\n'
        b'  oo_index_at:u64 = int.add.wrap(oo_row,oo_index_field)\n'
        b'  oo_index:u64 = call(oo_s,oo_index_at) {"function":"so_uget"}\n'
        b'  oo_zero:u64 = const() {"value":0}\n'
        b'  oo_no_index:bool = compare.eq(oo_index,oo_zero)\n'
        b'  oo_total_field:u64 = const() {"value":8}\n'
        b'  oo_total:u64 = call(oo_s,oo_total_field) {"function":"getw"}\n'
        b'  oo_outside:bool = compare.ge(oo_index,oo_total)\n'
        b'  oo_no_descriptor:bool = bool.or(oo_no_index,oo_outside)\n'
        b'  branch oo_no_descriptor ordinary_byte_guard(oo_s,oo_row,oo_storage,oo_offset,oo_span) ordinary_descriptor(oo_s,oo_row,oo_storage,oo_offset,oo_span,oo_index)\n'
        b'block ordinary_descriptor(od_s:mutbuf,od_row:u64,od_storage:u64,od_offset:u64,od_span:u64,od_index:u64)\n'
        b'  od_node:u64 = call(od_s,od_index) {"function":"dptr"}\n'
        b'  od_field:u64 = const() {"value":232}\n'
        b'  od_at:u64 = int.add.wrap(od_node,od_field)\n'
        b'  od_raw:u64 = call(od_s,od_at) {"function":"getw"}\n'
        b'  od_high:u64 = const() {"value":9223372036854775808}\n'
        b'  od_mismatched_tag:bool = compare.ge(od_raw,od_high)\n'
        b'  branch od_mismatched_tag b7(od_s) ordinary_byte_guard(od_s,od_row,od_storage,od_offset,od_span)\n'
        b'block ordinary_byte_guard(ob_s:mutbuf,ob_row:u64,ob_storage:u64,ob_offset:u64,ob_span:u64)\n'
        b'  v50:u64 = const() {"value":1073741824}\n'
        b'  v51:bool = compare.gt(ob_offset,v50)\n'
        b'  branch v51 b7(ob_s) b8(ob_s,ob_row,ob_storage,ob_offset,ob_span)\n')
    live=actual['so_tensor_live']
    require(live.count(live_new)==1 and live.replace(live_new,live_old,1)==previous['so_tensor_live'],
            'complete owner lookup retains the original byte limit, span predicate and every successor argument')
    require(all(actual[name]==previous[name] for name in ('getw','so_uget','dptr')),
            'new ordinary-path guard uses only unchanged read helpers')
    require(set(re.findall(rb'\{"function":"([^"]+)"\}',live_new))==
            {b'so_f64_empty_offset_live',b'getw',b'so_uget',b'dptr'}
            and not re.search(rb'(?:store|write|allocate|alloc|syscall)\(',live_new),
            'ordinary owner path has no new write, allocation or syscall')
    # All original physical-owner contracts require byte offsets at most 1 GiB.
    # Both new tag tests are false throughout that interval. The same old byte
    # limit still rejects larger ordinary offsets; mismatched descriptor tags
    # get the existing failure result, rather than being mistaken for an alias.
    ordinary_maximum=1073741824
    tag=1 << 63
    require(0 <= ordinary_maximum < tag and ordinary_maximum // 8 == 134217728,
            'all original admitted byte offsets are disjoint from tagged empty offsets')
    for offset in (0,1,ordinary_maximum-1,ordinary_maximum):
        require(not(offset >= tag),'new tag route is false on original owner boundaries')
    old_guard = b'  v85:bool = bool.or(v45,v84)\n'
    new_guard = (b'  as_strided_opcode:u64 = const() {"value":111}\n'
        b'  other_f64_command:bool = compare.ne(v19,as_strided_opcode)\n'
        b'  rejected_f64:bool = bool.and(v45,other_f64_command)\n'
        b'  v85:bool = bool.or(rejected_f64,v84)\n')
    command = actual['to_command']
    require(command.count(new_guard) == 1 and command.replace(new_guard, old_guard, 1) == previous['to_command'],
            'only the exact F64 opcode111 refusal guard changes in the complete dispatcher')
    dtype_guard = (b'  v44:u64 = const() {"value":4}\n'
        b'  v45:bool = compare.eq(v43,v44)\n'
        b'  v80:u64 = const() {"value":5}\n'
        b'  v81:u64 = const() {"value":6}\n'
        b'  v82:bool = compare.eq(v43,v80)\n'
        b'  v83:bool = compare.eq(v43,v81)\n'
        b'  v84:bool = bool.or(v82,v83)\n')
    require(dtype_guard + new_guard in command
            and b'  branch v85 b3(v15,v14,v17,v19) b4(v13,v14,v15,v16,v17,v18)\n' in command,
            'actual boolean operands, fixed dtype constants and unchanged successor arguments')
    differing = []
    for dtype in range(7):
        for opcode in range(128):
            before = dtype == 4 or dtype in (5,6)
            after = (dtype == 4 and opcode != 111) or dtype in (5,6)
            if before != after:
                differing.append((dtype,opcode))
    require(differing == [(4,111)], 'all other public/private dtype/opcode refusal routes are identical')

    old_constructor = (b'block b12(v120:readbuf,v121:mutbuf,v122:u64,v123:u64,v124:u64,v125:u64,v126:u64)\n'
        b'  v129:u64 = call(v120,v121,v122,v123,v124) {"function":"rng_strided_view"}\n'
        b'  v130:u64 = const() {"value":0}\n'
        b'  v131:bool = compare.eq(v129,v130)\n'
        b'  branch v131 b13(v120,v121) b14(v120,v121,v125,v126,v129)\n')
    new_constructor = (b'block b12(v120:readbuf,v121:mutbuf,v122:u64,v123:u64,v124:u64,v125:u64,v126:u64)\n'
        b'  constructor_dtype:u64 = call(v121,v125) {"function":"tensor_dtype"}\n'
        b'  public_f64:u64 = const() {"value":4}\n'
        b'  direct_f64:bool = compare.eq(constructor_dtype,public_f64)\n'
        b'  branch direct_f64 f64_constructor(v120,v121,v122,v123,v124,v125,v126) guarded_constructor(v120,v121,v122,v123,v124,v125,v126)\n'
        b'block f64_constructor(fr:readbuf,fs:mutbuf,fh:u64,fp:u64,fe:u64,fi:u64,fd:u64)\n'
        b'  fv:u64 = call(fr,fs,fh,fp,fe) {"function":"public_f64_empty_strided_view"}\n'
        b'  jump constructed(fr,fs,fi,fd,fv)\n'
        b'block guarded_constructor(lr:readbuf,ls:mutbuf,lh:u64,lp:u64,le:u64,li:u64,ld:u64)\n'
        b'  lv:u64 = call(lr,ls,lh,lp,le) {"function":"rng_strided_view"}\n'
        b'  jump constructed(lr,ls,li,ld,lv)\n'
        b'block constructed(cr:readbuf,cs:mutbuf,ci:u64,cd:u64,cv:u64)\n'
        b'  cz:u64 = const() {"value":0}\n'
        b'  failed_constructor:bool = compare.eq(cv,cz)\n'
        b'  branch failed_constructor b13(cr,cs) b14(cr,cs,ci,cd,cv)\n')
    constructor=actual['to_strided_view']
    require(constructor.count(new_constructor)==1
            and constructor.replace(new_constructor,old_constructor,1)==previous['to_strided_view'],
            'complete view body has exactly one guarded constructor replacement')
    require(all(actual[name]==previous[name] for name in ('tensor_dtype','rng_strided_view',
            'scalar_previous_rng_strided_view','typed_descriptor','scalar_previous_typed_descriptor')),
            'all original shared constructors and actual dtype reader remain byte-identical')

    audit=actual['f64_dtype_audit']
    calls=re.findall(rb'\{"function":"([^"]+)"\}',audit)
    require(set(calls)=={b'scalar_previous_dtype_audit',b'getw',b'dptr'}
            and calls.count(b'scalar_previous_dtype_audit')==1
            and actual['scalar_previous_dtype_audit']==previous['scalar_previous_dtype_audit']
            and b'  prior:u64 = call(s) {"function":"scalar_previous_dtype_audit"}\n'
                b'  zero:u64 = const() {"value":0}\n'
                b'  failed:bool = compare.ne(prior,zero)\n'
                b'  branch failed done(prior) begin(s)\n' in audit
            and b'  four4:u64 = const() {"value":4}\n'
                b'  zero4:u64 = const() {"value":0}\n'
                b'  is_f644:bool = compare.eq(dtype4,four4)\n'
                b'  active4:bool = compare.ne(live4,zero4)\n'
                b'  admitted4:bool = bool.and(is_f644,active4)\n'
                b'  branch admitted4 no_ad(s4,index4,node4) next(s4,index4)\n' in audit,
            'actual audit first runs the unchanged original, then checks only live public dtype4')
    require(not re.search(rb'(?:store|write|allocate|alloc|syscall)\(',audit),
            'new dtype scan contains no additional direct write, allocation or syscall')
    fields=(24,32,40,48,56,288,384,584,592,608,616,624,632)
    for index,field in enumerate(fields):
        expected=('  offset_%d:u64 = const() {"value":%d}\n'
            '  address_%d:u64 = int.add.wrap(node5,offset_%d)\n'
            '  word_%d:u64 = call(s5,address_%d) {"function":"getw"}\n') % (
                index,field,index,index,index,index)
        require(expected.encode() in audit,'original no-AD zero field read: '+str(field))
        if index:
            before='word_0' if index==1 else 'accumulated_%d'%(index-1)
            require(('  accumulated_%d:u64 = bit.or(%s,word_%d)\n'%(index,before,index)).encode() in audit,
                    'complete no-AD zero accumulator: '+str(field))
    require(b'  forbidden5:bool = compare.ne(accumulated_12,zero5)\n'
            b'  branch forbidden5 invalid() view_state(s5,index5,node5)\n' in audit
            and all(actual[name]==previous[name] for name in ('history_audit','to_view_audit',
                'to_view_base_node','to_view_root_retain','private_tensor','tensor_row_release')),
            'every original zero predicate and all root, incarnation, Version and release audits remain')
    return dict(schema='heptatorch.public-f64-view.original-domain-paths.v1',
        source_sha256=SOURCE,parent_source_sha256=PARENT_SOURCE,
        complete_source_inverse=inverse,
        old_to_command_sha256=sha(previous['to_command']),actual_to_command_sha256=sha(command),
        actual_to_strided_view_sha256=sha(constructor),actual_f64_dtype_audit_sha256=sha(audit),
        old_bind_alias_sha256=sha(previous['bind_alias']),actual_bind_alias_sha256=sha(alias),
        old_so_tensor_live_sha256=sha(previous['so_tensor_live']),actual_so_tensor_live_sha256=sha(live),
        original_byte_offset_domain=[0,ordinary_maximum],tagged_offset_bit=63,
        ordinary_alias_span_block_parameter_inverse_exact=True,
        ordinary_owner_byte_limit_span_and_successor_arguments_retained=True,
        added_ordinary_owner_guard_uses_only_unchanged_read_helpers=True,
        original_legal_owner_offsets_never_enter_new_tag_helper=True,
        mismatched_descriptor_tag_uses_original_failure_result=True,
        changed_dispatcher_requests=[dict(dtype=4,opcode=111)],
        original_resource_dtype_domain=[0,1,2,3],
        original_domain_dispatcher_guard_unchanged=True,
        original_non_f64_constructor_and_all_arguments_retained=True,
        additional_constructor_read_only_lookup='tensor_dtype',
        original_dtype_audit_called_first_and_unchanged=True,
        new_audit_checks_only_live_public_dtype4=True,
        retained_no_AD_zero_fields=list(fields),
        original_root_incarnation_version_release_audits_unchanged=True,
        actual_executable_replaced=False,native_executed=False,Torch_executed=False,
        original_resource_requests_and_physical_readback_still_required=True,
        runtime_qualification_transferred=False)
