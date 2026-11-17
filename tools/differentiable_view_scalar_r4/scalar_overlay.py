"""R4 successor: genuine wrapped Scalar64, precise cast, original Saved Tensor."""
import json
from authoring_support import functions, once, sha
PARENT_SOURCE='67307c29c29d3f35c288f1775d883b179a6d1ab6497ac8da004b3a713b20db21'
PARENT_IMAGE='ef877aa61320746f71956fd34b4d18d5cf4e164e5a0274e3c98301d4fd288ba1'
MATH_SHA='b410bef7152009f98d58559598e223dfe0d0f84cbd0fc91a66e9f5a0354707f0'
RUNTIME_SHA='0b9d5e40bdf79d103b4fb4adbb7aa21de63e52c15fc4aff9e3b97bb9a54a44cc'

def apply(e, combined, math, runtime, previous, out):
    original=dict(combined)
    assert sha((previous/'runtime.hepta').read_bytes())==PARENT_SOURCE
    assert sha(math)==MATH_SHA and sha(runtime)==RUNTIME_SHA
    receipts=[]
    aliases={
      'typed_descriptor':'scalar_previous_typed_descriptor',
      'typed_from_wire':'scalar_previous_typed_from_wire',
      'dtype_audit':'scalar_previous_dtype_audit',
      'to_copy_tensor':'scalar_previous_to_copy_tensor',
      'transpose':'scalar_previous_transpose',
      'command_argument':'scalar_previous_command_argument',
      'typed_command':'scalar_previous_typed_command',
      'to_command':'scalar_previous_to_command',
      'data_alias':'scalar_previous_data_alias',
      'save_public':'scalar_previous_save_public',
      'rng_strided_view':'scalar_previous_rng_strided_view',
      'back_binary':'scalar_previous_back_binary',
      'incoming_describe':'scalar_previous_incoming_describe',
      'incoming_mark_preserve_targets':'scalar_previous_incoming_mark_preserve_targets',
    }
    for old,new in aliases.items():
        before=original[old]
        body=once(before,('fn '+old+'(').encode(),('fn '+new+'(').encode())
        inverse=body
        edits=[]
        if old=='dtype_audit':
            a=b'  v89:u64 = const() {"value":2}\n'
            b=b'  v89:u64 = const() {"value":3}\n'
            body=once(body,a,b)
            inverse=once(body,b,a)
            edits.append({'old':a.decode(),'new':b.decode()})
        assert once(inverse,('fn '+new+'(').encode(),('fn '+old+'(').encode())==before
        combined[new]=body
        e.add_context(new,body)
        receipts.append({'kind':'parent-alias','original':old,'alias':new,
           'parent_sha256':sha(before),'sha256':sha(body),'edits':edits,'exact_inverse':True})
    base=original['back_binary']
    assert sha(base)=='9d0c78d3b2a1d3f3b06429b8c61a171ae0f724110645ed0d4a36f0f9ac3dd0f5'
    clone=once(base,b'fn back_binary(',b'fn wrapped_scalar_back_binary(')
    mul_edits=[]
    for a in (b'  v122:u64 = call(v105,v121) {"function":"mulf"}\n',
              b'  v301:u64 = call(v284,v300) {"function":"mulf"}\n'):
        b=a.replace(b'"mulf"',b'"wrapped_mul_f32"')
        clone=once(clone,a,b);mul_edits.append((a,b))
    inv=clone
    for a,b in reversed(mul_edits):inv=once(inv,b,a)
    assert once(inv,b'fn wrapped_scalar_back_binary(',b'fn back_binary(')==base
    combined['wrapped_scalar_back_binary']=clone
    e.add_context('wrapped_scalar_back_binary',clone)
    receipts.append({'kind':'wrapped-scalar-multiply-backward-only','parent_sha256':sha(base),
      'sha256':sha(clone),'exact_inverse':True,
      'callsite_edits':[{'old':a.decode(),'new':b.decode()} for a,b in mul_edits],
      'dispatch_guard':'actual op3 and right operand dtype3; all other binary paths use exact parent'})
    fragment=e.emit((math+b'\n'+runtime).decode())
    emitted=functions(fragment)
    replacements=set(aliases)|{'dtype_width','binary_backward_read'}
    assert set(original)&set(emitted)==replacements
    assert len(emitted)==21
    combined.update(emitted)
    (out/'wrapped-scalar.hepta').write_bytes(fragment)
    before=original['so_audit_records']
    body=before
    edits=[]
    for reg in ('v487','v555'):
        a=('  '+reg+':u64 = const() {"value":2}\n').encode()
        b=('  '+reg+':u64 = const() {"value":3}\n').encode()
        body=once(body,a,b);edits.append((a,b))
    inverse=body
    for a,b in reversed(edits):inverse=once(inverse,b,a)
    assert inverse==before
    combined['so_audit_records']=body
    receipts.append({'kind':'public-dtype-enum','function':'so_audit_records',
       'parent_sha256':sha(before),'sha256':sha(body),'exact_inverse':True,
       'edits':[{'old':a.decode(),'new':b.decode()} for a,b in edits],
       'private_hold_dtype_zero_guard_unchanged':True})
    before,after=b'{"value":113}',b'{"value":114}'
    entry=once(original['entry'],before,after)
    assert once(entry,after,before)==original['entry']
    combined['entry']=entry
    receipts.append({'kind':'public-dispatch','function':'entry','parent_sha256':sha(original['entry']),
       'sha256':sha(entry),'old':before.decode(),'new':after.decode(),'exact_inverse':True})
    for name in ('init_state','state_header','so_state_init','so_state_header'):
        outer=name in ('init_state','state_header')
        old_tag=b'HTVSHP01' if outer else b'HTOVSHP1'
        new_tag=b'HTVSHP02' if outer else b'HTOVSHP2'
        a=('{"value":'+str(int.from_bytes(old_tag,'little'))+'}').encode()
        b=('{"value":'+str(int.from_bytes(new_tag,'little'))+'}').encode()
        body=once(original[name],a,b)
        assert once(body,b,a)==original[name]
        combined[name]=body
        receipts.append({'kind':'format-tag','function':name,'parent_sha256':sha(original[name]),
           'sha256':sha(body),'old':a.decode(),'new':b.decode(),'exact_inverse':True})
    before=original['retained_apply']
    old_call=b'  v220:u64 = call(v219,v209) {"function":"addf"}\n'
    new_call=b'  v220:u64 = call(v219,v209) {"function":"scalar_add_preserving_bits"}\n'
    assert sha(original['scalar_add_preserving_bits'])=='b9d272f526920879e66970641f60cfcc0fc1d71aaf3c285e2781a5a909c6a436'
    body=once(before,old_call,new_call)
    assert once(body,new_call,old_call)==before
    combined['retained_apply']=body
    receipts.append({'kind':'retained-ieee-addition','function':'retained_apply',
        'parent_sha256':sha(before),'sha256':sha(body),'exact_inverse':True,
        'old':old_call.decode(),'new':new_call.decode(),'global_addf_unchanged':True})
    changed={n for n in original if original[n]!=combined[n]}
    assert changed==replacements|{'retained_apply','so_audit_records','entry','init_state','state_header','so_state_init','so_state_header'}
    for name in ('integer_binary','integer_result_dtype','integer_promote_operand','integer_as_f32',
       'to_convert_value','divf','addf','mulf','dist_probability_to32',
       'shape_transform','shape_compute_strides','shape_infer_dimensions','shape_fresh_alias_version',
       'record','binary_saved_bind','binary_saved_release','binary_saved_get',
       'to_as_strided_backward','history_capture','to_view_refresh'):
        assert combined[name]==original[name],name
    (out/'overlay-inverse.json').write_text(json.dumps(receipts,indent=2)+'\n')
    (out/'wrapped-scalar-custody.json').write_text(json.dumps({
      'schema':'heptatorch.wrapped-scalar64.source.v1','parent_source_sha256':PARENT_SOURCE,
      'parent_native_sha256':PARENT_IMAGE,'math_sha256':sha(math),'runtime_sha256':sha(runtime),
      'changed_parent_functions':sorted(changed),'added_functions':sorted(set(combined)-set(original)),
      'descriptor_bytes':672,'state_format':'HTVSHP02','backing_format':'HTOVSHP2',
      'new_commands':[114],'dtype3':'internal rank-zero float64 wrapped scalar; no requires_grad',
      'saved_tensor':'original eight-byte scalar, never promoted F32 replacement',
      'generic_103_dtype3_still_rejected':True,'new_transpose_preserves_TBackward':True,
      'runtime_qualified':False,'native_tests_run_here':False,'torch_executed':False,
      'reference_sha256':'fecf1956790f78a9b1238f535c1c7cce9f1237b211c389ac0736d88687bbb9c9',
      'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590
    },indent=2)+'\n')
    return combined
