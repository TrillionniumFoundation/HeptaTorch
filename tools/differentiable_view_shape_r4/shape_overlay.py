"""R4 concrete view/reshape overlay. Runtime shape and Tensor work remains HIA."""
import json
from authoring_support import functions, once, sha

PARENT_SOURCE = '3c2c0a125bcac6d47075bf58fc3f78c0e7b9a0b1370757981b99dcd3793b21c6'
PARENT_IMAGE = '7841b19051960b6fa13cb98841dbb8e4d6ead97f03e2ad65af415e1a35b03221'
GEOMETRY_SHA = '5861fa3e46add8b04747c64456a6595ff833e3ab5102bb011a83a7ff1f634613'
CONSTRUCTORS_SHA = '21d13c9a959bd33d83fb9244c233518fe8ce969f89fbb38c7cd8f871c9581b53'

def apply(e, combined, geometry, constructors, previous, out):
    original = dict(combined)
    raw = (previous / 'runtime.hepta').read_bytes()
    assert sha(raw) == PARENT_SOURCE and functions(raw) == original
    assert sha(geometry) == GEOMETRY_SHA and sha(constructors) == CONSTRUCTORS_SHA
    aliases = {
        'to_command': 'shape_previous_to_command',
        'partial_apply_node': 'shape_previous_partial_apply_node',
        'gradient_direct': 'shape_previous_gradient_direct',
        'incoming_describe': 'shape_previous_incoming_describe',
    }
    receipts = []
    for old, new in aliases.items():
        assert new not in combined
        before = ('fn '+old+'(').encode()
        after = ('fn '+new+'(').encode()
        body = once(original[old], before, after)
        assert once(body, after, before) == original[old]
        combined[new] = body
        e.add_context(new, body)
        receipts.append({'kind':'parent-alias','original':old,'alias':new,
                         'parent_sha256':sha(original[old]),'sha256':sha(body),
                         'exact_inverse':True})
    fragment = e.emit((geometry + b'\n' + constructors).decode())
    emitted = functions(fragment)
    assert set(original).intersection(emitted) == set(aliases)
    assert len(emitted) == 11
    combined.update(emitted)
    (out/'shape-view.hepta').write_bytes(fragment)
    before, after = b'{"value":111}', b'{"value":113}'
    entry = once(original['entry'],before,after)
    assert once(entry,after,before) == original['entry']
    combined['entry'] = entry
    receipts.append({'kind':'public-dispatch','function':'entry','parent_sha256':sha(original['entry']),
                     'sha256':sha(entry),'old':before.decode(),'new':after.decode(),'exact_inverse':True})
    for name in ('init_state','state_header','so_state_init','so_state_header'):
        outer = name in ('init_state','state_header')
        old_tag = b'HTVAD003' if outer else b'HTOVAD03'
        new_tag = b'HTVSHP01' if outer else b'HTOVSHP1'
        old = ('{"value":'+str(int.from_bytes(old_tag,'little'))+'}').encode()
        new = ('{"value":'+str(int.from_bytes(new_tag,'little'))+'}').encode()
        body = once(original[name],old,new)
        assert once(body,new,old) == original[name]
        combined[name] = body
        receipts.append({'kind':'format-tag','function':name,'parent_sha256':sha(original[name]),
                         'sha256':sha(body),'old':old.decode(),'new':new.decode(),'exact_inverse':True})
    changed = {n for n in original if original[n] != combined[n]}
    assert changed == set(aliases) | {'entry','init_state','state_header','so_state_init','so_state_header'}
    for name in ('to_strided_view','to_view_refresh','record','record_body','history_original_record',
                 'incoming_reshape_layout','gradient_accumulate','incoming_obeys_contract',
                 'incoming_adopt_gradient','to_as_strided_backward','divf','addf','dview_coverage_divide'):
        assert combined[name] == original[name], name
    (out/'overlay-inverse.json').write_text(json.dumps(receipts,indent=2)+'\n')
    (out/'shape-view-custody.json').write_text(json.dumps({
        'schema':'heptatorch.concrete-view-reshape.source.v1',
        'parent_source_sha256':PARENT_SOURCE,'parent_native_sha256':PARENT_IMAGE,
        'geometry_sha256':sha(geometry),'constructors_sha256':sha(constructors),
        'changed_parent_functions':sorted(changed),'added_functions':sorted(set(combined)-set(original)),
        'descriptor_bytes':672,'state_format':'HTVSHP01','backing_format':'HTOVSHP1',
        'new_commands':[112,113],'new_autograd_ops':{'45':'ReshapeAliasBackward','46':'UnsafeViewBackward'},
        'existing_view_autograd_op':16,'as_strided_autograd_op44_reused_for_new_ops':False,
        'runtime_qualified':False,'native_tests_run_here':False,'torch_executed':False,
        'reference_sha256':'fecf1956790f78a9b1238f535c1c7cce9f1237b211c389ac0736d88687bbb9c9',
        'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590,
    },indent=2)+'\n')
    return combined
