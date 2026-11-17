"""R3 input geometry: native materialization, exact parent aliases and scoped arithmetic."""
import json
from authoring_support import functions, once, sha

PARENT_SOURCE = '197bd52db4815f575ea3e9055d3f5556900314bd2f4fbb262e80cea76ea16ef2'
PARENT_IMAGE = '13cee2163e8f62081db35e831885714789ac5117fb80c030a1ccaf7194e0821c'
AUTHOR_SHA = 'bdd5c9d06093db032c3cb25ed541449f531145288ce56f6b97d3ca634064137e'
HELPER_SHA = '4de09bb940b9e63f625c5629583b7dce7c1487c2bf3b84a88a1faae9c360fb34'

def apply(e, combined, source, helper, previous, out):
    original = dict(combined)
    parent_raw = (previous / 'runtime.hepta').read_bytes()
    assert sha(parent_raw) == PARENT_SOURCE
    assert functions(parent_raw) == original
    assert sha(source) == AUTHOR_SHA and sha(helper) == HELPER_SHA
    receipts = []
    aliases = {
        'incoming_describe': 'dview_previous_incoming_describe',
        'incoming_construct': 'dview_previous_incoming_construct',
        'incoming_mark_preserve_targets': 'dview_previous_mark_preserve_targets',
    }
    for old, new in aliases.items():
        assert new not in combined
        body = once(original[old], ('fn '+old+'(').encode(), ('fn '+new+'(').encode())
        assert once(body, ('fn '+new+'(').encode(), ('fn '+old+'(').encode()) == original[old]
        combined[new] = body
        e.add_context(new, body)
        receipts.append({'kind':'parent-alias','original':old,'alias':new,
                         'parent_sha256':sha(original[old]),'sha256':sha(body),
                         'exact_inverse':True})
    assert set(functions(helper)) == {'dview_coverage_divide'}
    combined.update(functions(helper))
    e.add_context('dview_coverage_divide', helper)
    fragment = e.emit(source.decode())
    emitted = functions(fragment)
    replaced = {'incoming_describe','incoming_construct','incoming_mark_preserve_targets',
                'to_as_strided_backward'}
    assert set(original).intersection(emitted) == replaced
    combined.update(emitted)
    (out/'incoming-geometry.hepta').write_bytes(fragment)
    reserve = original['incoming_reserve_preserve_storage']
    old = b'  v20:u64 = int.mul.wrap(v18,v19)\n'
    new = b'  v20:u64 = call(v11,v13) {"function":"dview_preserve_capacity"}\n'
    revised = once(reserve,old,new)
    assert once(revised,new,old) == reserve
    combined['incoming_reserve_preserve_storage'] = revised
    receipts.append({'kind':'reserve-size','function':'incoming_reserve_preserve_storage',
                     'parent_sha256':sha(reserve),'sha256':sha(revised),'exact_inverse':True,
                     'old':old.decode(),'new':new.decode()})
    for name in ('init_state','state_header','so_state_init','so_state_header'):
        outer = name in ('init_state','state_header')
        before = b'HTVAD002' if outer else b'HTOVAD02'
        after = b'HTVAD003' if outer else b'HTOVAD03'
        old = ('{"value":'+str(int.from_bytes(before,'little'))+'}').encode()
        new = ('{"value":'+str(int.from_bytes(after,'little'))+'}').encode()
        changed = once(original[name],old,new)
        assert once(changed,new,old) == original[name]
        combined[name] = changed
        receipts.append({'kind':'format-tag','function':name,'parent_sha256':sha(original[name]),
                         'sha256':sha(changed),'exact_inverse':True})
    changed = {n for n in original if combined[n] != original[n]}
    assert changed == replaced | {'incoming_reserve_preserve_storage','init_state','state_header',
                                   'so_state_init','so_state_header'}
    for name in ('gradient_accumulate','incoming_obeys_contract','incoming_adopt_gradient',
                 'incoming_drop_cache','to_zero_factory_incoming','divf','addf',
                 'to_strided_view','to_view_refresh'):
        assert combined[name] == original[name]
    added = sorted(set(combined)-set(original))
    (out/'overlay-inverse.json').write_text(json.dumps(receipts,indent=2)+'\n')
    (out/'incoming-geometry-custody.json').write_text(json.dumps({
        'schema':'heptatorch.direct-view-input-geometry-source.v1',
        'parent_source_sha256':PARENT_SOURCE,'parent_native_sha256':PARENT_IMAGE,
        'author_sha256':sha(source),'coverage_helper_sha256':sha(helper),
        'changed_parent_functions':sorted(changed),'added_functions':added,
        'descriptor_bytes':672,'state_format':'HTVAD003','backing_format':'HTOVAD03',
        'global_divf_unchanged':True,'global_addf_unchanged':True,
        'gradient_layout_contract_unchanged':True,'real_private_cache_tail_release':True,
        'public_gradient_version_reset':False,'empty_factory_branch_unchanged':True,
        'runtime_qualified':False,'native_tests_run_here':False,'torch_executed':False,
        'reference_sha256':'f4abe6a750ecc791b3a7bc7aa507b4c77eb0a2fcccf45b0c6a828db0e01490d3',
    },indent=2)+'\n')
    return combined
