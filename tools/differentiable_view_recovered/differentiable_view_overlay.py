"""Recovery R1: exact614 parent, one emitted constructor and four tag replacements."""
import ast
import json
from authoring_support import functions, once, sha

PARENT09B = '4f8f73f463f681b5eb29423d71a35a0b42d807fd92bf976bbf76c1970fbcee33'
PARENT_IMAGE = '53fb43b869858f020c0b936a65f00f3b14b97d9e37038a74c357e1da471fe083'


def apply(e, combined, source, baseline_authoring, previous, out):
    before = b'hia-asm.v1 hepta.hia.program.v8 tensor_to_copy_runtime\n' + b''.join(combined.values())
    assert sha(before) == PARENT09B
    assert (previous / 'runtime.hepta').read_bytes() == before
    assert sha((previous / 'tensor-to-copy-session').read_bytes()) == PARENT_IMAGE
    assert sha(source) == '934c039186eb794b2294cf3cc4713685c14a2b1071b13834779b39e16bba3021'
    original = dict(combined)
    baseline_fragment = e.emit(ast.unparse(ast.parse(baseline_authoring).body[0]))
    assert functions(baseline_fragment) == {'to_strided_view': original['to_strided_view']}
    emitted = e.emit(source.decode())
    assert set(functions(emitted)) == {'to_strided_view'}
    (out / 'differentiable-view.hepta').write_bytes(emitted)
    combined.update(functions(emitted))
    receipts = []
    for name in ('init_state', 'state_header', 'so_state_init', 'so_state_header'):
        outer = name in ('init_state', 'state_header')
        old_tag = b'HTTOCP08' if outer else b'HTOTOCP8'
        new_tag = b'HTVAD001' if outer else b'HTOVAD01'
        old = ('{"value":' + str(int.from_bytes(old_tag, 'little')) + '}').encode()
        new = ('{"value":' + str(int.from_bytes(new_tag, 'little')) + '}').encode()
        body = combined[name]
        revised = once(body, old, new)
        assert once(revised, new, old) == body
        combined[name] = revised
        receipts.append({'function': name, 'before_sha256': sha(body),
                         'after_sha256': sha(revised), 'exact_inverse': True})
    changed = [name for name in original if original[name] != combined[name]]
    assert set(changed) == {'to_strided_view', 'init_state', 'state_header', 'so_state_init', 'so_state_header'}
    assert set(combined) == set(original) and len(combined) == 614
    rows = [{'name': name, 'parent_sha256': sha(original[name]), 'sha256': sha(body),
             'unchanged_parent09b': body == original[name]} for name, body in combined.items()]
    (out / 'direct-view-source-map.json').write_text(json.dumps(rows, indent=2) + '\n')
    (out / 'differentiable-view-custody.json').write_text(json.dumps({
        'schema': 'heptatorch.differentiable-view.overlay-custody.v1',
        'orchestration_revision': 'RECOVERED_SOURCE_R1',
        'parent_source_sha256': PARENT09B, 'parent_native_sha256': PARENT_IMAGE,
        'parent_complete_reconstruction_exact': True,
        'baseline_constructor_reemission_exact': True,
        'constructor_author_sha256': sha(source),
        'descriptor_bytes': 672, 'descriptor_words': 84,
        'new_state_format': 'HTVAD001', 'new_backing_format': 'HTOVAD01',
        'changed_functions': changed, 'added_functions': [], 'tag_receipts': receipts,
        'tensor_base_role': 'Existing640/648 ultimate-base strong Tensor ownership.',
        'initial_ad_edge': 'Operation44 captures immediate source with stored own/derived edge flags.',
        'saved_tensor_holds_added': False, 'compiler_limits_changed': False,
        'global_addf_changed': False, 'runtime_qualified': False,
        'fixed_reference_sha256': '2270d5ba925bcf91c56db94fac663695e69079c74edbf80aff8fa086e7c81665',
        'reference_compared_here': False
    }, indent=2) + '\n')
    return combined
