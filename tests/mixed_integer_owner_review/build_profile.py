#!/usr/bin/env python3
"""Independent new Saved-original-T ownership equations; preserve parent07."""
import ast,hashlib,json
from pathlib import Path


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    original=(root/'parent_owner640_integer07.py').read_bytes()
    assert sha(original)=='fd7a5be1ba4c1f9dbcfd6955e6fe5eb63860e55cdb87cd9556a969be4c8236d0'
    text=original.decode();edits=[]
    def replace(name,before,after,count=1):
        nonlocal text
        assert text.count(before)==count,(name,text.count(before),count)
        text=text.replace(before,after);edits.append({'name':name,'before':before,'after':after,'count':count})
    for name,old,new,count in [
        ('schema','heptatorch.integer-owner-observer07.v1','heptatorch.mixed-integer-owner-observer10.v1',1),
        ('profile','integer-tensor-544-general-owner07','mixed-integer-tensor-556-general-owner10',1),
        ('source','adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62','5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49',1),
        ('image','b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08','69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6',1),
        ('state','HTINT002','HTMXAD02',2),('backing','HTOINT02','HTOMXD02',2)]:
        replace(name,old,new,count)
    replace('raw_saved_Tensor_pairs',
        '"optimizer_blobs": {}, "generator_states": {}, "saved_layouts": {},',
        '"optimizer_blobs": {}, "generator_states": {}, "saved_layouts": {}, "binary_saved_roots": {},')
    replace('decode_real_private_Hold_bytes',
        '        result["edges"][key] = edges\n',
        '''        result["edges"][key] = edges
        if w(node, 584) > 65535:
            original_tensor_pairs = hold(w(node, 584))
            need(len(original_tensor_pairs) == 32, "decode-binary-Saved-Hold-size")
            result["binary_saved_roots"][key] = list(struct.unpack('<4Q', original_tensor_pairs))
''')
    replace('empty_no_unreported_roles',
        '"optimizer_blobs", "generator_states", "saved_layouts")), "empty-owner")',
        '"optimizer_blobs", "generator_states", "saved_layouts", "binary_saved_roots")), "empty-owner")')
    replace('binary_pair_inventory',
        'accumulator_arcs, weak_hooks, masks, integer_views = [], [], [], []',
        '''accumulator_arcs, weak_hooks, masks, integer_views, binary_saved_edges = [], [], [], [], []
    need(set(snapshot["binary_saved_roots"]) == {str(i) for i,n in live.items() if w(n,584)>65535},
         "binary-Saved-pair-coverage")''')
    replace('rowless_I64_metadata_cannot_hold_real_base',
        '        integer_view = w(node, 576) == 2 and w(node, 408) == 1',
        '''        integer_view = w(node, 576) == 2 and w(node, 408) == 1 and w(node, 240) == 1
        if w(node,576) == 2 and w(node,408) == 1 and w(node,240) == 0:
            need(mask == 0 and w(node,592) == 0, "I64-rowless-view-base-retained", node=i)''')
    replace('new_Hold_role_strict_before_old_node_roles',
        '        if integer_view:\n',
        '''        if mask > 65535:
            region = private_hold(mask, f"binary-Saved:{i}", 32)
            need(w(node, 592) == TAG + region[2], "binary-Saved-relative-address", node=i)
            need(w(node, 576) == 0 and w(node, 32) == 1 and op in (3,4)
                 and w(node, 328) == 0, "binary-Saved-owner-role", node=i)
            values = snapshot["binary_saved_roots"][str(i)]
            need(len(values) == 4 and all(type(x) is int and 0 <= x <= MAX_U64 for x in values),
                 "binary-Saved-pair-width", node=i)
            need(sha(struct.pack('<4Q',*values)) == snapshot["hold_regions"][str(mask)]["sha256"],
                 "binary-Saved-Hold-content", node=i)
            populated = 0
            for slot in range(2):
                target, incarnation = values[2*slot:2*slot+2]
                saved = w(node,280+8*slot)
                needed = w(node,80) if slot == 0 else (w(node,72)|w(node,80) if op==4 else w(node,72))
                need(bool(target) == bool(saved) == bool(needed), "binary-Saved-slot-presence", node=i, slot=slot)
                if target:
                    tn = tensor(target,incarnation,"binary-Saved-original-Tensor-incarnation")
                    need(incarnation < w(node,320), "binary-Saved-original-order", node=i, slot=slot)
                    sr = lookup(5,saved)
                    need(sr[2] == w(tn,216) and sr[3] == w(tn,224)
                         and sr[5] == w(tn,232) and sr[7] == 0,
                         "binary-Saved-original-S-V-offset", node=i, slot=slot)
                    owned_refs[target] += 1; strong_arcs[i].append(target)
                    binary_saved_edges.append([i,slot,target,incarnation,saved,mask])
                    populated += 1
                else:
                    need(incarnation == 0, "binary-Saved-absent-incarnation", node=i, slot=slot)
            need(populated > 0,"binary-Saved-empty-Hold",node=i)
        elif integer_view:
''')
    replace('report_actual_Saved_Tensor_arcs',
        '"I64_ultimate_base_edges": integer_views,',
        '"I64_ultimate_base_edges": integer_views, "binary_Saved_original_Tensor_edges": binary_saved_edges,')
    replace('new_actual_differentiable_out_of_place_contract',
        'and 1 <= command[1] <= 7 and w(left, 32) == w(right, 32) == 0,',
        'and 1 <= command[1] <= 7 and (code == 103 or w(left, 32) == w(right, 32) == 0),')
    replace('explicit_existing_gradient_read_has_zero_descriptors',
        'elif code in (13, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 54,',
        'elif code in (13, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 54,')
    replace('zero_allocation_actual_Saved_accessor',
        '    elif code == 106:\n',
        '''    elif code == 107:
        _, node = public_argument(command[1])
        need(command[2] in (0,1) and not any(command[3:]), "Saved-accessor-command-shape")
        allocated = 0  # publish_tensor toggles an existing live T, no new descriptor
    elif code == 106:
''')
    reverse=text
    for e in reversed(edits):
        assert reverse.count(e['after'])==e['count'],e['name']
        reverse=reverse.replace(e['after'],e['before'])
    assert reverse.encode()==original
    a={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    b={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    changed=sorted(n for n in a if a[n]!=b[n]);assert changed==['decode_owner','pin_profile','verify_isolated_transition','verify_snapshot']
    path=root/'owner640_mixed10.py';path.write_text(text)
    receipt={'status':'SOURCE_ADAPTED_NOT_NATIVE_QUALIFIED','parent07_sha256':sha(original),
             'source_sha256':sha(path.read_bytes()),'inverse_exact_parent07':True,
             'changed_definitions':changed,'unchanged_definitions':sorted(set(a)-set(changed)),
             'edits':edits,'script_sha256':sha(Path(__file__).read_bytes()),
             'source_native_sha256':'5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49',
             'native_image_sha256':'69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6'}
    (root/'adaptation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('edits','unchanged_definitions')}))


if __name__=='__main__':main()
