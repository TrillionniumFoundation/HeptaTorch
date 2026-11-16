#!/usr/bin/env python3
"""Author a native flat-image linker from the frozen borrowed-module gate.

This is a source adapter, not a host linker. Object admission, symbol/effect
matching, relocation and output publication execute in the emitted HIA tool.
"""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import json
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
MAGIC = b'HIAFLK01'
EXPECTED_ADAPTER = 'a9046f0ee8c6489abd34afc41b5dd572e7174e05d8a8998f5e4ba76f8aa9dad8'
EXPECTED_ADDITIONS = 'c780a60a420eeac8ccdfdd20ea345fdd280e23792b2e6b67177f5c1fc8f09890'
EXPECTED_AUTHORING = '0b6cba389162e8aa25c11b160db259e09ed816a3f5d5dc65240230b7b3ea58ef'
EXPECTED_HIA = '5d32ecd85c32388b649e34038e423996c78011a3ccd8b2cbe17e4613c2fdc9d4'

# Exact upstream flat wrapper: six SysV words become three 16-byte descriptors
# at rbp-64, rbp-48, rbp-32. rdi=argument area, rsi=result slot rbp-8.
# Only the call's rel32 displacement at byte65 is filled by the native linker.
BRIDGE = bytes.fromhex(
    '55 48 89 e5 48 83 ec 40 '
    '48 89 bd c0 ff ff ff 48 89 b5 c8 ff ff ff '
    '48 89 95 d0 ff ff ff 48 89 8d d8 ff ff ff '
    '4c 89 85 e0 ff ff ff 4c 89 8d e8 ff ff ff '
    '48 8d bd c0 ff ff ff 48 8d b5 f8 ff ff ff '
    'e8 00 00 00 00 48 8b 85 f8 ff ff ff c9 c3')
assert len(BRIDGE) == 78 and BRIDGE[64] == 0xe8


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--capacity-adaptation', type=Path, required=True)
    p.add_argument('--module-tool-dir', type=Path,
                   default=Path(__file__).resolve().parents[1]/'modules')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    modules = args.module_tool_dir.resolve(); upstream = args.source_root.resolve()
    assert digest((modules/'build_module_linker.py').read_bytes()) == EXPECTED_ADAPTER
    assert digest((modules/'module_linker_additions.txt').read_bytes()) == EXPECTED_ADDITIONS
    baseline = output/'borrowed-baseline'
    result = subprocess.run([sys.executable,'-I','-S','-B',str(modules/'build_module_linker.py'),
                             '--source-root',str(upstream),'--capacity-adaptation',
                             str(args.capacity_adaptation.resolve()),'--output',str(baseline)],
                            capture_output=True,timeout=30)
    (output/'baseline-authoring.stdout').write_bytes(result.stdout)
    (output/'baseline-authoring.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr
    original = (baseline/'linker-authoring.txt').read_text()
    original_hia = (baseline/'module-linker.hepta').read_bytes()
    assert digest(original.encode()) == EXPECTED_AUTHORING
    assert digest(original_hia) == EXPECTED_HIA
    receipt = json.loads((baseline/'linker-authoring.json').read_text())
    closure = receipt['upstream_source_files_verified_before_and_after']
    for path, expected in closure.items():
        assert digest((upstream/path).read_bytes()) == expected
    sys.path.insert(0,str(upstream))
    from tools import generate_hia_elf_linker as generator
    from tools.generate_hia_selfhost import compact_authoring_cfg,total_scalar_functions
    from hepta.hia_agent_assembly import render_assembly,assemble_program

    def document(source, name):
        nodes = ast.parse(source).body
        names = {node.name:'f'+str(i) for i,node in enumerate(nodes)}
        signatures = {n.name:([a.annotation.id for a in n.args.args],n.returns.id) for n in nodes}
        total = total_scalar_functions(nodes); functions=[]
        for node in nodes:
            b = generator.LinkerBuilder(node,names,signatures,total); b.statements(node.body)
            functions.append({'name':names[node.name],'params':[b.param(v) for v in b.params],
                              'returns':[b.returns],'effects':['read','write','trap'],
                              'entry':'b0','blocks':b.blocks})
        return compact_authoring_cfg({'schema':'hepta.hia.program.v8','module':name,'functions':functions})

    assert render_assembly(document(original,'borrowed_module_linker')).encode() == original_hia
    old_magic = str(int.from_bytes(b'HIABLK01','little'))
    assert original.count(old_magic) == 1
    modified = original.replace(old_magic,str(int.from_bytes(MAGIC,'little')),1)
    header = ['def headers(o:mutbuf, w:mutbuf) -> u64:']
    for offset in range(0,len(BRIDGE),8):
        chunk=BRIDGE[offset:offset+8]
        header.append(f'    write(o, {offset}, {int.from_bytes(chunk,"little")}, {len(chunk)})')
    header += ['    write(o, 65, get(w, 88) - 69, 4)','    return 0']
    start=modified.index('\ndef headers('); end=modified.index('\ndef property_note(',start)
    modified=modified[:start]+'\n'+'\n'.join(header)+'\n'+modified[end:]
    old_nodes={n.name:ast.get_source_segment(original,n) for n in ast.parse(original).body}
    new_nodes={n.name:ast.get_source_segment(modified,n) for n in ast.parse(modified).body}
    changed_authoring=[n for n in old_nodes if old_nodes[n]!=new_nodes[n]]
    assert changed_authoring == ['link','headers'] and len(new_nodes)==27
    assert new_nodes['link'].replace(str(int.from_bytes(MAGIC,'little')),old_magic,1) == old_nodes['link']
    doc=document(modified,'borrowed_flat_module_linker')
    hia=render_assembly(doc).encode(); assert assemble_program(hia).as_dict()==doc
    profile=json.loads((Path(__file__).resolve().parent/'profile.json').read_text())
    assert digest(hia)==profile['flat_linker_source_sha256']
    extract=lambda data:dict(re.findall(rb'fn ([^(]+)(\(.*?\nend\n?)',data,re.S))
    old_functions,new_functions=extract(original_hia),extract(hia)
    changed_hia=[n.decode() for n in old_functions if old_functions[n]!=new_functions[n]]
    assert changed_hia == ['f0','f6']
    (output/'flat-linker-authoring.txt').write_text(modified)
    (output/'flat-linker-authoring.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True),modified.splitlines(True),fromfile='borrowed-linker',tofile='flat-linker')))
    (output/'flat-linker.hepta').write_bytes(hia)
    (output/'entry-bridge.bin').write_bytes(BRIDGE)
    (output/'linker-packer.hepta').write_bytes((baseline/'linker-packer.hepta').read_bytes())
    functions=doc['functions']
    counts={'functions':len(functions),'blocks':sum(len(f['blocks']) for f in functions),
            'instructions':sum(len(b['ops']) for f in functions for b in f['blocks']),
            'source_bytes':len(hia)}
    assert counts['functions']<=256 and counts['blocks']<=2560 and counts['instructions']<=8192
    for path, expected in closure.items():
        assert digest((upstream/path).read_bytes()) == expected
    assert digest((modules/'build_module_linker.py').read_bytes()) == EXPECTED_ADAPTER
    assert digest((modules/'module_linker_additions.txt').read_bytes()) == EXPECTED_ADDITIONS
    report={'schema':'heptatorch.borrowed-flat-linker-authoring.v1',
            'native_compilation_performed':False,'native_linking_performed':False,
            'adapter_sha256':digest(Path(__file__).read_bytes()),
            'frozen_module_adapter_sha256':EXPECTED_ADAPTER,
            'frozen_module_additions_sha256':EXPECTED_ADDITIONS,
            'frozen_module_linker_source_sha256':EXPECTED_HIA,
            'frozen_module_linker_authoring_sha256':EXPECTED_AUTHORING,
            'flat_linker_source_sha256':digest(hia),'source_structure':counts,
            'changed_authoring_functions':changed_authoring,'changed_hia_functions':changed_hia,
            'unchanged_hia_functions':25,'all_native_admission_functions_unchanged':True,
            'link_change_only_wire_magic':True,'wire_magic':MAGIC.decode(),
            'object_interface_magic':'HIAOBJ01','object_interface_version':1,
            'bridge_bytes':len(BRIDGE),'bridge_call_displacement_offset':65,
            'bridge_call_return_offset':69,'bridge_sha256':digest(BRIDGE),
            'object_text_origin':1024,'output_format':'raw position-independent image',
            'entry_parameters':['readbuf','mutbuf','mutbuf'],'entry_result':'u32',
            'source_objects_must_come_from_verified_native_compiler':True,
            'limits':{'objects':64,'input_bytes':8<<20,'output_bytes':8<<20,'workspace_bytes':16<<20},
            'upstream_source_files_verified_before_and_after':closure,
            'packer_source_sha256':digest((output/'linker-packer.hepta').read_bytes())}
    (output/'flat-linker-authoring.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='upstream_source_files_verified_before_and_after'},indent=2))


if __name__=='__main__':main()
