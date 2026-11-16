#!/usr/bin/env python3
"""Narrow independent review of borrowed flat-linker entry/PIC/refusal paths."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import struct
import subprocess
import sys

EXPECTED_LINKER = '4c36e3409b397788664f1501896aff2719f667306795cb5fa95061d710fd6139'
EXPECTED_IMAGE = '56e592b7ee6c797a805e5e63a9814a411f0792a1f05c79f77087d18e08278b61'
EXPECTED_SOURCE = 'f2096b101480369d8e51b507fe1d6266b29818ebcd741c83ef6b7f8de4a20a19'
EXPECTED_OLD_SOURCE = '5d32ecd85c32388b649e34038e423996c78011a3ccd8b2cbe17e4613c2fdc9d4'
EXPECTED_OBSERVER = '79305c1d817bb0702b8ad0777f9bd62f82737134f4e850f9ad8d2cc990b4b0b0'
CHILD = r'''from pathlib import Path
import ctypes,hashlib,json,mmap,resource,sys
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(256<<20,256<<20))
root=Path(sys.argv[1]);cfg=json.loads((root/'execute.json').read_text());raw=(root/'application.flat').read_bytes()
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(raw)==cfg['image_sha256']
libc=ctypes.CDLL(None);libc.mprotect.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
regions=[];rows=[]
for shift in (0,4096):
 code=mmap.mmap(-1,len(raw)+4096,prot=mmap.PROT_READ|mmap.PROT_WRITE)
 code[shift:shift+len(raw)]=raw
 address=ctypes.addressof(ctypes.c_char.from_buffer(code))
 assert libc.mprotect(address,len(raw)+4096,mmap.PROT_READ|mmap.PROT_EXEC)==0
 regions.append(code)
 data=bytes.fromhex(cfg['input_hex']);expected=bytes.fromhex(cfg['expected_hex'])
 source=ctypes.create_string_buffer(data,max(1,len(data)))
 output=ctypes.create_string_buffer(bytes([0xA5])*96)
 workspace=ctypes.create_string_buffer(bytes([0x5A])*128)
 fn=ctypes.CFUNCTYPE(ctypes.c_uint64,*([ctypes.c_void_p,ctypes.c_uint64]*3))(address+shift)
 result=int(fn(ctypes.addressof(source),len(data),ctypes.addressof(output)+32,32,ctypes.addressof(workspace)+32,64))
 actual=output.raw[32:32+result] if result<=32 else b''
 row={'mapping_entry_address':address+shift,'mapping_prefix_bytes':shift,'result':result,
      'actual_hex':actual.hex(),'source_unchanged':source.raw[:len(data)]==data,
      'output_guards_unchanged':output.raw[:32]==bytes([0xA5])*32 and output.raw[64:96]==bytes([0xA5])*32,
      'workspace_guards_unchanged':workspace.raw[:32]==bytes([0x5A])*32 and workspace.raw[96:128]==bytes([0x5A])*32}
 assert result==len(expected) and actual==expected and row['source_unchanged'] and row['output_guards_unchanged'] and row['workspace_guards_unchanged'],row
 rows.append(row)
assert rows[0]['mapping_entry_address']!=rows[1]['mapping_entry_address']
print(json.dumps({'status':'PASS','two_different_mapping_addresses':True,'rows':rows}))
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native-dir', type=Path, required=True)
    p.add_argument('--generated-dir', type=Path, required=True)
    p.add_argument('--prior-review-dir', type=Path, required=True)
    p.add_argument('--prior-report', type=Path, required=True)
    p.add_argument('--abi-observer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    exe = (a.native_dir/'flat-linker').resolve()
    image = (a.native_dir/'flat-linker.image').resolve()
    assert sha(exe.read_bytes()) == EXPECTED_LINKER
    assert sha(image.read_bytes()) == EXPECTED_IMAGE
    old_source = (a.generated_dir/'borrowed-baseline/module-linker.hepta').read_bytes()
    source = (a.generated_dir/'flat-linker.hepta').read_bytes()
    assert sha(old_source) == EXPECTED_OLD_SOURCE and sha(source) == EXPECTED_SOURCE
    extract = lambda raw: dict(re.findall(rb'^fn ([^(]+)(\(.*?\nend\n)', raw, re.M|re.S))
    old_functions, new_functions = extract(old_source), extract(source)
    assert len(old_functions) == len(new_functions) == 27
    changed = [name.decode() for name in old_functions if old_functions[name] != new_functions[name]]
    assert changed == ['f0','f6'], changed
    old_author = (a.generated_dir/'borrowed-baseline/linker-authoring.txt').read_text()
    new_author = (a.generated_dir/'flat-linker-authoring.txt').read_text()
    old_ast = {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(old_author).body}
    new_ast = {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(new_author).body}
    assert [name for name in old_ast if old_ast[name] != new_ast[name]] == ['link','headers']
    new_magic, old_magic = int.from_bytes(b'HIAFLK01','little'), int.from_bytes(b'HIABLK01','little')
    assert new_ast['link'].replace(str(new_magic),str(old_magic),1) == old_ast['link']
    bridge = (a.generated_dir/'entry-bridge.bin').read_bytes()
    assert len(bridge) == 78 and bridge[64] == 0xE8
    assert sha(bridge) == '3abd320129a48d28c1933e70185e4d375c8e47ade38bce73efe34cee23719f6d'
    prior_raw = a.prior_report.read_bytes(); prior = json.loads(prior_raw)
    prior_rows = {row['case']:row for row in prior['rows'] if row['kind'].startswith('native-link')}
    assert prior['status'] == 'PASS' and prior['compiler_sha256'] == 'e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59'
    rows, valid_images = [], {}

    def run_link(name, expected_ok):
        old = (a.prior_review_dir/(name+'.request')).read_bytes()
        assert old[:8] == b'HIABLK01' and sha(old) == prior_rows[name]['request_sha256']
        request = b'HIAFLK01'+old[8:]
        assert request[8:] == old[8:]
        (out/(name+'.request.bin')).write_bytes(request)
        result = subprocess.run([str(exe)], input=request, capture_output=True, env={'PATH':''}, timeout=30)
        (out/(name+'.link.stdout')).write_bytes(result.stdout)
        (out/(name+'.link.stderr')).write_bytes(result.stderr)
        row = {'case':name, 'kind':'flat-link-success' if expected_ok else 'flat-link-refusal',
               'original_request_sha256':sha(old), 'request_sha256':sha(request),
               'only_request_change':'HIABLK01 -> HIAFLK01',
               'exit_code':result.returncode, 'output_bytes':len(result.stdout)}
        if expected_ok:
            old_executable = (a.prior_review_dir/(name+'.native')).read_bytes()
            assert result.returncode == 0 and not result.stderr and len(result.stdout) == len(old_executable)
            assert result.stdout[1024:] == old_executable[1024:]
            expected_entry = struct.unpack_from('<i',old_executable,370)[0]+374
            actual_entry = struct.unpack_from('<i',result.stdout,65)[0]+69
            expected_bridge = bytearray(bridge)
            struct.pack_into('<i',expected_bridge,65,expected_entry-69)
            assert actual_entry == expected_entry >= 1024
            assert result.stdout[:78] == expected_bridge and result.stdout[78:1024] == bytes(1024-78)
            row.update(entry_offset=actual_entry, body_after_1024_byte_identical=True,
                       exact_78_byte_entry_bridge=True, image_sha256=sha(result.stdout))
            valid_images[name] = result.stdout
        else:
            assert result.returncode == 2 and not result.stdout
        row['pass'] = True
        rows.append(row)

    positives = [('cross-resource-mutation','7f','8081'),
                 ('reordered-objects','7f','8081'), ('nested-borrow','','5b')]
    for name, input_hex, expected_hex in positives:
        run_link(name, True)
        folder=out/name; folder.mkdir()
        (folder/'application.flat').write_bytes(valid_images[name])
        (folder/'execute.json').write_text(json.dumps({'image_sha256':sha(valid_images[name]),
                                                      'input_hex':input_hex,'expected_hex':expected_hex})+'\n')
        (folder/'child.py').write_text(CHILD)
        result=subprocess.run([sys.executable,'-I','-S','-B',str(folder/'child.py'),str(folder)],
                              capture_output=True,timeout=10)
        (folder/'execution.stdout').write_bytes(result.stdout)
        (folder/'execution.stderr').write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (name,result.returncode,result.stderr)
        observed=json.loads(result.stdout)
        rows.append({'case':name+'-two-load-addresses','kind':'raw-PIC-execution',
                     'input_hex':input_hex,'expected_hex':expected_hex,
                     'expectation_source':'Previously frozen borrowed-module source/reference case',
                     'pass':True,**observed})
    for name in ('corrupt-abi-version','link-buffer-mode-mismatch','cross-module-cycle-refused',
                 'overlapping-relocations-refused','unresolved-import','corrupt-relocation-outside-text',
                 'wrong-entry-signature','corrupt-owned-interface'):
        run_link(name, False)

    # Reuse the already inspected disposable ABI observer only as an execution
    # harness. It imports no compiler/linker/HIA runtime and writes no files.
    observer = a.abi_observer.resolve(); observer_sha = sha(observer.read_bytes())
    assert observer_sha == EXPECTED_OBSERVER
    capacity_cases = [
        ('valid-dirty-workspace','reordered-objects',4096,16<<20,4096,False),
        ('short-output-no-publication','reordered-objects',4095,16<<20,0xFFFFFFFE,True),
        ('short-workspace-no-mutation','reordered-objects',4096,(16<<20)-1,0xFFFFFFFE,True),
        ('bad-interface-no-publication','corrupt-abi-version',4096,16<<20,0xFFFFFFFF,True),
    ]
    for name,request_name,capacity,workspace,expected_result,failure in capacity_cases:
        request_path=out/(request_name+'.request.bin')
        config={'image':str(image),'image_sha256':EXPECTED_IMAGE,
                'request':str(request_path),'request_sha256':sha(request_path.read_bytes()),
                'capacity':capacity,'workspace':workspace,'workspace_fill':90}
        config_path=out/(name+'.abi.json');config_path.write_text(json.dumps(config,indent=2)+'\n')
        result=subprocess.run([sys.executable,'-I','-S','-B',str(observer),str(config_path)],capture_output=True,timeout=25)
        (out/(name+'.observe.stdout')).write_bytes(result.stdout)
        (out/(name+'.observe.stderr')).write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr,(name,result)
        observed=json.loads(result.stdout)
        assert observed['result']==expected_result and not observed['forbidden_modules'],(name,observed)
        assert observed['source_unchanged'] and observed['output_guards_unchanged'] and observed['workspace_guards_unchanged']
        if failure: assert observed['output_unchanged']
        else: assert observed['produced_sha256']==sha(valid_images[request_name])
        if name=='short-workspace-no-mutation': assert observed['workspace_unchanged']
        rows.append({'case':name,'kind':'flat-ABI-capacity-atomicity','capacity':capacity,
                     'workspace':workspace,'pass':True,**observed})
    report={'schema':'heptatorch.independent-flat-linker-review.v1','status':'PASS',
            'linker_source_sha256':EXPECTED_SOURCE,'linker_executable_sha256':EXPECTED_LINKER,
            'linker_flat_image_sha256':EXPECTED_IMAGE,'frozen_baseline_source_sha256':EXPECTED_OLD_SOURCE,
            'changed_hia_functions':changed,'unchanged_hia_functions':25,
            'link_change_only_wire_magic':True,'selected_entry_bridge_bytes':78,
            'object_text_origin':1024,'prior_review_receipt_sha256':sha(prior_raw),
            'observer_script_sha256':sha(Path(__file__).read_bytes()),
            'execution_child_sha256':sha(CHILD.encode()),'abi_observer_sha256':observer_sha,
            'checks':len(rows),'successful_links':3,'raw_image_executions':6,
            'malformed_request_refusals':8,'flat_abi_capacity_atomicity_checks':4,
            'host_role':'Source/interface comparison, request framing and stdlib native ABI observation only',
            'scope':'Borrowed ABI1 raw-PIC output and selected old malformed requests. Separate from owned ABI2 and from full tensor session numerical qualification.',
            'rows':rows}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
