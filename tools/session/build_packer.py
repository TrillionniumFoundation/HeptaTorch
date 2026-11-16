#!/usr/bin/env python3
"""Author an 8-MiB-image HIA session packer and its real wide I/O wrapper.

This stage verifies upstream source and emits HIA. Native compilation and
native ELF packing are a distinct, reproducible stage in rebuild_packer.py.
"""
from pathlib import Path
from types import FunctionType
import argparse
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode=True
SPEC=Path(__file__).resolve().parent
MIB=1<<20
IMAGE_LIMIT=8*MIB
WIRE_LIMIT=IMAGE_LIMIT+16
OUTPUT_CAPACITY=IMAGE_LIMIT+4096
WORKSPACE_CAPACITY=4096
CONTRACT_PIN='8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'
CAPACITY_PIN='1926437c96c413b891b4c0b61d1555434598a40067855241e0d2a53b1bee1292'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def private(original,**changes):
    namespace=dict(original.__globals__);namespace.update(changes)
    result=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    result.__kwdefaults__=original.__kwdefaults__
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--capacity-adaptation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    upstream=args.source_root.resolve()
    capacity=args.capacity_adaptation.read_bytes();assert sha(capacity)==CAPACITY_PIN
    contract=(upstream/'docs/HIA_SELFHOST_CORE_SUPPORT.json').read_bytes();assert sha(contract)==CONTRACT_PIN
    pins=json.loads(contract)['source_identity']['files']
    closure=set(json.loads(capacity)['upstream_source_files_loaded'])|{
        'tools/generate_hia_native_filter.py','tools/generate_hia_elf_linker.py'}
    for path in closure:assert sha((upstream/path).read_bytes())==pins[path]
    sys.path.insert(0,str(upstream))
    from tools import generate_hia_native_filter as generator
    from hepta.hia_agent_assembly import render_assembly,assemble_program
    assert render_assembly(generator.document()).encode()==(
        upstream/'examples/hia_bootstrap/native_filter_packer.hepta').read_bytes()
    spec=importlib.util.spec_from_file_location('session_wrapper',SPEC/'wrapper.py')
    wrapper_source=(SPEC/'wrapper.py').read_bytes()
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    labels={};prefix,hole=wrapper.session_template(receipt=labels)
    assert len(prefix)==774 and hole==465
    output_limit=IMAGE_LIMIT+len(prefix)
    author=private(generator.source_program,template=wrapper.session_template)
    original=author(output_bytes=MIB,workspace_bytes=MIB)
    original_doc=private(generator.document,source_program=author)(
        output_bytes=MIB,workspace_bytes=MIB,module='selfhost_exec')
    assert sha(render_assembly(original_doc).encode())=='99058dae95b008303981710f0d8ee348da0976f173d3662c5e5f5de150fdfe6b'
    old_header='    if length(s) < 17 or length(s) > 1048576:\n'
    old_size='    if size > 1048576 or length(o) < size:\n'
    assert original.count(old_header)==original.count(old_size)==1
    modified=original.replace(old_header,f'    if length(s) < 17 or length(s) > {WIRE_LIMIT}:\n',1)
    modified=modified.replace(old_size,f'    if size > {output_limit} or length(o) < size:\n',1)

    def source_program(**unused):return modified
    document=private(generator.document,source_program=source_program)
    doc=document(module='wide_session_packer')
    source=render_assembly(doc).encode();assert assemble_program(source).as_dict()==doc
    # Bootstrap the wide packer from a small real HIA packer. Its emitted
    # native wrapper genuinely admits wire_limit bytes and allocates enough
    # output + workspace; changing only the HIA size test would be insufficient.
    carrier_template=private(generator.template,WIRE_BYTES=WIRE_LIMIT,
                             OUTPUT_OFFSET=IMAGE_LIMIT+4096)
    carrier_author=private(generator.source_program,template=carrier_template)
    carrier_document=private(generator.document,source_program=carrier_author)
    carrier_doc=carrier_document(output_bytes=OUTPUT_CAPACITY,
                                 workspace_bytes=WORKSPACE_CAPACITY,module='selfhost_exec')
    carrier=render_assembly(carrier_doc).encode()
    assert assemble_program(carrier).as_dict()==carrier_doc
    profile=json.loads((SPEC/'profile.json').read_text())
    assert sha(source)==profile['wide_packer_source_sha256']
    assert sha(carrier)==profile['tool_packer_source_sha256']
    (out/'session-packer.hepta').write_bytes(source)
    (out/'session-packer-authoring.txt').write_text(modified)
    (out/'tool-packer.hepta').write_bytes(carrier)
    (out/'tool-packer-authoring.txt').write_text(carrier_author(
        output_bytes=OUTPUT_CAPACITY,workspace_bytes=WORKSPACE_CAPACITY))
    (out/'session-prefix.bin').write_bytes(prefix)
    (out/'wrapper-labels.json').write_text(json.dumps(labels,indent=2)+'\n')
    def counts(doc):return {'functions':len(doc['functions']),
                            'blocks':sum(len(f['blocks']) for f in doc['functions']),
                            'instructions':sum(len(b['ops']) for f in doc['functions'] for b in f['blocks'])}
    for value in (counts(doc),counts(carrier_doc)):
        assert value['functions']<=256 and value['blocks']<=2560 and value['instructions']<=8192
    loaded={}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename is None:continue
        path=Path(filename).resolve()
        try:relative=path.relative_to(upstream).as_posix()
        except ValueError:continue
        assert relative in closure and sha(path.read_bytes())==pins[relative],relative
        loaded[relative]=pins[relative]
    assert set(loaded)==closure
    for path in closure:assert sha((upstream/path).read_bytes())==pins[path]
    assert (SPEC/'wrapper.py').read_bytes()==wrapper_source
    report={'schema':'heptatorch.wide-session-packer-authoring.v1',
            'native_compilation_performed':False,'native_packing_performed':False,
            'source_sha256':sha(source),'source_structure':counts(doc),
            'tool_packer_source_sha256':sha(carrier),'tool_packer_structure':counts(carrier_doc),
            'adapter_sha256':sha(Path(__file__).read_bytes()),
            'wrapper_authoring_sha256':sha(wrapper_source),
            'session_prefix_sha256':sha(prefix),'session_prefix_bytes':len(prefix),
            'session_call_hole':hole,'wire_magic':'HIAFLT01',
            'image_limit_bytes':IMAGE_LIMIT,'input_limit_bytes':WIRE_LIMIT,
            'packed_image_limit_bytes':output_limit,
            'native_wrapper_input_limit_bytes':WIRE_LIMIT,
            'native_wrapper_output_capacity_bytes':OUTPUT_CAPACITY,
            'native_wrapper_workspace_bytes':WORKSPACE_CAPACITY,
            'native_wrapper_output_offset':IMAGE_LIMIT+4096,
            'native_wrapper_mapping_bytes':(IMAGE_LIMIT+4096)+OUTPUT_CAPACITY+WORKSPACE_CAPACITY,
            'packed_tensor_workspace_bytes_each':MIB,'packed_tensor_workspace_copies':2,
            'upstream_source_files_verified_before_and_after':loaded}
    (out/'authoring.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='upstream_source_files_verified_before_and_after'},indent=2))


if __name__=='__main__':main()
