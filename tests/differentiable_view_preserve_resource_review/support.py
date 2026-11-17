"""Strict inputs and byte transport for the independent to-copy controls."""
from pathlib import Path
import hashlib
import importlib.util
import json
import lzma
import struct

SOURCE='4f8f73f463f681b5eb29423d71a35a0b42d807fd92bf976bbf76c1970fbcee33'
IMAGE='53fb43b869858f020c0b936a65f00f3b14b97d9e37038a74c357e1da471fe083'
REFERENCE='00bdbbe17709b0cb737161faf944eae873c09b274311dd4c74a28c251e20b6be'
PROBE='07f09ea7e0ec6b8aa9195f9d059ea905753e86170f48ce6c6b23fb6d58923251'
PROTOCOL='7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
COMPILER='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER='0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
ENTRY='54f81a65652e2c2ba9eb3f04d608e89b71c92c5ad33ff7061e9a03b89f45f9ea'
OBS='6d230e2df25e04e32f76fb10fd0936ec9d220b60acda37c40b52a45633372312'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2)+'\n')


class Custody:
    def __init__(self): self.inputs={}
    def bind(self,path,pin=None):
        path=Path(path).resolve();assert path.is_file()
        raw=path.read_bytes();h=sha(raw)
        assert pin is None or h==pin,(str(path),h,pin)
        self.inputs[str(path)]={'bytes':len(raw),'sha256':h,'mode':path.stat().st_mode&511}
        return raw
    def finish(self):
        for p,r in self.inputs.items():
            path=Path(p);assert len(path.read_bytes())==r['bytes'] and sha(path.read_bytes())==r['sha256']
            assert path.stat().st_mode&511==r['mode']
        return self.inputs


def load(custody,build,protocol,reference):
    build=Path(build).resolve();protocol=Path(protocol).resolve();here=Path(__file__).parent
    for name in ('support.py','owner_state.py','owner_view05.py','observer-adaptation.json','parent_owner_mixed10.py'):
        custody.bind(here/name,OBS if name=='owner_view05.py' else None)
    custody.bind(build/'runtime.hepta',SOURCE)
    image=build/'tensor-to-copy-session';custody.bind(image,IMAGE)
    info=json.loads(custody.bind(build/'build.json'))
    assert info['status']=='PASS' and info['source_sha256']==SOURCE and info['native_sha256']==IMAGE
    assert info['state_format']=='HTTOCP08' and info['backing_format']=='HTOTOCP8' and info['descriptor_bytes']==672
    assert info['functions']==614 and len(info['units'])==12 and info['link_exit']==0
    assert info['compiler_sha256']==COMPILER and info['linker_sha256']=='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
    assert len({u['name'] for u in info['units']})==12
    for unit in info['units']:
        assert unit['compiler_exit']==0
        for suffix,key in (('.hepta','source_sha256'),('.o','object_sha256')):
            custody.bind(build/(unit['name']+suffix),unit[key])
    custody.bind(protocol,PROTOCOL);custody.bind(protocol.with_name('owner_client.py'),CLIENT)
    spec=importlib.util.spec_from_file_location('to_copy_control_wire',protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    observed=json.loads(custody.bind(reference,REFERENCE))
    assert observed['status']=='OBSERVED' and observed['cases']==54 and len(observed['rows'])==54
    assert observed['probe_error_count']==0 and observed['probe_sha256']==PROBE
    assert observed['conversion_variants_observed']==186 and observed['source_files_unchanged'] is True
    checks=observed['oracle_provenance']['checks']
    assert observed['oracle_provenance']['status']=='PASS'
    assert set(checks)=={'python_matches','platform_matches','version_backend_matches','git_matches_oracle_build_pin',
        'distribution_version_matches','direct_url_matches','direct_url_archive_sha256_matches','pip_report_url_matches',
        'pip_report_archive_sha256_matches','pip_report_version_matches','pip_report_requested'}
    assert all(v is True for v in checks.values())
    for h,state in observed['states'].items():
        raw=bytes.fromhex(state['hex']);assert len(raw)==state['bytes']==5056 and sha(raw)==h
    return wire,image,info,observed


def create_packet(dtype,shape,raw,requires=0,opcode=97):
    assert len(shape)<=4 and opcode in (97,110)
    config=struct.pack('<8Q',len(shape),requires,*shape,*([0]*(4-len(shape))),160,len(raw))
    return b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)+struct.pack('<8Q',opcode,dtype,96,0,0,0,0,0)+config+raw


def raw_file(root,raw):
    digest=sha(raw);path=Path(root)/(digest+'.owner.xz')
    if not path.exists():
        path.write_bytes(lzma.compress(raw,preset=1))
        assert lzma.decompress(path.read_bytes())==raw
    return {'path':path.name,'bytes':len(raw),'sha256':digest,'archive_sha256':sha(path.read_bytes())}


def typed_bytes(dtype,values):
    if dtype==0:return struct.pack('<'+'f'*len(values),*values)
    width=(4,1,8)[dtype]
    return b''.join((int(v)&((1<<(width*8))-1)).to_bytes(width,'little') for v in values)
