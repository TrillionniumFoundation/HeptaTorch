"""Verify the complete Saved lifetime packet without running Torch or native code."""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tarfile

sha=lambda raw:hashlib.sha256(raw).hexdigest()


def single_xz(raw):
    decoder=lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    expanded=decoder.decompress(raw)
    assert decoder.eof and not decoder.unused_data
    return expanded


def frame_count(raw,magic):
    count=offset=0
    while offset<len(raw):
        assert offset+16<=len(raw) and raw[offset:offset+8]==magic
        size=struct.unpack_from('<Q',raw,offset+8)[0]
        assert 32<=size<=1048576 and offset+16+size<=len(raw)
        offset+=16+size;count+=1
    assert offset==len(raw)
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--expected-manifest-sha256',required=True)
    a=parser.parse_args();root=a.package_root.resolve();val=root/'validation/saved-public-lifetime'
    raw=(val/'package-manifest.json').read_bytes();assert sha(raw)==a.expected_manifest_sha256
    package=json.loads(raw);assert package['file_count']==len(package['files'])
    for name,e in package['files'].items():
        p=root/name;b=p.read_bytes()
        assert not p.is_symlink() and len(b)==e['bytes'] and sha(b)==e['sha256']
        assert stat.S_IMODE(p.stat().st_mode)==e['mode']
    q=json.loads((val/'qualification.json').read_bytes())
    archive=(val/'raw-native-history.tar.xz').read_bytes()
    raw_manifest=(val/'raw-native-history-manifest.json').read_bytes()
    assert len(archive)==q['archive_bytes'] and sha(archive)==q['archive_sha256']
    assert sha(raw_manifest)==q['raw_manifest_sha256']
    manifest=json.loads(raw_manifest);wanted=manifest['files'];expanded=single_xz(archive)
    seen=set();frames={};owner_snapshots=0;payload_bytes=0;selected={}
    with tarfile.open(fileobj=io.BytesIO(expanded),mode='r:') as tar:
        for m in tar:
            name=m.name;p=PurePosixPath(name)
            assert m.isfile() and name not in seen and name in wanted
            assert not p.is_absolute() and '..' not in p.parts
            b=tar.extractfile(m).read();e=wanted[name]
            assert len(b)==m.size==e['bytes'] and sha(b)==e['sha256'] and m.mode==e['mode']
            seen.add(name);payload_bytes+=len(b)
            if name.endswith('.owner.xz'):
                owner=single_xz(b);assert len(owner)>640 and sha(owner)==p.name.removesuffix('.owner.xz')
                owner_snapshots+=1
            if name.endswith('/requests.frames.bin'):
                frames[name]=frame_count(b,b'HTOWNR01')
            if name.endswith('/responses.frames.bin'):
                frames[name]=frame_count(b,b'HTOWNS01')
            if name in ('native09/build.json','native10/build.json'):
                selected[name]=json.loads(b)
    assert seen==set(wanted) and len(seen)==manifest['members']==q['raw_members']
    assert payload_bytes==manifest['expanded_file_bytes']
    for prefix,source,image in (
        ('native09','d26e33363e1d1e43ba03b0289551b75826f348739e32e77b5e285df578088068','a1479a8773eb52937748bd90a1c4a79cc8d38c628be687152b9a8a97b721fb8c'),
        ('native10','5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49','69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6')):
        build=selected[prefix+'/build.json']
        assert build['status']=='PASS' and len(build['units'])==11
        assert build['source_sha256']==wanted[prefix+'/runtime.hepta']['sha256']==source
        assert build['native_sha256']==wanted[prefix+'/tensor-mixed-autograd-session']['sha256']==image
        for unit in build['units']:
            assert wanted[prefix+'/'+unit['name']+'.hepta']['sha256']==unit['source_sha256']
            assert wanted[prefix+'/'+unit['name']+'.o']['sha256']==unit['object_sha256']
    for filename,prefix,expected in (
        ('native09-failure-original.json','original-numerical-failure09',(14,13,1,2034,8768,2148,25863888,470208)),
        ('native10-pass-original.json','original-pass10',(14,14,0,2058,8842,2166,27261936,480320))):
        b=(val/filename).read_bytes();report=json.loads(b)
        assert tuple(report[k] for k in ('cases','passed','failed','frames','checks','exact_words','rollback_bytes','rng_bytes'))==expected
        assert report['reference_indices']==list(range(14)) and report['all_inputs_unchanged'] is True
        assert len(report['provenance_checks'])==11 and all(x is True for x in report['provenance_checks'].values())
        assert report['reference_sha256']=='c80ee752f7d7fffe3e808eb4720de0a0447e146e1b20eadc4b219dbbff96c3b5'
        for row in report['rows']:
            for name in ('requests','responses'):
                path=prefix+'/case-'+str(row['index']).zfill(2)+'/raw/'+name+'.frames.bin'
                assert frames[path]==row['frames']
        assert sum(x['frames'] for x in report['rows'])==report['frames']
    reference=(val/'fixed-reference-original.json').read_bytes();r=json.loads(reference)
    assert sha(reference)=='c80ee752f7d7fffe3e808eb4720de0a0447e146e1b20eadc4b219dbbff96c3b5'
    assert r['status']=='OBSERVED' and r['cases']==14 and r['probe_error_count']==0
    print(json.dumps(dict(status='PASS_EXACT_PACKAGE_AND_RAW_CUSTODY',payload_files=len(package['files']),
        raw_members=len(seen),raw_file_bytes=payload_bytes,complete_owner_snapshots=owner_snapshots,
        both_original_native_images_and_22_units_verified=True,native09_original_failures=1,
        native10_original_passed=14,new_native_execution=False,new_Torch_execution=False,whole_project_decision='HOLD')))


if __name__=='__main__':
    main()
