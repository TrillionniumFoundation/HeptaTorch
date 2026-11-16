"""Verify frozen mixed-autograd source and complete historical/native evidence bytes.

This verifier executes no Torch or native image. Optional extraction restores
one named evidence archive into a newly created destination.
"""
import argparse
import hashlib
import json
import lzma
from pathlib import Path,PurePosixPath
import stat
import tarfile

SHA=lambda raw:hashlib.sha256(raw).hexdigest()


def members_of(value):
    for key in ('files','rows','members'):
        result=value.get(key)
        if isinstance(result,dict):return result
        if isinstance(result,list):
            pairs=[(row['path'],row) for row in result]
            assert len({n for n,_ in pairs})==len(pairs)
            return dict(pairs)
    raise AssertionError('No complete member inventory')


def single_xz(path):
    decoder=lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    with path.open('rb') as source:
        while data:=source.read(65536):
            assert not decoder.eof
            decoder.decompress(data,max_length=1<<20)
            while not decoder.needs_input and not decoder.eof:
                decoder.decompress(b'',max_length=1<<20)
            assert not decoder.unused_data
    assert decoder.eof


def verify_archive(path,manifest):
    wanted=members_of(manifest);seen=set();total=0
    single_xz(path)
    with tarfile.open(path,'r:xz') as tar:
        for member in tar:
            pure=PurePosixPath(member.name)
            assert member.isfile() and not pure.is_absolute() and '..' not in pure.parts
            assert member.name in wanted and member.name not in seen
            raw=tar.extractfile(member).read();pin=wanted[member.name]
            assert len(raw)==pin['bytes']==member.size
            assert SHA(raw)==pin['sha256'] and member.mode==pin['mode']
            seen.add(member.name);total+=len(raw)
    assert seen==set(wanted)
    return {'members':len(seen),'file_bytes':total,'single_xz_EOF':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--extract-evidence')
    p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.package_root.resolve();v=root/'validation/mixed-integer-autograd-evidence'
    manifest_raw=(v/'handoff-manifest.json').read_bytes()
    assert SHA(manifest_raw)==a.expected_manifest_sha256
    manifest=json.loads(manifest_raw)
    assert len(manifest['files'])==manifest['file_count']
    for name,pin in manifest['files'].items():
        rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
        path=root/name;raw=path.read_bytes()
        assert not path.is_symlink()
        assert len(raw)==pin['bytes'] and SHA(raw)==pin['sha256']
        assert stat.S_IMODE(path.stat().st_mode)==pin['mode']
    source_manifest_raw=(root/'tools/mixed_integer_autograd/package-manifest.json').read_bytes()
    assert SHA(source_manifest_raw)=='8e8f775ea42fbec4898af318bc0f51be6dcb1510a1eff13b96e2e8f74cf6daca'
    source_manifest=json.loads(source_manifest_raw)
    assert source_manifest['status']=='PREPARED_NOT_RUN' and len(source_manifest['files'])==85
    for name,pin in source_manifest['files'].items():assert manifest['files'][name]==pin
    q=json.loads((v/'qualification.json').read_bytes())
    raw=(v/q['cold_report_path']).read_bytes();assert SHA(raw)==q['cold_report_sha256']
    cold=json.loads(raw)
    assert cold['status']=='PASS' and cold['step_count']==5 and cold['native_cold_rebuild_executed'] is True
    assert cold['source_sha256']==q['source_sha256'] and cold['native_sha256']==q['native_sha256']
    assert cold['all_package_members_unchanged'] is True and cold['all_external_package_members_unchanged'] is True
    assert cold['all_named_gate_statuses_and_counts_matched'] is True
    expected=['cold-native-build','fixed-mixed-actual36','inherited-I64-eight','legacy523-twelve','fixed-saved-public14']
    assert [row['name'] for row in cold['steps']]==expected
    for row in cold['steps']:
        assert row['status']=='PASS' and row['exit_code']==0
        assert row['process_started'] is True and row['process_completed'] is True
    index=json.loads((v/'archive-index.json').read_bytes());counts={}
    for item in index['archives']:
        archive=v/item['archive'];mr=(v/item['manifest']).read_bytes()
        assert SHA(mr)==item['manifest_sha256']
        detail=json.loads(mr);ar=archive.read_bytes()
        assert len(ar)==item['bytes']==detail['archive_bytes']
        assert SHA(ar)==item['sha256']==detail['archive_sha256']
        counts[item['name']]=verify_archive(archive,detail)
    if a.extract_evidence:
        assert a.output is not None and not a.output.exists()
        selected=[r for r in index['archives'] if r['name']==a.extract_evidence];assert len(selected)==1
        a.output.mkdir(parents=True,exist_ok=False)
        detail=json.loads((v/selected[0]['manifest']).read_bytes());wanted=members_of(detail)
        with tarfile.open(v/selected[0]['archive'],'r:xz') as tar:
            for member in tar:
                dest=a.output/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as target:target.write(tar.extractfile(member).read())
                dest.chmod(wanted[member.name]['mode'])
        for name,pin in wanted.items():
            dest=a.output/name
            assert SHA(dest.read_bytes())==pin['sha256'] and stat.S_IMODE(dest.stat().st_mode)==pin['mode']
    else:
        assert a.output is None
    print(json.dumps({'status':'PASS_BYTE_CUSTODY_ONLY','payload_files':manifest['file_count'],
          'archives':counts,'cold_native_gates_recorded':5,'new_native_execution':False,
          'new_Torch_execution':False,'full_pytorch_rewrite_complete':False,
          'whole_system_decision':'HOLD'}))


if __name__=='__main__':main()
