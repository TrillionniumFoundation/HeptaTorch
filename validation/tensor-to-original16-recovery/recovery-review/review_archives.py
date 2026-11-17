#!/usr/bin/env python3
"""Read-only byte recovery of two original frozen archives; no compiler/Torch."""
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import traceback

BASE = Path('/workspace/scratch/d26699e25fea')
OUT = Path('/dev/shm/heptatorch-d26699e25fea/hia-frozen-source-recovery-01')
RECEIPT = BASE / 'hia-recovery-review-20261116'
sha = lambda b: hashlib.sha256(b).hexdigest()
STEMS = ('copy4', 'reduction4', 'lanes2', 'branches6')

def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def selected(name, kind):
    if kind == 'producer':
        return (name in {
            'tools/tensor_to_copy/package-manifest.json',
            'tools/tensor_to_copy/profile.json',
            'tools/tensor_to_copy/qualify.py',
            'tools/tensor_to_copy/archive_evidence.py',
            'tests/tensor_to_copy_review/native_observer09.py',
            'tests/tensor_to_copy_review/owner_client.py',
            'tests/tensor_to_copy_review/protocol.py'}
            or name.startswith('tests/pytorch_as_strided_')
            or name.startswith('tests/tensor_to_copy_review/replay_as_strided_')
            or name.startswith('tests/tensor_to_copy_review/source-inverses/as-strided-')
            or name.startswith('validation/tensor-to-copy/reference/as-strided-')
            or name.startswith('validation/tensor-to-copy/reference-packets/as-strided-')
            or (name.startswith('validation/tensor-to-copy/original-reports/as-strided-') and '09b' in name))
    return (name in {'original/report.json', 'original/invocation.json',
                     'original/inputs-before-execution.json', 'original/producer/report.json'}
            or name.startswith('root/')
            or any(name.startswith('original/producer/fixed-as-strided-'+stem+'/')
                   or name.startswith('original/producer/fixed-as-strided-'+stem+'.')
                   or name.startswith('original/producer/evidence/fixed-as-strided-'+stem+'.')
                   for stem in STEMS))

def verify(kind, dirname, expected_manifest, expected_archive):
    root = BASE/'tensor-to-copy-checkpoints'/dirname
    mb = (root/'manifest.json').read_bytes()
    assert sha(mb) == expected_manifest
    manifest = json.loads(mb)
    arc = root/manifest['archive']
    before = arc.stat()
    compressed = arc.read_bytes()
    assert len(compressed) == manifest['archive_bytes']
    assert sha(compressed) == manifest['archive_sha256'] == expected_archive
    decoder = lzma.LZMADecompressor()
    raw = decoder.decompress(compressed)
    assert decoder.eof and decoder.unused_data == b''
    got, copied, total = {}, {}, 0
    destroot = OUT/kind
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as tf:
        for member in tf:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert str(path) == member.name and member.name not in got
            assert member.name in manifest['members']
            item = manifest['members'][member.name]
            data = tf.extractfile(member).read()
            actual = {'bytes':len(data), 'sha256':sha(data), 'mode':member.mode}
            assert len(data) == member.size
            assert all(actual[k] == item[k] for k in actual), (member.name, actual, item)
            got[member.name] = actual
            total += len(data)
            if selected(member.name, kind):
                p = destroot.joinpath(*path.parts)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open('xb') as out: out.write(data)
                os.chmod(p, member.mode)
                assert p.read_bytes() == data and stat.S_IMODE(p.stat().st_mode) == member.mode
                copied[member.name] = {'path':str(p), **actual}
    assert set(got) == set(manifest['members'])
    assert len(got) == manifest['member_count'] and total == manifest['original_bytes']
    after = arc.stat()
    assert all(getattr(before, k) == getattr(after, k) for k in ('st_dev','st_ino','st_size','st_mode','st_mtime_ns','st_ctime_ns'))
    assert (root/'manifest.json').read_bytes() == mb
    (destroot/'original-checkpoint-manifest.json').write_bytes(mb)
    write_json(OUT/(kind+'-verified-members.json'), got)
    return {'archive_path':str(arc), 'archive_bytes':len(compressed),
            'archive_sha256':sha(compressed), 'manifest_sha256':sha(mb),
            'tar_bytes':len(raw), 'tar_sha256':sha(raw), 'single_xz_eof':True,
            'members':len(got), 'expanded_bytes':total, 'all_members_exact':True,
            'source_unchanged':True, 'extracted_subset':copied,
            'original_status':manifest['status']}

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    report = {'schema':'heptatorch.to09b-original-byte-recovery.v1', 'status':'RUNNING',
              'native_executions':0, 'torch_executions':0, 'compiler_executions':0,
              'original_result_reclassification':False}
    try:
        report['producer'] = verify('producer','portable09b-prepared',
            'aff1d7a34dbc60a17e994347ea6af933ba7ec1f8580d5468063ea31d6250569a',
            '2bdc1c3d140db2ae51fa0eb605504d2f672a1098912780578f30595289918ad4')
        pm=OUT/'producer/tools/tensor_to_copy/package-manifest.json'
        assert sha(pm.read_bytes()) == '5179e869868633ff38c0a357039b8f223f5d9672a6453af6187d4d2bd025e3a1'
        report['cold01'] = verify('cold01','root-cold-01-failure',
            '005764b611d41f97281cebbcba151e6168b235c5b7b97713be6ae7d9ac670cf1',
            'b82492244b04c7c67a932b720201a3015e9533a28458f188406de76c39eaaaed')
        failure=OUT/'cold01/original/report.json'
        assert sha(failure.read_bytes()) == '245f664be05c7c22b451991b6bbe5c4ec93d05b8ab94aff14a967e3c11c02042'
        assert json.loads(failure.read_bytes())['status'] == 'FAIL'
        report['status']='PASS_BYTES_RECOVERED_ORIGINAL_RESULTS_UNCHANGED'
    except BaseException:
        report['status']='FAIL_BYTE_RECOVERY';report['error']=traceback.format_exc()
    write_json(RECEIPT/'archive-review.json', report)
    print(json.dumps({'status':report['status'],'receipt':str(RECEIPT/'archive-review.json'),
      'sha256':sha((RECEIPT/'archive-review.json').read_bytes()),
      'producer_members':report.get('producer',{}).get('members'),
      'cold01_members':report.get('cold01',{}).get('members'),'error':report.get('error')}))
    return 0 if report['status'].startswith('PASS_') else 1

if __name__ == '__main__':
    raise SystemExit(main())
