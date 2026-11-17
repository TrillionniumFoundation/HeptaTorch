"""Verify the original reference packet and optionally restore/replay pure admission."""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile


def sha(b):
    return hashlib.sha256(b).hexdigest()


def archive_members(archive, manifest):
    raw=archive.read_bytes()
    assert len(raw)==manifest['archive_bytes'] and sha(raw)==manifest['archive_sha256']
    dec=lzma.LZMADecompressor(format=lzma.FORMAT_XZ);tar=dec.decompress(raw)
    assert dec.eof and not dec.unused_data
    assert len(tar)==manifest['tar_bytes'] and sha(tar)==manifest['tar_sha256']
    got={}
    with tarfile.open(fileobj=io.BytesIO(tar),mode='r:') as tf:
        for m in tf:
            assert m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
            assert m.name not in got and m.name in manifest['files']
            b=tf.extractfile(m).read();e=manifest['files'][m.name]
            assert len(b)==m.size==e['bytes'] and sha(b)==e['sha256'] and m.mode==e['mode']
            got[m.name]=(b,m.mode)
    assert set(got)==set(manifest['files']) and len(got)==manifest['file_count']
    assert sum(len(b) for b,mode in got.values())==manifest['expanded_bytes']
    return got


def restore(members,root):
    root.mkdir(parents=True,exist_ok=False)
    for name,(b,mode) in members.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);p.chmod(mode)
        assert p.read_bytes()==b and stat.S_IMODE(p.stat().st_mode)==mode


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--packet',type=Path,required=True)
    ap.add_argument('--restore',type=Path)
    ap.add_argument('--admission-output',type=Path)
    a=ap.parse_args();packet=a.packet.resolve()
    m=json.loads((packet/'manifest.json').read_bytes());outer=json.loads((packet/'outer.json').read_bytes())
    for name,record in outer['files'].items():
        p=packet/name;b=p.read_bytes();assert len(b)==record['bytes'] and sha(b)==record['sha256']
        # Outer files may be installed into Git as 0644. Their original modes are
        # explicitly recorded; archive member modes remain exact and restored.
        assert stat.S_IMODE(p.stat().st_mode) in (record['mode'],0o644)
    got=archive_members(packet/'complete.tar.xz',m)
    if a.restore:
        root=a.restore.resolve();restore(got,root)
        om=json.loads((root/'original-packet/manifest.json').read_bytes())
        originals=archive_members(root/'original-packet/complete.tar.xz',om)
        restore(originals,root/'original-packet/payload')
        am=json.loads((root/'ad-source-packet/manifest.json').read_bytes())
        sources=archive_members(root/'ad-source-packet/complete.tar.xz',am)
        restore(sources,root/'ad-source-packet/payload')
        if a.admission_output:
            cmd=[sys.executable,'-I','-S','-B',str(root/'admit_actual26_r03.py'),'--root',str(root),'--output',str(a.admission_output.resolve())]
            result=subprocess.run(cmd,stdin=subprocess.DEVNULL,capture_output=True)
            assert result.returncode==0,(result.returncode,result.stdout,result.stderr)
            fresh=(a.admission_output.resolve()/'report.json').read_bytes()
            original=(root/'admission-r03-first/report.json').read_bytes()
            assert fresh==original,'Pure replay must reproduce the entire original admission bytes'
    else:
        assert a.admission_output is None
        # Check both nested originals without retaining an expanded directory.
        for prefix in ('original-packet','ad-source-packet'):
            nm=json.loads(got[prefix+'/manifest.json'][0]);raw=got[prefix+'/complete.tar.xz'][0]
            class BytesPath:
                def read_bytes(self): return raw
            archive_members(BytesPath(),nm)
    print(json.dumps({'status':'PASS','outer_members':len(got),'all_bytes_modes_single_XZ_verified':True,
                      'restored':bool(a.restore),'pure_admission_replayed':bool(a.admission_output),
                      'Torch_executed':False,'native_executed':False}))


if __name__=='__main__':main()
