#!/usr/bin/env python3
"""Verify unchanged packet bytes and all original archived run members, no execution."""
import argparse,hashlib,json,lzma,stat,tarfile
from pathlib import Path,PurePosixPath

def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def member_name(name):
    p=PurePosixPath(name)
    assert name and str(p)==name and not p.is_absolute() and all(x not in ('','..','.') for x in p.parts)
    return p

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    a=p.parse_args();root=a.repo.resolve();v=root/'validation/mixed-integer-owner-review'
    package=json.loads((v/'package-manifest.json').read_bytes())
    for row in package['files']:
        name=member_name(row['path']);f=root/str(name)
        assert f.is_file() and not f.is_symlink() and root in f.resolve().parents
        assert f.stat().st_size==row['bytes'] and stat.S_IMODE(f.stat().st_mode)==row['mode'] and sha_file(f)==row['sha256']
    result=[]
    for spec in json.loads((v/'qualification.json').read_bytes())['archives']:
        archive=root/str(member_name(spec['path']));m=json.loads((root/str(member_name(spec['manifest']))).read_bytes())
        assert archive.name==m['archive']['path'] and archive.stat().st_size==m['archive']['bytes']
        assert sha_file(archive)==m['archive']['sha256']
        seen=set();total=0
        with tarfile.open(archive,'r:xz') as tar:
            for info in tar:
                member_name(info.name)
                assert info.isfile() and info.name not in seen and info.name in m['members']
                seen.add(info.name);row=m['members'][info.name]
                h=hashlib.sha256();length=0;stream=tar.extractfile(info)
                for block in iter(lambda:stream.read(1<<20),b''):h.update(block);length+=len(block)
                assert length==info.size==row['bytes'] and info.mode==row['mode'] and h.hexdigest()==row['sha256']
                total+=length
        assert seen==set(m['members']) and len(seen)==m['member_count'] and total==m['uncompressed_file_bytes']
        dec=lzma.LZMADecompressor();tar_sha=hashlib.sha256();tar_length=0
        with archive.open('rb') as stream:
            while not dec.eof:
                block=stream.read(1<<20) if dec.needs_input else b''
                assert block or not dec.needs_input
                raw=dec.decompress(block,max_length=1<<20);tar_sha.update(raw);tar_length+=len(raw)
            assert not dec.unused_data and stream.read(1)==b''
        result.append({'archive':spec['path'],'sha256':m['archive']['sha256'],'members':len(seen),
                       'file_bytes':total,'single_XZ_EOF':True,'tar_bytes':tar_length,'tar_sha256':tar_sha.hexdigest()})
    print(json.dumps({'status':'BYTE_CUSTODY_PASS','package_payloads':len(package['files']),
        'archives':result,'native_executed':False,'Torch_executed':False}))

if __name__=='__main__':main()
