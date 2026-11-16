"""Archive completed native evidence, verify all bytes and XZ EOF, then compact."""
import hashlib
import json
import lzma
from pathlib import Path
import stat
import tarfile

sha=lambda raw:hashlib.sha256(raw).hexdigest()


def archive(directory,archive_directory,name):
    directory,archive_directory=Path(directory),Path(archive_directory)
    archive_directory.mkdir(parents=True,exist_ok=True)
    path=archive_directory/(name+'.tar.xz');assert not path.exists();members={}
    with lzma.open(path,'wb',preset=3) as xz:
        with tarfile.open(fileobj=xz,mode='w|') as tar:
            for f in sorted(directory.rglob('*')):
                if not f.is_file():continue
                relative=f.relative_to(directory).as_posix();raw=f.read_bytes()
                info=tar.gettarinfo(str(f),arcname=relative);assert info.isfile()
                members[relative]={'bytes':len(raw),'sha256':sha(raw),'mode':stat.S_IMODE(f.stat().st_mode)}
                with f.open('rb') as source:tar.addfile(info,source)
    seen=set()
    with tarfile.open(path,'r:xz') as tar:
        for item in tar:
            pin=members[item.name];raw=tar.extractfile(item).read()
            assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256'] and item.mode==pin['mode'];seen.add(item.name)
    assert seen==set(members)
    decoder=lzma.LZMADecompressor()
    with path.open('rb') as reader:
        while compressed:=reader.read(65536):
            assert not decoder.eof
            decoder.decompress(compressed)
            assert not decoder.unused_data
    assert decoder.eof
    removed=[]
    for name2,pin in members.items():
        f=directory/name2
        if f.suffix in ('.json','.py','.md','.txt','.stdout','.stderr'):continue
        assert sha(f.read_bytes())==pin['sha256'] and stat.S_IMODE(f.stat().st_mode)==pin['mode']
        f.unlink();removed.append(name2)
    receipt={'archive':path.name,'archive_bytes':path.stat().st_size,'archive_sha256':sha(path.read_bytes()),
             'member_count':len(members),'original_bytes':sum(p['bytes'] for p in members.values()),'members':members,
             'all_member_bytes_sizes_modes_verified':True,'xz_complete_EOF_verified':True,'single_xz_stream_verified':True,
             'expanded_verified_raw_removed':removed,'restoration_base':str(directory)}
    manifest=archive_directory/(name+'.manifest.json');manifest.write_text(json.dumps(receipt,indent=2)+'\n')
    (directory/'RESTORE.json').write_text(json.dumps({'archive':str(path),'manifest':str(manifest)},indent=2)+'\n')
    return {k:v for k,v in receipt.items() if k not in ('members','expanded_verified_raw_removed')}
