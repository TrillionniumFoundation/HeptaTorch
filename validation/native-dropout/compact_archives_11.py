from pathlib import Path
import runpy,gzip,lzma,tarfile,hashlib,json
R=Path('/dev/shm/heptatorch-d26699e25fea/native-dropout-dtype-extension')
ctx=runpy.run_path(str(R/'archive_handoff_evidence.py'))
ctx['archive']('portable-cold-11-first',[(n,R/n) for n in ('portable-cold-11-first','portable-cold-11-first.stdout','portable-cold-11-first.stderr')])
D=R/'archive-staging-11';summary=[]
for path in sorted(D.glob('*.tar.gz')):
    record=json.loads(path.with_name(path.name[:-7]+'.manifest.json').read_bytes())
    new=path.with_name(path.name[:-7]+'.tar.xz')
    hin=hashlib.sha256()
    with gzip.open(path,'rb') as source,lzma.open(new,'wb',preset=6) as target:
        while True:
            part=source.read(1024*1024)
            if not part:break
            hin.update(part);target.write(part)
    hout=hashlib.sha256()
    with lzma.open(new,'rb') as stream:
        while True:
            part=stream.read(1024*1024)
            if not part:break
            hout.update(part)
    assert hin.digest()==hout.digest()
    with tarfile.open(new,'r:xz') as archive:
        members=archive.getmembers();assert len(members)==record['members']
        assert {x.name for x in members}==set(record['files'])
        for member in members:
            raw=archive.extractfile(member).read();pin=record['files'][member.name]
            assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'] and member.mode&0o777==pin['mode']
    record['original_gzip']={'sha256':record['archive_sha256'],'bytes':record['archive_bytes']}
    record['archive_format']='tar.xz';record['archive_sha256']=hashlib.sha256(new.read_bytes()).hexdigest();record['archive_bytes']=new.stat().st_size
    record['tar_bytes_sha256']=hin.hexdigest();record['all_member_bytes_and_modes_verified']=True
    m=new.with_name(new.name[:-7]+'.xz-manifest.json');m.write_text(json.dumps(record,indent=2)+'\n')
    row={k:v for k,v in record.items() if k not in ('files',)};row['manifest_sha256']=hashlib.sha256(m.read_bytes()).hexdigest();summary.append(row);print(json.dumps(row),flush=True)
(D/'compact-summary.json').write_text(json.dumps({'status':'VERIFIED','archives':summary,'total_archive_bytes':sum(x['archive_bytes'] for x in summary)},indent=2)+'\n')
