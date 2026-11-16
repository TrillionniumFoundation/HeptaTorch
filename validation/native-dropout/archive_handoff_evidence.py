"""Archive completed evidence; check every member against its original bytes/mode."""
from pathlib import Path
import gzip,hashlib,json,tarfile,io
R=Path('/dev/shm/heptatorch-d26699e25fea/native-dropout-dtype-extension')
S=Path('/workspace/scratch/d26699e25fea/native-dropout-dtype-extension')
OUT=R/'archive-staging-11';OUT.mkdir(exist_ok=True)
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def archive(name,entries):
    files={}
    for logical,p in entries:
        if p.is_dir():
            for f in sorted(p.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts:
                    files[logical+'/'+f.relative_to(p).as_posix()]=f
        else:
            assert p.is_file(),p;files[logical]=p
    manifest={'schema':'heptatorch.verified-evidence-archive.v1','name':name,'files':{}}
    for logical,p in sorted(files.items()):
        raw=p.read_bytes();manifest['files'][logical]={'bytes':len(raw),'sha256':sha(raw),'mode':p.stat().st_mode&0o777}
    a=OUT/(name+'.tar.gz')
    with a.open('wb') as rawout,gzip.GzipFile(filename='',mode='wb',fileobj=rawout,mtime=0,compresslevel=9) as gz,tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
        for logical,p in sorted(files.items()):
            raw=p.read_bytes();pin=manifest['files'][logical]
            assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256']
            ti=tarfile.TarInfo(logical);ti.size=len(raw);ti.mode=pin['mode'];ti.mtime=0
            tar.addfile(ti,io.BytesIO(raw))
    with tarfile.open(a,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==len(files)
        assert {m.name for m in members}==set(files)
        for m in members:
            raw=tar.extractfile(m).read();pin=manifest['files'][m.name]
            assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256'] and m.mode&0o777==pin['mode']
            p=files[m.name];assert p.read_bytes()==raw and p.stat().st_mode&0o777==pin['mode']
    manifest.update(status='VERIFIED',archive_bytes=a.stat().st_size,archive_sha256=sha(a.read_bytes()),members=len(files),uncompressed_bytes=sum(p['bytes'] for p in manifest['files'].values()),all_member_bytes_and_modes_verified=True,originals_removed=False)
    m=OUT/(name+'.manifest.json');m.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'}),flush=True)

def select(names):return [(n,R/n) for n in names]
if __name__ == '__main__':
    archive('authoring-candidates01-05',select(['candidate-'+f'{i:02}' for i in range(1,6)]))
    archive('authoring-candidates06-10',select(['candidate-'+f'{i:02}' for i in range(6,11)]+[f'candidate-{i:02}.{e}' for i in range(7,11) for e in ('stdout','stderr')]))
    old=['actual28-04-first','actual28-05','actual28-06-first','allocation-07-first','allocation-07-second','dtype-admission-07-first','dtype-admission-07-second','dtype-admission-07-third','existing-optimizer-10-first','history-controls-06-first','history-mul-controls-07-first','hold-read-04-first','inherited-05-first','inherited-05-second','inherited-06-first','inherited-06-second','lifetimes-05-first','lifetimes-06-first','logical-not-07-first','optimizer-history-07-first','original8-05-first','original8-06-first','retained12-05','retained12-06-first','sgd-layout-10-first','slice-04-first','slice-04-second','slice-05','slice-06-first']
    archive('historical-native-executions',select(old))
    archive('final11-native-and-authors',select(['candidate-11','candidate-11.stdout','candidate-11.stderr']))
    archive('final11-original-qualification',select(['inherited-11-first','inherited-11-first.stdout','inherited-11-first.stderr','local-controls-11-first','local-controls-11-admitted.json','existing-optimizer-11-first','existing-optimizer-11-first.stdout','existing-optimizer-11-first.stderr','sgd-layout-11-first','sgd-layout-11-first.stdout','sgd-layout-11-first.stderr']))
    source=[('source/'+p.relative_to(S).as_posix(),p) for p in S.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.gz','.xz')]
    archive('source-history-and-reference-custody',source+[('sgd-layout-actual-01',R/'sgd-layout-actual-01')])
