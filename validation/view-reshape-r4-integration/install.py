#!/usr/bin/env python3
"""Install immutable R4 integration receipts and retain original control fixtures."""
from pathlib import Path
import hashlib,io,json,lzma,os,stat,subprocess,sys,tarfile,traceback
R=Path('/home/qian-qi/heptatorch-continuation-20261116/runs')
REPO=R/'view-reshape-r4-integration-01/repo'
SOURCE=R/'root-r4-integration-evidence-source-01'
OUT=REPO/'validation/view-reshape-r4-integration'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):
    s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1,p
    b=p.read_bytes();return b,{'bytes':len(b),'sha256':sha(b),'mode':stat.S_IMODE(s.st_mode)}
def write(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b)
    p.chmod(0o644);assert p.read_bytes()==b
def dump(p,v):write(p,(json.dumps(v,indent=2,sort_keys=True)+'\n').encode())
OUT.mkdir(exist_ok=False)
report={'schema':'heptatorch.r4.integration-evidence-install.v1','status':'RUNNING','native_executed':False,'compiler_executed':False,'Torch_executed':False,'closed_schema_count':0,'whole_pytorch_gate':'HOLD','files':{},'source_inputs':{}}
code=1
try:
    plan=json.loads((SOURCE/'plan.json').read_bytes())
    for row in plan:
        source=Path(row['source']);b,m=read(source)
        if row['sha256'] is not None:assert m['sha256']==row['sha256'],source
        dest=OUT/row['destination'];write(dest,b)
        report['files'][row['destination']]={**m,'source_path':str(source),'installed_mode':0o644}
        report['source_inputs'][str(source)]=m
    for name in ['plan.json','tool-receipts.json','install.py']:
        b,m=read(SOURCE/name);write(OUT/name,b);report['source_inputs'][str(SOURCE/name)]=m
        report['files'][name]={**m,'source_path':str(SOURCE/name),'installed_mode':0o644}
    members={}
    archive=OUT/'control-fixtures-original.tar.xz'
    with tarfile.open(archive,'x:xz',format=tarfile.PAX_FORMAT,preset=1) as tf:
        for dirname in ['root-r4-ci-sealer-controls-01','root-r4-ci-sealer-controls-02']:
            for path in sorted((R/dirname).rglob('*')):
                if path.is_dir():continue
                b,m=read(path);name=str(path.relative_to(R))
                row=tarfile.TarInfo(name);row.size=len(b);row.mode=m['mode'];row.mtime=0;tf.addfile(row,io.BytesIO(b))
                members[name]={**m,'source_path':str(path)};report['source_inputs'][str(path)]=m
    packed,am=read(archive);d=lzma.LZMADecompressor();d.decompress(packed)
    assert d.eof and not d.unused_data
    with tarfile.open(archive,'r:xz') as tf:
        rows=tf.getmembers();assert len(rows)==len(members) and {r.name for r in rows}==set(members)
        for row in rows:
            b=tf.extractfile(row).read();m=members[row.name]
            assert row.isfile() and len(b)==m['bytes'] and sha(b)==m['sha256'] and row.mode==m['mode']
        end=max(r.offset_data+((r.size+511)//512)*512 for r in rows);tf.fileobj.seek(end);tail=tf.fileobj.read()
        assert len(tail)>=1024 and not any(tail)
    dump(OUT/'control-fixtures-manifest.json',{'schema':'heptatorch.r4.synthetic-file-controls-originals.v1','fixture_only':True,'production_reference':False,'archive':am,'members':members,'member_count':len(members)})
    p=subprocess.run([sys.executable,'-I','-S','-B','tools/check_acceptance.py'],cwd=REPO,stdin=subprocess.DEVNULL,capture_output=True,timeout=60)
    write(OUT/'acceptance-source-check.stdout',p.stdout);write(OUT/'acceptance-source-check.stderr',p.stderr)
    report['acceptance_source_check']={'exit_code':p.returncode,'stdout':{'bytes':len(p.stdout),'sha256':sha(p.stdout)},'stderr':{'bytes':len(p.stderr),'sha256':sha(p.stderr)}}
    assert p.returncode==0 and not p.stderr,('original acceptance source guard',p.returncode,p.stderr.decode())
    for spelling,m in report['source_inputs'].items():assert read(Path(spelling))[1]==m,spelling
    report.update(status='PASS_CREATE_ONLY_EVIDENCE_INSTALL',all_source_bytes_modes_unchanged=True,control_archive=am,control_members=len(members),historical_failures_preserved=True)
    code=0
except BaseException as e:report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
dump(OUT/'report.json',report)
print(json.dumps({'status':report['status'],'error':report.get('error'),'files':len(report['files']),'control_members':report.get('control_members')}))
raise SystemExit(code)
