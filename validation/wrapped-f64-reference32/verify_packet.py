"""Verify and optionally restore the immutable actual32 evidence and replay pure admission."""
import argparse,hashlib,io,json,lzma,stat,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
def sha(b):return hashlib.sha256(b).hexdigest()
def unpack(raw,m,nested=False):
    if nested:
        assert len(raw)==m['archive']['bytes'] and sha(raw)==m['archive']['sha256']
        expected_tar=m['tar']
    else:
        assert len(raw)==m['archive_bytes'] and sha(raw)==m['archive_sha256']
        expected_tar={'bytes':m['tar_bytes'],'sha256':m['tar_sha256']}
    d=lzma.LZMADecompressor(format=lzma.FORMAT_XZ);t=d.decompress(raw)
    assert d.eof and not d.unused_data and len(t)==expected_tar['bytes'] and sha(t)==expected_tar['sha256']
    got={}
    with tarfile.open(fileobj=io.BytesIO(t),mode='r:') as tf:
        for x in tf:
            assert x.isfile() and not PurePosixPath(x.name).is_absolute() and '..' not in PurePosixPath(x.name).parts
            assert x.name not in got and x.name in m['files']
            b=tf.extractfile(x).read();v=m['files'][x.name]
            assert len(b)==x.size==v['bytes'] and sha(b)==v['sha256'] and x.mode==v['mode']
            got[x.name]=(b,x.mode)
    assert set(got)==set(m['files']) and len(got)==m['file_count']
    assert sum(len(b)for b,mode in got.values())==m['expanded_bytes']
    return got
def restore(members,root):
    root.mkdir(parents=True,exist_ok=False)
    for name,(b,mode)in members.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(b);p.chmod(mode)
        assert p.read_bytes()==b and stat.S_IMODE(p.stat().st_mode)==mode
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--packet',type=Path,required=True);ap.add_argument('--restore',type=Path);ap.add_argument('--admission-output',type=Path)
    a=ap.parse_args();p=a.packet.resolve();m=json.loads((p/'manifest.json').read_bytes());outer=json.loads((p/'outer.json').read_bytes())
    for n,v in outer['files'].items():
        q=p/n;b=q.read_bytes()
        assert len(b)==v['bytes'] and sha(b)==v['sha256']
        assert stat.S_IMODE(q.stat().st_mode)in(v['mode'],0o644)
    members=unpack((p/'complete.tar.xz').read_bytes(),m)
    nm=members['original-packet/manifest.json'][0];nr=members['original-packet/complete.tar.xz'][0]
    assert sha(nm)=='9380a48145c632f8cabde4596d86de6968716211796ea1d42ca171c3f1e24a26'
    assert sha(nr)=='77e032dca727bd901471b95d15148e9936bf46da20c9141370e219eb4e2b273a'
    original=unpack(nr,json.loads(nm),True);assert len(original)==91
    pins={'admit_actual32.py':'81a6d312d1428dd2d94e3a1cfdd3a0ae1fddc0314032e09c677c454e5f0ed331',
          'parent_admit_actual26_r03.py':'be8c14f2161dbae865a2e8fc0d79c5747a3065538f746ba850a1091d4d0429c3',
          'admission-01/report.json':'767923408156d36d1f2247ce85b16f70f228bcf222a6f2068bf1b18c118f1e6d'}
    for n,pin in pins.items():assert sha(members[n][0])==pin
    if a.restore:
        root=a.restore.resolve();restore(members,root);restore(original,root/'original-packet/payload')
        if a.admission_output:
            output=a.admission_output.resolve()
            command=[sys.executable,'-I','-S','-B',str(root/'admit_actual32.py'),'--root',str(root),'--output',str(output)]
            r=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True)
            assert r.returncode==0,(r.returncode,r.stdout,r.stderr)
            assert (output/'report.json').read_bytes()==members['admission-01/report.json'][0]
    else:assert a.admission_output is None
    print(json.dumps({'status':'PASS','outer_members':len(members),'original_members':len(original),
      'all_bytes_modes_single_XZ_verified':True,'restored':bool(a.restore),'pure_admission_replayed':bool(a.admission_output),
      'Torch_executed':False,'native_executed':False,'original_actual_status':'OBSERVED'}))
if __name__=='__main__':main()
