#!/usr/bin/env python3
"""Actual pinned PyTorch comparison; native executable has no Python runtime."""
import hashlib, importlib.metadata, json, platform, random, struct, subprocess, sys
from pathlib import Path
assert platform.machine()=="x86_64" and sys.platform=="linux" and list(sys.version_info[:2])==[3,12]
lock=json.loads(Path("compatibility/pytorch-oracle-lock.json").read_text())["oracle"]
assert json.loads(Path("compatibility/upstream-lock.json").read_text())["commit"]=="31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457"
import torch
assert torch.__version__==lock["version"] and torch.version.git_version==lock["git_commit"]
assert {k:getattr(torch.version,k) for k in lock["backend_versions"]}==lock["backend_versions"]
dist=importlib.metadata.distribution("torch")
assert dist.version==lock["version"]
direct=json.loads(dist.read_text("direct_url.json"))
assert direct["url"]==lock["wheel"]["url"] and direct["archive_info"]["hashes"]["sha256"]==lock["wheel"]["sha256"]
report=json.loads(Path(lock["install_report"]).read_text())
r=[i for i in report["install"] if i.get("metadata",{}).get("name","").lower()=="torch"]
assert len(r)==1 and r[0]["requested"] is True and r[0]["metadata"]["version"]==lock["version"]
assert r[0]["download_info"]["url"]==lock["wheel"]["url"]
assert r[0]["download_info"]["archive_info"]["hashes"]["sha256"]==lock["wheel"]["sha256"]
compiler=Path(sys.argv[1]).resolve()
assert hashlib.sha256(compiler.read_bytes()).hexdigest()=="87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54"
source=Path("tools/dtype-bf16/bf16.hepta")
images=[]
for _ in range(2):
    result=subprocess.run([str(compiler)],input=source.read_bytes(),capture_output=True,timeout=45,env={"PATH":""})
    assert result.returncode==0 and not result.stderr and result.stdout.startswith(b"\x7fELF"),(result.returncode,result.stderr)
    images.append(result.stdout)
assert images[0]==images[1]
exe=Path("build/bf16-native");exe.parent.mkdir(exist_ok=True);exe.write_bytes(images[0]);exe.chmod(0o755)
cases=0; exact=0;nans=0
def call(packet):
    p=subprocess.run([str(exe)],input=packet,capture_output=True,timeout=25,env={"PATH":""})
    assert p.returncode==0 and not p.stderr,(p.returncode,p.stderr)
    return p.stdout
def compare(mode,values):
    global cases,exact,nans
    unit=4 if mode==0 else 2
    packet=struct.pack("<II",mode,len(values))+b"".join(v.to_bytes(unit,"little") for v in values)
    got=call(packet)
    assert got[:4]==bytes(4) and len(got)==4+len(values)*(2 if mode==0 else 4),(mode,len(values),got[:20])
    outunit=2 if mode==0 else 4
    found=[int.from_bytes(got[4+i*outunit:4+(i+1)*outunit],"little") for i in range(len(values))]
    if not values:cases+=1;return
    buf=bytearray(packet[8:])
    if mode==0:
        original=torch.frombuffer(buf,dtype=torch.float32).clone()
        reference=original.to(dtype=torch.bfloat16).view(torch.int16)
        nanmask=0x7f800000; payload=0x007fffff; targetmask=0x7f80;targetpayload=0x7f;mask=0xffff
    else:
        original=torch.frombuffer(buf,dtype=torch.bfloat16).clone()
        reference=original.to(dtype=torch.float32).view(torch.int32)
        nanmask=0x7f80; payload=0x7f;targetmask=0x7f800000;targetpayload=0x7fffff;mask=0xffffffff
    want=[v&mask for v in reference.tolist()]
    for i,(src,a,b) in enumerate(zip(values,found,want)):
        if src&nanmask==nanmask and src&payload:
            assert a&targetmask==targetmask and a&targetpayload,(i,hex(src),hex(a))
            nans+=1
        else:
            assert a==b,(mode,i,hex(src),hex(a),hex(b))
            exact+=1
    cases+=1
rng=random.Random(1729)
compare(0,[0,0x80000000,0x7f800000,0xff800000,0x7f7fffff,0xff7fffff,0x00000001,0x007fffff,
           0x3f800000,0x3f807fff,0x3f808000,0x3f818000,0x7f800001,0x7fc00000,0xffffffff])
for i in range(12):compare(0,[rng.getrandbits(32) for _ in range(1 if i==0 else (19 if i<3 else 1301))])
compare(0,[(i<<16)|0x8000 for i in range(0,65536,7)])
compare(1,list(range(65536)))
compare(0,[]);compare(1,[])
for packet in (bytes(7),struct.pack("<II",2,0),struct.pack("<II",0,1)+bytes(2),
               struct.pack("<II",0,200001),struct.pack("<II",1,0)+b"x"):
    assert call(packet)==b"\x01\x00\x00\x00"
    cases+=1
print(json.dumps({"schema":"heptatorch.native-bf16-oracle.v1","status":"PASS",
                  "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                  "compiler_sha256":hashlib.sha256(compiler.read_bytes()).hexdigest(),
                  "native_executable_sha256":hashlib.sha256(images[0]).hexdigest(),
                  "cases":cases,"exact_non_nan":exact,"nan_classification":nans,
                  "scope":"standalone native HIA CPU bit conversion; not integrated Tensor dtype parity"}))
