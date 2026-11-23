"""Derive BF16 raw constructor tests and retain previous F16/layout fixtures exactly."""
from pathlib import Path
import sys,hashlib
setup,prev,d=map(Path,sys.argv[1:])
def load(base,name,digest):
 p=base/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
 return p.read_text()
def one(t,a,b):
 assert t.count(a)==1,(a,t.count(a))
 return t.replace(a,b,1)
PRED='b680ef8390e5438240733d8c20a55931b3e059f41b40f8f86db08e153021609e'
ELF='d75b5e51dbef318a70d233e1c68f50bd2cc22ce7eb0f111c3a730ecaa91dafdb'
o=load(setup,'oracle_f16_view.py','9adcfcc548e50a988d0ef958a563c9e75981a2ff3ef60b62120fb1014c5cf607')
o=one(o,'torch.float16','torch.bfloat16')
o=one(o,'words=[0,0x8000,1,0x8001,0x7c00,0xfc00,0x7e55,0x7d41,0xfe32,0xfd04,0x3c00,0xbc00,0x3c01,0xb802,0x0400,0x8400,0x03ff,0x83ff,0x7bff,0xfbff,0x4000,0x4401,0x3555,0x481b]',
      'words=[0,0x8000,1,0x8001,0x7f80,0xff80,0x7fc1,0x7fa1,0xffc3,0xffa3,0x3f80,0xbf80,0x3f81,0xb801,0x0080,0x8080,0x007f,0x807f,0x7f7f,0xff7f,0x4000,0x4401,0x3555,0x481b]')
(d/'oracle_bf16_view.py').write_text(o)
n=load(setup,'native_f16_view.py','f8d79fd58dee38a44fc58de157ac8721324ec4f044c611091ddbef5a10f5d596')
n=one(n,'6c8e5d338f63f7cf2cd4437b1f49294e635d6f07e9927b4b1f0ccdd11d8a1095',PRED)
n=one(n,'2196cd3811604609925ba6da965b105f35cd0336e3ac6f7f7b7fab44df745ebb',ELF)
n=one(n,'def make(shape,words,dtype=5):','def make(shape,words,dtype=6):')
n=one(n,"  bf=make([6],[0,1,2,3,4,5],6)\n  refuse('bf16-still-denied',7,view(bf,[2,3],[1,2]))\n",'')
n=one(n,'new-f16-view-is-still-not-out-covered-','new-bf16-view-is-still-not-out-covered-')
n=one(n,'f16-root-grad-toggle','bf16-root-grad-toggle')
(d/'native_bf16_view.py').write_text(n)
p=load(setup,'pure_f16_view.py','251e88956f164faf26d4d0a8190ce8b265614142915e5b179ad39fbac81c6eb0')
assert p.count('[72]==5')==5
p=p.replace('[72]==5','[72]==6')
p=one(p,'(FACTORY,5)','(FACTORY,6)')
p=one(p,'new-f16-view-is-still-not-out-covered-','new-bf16-view-is-still-not-out-covered-')
p=one(p,'f16-root-grad-toggle','bf16-root-grad-toggle')
p=one(p,"   if x['case']=='bf16-still-denied':\n    assert qs[i][0][0]==111 and Owner(before).node(qs[i][0][1])[72]==6\n",'')
(d/'pure_bf16_view.py').write_text(p)
r=load(prev,'native_f16_view.py','78e2e8366f1963c4bdd77c3a16187ca98516542a9367f1da73b2e89c83923cf6')
r=one(r,PRED,ELF)
r=one(r,"  bf=make([6],[0,1,2,3,4,5],6)\n  refuse('bf16-still-denied',7,view(bf,[2,3],[1,2]))\n",'')
(d/'retain_f16_view.py').write_text(r)
rr=load(prev,'native_observe.py','efad1e3e32bbb5a59997d8eef38e51562500bd1f84d2df1c3ec8907d9a761f74')
(d/'retain_layout.py').write_text(one(rr,PRED,ELF))
rr=load(prev,'retained79_observe.py','632a9d527b4740bb73f93a07c1112bbbf02077111379bb2f41f250c1b0ab5fe2')
(d/'retain79.py').write_text(one(rr,PRED,ELF))
