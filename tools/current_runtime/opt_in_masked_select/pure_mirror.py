"""Compare archived clean/overcap-recovery requests, owner bytes and results offline."""
from pathlib import Path
import sys,json,zlib
sys.path.insert(0,str(Path(__file__).parent))
from pure import frames,req,resp,sha,Owner
d=Path(sys.argv[1]); f=json.loads((d/'mirror.json').read_bytes())
E='75401b00356b8941960fae2a2207c0e046fb0da36cbcef7de26bef41ec4a3deb'
assert f['elf_sha256']==E
res={}
for a in f['cases']:
 label=a['mask'];q0=d/(label+'-clean');q1=d/(label+'-reject')
 assert a['clean_vs_reject'][0]==a['clean_vs_reject'][1]
 c=a['clean_vs_reject'][0]
 def mem(root,n):
  b=zlib.decompress((root/(n+'.zlib')).read_bytes());assert sha(b)==n;return b
 assert mem(q0,c['start_owner'])==mem(q1,c['start_owner'])==mem(q1,c['post_check_owner'])
 after=mem(q0,c['result_owner']);assert after==mem(q1,c['result_owner'])
 h,st,v=c['ids'];o=Owner(after);ds=o.node(h);assert ds[27:29]==(st,v) and ds[12]==3 and ds[16]==1
 assert sha(o.storage(st)[2])==c['raw_words_sha256'] and o.row(3,v)[1][2]==0
 Q0=list(frames(q0/'requests.frames.bin',b'HTOWNR01'));Q1=list(frames(q1/'requests.frames.bin',b'HTOWNR01'))
 R0=list(frames(q0/'responses.frames.bin',b'HTOWNS01'));R1=list(frames(q1/'responses.frames.bin',b'HTOWNS01'))
 j=len(Q1)-2;assert resp(R1[j])==(3,[]) and req(Q1[j])[0][0]==130
 assert Q0==Q1[:j]+Q1[j+1:] and R0==R1[:j]+R1[j+1:]
 res[label]=dict(clean_frames=len(Q0),reject_frames=len(Q1),exact_post_owner=True,owner_ids=c['ids'])
print(json.dumps(res,sort_keys=True,indent=2))
