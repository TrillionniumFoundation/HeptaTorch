"""Independent offline decoder: no project protocol, producer imports, or native calls."""
import hashlib,json,struct,sys,zlib
from pathlib import Path
def words(raw,n,at=0):return struct.unpack_from('<'+'Q'*n,raw,at)
def unpack_frames(path,magic):
 b=path.read_bytes(); off=0
 while off<len(b):
  assert b[off:off+8]==magic
  (length,)=words(b,1,off+8); off+=16
  assert 32<=length<=1<<20 and off+length<=len(b)
  yield b[off:off+length];off+=length
 assert off==len(b)
def req(b):
 assert b[:8]==b'HTORCH03'
 n,ln,z=words(b,3,8);assert ln==len(b) and z==0
 return [words(b,8,32+i*64) for i in range(n)]
def res(b):
 assert b[:8]==b'HTORSP03'
 version,n,st=words(b,3,8);assert version==3
 if st:
  assert n==0 and len(b)==32
  return st,[]
 out=[];off=32
 for i in range(n):
  op,e,val,ln=words(b,4,off);assert e==0;off+=32
  assert off+ln<=len(b)
  out.append((op,val,b[off:off+ln]));off+=ln
 assert off==len(b)
 return st,out
def calc(d):
 d=Path(d);rep=json.loads((d/'report.json').read_bytes())
 qs=list(map(req,unpack_frames(d/'requests.frames.bin',b'HTOWNR01')))
 rs=list(map(res,unpack_frames(d/'responses.frames.bin',b'HTOWNS01')))
 assert len(qs)==len(rs)==rep['frames']
 for q,(st,r) in zip(qs,rs):
  if not st:assert [q_[0] for q_ in q]==[r_[0] for r_ in r]
 sha=lambda b:hashlib.sha256(b).hexdigest()
 total=refused=byteproofs=0
 for row in rep['rows']:
  if 'status' in row:
   no=row['frame']; status, records=rs[no-1]
   a,b=[zlib.decompress((d/f'{no:03d}-{side}.owner.zlib').read_bytes()) for side in ['pre','post']]
   assert len(a)==row['length'] and a==b and sha(a)==row['sha'] and (status,records)==(row['status'],[])
   refused+=1;byteproofs+=len(a)
  else:
   i=row['write_frame']-1
   while [x[0] for x in qs[i]]!=[128,128,98]:
    i+=1;assert i<row['write_frame']+2
   q=qs[i];status,rr=rs[i]; out=row['destination'];n=row['numel']
   assert status==0 and q[0][2]==out and q[1][2]==0xfe000001 and q[2][1]==0xfe000002
   assert rr[0]==rr[1]==(128,out,b'') and rr[2][0:2]==(98,n)
   assert len(rr[2][2])==n and set(rr[2][2])<={0,1} and sha(rr[2][2])==row['output_sha']
   in0=rs[i-4:i-2]; out0=rs[i-2:i]; out1=rs[i+1:i+3]; in1=rs[i+3:i+5]
   assert in0==in1
   old=words(out0[0][1][0][2],16); new=words(out1[0][1][0][2],16)
   assert new[:13]==old[:13] and new[13]==old[13]+2 and new[14:]==old[14:], (old,new)
   assert out1[1][1][0][2]==rr[2][2]
   total+=n
 return dict(image=rep['image'],frames=len(rs),positive_rows=sum('numel' in x for x in rep['rows']),boolean_elements=total,rollback_rows=refused,rollback_owner_bytes=byteproofs,requests_sha256=sha((d/'requests.frames.bin').read_bytes()),responses_sha256=sha((d/'responses.frames.bin').read_bytes()))
print(json.dumps([calc(p) for p in sys.argv[1:]],indent=2))
