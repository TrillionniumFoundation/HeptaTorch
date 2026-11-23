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
 total=refused=byteproofs=self_rows=self_values=numeric_rows=numeric_words=numeric_bytes=0
 for row in rep['rows']:
  if 'status' in row:
   no=row['frame']; status, records=rs[no-1]
   a,b=[zlib.decompress((d/f'{no:03d}-{side}.owner.zlib').read_bytes()) for side in ['pre','post']]
   assert len(a)==row['length'] and a==b and sha(a)==row['sha'] and (status,records)==(row['status'],[])
   refused+=1;byteproofs+=len(a)
  elif row.get('self_alias'):
   j=row['paired_frame']-1; h=row['destination']; n=row['numel']; ref1=0xfe000001;ref3=0xfe000003
   before=bytes.fromhex(row['expected_restoration']); inv=bytes.fromhex(row['expected_inversion'])
   if n:
    assert qs[j]==[(128,h,h,0,0,0,0,0),(98,ref1,0,n,0,0,0,0),(128,ref1,ref1,0,0,0,0,0),(98,ref3,0,n,0,0,0,0)]
    assert rs[j]==(0,[(128,h,b''),(98,n,inv),(128,h,b''),(98,n,before)])
    old_meta=rs[j-3][1][0][2]
    assert rs[j-2][1][0][2]==before and rs[j-1]==(1,[])
    assert inv==bytes(1-i for i in before)
   else:
    assert qs[j]==qs[j-3]==[(128,h,h,0,0,0,0,0)]
    assert rs[j]==rs[j-3]==(0,[(128,h,b'')])
    old_meta=rs[j-6][1][0][2]
    mid=words(rs[j-2][1][0][2],16)
    assert mid[13]==row['version_before']+1
    assert rs[j-4]==(1,[])
   new_meta=rs[j+1][1][0][2]
   old=words(old_meta,16);new=words(new_meta,16)
   assert new[:13]==old[:13] and new[13]==old[13]+2 and new[14:]==old[14:]
   assert new[13]==row['version_after'] and old[13]==row['version_before']
   assert rs[j+2]==(0,[(98,n,before)])
   self_rows+=1;self_values+=2*n
  else:
   i=row['write_frame']-1
   while [x[0] for x in qs[i]]!=[128,128,98]:
    i+=1;assert i<row['write_frame']+2
   q=qs[i];status,rr=rs[i]; out=row['destination'];n=row['numel']
   assert status==0 and q[0][2]==out and q[1][2]==0xfe000001 and q[2][1]==0xfe000002
   assert rr[0]==rr[1]==(128,out,b'') and rr[2][0:2]==(98,n)
   numeric=row.get('numeric_out',False)
   width=row.get('output_width',1); one=row.get('raw_one',1)
   payload=rr[2][2]
   assert len(payload)==n*width and sha(payload)==row['output_sha']
   assert all(int.from_bytes(payload[j:j+width],'little') in (0,one) for j in range(0,len(payload),width))
   out1=rs[i+1:i+3]; in1=rs[i+3:i+5]
   if numeric:
    assert rs[i-1]==(1,[])
    in0=rs[i-5:i-3];out0=rs[i-3:i-1]
    assert words(out0[0][1][1][2],4)[:2]==(row['output_tag'],width)
    assert out0[0][1][1]==out1[0][1][1]
    source=words(in0[0][1][1][2],4)[0]
    src_width=words(in0[0][1][1][2],4)[1]
    src_data=in0[1][1][0][2]
    mask=(1<<(src_width*8-1))-1 if source in (0,4,5,6) else (1<<(src_width*8))-1
    computed=b''.join((one if int.from_bytes(src_data[j:j+src_width],'little') & mask==0 else 0).to_bytes(width,'little') for j in range(0,len(src_data),src_width))
    assert len(src_data)==src_width*n and computed==payload
    numeric_rows+=1;numeric_words+=n;numeric_bytes+=len(payload)
   else:
    in0=rs[i-4:i-2]; out0=rs[i-2:i]
    total+=n
   assert in0==in1
   old=words(out0[0][1][0][2],16); new=words(out1[0][1][0][2],16)
   assert new[:13]==old[:13] and new[13]==old[13]+2 and new[14:]==old[14:], (old,new)
   assert out1[1][1][0][2]==rr[2][2]
 return dict(image=rep['image'],frames=len(rs),disjoint_positive_rows=sum('numel' in x and not (x.get('self_alias') or x.get('numeric_out')) for x in rep['rows']),disjoint_boolean_elements=total,exact_self_rows=self_rows,exact_self_boolean_elements=self_values,numeric_rows=numeric_rows,numeric_words=numeric_words,numeric_bytes=numeric_bytes,rollback_rows=refused,rollback_owner_bytes=byteproofs,requests_sha256=sha((d/'requests.frames.bin').read_bytes()),responses_sha256=sha((d/'responses.frames.bin').read_bytes()))
print(json.dumps([calc(p) for p in sys.argv[1:]],indent=2))
