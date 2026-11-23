from pathlib import Path
import sys
root=Path(sys.argv[1]);dst=Path(sys.argv[2])
new='d160a44ae8857aab581f24d41f92cb45eb6ec07cc5d90a655a68cda9760c07b2'
old='7ebb937c818ce8ab36791b94a829a3a2a4aad1c3599eb80c80d38789c72149d9'
def edit(file,*pairs):
 s=(root/file).read_text()
 for a,b in pairs:
  assert s.count(a)==1,(file,a,s.count(a));s=s.replace(a,b)
 (dst/file).write_text(s)
edit('oracle_fixture.py',("(D/'official.fixture.json')",'''def f64_raw_tensor(words):
 return torch.tensor([w if w<1<<63 else w-(1<<64) for w in words],dtype=torch.int64).view(torch.float64)
for j in range(4):
 boundary=j==3
 root_values=[0x3ff0000000000000,0x8000000000000000,0x7ff8000000001111,0x7ff0000000004321,0xbff0000000000000] if boundary else [
  0x7ff0000000005432,0xfff8000000003333,0x7ff0000000000000,0x0000000000000001,0x3ff0000000000000,
  0x8000000000000001,0x8000000000000000,0x7fefffffffffffff,0xfff0000000000000,0x0000000000000000]
 shape,strides,offset=([2,2],[1,2],1) if boundary else ([2,3],[1,2],2)
 probes=[
  [0x7ff8000000002345,1,0x7ff0000000000000,0xfff0000000000012,0xfff0000000000000,0x7fffffffffffffff],
  [0x7ff0000000005432,0x8000000000000001,0x000fffffffffffff,0x3ff0000000000000,0xfff8000000001111,0x0010000000000000],
  [0xfff800000000deb1,0x800fffffffffffff,0xbff0000000000000,0xfff0000000000000,0x3ff0000000000000,1]
 ]
 values=[0,0x7ff0000000004321,0x8000000000000000,0xfff8000000007777] if boundary else [
  ((0x8000000000000000 if i&1 else 0) if i>>j&1 else probes[j][i]) for i in range(6)]
 base=f64_raw_tensor(root_values)
 dst=base.as_strided(shape,strides,offset);sib=base.as_strided(shape,strides,offset)
 src=f64_raw_tensor(values).reshape(shape)
 before=data(base);initial=data(src);version=base._version
 assert before==b''.join(v.to_bytes(8,'little') for v in root_values).hex()
 assert initial==b''.join(v.to_bytes(8,'little') for v in values).hex()
 ptr=dst.data_ptr(); storage=dst.untyped_storage().data_ptr()
 assert not dst.is_contiguous() and dst.storage_offset()==offset and dst._base is base
 assert torch.logical_not(src,out=dst) is dst and dst.data_ptr()==ptr
 assert dst.untyped_storage().data_ptr()==storage
 assert dst._version==base._version==sib._version==version+1 and data(src)==initial
 assert data(sib)==data(dst)
 rows.append(dict(dtype=4,base_shape=[len(root_values)],shape=shape,strides=strides,offset=offset,
   prior_exact_boundary=boundary,root_values=root_values,base_initial=before,input=values,
   initial=initial,logical=data(dst),base_physical=data(base),version_delta=1))
(D/'official.fixture.json')'''))
edit('native_observe.py',('f24a571bebd4f85e5c55dbb2ed55eef4d237bea32f0a3baea3e893a0187b500b',old),("'candidate':'"+old,"'candidate':'"+new),
("if mode=='baseline' and not (offset and dtype==2):continue","if mode=='baseline' and not (offset and dtype==4):continue"),
("  if offset and dtype in (1,2) and not row['prior_exact_boundary']:","  if offset and dtype in (1,2,4) and not row['prior_exact_boundary']:"),
("negative({1:'bool',2:'i64'}[dtype]+'-offset-partially-overlapping-'+str(j),[(128,partial,dst)])","negative({1:'bool',2:'i64',4:'f64'}[dtype]+'-offset-partially-overlapping-'+str(j),[(128,partial,dst)])"),
("  unsupported('f64-nonzero-offset',[5],[2,2],[1,2],1,4)","""  unsupported('f64-offset-contiguous',[10],[2,3],[3,1],2,4)
  unsupported('f64-offset-gapped',[12],[2,3],[1,3],2,4)
  unsupported('f64-offset-overlap',[10],[2,3],[1,1],2,4)
  unsupported('f64-offset-zero-stride',[10],[2,3],[0,1],2,4)"""))
edit('pure_readback.py',("width={0:4,1:1,2:8}[dtype]","width={0:4,1:1,2:8,4:8}[dtype]"),
("('offset-same-storage-','bool-offset-partially-overlapping-','i64-offset-partially-overlapping-')","('offset-same-storage-','bool-offset-partially-overlapping-','i64-offset-partially-overlapping-','f64-offset-partially-overlapping-')"),
("dtype=src[72];width={1:1,2:8}[dtype]","dtype=src[72];width={1:1,2:8,4:8}[dtype]"),
("key={1:'bool',2:'i64'}[dtype]","key={1:'bool',2:'i64',4:'f64'}[dtype]"),
("     if src[72] in (1,2):\n      key={1:'bool',2:'i64'}[src[72]]","     if src[72] in (1,2,4):\n      key={1:'bool',2:'i64',4:'f64'}[src[72]]"),
("assert vn[72]==sn[72]==2","assert vn[72]==sn[72]==4"),("assert rn[72]==2 and rn[83]==0","assert rn[72]==4 and rn[83]==0"),
("key={0:'f32',1:'bool',2:'i64'}[dtype]","key={0:'f32',1:'bool',2:'i64',4:'f64'}[dtype]"))
edit('retained79_observe.py',(old,new))
