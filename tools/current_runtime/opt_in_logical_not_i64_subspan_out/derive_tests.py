from pathlib import Path
import sys
root=Path(sys.argv[1]);dst=Path(sys.argv[2])
new='7ebb937c818ce8ab36791b94a829a3a2a4aad1c3599eb80c80d38789c72149d9'
old='f24a571bebd4f85e5c55dbb2ed55eef4d237bea32f0a3baea3e893a0187b500b'
def edit(file,*pairs):
 s=(root/file).read_text()
 for a,b in pairs:
  assert s.count(a)==1,(file,a,s.count(a));s=s.replace(a,b)
 (dst/file).write_text(s)
edit('oracle_fixture.py',("(D/'official.fixture.json')",'''for j in range(4):
 boundary=j==3
 mini,maxi=-(1<<63),(1<<63)-1
 root_values=[mini,-21,27,maxi,-455] if boundary else [mini,maxi,-31,57,-121,208,-677,982,-127,1234]
 shape,strides,offset=([2,2],[1,2],1) if boundary else ([2,3],[1,2],2)
 extrema=[mini,maxi,-(1<<40),1<<40,-7,2]
 values=[0,mini,0,maxi] if boundary else [0 if i>>j&1 else extrema[i] for i in range(6)]
 base=torch.tensor(root_values,dtype=torch.int64)
 dst=base.as_strided(shape,strides,offset);sib=base.as_strided(shape,strides,offset)
 src=torch.tensor(values,dtype=torch.int64).reshape(shape)
 before=data(base);initial=data(src);version=base._version
 ptr=dst.data_ptr(); storage=dst.untyped_storage().data_ptr()
 assert not dst.is_contiguous() and dst.storage_offset()==offset and dst._base is base
 assert torch.logical_not(src,out=dst) is dst and dst.data_ptr()==ptr
 assert dst.untyped_storage().data_ptr()==storage
 assert dst._version==base._version==sib._version==version+1 and data(src)==initial
 assert data(sib)==data(dst)
 rows.append(dict(dtype=2,base_shape=[len(root_values)],shape=shape,strides=strides,offset=offset,
   prior_exact_boundary=boundary,root_values=root_values,base_initial=before,input=values,
   initial=initial,logical=data(dst),base_physical=data(base),version_delta=1))
(D/'official.fixture.json')'''))
edit('native_observe.py',('b8558c915c0c1b475bfcbd46e654dda64930a30e18e835d3504e8b3c95851bef',old),("'candidate':'"+old,"'candidate':'"+new),
("if mode=='baseline' and not (offset and dtype==1):continue","if mode=='baseline' and not (offset and dtype==2):continue"),
("out_meta['storage_offset']==offset*(4 if dtype==0 else 1)","out_meta['storage_offset']==offset*{0:4,1:1,2:8,4:8}[dtype]"),
("  if offset and dtype==1 and not row['prior_exact_boundary']:","  if offset and dtype in (1,2) and not row['prior_exact_boundary']:"),
("negative('bool-offset-partially-overlapping-'+str(j),[(128,partial,dst)])","negative({1:'bool',2:'i64'}[dtype]+'-offset-partially-overlapping-'+str(j),[(128,partial,dst)])"),
("  unsupported('i64-nonzero-offset',[5],[2,2],[1,2],1,2)","""  unsupported('i64-offset-contiguous',[10],[2,3],[3,1],2,2)
  unsupported('i64-offset-gapped',[12],[2,3],[1,3],2,2)
  unsupported('i64-offset-overlap',[10],[2,3],[1,1],2,2)
  unsupported('i64-offset-zero-stride',[10],[2,3],[0,1],2,2)"""))
edit('pure_readback.py',("width={0:4,1:1}[dtype]","width={0:4,1:1,2:8}[dtype]"),
("('offset-same-storage-','bool-offset-partially-overlapping-')","('offset-same-storage-','bool-offset-partially-overlapping-','i64-offset-partially-overlapping-')"),
("""    if x['case'].startswith('bool-offset-partially-overlapping-'):
     assert (difference,src[72])==(1,1) and src[1]==dst[1]>difference
     counts['bool_partial_offset_alias_rollbacks']=counts.get('bool_partial_offset_alias_rollbacks',0)+1""","""    if 'partially-overlapping' in x['case']:
     dtype=src[72];width={1:1,2:8}[dtype]
     assert difference==width and src[1]==dst[1]>1
     key={1:'bool',2:'i64'}[dtype]+'_partial_offset_alias_rollbacks'
     counts[key]=counts.get(key,0)+1"""),
("     if src[72]==1:counts['bool_same_start_offset_alias_rollbacks']=counts.get('bool_same_start_offset_alias_rollbacks',0)+1","     if src[72] in (1,2):\n      key={1:'bool',2:'i64'}[src[72]]+'_same_start_offset_alias_rollbacks'; counts[key]=counts.get(key,0)+1"),
("assert vn[72]==sn[72]==1","assert vn[72]==sn[72]==2"),("assert rn[72]==1 and rn[83]==0","assert rn[72]==2 and rn[83]==0"),
("key={0:'f32',1:'bool'}[dtype]","key={0:'f32',1:'bool',2:'i64'}[dtype]"))
edit('retained79_observe.py',(old,new))
