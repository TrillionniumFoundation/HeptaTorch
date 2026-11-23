from pathlib import Path
import sys
root=Path(sys.argv[1]);dst=Path(sys.argv[2])
new='6c8e5d338f63f7cf2cd4437b1f49294e635d6f07e9927b4b1f0ccdd11d8a1095'
old='48d7313c1fd8aa5d687db8a9eb5a62dea765f3cc2d641a37eec89efb2098df37'
def edit(file,*pairs):
 s=(root/file).read_text()
 for a,b in pairs:
  assert s.count(a)==1,(file,a,s.count(a));s=s.replace(a,b)
 (dst/file).write_text(s)
edit('oracle_fixture.py',("(D/'official.fixture.json')",'''for dtype in (1,2,4):
 for j in range(4):
  boundary=j==3
  shape,strides,offset=[2,3],[3,1],4 if boundary else 2
  roots={
   1:[1,0,1,1,0,0,1,0,0,1],
   2:[-(1<<63),(1<<63)-1,-31,57,-121,208,-677,982,-127,1234],
   4:[0x7ff0000000005432,0xfff8000000003333,0x7ff0000000000000,1,0x3ff0000000000000,
      0x8000000000000001,0x8000000000000000,0x7fefffffffffffff,0xfff0000000000000,0]}
  root_values=roots[dtype]
  probes={
   1:[1]*6,
   2:[-(1<<63),(1<<63)-1,-(1<<40),1<<40,-7,2],
   4:[0x7ff0000000001234,0xfff8000000002314,0x8000000000000001,0x7ff0000000000000,0xfff0000000000000,1]}[dtype]
  values=[((0x8000000000000000 if dtype==4 and i&1 else 0) if (not i%2 if boundary else i>>j&1) else probes[i]) for i in range(6)]
  to_tensor=lambda w: f64_raw_tensor(w) if dtype==4 else torch.tensor(w,dtype={1:torch.bool,2:torch.int64}[dtype])
  base=to_tensor(root_values)
  dst=base.as_strided(shape,strides,offset);sib=base.as_strided(shape,strides,offset)
  src=to_tensor(values).reshape(shape)
  before=data(base);initial=data(src);version=base._version
  if dtype==4:
   assert before==b''.join(v.to_bytes(8,'little') for v in root_values).hex()
   assert initial==b''.join(v.to_bytes(8,'little') for v in values).hex()
  ptr=dst.data_ptr(); storage=dst.untyped_storage().data_ptr()
  assert dst.is_contiguous() and dst.storage_offset()==offset and dst._base is base
  assert torch.logical_not(src,out=dst) is dst and dst.data_ptr()==ptr
  assert dst.untyped_storage().data_ptr()==storage
  assert dst._version==base._version==sib._version==version+1 and data(src)==initial
  assert data(sib)==data(dst)
  rows.append(dict(dtype=dtype,row_contiguous=True,base_shape=[len(root_values)],shape=shape,strides=strides,offset=offset,
    prior_exact_boundary=boundary,root_values=root_values,base_initial=before,input=values,
    initial=initial,logical=data(dst),base_physical=data(base),version_delta=1))
(D/'official.fixture.json')'''))
edit('native_observe.py',('d160a44ae8857aab581f24d41f92cb45eb6ec07cc5d90a655a68cda9760c07b2',old),("'candidate':'"+old,"'candidate':'"+new),
("if mode=='baseline' and not row.get('row_contiguous'):continue","if mode=='baseline' and not (row.get('row_contiguous') and dtype in (1,2,4)):continue"),
("negative({0:'f32-contig',1:'bool',2:'i64',4:'f64'}[dtype]+'-offset-partially-overlapping-'+str(j),[(128,partial,dst)])","negative({0:'f32',1:'bool',2:'i64',4:'f64'}[dtype]+('-contig' if row.get('row_contiguous') else '')+'-offset-partially-overlapping-'+str(j),[(128,partial,dst)])"),
("  unsupported('bool-offset-contiguous',[10],[2,3],[3,1],2,1)","  unsupported('zero-offset-retained-contiguous-bool',[6],[2,3],[3,1],0,1)"),
("  unsupported('i64-offset-contiguous',[10],[2,3],[3,1],2,2)","  unsupported('zero-offset-retained-contiguous-i64',[6],[2,3],[3,1],0,2)"),
("  unsupported('f64-offset-contiguous',[10],[2,3],[3,1],2,4)","  unsupported('zero-offset-retained-contiguous-f64',[6],[2,3],[3,1],0,4)"))
edit('pure_readback.py',
("('offset-same-storage-','bool-offset-partially-overlapping-','i64-offset-partially-overlapping-','f64-offset-partially-overlapping-','f32-contig-offset-partially-overlapping-')","('offset-same-storage-','bool-offset-partially-overlapping-','i64-offset-partially-overlapping-','f64-offset-partially-overlapping-','f32-contig-offset-partially-overlapping-','bool-contig-offset-partially-overlapping-','i64-contig-offset-partially-overlapping-','f64-contig-offset-partially-overlapping-')"),
("     key={0:'f32_contig',1:'bool',2:'i64',4:'f64'}[dtype]+'_partial_offset_alias_rollbacks'","     key={0:'f32',1:'bool',2:'i64',4:'f64'}[dtype]+('_contig' if src[16:18]==dst[16:18]==(3,1) else '')+'_partial_offset_alias_rollbacks'"),
("""     if src[72]==0 and src[16:18]==dst[16:18]==(3,1):
      counts['f32_contig_same_start_offset_alias_rollbacks']=counts.get('f32_contig_same_start_offset_alias_rollbacks',0)+1
     if src[72] in (1,2,4):
      key={1:'bool',2:'i64',4:'f64'}[src[72]]+'_same_start_offset_alias_rollbacks'; counts[key]=counts.get(key,0)+1""","""     if src[72] in (1,2,4) or src[16:18]==dst[16:18]==(3,1):
      key={0:'f32',1:'bool',2:'i64',4:'f64'}[src[72]]+('_contig' if src[16:18]==dst[16:18]==(3,1) else '')+'_same_start_offset_alias_rollbacks'
      counts[key]=counts.get(key,0)+1"""),
("""    'zero-offset-retained-contiguous-f32':(0,0),'bool-offset-contiguous':(1,2),
    'i64-offset-contiguous':(2,16),'f64-offset-contiguous':(4,16)}""","""    'zero-offset-retained-contiguous-f32':(0,0),'zero-offset-retained-contiguous-bool':(1,0),
    'zero-offset-retained-contiguous-i64':(2,0),'zero-offset-retained-contiguous-f64':(4,0)}"""),
("assert vn[72]==sn[72]==0","assert vn[72]==sn[72] and vn[72] in (1,2,4)"),
("assert rn[72]==0 and rn[83]==0","assert rn[72]==vn[72] and rn[83]==0"),
("   key=('f32_contig' if f.get('row_contiguous') else {0:'f32',1:'bool',2:'i64',4:'f64'}[dtype])+","   key={0:'f32',1:'bool',2:'i64',4:'f64'}[dtype]+('_contig' if f.get('row_contiguous') else '')+"))
edit('retained79_observe.py',(old,new))
