from pathlib import Path
import sys
root=Path(sys.argv[1]);dst=Path(sys.argv[2])
new='48d7313c1fd8aa5d687db8a9eb5a62dea765f3cc2d641a37eec89efb2098df37'
old='d160a44ae8857aab581f24d41f92cb45eb6ec07cc5d90a655a68cda9760c07b2'
def edit(file,*pairs):
 s=(root/file).read_text()
 for a,b in pairs:
  assert s.count(a)==1,(file,a,s.count(a));s=s.replace(a,b)
 (dst/file).write_text(s)
edit('oracle_fixture.py',("(D/'official.fixture.json')",'''for j in range(4):
 boundary=j==3
 root_values=[-201,-202,203,-204,205,-206,207,-208,209,-210]
 shape,strides,offset=[2,3],[3,1],4 if boundary else 2
 values=[-0.0,7,0.0,-11,-0.0,13] if boundary else [
  ((-0.0 if i&1 else 0.0) if i>>j&1 else 301+i) for i in range(6)]
 base=torch.tensor(root_values,dtype=torch.float32)
 dst=base.as_strided(shape,strides,offset);sib=base.as_strided(shape,strides,offset)
 src=torch.tensor(values,dtype=torch.float32).reshape(shape)
 before=data(base);initial=data(src);version=base._version
 ptr=dst.data_ptr(); storage=dst.untyped_storage().data_ptr()
 assert dst.is_contiguous() and dst.storage_offset()==offset and dst._base is base
 assert torch.logical_not(src,out=dst) is dst and dst.data_ptr()==ptr
 assert dst.untyped_storage().data_ptr()==storage
 assert dst._version==base._version==sib._version==version+1 and data(src)==initial
 assert data(sib)==data(dst)
 rows.append(dict(dtype=0,row_contiguous=True,base_shape=[len(root_values)],shape=shape,strides=strides,offset=offset,
   prior_exact_boundary=boundary,root_values=root_values,base_initial=before,input=values,
   initial=initial,logical=data(dst),base_physical=data(base),version_delta=1))
(D/'official.fixture.json')'''))
edit('native_observe.py',('7ebb937c818ce8ab36791b94a829a3a2a4aad1c3599eb80c80d38789c72149d9',old),("'candidate':'"+old,"'candidate':'"+new),
("if mode=='baseline' and not (offset and dtype==4):continue","if mode=='baseline' and not row.get('row_contiguous'):continue"),
("  if offset and dtype in (1,2,4) and not row['prior_exact_boundary']:","  if offset and (dtype in (1,2,4) or row.get('row_contiguous')) and not row['prior_exact_boundary']:"),
("negative({1:'bool',2:'i64',4:'f64'}[dtype]+'-offset-partially-overlapping-'+str(j),[(128,partial,dst)])","negative({0:'f32-contig',1:'bool',2:'i64',4:'f64'}[dtype]+'-offset-partially-overlapping-'+str(j),[(128,partial,dst)])"),
("  unsupported('nonzero-offset-contiguous',[10],[2,3],[3,1],2)","  unsupported('zero-offset-retained-contiguous-f32',[6],[2,3],[3,1],0)"))
edit('pure_readback.py',
("   if r['mode']=='baseline':","""   bounded_refusal={
    'zero-offset-retained-contiguous-f32':(0,0),'bool-offset-contiguous':(1,2),
    'i64-offset-contiguous':(2,16),'f64-offset-contiguous':(4,16)}
   if x['case'] in bounded_refusal:
    c,=qs[idx];assert c[0]==128 and c[1]!=c[2] and rs[idx]==(7,[])
    a=Owner(raw(x['pre']));src=a.node(c[1]);dst=a.node(c[2])
    dtype,offset=bounded_refusal[x['case']]
    assert src[72]==dst[72]==dtype and src[83]==0 and dst[83]==1
    assert src[27]!=dst[27] and src[28]!=dst[28] and src[29]==0 and dst[29]==offset
    assert src[12:14]==dst[12:14]==(2,3) and src[16:18]==dst[16:18]==(3,1)
    counts['preserved_contiguous_view_refusals']=counts.get('preserved_contiguous_view_refusals',0)+1
   if r['mode']=='baseline':"""),
("('offset-same-storage-','bool-offset-partially-overlapping-','i64-offset-partially-overlapping-','f64-offset-partially-overlapping-')","('offset-same-storage-','bool-offset-partially-overlapping-','i64-offset-partially-overlapping-','f64-offset-partially-overlapping-','f32-contig-offset-partially-overlapping-')"),
("dtype=src[72];width={1:1,2:8,4:8}[dtype]","dtype=src[72];width={0:4,1:1,2:8,4:8}[dtype]"),
("key={1:'bool',2:'i64',4:'f64'}[dtype]","key={0:'f32_contig',1:'bool',2:'i64',4:'f64'}[dtype]"),
("     if src[72] in (1,2,4):","""     if src[72]==0 and src[16:18]==dst[16:18]==(3,1):
      counts['f32_contig_same_start_offset_alias_rollbacks']=counts.get('f32_contig_same_start_offset_alias_rollbacks',0)+1
     if src[72] in (1,2,4):"""),
("assert vn[72]==sn[72]==4","assert vn[72]==sn[72]==0"),("assert rn[72]==4 and rn[83]==0","assert rn[72]==0 and rn[83]==0"),
("   key={0:'f32',1:'bool',2:'i64',4:'f64'}[dtype]+","   key=('f32_contig' if f.get('row_contiguous') else {0:'f32',1:'bool',2:'i64',4:'f64'}[dtype])+"))
edit('retained79_observe.py',(old,new))
