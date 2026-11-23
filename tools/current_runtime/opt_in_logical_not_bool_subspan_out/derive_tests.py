from pathlib import Path
import sys
root=Path(sys.argv[1])
dst=Path(sys.argv[2])
new='f24a571bebd4f85e5c55dbb2ed55eef4d237bea32f0a3baea3e893a0187b500b'
old='b8558c915c0c1b475bfcbd46e654dda64930a30e18e835d3504e8b3c95851bef'
def edit(file,*pairs):
 s=(root/file).read_text()
 for a,b in pairs:
  assert s.count(a)==1,(file,a,s.count(a));s=s.replace(a,b)
 (dst/file).write_text(s)
edit('oracle_fixture.py',("(D/'official.fixture.json')",'''for j in range(4):
 boundary=j==3
 root_values=[1]*5 if boundary else [1,0,1,1,0,0,1,0,0,1]
 shape,strides,offset=([2,2],[1,2],1) if boundary else ([2,3],[1,2],2)
 values=[0]*4 if boundary else [0 if i>>j&1 else 1 for i in range(6)]
 base=torch.tensor(root_values,dtype=torch.bool)
 dst=base.as_strided(shape,strides,offset);sib=base.as_strided(shape,strides,offset)
 src=torch.tensor(values,dtype=torch.bool).reshape(shape)
 before=data(base);initial=data(src);version=base._version
 ptr=dst.data_ptr(); storage=dst.untyped_storage().data_ptr()
 assert not dst.is_contiguous() and dst.storage_offset()==offset and dst._base is base
 assert torch.logical_not(src,out=dst) is dst and dst.data_ptr()==ptr
 assert dst.untyped_storage().data_ptr()==storage
 assert dst._version==base._version==sib._version==version+1 and data(src)==initial
 assert data(sib)==data(dst)
 rows.append(dict(dtype=1,base_shape=[len(root_values)],shape=shape,strides=strides,offset=offset,
   prior_exact_boundary=boundary,root_values=root_values,base_initial=before,input=values,
   initial=initial,logical=data(dst),base_physical=data(base),version_delta=1))
(D/'official.fixture.json')'''))
edit('native_observe.py',('dc00ed3e71ac10dc1550a9f331870785a084a7c27971e968d4ca61c68bf9d6ed',old),("'candidate':'"+old,"'candidate':'"+new),
("if mode=='baseline' and not offset:continue","if mode=='baseline' and not (offset and dtype==1):continue"),
("  if offset:negative('f32-offset-same-storage-'+str(j),[(128,sib,dst)])","""  if offset:negative('offset-same-storage-'+str(j),[(128,sib,dst)])
  if offset and dtype==1 and not row['prior_exact_boundary']:
   st,r=shape_view(root,row['shape'],row['strides'],offset+1)
   assert st==0 and r[0]['opcode']==111,(st,r)
   partial=r[0]['value'];assert partial not in (sib,dst)
   assert inspect(partial)['meta']['storage_id']==out_meta['storage_id']
   negative('bool-offset-partially-overlapping-'+str(j),[(128,partial,dst)])"""),
("  unsupported('bool-nonzero-offset',[5],[2,2],[1,2],1,1)","""  unsupported('bool-offset-contiguous',[10],[2,3],[3,1],2,1)
  unsupported('bool-offset-gapped',[12],[2,3],[1,3],2,1)
  unsupported('bool-offset-overlap',[10],[2,3],[1,1],2,1)
  unsupported('bool-offset-zero-stride',[10],[2,3],[0,1],2,1)"""))
edit('pure_readback.py',("  elems=f['offset'];span=4*len(f['input'])\n  assert r[72]==v[72]==0","  elems=f['offset'];dtype=f['dtype'];width={0:4,1:1}[dtype];span=width*len(f['input'])\n  assert r[72]==v[72]==dtype"),
("v[2]-r[2]==4*elems","v[2]-r[2]==width*elems"),("span==4*(1+sum","span==width*(1+sum"),
("live==kr[5]==4*r[1]==4*len(f['root_values']) and 0<4*elems<=live-span and kv[5]==live-4*elems","live==kr[5]==width*r[1]==width*len(f['root_values']) and 0<width*elems<=live-span and kv[5]==live-width*elems"),
("assert vn[72]==sn[72]==0","assert vn[72]==sn[72]==1"),("assert rn[72]==0 and rn[83]==0","assert rn[72]==1 and rn[83]==0"),
("   key='prior_end_boundary_successes' if f.get('prior_exact_boundary') else 'f32_offset_mask_successes'","   key={0:'f32',1:'bool'}[dtype]+('_end_boundary_successes' if f.get('prior_exact_boundary') else '_offset_mask_successes')"),
("   if r['mode']=='baseline':","""   if x['case'].startswith(('offset-same-storage-','bool-offset-partially-overlapping-')):
    c,=qs[idx];assert c[0]==128 and c[1]!=c[2]
    a=Owner(raw(x['pre']));src=a.node(c[1]);dst=a.node(c[2])
    assert src[27:29]==dst[27:29] and src[83]==dst[83]==1 and src[72]==dst[72] and src[29]>0 and dst[29]>0
    difference=src[29]-dst[29]
    if x['case'].startswith('bool-offset-partially-overlapping-'):
     assert (difference,src[72])==(1,1) and src[1]==dst[1]>difference
     counts['bool_partial_offset_alias_rollbacks']=counts.get('bool_partial_offset_alias_rollbacks',0)+1
    else:
     assert difference==0
     if src[72]==1:counts['bool_same_start_offset_alias_rollbacks']=counts.get('bool_same_start_offset_alias_rollbacks',0)+1
   if r['mode']=='baseline':"""))
edit('retained79_observe.py',(old,new))
