from pathlib import Path
import sys
root,ctor,dst=map(Path,sys.argv[1:])
new='b680ef8390e5438240733d8c20a55931b3e059f41b40f8f86db08e153021609e'
old='2196cd3811604609925ba6da965b105f35cd0336e3ac6f7f7b7fab44df745ebb'
old_layout='6c8e5d338f63f7cf2cd4437b1f49294e635d6f07e9927b4b1f0ccdd11d8a1095'
def edit(file,*pairs,fromdir=None):
 s=((fromdir or root)/file).read_text()
 for a,b in pairs:
  assert s.count(a)==1,(file,a,s.count(a));s=s.replace(a,b)
 (dst/file).write_text(s)
edit('oracle_fixture.py',("(D/'official.fixture.json')",'''def f16_raw_tensor(words):
 return torch.tensor([w if w<1<<15 else w-(1<<16) for w in words],dtype=torch.int16).view(torch.float16)
special=[0,0x8000,1,0x8001,0x7c00,0xfc00,0x7e55,0x7d41,0xfe32,0xfd04,0x3c00,0xbc00,
         0x3c01,0xb802,0x0400,0x8400,0x03ff,0x83ff,0x7bff,0xfbff,0x4000,0x4401,0x3555,0x481b]
for index,(bs,shape,strides) in enumerate(([ [3,2],[2,3],[1,2] ],[ [2,3,4],[3,4,2],[4,1,12] ],[ [3,2],[2,3],[1,2] ])):
 n=__import__('math').prod(bs)
 root_values=special[(index+1)*3:]+special[:(index+1)*3]
 root_values=root_values[:n]
 values=[[0,0x8000,1,0xfc00,0x7d41,0xfe32],special,[0x8001,0x7e55,0,0x7c00,0x8000,0xfbff]][index]
 base=f16_raw_tensor(root_values)
 dst=base.as_strided(shape,strides,0);sib=base.as_strided(shape,strides,0)
 src=f16_raw_tensor(values).reshape(shape)
 before=data(base);initial=data(src);version=base._version
 assert before==b''.join(v.to_bytes(2,'little') for v in root_values).hex()
 assert initial==b''.join(v.to_bytes(2,'little') for v in values).hex()
 ptr=dst.data_ptr();storage=dst.untyped_storage().data_ptr()
 assert not dst.is_contiguous() and dst.storage_offset()==0 and dst._base is base
 assert torch.logical_not(src,out=dst) is dst and dst.data_ptr()==ptr
 assert dst.untyped_storage().data_ptr()==storage
 assert dst._version==base._version==sib._version==version+1 and data(src)==initial
 assert data(sib)==data(dst)
 rows.append(dict(dtype=5,base_shape=bs,shape=shape,strides=strides,root_values=root_values,base_initial=before,
    input=values,initial=initial,logical=data(dst),base_physical=data(base),version_delta=1))
(D/'official.fixture.json')'''))
edit('native_observe.py',("[v&((1<<64)-1) for v in vals]","[v&((1<<(16 if dtype==5 else 64))-1) for v in vals]"),('48d7313c1fd8aa5d687db8a9eb5a62dea765f3cc2d641a37eec89efb2098df37',old),("'candidate':'"+old_layout,"'candidate':'"+new),
("if mode=='baseline' and not (row.get('row_contiguous') and dtype in (1,2,4)):continue","if mode=='baseline' and dtype!=5:continue"),
("if dtype in (1,2,4) or offset:","if dtype in (1,2,4,5) or offset:"),
("offset*{0:4,1:1,2:8,4:8}[dtype]","offset*{0:4,1:1,2:8,4:8,5:2}[dtype]"),
("  negative('later-invalid-'+str(j),[(128,source,dst),(128,0,dst)],1)","""  if dtype==5:
   negative('f16-distinct-same-storage-'+str(j),[(128,sib,dst)])
   st,r=shape_view(source,row['shape'],row['strides'],0); assert st==0 and r[0]['opcode']==111
   negative('f16-disjoint-view-source-'+str(j),[(128,r[0]['value'],dst)])
   wrong=make([len(row['input'])],[0]*len(row['input']),5)
   negative('f16-equal-count-shape-mismatch-'+str(j),[(128,wrong,dst)])
  negative('later-invalid-'+str(j),[(128,source,dst),(128,0,dst)],1)"""),
("  rows.append(dict(kind='positive',case=j,construct=(construct_dst,construct_sib),frame=frame,before=before,after=after,src=source,root=root,dst=dst,sib=sib,fixture=row,old=old,new=now))","""  rows.append(dict(kind='positive',case=j,construct=(construct_dst,construct_sib),frame=frame,before=before,after=after,src=source,root=root,dst=dst,sib=sib,fixture=row,old=old,new=now))
  if dtype==5:
   weak=run((42,root))[0]['value']; weak_frame=s.frames
   first_handle,second_handle,survivor,expected=(dst,root,root,row['base_physical']) if len(row['shape'])==3 else (root,dst,dst,row['logical'])
   run((41,first_handle)); drop_first=s.frames
   assert run((98,survivor,0,len(row['input'])))[0]['payload'].hex()==expected
   first_read=s.frames
   run((41,second_handle)); drop_second=s.frames
   hold=run((43,weak))[0]['value']; hold_frame=s.frames; run((45,hold))
   assert run((98,sib,0,len(row['input'])))[0]['payload'].hex()==row['logical']
   sibling_read=s.frames
   run((41,sib)); drop_third=s.frames
   negative('f16-written-last-view-weak-expired-'+str(j),[(43,weak),(128,source,dst)],6)
   run((44,weak))
   rows.append(dict(kind='f16_lifetime',case=j,weak_frame=weak_frame,weak=weak,hold=hold,hold_frame=hold_frame,drop=(drop_first,drop_second,drop_third),first_read=first_read,first_handle=first_handle,second_handle=second_handle,survivor=survivor,sibling_read=sibling_read,root=root,dst=dst,sib=sib))"""),
("  unsupported('zero-offset-retained-contiguous-f32',[6],[2,3],[3,1],0)","""  unsupported('zero-offset-retained-contiguous-f32',[6],[2,3],[3,1],0)
  unsupported('zero-offset-retained-contiguous-f16',[6],[2,3],[3,1],0,5)"""))
edit('pure_readback.py',
("  if x['kind']=='setup_refusal':","""  if x['kind']=='f16_lifetime':
   f=official['rows'][x['case']]; n=len(f['input'])
   wf=x['weak_frame']-1; assert qs[wf][0][:2]==(42,x['root']) and rs[wf]==(0,[(42,x['weak'],b'')])
   first,second,last=x['drop']; a,b,c=[v-1 for v in (first,second,last)]
   assert wf<a<b<c
   for idx,handle in ((a,x['first_handle']),(b,x['second_handle']),(c,x['sib'])):
    assert qs[idx][0][:2]==(41,handle) and rs[idx]==(0,[(41,1,b'')])
   fr=x['first_read']-1; sr=x['sibling_read']-1; hf=x['hold_frame']-1
   assert a<fr<b<hf<sr<c
   assert qs[hf][0][:2]==(43,x['weak']) and rs[hf]==(0,[(43,x['hold'],b'')])
   raw_first=f['base_physical'] if x['survivor']==x['root'] else f['logical']
   assert qs[fr][0][:4]==(98,x['survivor'],0,n) and rs[fr]==(0,[(98,n,bytes.fromhex(raw_first))])
   assert qs[sr][0][:4]==(98,x['sib'],0,n) and rs[sr]==(0,[(98,n,bytes.fromhex(f['logical']))])
   assert qs[c+1][0][:2]==(43,x['weak']) and rs[c+1]==(6,[])
   counts['f16_written_lifetimes']=counts.get('f16_written_lifetimes',0)+1
   continue
  if x['kind']=='setup_refusal':"""),
("    'zero-offset-retained-contiguous-i64':(2,0),'zero-offset-retained-contiguous-f64':(4,0)}","    'zero-offset-retained-contiguous-i64':(2,0),'zero-offset-retained-contiguous-f64':(4,0),'zero-offset-retained-contiguous-f16':(5,0)}"),
("assert vn[72]==sn[72] and vn[72] in (1,2,4)","assert vn[72]==sn[72]==5"),
("direct_config(i,root,dst,f);validate_span(a,root,dst,f)","direct_config(i,root,dst,f);assert a.row(1,dst)[1][4:6]==(0,2*len(f['input']))"),
("dtype=f['dtype'];assert dtype in (0,1,2,4)","dtype=f['dtype'];assert dtype in (0,1,2,4,5)"),
("width={0:4,1:1,2:8,4:8}[dtype]\n  if dtype in (1,2,4) or offset:","width={0:4,1:1,2:8,4:8,5:2}[dtype]\n  if dtype in (1,2,4,5) or offset:"),
("if dtype in (1,2) else struct.pack('<I',0x3f800000 if v==0 else 0)","if dtype in (1,2) else ((0x3c00 if v&0x7fff==0 else 0).to_bytes(2,'little') if dtype==5 else struct.pack('<I',0x3f800000 if v==0 else 0))"),
("   assert len(old)>=width*n and old[:width*n]==","   if dtype==5:\n    assert old[:width*n].hex()==f['base_initial']\n    assert a.row(1,h)[1][4:6]==a.row(1,x['sib'])[1][4:6]==(0,width*n)\n   else:assert len(old)>=width*n and old[:width*n]=="),
("key={0:'f32_successes',1:'bool_successes',2:'i64_successes',4:'f64_successes'}[dtype]","key={0:'f32_successes',1:'bool_successes',2:'i64_successes',4:'f64_successes',5:'f16_successes'}[dtype]"))
edit('retained79_observe.py',(old_layout,new))
edit('native_f16_view.py',("'candidate':'"+old,"'candidate':'"+new),
("  refuse('new-f16-view-is-still-not-out-covered-'+row['label'],7,(128,source,v))","""  if row['label']=='rank2-row-major':
   refuse('new-f16-view-is-still-not-out-covered-'+row['label'],7,(128,source,v))
  else:
   refuse('accepted-permuted-out-followed-by-invalid-'+row['label'],1,(128,source,v),(128,0,v))"""),fromdir=ctor)
edit('pure_f16_view.py',("   if x['case']=='bf16-still-denied':","""   if x['case'].startswith('accepted-permuted-out-followed-by-invalid-'):
    a,b=qs[i];assert a[0]==b[0]==128 and b[1]==0 and a[2]==b[2] and rs[i]==(1,[])
    oo=Owner(before)
    assert oo.node(a[1])[72]==oo.node(a[2])[72]==5 and oo.node(a[2])[83]==1
    totals['scoped_out_then_invalid_rollbacks']=totals.get('scoped_out_then_invalid_rollbacks',0)+1
   if x['case']=='bf16-still-denied':"""),fromdir=ctor)
