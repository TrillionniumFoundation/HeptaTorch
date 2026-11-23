from pathlib import Path
import sys
root,ctor,dst=map(Path,sys.argv[1:])
new='87e7d880848f363374d020536d16b5a67ed17a60b22189b23bf0eb9e782a2c11'
old='d75b5e51dbef318a70d233e1c68f50bd2cc22ce7eb0f111c3a730ecaa91dafdb'
old_layout='b680ef8390e5438240733d8c20a55931b3e059f41b40f8f86db08e153021609e'
def edit(file,*pairs,fromdir=None):
 s=((fromdir or root)/file).read_text()
 for a,b in pairs:
  assert s.count(a)==1,(file,a,s.count(a));s=s.replace(a,b)
 (dst/file).write_text(s)
edit('oracle_fixture.py',("(D/'official.fixture.json')",'''def bf16_raw_tensor(words):
 return torch.tensor([w if w<1<<15 else w-(1<<16) for w in words],dtype=torch.int16).view(torch.bfloat16)
special=[0,0x8000,1,0x8001,0x7f80,0xff80,0x7fc5,0x7f81,0xffd2,0xff93,0x3f80,0xbf80,
         0x3f81,0xbf02,0x0080,0x8080,0x007f,0x807f,0x7f7f,0xff7f,0x4000,0x407d,0x3eab,0x4123]
for index,(bs,shape,strides) in enumerate(([ [3,2],[2,3],[1,2] ],[ [2,3,4],[3,4,2],[4,1,12] ],[ [3,2],[2,3],[1,2] ])):
 n=__import__('math').prod(bs)
 root_values=special[(index+1)*3:]+special[:(index+1)*3]
 root_values=root_values[:n]
 values=[[0,0x8000,1,0xff80,0x7f81,0xffd2],special,[0x8001,0x7fc5,0,0x7f80,0x8000,0xff7f]][index]
 base=bf16_raw_tensor(root_values)
 dst=base.as_strided(shape,strides,0);sib=base.as_strided(shape,strides,0)
 src=bf16_raw_tensor(values).reshape(shape)
 before=data(base);initial=data(src);version=base._version
 assert before==b''.join(v.to_bytes(2,'little') for v in root_values).hex()
 assert initial==b''.join(v.to_bytes(2,'little') for v in values).hex()
 ptr=dst.data_ptr();storage=dst.untyped_storage().data_ptr()
 assert not dst.is_contiguous() and dst.storage_offset()==0 and dst._base is base
 assert torch.logical_not(src,out=dst) is dst and dst.data_ptr()==ptr
 assert dst.untyped_storage().data_ptr()==storage
 assert dst._version==base._version==sib._version==version+1 and data(src)==initial
 assert data(sib)==data(dst)
 rows.append(dict(dtype=6,base_shape=bs,shape=shape,strides=strides,root_values=root_values,base_initial=before,
    input=values,initial=initial,logical=data(dst),base_physical=data(base),version_delta=1))
(D/'official.fixture.json')'''))
edit('native_observe.py',('2196cd3811604609925ba6da965b105f35cd0336e3ac6f7f7b7fab44df745ebb',old),("'candidate':'"+old_layout,"'candidate':'"+new),
("16 if dtype==5 else 64","16 if dtype in (5,6) else 64"),
("if mode=='baseline' and dtype!=5:continue","if mode=='baseline' and dtype!=6:continue"),
("if dtype in (1,2,4,5) or offset:","if dtype in (1,2,4,5,6) or offset:"),
("offset*{0:4,1:1,2:8,4:8,5:2}[dtype]","offset*{0:4,1:1,2:8,4:8,5:2,6:2}[dtype]"),
("""  if dtype==5:
   negative('f16-distinct-same-storage-'+str(j),[(128,sib,dst)])
   st,r=shape_view(source,row['shape'],row['strides'],0); assert st==0 and r[0]['opcode']==111
   negative('f16-disjoint-view-source-'+str(j),[(128,r[0]['value'],dst)])
   wrong=make([len(row['input'])],[0]*len(row['input']),5)
   negative('f16-equal-count-shape-mismatch-'+str(j),[(128,wrong,dst)])""","""  if dtype in (5,6):
   half='f16' if dtype==5 else 'bf16'
   negative(half+'-distinct-same-storage-'+str(j),[(128,sib,dst)])
   st,r=shape_view(source,row['shape'],row['strides'],0); assert st==0 and r[0]['opcode']==111
   negative(half+'-disjoint-view-source-'+str(j),[(128,r[0]['value'],dst)])
   wrong=make([len(row['input'])],[0]*len(row['input']),dtype)
   negative(half+'-equal-count-shape-mismatch-'+str(j),[(128,wrong,dst)])"""),
("  if dtype==5:\n   weak=","  if dtype in (5,6):\n   half='f16' if dtype==5 else 'bf16'\n   weak="),
("negative('f16-written-last-view-weak-expired-'+str(j)","negative(half+'-written-last-view-weak-expired-'+str(j)"),
("dict(kind='f16_lifetime',case=j","dict(kind=half+'_lifetime',case=j"),
("  unsupported('zero-offset-retained-contiguous-f16',[6],[2,3],[3,1],0,5)","""  unsupported('zero-offset-retained-contiguous-f16',[6],[2,3],[3,1],0,5)
  unsupported('zero-offset-retained-contiguous-bf16',[6],[2,3],[3,1],0,6)"""))
edit('pure_readback.py',
("if x['kind']=='f16_lifetime':","if x['kind'] in ('f16_lifetime','bf16_lifetime'):"),
("   counts['f16_written_lifetimes']=counts.get('f16_written_lifetimes',0)+1","   half={5:'f16',6:'bf16'}[f['dtype']]; assert x['kind']==half+'_lifetime'; key=half+'_written_lifetimes'; counts[key]=counts.get(key,0)+1"),
("'zero-offset-retained-contiguous-f16':(5,0)}","'zero-offset-retained-contiguous-f16':(5,0),'zero-offset-retained-contiguous-bf16':(6,0)}"),
("assert vn[72]==sn[72]==5","assert vn[72]==sn[72]==6"),
("dtype=f['dtype'];assert dtype in (0,1,2,4,5)","dtype=f['dtype'];assert dtype in (0,1,2,4,5,6)"),
("width={0:4,1:1,2:8,4:8,5:2}[dtype]\n  if dtype in (1,2,4,5) or offset:","width={0:4,1:1,2:8,4:8,5:2,6:2}[dtype]\n  if dtype in (1,2,4,5,6) or offset:"),
("(0x3c00 if v&0x7fff==0 else 0).to_bytes(2,'little') if dtype==5","(({5:0x3c00,6:0x3f80}[dtype]) if v&0x7fff==0 else 0).to_bytes(2,'little') if dtype in (5,6)"),
("   if dtype==5:\n    assert old","   if dtype in (5,6):\n    assert old"),
("5:'f16_successes'}[dtype]","5:'f16_successes',6:'bf16_successes'}[dtype]"))
edit('retained79_observe.py',(old_layout,new))
edit('native_bf16_view.py',("'candidate':'"+old,"'candidate':'"+new),
("  refuse('new-bf16-view-is-still-not-out-covered-'+row['label'],7,(128,source,v))","""  if row['label']=='rank2-row-major':
   refuse('new-bf16-view-is-still-not-out-covered-'+row['label'],7,(128,source,v))
  else:
   refuse('accepted-bf16-permuted-out-followed-by-invalid-'+row['label'],1,(128,source,v),(128,0,v))"""),fromdir=ctor)
edit('pure_bf16_view.py',("   if x['case']=='f16-still-covered':","   if x['case']=='f16-still-covered':") if False else ("   if x['case'].startswith('valid-view-followed-by-stale-view-'):","""   if x['case'].startswith('accepted-bf16-permuted-out-followed-by-invalid-'):
    a,b=qs[i];assert a[0]==b[0]==128 and b[1]==0 and a[2]==b[2] and rs[i]==(1,[])
    oo=Owner(before)
    assert oo.node(a[1])[72]==oo.node(a[2])[72]==6 and oo.node(a[2])[83]==1
    totals['scoped_out_then_invalid_rollbacks']=totals.get('scoped_out_then_invalid_rollbacks',0)+1
   if x['case'].startswith('valid-view-followed-by-stale-view-'):"""),fromdir=ctor)
