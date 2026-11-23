"""New-layout native observations. Never imports Torch."""
from pathlib import Path
import hashlib,json,struct,sys,zlib
D=Path(__file__).resolve().parent
H,protocol,image,out,mode=map(Path,sys.argv[1:]); mode=mode.name
for n,sha in [('test_runtime_candidate.py','40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'),
              ('test_component.py','97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c')]:
 assert hashlib.sha256((H/n).read_bytes()).hexdigest()==sha
sys.path.insert(0,str(H))
from test_runtime_candidate import factory, load_wire
assert 'torch' not in sys.modules
image_pin={'baseline':'f24a571bebd4f85e5c55dbb2ed55eef4d237bea32f0a3baea3e893a0187b500b','candidate':'7ebb937c818ce8ab36791b94a829a3a2a4aad1c3599eb80c80d38789c72149d9'}[mode]
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(image.read_bytes())==image_pin
fixture_raw=(D/'official.fixture.json').read_bytes();fixtures=json.loads(fixture_raw)
w=load_wire(protocol);s=w.Session(image,directory=out);(out/'owners').mkdir()
rows=[]
def packet(*commands):return w.parse(s.exchange(w.packet(commands)))
def run(*commands):
 status,rr=packet(*commands)
 assert not status and [r['opcode'] for r in rr]==[x[0] for x in commands],(status,commands,rr)
 return rr
def make(shape,vals,dtype=0):
 words=[int.from_bytes(struct.pack('<f',v),'little') for v in vals] if dtype==0 else ([int(bool(v)) for v in vals] if dtype==1 else [v&((1<<64)-1) for v in vals])
 st,r=w.parse(s.exchange(b''.join(factory(dtype,shape,words))));assert st==0 and r[0]['opcode']==97
 return r[0]['value']
def snap():
 r=s.owner();raw=r['bytes'];key=sha(raw)
 p=out/'owners'/(key+'.zlib')
 if p.exists():assert zlib.decompress(p.read_bytes())==raw
 else:p.write_bytes(zlib.compress(raw,6))
 return dict(sha256=key,length=len(raw),ptr=r['pointer'])
def inspect(h):
 rr=run((13,h),(96,h,0));meta=w.meta(rr[0]);n=meta['numel'];res=run((98,h,0,n))[0]
 return dict(meta=meta,owner=rr[1]['payload'].hex(),raw=res['payload'].hex())
def negative(name,commands,wanted=7):
 pre=snap();st,res=packet(*commands);f=s.frames;post=snap()
 rows.append(dict(case=name,kind='refusal',frame=f,status=st,expected_status=wanted,pre=pre,post=post))
 assert st==wanted and not res and pre==post,(name,st,res,pre,post)
def shape_view(h,shape,strides,offset):
 conf=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),offset)
 return packet((111,h,w.Payload(conf),80))
def layout(root,perm):
 cur=list(range(len(perm)));out=root
 for i,want in enumerate(perm):
  j=cur.index(want)
  if i!=j:
   out=run((6,out,i,j))[0]['value'];cur[i],cur[j]=cur[j],cur[i]
 return out
def ensure_version(old,after,raw):
 target=dict(old['meta']);target['version']+=1
 assert after==dict(meta=target,owner=old['owner'],raw=raw),(old,after,target,raw)
try:
 g=run((60,1729))[0]['value'];rng=run((63,g))[0]['payload']
 for j,row in enumerate(fixtures['rows']):
  dtype=row['dtype'];offset=row.get('offset',0);rn=__import__('math').prod(row['base_shape'])
  if mode=='baseline' and not (offset and dtype==2):continue
  source=make(row['shape'],row['input'],dtype);root=make(row['base_shape'],row.get('root_values',[0xc000000000000000 if dtype==4 else -2]*rn),dtype)
  if dtype in (1,2,4) or offset:
   st,r=shape_view(root,row['shape'],row['strides'],offset)
   assert st==0 and r[0]['opcode']==111,(st,r)
   dst=r[0]['value'];construct_dst=s.frames
   st,r=shape_view(root,row['shape'],row['strides'],offset)
   assert st==0 and r[0]['opcode']==111,(st,r)
   sib=r[0]['value'];construct_sib=s.frames
  else:dst=layout(root,row['permutation']);sib=layout(root,row['permutation']);construct_dst=construct_sib=None
  old={n:inspect(h) for n,h in [('source',source),('root',root),('dst',dst),('sib',sib)]}
  root_meta=old['root']['meta'];out_meta=old['dst']['meta'];src_meta=old['source']['meta']
  assert tuple(out_meta['strides'][:len(row['strides'])])==tuple(row['strides']) and out_meta['storage_offset']==offset*{0:4,1:1,2:8,4:8}[dtype]
  for k in ('storage_id','version_id'):assert out_meta[k]==root_meta[k]==old['sib']['meta'][k] and src_meta[k]!=out_meta[k]
  assert old['source']['raw']==row['initial']
  if mode=='baseline':
   negative('old-permutation-'+str(j),[(128,source,dst)])
   continue
  if offset:negative('offset-same-storage-'+str(j),[(128,sib,dst)])
  if offset and dtype in (1,2) and not row['prior_exact_boundary']:
   st,r=shape_view(root,row['shape'],row['strides'],offset+1)
   assert st==0 and r[0]['opcode']==111,(st,r)
   partial=r[0]['value'];assert partial not in (sib,dst)
   assert inspect(partial)['meta']['storage_id']==out_meta['storage_id']
   negative({1:'bool',2:'i64'}[dtype]+'-offset-partially-overlapping-'+str(j),[(128,partial,dst)])
  negative('later-invalid-'+str(j),[(128,source,dst),(128,0,dst)],1)
  before=snap()
  ops=[(128,source,dst),(98,w.reference(1),0,len(row['input'])),(98,root,0,rn),(98,sib,0,len(row['input']))]
  rr=run(*ops); frame=s.frames; after=snap()
  assert rr[0]['value']==dst and rr[0]['payload']==b''
  assert [v['payload'].hex() for v in rr[1:]]==[row['logical'],row['base_physical'],row['logical']]
  now={n:inspect(h) for n,h in [('source',source),('root',root),('dst',dst),('sib',sib)]}
  assert now['source']==old['source']
  for n in ('root','dst','sib'):ensure_version(old[n],now[n],row['base_physical'] if n=='root' else row['logical'])
  rows.append(dict(kind='positive',case=j,construct=(construct_dst,construct_sib),frame=frame,before=before,after=after,src=source,root=root,dst=dst,sib=sib,fixture=row,old=old,new=now))
 if mode=='candidate':
  def unsupported(name,base_shape,shape,strides,offset,dtype=0):
   h=make(base_shape,[-2]*__import__('math').prod(base_shape),dtype);source=make(shape,[0]*__import__('math').prod(shape),dtype)
   st,rr=shape_view(h,shape,strides,offset);f=s.frames
   if st:
    rows.append(dict(kind='setup_refusal',case=name,status=st,frame=f));return
   dst=rr[0]['value'];negative(name,[(128,source,dst)])
  unsupported('positive-repeated-active-strides',[2,2],[2,2],[1,1],0)
  unsupported('expanded-zero-stride',[2,2],[2,2],[0,1],0)
  unsupported('gapped-positive-stride',[5],[2,2],[1,3],0)
  unsupported('nonzero-offset-contiguous',[10],[2,3],[3,1],2)
  unsupported('nonzero-offset-gapped',[12],[2,3],[1,3],2)
  unsupported('nonzero-offset-overlap',[10],[2,3],[1,1],2)
  unsupported('nonzero-offset-zero-stride',[10],[2,3],[0,1],2)
  unsupported('empty-destination-view',[0,2],[2,0],[1,2],0)
  unsupported('bool-positive-repeated-active-strides',[2,2],[2,2],[1,1],0,1)
  unsupported('bool-expanded-zero-stride',[2,2],[2,2],[0,1],0,1)
  unsupported('bool-gapped-positive-stride',[5],[2,2],[1,3],0,1)
  unsupported('bool-offset-contiguous',[10],[2,3],[3,1],2,1)
  unsupported('bool-offset-gapped',[12],[2,3],[1,3],2,1)
  unsupported('bool-offset-overlap',[10],[2,3],[1,1],2,1)
  unsupported('bool-offset-zero-stride',[10],[2,3],[0,1],2,1)
  unsupported('bool-empty-destination-view',[0,2],[2,0],[1,2],0,1)
  br=make([3,2],[1]*6,1)
  def bool_view(h):
   st,r=shape_view(h,[2,3],[1,2],0);assert st==0 and r[0]['opcode']==111
   return r[0]['value']
  bd=bool_view(br);bs=bool_view(br);assert bd!=bs
  neg_src=make([3,2],[0]*6,1);separate=bool_view(neg_src)
  negative('bool-wrong-shape-equal-count',[(128,neg_src,bd)])
  negative('bool-same-storage-through-distinct-alias',[(128,bs,bd)])
  negative('bool-disjoint-source-view',[(128,separate,bd)])
  unsupported('i64-positive-repeated-active-strides',[2,2],[2,2],[1,1],0,2)
  unsupported('i64-expanded-zero-stride',[2,2],[2,2],[0,1],0,2)
  unsupported('i64-gapped-positive-stride',[5],[2,2],[1,3],0,2)
  unsupported('i64-offset-contiguous',[10],[2,3],[3,1],2,2)
  unsupported('i64-offset-gapped',[12],[2,3],[1,3],2,2)
  unsupported('i64-offset-overlap',[10],[2,3],[1,1],2,2)
  unsupported('i64-offset-zero-stride',[10],[2,3],[0,1],2,2)
  unsupported('i64-empty-destination-view',[0,2],[2,0],[1,2],0,2)
  br=make([3,2],[1]*6,2)
  def bool_view(h):
   st,r=shape_view(h,[2,3],[1,2],0);assert st==0 and r[0]['opcode']==111
   return r[0]['value']
  bd=bool_view(br);bs=bool_view(br);assert bd!=bs
  neg_src=make([3,2],[0]*6,2);separate=bool_view(neg_src)
  negative('i64-wrong-shape-equal-count',[(128,neg_src,bd)])
  negative('i64-same-storage-through-distinct-alias',[(128,bs,bd)])
  negative('i64-disjoint-source-view',[(128,separate,bd)])
  unsupported('f64-positive-repeated-active-strides',[2,2],[2,2],[1,1],0,4)
  unsupported('f64-expanded-zero-stride',[2,2],[2,2],[0,1],0,4)
  unsupported('f64-gapped-positive-stride',[5],[2,2],[1,3],0,4)
  unsupported('f64-nonzero-offset',[5],[2,2],[1,2],1,4)
  unsupported('f64-empty-destination-view',[0,2],[2,0],[1,2],0,4)
  br=make([3,2],[0x3ff0000000000000]*6,4)
  def bool_view(h):
   st,r=shape_view(h,[2,3],[1,2],0);assert st==0 and r[0]['opcode']==111
   return r[0]['value']
  bd=bool_view(br);bs=bool_view(br);assert bd!=bs
  neg_src=make([3,2],[0]*6,4);separate=bool_view(neg_src)
  negative('f64-wrong-shape-equal-count',[(128,neg_src,bd)])
  negative('f64-same-storage-through-distinct-alias',[(128,bs,bd)])
  negative('f64-disjoint-source-view',[(128,separate,bd)])
  root=make([3,2],[-2]*6);out_view=layout(root,[1,0]);src=make([2,3],[0]*6)
  wrong=make([3,2],[0]*6)
  negative('wrong-shape-equal-count',[(128,wrong,out_view)])
  sibling=layout(root,[1,0])
  assert sibling!=out_view and inspect(sibling)['meta']['storage_id']==inspect(out_view)['meta']['storage_id']
  negative('same-storage-through-distinct-alias',[(128,sibling,out_view)])
  source_root=make([3,2],[0]*6); source_view=layout(source_root,[1,0])
  negative('separate-source-view',[(128,source_view,out_view)])
  negative('stale-attempt-same-frame-root-toggle',[(90,root,1),(128,src,out_view)])
  st,rr=packet((90,root,1));f=s.frames
  if st:rows.append(dict(kind='setup_refusal',case='root-toggle',frame=f,status=st))
  else:
   m=inspect(out_view)['meta'];row=dict(kind='root_toggle_state',case='root-toggle',view_requires_grad=m['requires_grad'],root_requires_grad=inspect(root)['meta']['requires_grad'])
   rows.append(row);negative('root-grad-after-creation',[(128,src,out_view)])
 assert run((63,g))[0]['payload']==rng
 result=dict(mode=mode,elf=image_pin,fixture_sha256=sha(fixture_raw),rows=rows,native_frames=s.frames,no_torch='torch' not in sys.modules)
except BaseException as e:
 result=dict(mode=mode,elf=image_pin,rows=rows,native_frames=s.frames,error=repr(e))
 raise
finally:
 (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 s.finish()
