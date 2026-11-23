"""Detached stdlib-only rank-3 CPU dtype nonzero predictions."""
from pathlib import Path
import hashlib,json,struct,sys,itertools,math
SPECS={
 'int64':(2,64,0,0),'float32':(0,32,8,23),'float64':(4,64,11,52),
 'float16':(5,16,5,10),'bfloat16':(6,16,8,7),'bool':(1,8,0,0)
}
sha=lambda data:hashlib.sha256(data).hexdigest()
def patterns(dt):
 _,bits,e,f=SPECS[dt]
 sign=1<<(bits-1)
 if dt=='int64':
  return [0,sign,0,sign-1,sign+1,1,0,(1<<bits)-1,1<<32,0,sign+17,0,
          sign+65536,1<<62,0,2,0,sign+(1<<62),0]
 if dt=='bool':return [0,1,0,1,1,0]
 all_exp=((1<<e)-1)<<f
 payload=1<<(f-1)
 normal=1<<f
 one=((1<<(e-1))-1)<<f
 return [0,sign,1,sign|1,normal-1,sign|(normal-1),normal,sign|normal,
   0,one,sign|one,all_exp-1,sign|(all_exp-1),all_exp,sign|all_exp,
   all_exp|payload|3,sign|all_exp|payload|3,all_exp|1,sign|all_exp|1,
   all_exp|payload,sign|all_exp|payload,all_exp|(payload-1),sign|all_exp|(payload-1),sign,0]
def grid(shape):return itertools.product(*(range(d) for d in shape))
def source(dt,shape,r):
 n=math.prod(shape)
 _,bits,e,_=SPECS[dt]
 kind=r['kind']
 if kind=='words':
  seq=r['words'];assert len(seq)==n
 elif kind=='cycle':
  offset=r['phase'];a=patterns(dt)
  seq=(a[(i+offset)%len(a)] for i in range(n))
 elif kind=='signed-zero':
  seq=((1<<(bits-1))*int(e!=0 and (i%2)!=0) for i in range(n))
 elif kind=='points':
  m={tuple(p[:3]):p[3] for p in r['points']}
  assert len(m)==len(r['points']) and all(all(0<=k<d for k,d in zip(p,shape,strict=True)) for p in m)
  seq=(m.get(p,0) for p in grid(shape))
 else:raise AssertionError(kind)
 return b''.join(w.to_bytes(bits//8,'little') for w in seq)
def pack(ws):
 return b''.join(struct.pack('<q',w) for w in ws)
def make(dt,label,shape,r):
 a,b,c=shape;tag,bits,exp,_=SPECS[dt];sz=bits//8
 raw=source(dt,shape,r);assert len(raw)==a*b*c*sz
 bits_mask=(1<<(bits-int(exp>0)))-1
 chosen=[]
 # Enumerate in deliberately non-rowmajor order, then explicitly sort coordinate tuples.
 for inner in range(c):
  for outer in range(a):
   for middle in range(b):
    off=((outer*b+middle)*c+inner)*sz
    word=int.from_bytes(raw[off:off+sz],'little')
    if dt=='bool':assert word in (0,1)
    if word&bits_mask:chosen.append((outer,middle,inner))
 chosen.sort()
 k=len(chosen)
 logical=pack(v for coords in chosen for v in coords)
 columns=[pack(p[axis] for p in chosen) for axis in range(3)]
 physical=b''.join(columns)
 assert a*b*c<=262144 and 3*k<=262144 and len(logical)==len(physical)==24*k
 expect=dict(dtype='int64',shape=[k,3],strides=[1,k],version=0,offset=0,view=False,
     requires_grad=False,is_contiguous=k<=1,storage_bytes=24*k,
     logical_sha256=sha(logical),physical_sha256=sha(physical),column_sha256=list(map(sha,columns)),
     first8=chosen[:8],last8=chosen[-8:])
 if k<25:expect.update(logical_hex=logical.hex(),physical_hex=physical.hex())
 return dict(label=dt+'-'+label,input_dtype=dt,input_tag=tag,input_shape=shape,
    input_strides=[max(b,1)*max(c,1),max(c,1),1],input_count=a*b*c,input_raw_sha256=sha(raw),recipe=r,k=k,
    input_must_remain_byte_and_metadata_identical=True,
    predicted_parent_status_if_ordinary_setup_proven=0 if dt=='bool' else 7,expected_torch=expect)
def rr(words):return dict(kind='words',words=words)
def pp(*points):return dict(kind='points',points=points)
def cc(phase=0):return dict(kind='cycle',phase=phase)
Z=dict(kind='signed-zero')
cases=[]
for dt,(tag,bits,e,f) in SPECS.items():
 if dt=='bool':continue
 if e:
  high=(1<<(bits-1))|(((1<<e)-1)<<f)|1
  other=(((1<<e)-1)<<f)|(1<<(f-1))|1
 else:high=1<<(bits-1);other=(1<<bits)-1
 cases+= [
  make(dt,'empty-leading',[0,3,7],rr([])),make(dt,'empty-middle',[3,0,7],rr([])),
  make(dt,'empty-trailing',[3,7,0],rr([])),make(dt,'empty-two-axes',[0,7,0],rr([])),
  make(dt,'ordinary-all-false',[3,4,5],Z),make(dt,'one-corner-true',[3,4,5],pp((2,3,4,high))),
  make(dt,'patterns-asymmetric',[3,4,7],cc(0)),
  make(dt,'patterns-rotated-axes',[7,3,4],cc(8)),
  make(dt,'patterns-singleton-first',[1,5,7],cc(0)),
  make(dt,'patterns-singleton-middle',[5,1,7],cc(0)),
  make(dt,'patterns-singleton-last',[5,7,1],cc(0)),
  make(dt,'sparse-asymmetric-order',[5,6,9],pp((4,5,8,high),(1,0,3,other),(0,5,8,1),(0,2,1,high),
             (3,5,2,other),(2,1,3,1),(1,3,8,high),(4,0,1,other))),
 ]
cases += [make('bool','retained-one-corner',[4,3,5],pp((3,2,4,1))),
          make('bool','retained-empty-geometry',[3,0,7],rr([])),
          make('bool','retained-asymmetric',[3,4,7],pp((2,3,6,1),(1,3,1,1),(0,2,2,1),(0,0,5,1),
                (0,1,0,1),(2,1,4,1),(1,0,3,1)))]
head,tree,dest,prior_path=sys.argv[1:5];assert len(sys.argv)==5
prior=Path(prior_path).read_bytes()
assert sha(prior)=='0463be693db91a23d5cc6a13f53affa99f01a8abb8fdf357d8abdc23e295d161'
by_label={r['label']:r for r in json.loads(prior)['cases']}
retained_mapping={
 'bool-retained-one-corner':'one-corner-only','bool-retained-empty-geometry':'zero-second-axis',
 'bool-retained-asymmetric':'asym-columns'}
for r in cases:
 if r['label'] not in retained_mapping:continue
 name=retained_mapping[r['label']]
 old=by_label[name];assert old['input_shape']==r['input_shape'] and old['input_sha256']==r['input_raw_sha256']
 before=old['expected_torch'];after=r['expected_torch']
 assert all(after[k]==before[k] for k in ('shape','strides','version','storage_bytes','logical_sha256','physical_sha256','column_sha256'))
 r['retained_rank3_bool_fixture']=name
assert all(len(x)==40 and all(c in '0123456789abcdef' for c in x) for x in (head,tree))

summ=dict(cases=len(cases),new=sum(c['input_dtype']!='bool' for c in cases),
 retained_bool=sum(c['input_dtype']=='bool' for c in cases),
 words_each_order=sum(3*c['k'] for c in cases),new_words_each_order=sum(3*c['k'] for c in cases if c['input_dtype']!='bool'),
 case_k0=sum(c['k']==0 for c in cases),case_k1=sum(c['k']==1 for c in cases),
 case_distinct_orders=sum(c['expected_torch']['logical_sha256']!=c['expected_torch']['physical_sha256'] for c in cases),
 max_n=max(c['input_count'] for c in cases),max_k=max(c['k'] for c in cases))
out=dict(tier='DETACHED_ORDINARY_STD_LIB_PREDICTIONS',parent_git=head,parent_tree=tree,
 parent_source='ce78e4be4c06b1b05dc3221d3108ecd4241a1787e8ba6422f17154f6c5b1df87',
 parent_elf='7ebdbbae563577e908e01c8375a6047770db12781eaebbf5255a35fcbe8f1605',
 pinned_torch='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
 retained_rank3_bool_prereg_sha256=sha(prior),summary=summ,cases=cases,notes=[
 'Only usual rank3 owned CPU contiguous zero-offset non-grad valid inputs; no malformed/fault, pressure, out/tuple, other axes, real cap or dispatch-default claims.',
 'Independent IEEE raw word tests: either signed zero false, signed q/sNaN and any subnormal positive/negative true; I64 includes entire signed word. Use exact byte-based input construction, no Python floating cast or FTZ changes.',
 'Future exact shape [K,3], element strides [1,K], contiguous iff K<=1; independently compare lexicographic logical I64 coordinates and column-major physical raw storage plus each of the three independent column digests, size 24*K, Version0, fresh distinct Tensor/Storage/Version incl K0 (not pointer equality).',
 'No archived non-Bool rank3 refusal identified in the prior Bool qualification: exact current-head predicted status7 for new dtypes only, pending real old ordinary setup. Retained Bool witness previously measured and native successful; no relabeling.',
 'All 60 new small cases fit a literal public ordinary op97 Bool/typed raw frame, so no cast, reshaping, large allocation fallback or other lane required. Cap behavior inherited but not tested here.'
 ])
raw=(json.dumps(out,sort_keys=True,separators=(',',':'))+'\n').encode()
with Path(dest).open('xb') as fp:fp.write(raw)
print(len(raw),sha(raw),summ)
