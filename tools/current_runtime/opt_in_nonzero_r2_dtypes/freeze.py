"""Detached stdlib-only rank-2 CPU dtype nonzero predictions."""
from pathlib import Path
import hashlib,json,struct,sys
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
def source(dt,shape,r):
 rows,cols=shape;n=rows*cols
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
  m={(i,j):w for i,j,w in r['triples']}
  assert len(m)==len(r['triples']) and all(0<=i<rows and 0<=j<cols for i,j in m)
  seq=(m.get((i,j),0) for i in range(rows) for j in range(cols))
 else:raise AssertionError(kind)
 return b''.join(w.to_bytes(bits//8,'little') for w in seq)
def pack(ws):
 return b''.join(struct.pack('<q',w) for w in ws)
def make(dt,label,shape,r):
 n0,n1=shape;tag,bits,exp,_=SPECS[dt];sz=bits//8
 b=source(dt,shape,r);assert len(b)==n0*n1*sz
 magnitude=(1<<(bits-int(exp>0)))-1
 chosen=[]
 for col in range(n1):
  for row in range(n0):
   off=(row*n1+col)*sz
   word=int.from_bytes(b[off:off+sz],'little')
   if dt=='bool':assert word in (0,1)
   if word&magnitude:chosen.append((row,col))
 chosen.sort()
 k=len(chosen)
 logical=pack(v for rc in chosen for v in rc)
 physical=pack(v for axis in (0,1) for v in [rc[axis] for rc in chosen])
 assert n0*n1<=262144 and 2*k<=262144 and len(logical)==len(physical)==16*k
 expect=dict(dtype='int64',shape=[k,2],strides=[1,k],version=0,offset=0,view=False,
     requires_grad=False,is_contiguous=k<=1,storage_bytes=16*k,
     logical_sha256=sha(logical),physical_sha256=sha(physical),first8=chosen[:8],last8=chosen[-8:])
 if k<25:expect.update(logical_hex=logical.hex(),physical_hex=physical.hex())
 return dict(label=dt+'-'+label,input_dtype=dt,input_tag=tag,input_shape=shape,
    input_strides=[max(n1,1),1],input_count=n0*n1,input_raw_sha256=sha(b),recipe=r,k=k,
    predicted_parent_status_if_ordinary_setup_proven=0 if dt=='bool' else 7,expected_torch=expect)
def rr(words):return dict(kind='words',words=words)
def pp(*triples):return dict(kind='points',triples=triples)
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
  make(dt,'empty-leading',[0,7],rr([])),make(dt,'empty-trailing',[7,0],rr([])),
  make(dt,'empty-both',[0,0],rr([])),make(dt,'normal-shape-all-false',[4,5],Z),
  make(dt,'one-corner-true',[3,7],pp((2,6,high))),
  make(dt,'patterns-irregular-row-boundaries',[5,7],cc(0)),
  make(dt,'patterns-irregular-swapped',[7,5],cc(8)),
  make(dt,'patterns-single-row',[1,29],cc(0)),
  make(dt,'patterns-single-column',[29,1],cc(0)),
  make(dt,'uneven-sparse-order',[17,19],pp((16,18,high),(1,0,other),(0,18,1),(0,2,high),(6,5,other),(2,13,1),(6,3,high))),
 ]
cases += [make('bool','retained-old-witness',[2,3],rr([1,0,1,0,1,0])),
          make('bool','retained-empty-geometry',[7,0],rr([]))]
head,tree,dest=sys.argv[1:4];assert len(sys.argv)==4
assert all(len(x)==40 and all(c in '0123456789abcdef' for c in x) for x in (head,tree))
assert head.startswith('2b5b2fc') and tree.startswith('e4ca3d95')
summ=dict(cases=len(cases),new=sum(c['input_dtype']!='bool' for c in cases),
 retained_bool=sum(c['input_dtype']=='bool' for c in cases),
 words_each_order=sum(2*c['k'] for c in cases),new_words_each_order=sum(2*c['k'] for c in cases if c['input_dtype']!='bool'),
 case_k0=sum(c['k']==0 for c in cases),case_k1=sum(c['k']==1 for c in cases),
 case_distinct_orders=sum(c['expected_torch']['logical_sha256']!=c['expected_torch']['physical_sha256'] for c in cases),
 max_n=max(c['input_count'] for c in cases),max_k=max(c['k'] for c in cases))
out=dict(tier='DETACHED_ORDINARY_STD_LIB_PREDICTIONS',parent_git=head,parent_tree=tree,
 parent_source='8420d6599d484e2b6844576a0f60207d6b318cb21bda269f0d60bf135bac10db',
 parent_elf='ac17c641a0e90a08a79dc913e0a3afd2476797ab40d5a48d06190f3b8fd12d33',
 pinned_torch='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
 summary=summ,cases=cases,notes=[
 'Only usual rank2 owned CPU contiguous zero-offset non-grad valid inputs; no malformed/fault, pressure, out/tuple, other axes, real cap or dispatch-default claims.',
 'Independent IEEE raw word tests: either signed zero false, signed q/sNaN and any subnormal positive/negative true; I64 includes entire signed word. Use exact byte-based input construction, no Python floating cast or FTZ changes.',
 'Future exact shape [K,2], element strides [1,K], contiguous iff K<=1; independently compare lexicographic logical I64 coordinates and column-major physical raw storage, size 16*K, Version0, fresh distinct Tensor/Storage/Version incl K0 (not pointer equality).',
 'No archived non-Bool rank2 refusal identified in the prior Bool qualification: exact current-head predicted status7 for new dtypes only, pending real old ordinary setup. Retained Bool witness previously measured and native successful; no relabeling.',
 'All 50 new small cases fit a literal public ordinary op97 Bool/typed raw frame, so no cast, reshaping, large allocation fallback or other lane required. Cap behavior inherited but not tested here.'
 ])
raw=(json.dumps(out,sort_keys=True,separators=(',',':'))+'\n').encode()
with Path(dest).open('xb') as fp:fp.write(raw)
print(len(raw),sha(raw),summ)
