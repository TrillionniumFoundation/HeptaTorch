"""Detached Python stdlib predictions for ordinary CPU rank-3 Bool nonzero."""
import hashlib,itertools,json,math,struct,sys
from pathlib import Path
SHA=lambda b:hashlib.sha256(b).hexdigest()
NMAX=262144;KMAX=NMAX//3
def grid(shape):return itertools.product(*(range(d) for d in shape))
def source(shape,recipe):
 n=math.prod(shape);kind=recipe['kind']
 if kind=='none':return bytes(n)
 if kind=='all':return bytes([1])*n
 if kind=='prefix':
  k=recipe['k'];assert 0<=k<=n
  return bytes([1])*k+bytes(n-k)
 if kind=='points':
  pts={tuple(p) for p in recipe['coords']}
  assert len(pts)==len(recipe['coords']) and all(len(p)==3 and all(0<=a<b for a,b in zip(p,shape,strict=True)) for p in pts)
  return bytes(int(p in pts) for p in grid(shape))
 if kind=='skew':
  return bytes(int((13*a+17*b+23*c+3*a*b+5*b*c+a*c)%37 in (0,3,8,12,29)) for a,b,c in grid(shape))
 raise AssertionError(kind)
def words(vals):
 return b''.join(struct.pack('<q',v) for v in vals)
def make(name,shape,recipe):
 assert len(shape)==3 and all(0<=axis<=NMAX for axis in shape) and math.prod(shape)<=NMAX
 b=source(shape,recipe)
 # zipping an explicit Cartesian enumerator independently determines row lexicographic order.
 coords=[p for p,w in zip(grid(shape),b,strict=True) if w]
 assert all(w in (0,1) for w in b) and len(coords)==sum(b)
 k=len(coords);logical=words(v for p in coords for v in p)
 # Transpose independently before packing backing Storage's contiguous columns.
 physical=words(v for col in zip(*coords) for v in col)
 assert len(logical)==len(physical)==24*k
 by_axis=[SHA(words(p[d] for p in coords)) for d in range(3)]
 exp=dict(shape=[k,3],strides=[1,k],is_contiguous=k<=1,storage_bytes=24*k,
          offset=0,dtype='int64',version=0,requires_grad=False,view=False,
          logical_sha256=SHA(logical),physical_sha256=SHA(physical),column_sha256=by_axis,
          first8=coords[:8],last8=coords[-8:])
 if k<=24:exp.update(logical_hex=logical.hex(),physical_hex=physical.hex())
 _,d1,d2=shape
 return dict(label=name,input_shape=shape,input_strides=[max(d1,1)*max(d2,1),max(d2,1),1],
             input_dtype='bool',input_tag=1,input_count=len(b),input_sha256=SHA(b),recipe=recipe,k=k,
             candidate_policy='ordinary' if k<=KMAX else 'REFERENCE_ONLY_ABOVE_3K_BOUND',
             predicted_exact_parent_status_if_valid_ordinary_constructed=7 if k<=KMAX else None,
             expected_torch=exp)
Z=dict(kind='none');A=dict(kind='all');M=dict(kind='skew')
P=lambda *coords:dict(kind='points',coords=coords)
T=lambda k:dict(kind='prefix',k=k)
raw_cases=[
 ('zero-first-axis',[0,3,7],Z),('zero-second-axis',[3,0,7],Z),('zero-third-axis',[3,7,0],Z),
 ('double-zero-leading',[0,0,5],Z),('double-zero-outer',[0,5,0],Z),('double-zero-trailing',[5,0,0],Z),('all-axes-zero',[0,0,0],Z),
 ('nonempty-all-false',[4,3,5],Z),('one-corner-only',[4,3,5],P((3,2,4))),
 ('unit-cube-false',[1,1,1],Z),('unit-cube-true',[1,1,1],A),
 ('singleton-first',[1,4,7],P((0,0,6),(0,1,0),(0,3,4))),
 ('singleton-second',[4,1,7],P((0,0,6),(1,0,0),(3,0,4))),
 ('singleton-third',[4,7,1],P((0,6,0),(1,0,0),(3,4,0))),
 ('asym-columns',[3,4,7],P((2,3,6),(1,3,1),(0,2,2),(0,0,5),(0,1,0),(2,1,4),(1,0,3))),
 ('asym-columns-permuted',[7,3,4],P((6,2,3),(1,1,3),(2,0,2),(5,0,0),(0,0,1),(4,2,1),(3,1,0))),
 ('row-and-plane-carry',[3,4,5],P((0,0,4),(0,1,0),(0,3,4),(1,0,0),(1,3,4),(2,0,0))),
 ('dense-low',[2,3,4],A),('skew-mixed',[11,7,5],M),
 ('input-bound-false',[64,64,64],Z),
 ('input-bound-sparse',[64,64,64],P((0,0,0),(0,0,63),(0,63,63),(1,0,0),(31,63,63),(32,0,0),(63,63,63))),
 ('output-one-before-bound',[10,93,94],T(KMAX-1)),
 ('output-at-exact-K-bound',[10,93,94],T(KMAX)),
 ('reference-output-first-above-bound',[10,93,94],T(KMAX+1)),
]
cases=[make(*c) for c in raw_cases]
ad=[x for x in cases if x['candidate_policy']=='ordinary']
summary=dict(cases=len(cases),new_scope=len(ad),reference_only=len(cases)-len(ad),
    scope_words_each_order=sum(3*c['k'] for c in ad),all_words_each_order=sum(3*c['k'] for c in cases),
    scope_k0=sum(c['k']==0 for c in ad),scope_k1=sum(c['k']==1 for c in ad),
    scope_orders_distinct=sum(c['expected_torch']['logical_sha256']!=c['expected_torch']['physical_sha256'] for c in ad),
    exact_input_cap=sum(c['input_count']==NMAX for c in ad),exact_output_k_cap=sum(c['k']==KMAX for c in ad))
out=dict(tier='DETACHED_ORDINARY_NUMERICAL_STD_LIB_PREDICTIONS',
    parent_git='cf89ed3b9969ca2f548738b211e7b756ae2e0913',parent_tree='c028424ac03c7e4900a450042efc8a4edfb5263b',
    parent_source='c76588477d02e69e96fd86c8677430602c0c0f6afae198c07a0e60e906360155',
    parent_elf='c971732c76049f2b4d760177085f81f73ba602c7880e075625fb8a15bac06c54',
    torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
    torch_wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
    cap=dict(max_input_words=NMAX,max_logical_output_words=NMAX,max_selected=KMAX,
      largest_representable_scoped_3K=3*KMAX,smallest_out_of_scope_3K=3*(KMAX+1)),
    cases=cases,summary=summary,notes=[
       'Valid ordinary owned non-grad contiguous rank-3 canonical Bool inputs only. No malformed, fault, pressure, cap refusal, views, private types, Cat, or held work.',
       'Candidate policy expected at most N=262144 and K=floor(262144/3)=87381 where 3K=262143. The above 87382 output case is Torch reference only, not a native trial or allocator/OOM assertion.',
       'Future independent ordinary raw physical readback compares all three columns and exact tensor/storage/version tokens, including K0 with no data access. Logical route compares flattened lexicographic coordinate triplets.',
       'Zero in any input axis produces K0 and no division or payload address; declared input strides retain max(axis,1) recursion, output [0,3] stride [1,0]. For K=1 expect [1,3] strides [1,1] and C-contiguous.',
       'No archived exact rank-3 Bool status7 witness located in prior rank-1/rank-2 ordinary proof: verify fresh actual predecessor refusal with legal setup, do not relabel prior traces.',
       'Preserve rank1 and rank2 ordinary positive histories; no opts/default/tuple/out changes. Torch/native measurement pending.'
    ])
b=(json.dumps(out,sort_keys=True,separators=(',',':'))+'\n').encode()
with Path(sys.argv[1]).open('xb') as fp:fp.write(b)
print(len(b),SHA(b),summary)
