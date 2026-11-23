"""Detached independent stdlib oracle for ordinary rank4 Bool nonzero."""
import hashlib,itertools,json,math,struct,sys
from pathlib import Path
sha=lambda x:hashlib.sha256(x).hexdigest()
CAP=262144;KCAP=CAP//4
coords=lambda shape:itertools.product(*(range(n) for n in shape))
pack=lambda values:b''.join(struct.pack('<q',v) for v in values)
def source(shape,r):
 n=math.prod(shape);kind=r['kind']
 if kind=='zero':return bytes(n)
 if kind=='all':return bytes([1])*n
 if kind=='prefix':
  k=r['k'];assert 0<=k<=n
  return bytes([1])*k+bytes(n-k)
 if kind=='points':
  p={tuple(c) for c in r['indices']};assert len(p)==len(r['indices'])
  assert all(all(0<=i<d for i,d in zip(c,shape,strict=True)) for c in p)
  return bytes(int(c in p) for c in coords(shape))
 if kind=='skew':
  return bytes(int((11*a+17*b+23*c+31*d+3*a*d+b*c)%47 in (0,1,11,17,29,37)) for a,b,c,d in coords(shape))
 raise AssertionError(kind)
def make(label,shape,r):
 assert len(shape)==4 and all(0<=x<=CAP for x in shape)
 n=math.prod(shape);assert n<=CAP
 bits=source(shape,r);assert len(bits)==n and all(v in (0,1) for v in bits)
 tuples=[c for c,b in zip(coords(shape),bits,strict=True) if b]; k=len(tuples)
 assert k==sum(bits) and k<=KCAP
 logical=pack(v for c in tuples for v in c)
 # Independently pack physical interleave by four storage columns, not row tuples.
 cols=[pack(c[i] for c in tuples) for i in range(4)]
 physical=b''.join(cols)
 assert len(logical)==len(physical)==32*k
 expected=dict(dtype='int64',shape=[k,4],strides=[1,k],contiguous=k<=1,
  storage_bytes=32*k,version=0,offset=0,requires_grad=False,view=False,
  logical_sha256=sha(logical),physical_sha256=sha(physical),column_sha256=[sha(c) for c in cols],
  head=tuples[:6],tail=tuples[-6:])
 if k<=25:
  expected.update(logical_hex=logical.hex(),physical_hex=physical.hex())
 st=[math.prod(max(x,1) for x in shape[a+1:]) for a in range(4)]
 return dict(label=label,input_dtype='bool',input_shape=shape,input_strides=st,input_count=n,
  input_sha256=sha(bits),input_byte_and_metadata_unchanged=True,recipe=r,k=k,
  predicted_parent_status_if_normal_valid_setup=7,expected_torch=expected)
Z=dict(kind='zero');A=dict(kind='all');M=dict(kind='skew')
P=lambda *c:dict(kind='points',indices=c)
T=lambda k:dict(kind='prefix',k=k)
entries=[
 ('zero-outer-axis',[0,3,4,5],Z),('zero-interior-1',[3,0,4,5],Z),
 ('zero-interior-2',[3,4,0,5],Z),('zero-inner-axis',[3,4,5,0],Z),
 ('zero-separate-outer-inner',[0,3,4,0],Z),('zero-both-interior',[2,0,0,5],Z),
 ('zero-all',[0,0,0,0],Z),('unit-only-false',[1,1,1,1],Z),('unit-only-true',[1,1,1,1],A),
 ('all-false-asymmetric',[2,3,4,5],Z),('one-corner',[2,3,4,5],P((1,2,3,4))),
 ('singleton-outer',[1,3,4,5],P((0,0,3,4),(0,1,0,0),(0,2,2,3))),
 ('singleton-interior-1',[3,1,4,5],P((0,0,3,4),(1,0,0,0),(2,0,2,3))),
 ('singleton-interior-2',[3,4,1,5],P((0,3,0,4),(1,0,0,0),(2,2,0,3))),
 ('singleton-inner',[3,4,5,1],P((0,3,4,0),(1,0,0,0),(2,2,3,0))),
 ('asymmetric-all-four',[2,3,4,5],P((1,2,3,4),(0,0,0,3),(1,1,1,0),(0,2,3,0),(0,1,2,4))),
 ('rotated-asymmetric-all-four',[5,2,3,4],P((4,1,2,3),(0,1,0,3),(1,0,1,0),(3,1,2,0),(2,0,2,1))),
 ('carries-all-three-boundaries',[2,3,4,5],P((0,0,0,4),(0,0,1,0),(0,0,3,4),(0,1,0,0),(0,2,3,4),(1,0,0,0))),
 ('dense-low',[2,3,2,2],A),('skew-mixed',[3,5,6,7],M),
 ('optional-input-cap-false',[16,16,32,32],Z),
 ('optional-input-cap-sparse',[16,16,32,32],P((0,0,0,31),(0,0,1,0),(0,15,31,31),(1,0,0,0),(15,15,31,31))),
 ('optional-output-k-cap-minus-one',[16,16,16,16],T(KCAP-1)),
 ('optional-output-k-cap-exact',[16,16,16,16],A),
]
cases=[make(*x) for x in entries]
for row in cases:row['optional_in_cap_reference']=row['label'].startswith('optional-')
basic=[r for r in cases if not r['optional_in_cap_reference']]
source_path,elf_path,tree,out=sys.argv[1:]
s=Path(source_path).read_bytes();elf=Path(elf_path).read_bytes()
assert sha(s)=='b6d42642301c7b35da2632c9ef64db61715f2f21dcb24547b9b34aaba98dba52'
assert sha(elf)=='db28aa0c8f2ca4f59ccf7a8d52eb30a324d8d1464fdd7ac7a3cf920df5353772'
assert tree.startswith('42b71a4') and len(tree) in (7,40) and all(x in '0123456789abcdef' for x in tree)
out_obj=dict(tier='DETACHED_STD_LIB_ORDINARY_RANK4_BOOL',parent_git='e11598e93472bbed0e4d1c8e72ae63e800db19df',
 parent_tree_binding=tree,parent_tree_binding_is_full=len(tree)==40,source_sha256=sha(s),elf_sha256=sha(elf),
 torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
 bounds=dict(input_count_max=CAP,output_element_count_max=CAP,selected_rows_max=KCAP),
 cases=cases,summary=dict(cases=len(cases),base_cases=len(basic),optional_in_cap=len(cases)-len(basic),
   base_logical_and_physical_words_each=sum(4*x['k'] for x in basic),all_words_each=sum(4*x['k'] for x in cases),
   base_max_input=max(x['input_count'] for x in basic),k0=sum(x['k']==0 for x in cases),
   k1=sum(x['k']==1 for x in cases),order_discriminants=sum(x['expected_torch']['logical_sha256']!=x['expected_torch']['physical_sha256'] for x in cases)),
 notes=[
   'Only canonical Bool ordinary owned contiguous CPU rank-4 zero-offset and no-grad. Strict N=product(axes), each validated, N≤262144 and K≤65536 so 4K≤262144. The optional four cap references are within that numerical policy; no above-cap or stress trial or assertion.',
   'Predictions include canonical input strides recursively using max(axis,1), and fresh ordinary I64 shape [K,4], strides [1,K], Version0, storage 32K bytes, no view, row-major contiguous iff K≤1, unchanged input.',
   'Positive K writes separate physical axis columns j+d*K into locally checked allocation prior to metadata transpose; logical coordinates in row-major lexicographic order and four independent column hashes. Empty input/K0 produces no coordinate division or data access and strides [1,0].',
   'Current exact predecessor routes only rank2/3 to shared kernel and falls back to rank1 guard; rank4 status7 is predicted, not measured. No actual old archived rank4 ordinary case claimed. No numeric rank4, runtime/source change, default, held, fault or invalid scope.'
 ])
raw=(json.dumps(out_obj,sort_keys=True,separators=(',',':'))+'\n').encode()
with Path(out).open('xb') as f:f.write(raw)
print(len(raw),sha(raw),out_obj['summary'])
