"""Detached stdlib-only rank-4 CPU dtype nonzero predictions."""
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
  m={tuple(p[:4]):p[4] for p in r['points']}
  assert len(m)==len(r['points']) and all(all(0<=k<d for k,d in zip(p,shape,strict=True)) for p in m)
  seq=(m.get(p,0) for p in grid(shape))
 else:raise AssertionError(kind)
 return b''.join(w.to_bytes(bits//8,'little') for w in seq)
def pack(ws):
 return b''.join(struct.pack('<q',w) for w in ws)
def make(dt,label,shape,r):
 a,b,c,d=shape;tag,bits,exp,_=SPECS[dt];sz=bits//8
 raw=source(dt,shape,r);assert len(raw)==a*b*c*d*sz
 bits_mask=(1<<(bits-int(exp>0)))-1
 chosen=[]
 # Enumerate in deliberately non-rowmajor order, then explicitly sort coordinate tuples.
 for inner in range(d):
  for second in range(b):
   for outer in range(a):
    for third in range(c):
     off=(((outer*b+second)*c+third)*d+inner)*sz
     word=int.from_bytes(raw[off:off+sz],'little')
     if dt=='bool':assert word in (0,1)
     if word&bits_mask:chosen.append((outer,second,third,inner))
 chosen.sort()
 k=len(chosen)
 logical=pack(v for coords in chosen for v in coords)
 columns=[pack(p[axis] for p in chosen) for axis in range(4)]
 physical=b''.join(columns)
 assert a*b*c*d<=262144 and 4*k<=262144 and len(logical)==len(physical)==32*k
 expect=dict(dtype='int64',shape=[k,4],strides=[1,k],version=0,offset=0,view=False,
     requires_grad=False,is_contiguous=k<=1,storage_bytes=32*k,
     logical_sha256=sha(logical),physical_sha256=sha(physical),column_sha256=list(map(sha,columns)),
     first8=chosen[:8],last8=chosen[-8:])
 if k<25:expect.update(logical_hex=logical.hex(),physical_hex=physical.hex())
 return dict(label=dt+'-'+label,input_dtype=dt,input_tag=tag,input_shape=shape,
    input_strides=[math.prod(max(x,1) for x in shape[i+1:]) for i in range(4)],input_count=a*b*c*d,input_raw_sha256=sha(raw),recipe=r,k=k,
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
 cases += [
  make(dt,'empty-first',[0,2,3,5],rr([])), make(dt,'empty-second',[2,0,3,5],rr([])),
  make(dt,'empty-third',[2,3,0,5],rr([])), make(dt,'empty-fourth',[2,3,5,0],rr([])),
  make(dt,'empty-separated',[0,3,0,5],rr([])),
  make(dt,'all-false',[2,3,4,5],Z),make(dt,'one-corner',[2,3,4,5],pp((1,2,3,4,high))),
  make(dt,'full-raw-patterns',[2,3,4,5],cc(0)),
  make(dt,'singleton-first',[1,3,4,5],cc(7)),
  make(dt,'singleton-second',[3,1,4,5],cc(9)),
  make(dt,'singleton-third',[3,4,1,5],cc(11)),
  make(dt,'singleton-fourth',[3,4,5,1],cc(13)),
  make(dt,'sparse-all-columns',[3,4,5,6],pp(
       (2,3,4,5,high),(1,0,3,2,other),(0,3,4,1,1),(0,2,1,5,high),
       (2,3,2,0,other),(2,1,3,3,1),(1,3,1,4,high),(2,0,1,2,other))),
 ]
mapping={'bool-retained-empty':'zero-interior-2','bool-retained-corner':'one-corner','bool-retained-asymmetric':'asymmetric-all-four'}
cases += [
 make('bool','retained-empty',[3,4,0,5],rr([])),
 make('bool','retained-corner',[2,3,4,5],pp((1,2,3,4,1))),
 make('bool','retained-asymmetric',[2,3,4,5],pp((1,2,3,4,1),(0,0,0,3,1),(1,1,1,0,1),(0,2,3,0,1),(0,1,2,4,1))),
]
if __name__=='__main__':
 head,tree,dest,prior_path=sys.argv[1:]; assert all(len(s)==40 and all(c in '0123456789abcdef' for c in s) for s in (head,tree))
 prior=Path(prior_path).read_bytes();assert sha(prior)=='a79b5753ed6ca6f2972edd4629aafb299789eeae76e431e6f1c4aa8370dbd876'
 original={x['label']:x for x in json.loads(prior)['cases']}
 for c in cases:
  if c['label'] not in mapping:continue
  name=mapping[c['label']]; old=original[name]; assert not old['optional_in_cap_reference']
  assert (old['input_shape'],old['input_strides'],old['input_sha256'])==(c['input_shape'],c['input_strides'],c['input_raw_sha256'])
  prev=old['expected_torch']; now=c['expected_torch']
  assert all(prev[k]==now[k] for k in ('shape','strides','version','offset','storage_bytes','logical_sha256','physical_sha256','column_sha256'))
  c['retained_rank4_bool_fixture']=name
 # Check the dedicated ordinary full-pattern row includes each signed bit class before measuring.
 for dt in SPECS:
  if dt=='bool':continue
  c=next(c for c in cases if c['label']==dt+'-full-raw-patterns'); _,bits,_,_=SPECS[dt]; w=bits//8
  raw=source(dt,c['input_shape'],c['recipe'])
  assert set(patterns(dt))=={int.from_bytes(raw[i:i+w],'little') for i in range(0,len(raw),w)}
 assert len(cases)==68 and len({c['label'] for c in cases})==68 and max(c['input_count'] for c in cases)==360
 summary=dict(cases=len(cases),new=sum(c['input_dtype']!='bool' for c in cases),retained_bool=len(mapping),
  words_each_order=sum(4*c['k'] for c in cases),new_words_each_order=sum(4*c['k'] for c in cases if c['input_dtype']!='bool'),
  case_k0=sum(c['k']==0 for c in cases),case_k1=sum(c['k']==1 for c in cases),
  case_distinct_orders=sum(c['expected_torch']['logical_sha256']!=c['expected_torch']['physical_sha256'] for c in cases),
  max_n=max(c['input_count'] for c in cases),max_k=max(c['k'] for c in cases))
 out=dict(tier='DETACHED_ORDINARY_STD_LIB_PREDICTIONS',parent_git=head,parent_tree=tree,
  parent_source='b9bf71e3224190ce9db80a34a30620aa8488d27047472601f5070b995fa92891',
  parent_elf='6e2c23426239421cf62949c56d6ee80ac9326cef97de3529c2ccd695d5cc002c',
  pinned_torch='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
  retained_rank4_bool_prereg_sha256=sha(prior),summary=summary,cases=cases,notes=[
   'Only rank4 ordinary owned CPU contiguous zero-offset no-grad valid numeric inputs. No stress, cap, unsupported layouts, out/tuple, or general dispatch claims.',
   'Both signed IEEE zeros false; signed subnormal/normal/inf and q/sNaN true. All I64 nonzero words true, including signed extrema and high-bit-only. Use literal typed bytes, not floating conversion.',
   'Expected one fresh I64 tensor [K,4], strides [1,K], storage bytes 32K and offset/Version zero, contiguous iff K<=1 even for empty. Independently compare logical tuples and each physical column. Input bytes and metadata must remain identical.',
   'Numeric prior status7 predicted only after legitimate ordinary typed setup; not actual evidence. Three Bool controls are exact geometry/bytes/output bindings to the prior measured and native qualified corpus.',
   'Small literal op97 typed input setup planned, no casts or high-cap fixture.'
  ])
 raw=(json.dumps(out,sort_keys=True,separators=(',',':'))+'\n').encode()
 with Path(dest).open('xb') as f:f.write(raw)
 print(len(raw),sha(raw),summary)
