"""Independent preregistration only; no Torch, HIA, producer or tool imports."""
from pathlib import Path
import hashlib,itertools,json,math,struct,sys
CAP=262144
LOW=-(1<<63); HIGH=(1<<63)-1
PAT=(LOW,LOW+1,LOW+0xffffffff,LOW+(1<<40),-(1<<62),-(1<<53)-1,-(1<<32)-1,-(1<<32),-1,
 0,1,0x7fffffff,0x80000000,0xffffffff,1<<32,(1<<53)+1,(1<<62)+0x5a,0x5555555555555555,HIGH-1,HIGH)
def sha(b):return hashlib.sha256(b).hexdigest()
def encode_signed(values):
 return b''.join(struct.pack('<q',v) for v in values)
def coords(shape):return itertools.product(*(range(q) for q in shape))
def source_index(i,shape,config):
 kind=config['kind'];seed=config['seed']
 if kind=='signed_pattern': return PAT[(i+seed)%len(PAT)]
 assert kind=='i64_add_cartesian' and len(shape)==2
 row,col=divmod(i,shape[1]); phase=row%4; j=row*8192+seed
 lhs=(LOW+j, -((1<<60)+j), (1<<61)+j, HIGH-(1<<24)+j)[phase]
 rhs=13*col
 assert LOW<=lhs+rhs<=HIGH
 return lhs+rhs
def flag(i,n,c):
 k=c['kind']
 if k=='literal':return c['bits'][i]
 if k=='strided':return int(i%c['period'] in c['remainders'])
 if k=='all':return 1
 if k=='none':return 0
 if k=='edges':return int(i in (0,n-1))
 raise AssertionError(k)
def mk(label,x,m,source,mask,status=0):
 n=math.prod(x);nm=math.prod(m); delta=len(x)-len(m)
 assert 1<=len(x)<=4 and 0<=delta and all(a in (1,b) for a,b in zip(m,x[delta:],strict=True))
 assert n<=CAP if status==0 else n>CAP and status==3
 bits=bytes(flag(i,nm,mask) for i in range(nm));v=hashlib.sha256();h=hashlib.sha256();raw=bytearray();points=[]
 lookup=dict(zip(coords(m),bits,strict=True));k=0
 for i,c in enumerate(coords(x)):
  mc=tuple(0 if a==1 else b for a,b in zip(m,c[delta:],strict=True))
  w=source_index(i,x,source);b=struct.pack('<q',w);v.update(b)
  if not lookup[mc]:continue
  h.update(b)
  if n<=256:raw.extend(b)
  if k<16:points.append(dict(source_index=i,value_coords=c,mask_coords=mc,result_index=k,signed=str(w),word_hex=b.hex()))
  k+=1
 e=dict(torch_raw_sha256=h.hexdigest(),torch_raw_bytes=8*k,metadata=dict(dtype='int64',shape=[k],strides=[1],offset=0,version=0,storage_bytes=8*k),witnesses=points)
 if n<=256:e['raw_i64le_hex']=raw.hex()
 return dict(label=label,value_shape=x,mask_shape=m,value_count=n,mask_count=nm,source=source,mask=mask,
  old_status=7,new_hypothesized_status=status,pre_source_sha256=v.hexdigest(),pre_mask_sha256=sha(bits),expected=e)
lit=lambda *bits:dict(kind='literal',bits=bits)
rep=lambda n,*r:dict(kind='strided',period=n,remainders=r)
all_on=dict(kind='all'); all_off=dict(kind='none');edges=dict(kind='edges')
A=[
 ('equal-all-signed',[20],[20],all_on),
 ('equal-none-rank2',[4,5],[4,5],all_off),
 ('equal-alternate-rank2',[4,5],[4,5],rep(2,0)),
 ('equal-irregular-rank3',[2,3,4],[2,3,4],rep(7,1,2,5)),
 ('equal-edges-rank4',[2,2,3,4],[2,2,3,4],edges),
 ('scalar-true-extremes',[20],[],lit(1)),
 ('scalar-false',[2,3,4],[],lit(0)),
 ('missing-leading',[2,3,4],[4],lit(0,1,0,1)),
 ('middle-singleton',[2,3,4],[2,1,4],lit(0,1,0,1,1,0,1,0)),
 ('interior-and-trailing-singleton',[2,3,4],[1,3,1],lit(1,0,1)),
 ('rank4-multiple-singleton',[2,3,2,4],[1,3,1,4],rep(4,0,3)),
 ('zero-target-rank1',[0],[0],all_off),
 ('zero-target-scalar-true',[2,0,3],[],lit(1)),
 ('zero-target-trailing',[3,0],[3,1],lit(1,0,1)),
 ('zero-target-with-zero-mask',[0,4],[0,1],all_off),
 ('chunk-boundary-32767',[32767],[32767],all_on),
 ('chunk-boundary-32768',[32768],[32768],all_on),
 ('chunk-boundary-32769',[32769],[32769],all_on)]
R=[mk(n,x,m,dict(kind='signed_pattern',seed=0 if n in ('equal-all-signed','scalar-true-extremes') else i),mask) for i,(n,x,m,mask) in enumerate(A)]
C=[
 ('exact-cap-equal-edge',[512,512],[512,512],edges,0),
 ('exact-cap-scalar-all',[512,512],[],lit(1),0),
 ('exact-cap-scalar-none',[512,512],[],lit(0),0),
 ('exact-cap-partial-col',[512,512],[512],rep(127,1,3),0),
 ('overcap-equal-edge',[513,512],[513,512],edges,3),
 ('overcap-scalar-none',[513,512],[],lit(0),3)]
R += [mk(n,x,m,dict(kind='i64_add_cartesian',seed=101+i),mask,s) for i,(n,x,m,mask,s) in enumerate(C)]
controls=[
 dict(label='equal-count-transpose',value_shape=[2,3],mask_shape=[3,2],torch_accept=False,new=7),
 dict(label='higher-rank-mask',value_shape=[3,4],mask_shape=[1,3,4],torch_accept=True,new=7),
 dict(label='symmetric-value-expansion',value_shape=[2,1,4],mask_shape=[1,3,4],torch_accept=True,new=7),
 dict(label='incompatible-empty',value_shape=[2,0,4],mask_shape=[1,2,4],torch_accept=False,new=7),
 dict(label='mask-zero-changes-value-shape',value_shape=[2,1,4],mask_shape=[1,0,4],torch_accept=True,new=7),
 dict(label='rank0-value',value_shape=[],mask_shape=[],torch_accept=True,new=7),
 dict(label='nonbool-value-used-as-mask',torch_accept=False,new=7),
 dict(label='other-value-F64-remains-unadmitted',torch_accept=True,new=7),
 dict(label='actual-I64-value-offset-view',torch_accept=True,new=7),
 dict(label='actual-Bool-mask-offset-view',torch_accept=True,new=7),
 dict(label='stale-input-or-mask',new=1),
 dict(label='nonzero-padding',new=7),
 dict(label='valid-I64-compaction-then-unknown',new=4)]
summary=dict(cases=len(R),successes=sum(x['new_hypothesized_status']==0 for x in R),controls=len(controls),
 output_words_candidate=sum(x['expected']['torch_raw_bytes']//8 for x in R if x['new_hypothesized_status']==0),
 output_words_torch=sum(x['expected']['torch_raw_bytes']//8 for x in R),zero_targets=sum(x['value_count']==0 for x in R),
 cap_cases=sum(x['value_count']==CAP for x in R),overcap_cases=sum(x['value_count']>CAP for x in R))
meta=dict(kind='UNMEASURED_FROZEN_PREDICTIONS_ONLY',repo='TrillionniumFoundation/HeptaTorch',
 parent_git='695c770273ef7a1260272d54c14610dbb0e282cf',parent_tree='48c5bdcc1aa7c8eba6983140e8f4514f558d9519',
 parent_combined='7fea1364d10781f1b8f8c8d7f62973e6247d8d708b8af80a831e503e953b9960',
 parent_elf='458e9cffa53dbb77a4474a992775aa62b64dd6c25b314c0cfd0e1ab8fbb7494f',
 locked_torch='2.15.0+cpu',locked_torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',threads=1,
 large_fixture_native_setup='Hypothesis only: independent ordinary rank2 I64 [R,1] and [1,C] raw factory, add via existing functional integer broadcast; verify actual complete word and owner state before any masked_select test. If unavailable do not substitute a new fixture without review.',
 controls=controls,cases=R,summary=summary,
 assumptions=['I64 raw is signed <q encoded modulo 2**64 only by lexically signed source; never a floating intermediary.',
 'Read output chunks <=32768 I64 words including exact final position; cap is input N not K.',
 'Status7 old I64-value claims are prospective only until exact #67 real request replay; existing I64-mask refusals do not count.'])
b=(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n').encode()
with Path(sys.argv[1]).open('xb') as out:out.write(b)
print(len(b),sha(b),summary)
