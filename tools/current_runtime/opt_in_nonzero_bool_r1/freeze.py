"""Independent stdlib-only predictions; does not import Torch or invoke native."""
from pathlib import Path
import hashlib,struct,sys,json
CAP=262144
def sha(b):return hashlib.sha256(b).hexdigest()
def source(n,r):
 k=r['kind']
 if k=='all':return b'\1'*n
 if k=='none':return bytes(n)
 if k=='points':
  hits=tuple(r['positions']);assert len(set(hits))==len(hits) and all(0<=i<n for i in hits)
  b=bytearray(n)
  for i in hits:b[i]=1
  return bytes(b)
 if k=='even':return bytes(1-i%2 for i in range(n))
 if k=='pattern':return bytes(int((17*i+3)%31 in (0,3,6,7,17,29)) for i in range(n))
 raise AssertionError(k)
def make(label,n,recipe,status=0):
 a=source(n,recipe);out=hashlib.sha256();lit=bytearray();count=0;head=[];tail=[]
 assert all(x in (0,1) for x in a)
 for i,v in enumerate(a):
  if not v:continue
  q=struct.pack('<q',i);out.update(q);count+=1
  if n<=256:lit.extend(q)
  if len(head)<18:head.append(i)
  tail.append(i)
  if len(tail)>18:tail.pop(0)
 p=dict(dtype='int64',shape=[count,1],strides_hypothesis=[1,count],storage_bytes=8*count,offset=0,requires_grad=False,
          fresh_storage_and_version=True,view=False,version_counter='MEASURE_PINNED_TORCH_FIRST',
          raw_i64le_sha256=out.hexdigest(),head_first18=head,tail_last18=tail)
 if n<=256:p['raw_i64le_hex']=lit.hex()
 assert (n<=CAP and status==0) or (n>CAP and status==3)
 return dict(label=label,input_shape=[n],input_dtype='bool',input_count=n,recipe=recipe,input_raw_sha256=sha(a),hypothesized_candidate_status=status,expected_torch=p)
pt=lambda *i:dict(kind='points',positions=i)
A=dict(kind='all'); Z=dict(kind='none');E=dict(kind='even');P=dict(kind='pattern')
R=[
 ('true-empty-input',0,Z),('one-false',1,Z),('one-true',1,A),
 ('all-false-nonempty',33,Z),('sparse-in-middle-k1',37,pt(17)),
 ('two-at-ends-k2',37,pt(0,36)),('sentinel-order-k7',257,pt(250,127,0,1,63,128,200)),
 ('dense-k19',19,A),('even-k129',257,E),('irregular-k-not-nonzero-count',5003,P),
 *[(f'grain-{n}',n,pt(0,7,8,n//2,n-2,n-1)) for n in (32767,32768,32769)],
 ('multi-grain-unequal-chunk-count',131073,pt(0,1,32767,32768,32770,65534,65535,65536,98304,131071,131072)),
 ('cap-dense',CAP,A),('cap-all-false',CAP,Z),('cap-sparse',CAP,pt(0,32767,32768,131071,131072,CAP-1))]
cases=[make(*x) for x in R]+[make('overcap-K0',CAP+1,Z,3),make('overcap-K2',CAP+1,pt(0,CAP),3)]
guards=[
 dict(label='scalar-false',torch_shape=[0,0],torch_strides_hypothesis=[1,0],torch_storage_bytes=0,local_status=7),
 dict(label='scalar-true',torch_shape=[1,0],torch_strides_hypothesis=[1,1],torch_storage_bytes=0,local_status=7),
 dict(label='rank2-control',torch_shape=[3,2],torch_strides_hypothesis=[1,3],torch_coords=[[0,0],[0,2],[1,1]],local_status=7),
 dict(label='f32-truth-control',local_status=7,torch_scope_only=True),
 dict(label='i64-truth-control',local_status=7,torch_scope_only=True),
 dict(label='actual-offset-Bool-view',local_status=7,torch_scope_only=True),
 dict(label='stale-source',local_status=1),dict(label='padding',local_status=7),
 dict(label='valid-nonzero-followed-by-unknown',local_status=4),
 dict(label='invalid-Bool-raw-factory',setup_only=True,never_inject_noncanonical_owner=True)]
N=lambda rows:sum(a['expected_torch']['shape'][0] for a in rows)
summary=dict(cases=len(cases),candidate_successes=sum(a['hypothesized_candidate_status']==0 for a in cases),
 oracle_words=N(cases),candidate_words=N([a for a in cases if a['hypothesized_candidate_status']==0]),
 candidate_zero_k=sum(a['expected_torch']['shape'][0]==0 for a in cases if a['hypothesized_candidate_status']==0),
 exact_input_cap=sum(a['input_count']==CAP for a in cases),over_input_cap=sum(a['input_count']>CAP for a in cases),controls=len(guards))
out=dict(tier='FROZEN_STD_LIB_PREDICTIONS_ONLY',baseline_git='e9444902df9878060ab389a70ee957c1b7f4a887',
 combined='0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43',
 native='ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e',
 torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 torch_wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
 torch_profile='Measure fresh nonzero.default, one I64 output (not tuple/out). Threads1 all; representative threads2/4, actual Version counter and shape/element strides of empty/scalar. No pointer as identity for empty.',
 wire='No opcode reservation; 131 is occupied by logical-and; 132 has no local hit but requires full branch census. New op baseline refusal status4 is prospective pending exact native old capture.',
 native_plans=['Check true public Bool input shape/count, ordinary owned canonical offset0, live Tensor+Storage+Version and input cap before either pass.',
 'Check K0 owning descriptor and actual metadata [0,1] stride [1,0] independent of zero-byte sentinel; output and input tokens/storage/version distinct including both empty.',
 'Read exact output qwords <=32768 indices per chunk; crosscheck logical order and actual raw storage, element stride units, materialized result no-view and source unchanged.',
 'Exercise both destruction orders, weak acquisition then expiry/releases, immediate same-frame handle read and one committed-success plus unknown suffix rollback under exact pre-owner. Refusal status3 tests proposed before K analysis, not artificial OOM.',
 'Keep exact fresh counter from real locked wheel; do not globally bump/set strides or Version. Verify owned row and tensor descriptor are both consistent before private pre-publish metadata assignment.'],
 cases=cases,guards=guards,summary=summary)
b=(json.dumps(out,sort_keys=True,separators=(',',':'))+'\n').encode()
with Path(sys.argv[1]).open('xb') as f:f.write(b)
print(len(b),sha(b),summary)
