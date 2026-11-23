"""Detached current800 coexistence ordinary numerical fixture; stdlib only."""
from pathlib import Path
import hashlib, importlib.util, json, math, sys
P=Path(__file__).resolve().parent
assert hashlib.sha256((P/'oracle_v2.py').read_bytes()).hexdigest()=='76116587e0d78e0f65b26c9b4752aaa98d1d29bae8192ce9cda5837916747740'
s=importlib.util.spec_from_file_location('frozen_v2',P/'oracle_v2.py')
v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
sha=lambda b:hashlib.sha256(b).hexdigest()
v2=(P/'fixtures_v2.json').read_bytes()
assert sha(v2)=='7f492ae8d36bdb1c6523a5a42d09c2009702298df6798ec69258838071d060b5'
old=json.loads(v2)
spec=v.dtype_data
def nonzero(kind,shape,data):
 coords=[p for p,w in zip(v.row(shape),data,strict=True) if v.word_truth(kind,w)]
 return spec('I64',[j for p in coords for j in p],[len(coords),len(shape)],[1,len(coords)])
def case(label,kind,sh,co,msk,x,y,tailkind,tailshape,tail,where_to_mask):
 assert len(x)==len(y)==math.prod(sh) and len(msk)==math.prod(co)
 assert len(tail)==math.prod(tailshape)
 w=[a if v.broad(msk,co,p) else b for p,a,b in zip(v.row(sh),x,y,strict=True)]
 if where_to_mask:
  assert kind=='Bool'
  lattice=v.joint(sh,tailshape)
  masked=[v.broad(tail,tailshape,p) for p in v.row(lattice) if v.broad(w,sh,p)]
  args=['data','where']
  ins=dict(data=spec(tailkind,tail,tailshape))
  out=spec(tailkind,masked,[len(masked)])
 else:
  assert tailkind=='Bool'
  lattice=v.joint(sh,tailshape)
  masked=[v.broad(w,sh,p) for p in v.row(lattice) if v.broad(tail,tailshape,p)]
  args=['where','mask']
  ins=dict(mask=spec('Bool',tail,tailshape))
  out=spec(kind,masked,[len(masked)])
 assert len(sh)==1 or (len(sh)==2 and kind=='Bool')
 return dict(id=label,inputs=ins|dict(condition=spec('Bool',msk,co),lhs=spec(kind,x,sh),rhs=spec(kind,y,sh)),
  expect=dict(where=spec(kind,w,sh),masked_select=out,nonzero_where=nonzero(kind,sh,w)),
  composed_reference=[['where',129,'condition','lhs','rhs'],['masked_select',130,*args],['nonzero_where',135,'where']],
  standalone_reference=dict(where=['condition','lhs','rhs'],masked_select=[k if k!='where' else 'materialized expected.where' for k in args],nonzero=['materialized expected.where']),
  requires=['where ordinary same dtype and same value shape', 'mask select ordinary exact frozen #70', 'nonzero ordinary exact frozen #70'])
new=[]
for i,k in enumerate(v.TYPES):
 p=v.TYPES[k][1];q=len(p);n=q*2
 # Choose alternate values across both parents; preserve selected raw NaN bytes.
 new.append(case(f'where-values-{k}-r1',k,[n],[n],[int(j%3==0) for j in range(n)],
  v.words(k,n,i),v.words(k,n,i+2),'Bool',[n],[int(j%4!=1) for j in range(n)],False))
 new.append(case(f'where-mask-{k}-r2','Bool',[3,4],[3,4],[int((j+i)%3==1) for j in range(12)],
  [1,0,1,0,1,1,0,0,0,0,1,1],[0,1,1,0,1,0,1,0,0,1,0,1],k,[3,4],v.words(k,12,i),True))
# Scalar here is the Bool condition, not a mixed/extra numeric operand.
new.append(case('where-scalar-condition-f32','F32',[9],[],[1],v.words('F32',9,0),v.words('F32',9,3),'Bool',[1],[1],False))
# Rank1 empty output avoids imposing an unqualified noncanonical current800 stride.
new.append(case('where-empty-bool-r1','Bool',[0],[0],[],[],[],'F64',[0],[],True))
orig={c['id']:c for c in old['cases']}
d={'schema':'heptatorch.current800.coexistence.numerical.v1',
 'projection_parents':{'where':{'pr':66,'git_sha':'7583ea54756f6c76f722b95b2e569f417d584fc4','source_sha256':'43999e8e9afa40b78f24007697623d5a297308643bff160392fec49d388fe26b','native_sha256':'2ee8c061f44e0496321141992881ca59557a22ccb7782e5b7cf1a1a313fce8f6'},
  'coexistence':{'pr':70,'git_sha':'b1e1fd20de94fa12068ce1e33da2bc7c6f1f8026','source_sha256':'d64f15921b169ed48199356989aa1f31dc7937fed1f772e27e0e291b9717cc1f','native_sha256':'31a267d82b274ae81a956a03f8da1c9eb5a28bbcc903f78e56744333d1a88d4d'}},
 'main':old['main'],'status':'pure predictions; no measured claim',
 'unchanged_v2_sha256':sha(v2),'unchanged_v2_case_ids':list(orig),
 'new_where_cases':new,'cases':[ *old['cases'],*new ],
 'constraints':{'owner':old['constraints']['owners'],'observer':old['constraints']['observation'],
  'scope':'No public-F64 encoded empty layouts or mutable/out, views, gradients, operator failures, stress or unsupported requests. Canonical empty ordinary roots only. Op129 values use same public dtype and same shape; only one genuine rank-zero Bool condition probe. Numeric nonzero always rank1, Bool nonzero rank1 or rank2.',
  'isolated':'Use #66 exact where separately for 14 where outputs. Use either #70 exact prior combined or the three frozen standalone images for additive-family reference opcodes with materialized where result; no ownership handles migrate between images.',
  'activation':'Compare unchanged canonical original dispatch and ownership source statically. This fixture measures ordinary coexistence/cross-operator composition only, not #65 mutation, default routing, release/gating or full prior corpora.'}}
assert len(d['cases'])==50 and sum(len(c['expect']) for c in d['cases'])==171
for c in d['cases']:
 for n,op,*arg in c['composed_reference']:
  if op==135:
   q=(c['inputs']|c['expect'])[arg[0]]; assert len(q['shape'])==1 or (len(q['shape'])==2 and q['dtype']=='Bool')
out=(json.dumps(d,sort_keys=True,separators=(',',':'))+'\n')
if len(sys.argv)>1: assert Path(sys.argv[1]).read_text()==out
else: print(out,end='')
