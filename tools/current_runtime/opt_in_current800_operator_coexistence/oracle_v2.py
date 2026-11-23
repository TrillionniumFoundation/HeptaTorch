"""Numerical fixture generator only; stdlib, no source or runtime access."""
import hashlib,itertools,json,math,sys
from pathlib import Path

BASE='e9444902df9878060ab389a70ee957c1b7f4a887'
TYPES={
 'Bool':(1,[0,1,1,0]),
 'I64':(8,[0,1,0x8000000000000000,0x7fffffffffffffff,0xffffffffffffffff,3]),
 'F32':(4,[0,0x80000000,0x1,0x80000001,0x7f800000,0xff800000,0x7fcab123,0xff812345,0x3f000000]),
 'F64':(8,[0,0x8000000000000000,1,0x8000000000000001,0x7ff0000000000000,0xfff0000000000000,0x7ff8abcde1234567,0xfff0123456789abc,0x3ff0000000000000]),
 'F16':(2,[0,0x8000,1,0x8001,0x7c00,0xfc00,0x7e01,0xfc09,0x3800]),
 'BF16':(2,[0,0x8000,1,0x8001,0x7f80,0xff80,0x7fc3,0xff85,0x3f00]),
}
OPS={'and':131,'or':133,'xor':134}
def row(a):
 return list(itertools.product(*(range(n) for n in a)))
def at(data,sh,p):
 off=0
 for k,n in zip(p,sh,strict=True):
  off=off*n+k
 return data[off]
def broad(data,sh,p):
 return at(data,sh,[0 if n==1 else k for k,n in zip(p[len(p)-len(sh):],sh,strict=True)])
def joint(a,b):
 n=max(len(a),len(b)); out=[]
 for x,y in zip([1]*(n-len(a))+a,[1]*(n-len(b))+b,strict=True):
  if x==y or y==1: out.append(x)
  elif x==1: out.append(y)
  else: raise ValueError('nonbroadcast shape not an admitted fixture')
 return out
def dtype_data(kind,words,shape,strides=None):
 width,_=TYPES[kind]
 data=b''.join(x.to_bytes(width,'little') for x in words)
 assert len(words)==math.prod(shape)
 if strides is None: strides=[math.prod(shape[i+1:]) for i in range(len(shape))]
 return dict(dtype=kind,shape=shape,strides=strides,raw_le=data.hex(),sha256=hashlib.sha256(data).hexdigest())
def words(kind,n,offset):
 p=TYPES[kind][1]
 return [p[(i+offset)%len(p)] for i in range(n)]
def word_truth(kind,v):
 if kind in ('I64','Bool'): return v!=0
 return (v & ((1 << (TYPES[kind][0]*8-1))-1))!=0
def mask(a,b,op):
 return {'and':a and b,'or':a or b,'xor':a!=b}[op]
def mask_fixtures(rank,op,i):
 if rank==1:
  if op=='and':return [9],[1,1,0,1,0,1,1,0,1],[1],[1],[9]
  if op=='or':return [],[0],[9],[1,0,1,0,0,1,0,1,1],[9]
  return [9],[1,0,1,0,0,1,0,1,1],[9],[1,0,1,0,0,1,0,1,1],[9]
 if op=='and': return [3,1],[1,0,1],[1,4],[0,1,1,1],[3,4]
 if op=='or': return [3,4],[0,1,0,0,1,1,0,0,0,0,0,1],[],[0],[3,1]
 return [3,4],[0,1,0,1,1,0,1,0,0,1,1,0],[1,4],[0,1,1,1],[3,4]
def case(kind,op,rank,i):
 ash,a,bsh,b,vsh=mask_fixtures(rank,op,i)
 mshape=joint(ash,bsh)
 mcoords=row(mshape)
 m=[int(mask(broad(a,ash,p),broad(b,bsh,p),op)) for p in mcoords]
 v=words(kind,math.prod(vsh),0)
 selshape=joint(vsh,mshape)
 sel=[broad(v,vsh,p) for p in row(selshape) if broad(m,mshape,p)]
 orig=[list(p) for p,u in zip(mcoords,m,strict=True) if u]
 typed=[list(p) for p,w in zip(row(vsh),v,strict=True) if word_truth(kind,w)]
 nn=lambda pp,r:dtype_data('I64',[x for q in pp for x in q],[len(pp),r],[1,len(pp)])
 out=dict(logic=dtype_data('Bool',m,mshape),masked_select=dtype_data(kind,sel,[len(sel)]),nonzero_mask=nn(orig,rank))
 calls=[['logic',OPS[op],'lhs','rhs'],['masked_select',130,'values','logic'],['nonzero_mask',135,'logic']]
 ref_nonzero=['materialized expected.logic']
 requirements=['bool logical ordinary','masked_select ordinary raw '+kind,'nonzero ordinary rank '+str(rank)+' Bool']
 coords={'mask_rows':orig}
 if len(vsh)==1 or kind=='Bool':
  out['nonzero_values']=nn(typed,len(vsh))
  calls.append(['nonzero_values',135,'values'])
  ref_nonzero.append('values')
  requirements.append('nonzero ordinary rank '+str(len(vsh))+' '+kind)
  coords['value_truth_rows']=typed
 return dict(id=f'{kind}-{op}-r{rank}',expect=out,inputs=dict(lhs=dtype_data('Bool',a,ash),rhs=dtype_data('Bool',b,bsh),values=dtype_data(kind,v,vsh)),standalone_reference={
  'logic':['lhs','rhs'], 'masked_select':['values','materialized expected.logic'], 'nonzero':ref_nonzero},
  composed_reference=calls,coordinate_checks=coords,requires=requirements)
def build():
 cases=[case(k,o,r,i) for i,k in enumerate(TYPES) for o in OPS for r in (1,2)]
 for c in cases:
  for name,op,*args in c['composed_reference']:
   if op==135:
    d=c['expect']['logic'] if args==['logic'] else c['inputs']['values']
    assert len(d['shape'])==1 or (len(d['shape'])==2 and d['dtype']=='Bool')
 return {'schema':'heptatorch.coexistence.numerical.v2','main':BASE,'exact_draft_heads':{'67':'270fde9f8b597c1444fd711026121fef4845b36d','68':'dc4c9fd5efb9526241ea21ac83a4fa5b9196570f','69':'2b5b2fcd8b7937d5441317da1becf9cbef360a1b'},'status':'preregistered pure predictions only','cases':cases,'num_cases':len(cases),
 'constraints':{'same_image':'all opcodes in each composed route use one combined ELF; exact isolated references use one dedicated pinned image each, never marshal raw tokens across images','scope':'public ordinary CPU roots; row-major input payload and offset 0, no grad. Rank 2 nonzero only on Bool (logic intermediate or optional Bool value input); native typed original nonzero is included only for rank1. No rank2 typed-original request.',
 'owners':'each operation output expected fresh owned Tensor, Storage and Version0, disjoint identities versus every co-live input/output; input payload/metadata/version unchanged. Do not require raw IDs to coincide across sessions or all outputs to have monotone physical addresses.',
 'observation':'compare selected bytes including all floating NaN payloads without conversion; logical data must be C-order, rank-2 I64 nonzero coordinates need independent strided owner decoding, not physical byte order; expected.raw_le always logical row-major, strides enforce physical mapping.',
 'evidence':'hold supported ordinary expected cases only; record unsupported exact-parent lanes as out-of-scope rather than substitutes; these fixtures imply neither complete previous corpus retention nor production/default readiness.'}}
if __name__=='__main__':
 x=json.dumps(build(),sort_keys=True,separators=(',',':'))+'\n'
 if len(sys.argv)>1:
  assert Path(sys.argv[1]).read_text()==x
 else: print(x,end='')
