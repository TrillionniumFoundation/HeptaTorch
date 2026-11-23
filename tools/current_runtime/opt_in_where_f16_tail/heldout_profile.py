"""Pinned official CPU Torch fixture only; never imports the runtime."""
import hashlib,itertools,json,math,sys
from pathlib import Path
H,L,I,out=map(Path,sys.argv[1:])
assert hashlib.sha256((H/'qualify_runtime_torch.py').read_bytes()).hexdigest()=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
torch,oracle=load_oracle(L,I)
DT={0:(4,torch.int32,torch.float32),1:(1,torch.uint8,torch.bool),2:(8,torch.int64,torch.int64),
    4:(8,torch.int64,torch.float64),5:(2,torch.int16,torch.float16),6:(2,torch.int16,torch.bfloat16)}
def floating(sign,inf,qnan,one):
 return [0,sign,1,sign|1,inf,sign|inf,inf|qnan|3,inf|7,sign|inf|qnan|5,sign|inf|9,one,sign|one]
BITS={0:floating(1<<31,0x7f800000,1<<22,0x3f800000),
      1:[0,1,1,0,1,0,0,1,0,1,0,1],
      2:[0,1,2**64-1,2**63,2**63-1,2**63+1,9,2**64-8,73,2**64-15,37,11],
      4:floating(1<<63,0x7ff0000000000000,1<<51,0x3ff0000000000000),
      5:floating(1<<15,0x7c00,1<<9,0x3c00),
      6:floating(1<<15,0x7f80,1<<6,0x3f80)}
def make(tag,shape,words):
 w,it,t=DT[tag]
 if words: return torch.frombuffer(bytearray(b''.join(a.to_bytes(w,'little') for a in words)),dtype=it).view(t).reshape(shape)
 return torch.empty(shape,dtype=t)
def raw(t):
 return bytes(t.contiguous().reshape(-1).view(torch.uint8).tolist())
rows=[]; bad=[]
from array import array
from collections import Counter
assert sys.byteorder=='little'
W={0:4,5:2,6:2}
DATA={
 5:[0x7c03,0xfc09,0x7e03,0xfe19,0x7dff,0xfdff,0x7fff,0xffff,0,0x8000,1,0x8001,0x3ff,0x83ff,0x400,0x8400,
    0x7bff,0xfbff,0x7c00,0xfc00,0x3c00,0xbc00,0x3555,0xb001],
 6:[0x7f81,0xff81,0x7fc7,0xffd5,0,0x8000,0x7f80,0xff80],
 0:[0x7f800001,0xff800003,0x7fc00007,0xffc00777,0,0x80000000,0x7f800000,0xff800000]}
sha=lambda b:hashlib.sha256(b).hexdigest()
def words(t,w):
 a=array('H' if w==2 else 'I');a.frombytes(memoryview(t.reshape(-1).contiguous().view(torch.uint8).numpy()))
 return a
def transform(w):
 sign=(w&0x8000)<<16; exp=(w>>10)&31; frac=w&1023
 if exp==31:return sign|0x7f800000|(frac<<13)|(0x400000 if frac else 0)
 if exp:return sign|((exp+112)<<23)|(frac<<13)
 if not frac:return sign
 high=frac.bit_length()-1
 return sign|((high+103)<<23)|((frac-(1<<high))<<(23-high))
def is_nan(w):return (w&0x7c00)==0x7c00 and bool(w&0x3ff)
def hashwords(x):return sha(x.tobytes())
def tally_fail(actual,expected,stage,n,tag,order,ns,ms,src):
 if len(actual)!=len(expected):raise AssertionError('exact result length mismatch')
 count=0
 for i,(a,e) in enumerate(zip(actual,expected)):
  if a!=e:
   count+=1
   if len(bad)<512:bad.append(dict(stage=stage,count_source=n,tag=tag,order=order,index=i,expected=hex(e),observed=hex(a),source_shape=ns,mask_shape=ms,source=src))
 return count
# N in [32768,32769] is explicitly diagnostic, not part of the candidate.
for n in (7,8,9,15,16,17,31,32,33,32760,32767,32768,32769):
 cols=8 if n<=32768 else 2
 ns=[n,1]; rs=[1,cols]; ms=[1,cols]
 cutoff=n-n%8
 seed=DATA[5]
 x=array('H',(seed[i%len(seed)] for i in range(n)))
 # Use both signs and classes at repeated-axis neighbors around the last full group.
 for j in range(max(0,cutoff-4),n):
  x[j]=seed[(j-(cutoff-4))%8]
 tx=make(5,ns,x)
 tx_before=words(tx,2)
 cast=words(tx.to(torch.float32),4)
 model=array('I',(0x7fffffff if i>=cutoff and is_nan(w) else transform(w) for i,w in enumerate(x)))
 miss=tally_fail(cast,model,'to',n,5,'only',ns,ms,None)
 selected=[i for i in sorted({0,n-1,*range(max(0,cutoff-4),min(n,cutoff+7))})]
 rows.append(dict(stage='to',source_count=n,cutoff=cutoff,source_shape=ns,qualifying=n<32768,mismatches=miss,
   source_words_sha256=hashwords(x),observed_sha256=hashwords(cast),expected_sha256=hashwords(model),
   witness=[[i,hex(x[i]),hex(cast[i]),hex(model[i])] for i in selected]))
 for tag in (0,6):
  other=array('I' if tag==0 else 'H',DATA[tag][:cols]); oy=make(tag,rs,other)
  other_model=array('I',(w if tag==0 else w<<16 for w in other))
  for left in (True,False):
   mask=[int((j%2==0)==left) for j in range(cols)]
   mo=make(1,ms,mask)
   l,r=(tx,oy) if left else (oy,tx)
   t=torch.ops.aten.where.self(mo,l,r)
   assert t.dtype==torch.float32 and list(t.shape)==[n,cols] and str(t.device)=='cpu' and t._version==0
   assert t.untyped_storage()._cdata not in {l.untyped_storage()._cdata,r.untyped_storage()._cdata,mo.untyped_storage()._cdata}
   actual=words(t,4)
   expected=array('I',(model[i] if j%2==0 else other_model[j] for i in range(n) for j in range(cols)))
   m=tally_fail(actual,expected,'where',n,tag,'f16-left' if left else 'f16-right',ns,ms,None)
   assert words(tx,2)==tx_before==x and words(oy,W[tag])==other
   rows.append(dict(stage='where',source_count=n,cutoff=cutoff,source_shape=ns,other_shape=rs,mask_shape=ms,
     target_shape=[n,cols],qualifying=n<32768,mismatches=m,other_dtype=tag,f16_left=left,
     source_words_sha256=hashwords(x),mask=mask,other_input_hex=other.tobytes().hex(),
     observed_sha256=hashwords(actual),expected_sha256=hashwords(expected),
     witness=[[i,j,hex(x[i]),hex(actual[i*cols+j]),hex(expected[i*cols+j])] for i in selected for j in (0,cols-2)]))
summary=dict(rows=len(rows),qualified_rows=sum(r['qualifying'] for r in rows),postcap_rows=sum(not r['qualifying'] for r in rows),
 qualifying_mismatches=sum(r['mismatches'] for r in rows if r['qualifying']),
 diagnostic_mismatches=sum(r['mismatches'] for r in rows if not r['qualifying']),
 valid_where_words=sum(math.prod(r['target_shape']) for r in rows if r['stage']=='where' and r['qualifying']),
 valid_cast_words=sum(r['source_count'] for r in rows if r['stage']=='to' and r['qualifying']))
out.write_text(json.dumps(dict(rows=rows,summary=summary,oracle=oracle,mismatch_samples=bad),sort_keys=True,indent=2)+'\n')
print(json.dumps(summary,sort_keys=True))
