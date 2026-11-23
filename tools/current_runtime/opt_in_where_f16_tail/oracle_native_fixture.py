"""Pinned official CPU Torch fixture only; never imports the runtime."""
import hashlib,itertools,json,math,sys
from pathlib import Path
H,L,I,out=map(Path,sys.argv[1:5])
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

held_path=Path(sys.argv[5]); held_raw=held_path.read_bytes()
assert sha(held_raw)=='0190d75767f242a51f53c46ab68c0fda0700f251109d18fc7175a04652e6254a'
held={ (r['source_count'],r['other_dtype'],r['f16_left']):r for r in json.loads(held_raw)['rows'] if r['stage']=='where'}
def input_words(n):
 cutoff=n-n%8;seed=DATA[5]
 x=array('H',(seed[i%len(seed)] for i in range(n)))
 for j in range(max(0,cutoff-4),n):x[j]=seed[(j-(cutoff-4))%8]
 return x
def packed(seq,t):
 return b''.join(v.to_bytes(1 if t==1 else W[t],'little') for v in seq)
def add(label,xt,yt,ms,xs,ys,mask,x,y,key=None):
 assert {xt,yt} in ({5,0},{5,6}) and xs and ys
 tx=make(xt,xs,x);ty=make(yt,ys,y);m=make(1,ms,mask)
 actual=torch.ops.aten.where.self(m,tx,ty)
 assert str(actual.device)=='cpu' and actual.dtype==torch.float32 and actual._version==0
 assert actual.untyped_storage()._cdata not in {r.untyped_storage()._cdata for r in (m,tx,ty)}
 assert words(tx,W[xt]).tobytes()==packed(x,xt) and words(ty,W[yt]).tobytes()==packed(y,yt)
 blob=words(actual,4).tobytes();dims=list(actual.shape)
 if key is not None:
  ref=held[key]
  assert sha(blob)==ref['observed_sha256']==ref['expected_sha256'] and not ref['mismatches']
  assert sha(packed(x if xt==5 else y,5))==ref['source_words_sha256']
 rows.append(dict(label=label,dtype=0,width=4,left_dtype=xt,right_dtype=yt,shape=dims,mask_shape=ms,left_shape=xs,right_shape=ys,
   strides=list(actual.stride()),mask=mask,left=list(x),right=list(y),mask_hex=packed(mask,1).hex(),left_hex=packed(x,xt).hex(),
   right_hex=packed(y,yt).hex(),result_hex=blob.hex(),heldout_matches=key is not None))
for n in (7,8,9,15,16,17,32767):
 cols=2 if n==32767 else 8
 x=input_words(n)
 for tag in (0,6):
  y=DATA[tag][:cols]
  for left in (True,False):
   mask=[int((c%2==0)==left) for c in range(cols)]
   label=f'repeat-n{n}-{tag}-f16-'+('left' if left else 'right')
   key=(n,tag,left) if cols==8 else None
   if left:add(label,5,tag,[1,cols],[n,1],[1,cols],mask,x,y,key)
   else:add(label,tag,5,[1,cols],[1,cols],[n,1],mask,y,x,key)
for tag in (0,6):
 y=(DATA[tag]*2)[:4]
 add(f'rank4-{tag}-f16-left',5,tag,[1,2,1,3],[3,1,1,3],[1,2,2,1],[1,0,1,0,1,0],input_words(9),y)
 add(f'rank4-{tag}-f16-right',tag,5,[1,2,1,3],[1,2,2,1],[3,1,1,3],[0,1,0,1,0,1],y,input_words(9))
 add(f'empty-last-{tag}-f16-left',5,tag,[3,1],[3,0],[3,1],[1,0,1],[],DATA[tag][:3])
 add(f'empty-prefix-{tag}-f16-right',tag,5,[1,0,1],[2,0,4],[1,0,1],[],[],[])
 add(f'historical-{tag}-f16-left',5,tag,[],[2,4],[2,4],[1],list(range(8)),list(range(8)))
 add(f'historical-{tag}-f16-right',tag,5,[],[2,4],[2,4],[1],list(range(8)),list(range(8)))
out.write_text(json.dumps(dict(rows=rows,oracle=oracle,heldout_reference_sha256=sha(held_raw)),sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(rows=len(rows),heldout_matches=sum(r['heldout_matches'] for r in rows),result_words=sum(math.prod(r['shape']) for r in rows))))
