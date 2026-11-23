"""Integer-raw, stdlib-only expectations; no observed native or official outcomes."""
from pathlib import Path
import hashlib,itertools,json,math,struct,sys
CAP=262144
PATTERN=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
         0x7f800000,0xff800000,0x7fc12345,0xffc23456,0x7f812345,0xff823456,
         0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff]
sha=lambda p:hashlib.sha256(p).hexdigest()
def expect(shape,source,mask):
 assert 1<=len(shape)<=4 and all(0<=x<=CAP for x in shape)
 n=math.prod(shape);assert n<=CAP
 it=iter(source); bits=iter(mask)
 i=k=0; h=hashlib.sha256(); checks=[]
 for loc in itertools.product(*(range(d) for d in shape)):
  b=next(bits);w=next(it)
  assert b in (0,1) and 0<=w<=0xffffffff
  if b:
   if k<16:checks.append(dict(index=i,coordinate=list(loc),result_index=k,word=f'{w:08x}'))
   k+=1;h.update(struct.pack('<I',w))
  i+=1
 sentinel=object(); assert next(it,sentinel) is sentinel and next(bits,sentinel) is sentinel and i==n
 return dict(shape=[k],strides=[1],dtype='float32',numel=k,raw_u32le_sha256=h.hexdigest(),beginning=checks)
def source(kind,n,seed):
 if kind=='special':return (PATTERN[(i+seed)%len(PATTERN)] for i in range(n))
 if kind=='ordinal':return (0x3f000000+(i*13+seed)%(1<<23) for i in range(n))
 raise ValueError(kind)
def mask(kind,n):
 if kind=='all':return itertools.repeat(1,n)
 if kind=='none':return itertools.repeat(0,n)
 if kind=='even':return (int(i%2==0) for i in range(n))
 if kind=='odd':return (i%2 for i in range(n))
 if kind=='edge':return (int(i==0 or i==n-1) for i in range(n))
 if kind=='interior':return (int(i%6 in (1,4,5)) for i in range(n))
 if kind=='late':return (int(i==n-1) for i in range(n))
 raise ValueError(kind)
recipe=[
 ('empty-r1',[0],'special','none'),('empty-r2',[4,0],'special','none'),
 ('empty-r3',[2,0,3],'special','none'),('empty-r4',[1,2,0,4],'special','none'),
 ('one-false',[1],'special','none'),('one-true',[1],'special','all'),
 ('all-specials-r1',[18],'special','all'),('all-specials-r2',[3,6],'special','all'),
 ('none-nonempty-r3',[2,3,4],'special','none'),
 ('even-special-r2',[3,6],'special','even'),('odd-special-r2',[3,6],'special','odd'),
 ('row-order-r2',[3,7],'ordinal','interior'),('row-order-r3',[2,3,5],'ordinal','interior'),
 ('row-order-r4',[2,2,3,5],'ordinal','interior'),('singleton-r4',[2,1,3,4],'ordinal','odd'),
 ('edge-r3',[3,5,7],'special','edge'),('last-r4',[2,2,2,2],'ordinal','late'),
 *[(f'chunk-{x}',[x],'ordinal','all') for x in (255,256,257,1023,1024,1025,32767,32768,32769)],
 ('exact-cap-sparse',[512,512],'ordinal','edge'),('exact-cap-all',[512,512],'ordinal','all')]
positive=[]
for seed,(label,shape,src,msk) in enumerate(recipe):
 n=math.prod(shape); out=expect(shape,source(src,n,seed),mask(msk,n))
 if n<=257:
  out['raw_u32_le_hex']=''.join(struct.pack('<I',w).hex() for w,b in zip(source(src,n,seed),mask(msk,n),strict=True) if b)
  assert sha(bytes.fromhex(out['raw_u32_le_hex']))==out['raw_u32le_sha256']
 positive.append(dict(label=label,input_shape=shape,count=n,recipe=dict(source=src,mask=msk,seed=seed),expected=out))
guards=[
 dict(label='input-count-overcap-even-if-output-two',value_shape=[513,512],mask_shape=[513,512],value_recipe='ordinal',mask_recipe='edge',capacity_input=262656,mathematically_selected=2,hypothesized_status=3),
 dict(label='input-count-overcap-even-if-output-zero',value_shape=[513,512],mask_shape=[513,512],value_recipe='ordinal',mask_recipe='none',capacity_input=262656,mathematically_selected=0,hypothesized_status=3),
 dict(label='same-count-shape-mismatch',value_shape=[2,3],mask_shape=[3,2],hypothesized_status=7),
 dict(label='valid-pytorch-broadcast-outside-slice',value_shape=[2,3],mask_shape=[1,3],hypothesized_status=7),
 dict(label='zero-rank-outside-slice',value_shape=[],mask_shape=[],hypothesized_status=7),
 dict(label='wrong-public-mask-dtype',mask_tag=2,hypothesized_status=7),
 dict(label='valid-ordinary-plus-positive-offset-view',hypothesized_status=7),
 dict(label='actual-f32-requires-grad-input',hypothesized_status=7),
 dict(label='stale-input-or-mask-resolved-before-access',hypothesized_status=1),
 dict(label='reserved-word-nonzero',hypothesized_status=7),
 dict(label='good-compaction-followed-by-unknown-command',hypothesized_status=4),
]
summary=dict(rows=len(positive),input_words=sum(x['count'] for x in positive),selected_words=sum(x['expected']['numel'] for x in positive),
             zero_output_rows=sum(x['expected']['numel']==0 for x in positive),guards=len(guards))
assert summary['zero_output_rows']==6
obj=dict(kind='PRE_REGISTERED_PREDICTIONS_ONLY',positive=positive,guards=guards,summary=summary)
doc=(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n').encode()
with Path(sys.argv[1]).open('xb') as out:out.write(doc)
print(json.dumps(dict(summary=summary,bytes=len(doc),sha256=sha(doc)),sort_keys=True))
