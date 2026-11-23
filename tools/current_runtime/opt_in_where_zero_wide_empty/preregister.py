"""Pure model for pinned TensorIterator where output metadata; not an official oracle."""
import itertools,json,math,hashlib,sys
from pathlib import Path
P=Path('/tmp/hepta-where-zero-wide-prereg')

def canonical(shape):
    return [math.prod(max(1,d) for d in shape[i+1:]) for i in range(len(shape))]

def applied(orig,sh):
    offset=len(sh)-len(orig)
    assert offset>=0
    return [0]*offset+[0 if a==1 and b!=1 else s for a,b,s in zip(orig,sh[offset:],canonical(orig),strict=True)]

def model(mask,x,y):
    root=[mask,x,y]
    ranks=[len(v) for v in root]
    n=max(ranks)
    assert 1<=n<=4
    assert sum(not shape for shape in (x,y))==1
    axes=[set(z)-{1} for z in zip(*[[1]*(n-len(v))+v for v in root])]
    if any(len(z)>1 for z in axes):return None
    sh=[next(iter(a),1) for a in axes]
    elems=math.prod(sh)
    strides=[applied(v,sh) for v in root]  # ordered mask, x, y; promotion changes only rank-0's dtype.
    order=list(range(n-1,-1,-1))
    def compare(p,q):
        for row in strides:
            a,b=row[p],row[q]
            if not a or not b:continue
            if a!=b:return (a>b)-(a<b)
            if sh[p]>sh[q]:return 1
        return 0
    for i in range(1,n):
        pivot=i
        for j in range(i-1,-1,-1):
            c=compare(order[j],order[pivot])
            if c>0:order[j],order[pivot]=order[pivot],order[j];pivot=j
            elif c<0:break
    canonical_output=order==list(range(n-1,-1,-1))
    s=canonical(sh)
    if not canonical_output:
        stride=1
        for dim in order:
            s[dim]=stride;stride*=sh[dim]
    return dict(shape=sh,strides=s,order_fast_to_slow=order,operand_effective_strides=strides,
                elements=elems,contiguous_allocation=canonical_output,
                differs_from_canonical=s!=canonical(sh))

E=[
 ('mask-empty-r1',[0],[1]),
 ('value-empty-r1',[],[0]),
 ('same-empty-r1',[0],[0]),
 ('mask-trailing-zero',[1,0],[2,1]),
 ('value-leading-zero',[1,3],[0,3]),
 ('mask-leading-zero-with-singleton',[0,1],[1,3]),
 ('ordered-singleton-tie',[0,1,1],[1,4]),
 ('mask-zero-middle',[2,0,1],[1,1,4]),
 ('value-zero-middle',[1,1,3],[2,0,1]),
 ('two-zero-axes',[0,1,0],[1,3,1]),
 ('nonempty-mask-with-empty-value',[2,1,4],[1,0,4]),
 ('rank4-mask-extra-axis',[0,1,1,1],[1,3,4]),
 ('rank4-middle-and-last',[1,0,1,0],[2,1,3,1]),
 ('rank4-value-large-prefix',[1,1,2,0],[512,1,2,1]),
 ('rank4-scalar-mask',[],[2,1,0,4]),
 ('rank4-ambiguous',[1,1,1,0],[2,0,1,1]),
]
R=[
 ('zero-vs-two',[0],[2]),
 ('two-vs-zero',[2],[0]),
 ('earlier-zero-masks-later-error',[0,1,2],[0,3,4]),
 ('later-zero-does-not-hide-error',[2,0],[3,0]),
 ('unaligned-rank4-conflict',[0,1,2,1],[3,4,1]),
 ('shared-zero-but-wrong-nonzero-axis',[0,2,1],[3,4]),
]
RAW={0:0xff812345,2:2**63,5:0xfc39,6:0xff89}
F64=0xfff0123456789abc
rece=json.loads((P/'oracle-receipt.json').read_bytes())
frozen=json.loads((P/'expected.json').read_bytes())
prior={x['label']:x for x in rece['where'] if x['words']==0}
known=[]
for row in frozen['positive']:
    if row['label'] not in prior:continue
    a=prior[row['label']]
    args=[row[k]['shape'] for k in ('mask','x','y')]
    prediction=model(*args)
    assert prediction['shape']==a['shape'] and prediction['strides']==a['stride'] and prediction['elements']==0
    known.append(dict(label=row['label'],args=args,torch_recorded_stride=a['stride'],prediction=prediction))
assert len(known)==16
rows=[];rejects=[]
for label,m,v in E+R:
    for tag in RAW:
        for rev in (False,True):
            x,y=(v,[]) if rev else ([],v)
            prediction=model(m,x,y)
            is_valid=(label,m,v) in E
            assert (prediction is not None)==is_valid
            item=dict(label=f'{label}-tag{tag}-reverse{int(rev)}',mask_shape=m,left_shape=x,right_shape=y,
                      left_dtype=4 if rev else tag,right_dtype=tag if rev else 4,
                      scalar_word=RAW[tag],f64_words=[F64]*math.prod(v),mask_words=[1]*math.prod(m))
            if is_valid:
                assert prediction['elements']==0
                item['prediction']=prediction;rows.append(item)
            else:item['semantic_prediction']='locked Torch shape error; native status7 policy if this scope is later introduced';rejects.append(item)
assert {len(t['prediction']['shape']) for t in rows}=={1,2,3,4}
assert any(t['prediction']['differs_from_canonical'] for t in rows)
print(json.dumps(dict(parent_head='8c7f5b84cd1280d8a4868036fba88abd8d95d6aa',parent_tree='71165daa05849e3cfac262d5e58437ad07df444a',
 scope='pure metadata only; CPU Tensor where; canonical roots; scalar+F64 dim; no oracle execution',
 prior_oracle_receipt_sha256=hashlib.sha256((P/'oracle-receipt.json').read_bytes()).hexdigest(),
 pinned_tensor_iterator='https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/TensorIterator.cpp#L226-L301',
 known=known,empty=rows,shape_errors=rejects),sort_keys=True,separators=(',',':')))
