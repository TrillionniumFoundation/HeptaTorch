"""Check packed permutation encoding against independent list insertion model."""
import itertools,json,sys
from pathlib import Path
p=Path(__file__).parent
def permutation(shape, effective):
    r=len(shape); initial=(0,0,1,6,27)[r]; bits=initial
    def cmd(a,b):
        for vec in effective:
            i,j=vec[a],vec[b]
            if i==0 or j==0:continue
            if i<j:return 1
            if i>j or shape[a]>shape[b]:return 2
        return 0
    for i in range(1,r):
        left=i; pivot=i
        while left:
            j=left-1
            a=(bits>>(j*2))&3
            b=(bits>>(pivot*2))&3
            cmp=cmd(a,b)
            if cmp==1:break
            if cmp==2:
                raw=a^b
                bits^=raw<<(2*j)^raw<<(2*pivot)
                pivot=j
            left=j
    return [3&(bits>>(2*i)) for i in range(r)], bits==initial
r=json.loads((p/'preregister.json').read_bytes())
rows=[a['prediction'] for a in r['known']+r['empty']]
for d in rows:
    assert permutation(d['shape'],d['operand_effective_strides'])==(d['order_fast_to_slow'],d['contiguous_allocation'])
print(json.dumps(dict(comparisons=len(rows),canonical=sum(x['contiguous_allocation'] for x in rows),explicit=sum(not x['contiguous_allocation'] for x in rows),HIA_executed=False)))
