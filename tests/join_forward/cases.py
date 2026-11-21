"""Literal F32 cat/stack recipes; expected values are independent actual Torch."""
import hashlib
import json
import math

WORDS=[0x3e800000,0xbfc00000,0x40000000,0x40500000,0xc0900000,0x3f000000,0x40c00000]
SPECIAL=[0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc01234,0xffc05678,0x7f800001,0xff800001]
SEEDS=[0x3f000000,0xbf800000,0x40000000,0x3e800000]
GRADS=((False,False),(False,True),(True,False),(True,True))
OPS={'cat':34,'stack':35}

def tensor(shape,grad=False,strides=None,offset=0,count=None,words=None):
    if count is None:
        count=math.prod(shape) if strides is None else offset+1+sum((size-1)*stride for size,stride in zip(shape,strides))
    return dict(shape=list(shape),strides=strides,offset=offset,storage_count=count,
        requires_grad=grad,words=WORDS if words is None else words)

def raw_input(spec):return b''.join(spec['words'][i%len(spec['words'])].to_bytes(4,'little') for i in range(spec['storage_count']))
def seed_raw(count):return b''.join(SEEDS[i%len(SEEDS)].to_bytes(4,'little') for i in range(count))

CASES=[]
def add(name,operation,inputs,axis,selection=None,**flags):
    row=dict(name=name,operation=operation,opcode=OPS[operation],inputs=inputs,axis=axis,
        selection=list(range(len(inputs))) if selection is None else selection,
        controls=False,allocation=False,native_limit=None)
    row.update(flags);CASES.append(row)

for operation,shapes in [('cat',([3],[2,3],[2,3,2],[2,2,3,2])),('stack',([],[3],[2,3],[2,3,2]))]:
    for shape in shapes:
        ranks=len(shape)+(operation=='stack')
        for axis in range(ranks):
            other=shape.copy()
            if operation=='cat':other[axis]+=1
            for signed in (axis,axis-ranks):
                for ga,gb in GRADS:
                    add(f'{operation}/rank{len(shape)}/axis{signed}/grad{int(ga)}{int(gb)}',operation,
                        [tensor(shape,ga),tensor(other,gb,words=list(reversed(WORDS)))],signed)

for operation in OPS:
    for count in (1,3,8):
        for mode in ('none','last','all'):
            inputs=[tensor([2,3],mode=='all' or mode=='last' and index==count-1,
                words=WORDS[index%7:]+WORDS[:index%7]) for index in range(count)]
            add(f'{operation}/list{count}/grad-{mode}',operation,inputs,1)
    for name,shape,strides,offset,count in (
        ('transpose',[3,2],[1,3],0,6),('gaps',[2,3],[7,2],1,13),
        ('broadcast',[2,3],[0,1],0,3),('permuted',[2,3,2],[1,4,2],0,12)):
        for axis in (0,-1):
            for ga,gb in GRADS:
                add(f'{operation}/{name}/axis{axis}/grad{int(ga)}{int(gb)}',operation,
                    [tensor(shape,ga,strides,offset,count),tensor(shape,gb,strides,offset,count,list(reversed(WORDS)))],axis)
    for selection in ([0,0],[0,1,0],[1,1,0,1]):
        for ga,gb in GRADS:
            # Both literal inputs remain checked even if the selection omits one.
            add(f'{operation}/alias{selection}/grad{int(ga)}{int(gb)}',operation,
                [tensor([2,3],ga),tensor([2,3],gb)],1,selection=selection)
    for axis in (0,-1):
        for ga,gb in GRADS:
            add(f'{operation}/all-special-bits/axis{axis}/grad{int(ga)}{int(gb)}',operation,
                [tensor([2,5],ga,words=SPECIAL),tensor([2,5],gb,words=list(reversed(SPECIAL)))],axis)

for mixed in (False,True):
    for axis in (0,1,-1):
        for ga,gb in GRADS:
            add(f'cat/channels-last/mixed{int(mixed)}/axis{axis}/grad{int(ga)}{int(gb)}','cat',
                [tensor([2,2,2,3],ga,[12,1,6,2],0,24),
                 tensor([2,2,2,3],gb,None if mixed else [12,1,6,2],0,24)],axis)
for grad in (False,True):
    add(f'stack/rank-capacity/grad{int(grad)}','stack',[tensor([2,2,2,2],grad),tensor([2,2,2,2],grad)],0,native_limit='output_rank')

EMPTY_CAT=[([0],[0],0),([0],[2,3],0),([2,3],[0],1),([2,0],[2,3],1),
    ([0,3],[2,3],0),([2,0],[2,0],0),([2,0],[2,0],1),([0],[2,0,3],-1)]
for number,(sa,sb,axis) in enumerate(EMPTY_CAT):
    for ga,gb in GRADS:
        add(f'cat/empty{number}/grad{int(ga)}{int(gb)}','cat',[tensor(sa,ga),tensor(sb,gb)],axis)
for shape in ([0],[2,0],[0,3],[2,0,3]):
    for axis in (0,-1):
        for ga,gb in GRADS:
            add(f'stack/empty{shape}/axis{axis}/grad{int(ga)}{int(gb)}','stack',[tensor(shape,ga),tensor(shape,gb)],axis)

for operation in OPS:
    add(operation+'/empty-list',operation,[],0)
    for axis in ((2,-3,1<<63,(1<<63)-1) if operation=='cat' else (3,-4,1<<63,(1<<63)-1)):
        for grad in (False,True):
            add(f'{operation}/invalid-axis{axis}/grad{int(grad)}',operation,[tensor([2,3],grad),tensor([2,3],grad)],axis)
    for number,(sa,sb,axis) in enumerate([([2,3],[3,3],1),([3],[1,3],0)] if operation=='cat' else [([2,3],[3,2],0),([3],[1,3],0)]):
        for ga,gb in GRADS:
            add(f'{operation}/invalid-shapes{number}/grad{int(ga)}{int(gb)}',operation,[tensor(sa,ga),tensor(sb,gb)],axis)
    add(operation+'/protocol-controls',operation,[tensor([2,3]),tensor([2,3]),tensor([2,3])],1,controls=True)
    add(operation+'/allocation-recovery',operation,[tensor([32768]),tensor([32768])],0,allocation=True)
for ga,gb in GRADS:
    add(f'cat/scalars/grad{int(ga)}{int(gb)}','cat',[tensor([],ga),tensor([],gb)],0)

CASE_SHA256=hashlib.sha256(json.dumps(CASES,sort_keys=True,separators=(',',':')).encode()).hexdigest()
