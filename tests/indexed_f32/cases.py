"""Literal F32 input grids; every expected output is independently observed Torch."""
import copy
import hashlib
import json
import math
import struct

WORDS=[0x3e800000,0xbfc00000,0x40000000,0x40500000,0xc0900000,0x3f000000,0x40c00000]
SPECIAL=[0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc01234,0xffc05678,0x7f800001,0xff800001]
SEEDS=[0x3f000000,0xbf800000,0x40000000,0x3e800000]
OPS={'gather':31,'scatter_add':36,'index_add':37}

def tensor(shape,grad=False,strides=None,offset=0,count=None,words=None):
    if count is None:
        count=math.prod(shape) if strides is None else offset+1+sum((size-1)*stride for size,stride in zip(shape,strides))
    return dict(shape=list(shape),strides=strides,offset=offset,storage_count=count,requires_grad=grad,words=words or WORDS)

def raw_input(spec):
    return b''.join(spec['words'][i%len(spec['words'])].to_bytes(4,'little') for i in range(spec['storage_count']))

def seed_raw(count):return b''.join(SEEDS[i%len(SEEDS)].to_bytes(4,'little') for i in range(count))

CASES=[]
def add(name,op,a,axis,index_shape,indices,b=None,alpha=0x3e800000,**flags):
    row=dict(name=name,operation=op,opcode=OPS[op],a=a,b=b,axis=axis,index_shape=index_shape,
        indices=indices,alpha_bits=alpha,controls=False,allocation=False,same_input=False)
    row.update(flags);CASES.append(row)

for shape in ([],[3],[2,3],[2,3,2],[2,2,3,2]):
    rank=len(shape)
    for axis in range(max(1,rank)):
        for signed in (axis,axis-max(1,rank)):
            extent=shape[axis] if rank else 1
            for grad in (False,True):
                index_shape=shape.copy()
                indices=[(i*2+1)%extent for i in range(math.prod(index_shape))]
                add(f'gather/rank{rank}/axis{signed}/grad{int(grad)}','gather',tensor(shape,grad),signed,index_shape,indices)
            for op in ('scatter_add','index_add'):
                bshape=shape.copy()
                if rank:bshape[axis]=4
                indices_shape=bshape if op=='scatter_add' else [4 if rank else 1]
                indices=[(i*2+1)%extent for i in range(math.prod(indices_shape))]
                for ga,gb in ((False,False),(False,True),(True,False),(True,True)):
                    add(f'{op}/rank{rank}/axis{signed}/grad{int(ga)}{int(gb)}',op,tensor(shape,ga),signed,
                        indices_shape,indices,tensor(bshape,gb),alpha=0xbf000000 if signed<0 else 0x3e800000)

LAYOUTS=[
    ('transpose',[3,2],[1,3],0,6),
    ('gaps',[2,3],[7,2],1,13),
    ('broadcast',[2,3],[0,1],0,3),
    ('channels-last',[2,2,2,3],[12,1,6,2],0,24),
]
for name,shape,strides,offset,count in LAYOUTS:
    for op in OPS:
        for signed in (0,-1):
            axis=signed%len(shape);extent=shape[axis]
            index_shape=shape if op!='index_add' else [extent]
            indices=[(i+1)%extent for i in range(math.prod(index_shape))]
            for grad in (False,True):
                a=tensor(shape,grad,strides,offset,count)
                b=None if op=='gather' else tensor(shape,grad,strides,offset,count,words=list(reversed(WORDS)))
                add(f'{op}/{name}/axis{signed}/grad{int(grad)}',op,a,signed,index_shape,indices,b)

for op in OPS:
    for grad in (False,True):
        for shape in ([0],[2,0],[0,3],[2,0,3]):
            axis=0
            idxshape=shape if op!='index_add' else [shape[axis]]
            indices=[i%max(1,shape[axis]) for i in range(math.prod(idxshape))]
            a=tensor(shape,grad);b=None if op=='gather' else tensor(shape,grad)
            add(f'{op}/empty{shape}/grad{int(grad)}',op,a,axis,idxshape,indices,b)

add('gather/empty-relaxed-other-dimension','gather',tensor([2,3]),0,[5,0],[])
add('scatter_add/empty-relaxed-ranks','scatter_add',tensor([2,3]),0,[0,2,3,4],[],tensor([4,5]))
add('index_add/empty-nonindex-axis-invalid-unused-indices','index_add',tensor([2,0]),0,[3],[99,-1,44],tensor([3,0]))
for grad in (False,True):
    add(f'gather/raw-NaN-signedzero/grad{int(grad)}','gather',tensor([2,5],grad,words=SPECIAL),0,[2,5],[1]*5+[0]*5)
    for op in ('scatter_add','index_add'):
        add(f'{op}/identical-self-source/grad{int(grad)}',op,tensor([2,3],grad),1,
            [2,3] if op=='scatter_add' else [3],[2,0,2,1,1,0] if op=='scatter_add' else [2,0,2],
            tensor([2,3],grad),same_input=True)

for op in OPS:
    for axis in (2,-3,1<<63,(1<<63)-1):
        add(f'{op}/invalid-axis{axis}',op,tensor([2,3]),axis,[2,3] if op!='index_add' else [2],
            [0]*6 if op!='index_add' else [0,1],None if op=='gather' else tensor([2,3]))
    for bad in (-1,2,(1<<63)-1,-(1<<63)):
        add(f'{op}/invalid-index{bad}',op,tensor([2,3]),0,[2,3] if op!='index_add' else [2],
            [0]*5+[bad] if op!='index_add' else [0,bad],None if op=='gather' else tensor([2,3]))
    add(f'{op}/protocol-controls',op,tensor([2,3]),0,[2,3] if op!='index_add' else [2],
        [1,0,1,0,1,0] if op!='index_add' else [1,0],None if op=='gather' else tensor([2,3]),controls=True)
    count=65536
    add(f'{op}/allocation-recovery',op,tensor([count]),0,[count],list(range(count-1,-1,-1)),
        None if op=='gather' else tensor([count]),allocation=True)

CASE_SHA256=hashlib.sha256(json.dumps(CASES,sort_keys=True,separators=(',',':')).encode()).hexdigest()
