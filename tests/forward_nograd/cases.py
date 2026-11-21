"""Literal F32 matmul and normalized-exponential inputs, without host expected values."""
import hashlib
import json
import math

WORDS=[0x3e800000,0xbfc00000,0x40000000,0x40500000,0xc0900000,0x3f000000,0x40c00000]
SEEDS=[0x3f000000,0xbf800000,0x40000000,0x3e800000]
GRADS=((False,False),(False,True),(True,False),(True,True))
OPS={'matmul':32,'softmax':20,'log_softmax':21}

def tensor(shape,grad=False,strides=None,offset=0,count=None,words=None):
    if count is None:
        count=math.prod(shape) if strides is None else offset+1+sum((size-1)*stride for size,stride in zip(shape,strides))
    return dict(shape=list(shape),strides=strides,offset=offset,storage_count=count,
        requires_grad=grad,words=WORDS if words is None else words)

def raw_input(spec):
    return b''.join(spec['words'][i%len(spec['words'])].to_bytes(4,'little') for i in range(spec['storage_count']))

def seed_raw(count):
    return b''.join(SEEDS[i%len(SEEDS)].to_bytes(4,'little') for i in range(count))

CASES=[]
def add(name,op,a,b=None,axis=0,**flags):
    row=dict(name=name,operation=op,opcode=OPS[op],a=a,b=b,axis=axis,
        same_input=False,controls=False,allocation=False,native_limit=None,
        comparison='exact' if op=='matmul' else 'float32',rtol=1e-5,atol=1e-6)
    row.update(flags);CASES.append(row)

PAIRS=[([3],[3]),([2,3],[3]),([3],[3,2]),([2,3],[3,4]),
    ([2,2,3],[3]),([3],[2,3,4]),([2,2,3],[2,3,4]),
    ([2,1,2,3],[1,3,3,2]),([1,2,3],[4,3,2]),
    ([2,1,2,3],[3,2]),([2,3],[2,1,3,2]),([2,3,1],[1,4]),([3],[2,2,3,2])]
for pair,(sa,sb) in enumerate(PAIRS):
    for ga,gb in GRADS:
        add(f'matmul/dense{pair}/grad{int(ga)}{int(gb)}','matmul',tensor(sa,ga),tensor(sb,gb,words=list(reversed(WORDS))))

LAYOUT_PAIRS=[
    ('both-transposed',([2,3],[1,2],0,6),([3,2],[1,3],0,6)),
    ('both-gapped',([2,3],[7,2],1,13),([3,2],[5,2],1,14)),
    ('broadcast-left',([2,3],[0,1],0,3),([3,2],None,0,6)),
    ('broadcast-right',([2,3],None,0,6),([3,2],[1,0],0,3)),
    ('permuted-batch',([2,2,3],[3,6,1],0,12),([2,3,2],[6,1,3],0,12)),
    ('channels-last',([2,2,2,3],[12,1,6,2],0,24),([2,2,3,2],[12,1,4,2],0,24)),
    ('gapped-vector',([3],[2],1,6),([3,2],[1,3],0,6)),
]
for name,(sa,ta,oa,ca),(sb,tb,ob,cb) in LAYOUT_PAIRS:
    for ga,gb in GRADS:
        add(f'matmul/{name}/grad{int(ga)}{int(gb)}','matmul',
            tensor(sa,ga,ta,oa,ca),tensor(sb,gb,tb,ob,cb,list(reversed(WORDS))))
for grad in (False,True):
    add(f'matmul/identical-input/grad{int(grad)}','matmul',tensor([3,3],grad),tensor([3,3],grad),same_input=True)

EMPTY_PAIRS=[([0],[0]),([2,0],[0,3]),([0,3],[3,2]),([2,3],[3,0]),
    ([0,2,3],[1,3,2]),([2,0,3],[3]),([3],[0,3,2]),
    ([262144,262144,0,1],[1,1,1,0]),([0],[0,2]),([3],[3,0]),
    ([0],[2,0,2]),([0],[2,2,0,2])]
for pair,(sa,sb) in enumerate(EMPTY_PAIRS):
    for ga,gb in GRADS:
        add(f'matmul/empty{pair}/grad{int(ga)}{int(gb)}','matmul',tensor(sa,ga),tensor(sb,gb))

SPECIALS=[('signedzero',[0,0x80000000,0x3f800000,0xbf800000,0,0x80000000]),
    ('subnormal',[1,0x80000001,0x00800000,0x80800000,2,0x80000002]),
    ('infinities',[0x7f800000,0xff800000,0,0x80000000,0x3f800000,0xbf800000]),
    ('NaN-payloads',[0x7fc01234,0xffc05678,0x7f800001,0xff800001,0x3f800000,0xbf800000])]
for name,words in SPECIALS:
    for ga,gb in GRADS:
        add(f'matmul/{name}/grad{int(ga)}{int(gb)}','matmul',tensor([2,3],ga,words=words),
            tensor([3,2],gb),comparison='float32')

INVALID_PAIRS=[([],[1]),([1],[]),([],[]),([3],[4]),([2,3],[4,2]),([2,2,3],[3,3,2])]
for pair,(sa,sb) in enumerate(INVALID_PAIRS):
    for ga,gb in GRADS:
        add(f'matmul/invalid{pair}/grad{int(ga)}{int(gb)}','matmul',tensor(sa,ga),tensor(sb,gb))

# These are independently accepted by Torch but deliberately exceed explicit
# public HIA capacity. They are retained as named refusals, not Torch errors.
add('matmul/output-capacity','matmul',tensor([4097,0]),tensor([0,4097]),native_limit='output_elements',comparison='float32')
add('matmul/work-capacity','matmul',tensor([129,256]),tensor([256,513]),native_limit='multiply_accumulates',comparison='float32')

for operation in ('softmax','log_softmax'):
    for shape in ([],[1],[3],[2,3],[2,3,2],[2,2,2,3]):
        for axis in range(max(1,len(shape))):
            for signed in (axis,axis-max(1,len(shape))):
                for grad in (False,True):
                    add(f'{operation}/shape{shape}/axis{signed}/grad{int(grad)}',operation,tensor(shape,grad),axis=signed)
    for name,shape,strides,offset,count in (
        ('transpose',[3,2],[1,3],0,6),('gaps',[2,3],[7,2],1,13),
        ('broadcast',[2,3],[0,1],0,3),('channels-last',[2,2,2,3],[12,1,6,2],0,24)):
        for axis in (0,-1):
            for grad in (False,True):
                add(f'{operation}/{name}/axis{axis}/grad{int(grad)}',operation,tensor(shape,grad,strides,offset,count),axis=axis)
    for shape in ([0],[2,0],[0,3],[2,0,3]):
        for axis in range(len(shape)):
            for signed in (axis,axis-len(shape)):
                for grad in (False,True):
                    add(f'{operation}/empty{shape}/axis{signed}/grad{int(grad)}',operation,tensor(shape,grad),axis=signed)
    for name,words in SPECIALS+[
        ('large-gap',[0x447a0000,0xc47a0000,0,0x3f800000,0xbf800000,0]),
        ('all-negative-infinity',[0xff800000]),('all-positive-infinity',[0x7f800000])]:
        for axis in (0,-1):
            for grad in (False,True):
                add(f'{operation}/{name}/axis{axis}/grad{int(grad)}',operation,tensor([2,3],grad,words=words),axis=axis)
    for word in (0,0x80000000,0x7f800000,0xff800000,0x7fc01234):
        for grad in (False,True):
            add(f'{operation}/scalar-{word:08x}/grad{int(grad)}',operation,tensor([],grad,words=[word]),axis=-1)
    for axis in (2,-3,1<<63,(1<<63)-1):
        for grad in (False,True):
            add(f'{operation}/invalid-axis{axis}/grad{int(grad)}',operation,tensor([2,3],grad),axis=axis)

add('matmul/protocol-controls','matmul',tensor([2,3]),tensor([3,2]),controls=True)
add('matmul/allocation-recovery','matmul',tensor([256,1]),tensor([1,256]),allocation=True)
for operation in ('softmax','log_softmax'):
    add(operation+'/protocol-controls',operation,tensor([2,3]),axis=1,controls=True)
    add(operation+'/allocation-recovery',operation,tensor([256,256]),axis=1,allocation=True)

CASE_SHA256=hashlib.sha256(json.dumps(CASES,sort_keys=True,separators=(',',':')).encode()).hexdigest()
