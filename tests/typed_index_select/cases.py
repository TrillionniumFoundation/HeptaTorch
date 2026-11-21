"""Literal, deterministic geometry and bit patterns; no expected outputs."""
import hashlib
import json
import math
import struct

DTYPES={0:'float32',1:'bool',2:'int64',4:'float64',5:'float16',6:'bfloat16'}
WIDTHS={0:4,1:1,2:8,4:8,5:2,6:2}
WORDS={
 0:(0,0x80000000,0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc00021,0xffc00034,0x7f800001,1,0x80000001,0x7f7fffff),
 1:(0,1,1,0,1,0),
 2:(0,1,2**63-1,2**63,2**64-1,2**53+1,2**60+3,2**64-2**60-7),
 4:(0,2**63,0x3ff0000000000000,0xbff0000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000021,0xfff8000000000034,0x7ff0000000000001,1,0x8000000000000001,0x7fefffffffffffff),
 5:(0,0x8000,0x3c00,0xbc00,0x7c00,0xfc00,0x7e21,0xfe34,0x7c01,1,0x8001,0x7bff),
 6:(0,0x8000,0x3f80,0xbf80,0x7f80,0xff80,0x7fc1,0xffc3,0x7f81,1,0x8001,0x7f7f),
}


def raw_input(spec):
    dtype=0 if spec.get('cast_from_f32') else spec['dtype']
    count=spec.get('storage_count',math.prod(spec['shape']))
    words=WORDS[dtype]
    if spec.get('requires_grad'):
        words=(0x3f800000,0x40000000,0x40400000,0x40800000)
    return b''.join(int(words[i%len(words)]).to_bytes(WIDTHS[dtype],'little') for i in range(count))


CASES=[]


def add(spec,axis,indices,*,label='',controls=False,allocation=False):
    CASES.append(dict(name=DTYPES[spec['dtype']]+'/'+label, input=spec,axis=axis,
        indices=list(indices),controls=controls,allocation=allocation))


for dtype in DTYPES:
    scalar=dict(dtype=dtype,shape=[],strides=None,offset=0)
    for axis,indices in ((0,[0]),(-1,[0]),(0,[]),(0,[0,0]),(0,[1]),(1,[0]),(-2,[0])):
        add(scalar,axis,indices,label='scalar')
    dense_shapes=([1],[12],[0],[2,3],[3,0],[0,3],[2,1,3],[2,3,2,2],[1,2,0,3])
    for shape in dense_shapes:
        spec=dict(dtype=dtype,shape=list(shape),strides=None,offset=0)
        for axis,dim in enumerate(shape):
            choices=([dim-1,0,dim-1],[]) if dim else ([],)
            for indices in choices:
                for selected in (axis,axis-len(shape)):
                    add(spec,selected,indices,label='dense')
        for axis,indices in ((0,[shape[0]]),(0,[-1]),(len(shape),[]),(-len(shape)-1,[])):
            add(spec,axis,indices,label='dense-error')
    for name,shape,strides,offset,count in (
        ('transpose',[2,3],[1,2],0,6),
        ('gaps',[2,3],[7,2],1,13),
        ('broadcast',[2,3],[0,1],0,3),
        ('channels-last',[2,3,2,2],[12,1,6,3],0,24),
        ('empty-view',[2,0,3],[11,3,1],2,7),
    ):
        spec=dict(dtype=dtype,shape=shape,strides=strides,offset=offset,storage_count=count,
                  cast_from_f32=dtype in (4,5,6))
        for axis,dim in enumerate(shape):
            indices=[dim-1,0,dim-1] if dim else []
            for selected in (axis,axis-len(shape)):
                add(spec,selected,indices,label=name)
    spec=dict(dtype=dtype,shape=[12],strides=None,offset=0)
    add(spec,0,[11,0,5,0],label='controls',controls=True)
    # Empty, special-value and multi-page copies are independent actual Torch observations.
    big=dict(dtype=dtype,shape=[4096],strides=None,offset=0)
    add(big,-1,list(range(4095,-1,-1)),label='multiple-pages')
    if dtype in (2,4):
        # The input is small; repeated output allocation must exhaust the real child limit.
        oom=dict(dtype=dtype,shape=[12],strides=None,offset=0)
        add(oom,0,[0,5,11,1]*16000,label='allocation',allocation=True)

# New scalar AD and retained F32 non-scalar AD both compare with real autograd.
for shape,axis,indices in (([],0,[0]),([],-1,[0]),([2,3],1,[2,0,2]),([2,3],-2,[1,0,1]),([2,3],1,[])):
    add(dict(dtype=0,shape=shape,strides=None,offset=0,requires_grad=True),axis,indices,label='f32-autograd')

# Preserve the exact original Bool factory and index_select request semantics.
add(dict(dtype=1,shape=[2,2],strides=None,offset=0,legacy_bool=True),0,[0,1],label='original-bool-successor')
CASE_SHA256=hashlib.sha256(json.dumps(CASES,sort_keys=True,separators=(',',':')).encode()).hexdigest()
