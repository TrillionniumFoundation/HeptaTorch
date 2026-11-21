"""Literal F32 cases for explicit forward geometry and unchanged AD paths."""
import hashlib
import json
import math
import struct

MASK=(1<<64)-1
EPS=struct.unpack('<I',struct.pack('<f',1e-5))[0]


def tensor(shape,requires=False,layout='contiguous',offset=0,pattern='signed'):
    shape=list(shape);stride=[];product=1
    for extent in reversed(shape):stride.insert(0,product);product*=extent
    storage_offset=0
    if layout=='contiguous':strides=None
    elif layout=='stepped':
        strides=[2*x for x in stride];storage_offset=1
    elif layout=='transposed':
        assert len(shape)>=2
        strides=stride.copy();strides[-2]=1;strides[-1]=shape[-2]
    elif layout=='channels_last':
        assert len(shape)==4
        n,c,h,w=shape;strides=[c*h*w,1,w*c,c];storage_offset=1
    else:raise ValueError(layout)
    storage_count=product if strides is None else storage_offset+sum((n-1)*s for n,s in zip(shape,strides))+1
    return dict(shape=shape,requires_grad=requires,strides=strides,offset=storage_offset,
                storage_count=storage_count,pattern=pattern,pattern_offset=offset,layout=layout)


def raw_input(spec):
    values=[]
    for i in range(spec['storage_count']):
        number=i+spec['pattern_offset']
        value=((number%17)-8)/4 if spec['pattern']=='signed' else (.5+(number%7)/4 if spec['pattern']=='variance' else (1+number%5)/4)
        values.append(struct.pack('<f',value))
    return b''.join(values)


def seed_raw(count):
    return b''.join(struct.pack('<f',((i%7)-3)/4) for i in range(count))


def catalog():
    result=[]
    def add(name,operation,inputs,args,rtol,atol,controls=False,allocation=False):
        result.append(dict(name=name,operation=operation,inputs=inputs,args=args,rtol=rtol,atol=atol,
                           controls=controls,allocation=allocation))
    conv_specs=[((1,1,3,3),(1,1,2,2),(1,1),(0,0),(1,1),1),
                ((1,2,5,5),(3,2,3,2),(1,1),(1,0),(1,1),1),
                ((1,4,5,6),(4,2,2,3),(2,1),(1,1),(1,1),2),
                ((1,2,5,5),(2,1,2,2),(1,1),(1,1),(2,2),2)]
    for ordinal,(xs,ws,stride,pad,dilation,groups) in enumerate(conv_specs):
        for layout in ('contiguous','stepped','transposed'):
            for mode,(gx,gw) in enumerate(((False,False),(True,False),(False,True),(True,True))):
                anchor=ordinal==0 and layout=='contiguous' and mode==0
                add(f'conv2d/{ordinal}/{layout}/grad-{mode}','conv2d',
                    [tensor(xs,gx,layout,2),tensor(ws,gw,layout,5)],
                    dict(stride=list(stride),padding=list(pad),dilation=list(dilation),groups=groups),
                    2e-4,2e-5,anchor,anchor)
    for padded in (False,True):
        for average,include in ((False,True),(True,True),(True,False)):
            for layout in ('contiguous','stepped','transposed'):
                for grad in (False,True):
                    anchor=not padded and average and include and layout=='contiguous' and not grad
                    add(f'pool2d/pad-{padded}/average-{average}/include-{include}/{layout}/grad-{grad}',
                        'pool2d',[tensor((1,2,5,6),grad,layout,3)],
                        dict(kernel=[3,2],stride=[2,1],padding=[1,1] if padded else [0,0],average=average,include=include),
                        1e-4,1e-6,anchor,anchor)
    shapes=[[],[5],[2,3],[2,3,4],[1,2,2,3]]
    for shape in shapes:
        rank=len(shape);axes=list(range(rank))+([-1] if rank else [0])
        for axis in axes:
            for keep in (False,True):
                for mean in (False,True):
                    for grad in (False,True):
                        anchor=shape==[2,3,4] and axis==1 and not keep and not mean and not grad
                        add(f'reduce/{shape}/axis-{axis}/keep-{keep}/mean-{mean}/grad-{grad}',
                            'reduce_axis',[tensor(shape,grad,offset=4)],dict(axis=axis&MASK,keep=keep,mean=mean),
                            1e-4,1e-6,anchor,anchor)
    for shape in ([2,3],[2,3,4],[1,2,2,3]):
        for keep in (False,True):
            for mean in (False,True):
                for grad in (False,True):
                    add(f'reduce/{shape}/stepped/last/keep-{keep}/mean-{mean}/grad-{grad}',
                        'reduce_axis',[tensor(shape,grad,'stepped',4)],dict(axis=MASK,keep=keep,mean=mean),1e-4,1e-6)
    for shape in ((2,3),(2,2,3),(1,2,2,3)):
        for affine in (False,True):
            for layout in ('contiguous','stepped','transposed'):
                for grad in (False,True):
                    inputs=[tensor(shape,grad,layout,2),tensor((3,),grad,offset=9) if affine else None,
                            tensor((3,),grad,offset=11) if affine else None]
                    add(f'layernorm/{shape}/affine-{affine}/{layout}/grad-{grad}','layernorm',inputs,
                        dict(epsilon_bits=EPS),5e-4,5e-5)
    for shape in ((2,3),(2,3,2),(2,3,2,2)):
        for affine in (False,True):
            for layout in ('contiguous','stepped','transposed'):
                for grad in (False,True):
                    inputs=[tensor(shape,grad,layout,2),tensor((3,),False,offset=1),
                            tensor((3,),False,pattern='variance'),tensor((3,),grad,offset=9) if affine else None,
                            tensor((3,),grad,offset=11) if affine else None]
                    add(f'batchnorm/{shape}/affine-{affine}/{layout}/grad-{grad}','batchnorm_eval',inputs,
                        dict(epsilon_bits=EPS),2e-4,2e-5)
    for shape in ((2,3),(2,2,3)):
        for layout in ('contiguous','stepped'):
            for mode,(gx,gp) in enumerate(((False,False),(True,False),(True,True))):
                add(f'linear/{shape}/{layout}/grad-{mode}','linear',
                    [tensor(shape,gx,layout),tensor((4,3),gp,offset=4),tensor((4,),gp,offset=9)],{},2e-4,2e-5)
    for shape in ((3,4),(1,3)):
        for layout in ('contiguous','stepped'):
            for grad in (False,True):
                add(f'cross-entropy/{shape}/{layout}/grad-{grad}','cross_entropy_probs',
                    [tensor(shape,grad,layout,3),tensor(shape,grad,layout,1,'positive')],{},1e-4,1e-6)
    # Exact output strides matter when channels-last input is also contiguous
    # because C=1 or H=W=1. Preserve the original320 rows and add both ambiguity
    # patterns plus an unambiguous channels-last control, with/without affine
    # parameters and gradients. Expected bytes/metadata come from actual Torch.
    for shape in ((2,3,1,1),(2,1,2,2),(2,3,2,2)):
        for affine in (False,True):
            for grad in (False,True):
                tail=shape[-1]
                add(f'layernorm/{shape}/affine-{affine}/channels_last/grad-{grad}',
                    'layernorm',[tensor(shape,grad,'channels_last',2),
                    tensor((tail,),grad,offset=9) if affine else None,
                    tensor((tail,),grad,offset=11) if affine else None],
                    dict(epsilon_bits=EPS),5e-4,5e-5)
                channels=shape[1]
                add(f'batchnorm/{shape}/affine-{affine}/channels_last/grad-{grad}',
                    'batchnorm_eval',[tensor(shape,grad,'channels_last',2),
                    tensor((channels,),False,offset=1),
                    tensor((channels,),False,pattern='variance'),
                    tensor((channels,),grad,offset=9) if affine else None,
                    tensor((channels,),grad,offset=11) if affine else None],
                    dict(epsilon_bits=EPS),2e-4,2e-5)
    assert len(result)==344 and len({r['name'] for r in result})==344
    assert sum(r['controls'] for r in result)==sum(r['allocation'] for r in result)==3
    return result


CASES=catalog()
CASE_SHA256=hashlib.sha256((json.dumps(CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
