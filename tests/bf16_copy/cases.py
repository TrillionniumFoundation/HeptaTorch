"""Literal input bits and layouts; expected values are observed from Torch."""

import hashlib
import itertools
import json
import math
import struct

WORDS=(0x7f800001,0x7fc00000,0x7fffffff,0xff800001,0xffc00000,0xffffffff,
       0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
       0x3f800000,0xbf800000,0x3f808000,0x3f818000,0x7f7fffff,0xff7fffff,
       0x7f800000,0xff800000,0x7f810000,0xff810000)


def input_words(spec):
    count=spec['input_count']
    if spec['input_kind']=='repeat':return [WORDS[i%len(WORDS)] for i in range(count)]
    if spec['input_kind']=='literal':return list(spec['words'])
    if spec['input_kind']=='upper16-grid':return [(upper<<16)|spec['lower16'] for upper in range(65536)]
    if spec['input_kind']=='xorshift32':
        result=[];value=0x453170c5
        for _ in range(count):
            value^=(value<<13)&0xffffffff;value^=value>>17;value^=(value<<5)&0xffffffff
            result.append(value)
        return result
    raise ValueError('unknown input recipe')


def raw_input(spec):
    words=input_words(spec)
    assert len(words)==spec['input_count']
    return struct.pack('<'+'I'*len(words),*words)


def inputs():
    result=[]
    def add(name,shape,strides=None,offset=0,kind='repeat',**extra):
        count=math.prod(shape) if strides is None else (offset+1+sum((d-1)*s for d,s in zip(shape,strides)) if math.prod(shape) else 0)
        result.append(dict(name=name,shape=list(shape),strides=None if strides is None else list(strides),
                           offset=offset,input_count=count,input_kind=kind,**extra))
    for size in (0,1,2,3,6,7,8,15,16,17,31,32,33,64,65):add('dense-%d'%size,[size])
    for i,word in enumerate(WORDS):add('scalar-%02d'%i,[],kind='literal',words=[word])
    for name,shape,strides,offset in (
        ('transpose',[3,2],[1,3],0),('transpose-offset',[3,2],[1,3],1),
        ('holes',[2,3],[7,2],1),('transpose-holes',[2,3],[1,4],0),
        ('overlap',[2,3],[1,1],0),('broadcast-inner',[2,3],[1,0],0),
        ('broadcast-outer',[2,3],[0,1],0),('broadcast-all',[2,3],[0,0],0),
        ('channels-last',[2,3,2,2],[12,1,6,3],0),
        ('singleton-stride',[2,1,3],[1,99,2],0),
        ('all-singleton',[1,1,1,1],[99,0,1,5],0),
        ('singleton-hole',[1,1,2],[3,0,19],0),
        ('singleton-overlap',[1,2,2],[1,3,0],1),
        ('singleton-leading',[1,1,2,2],[7,7,7,2],0),
        ('empty',[2,0,3],[8,99,2],0),('empty-transpose',[3,2,0],[1,3,3],0),
        ('scalar-view',[],[],2),('singleton-view',[1],[0],2),
        ('rank4-overlap',[2,2,2,2],[3,2,1,0],0),
        ('rank4-gaps',[2,2,2,2],[19,7,2,1],1),
    ):add(name,shape,strides,offset)
    for index,axes in enumerate(itertools.permutations(range(3))):
        shape=[2,3,2];strides=[0]*3;stride=1
        for axis in axes:strides[axis]=stride;stride*=shape[axis]
        add('rank3-permutation-%d'%index,shape,strides)
    for lower in (0,0x7fff,0x8000,0xffff):
        add('upper16-grid-%04x'%lower,[65536],kind='upper16-grid',lower16=lower)
    add('xorshift32',[65536],kind='xorshift32')
    return result


INPUTS=inputs()
CASES=[dict(spec,opcode=opcode,memory_format=mode) for spec in INPUTS for mode in (0,1) for opcode in (108,109)]
CASE_SHA256=hashlib.sha256((json.dumps(CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
