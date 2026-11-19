"""Literal input recipes only; expected classification values come from Torch."""

import hashlib
import json
import struct

OPERATIONS = {'isnan':120, 'isinf':121, 'isfinite':122, 'isposinf':123, 'isneginf':124}
DTYPES = {'float32':0, 'bool':1, 'int64':2, 'wrapped_float64':3, 'float64':4, 'float16':5, 'bfloat16':6}
WIDTHS = {0:4, 1:1, 2:8, 3:8, 4:8, 5:2, 6:2}
WORDS = {
    'float32':[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
               0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,
               0x7fc00000,0xffc00000,0x7f800001,0xff800001,0x7fffffff,0xffffffff],
    'float64':[0,0x8000000000000000,1,0x8000000000000001,0x000fffffffffffff,
               0x0010000000000000,0x3ff0000000000000,0xbff0000000000000,
               0x7fefffffffffffff,0xffefffffffffffff,0x7ff0000000000000,0xfff0000000000000,
               0x7ff8000000000000,0xfff8000000000000,0x7ff0000000000001,0xfff0000000000001,
               0x7fffffffffffffff,0xffffffffffffffff],
    'float16':[0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x3c00,0xbc00,0x7bff,0xfbff,
               0x7c00,0xfc00,0x7e00,0xfe00,0x7c01,0xfc01,0x7fff,0xffff],
    'bfloat16':[0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x3f80,0xbf80,0x7f7f,0xff7f,
                0x7f80,0xff80,0x7fc0,0xffc0,0x7f81,0xff81,0x7fff,0xffff],
    'int64':[0,1,2,0xffffffffffffffff,0x8000000000000000,0x7fffffffffffffff,
             0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000000,0x7f800000,
             0xff800000,0x8000,0x7c00,0x7f80],
    'bool':[0,1,0,1,1,0,1,0],
}
WORDS['wrapped_float64'] = WORDS['float64']


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def input_words(spec):
    kind=spec['input_kind'];dtype=spec['dtype']
    if kind=='literal':return spec['words']
    if kind=='all16':return range(65536)
    if kind=='f32-exponent-grid':return [sign | (exponent<<23) | mantissa for sign in (0,1<<31)
        for exponent in range(256) for mantissa in (0,1,0x3fffff,0x7fffff)]
    if kind=='f64-exponent-grid':return [sign | (exponent<<52) | mantissa for sign in (0,1<<63)
        for exponent in range(2048) for mantissa in (0,1,0x7ffffffffffff,0xfffffffffffff)]
    if kind=='repeat':
        words=WORDS[dtype];return [words[i%len(words)] for i in range(spec['input_count'])]
    raise ValueError('unknown literal input recipe')


def raw_input(spec):
    words=input_words(spec);width=WIDTHS[DTYPES[spec['dtype']]]
    return struct.pack('<'+{1:'B',2:'H',4:'I',8:'Q'}[width]*len(words),*words)


def inputs():
    result=[]
    def add(dtype,name,shape,input_count,*,strides=None,offset=0,requires_grad=False,kind='repeat',words=None):
        row=dict(name=dtype+'/'+name,dtype=dtype,native_dtype=DTYPES[dtype],shape=list(shape),
                 input_kind=kind,input_count=input_count,requires_grad=requires_grad,
                 strides=None if strides is None else list(strides),offset=offset)
        if words is not None:row['words']=list(words)
        result.append(row)
    for dtype in ('float32','bool','int64','float64','float16','bfloat16'):
        add(dtype,'special-bits',[len(WORDS[dtype])],len(WORDS[dtype]),kind='literal',words=WORDS[dtype])
        add(dtype,'rank4',[2,3,2,2],24)
        add(dtype,'rank0',[],1)
        add(dtype,'empty',[2,0,3],0)
        add(dtype,'all-singleton',[1,1,1,1],1)
        if dtype in ('float32','bool','int64'):
            for name,shape,strides,offset,count in [
                ('transpose',[3,2],[1,3],0,6),
                ('transpose-offset',[3,2],[1,3],1,8),
                ('holes',[2,3],[7,2],1,16),
                ('transpose-holes',[2,3],[1,4],0,10),
                ('overlap',[2,3],[1,1],0,4),
                ('zero-stride',[2,3],[0,1],0,3),
                ('channels-last',[2,3,2,2],[12,1,6,3],0,24),
                ('singleton-stride',[2,1,3],[1,99,2],0,6),
                ('all-singleton-stride',[1,1],[99,3],0,1),
                ('singleton-leading-gap',[1,1,2],[3,0,19],0,20),
                ('singleton-leading-overlap',[1,2,2],[1,3,0],1,5),
                ('empty-transpose',[3,2,0],[1,3,3],0,0),
                ('empty-holes',[2,0,3],[8,99,2],0,0),
                ('scalar-view',[],[],2,4),
            ]:add(dtype,name,shape,count,strides=strides,offset=offset)
        if dtype=='float32':
            add(dtype,'exponent-grid',[2048],2048,kind='f32-exponent-grid')
            add(dtype,'grad-leaf',[len(WORDS[dtype])],len(WORDS[dtype]),requires_grad=True)
            add(dtype,'grad-view',[2,1,3],6,strides=[1,99,2],requires_grad=True)
            add(dtype,'grad-singleton-leading-gap',[1,1,2,2],10,strides=[7,7,7,2],requires_grad=True)
        if dtype=='float64':add(dtype,'exponent-grid',[16384],16384,kind='f64-exponent-grid')
        if dtype in ('float16','bfloat16'):add(dtype,'all-65536-encodings',[65536],65536,kind='all16')
    for i,word in enumerate(WORDS['float64']):add('wrapped_float64','word-%02d'%i,[],1,kind='literal',words=[word])
    return result

INPUTS=inputs()
CASES=[dict(spec,operation=op,opcode=opcode) for spec in INPUTS for op,opcode in OPERATIONS.items()]
CASE_SHA256=hashlib.sha256(canonical(CASES)).hexdigest()
