"""Literal public input recipes; no expected arithmetic is computed here."""
import hashlib
import itertools
import json
import math
import struct

DTYPES={0:'float32',5:'float16',6:'bfloat16'}
WIDTHS={0:4,5:2,6:2}
WORDS={
    0:[0],
    5:[0x7c01,0xfc01,0x7e12,0xfe34,0,0x8000,0x0001,0x8001,0x03ff,0x0400,0x7bff,0xfbff,0x7c00,0xfc00,0x3c00,0xbc00],
    6:[0x7f81,0xff81,0x7fc1,0xffc2,0,0x8000,0x0001,0x8001,0x007f,0x0080,0x7f7f,0xff7f,0x7f80,0xff80,0x3f80,0xbf80],
}
LEGACY_SOURCE_SHA256='5ea842f9ffcde2bab021773d61e78ea33023b64ad37bbcd7a7a52b6f112bb25b'
LEGACY_ENTRY_SHA256='7b1db2c47be5e94c23455ea49fd4a74fde9184516f1e4c400a9d46538222fc0a'
SUCCESSORS=(('unsupported-half5-neg',5),('unsupported-half6-neg',6))


def input_spec(dtype,shape,*,strides=None,offset=0,words=None,all_words=False):
    shape=list(shape);count=math.prod(shape)
    storage=count if strides is None else (offset+1+sum((size-1)*stride for size,stride in zip(shape,strides)) if count else max(1,offset))
    spec=dict(dtype=dtype,shape=shape,strides=None if strides is None else list(strides),offset=offset,storage_count=storage)
    if words is not None:spec['words']=list(words)
    if all_words:spec['all_words']=True
    return spec


def raw_input(spec):
    if spec.get('all_words'):return struct.pack('<65536H',*range(65536))
    values=spec.get('words',WORDS[spec['dtype']]);width=WIDTHS[spec['dtype']]
    return b''.join(int(values[i%len(values)]).to_bytes(width,'little') for i in range(spec['storage_count']))


def overwrite_words(spec):
    shape=spec['source']['shape'] if 'cast' in spec else spec['shape']
    values=spec.get('overwrite_words')
    return [] if values is None else [values[i%len(values)] for i in range(math.prod(shape))]


def dense_strides(shape,order):
    strides=[0]*len(shape);width=1
    for axis in order:
        strides[axis]=width;width*=shape[axis]
    return strides


def recipes():
    for dtype in (5,6):
        yield dict(name=f'all-words-{dtype}',operation='neg',input=input_spec(dtype,[65536],all_words=True),allocation_control=True)
        for size in range(71):
            yield dict(name=f'dense-tail-{dtype}-{size}',operation='neg',input=input_spec(dtype,[size]))
        for index,word in enumerate(WORDS[dtype]):
            yield dict(name=f'scalar-word-{dtype}-{index}',operation='neg',input=input_spec(dtype,[],words=[word]))
        exponent,payload=(0x7c00,1024) if dtype==5 else (0x7f80,128)
        nan_words=[sign|exponent|fraction for sign in (0,0x8000) for fraction in range(1,payload)]
        for index,start in enumerate(range(0,len(nan_words),31)):
            words=nan_words[start:start+31]
            yield dict(name=f'all-scalar-nan-payloads-{dtype}-{index}',operation='neg',input=input_spec(dtype,[len(words)],words=words))
        layouts=[([],[],2),([1],[99],3),([1,3],[99,1],2),([3,1],[1,99],2),
            ([2,1,3],[3,99,1],2),([1,2,3],[999,3,1],2),([2,3,1],[3,1,999],2),
            ([2,3,2,2],[12,1,6,3],1),([1,3,2,2],[12,1,6,3],2),
            ([2,3],[7,2],2),([2,3],[1,2],2),([2,3],[0,0],2),([2,3],[1,0],2),
            ([2,3],[1,1],2),([2,0,3],[99,17,2],2),([0,2,3],[99,1,2],2),
            ([1,1,1,1],[9,7,5,3],2)]
        for shape in ([2,3,2,2],[2,3,5],[2,3,7],[2,3,11]):
            layouts.extend((shape,dense_strides(shape,order),3) for order in itertools.permutations(range(len(shape))))
        for index,(shape,strides,offset) in enumerate(layouts):
            source=input_spec(0,shape,strides=strides,offset=offset)
            yield dict(name=f'reachable-layout-{dtype}-{index}',operation='neg',
                input=dict(cast=dtype,source=source,overwrite_words=WORDS[dtype]))
    for label,dtype in SUCCESSORS:
        words=[0x3c00,0x8000] if dtype==5 else [0x3f80,0x8000]
        yield dict(name='successor-'+label,operation='neg',legacy_refusal_label=label,
            input=input_spec(dtype,[2],words=words),controls=True)


CASES=list(recipes())
CASE_SHA256=hashlib.sha256((json.dumps(CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
