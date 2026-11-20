"""Literal input recipes only; all expected outputs are actual fixed Torch data."""
import hashlib
import json
import math
import struct
from half_mul.legacy_controls import SUCCESSORS,WORDS as LEGACY_WORDS

DTYPES={0:'float32',1:'bool',2:'int64',4:'float64',5:'float16',6:'bfloat16'}
WIDTHS={0:4,1:1,2:8,4:8,5:2,6:2}
WORDS={
 0:[0x3f800000,0xbf800000,0x00000000,0x80000000,0x3f808000,0x477fe000,0x33800000,0x00800000,0x3f7fffff,0x4b800001],
 1:[0,1,1,0,1,0],
 2:[0,1,0xffffffffffffffff,65537,16777217,9007199254740993,0x8000000000000001,255],
 4:[0x3ff0000000000000,0xbff0000000000000,0,0x8000000000000000,0x3ff0100000000000,0x40f0000000000000,0x3e70000000000000,0x3ff0000010000000],
 5:[0x3c00,0xbc00,0,0x8000,0x3c01,0x7bff,0x0001,0x0400,0x3bff,0x3555],
 6:[0x3f80,0xbf80,0,0x8000,0x3f81,0x7f7f,0x0001,0x0080,0x3f7f,0x3eab],
}

def raw_input(spec):
    if spec.get('all_words'):return struct.pack('<65536H',*range(65536))
    values=spec.get('words',WORDS[spec['dtype']]);count=spec['storage_count']
    width=WIDTHS[spec['dtype']]
    return b''.join(int(values[i%len(values)]).to_bytes(width,'little') for i in range(count))


def input_spec(dtype,shape,*,strides=None,offset=0,words=None,all_words=False):
    shape=list(shape);count=math.prod(shape)
    storage=count if strides is None else (offset+1+sum((size-1)*stride for size,stride in zip(shape,strides)) if count else max(1,offset))
    spec=dict(dtype=dtype,shape=shape,strides=None if strides is None else list(strides),offset=offset,storage_count=storage)
    if words is not None:spec['words']=list(words)
    if all_words:spec['all_words']=True
    return spec


def layout_input(dtype,shape,strides,offset=0):
    # Public Half/F64 direct as_strided is a separate missing API. The existing
    # F32 view plus preserve-format conversion supplies real reachable layouts.
    if dtype in (4,5,6):
        return dict(cast=dtype,source=input_spec(0,shape,strides=strides,offset=offset))
    return input_spec(dtype,shape,strides=strides,offset=offset)


def recipes():
    pairs=[(a,b) for a in DTYPES for b in DTYPES if a in (5,6) or b in (5,6)]
    geometries=[([],[]),([],[3]),([3],[]),([1],[1]),([2,3],[2,3]),([2,1,3],[1,4,1])]
    for a,b in pairs:
        for k,(left,right) in enumerate(geometries):
            yield dict(name=f'promotion-{a}-{b}-{k}',left=input_spec(a,left),right=input_spec(b,right),controls=a==b and a in (5,6) and k==3)
    layouts=[((2,3),(3,1),(2,3),(1,2)),((2,3),(1,2),(2,3),(3,1)),
             ((2,3),(7,2),(2,3),(1,2)),((2,3),(1,2),(2,3),(7,2)),
             ((2,1,3),(5,99,1),(1,4,1),(0,2,0)),
             ((2,3,2,2),(12,1,6,3),(2,3,2,2),(12,4,2,1)),
             ((2,3,2,2),(12,4,2,1),(2,3,2,2),(12,1,6,3)),
             ((0,3),(3,1),(1,3),(3,1))]
    for a,b in [(5,5),(6,6),(5,6),(6,5),(5,0),(0,6),(5,4),(4,6),(2,5),(6,2)]:
        for k,(sh,st,rsh,rst) in enumerate(layouts):
            yield dict(name=f'layout-{a}-{b}-{k}',left=layout_input(a,sh,st,2),right=layout_input(b,rsh,rst,1))
    nan_words={0:[0x7fc12345,0xffc54321,0x7f800001,0xff800123],
               4:[0x7ff800123456789a,0xfff823456789abcd,0x7ff0000000000001,0xfff00123456789ab],
               5:[0x7e12,0xfe34,0x7c01,0xfc23],6:[0x7fc1,0xffd2,0x7f81,0xff93]}
    for a in nan_words:
        for b in nan_words:
            if a not in (5,6) and b not in (5,6):continue
            for shape in ([],[1],[33]):
                yield dict(name=f'paired-nan-{a}-{b}-{shape}',left=input_spec(a,shape,words=nan_words[a]),right=input_spec(b,shape,words=list(reversed(nan_words[b]))))
    for half in (5,6):
        for k,word in enumerate((0x3f800000,0xbf800000,0,0x3f808000,0x47c35000)):
            tensor=input_spec(half,[65536],all_words=True);scalar=input_spec(0,[],words=[word])
            for reverse in (False,True):
                yield dict(name=f'all-words-{half}-{k}-{int(reverse)}',left=scalar if reverse else tensor,right=tensor if reverse else scalar)
        for size in (0,1,15,16,17,31,32,33,63,64,65):
            words=[0x7c01,0xfc01,0x7e55,0xfeaa,0x7c00,0xfc00] if half==5 else [0x7f81,0xff81,0x7fc5,0xffea,0x7f80,0xff80]
            yield dict(name=f'nan-tail-{half}-{size}',left=input_spec(half,[size],words=words),right=input_spec(half,[1],words=[0]))
        for side in (0,1):
            tensor=input_spec(half,[2,3]);scalar=input_spec(4,[1,1],words=[0x3ff0100000000000])
            yield dict(name=f'dimensioned-singleton-original-opmath-{half}-{side}',left=scalar if side else tensor,right=tensor if side else scalar)
        for other in (0,2,4,half):
            yield dict(name=f'broadcast-error-{half}-{other}',left=input_spec(half,[2,3]),right=input_spec(other,[2,2]),expected_error=True)

    for label,left,right in SUCCESSORS:
        yield dict(name='successor-'+label,legacy_refusal_label=label,
            left=input_spec(left,[2],words=LEGACY_WORDS[left]),
            right=input_spec(right,[2],words=LEGACY_WORDS[right]))

CASES=list(recipes())
CASE_SHA256=hashlib.sha256((json.dumps(CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
