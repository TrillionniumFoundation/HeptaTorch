"""Literal independent Half true-division input recipes; real Torch supplies all expected outputs."""
import hashlib
import json
import math
import struct

DTYPES={0:'float32',1:'bool',2:'int64',4:'float64',5:'float16',6:'bfloat16'}


WIDTHS={0:4,1:1,2:8,4:8,5:2,6:2}


WORDS={
 0:[0x3f800000,0xbf800000,0,0x80000000,0x3f808000,0x477fe000,0x33800000,0x00800000,0x3f7fffff,0x4b800001],
 1:[0,1,1,0,1,0],
 2:[0,1,0xffffffffffffffff,65537,16777217,9007199254740993,0x8000000000000001,255],
 4:[0x3ff0000000000000,0xbff0000000000000,0,0x8000000000000000,0x3ff0100000000000,0x40f0000000000000,0x3e70000000000000,0x3ff0000010000000],
 5:[0x3c00,0xbc00,0,0x8000,0x3c01,0x7bff,0x0001,0x0400,0x3bff,0x3555],
 6:[0x3f80,0xbf80,0,0x8000,0x3f81,0x7f7f,0x0001,0x0080,0x3f7f,0x3eab],
}


NANS={
 0:[0x7fc12345,0xffc54321,0x7f800001,0xff800123],
 4:[0x7ff800123456789a,0xfff823456789abcd,0x7ff0000000000001,0xfff00123456789ab],
 5:[0x7e12,0xfe34,0x7c01,0xfc23],6:[0x7fc1,0xffd2,0x7f81,0xff93],
}


SPECIAL={
 0:[0,0x80000000,0x7f800000,0xff800000,0x00000001,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,0x3f800000,0xbf800000,0x33800000,0xb3800000,0x3f808000,0xbf808000],
 4:[0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,1,0x8000000000000001,0x000fffffffffffff,0x0010000000000000,0x7fefffffffffffff,0xffefffffffffffff,0x3ff0000000000000,0xbff0000000000000,0x3e70000000000000,0xbe70000000000000,0x3ff0100000000000,0xbff0100000000000],
 5:[0,0x8000,0x7c00,0xfc00,1,0x8001,0x03ff,0x0400,0x7bff,0xfbff,0x3c00,0xbc00,0x1000,0x9000,0x3c01,0xbc01],
 6:[0,0x8000,0x7f80,0xff80,1,0x8001,0x007f,0x0080,0x7f7f,0xff7f,0x3f80,0xbf80,0x3b80,0xbb80,0x3f81,0xbf81],
}


LEGACY_WORDS={0:[0x3f800000,0x80000000],1:[1,0],2:[1,0],4:[0x3ff0000000000000,0x8000000000000000],5:[0x3c00,0x8000],6:[0x3f80,0x8000]}


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
    shape=spec['source']['shape'] if 'cast' in spec else spec['shape'];values=spec.get('overwrite_words')
    return [] if values is None else [values[i%len(values)] for i in range(math.prod(shape))]


def layout_input(dtype,shape,strides,offset=0,words=None):
    if dtype in (4,5,6):
        spec=dict(cast=dtype,source=input_spec(0,shape,strides=strides,offset=offset))
        if words is not None:spec['overwrite_words']=list(words)
        return spec
    return input_spec(dtype,shape,strides=strides,offset=offset,words=words)


def dtype(spec):
    return spec['cast'] if 'cast' in spec else spec['dtype']


def case(group,name,left,right,**extra):
    return dict(group=group,name=name,operation=4,left=left,right=right,**extra)


def legacy_controls():
    for half in (5,6):
        for other in (0,1,2,4,5,6):
            yield (f'half-binary-left-{half}-{other}-4',half,other)
            if half!=other:yield (f'half-binary-right-{half}-{other}-4',other,half)


LEGACY_CONTROLS=tuple(legacy_controls())
assert len(LEGACY_CONTROLS)==22


def recipes():
    pairs=[(a,b) for a in DTYPES for b in DTYPES if a in (5,6) or b in (5,6)]
    geometries=[([],[]),([],[3]),([3],[]),([1],[1]),([2,3],[2,3]),
                ([2,1,3],[1,4,1]),([0,3],[1,3]),([2,1,3,1],[1,4,1,2])]
    for a,b in pairs:
        for k,(left,right) in enumerate(geometries):
            yield case('promotion',f'promotion-{a}-{b}-{k}',input_spec(a,left),input_spec(b,right),
                controls=a==b and a in (5,6) and k==3)
    layouts=[((2,3),(3,1),(2,3),(1,2)),((2,3),(1,2),(2,3),(3,1)),
             ((2,3),(7,2),(2,3),(1,2)),((2,3),(1,2),(2,3),(7,2)),
             ((2,1,3),(5,99,1),(1,4,1),(0,2,0)),
             ((2,3,2,2),(12,1,6,3),(2,3,2,2),(12,4,2,1)),
             ((2,3,2,2),(12,4,2,1),(2,3,2,2),(12,1,6,3)),
             ((0,3),(3,1),(1,3),(3,1)),
             ((2,3,5),(1,30,2),(2,3,5),(0,0,0))]
    for a,b in pairs:
        for k,(sh,st,rsh,rst) in enumerate(layouts):
            yield case('layout',f'layout-{a}-{b}-{k}',layout_input(a,sh,st,2),layout_input(b,rsh,rst,3))
    for a,b in pairs:
        if a not in NANS or b not in NANS:continue
        for size in (None,1,2,3,4,7,8,12,15,16,17,31,32,33,63,65):
            shape=[] if size is None else [size]
            yield case('paired_nan',f'paired-nan-{a}-{b}-{size}',input_spec(a,shape,words=NANS[a]),
                input_spec(b,shape,words=list(reversed(NANS[b]))))
        for phase in (0,1,7,13):
            right=SPECIAL[b][phase:]+SPECIAL[b][:phase]
            for shape in ([],[1],[65]):
                yield case('special',f'special-{a}-{b}-{phase}-{shape}',input_spec(a,shape,words=SPECIAL[a]),
                    input_spec(b,shape,words=right))
    divisors=(0,0x80000000,0x3f800000,0xbf800000,0x3f808000,0x33800000,0x7f800000,0x7fc54321)
    for half in (5,6):
        for k,word in enumerate(divisors):
            tensor=input_spec(half,[65536],all_words=True);scalar=input_spec(0,[],words=[word])
            for reverse in (False,True):
                yield case('all_words',f'all-words-{half}-{k}-{int(reverse)}',
                    scalar if reverse else tensor,tensor if reverse else scalar,
                    allocation_control=k==0 and not reverse)
        for other,words in ((0,(0x3f808000,0x00000001,0x7f7fffff,0x7f812345)),
                            (2,(0,16777217,9007199254740993,0x8000000000000000)),
                            (4,(0x3ff0100000000000,0x0000000000000001,0x7fefffffffffffff,0x7ff0000012345678))):
            for k,word in enumerate(words):
                for shape,strides in (([],[]),([1],[0]),([1,1],[99,0])):
                    for reverse in (False,True):
                        # Scalar shapes keep their actual nonzero offset where
                        # the public dtype admits it. F64 is reached through
                        # the existing F32 view + copy and its raw bits restored.
                        scalar=layout_input(other,shape,strides,2,[word])
                        tensor=layout_input(half,[2,3],[1,2],1,WORDS[half])
                        yield case('scalar_precision',f'scalar-precision-{half}-{other}-{k}-{shape}-{int(reverse)}',
                            scalar if reverse else tensor,tensor if reverse else scalar)
        for size in range(71):
            for reverse in (False,True):
                vector=input_spec(half,[size],words=NANS[half]+SPECIAL[half])
                scalar=input_spec(half,[1],words=[0])
                yield case('tail',f'tail-{half}-{size}-{int(reverse)}',
                    scalar if reverse else vector,vector if reverse else scalar)
        for other in DTYPES:
            for reverse in (False,True):
                left=input_spec(half,[2,3]);right=input_spec(other,[2,2])
                yield case('broadcast_error',f'broadcast-error-{half}-{other}-{int(reverse)}',
                    right if reverse else left,left if reverse else right,expected_error='broadcast')
        for k,(shape,left_strides,right_strides) in enumerate((
            ([2,3],[1,2],[3,1]),([2,3,11],[1,22,2],[1,22,2]),
            ([2,3,7],[21,1,3],[21,1,3]),([2,3,2,2],[12,1,6,3],[12,1,6,3]))):
            yield case('raw_nan_layout',f'raw-nan-layout-{half}-{k}',
                layout_input(half,shape,left_strides,2,NANS[half]),
                layout_input(half,shape,right_strides,1,list(reversed(NANS[half]))))
        for shape in ([],[0],[33]):
            same=input_spec(half,shape,words=NANS[half]+SPECIAL[half])
            yield case('same_handle',f'same-handle-{half}-{shape}',same,same,same_handle=True)
    for label,left,right in LEGACY_CONTROLS:
        yield case('legacy_division','successor-'+label,input_spec(left,[2],words=LEGACY_WORDS[left]),
            input_spec(right,[2],words=LEGACY_WORDS[right]),legacy_refusal_label=label,same_handle=left==right)


CASES=list(recipes())
GROUP_COUNTS={name:sum(case['group']==name for case in CASES) for name in dict.fromkeys(case['group'] for case in CASES)}
assert len(CASES)==len({case['name'] for case in CASES})
CASE_COUNT=len(CASES)
CASE_SHA256=hashlib.sha256((json.dumps(CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
