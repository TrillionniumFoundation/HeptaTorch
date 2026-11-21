"""Literal public input recipes. Arithmetic outputs come only from real Torch."""
import hashlib
import itertools
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
LEGACY_SOURCE_SHA256='5ea842f9ffcde2bab021773d61e78ea33023b64ad37bbcd7a7a52b6f112bb25b'
LEGACY_ENTRY_SHA256='7b1db2c47be5e94c23455ea49fd4a74fde9184516f1e4c400a9d46538222fc0a'
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


def case(name,operation,left,right,**extra):
    result=dict(name=name,operation=operation,left=left,right=right,**extra)
    if operation==2 and 1 in (dtype(left),dtype(right)):result['expected_error']='bool_subtraction'
    return result


def legacy_controls():
    for half in (5,6):
        for other in (0,1,2,4,5,6):
            for operation in (1,2):
                yield (f'half-binary-left-{half}-{other}-{operation}',operation,half,other)
                if half!=other:yield (f'half-binary-right-{half}-{other}-{operation}',operation,other,half)


LEGACY_CONTROLS=tuple(legacy_controls())
SUCCESSORS=tuple(row for row in LEGACY_CONTROLS if not (row[1]==2 and 1 in row[2:]))
BOOL_REFUSALS=tuple(row for row in LEGACY_CONTROLS if row not in SUCCESSORS)
MUL_CONTROL_SUCCESSORS=tuple((f'promotion-{half}-{half}-3:operation-{operation}',operation,half) for half in (5,6) for operation in (1,2))
assert len(LEGACY_CONTROLS)==44 and len(SUCCESSORS)==40 and len(BOOL_REFUSALS)==4 and len(MUL_CONTROL_SUCCESSORS)==4


def recipes():
    pairs=[(a,b) for a in DTYPES for b in DTYPES if a in (5,6) or b in (5,6)]
    geometries=[([],[]),([],[3]),([3],[]),([1],[1]),([2,3],[2,3]),([2,1,3],[1,4,1]),([0,3],[1,3])]
    for operation in (1,2):
        for a,b in pairs:
            for k,(left,right) in enumerate(geometries):
                yield case(f'promotion-{operation}-{a}-{b}-{k}',operation,input_spec(a,left),input_spec(b,right))
        layouts=[((2,3),(3,1),(2,3),(1,2)),((2,3),(1,2),(2,3),(3,1)),
                 ((2,3),(7,2),(2,3),(1,2)),((2,3),(1,2),(2,3),(7,2)),
                 ((2,1,3),(5,99,1),(1,4,1),(0,2,0)),
                 ((2,3,2,2),(12,1,6,3),(2,3,2,2),(12,4,2,1)),
                 ((2,3,2,2),(12,4,2,1),(2,3,2,2),(12,1,6,3)),
                 ((0,3),(3,1),(1,3),(3,1))]
        for a,b in [(5,5),(6,6),(5,6),(6,5),(5,0),(0,6),(5,4),(4,6),(2,5),(6,2)]:
            for k,(sh,st,rsh,rst) in enumerate(layouts):
                yield case(f'layout-{operation}-{a}-{b}-{k}',operation,layout_input(a,sh,st,2),layout_input(b,rsh,rst,1))
        for a,b in pairs:
            if a not in NANS or b not in NANS:continue
            for shape in ([],[1],[4],[7],[12],[15],[16],[17],[31],[32],[33],[65]):
                yield case(f'paired-nan-{operation}-{a}-{b}-{shape}',operation,
                    input_spec(a,shape,words=NANS[a]),input_spec(b,shape,words=list(reversed(NANS[b]))))
            for left,right in (([33],[]),([],[33])):
                yield case(f'paired-scalar-nan-{operation}-{a}-{b}-{left}',operation,
                    input_spec(a,left,words=NANS[a]),input_spec(b,right,words=list(reversed(NANS[b]))))
            for phase in (0,7):
                right=SPECIAL[b][phase:]+SPECIAL[b][:phase]
                yield case(f'exceptional-finite-{operation}-{a}-{b}-{phase}',operation,
                    input_spec(a,[64],words=SPECIAL[a]),input_spec(b,[64],words=right))
        for half in (5,6):
            for k,word in enumerate((0,0x3f800000,0xbf800000,0x3f808000,0x47c35000)):
                tensor=input_spec(half,[65536],all_words=True);scalar=input_spec(0,[],words=[word])
                for reverse in (False,True):
                    yield case(f'all-words-{operation}-{half}-{k}-{int(reverse)}',operation,
                        scalar if reverse else tensor,tensor if reverse else scalar,
                        allocation_control=(k==0 and not reverse))
            for size in (0,1,15,16,17,31,32,33,63,64,65):
                yield case(f'nan-tail-{operation}-{half}-{size}',operation,
                    input_spec(half,[size],words=NANS[half]),input_spec(half,[1],words=[0]))
            for side in (0,1):
                tensor=input_spec(half,[2,3]);scalar=input_spec(4,[1,1],words=[0x3ff0100000000000])
                yield case(f'dimensioned-singleton-cast-{operation}-{half}-{side}',operation,
                    scalar if side else tensor,tensor if side else scalar)
            for other in (0,1,2,4,half):
                yield case(f'broadcast-error-{operation}-{half}-{other}',operation,
                    input_spec(half,[2,3]),input_spec(other,[2,2]),expected_error='broadcast')
            for k,(shape,strides) in enumerate((([2,3],[1,2]),([2,3,11],[1,22,2]),([2,3,7],[21,1,3]),([2,3,2,2],[12,1,6,3]))):
                yield case(f'reachable-raw-nan-layout-{operation}-{half}-{k}',operation,
                    layout_input(half,shape,strides,2,NANS[half]),layout_input(half,shape,strides,1,list(reversed(NANS[half]))))
            same=input_spec(half,[33],words=NANS[half]+SPECIAL[half])
            yield case(f'same-handle-{operation}-{half}',operation,same,same,same_handle=True)
    for label,operation,left,right in LEGACY_CONTROLS:
        yield case('historical-'+label,operation,input_spec(left,[2],words=LEGACY_WORDS[left]),
            input_spec(right,[2],words=LEGACY_WORDS[right]),legacy_refusal_label=label,same_handle=(left==right))
    for label,operation,half in MUL_CONTROL_SUCCESSORS:
        yield case('mul-control-'+label,operation,input_spec(half,[1]),input_spec(half,[1]),
            multiplication_refusal_label=label,controls=True)


ORIGINAL_CASE_SHA256='553e1a15e63bc66a60faf1276e827410165c9ed8ff0b05d6fb013421930cc6d8'
LEGACY_FE_SUCCESSORS=tuple((f'same-frame-half-FE-binary-{half}',half) for half in (5,6))
CASES=list(recipes())
assert len(CASES)==1004 and hashlib.sha256((json.dumps(CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()==ORIGINAL_CASE_SHA256
CASES.extend(case('historical-'+label,1,input_spec(half,[2],words=[1,2]),
    input_spec(0,[2],words=LEGACY_WORDS[0]),same_handle=False,legacy_fe_successor_label=label)
    for label,half in LEGACY_FE_SUCCESSORS)
assert len(CASES)==1006
CASE_SHA256=hashlib.sha256((json.dumps(CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
