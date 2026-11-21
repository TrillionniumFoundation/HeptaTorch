"""Literal mixed public F64 inputs; only actual Torch computes expected results."""
import hashlib
import json
import math

DTYPES={0:'float32',1:'bool',2:'int64',4:'float64'}
WIDTHS={0:4,1:1,2:8,4:8}
OPERATIONS={1:'add',2:'sub',3:'mul',4:'true_divide'}
MASK=(1<<64)-1
WORDS={
 0:[0x3f800000,0xbf800000,0,0x80000000,0x3f808000,0x477fe000,0x33800000,0x00800000,0x3f7fffff,0x4b800001],
 1:[0,1,1,0,1,0],
 2:[0,1,MASK,65537,16777217,9007199254740993,0x8000000000000000,0x7fffffffffffffff,255],
 4:[0x3ff0000000000000,0xbff0000000000000,0,0x8000000000000000,0x3ff0100000000000,0x40f0000000000000,0x3e70000000000000,0x3ff0000010000000],
}
NANS={0:[0x7fc12345,0xffc54321,0x7f800001,0xff800123,0x7fffffff,0xffffffff],
 4:[0x7ff800123456789a,0xfff823456789abcd,0x7ff0000000000001,0xfff00123456789ab,0x7fffffffffffffff,0xffffffffffffffff]}
SPECIAL={
 0:[0,0x80000000,0x7f800000,0xff800000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,0x3f800000,0xbf800000,0x33800000,0xb3800000,0x3f808000,0xbf808000],
 4:[0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,1,0x8000000000000001,0x000fffffffffffff,0x0010000000000000,0x7fefffffffffffff,0xffefffffffffffff,0x3ff0000000000000,0xbff0000000000000,0x3e70000000000000,0xbe70000000000000,0x3ff0100000000000,0xbff0100000000000],
}
LEGACY_WORDS={0:[0x3f800000,0x80000000],1:[1,0],2:[1,0],4:[0x3ff0000000000000,0x8000000000000000]}

def input_spec(dtype,shape,*,strides=None,offset=0,words=None,word_series=None):
    shape=list(shape);count=math.prod(shape)
    storage=count if strides is None else (offset+1+sum((size-1)*stride for size,stride in zip(shape,strides)) if count else max(1,offset))
    result=dict(dtype=dtype,shape=shape,strides=None if strides is None else list(strides),offset=offset,storage_count=storage)
    if words is not None:result['words']=list(words)
    if word_series is not None:result['word_series']=word_series
    return result

def integer_boundary_words():
    values={0,1,-1,-(1<<63),(1<<63)-1}
    for offset in (0,1,2,3,510,511,512,513,1023,1024,1025,1535,1536,1537,2047,2048):
        values.add((1<<63)-1-offset);values.add(-(1<<63)+offset)
    for exponent in range(0,63):
        center=1<<exponent
        offsets={-3,-2,-1,0,1,2,3}
        if exponent>=53:
            half=1<<(exponent-53)
            offsets.update((-3*half-1,-3*half,-3*half+1,-half-1,-half,-half+1,
                half-1,half,half+1,3*half-1,3*half,3*half+1))
        for offset in offsets:
            for sign in (-1,1):
                value=sign*(center+offset)
                if -(1<<63)<=value<(1<<63):values.add(value)
    return [value&MASK for value in sorted(values)]

def integer_sample_words():
    value=0x6a09e667f3bcc909;result=[]
    for unused in range(8192):
        value^=(value<<13)&MASK;value^=value>>7;value^=(value<<17)&MASK;value&=MASK
        result.append(value)
    return result

SERIES={'integer_boundaries':integer_boundary_words(),'integer_samples':integer_sample_words()}

def raw_input(spec):
    words=SERIES[spec['word_series']] if 'word_series' in spec else spec.get('words',WORDS[spec['dtype']])
    return b''.join(int(words[i%len(words)]).to_bytes(WIDTHS[spec['dtype']],'little') for i in range(spec['storage_count']))

def overwrite_words(spec):
    shape=spec['source']['shape'] if 'cast' in spec else spec['shape'];values=spec.get('overwrite_words')
    return [] if values is None else [values[i%len(values)] for i in range(math.prod(shape))]

def layout_input(dtype,shape,strides,offset=0,words=None):
    if dtype==4:
        result=dict(cast=4,source=input_spec(0,shape,strides=strides,offset=offset))
        if words is not None:result['overwrite_words']=list(words)
        return result
    return input_spec(dtype,shape,strides=strides,offset=offset,words=words)

def dtype(spec):return spec['cast'] if 'cast' in spec else spec['dtype']

def case(group,name,operation,left,right,**extra):
    result=dict(group=group,name=name,operation=operation,left=left,right=right,**extra)
    if operation==2 and 1 in (dtype(left),dtype(right)):result['expected_error']='bool_subtraction'
    return result

def legacy_controls():
    for other,name in ((0,'F32'),(1,'Bool'),(2,'I64')):
        for operation in OPERATIONS:
            for order in ('left','right'):
                yield 'mixed-%s-op%d-F64-%s'%(name,operation,order),operation,4 if order=='left' else other,other if order=='left' else 4
    yield 'same-packet-F64-public-alias-mixed-pair',1,4,0

LEGACY_CONTROLS=tuple(legacy_controls())
PAIRS=((4,0),(0,4),(4,1),(1,4),(4,2),(2,4))

def recipes():
    geometries=[([],[]),([],[3]),([3],[]),([1],[1]),([2,3],[2,3]),
        ([2,1,3],[1,4,1]),([0,3],[1,3]),([2,1,3,1],[1,4,1,2])]
    for a,b in PAIRS:
        for operation in OPERATIONS:
            for k,(left,right) in enumerate(geometries):
                yield case('promotion',f'promotion-{a}-{b}-{operation}-{k}',operation,input_spec(a,left),input_spec(b,right),
                    controls=(a,b)==(4,0) and k==3)
    layouts=[((2,3),(3,1),(2,3),(1,2)),((2,3),(1,2),(2,3),(3,1)),
        ((2,3),(7,2),(2,3),(1,2)),((2,3),(1,2),(2,3),(7,2)),
        ((2,1,3),(5,99,1),(1,4,1),(0,2,0)),
        ((2,3,2,2),(12,1,6,3),(2,3,2,2),(12,4,2,1)),
        ((2,3,2,2),(12,4,2,1),(2,3,2,2),(12,1,6,3)),
        ((0,3),(3,1),(1,3),(3,1)),((2,3,5),(1,30,2),(2,3,5),(0,0,0)),
        ((2,3),(0,0),(2,3),(3,1)),((2,3),(1,1),(2,3),(1,2)),
        ((2,1,3),(3,99,1),(1,1,1),(0,0,0))]
    for a,b in PAIRS:
        for operation in OPERATIONS:
            for k,(ls,lt,rs,rt) in enumerate(layouts):
                yield case('layout',f'layout-{a}-{b}-{operation}-{k}',operation,
                    layout_input(a,ls,lt,2),layout_input(b,rs,rt,3))
    for a,b in ((4,0),(0,4)):
        for operation in OPERATIONS:
            for size in (None,0,1,2,3,4,7,8,12,15,16,17,31,32,33,63,64,65):
                shape=[] if size is None else [size]
                yield case('paired_nan',f'paired-nan-{a}-{b}-{operation}-{size}',operation,
                    input_spec(a,shape,words=NANS[a]),input_spec(b,shape,words=list(reversed(NANS[b]))))
            for phase in (0,1,7,13):
                right=SPECIAL[b][phase:]+SPECIAL[b][:phase]
                for shape in ([],[1],[65]):
                    yield case('special',f'special-{a}-{b}-{operation}-{phase}-{shape}',operation,
                        input_spec(a,shape,words=SPECIAL[a]),input_spec(b,shape,words=right))
            for size in range(66):
                for scalar_left in (False,True):
                    left=input_spec(a,[1] if scalar_left else [size],words=NANS[a]+SPECIAL[a])
                    right=input_spec(b,[size] if scalar_left else [1],words=NANS[b]+SPECIAL[b])
                    yield case('vector_tail',f'vector-tail-{a}-{b}-{operation}-{size}-{int(scalar_left)}',operation,left,right)
            for k,(shape,ls,rs) in enumerate((([2,3],[1,2],[3,1]),([2,3,11],[1,22,2],[1,22,2]),
                    ([2,3,11],[33,11,1],[1,22,2]),([2,3],[0,0],[1,2]))):
                yield case('nan_layout',f'nan-layout-{a}-{b}-{operation}-{k}',operation,
                    layout_input(a,shape,ls,2,NANS[a]),layout_input(b,shape,rs,3,list(reversed(NANS[b]))))
    for a,b in ((4,1),(1,4),(4,2),(2,4)):
        for operation in OPERATIONS:
            for size in (None,0,1,2,3,8,17,65):
                shape=[] if size is None else [size]
                yield case('single_nan',f'single-nan-{a}-{b}-{operation}-{size}',operation,
                    input_spec(a,shape,words=NANS.get(a,WORDS[a])),input_spec(b,shape,words=NANS.get(b,WORDS[b])))
    precision_words={0:[0x3f808001,0x00000001,0x7f7fffff,0x7f812345],
        2:[16777217,9007199254740993,0x7fffffffffffffff,0x8000000000000000],
        4:[0x3ff0000010000000,0x0000000000000001,0x7fefffffffffffff,0x7ff0000012345678]}
    for a,b in ((4,0),(0,4),(4,2),(2,4)):
        for operation in OPERATIONS:
            for scalar_left in (False,True):
                d=a if scalar_left else b
                for word_index,word in enumerate(precision_words[d]):
                    for k,(shape,strides) in enumerate((([],[]),([1],[0]),([1,1],[99,0]),([2,3],[0,0]))):
                        scalar=layout_input(d,shape,strides,2,[word])
                        array=layout_input(b if scalar_left else a,[2,3],[1,2],1)
                        yield case('scalar_precision',f'scalar-precision-{a}-{b}-{operation}-{int(scalar_left)}-{word_index}-{k}',operation,
                            scalar if scalar_left else array,array if scalar_left else scalar)
    for series in ('integer_boundaries','integer_samples'):
        count=len(SERIES[series])
        for operation in OPERATIONS:
            for integer_left in (False,True):
                integer=input_spec(2,[count],word_series=series)
                for scalar in (False,True):
                    real=input_spec(4,[] if scalar else [count],words=[0x3ff0000000000000,0xbff0000000000000,0x3ff0000000000001])
                    yield case('integer_precision',f'integer-precision-{series}-{operation}-{int(integer_left)}-{int(scalar)}',operation,
                        integer if integer_left else real,real if integer_left else integer)
    for operation in OPERATIONS:
        left=input_spec(2,[32768],word_series='integer_samples')
        right=input_spec(4,[],words=[0x3ff0000000000000])
        yield case('allocation',f'allocation-{operation}',operation,left,right,allocation_control=True)
    for a,b in PAIRS:
        for operation in OPERATIONS:
            yield case('broadcast_error',f'broadcast-error-{a}-{b}-{operation}',operation,
                input_spec(a,[2,3]),input_spec(b,[2,2]),expected_error='broadcast')
    for label,operation,a,b in LEGACY_CONTROLS:
        words=LEGACY_WORDS[a] if label!='same-packet-F64-public-alias-mixed-pair' else [0x3ff0000000000000,0x4000000000000000]
        yield case('legacy_entry',label,operation,input_spec(a,[2],words=words),input_spec(b,[2],words=LEGACY_WORDS[b]),
            legacy_refusal_label=label,legacy_same_frame=label=='same-packet-F64-public-alias-mixed-pair')
    for operation in OPERATIONS:
        for shape in ([],[17]):
            yield case('retained_same_f64',f'retained-same-f64-{operation}-{shape}',operation,
                input_spec(4,shape,words=[0x3ff0000000000000,0x8000000000000000,1,0x0010000000000000]),
                input_spec(4,shape,words=[0x3ff0000000000000,0x4000000000000000,0xbff0000000000000]))

CASES=list(recipes())
CASE_COUNT=len(CASES)
GROUP_COUNTS={group:sum(row['group']==group for row in CASES) for group in dict.fromkeys(row['group'] for row in CASES)}
assert CASE_COUNT==len({row['name'] for row in CASES})
assert len(LEGACY_CONTROLS)==25
CASE_SHA256=hashlib.sha256((json.dumps(CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
