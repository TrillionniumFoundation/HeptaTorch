"""Literal64-bit inputs only; no host computes expected arithmetic results."""
import hashlib
import json
import math

DTYPES={0:'float32',1:'bool',2:'int64',4:'float64'}
WIDTHS={0:4,1:1,2:8,4:8}
OPERATIONS={1:'add',2:'sub',3:'mul',4:'true_divide'}
MASK=(1<<64)-1
FRACTION=(1<<52)-1
NANS=[0x7ff800123456789a,0xfff823456789abcd,0x7ff0000000000001,
      0xfff00123456789ab,0x7fffffffffffffff,0xffffffffffffffff]
SPECIAL=[0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,
    1,0x8000000000000001,0x000fffffffffffff,0x0010000000000000,
    0x7fefffffffffffff,0xffefffffffffffff,0x3ff0000000000000,0xbff0000000000000,
    0x3e70000000000000,0xbe70000000000000,0x3ff0100000000000,0xbff0100000000000,
    0x3ff0000000000001,0x3fefffffffffffff]+NANS
WORDS={0:[0x3f800000,0xbf800000,0,0x80000000],4:SPECIAL}
LEGACY_CONTROLS=()

def sample(seed,count=8192):
    values=[];value=seed
    for unused in range(count):
        value^=(value<<13)&MASK;value^=value>>7;value^=(value<<17)&MASK;value&=MASK
        values.append(value)
    return values

def exponent_words():
    return [(sign<<63)|(exponent<<52)|fraction for exponent in range(2047)
        for fraction in (0,1,2,FRACTION-1,FRACTION) for sign in (0,1)]

SERIES={'random_left':sample(0x6a09e667f3bcc909),'random_right':sample(0xbb67ae8584caa73b),
        'finite_exponents':exponent_words()}
SERIES['subnormal']=[word&((1<<63)|FRACTION) for word in SERIES['random_left']]
SERIES['near_one']=[(word&((1<<63)|FRACTION))|((1022+(index&1))<<52)
    for index,word in enumerate(SERIES['random_right'])]

def input_spec(dtype,shape,*,strides=None,offset=0,words=None,word_series=None):
    shape=list(shape);count=math.prod(shape)
    storage=count if strides is None else (offset+1+sum((size-1)*stride for size,stride in zip(shape,strides)) if count else max(1,offset))
    value=dict(dtype=dtype,shape=shape,strides=None if strides is None else list(strides),offset=offset,storage_count=storage)
    if words is not None:value['words']=list(words)
    if word_series is not None:value['word_series']=word_series
    return value

def raw_input(spec):
    words=SERIES[spec['word_series']] if 'word_series' in spec else spec.get('words',WORDS[spec['dtype']])
    return b''.join(int(words[i%len(words)]).to_bytes(WIDTHS[spec['dtype']],'little') for i in range(spec['storage_count']))

def overwrite_words(spec):
    shape=spec['source']['shape'] if 'cast' in spec else spec['shape'];values=spec.get('overwrite_words')
    if 'overwrite_series' in spec:values=SERIES[spec['overwrite_series']]
    return [] if values is None else [values[i%len(values)] for i in range(math.prod(shape))]

def layout_input(shape,strides,offset=0,words=None):
    # Public dtype4 as_strided remains outside this arithmetic extension.
    # Existing F32 view -> F64 copy gives an actually reachable F64 layout.
    result=dict(cast=4,source=input_spec(0,shape,strides=strides,offset=offset))
    if words is not None:result['overwrite_words']=list(words)
    return result

def dtype(spec):return spec['cast'] if 'cast' in spec else spec['dtype']

def case(group,name,operation,left,right,**extra):
    return dict(group=group,name=name,operation=operation,left=left,right=right,**extra)

def recipes():
    for operation in OPERATIONS:
        for i,left in enumerate(SPECIAL):
            for j,right in enumerate(SPECIAL):
                yield case('scalar_pairs',f'scalar-pair-{operation}-{i}-{j}',operation,
                    input_spec(4,[],words=[left]),input_spec(4,[],words=[right]))
        lhs=NANS+SPECIAL[:18];rhs=NANS[3:]+NANS[:3]+list(reversed(SPECIAL[:18]))
        for count in range(66):
            for form in ('both','left_scalar','right_scalar'):
                left=[] if form=='left_scalar' else [count]
                right=[] if form=='right_scalar' else [count]
                yield case('vector_tail',f'vector-{operation}-{count}-{form}',operation,
                    input_spec(4,left,words=lhs),input_spec(4,right,words=rhs))
        for count in (0,1,4,8,9,17):
            yield case('same_handle',f'same-handle-{operation}-{count}',operation,
                input_spec(4,[count],words=NANS+SPECIAL),input_spec(4,[count],words=NANS+SPECIAL),same_handle=True)
        geometries=[([2,3],[1,2],0,[2,3],[3,1],0),([2,3],[1,2],7,[1,3],[3,1],3),
            ([3,5],[1,3],2,[3,1],[1,3],4),([3,9],[1,3],1,[1,9],[9,1],5),
            ([2,3],[3,1],5,[2,3],[1,2],2),([2,3,4],[12,1,3],0,[1,3,1],[3,1,1],0),
            ([2,3,2,4],[24,1,12,3],0,[1,3,1,4],[12,4,4,1],0),
            ([3,4],[8,2],3,[1,4],[4,1],1),([3,4],[0,1],2,[3,1],[1,1],0),
            ([2,3],[1,1],1,[2,1],[1,1],2),([0,3],[3,1],0,[1,3],[3,1],0),
            ([1,1],[7,2],3,[3,9],[1,3],4)]
        for index,(ls,lt,lo,rs,rt,ro) in enumerate(geometries):
            left=layout_input(ls,lt,lo,words=lhs);right=layout_input(rs,rt,ro,words=rhs)
            for reverse in (False,True):
                yield case('layout',f'layout-{operation}-{index}-{int(reverse)}',operation,
                    right if reverse else left,left if reverse else right)
        broadcasts=[([2,1,3],[1,4,1]),([2,1,3,1],[1,4,1,2]),([],[2,3,4]),
            ([2,3,4],[]),([1],[2,3,4]),([2,3,4],[1]),([1,1],[3,9]),([3,9],[1,1]),
            ([0,3],[1,3]),([2,0,3],[1,1,3])]
        for index,(left,right) in enumerate(broadcasts):
            yield case('broadcast',f'broadcast-{operation}-{index}',operation,
                input_spec(4,left,words=lhs),input_spec(4,right,words=rhs))
        for left_series,right_series in (('random_left','random_right'),('subnormal','near_one')):
            for reverse in (False,True):
                a,b=(right_series,left_series) if reverse else (left_series,right_series)
                yield case('raw_bits',f'raw-bits-{operation}-{a}-{b}',operation,
                    input_spec(4,[8192],word_series=a),input_spec(4,[8192],word_series=b))
        yield case('finite_exponents',f'finite-exponents-{operation}',operation,
            input_spec(4,[len(SERIES['finite_exponents'])],word_series='finite_exponents'),
            input_spec(4,[len(SERIES['finite_exponents'])],word_series='random_right'))
        yield case('controls',f'protocol-controls-{operation}',operation,
            input_spec(4,[1],words=[0x3ff0000000000000]),input_spec(4,[1],words=[0x4000000000000000]),controls=True)
        yield case('allocation',f'actual-allocation-{operation}',operation,
            input_spec(4,[65536],words=[0x3ff0000000000000]),input_spec(4,[65536],words=[0x4000000000000000]),allocation_control=True)
        yield case('broadcast_error',f'actual-broadcast-error-{operation}',operation,
            input_spec(4,[2],words=[0x3ff0000000000000]),input_spec(4,[3],words=[0x4000000000000000]),expected_error='broadcast')

ORIGINAL_CASE_COUNT=3288
ORIGINAL_CASE_SHA256='1d43f6550bb1b732004c77798caa593cde30a7b4234e63fd70cc725a8fe233c4'
ORIGINAL_CASES=list(recipes())
assert len(ORIGINAL_CASES)==ORIGINAL_CASE_COUNT
assert hashlib.sha256((json.dumps(ORIGINAL_CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()==ORIGINAL_CASE_SHA256

def singleton_recipes():
    # The v1 failed case exposed rank padding that disappears when every size
    # is one. These are new literal operands; all expected bytes come from the
    # independently executed fixed Torch reference, including every NaN bit.
    pairs={
        'singleton_rank_nan':[
            ('quiet-positive-negative',NANS[0],NANS[1]),
            ('quiet-negative-positive',NANS[1],NANS[0]),
            ('quiet-signaling',NANS[0],NANS[3]),
            ('signaling-quiet',NANS[3],NANS[0]),
            ('signaling-positive-negative',NANS[2],NANS[3]),
            ('signaling-negative-positive',NANS[3],NANS[2])],
        'singleton_rank_zero':[
            ('positive-negative',0,0x8000000000000000),
            ('negative-positive',0x8000000000000000,0),
            ('negative-negative',0x8000000000000000,0x8000000000000000),
            ('positive-positive',0,0)],
        'singleton_rank_finite':[
            ('normal-signs',0x3ff0000000000000,0xc000000000000000),
            ('subnormal',1,0x4000000000000000),
            ('largest-finite',0x7fefffffffffffff,0x4000000000000000)]}
    for operation in OPERATIONS:
        for left_rank in range(5):
            for right_rank in range(5):
                for group,values in pairs.items():
                    for name,left,right in values:
                        yield case(group,f'{group}-{operation}-{left_rank}-{right_rank}-{name}',operation,
                            input_spec(4,[1]*left_rank,words=[left]),
                            input_spec(4,[1]*right_rank,words=[right]))
                layouts=[('zero-zero',[0]*left_rank,[0]*right_rank),
                    ('zero-unit',[0]*left_rank,[1]*right_rank),
                    ('unit-zero',[1]*left_rank,[0]*right_rank),
                    ('gapped',[2+i for i in range(left_rank)],[7-i for i in range(right_rank)]),
                    ('alternating',[0 if i%2==0 else 3 for i in range(left_rank)],
                        [1 if i%2==0 else 7 for i in range(right_rank)])]
                for name,left_strides,right_strides in layouts:
                    yield case('singleton_rank_layout',f'singleton-layout-{operation}-{left_rank}-{right_rank}-{name}',operation,
                        layout_input([1]*left_rank,left_strides,3,words=[NANS[0]]),
                        layout_input([1]*right_rank,right_strides,2,words=[NANS[3]]))

CASES=ORIGINAL_CASES+list(singleton_recipes())
CASE_COUNT=len(CASES)
GROUP_COUNTS={group:sum(row['group']==group for row in CASES) for group in dict.fromkeys(row['group'] for row in CASES)}
assert len(SPECIAL)==24 and CASE_COUNT==len({row['name'] for row in CASES})
assert CASE_COUNT==5088 and CASES[:ORIGINAL_CASE_COUNT]==ORIGINAL_CASES
assert {key:GROUP_COUNTS[key] for key in ('singleton_rank_nan','singleton_rank_zero','singleton_rank_finite','singleton_rank_layout')}=={
    'singleton_rank_nan':600,'singleton_rank_zero':400,'singleton_rank_finite':300,'singleton_rank_layout':500}
assert all(dtype(row['left'])==dtype(row['right'])==4 for row in CASES)
CASE_SHA256=hashlib.sha256((json.dumps(CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
