"""Literal public input recipes; expected arithmetic is observed only in Torch."""
import hashlib
import itertools
import json
import math
import struct

DTYPES = {0: 'float32', 2: 'int64', 4: 'float64'}
WIDTHS = {0: 4, 2: 8, 4: 8}
WORDS = {
    0: [0],
    2: [0, 1, 2, (1 << 64)-1, (1 << 64)-2, 1 << 63, (1 << 63)+1,
        (1 << 63)-1, (1 << 63)-2, (1 << 32)-1, 1 << 32, (1 << 32)+1,
        (1 << 53)-1, 1 << 53, (1 << 53)+1],
    4: [0, 0x8000000000000000, 1, 0x8000000000000001,
        0x000fffffffffffff, 0x800fffffffffffff, 0x0010000000000000,
        0x8010000000000000, 0x3ff0000000000000, 0xbff0000000000000,
        0x7fefffffffffffff, 0xffefffffffffffff, 0x7ff0000000000000,
        0xfff0000000000000, 0x7ff0000000000001, 0xfff0000000000001,
        0x7ff8000000000001, 0xfff8000000000001,
        0x7fffffffffffffff, 0xffffffffffffffff],
}
F64_ORIGINAL_LABEL = 'unsupported-F64-neg'
F64_ORIGINAL_WORDS = [0x3ff0000000000000, 0x8000000000000000]


def input_spec(dtype, shape, *, strides=None, offset=0, words=None, grid=None):
    shape = list(shape)
    count = math.prod(shape)
    storage = count if strides is None else (
        offset+1+sum((size-1)*stride for size,stride in zip(shape,strides))
        if count else max(1,offset))
    spec = dict(dtype=dtype, shape=shape, strides=None if strides is None else list(strides),
                offset=offset, storage_count=storage)
    if words is not None: spec['words'] = list(words)
    if grid is not None: spec['grid'] = grid
    return spec


def raw_input(spec):
    kind = spec.get('grid')
    if kind == 'exponents':
        values = [sign | (exponent << 52) | fraction
                  for sign in (0, 1 << 63) for exponent in range(2048)
                  for fraction in (0, 1, 0x7ffffffffffff, 0xfffffffffffff)]
    elif kind == 'integer-boundaries':
        mask = (1 << 64)-1
        values = [(sign * (1 << bit) + delta) & mask
                  for sign in (-1, 1) for bit in range(64) for delta in (-2,-1,0,1,2)]
    elif kind == 'sha256-words':
        raw = b''.join(hashlib.sha256(b'HeptaTorch-wide-neg-literal-input-v1/' + struct.pack('<Q',i)).digest()
                       for i in range((spec['storage_count']+3)//4))
        return raw[:spec['storage_count']*8]
    elif kind is None:
        values = spec.get('words', WORDS[spec['dtype']])
    else:
        raise ValueError('unknown literal input recipe')
    return b''.join(int(values[i % len(values)]).to_bytes(WIDTHS[spec['dtype']], 'little')
                    for i in range(spec['storage_count']))


def overwrite_words(spec):
    shape = spec['source']['shape'] if 'cast' in spec else spec['shape']
    values = spec.get('overwrite_words')
    return [] if values is None else [values[i % len(values)] for i in range(math.prod(shape))]


def dense_strides(shape, order):
    result = [0]*len(shape)
    stride = 1
    for axis in order:
        result[axis] = stride
        stride *= shape[axis]
    return result


def recipes():
    for dtype in (2, 4):
        yield dict(name=f'large-bit-patterns-{dtype}', operation='neg',
                   input=input_spec(dtype,[65536],grid='sha256-words'),allocation_control=True)
        for size in range(71):
            yield dict(name=f'dense-tail-{dtype}-{size}', operation='neg',input=input_spec(dtype,[size]))
        for index,word in enumerate(WORDS[dtype]):
            yield dict(name=f'scalar-word-{dtype}-{index}',operation='neg',input=input_spec(dtype,[],words=[word]))
        layouts=[([],[],2),([1],[99],3),([1,3],[99,1],2),([3,1],[1,99],2),
                 ([2,1,3],[3,99,1],2),([1,2,3],[999,3,1],2),([2,3,1],[3,1,999],2),
                 ([2,3,2,2],[12,1,6,3],1),([1,3,2,2],[12,1,6,3],2),
                 ([2,3],[7,2],2),([2,3],[1,2],2),([2,3],[0,0],2),
                 ([2,3],[1,0],2),([2,3],[1,1],2),([2,0,3],[99,17,2],2),
                 ([0,2,3],[99,1,2],2),([1,1,1,1],[9,7,5,3],2)]
        for shape in ([2,3,2,2],[2,3,5],[2,3,7],[2,3,11]):
            layouts.extend((shape,dense_strides(shape,order),3)
                           for order in itertools.permutations(range(len(shape))))
        for index,(shape,strides,offset) in enumerate(layouts):
            if dtype == 2:
                source = input_spec(dtype,shape,strides=strides,offset=offset)
            else:
                source = dict(cast=dtype,source=input_spec(0,shape,strides=strides,offset=offset),
                              overwrite_words=WORDS[dtype])
            yield dict(name=f'reachable-layout-{dtype}-{index}',operation='neg',input=source)
    yield dict(name='i64-bit-boundaries',operation='neg',
               input=input_spec(2,[640],grid='integer-boundaries'))
    yield dict(name='f64-all-exponents',operation='neg',
               input=input_spec(4,[16384],grid='exponents'))
    for sign in (0,1 << 63):
        for bit in range(52):
            yield dict(name=f'f64-nan-scalar-sign{sign >> 63}-payload-bit{bit}',operation='neg',
                       input=input_spec(4,[],words=[sign | 0x7ff0000000000000 | (1 << bit)]))
    yield dict(name='i64-protocol-controls',operation='neg',
               input=input_spec(2,[2],words=[1,1 << 63]),controls=True)
    yield dict(name='successor-'+F64_ORIGINAL_LABEL,operation='neg',
               legacy_refusal_label=F64_ORIGINAL_LABEL,
               input=input_spec(4,[2],words=F64_ORIGINAL_WORDS),controls=True)


CASES = list(recipes())
CASE_SHA256 = hashlib.sha256((json.dumps(CASES,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
