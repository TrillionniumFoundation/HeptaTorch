"""Input recipes only; PyTorch observes every expected result independently."""
import hashlib
import json
import struct

MASK = (1 << 64) - 1
DTYPES = {'float32': 0, 'bool': 1, 'int64': 2, 'float64': 3}
OPERATIONS = {'any': 118, 'all': 119, 'sum_bool': 115}


def recipe(name, dtype, words, shape, strides, axes, keepdim=False, offset=0, requires_grad=False):
    return dict(name=name, dtype=dtype, words=list(words), shape=list(shape), strides=list(strides),
                axes=axes, keepdim=keepdim, offset=offset, requires_grad=requires_grad)


RECIPES = (
    recipe('bool_dense_all', 'bool', [0, 1, 0, 1, 1, 1], [2, 3], [3, 1], None),
    recipe('bool_no_axes', 'bool', [0, 1, 0, 1, 1, 1], [2, 3], [3, 1], []),
    recipe('bool_no_axes_keep', 'bool', [0, 1, 0, 1, 1, 1], [2, 3], [3, 1], [], True),
    recipe('bool_axis_zero', 'bool', [0, 1, 0, 1, 1, 1], [2, 3], [3, 1], [0]),
    recipe('bool_axis_negative', 'bool', [0, 1, 0, 1, 1, 1], [2, 3], [3, 1], [-1], True),
    recipe('bool_transpose', 'bool', [0, 1, 0, 1, 1, 1], [2, 3], [1, 2], [1]),
    recipe('bool_zero_stride', 'bool', [0, 1], [3, 2], [0, 1], [0]),
    recipe('bool_empty_reduction', 'bool', [], [0, 3], [3, 1], [0]),
    recipe('bool_empty_result', 'bool', [], [0, 3], [3, 1], [1]),
    recipe('bool_empty_no_axes', 'bool', [], [0, 3], [3, 1], [], True),
    recipe('bool_empty_all_keep', 'bool', [], [2, 0, 3], [3, 3, 1], None, True),
    recipe('bool_scalar_true_zero', 'bool', [1], [], [], [0]),
    recipe('bool_scalar_false_negative', 'bool', [0], [], [], [-1], True),
    recipe('bool_rank_four', 'bool', [0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1], [2,2,2,2], [8,4,2,1], [3,1], True),
    recipe('int64_extremes', 'int64', [0, -(1 << 63), (1 << 63)-1, 0, 256, -256], [2,3], [3,1], [0]),
    recipe('int64_cancellation', 'int64', [1,-1], [2], [1], None),
    recipe('int64_unsorted_axes', 'int64', [0,1,-1,0,256,0,0,0,-256,0,0,0], [2,2,3], [6,3,1], [-1,0]),
    recipe('int64_holes_offset', 'int64', [91,92,0,256,-256,93,94,0,0,0,95,96], [2,3], [5,1], [0], offset=2),
    recipe('int64_overlap', 'int64', [0,1,-1,0], [2,3], [1,1], [1]),
    recipe('int64_scalar_empty_axes', 'int64', [-(1<<63)], [], [], [], True),
    recipe('int64_empty_offset', 'int64', [1,-1,0,256,0,0], [2,0], [2,1], [1], offset=3),
    recipe('float32_signed_zero', 'float32', [0,0x80000000], [2], [1], None),
    recipe('float32_signed_zero_no_axes', 'float32', [0,0x80000000], [2], [1], []),
    recipe('float32_special_values', 'float32', [0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc00001,0xffc00123], [2,4], [4,1], [1]),
    recipe('float32_nan_only', 'float32', [0x7fc00001,0xffc00123], [2], [1], None),
    recipe('float32_grad_all', 'float32', [0,0x80000000,0x3f800000,0xbf800000,0x7fc00001,1], [2,3], [3,1], None, requires_grad=True),
    recipe('float32_grad_no_axes', 'float32', [0,0x80000000,0x3f800000,0xbf800000,0x7fc00001,1], [2,3], [3,1], [], requires_grad=True),
    recipe('float32_grad_transpose', 'float32', [0,0x80000000,0x3f800000,0xbf800000,0x7fc00001,1], [2,3], [1,2], [-1], True, requires_grad=True),
    recipe('float32_grad_overlap', 'float32', [0,0x3f800000,0xbf800000,0], [2,3], [1,1], [1], requires_grad=True),
    recipe('float32_grad_empty', 'float32', [], [0,3], [3,1], [0], requires_grad=True),
    recipe('float32_holes_offset', 'float32', [0x7fc00001,0x7f800000,0,0x80000000,1,0xff800000,0,0,0x3f800000,0,0,0], [2,3], [5,1], [0], offset=2),
    recipe('float32_scalar_negative_zero', 'float32', [0x80000000], [], [], [0], True, requires_grad=True),
    recipe('float32_scalar_nan', 'float32', [0x7fc00123], [], [], [], True),
    recipe('duplicate_normalized_axis', 'int64', [1,-1,0,0], [2,2], [2,1], [0,-2]),
    recipe('positive_axis_out_of_range', 'bool', [0,1,0,0], [2,2], [2,1], [2]),
    recipe('negative_axis_out_of_range', 'float32', [0,0,1,0], [2,2], [2,1], [-3]),
    recipe('scalar_axis_out_of_range', 'bool', [1], [], [], [1]),
    recipe('wrapped_float64_positive_zero', 'float64', [0], [], [], None),
    recipe('wrapped_float64_negative_zero', 'float64', [0x8000000000000000], [], [], None),
    recipe('wrapped_float64_negative_zero_no_axes', 'float64', [0x8000000000000000], [], [], [], True),
    recipe('wrapped_float64_subnormal', 'float64', [1], [], [], [-1]),
    recipe('wrapped_float64_negative_subnormal', 'float64', [0x8000000000000001], [], [], [0], True),
    recipe('wrapped_float64_nan', 'float64', [0x7ff8000000000001], [], [], None),
    recipe('wrapped_float64_negative_nan', 'float64', [0xfff8000000001234], [], [], []),
    recipe('wrapped_float64_infinity', 'float64', [0x7ff0000000000000], [], [], [0]),
    recipe('wrapped_float64_negative_infinity', 'float64', [0xfff0000000000000], [], [], [-1], True),
    recipe('float32_grad_duplicate_axis', 'float32', [0x3f800000]*4, [2,2], [2,1], [0,0], requires_grad=True),
    recipe('float32_grad_positive_axis_out_of_range', 'float32', [0x3f800000]*4, [2,2], [2,1], [2], requires_grad=True),
    recipe('float32_grad_negative_axis_out_of_range', 'float32', [0x3f800000]*4, [2,2], [2,1], [-3], requires_grad=True),
)
CASES = tuple(dict(spec, operation=op, name=op + '/' + spec['name']) for spec in RECIPES for op in OPERATIONS)


def raw_input(spec):
    dtype, words = spec['dtype'], spec['words']
    if dtype == 'bool':
        return bytes(words)
    fmt = {'float32':'I', 'int64':'q', 'float64':'Q'}[dtype]
    return struct.pack('<' + fmt * len(words), *words)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


CASE_SHA256 = hashlib.sha256(canonical(CASES)).hexdigest()
assert len(CASES) == 147 and len({x['name'] for x in CASES}) == len(CASES)
