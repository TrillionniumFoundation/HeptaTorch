"""Seventeen input specifications; contains no expected tensor values.

The values below are inputs, including integer rounding/cancellation probes.
All numerical outputs, cast layouts, errors, and lifetimes must come from the
fixed PyTorch CPU wheel through reference.py. No reference execution has been
performed when this source is prepared.
"""
import hashlib
import json
import struct

UPSTREAM = "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457"
MASK = (1 << 64) - 1
VALUES = (16777217, -16777216, 1, -1, 16777219, -16777218,
          (1 << 62) + 3, -(1 << 62) + 3, 0, 7, -7, 33554435, -33554432)


def patterned(count):
    return [VALUES[index % len(VALUES)] for index in range(count)]


def case(name, dtype, values, shape, strides, axes, keepdim=False, offset=0):
    return dict(name=name, dtype=dtype, values=list(values), shape=list(shape),
                strides=list(strides), offset=offset, axes=axes, keepdim=keepdim)


CASES = (
    case("bool_dense_none", "bool", [0, 1, 0, 1, 1, 0], [2, 3], [3, 1], None),
    case("bool_empty_axes_keep", "bool", [0, 1, 1, 0, 1, 0, 0, 1],
         [2, 2, 2], [4, 2, 1], [], True),
    case("i64_cast_before_reduce_rounding", "int64", [16777217, -16777216], [2], [1], None),
    case("i64_extremes_before_accumulation", "int64", [-(1 << 63), (1 << 63) - 1], [2], [1], []),
    case("scalar_i64_max_empty_axes", "int64", [(1 << 63) - 1], [], [], []),
    case("scalar_bool_negative_axis_keep", "bool", [1], [], [], [-1], True),
    case("i64_vector_inner_129", "int64", patterned(3 * 129), [3, 129], [129, 1], [1]),
    case("i64_vector_outer_17x16", "int64", patterned(17 * 16), [17, 16], [16, 1], [0], True),
    case("i64_dense_permutation_unsorted_axes", "int64", patterned(24),
         [2, 3, 4], [1, 8, 2], [-1, 0]),
    case("i64_holes_and_offset", "int64", patterned(17), [2, 3], [7, 2], [0], offset=3),
    case("i64_overlapping_view", "int64", patterned(8), [3, 4], [1, 1], [1], offset=1),
    case("bool_zero_stride", "bool", [1, 0], [5, 2], [0, 1], [0]),
    case("i64_empty_reduced_extent", "int64", [], [2, 0, 3], [3, 3, 1], [1], True),
    case("bool_empty_output", "bool", [], [0, 3], [3, 1], [1]),
    case("i64_rank4_singleton_multi_axes", "int64", patterned(24),
         [2, 1, 3, 4], [12, 12, 4, 1], [2, 0], True),
    case("i64_empty_offset_view", "int64", patterned(6), [2, 0], [2, 1], [-1], offset=3),
    case("i64_scalar_inner_7", "int64", patterned(49), [7, 7], [7, 1], [-1]),
)

# These are operation arguments, not fabricated errors. The exact exception
# classes/messages and the absence of source mutation are observed in Torch.
AXIS_CONTROLS = (
    dict(name="duplicate_normalized_axis", case_index=0, axes=[0, -2], keepdim=False),
    dict(name="positive_axis_out_of_range", case_index=0, axes=[2], keepdim=False),
    dict(name="negative_axis_out_of_range", case_index=0, axes=[-3], keepdim=True),
    dict(name="scalar_axis_out_of_range", case_index=5, axes=[1], keepdim=False),
)


def product(shape):
    result = 1
    for value in shape:
        result *= value
    return result


def raw_input(spec):
    if spec["dtype"] == "bool":
        assert all(type(x) is int and x in (0, 1) for x in spec["values"])
        return bytes(spec["values"])
    assert spec["dtype"] == "int64"
    return b"".join(struct.pack("<q", value) for value in spec["values"])


def validate():
    assert len(CASES) == 17 and len({x["name"] for x in CASES}) == 17
    assert len(AXIS_CONTROLS) == 4
    for row in CASES:
        assert len(row["shape"]) == len(row["strides"]) <= 4
        assert all(type(x) is int and x >= 0 for x in row["shape"] + row["strides"])
        assert 0 <= row["offset"] <= len(row["values"])
        assert product(row["shape"]) <= 262144
        if product(row["shape"]):
            last = row["offset"] + sum((n - 1) * s for n, s in zip(row["shape"], row["strides"]))
            assert last < len(row["values"]), row["name"]
        assert row["axes"] is None or isinstance(row["axes"], list) and len(row["axes"]) <= 4
        assert type(row["keepdim"]) is bool
        raw_input(row)
    for row in AXIS_CONTROLS:
        assert 0 <= row["case_index"] < len(CASES)
        assert all(type(x) is int for x in row["axes"])
    data = dict(cases=CASES, axis_controls=AXIS_CONTROLS)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


CASE_SHA256 = validate()
