"""Literal inputs for the Bool-destination sum comparison.

This module defines inputs and operation arguments only. Expected tensor values,
geometry, versions, errors, and lifetimes are observed by reference.py in Torch.
"""
import hashlib
import json
import struct

UPSTREAM = "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457"
VERSION = "2.15.0"
MASK = (1 << 64) - 1


def case(name, dtype, values, shape, strides, axes, keepdim=False, offset=0):
    return dict(name=name, dtype=dtype, values=list(values), shape=list(shape),
                strides=list(strides), offset=offset, axes=axes, keepdim=keepdim)


CASES = (
    case("bool_dense_none", "bool", [0, 1, 0, 1, 1, 0], [2, 3], [3, 1], None),
    case("bool_dense_empty_axes_keep", "bool", [0, 1, 0, 1, 1, 0], [2, 3], [3, 1], [], True),
    case("i64_signed_cancellation", "int64", [1, -1], [2], [1], None),
    case("i64_min_wrap_cancellation", "int64", [-(1 << 63)] * 2, [2], [1], []),
    case("i64_high_byte_cancellation", "int64", [256, -256, 0, 0], [2, 2], [2, 1], [0]),
    case("i64_high_bit_and_zero", "int64", [0, 1 << 62, -(1 << 62), 0], [2, 2], [2, 1], [-1], True),
    case("bool_all_zero_multi_axes", "bool", [0] * 12, [2, 2, 3], [6, 3, 1], [2, 0], True),
    case("i64_unsorted_negative_axes", "int64", [0, 1, -1, 0, 256, 0, 0, 0, -256, 0, 0, 0],
         [2, 2, 3], [6, 3, 1], [-1, 0]),
    case("bool_unsorted_positive_axes", "bool", [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
         [2, 2, 3], [6, 3, 1], [2, 0]),
    case("scalar_i64_none", "int64", [-(1 << 63)], [], [], None),
    case("scalar_i64_empty_axes_keep", "int64", [0], [], [], [], True),
    case("scalar_bool_axis_zero", "bool", [1], [], [], [0]),
    case("scalar_bool_axis_negative_one", "bool", [0], [], [], [-1], True),
    case("bool_transposed_strides", "bool", [0, 1, 0, 0, 1, 0], [2, 3], [1, 2], [1]),
    case("i64_holes_and_offset", "int64", [91, 92, 0, 256, -256, 93, 94, 0, 0, 0, 95, 96],
         [2, 3], [5, 1], [0], offset=2),
    case("i64_overlapping_strides", "int64", [0, 1, -1, 0], [2, 3], [1, 1], [1]),
    case("bool_zero_stride", "bool", [0, 1], [3, 2], [0, 1], [0]),
    case("bool_empty_reduced_extent", "bool", [], [0, 3], [3, 1], [0]),
    case("i64_empty_output", "int64", [], [0, 3], [3, 1], [1]),
    case("bool_empty_multi_axes_keep", "bool", [], [2, 0, 3], [3, 3, 1], [0, 1], True),
    case("bool_rank_four", "bool", [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
         [2, 2, 2, 2], [8, 4, 2, 1], [3, 1], True),
    case("duplicate_normalized_axis", "int64", [1, -1, 0, 0], [2, 2], [2, 1], [0, -2]),
    case("positive_axis_out_of_range", "bool", [0, 1, 0, 0], [2, 2], [2, 1], [2]),
    case("negative_axis_out_of_range", "int64", [1, 0, 0, -1], [2, 2], [2, 1], [-3]),
    case("scalar_axis_out_of_range", "bool", [1], [], [], [1]),
    case("i64_empty_offset_view", "int64", [1, -1, 0, 256, 0, 0], [2, 0], [2, 1], [1], offset=3),
    case("bool_negative_axis", "bool", [0, 1, 0, 0, 0, 0], [2, 3], [3, 1], [-1]),
)


def raw_input(spec):
    if spec["dtype"] == "bool":
        assert all(type(x) is int and x in (0, 1) for x in spec["values"])
        return bytes(spec["values"])
    assert spec["dtype"] == "int64"
    return b"".join(struct.pack("<q", x) for x in spec["values"])


def product(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def validate():
    assert len(CASES) == 27 and len({x["name"] for x in CASES}) == len(CASES)
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
    return hashlib.sha256(json.dumps(CASES, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


CASE_SHA256 = validate()
