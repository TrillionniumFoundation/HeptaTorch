"""Build only the exact opt-in logical-not integration source; leave production intact."""
import hashlib
import json
from pathlib import Path
import re

BASE_MANIFEST = "a5cf09f191db4887b4452001e0b72e75fffb2147f2476544cc3be49d8ffa2c85"
BASE_SOURCE = "aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c"
CANDIDATE_SOURCE = "e3fe2e56a812dec618716ae14f7b95b821d4dc3195caf4dd19ee54c96fe0b4ba"


def one(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one anchor {old!r}")
    return source.replace(old, new, 1)


def transform_predicate(source):
    source = one(source, b'fn predicate_bits(v0:u64,v1:u64,v2:u64)', b'fn predicate_bits(logical_raw:u64,logical_dtype:u64,logical_op:u64)')
    source = one(source, b'entry(b0)\n', b'entry(logical_not_choice)\n'
        b'block logical_not_choice(logical_raw:u64,logical_dtype:u64,logical_op:u64)\n'
        b'  typed_logical_not_opcode:u64 = const() {"value":127}\n'
        b'  typed_logical_not_selected:bool = compare.eq(logical_op,typed_logical_not_opcode)\n'
        b'  branch typed_logical_not_selected logical_not_result(logical_raw,logical_dtype) b0(logical_raw,logical_dtype,logical_op)\n'
        b'block logical_not_result(logical_raw_value:u64,logical_value_type:u64)\n'
        b'  logical_truth_value:u64 = call(logical_raw_value,logical_value_type) {"function":"logical_truth"}\n'
        b'  logical_false:u64 = const() {"value":0}\n'
        b'  logical_true:u64 = const() {"value":1}\n'
        b'  logical_is_zero:bool = compare.eq(logical_truth_value,logical_false)\n'
        b'  logical_inverted:u64 = select(logical_is_zero,logical_true,logical_false)\n'
        b'  ret(logical_inverted)\n')
    return source


def function(source, name, transform):
    rx = rb'^fn ' + name + rb'\([^\n]*\n.*?^end\n'
    values = re.findall(rx, source, re.M | re.S)
    if len(values) != 1:
        raise ValueError(f"function not unique: {name!r}")
    return one(source, values[0], transform(values[0]))


def broaden(source, tag, incoming):
    tag = tag.encode()
    before = b'  branch predicate_' + tag + b'_admitted'
    return one(source, before, b'  typed_logical_not_' + tag + b'_opcode:u64 = const() {"value":127}\n'
        b'  typed_logical_not_' + tag + b'_selected:bool = compare.eq(' + incoming + b',typed_logical_not_' + tag + b'_opcode)\n'
        b'  typed_logical_not_' + tag + b'_admitted:bool = bool.or(predicate_' + tag + b'_admitted,typed_logical_not_' + tag + b'_selected)\n'
        b'  branch typed_logical_not_' + tag + b'_admitted')


def apply_native(source):
    source = function(source, b'entry', lambda body: broaden(body, 'entry', b'v139'))
    # The inner guard uses a deliberately different preexisting name.
    source = function(source, b'predicate_command', lambda b: one(b, b'  branch predicate_admitted',
        b'  typed_logical_not_predicate_opcode:u64 = const() {"value":127}\n'
        b'  typed_logical_not_predicate_selected:bool = compare.eq(v6,typed_logical_not_predicate_opcode)\n'
        b'  typed_logical_not_predicate_admitted:bool = bool.or(predicate_admitted,typed_logical_not_predicate_selected)\n'
        b'  branch typed_logical_not_predicate_admitted'))
    return function(source, b'predicate_bits', transform_predicate)


def read_bound(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular source file: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"wrong #57 source bytes: {path}")
    return raw


def pinned_source(root):
    manifest = json.loads(read_bound(root / "manifest.json", BASE_MANIFEST))
    if manifest["runtime"]["path"] != "runtime.hepta" or manifest["runtime"]["sha256"] != BASE_SOURCE:
        raise ValueError("wrong #57 canonical runtime binding")
    units = manifest["units"]
    expected = ["rng03", *(f"runtime800_{i:02d}" for i in range(13)), "owners"]
    if [unit["name"] for unit in units] != expected or (root / "units").is_symlink():
        raise ValueError("wrong #57 partition roster")
    for unit in units:
        relative = f"units/{unit['name']}.hepta"
        if unit["path"] != relative:
            raise ValueError("wrong #57 partition path")
        read_bound(root / relative, unit["sha256"])
    return read_bound(root / "runtime.hepta", BASE_SOURCE)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.base_source_root.resolve(strict=True)
    destination = args.output.resolve()
    if args.output.is_symlink() or destination.is_relative_to(root):
        raise ValueError("write the experimental source outside the bound production tree")
    raw = apply_native(pinned_source(root))
    if hashlib.sha256(raw).hexdigest() != CANDIDATE_SOURCE:
        raise ValueError("candidate source digest differs")
    with destination.open('xb') as stream:
        stream.write(raw)


if __name__ == "__main__":
    main()
