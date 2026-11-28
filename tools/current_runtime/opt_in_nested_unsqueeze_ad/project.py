"""Extend #102 unsqueeze reverse mode to one contiguous prior kind-45 shape view."""
import argparse
import hashlib
from pathlib import Path

PARENT = "f58637bb9eda972f9a335c26f286538211390ea8b09a63eaf66c2e508e48f91e"
SOURCE = "876ebdb253483b9c76cb35c6ea0ec675c9d9e438b84ce2e4abf8b15fa9e5cb80"
OLD = b'''  scope_flags:u64 = bit.or(kind,storage_offset)
  scope_full:u64 = bit.or(scope_flags,view_kind)
  ordinary_leaf:bool = compare.eq(scope_full,g0)
  is_dense:bool = compare.ne(contiguous,g0)
  has_data:bool = compare.ne(cnt,g0)
  allowed_a:bool = bool.and(ordinary_leaf,is_dense)
  allowed:bool = bool.and(allowed_a,has_data)
  branch allowed wrap(sad,nad,dad,pad,rad) grad_error(sad)
'''
NEW = b'''  no_offset:bool = compare.eq(storage_offset,g0)
  is_dense:bool = compare.ne(contiguous,g0)
  has_data:bool = compare.ne(cnt,g0)
  allowed_a:bool = bool.and(no_offset,is_dense)
  allowed:bool = bool.and(allowed_a,has_data)
  branch allowed grad_kind(sad,nad,dad,pad,rad,kind,view_kind) grad_error(sad)
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    predecessor = a.parent.read_bytes()
    if sha(predecessor) != PARENT or predecessor.count(OLD) != 1:
        raise ValueError("exact #102 spatial/AD source required")
    body = Path(__file__).with_name("nested.hepta").read_bytes()
    candidate = predecessor.replace(OLD, NEW + body, 1)
    if sha(candidate) != SOURCE:
        raise ValueError("unexpected native nested-unsqueeze projection")
    with a.output.open("xb") as f:
        f.write(candidate)
    print(SOURCE)


if __name__ == "__main__":
    main()
