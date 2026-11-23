"""Source-only immutable same-shape CPU where.self projection; optional materialization for later build."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

NAMES = ("where_ordinary", "where_self_tensor", "where_self_command")
UNIT = "runtime800_12"
BUILDER = "9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d"
BASIS = {
    "abb362f523a09c74f3e9de6deb0327f4d1df984882d7da766bae7068dda3653a": (
        "ac64f4ef117bdeaf392598f50c9cfce98bf36530",
        "5ad66743b19201fe39c19c6c43d588d939e5c997464969833ef426cc8ac74235",
        "41bad1f2fa875305592fb7f75c1c7fb24cd1a6e2c60c2d7e9aa7a161482c286a"),
    "11343747e6c86b7776842d26c9eb60fc993cb02cddb21ca21b82ddcb9f88b5f3": (
        "809cffaee8e5b148156320245e3337bcf55fb5fd",
        "81e627001c50842453517155e3e1f8e2913ddf44f20e229375ebe2908a707d2a",
        "05b8221652bf28c29efc4f78090d5e7ad7f8773ec095b445bcc22d26e3757871"),
}

KEEP = b"""  branch ka b13(kr,ko,ks,kw,kn,ke,ki,kp,kflag,kop) b14(kr,ko,ks,kw,kn,ke,ki,kp,kflag,kop)
"""
ROUTE = b"""  where_op:u64 = const() {"value":129}
  where_selected:bool = compare.eq(kop,where_op)
  branch where_selected where_dispatch(kr,ko,ks,kw,kn,ke,ki,kp,kflag,kop) where_keep(kr,ko,ks,kw,kn,ke,ki,kp,kflag,kop,ka)
block where_keep(wkr:readbuf,wko:mutbuf,wks:mutbuf,wkw:mutbuf,wkn:u64,wke:u64,wki:u64,wkp:u64,wkflag:u64,wkop:u64,wka:bool)
  branch wka b13(wkr,wko,wks,wkw,wkn,wke,wki,wkp,wkflag,wkop) b14(wkr,wko,wks,wkw,wkn,wke,wki,wkp,wkflag,wkop)
block where_dispatch(wdr:readbuf,wdo:mutbuf,wds:mutbuf,wdw:mutbuf,wdn:u64,wde:u64,wdi:u64,wdp:u64,wdflag:u64,wdop:u64)
  where_frame_header:u64 = const() {"value":32}
  where_record_bytes:u64 = const() {"value":64}
  where_delta:u64 = int.mul.wrap(wdi,where_record_bytes)
  where_at:u64 = int.add.wrap(where_frame_header,where_delta)
  where_written:u64 = call(wdr,wdo,wds,where_at,wdop,wdp) {"function":"where_self_command"}
  jump b45(wdr,wdo,wds,wdw,wdn,wde,wdi,wdp,wdflag,wdop,where_written)
"""
EXPORT0 = b"exports(logical_out_layout,"
EXPORT1 = b"exports(" + b",".join(x.encode() for x in NAMES) + b",logical_out_layout,"
ANCHOR = b"fn logical_out_layout("

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def replace(raw, before, after):
    if raw.count(before) != 1:
        raise ValueError("projection anchor missing or ambiguous")
    return raw.replace(before, after, 1)

def transform(raw, inc, split, reverse=False):
    if not reverse and (re.search(rb'"value":\s*129\b', raw) or any(b"fn " + f.encode() + b"(" in raw for f in NAMES)):
        raise ValueError("wire 129 or symbol already present")
    pairs = [(KEEP, ROUTE), (ANCHOR, inc + ANCHOR)] + ([(EXPORT0, EXPORT1)] if split else [])
    for orig, new in reversed(pairs) if reverse else pairs:
        raw = replace(raw, new, orig) if reverse else replace(raw, orig, new)
    return raw

def load_tool(path):
    if sha(path.read_bytes()) != BUILDER:
        raise ValueError("unexpected source checker")
    spec = importlib.util.spec_from_file_location("pinned_builder", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool

def validate(source, checker, inc, dst=None):
    old_manifest_bytes = checker.read_file(source / "manifest.json")
    m = json.loads(old_manifest_bytes, object_pairs_hook=checker.unique_object)
    old_runtime, old_rows, _ = checker.check_sources(m, source, checker.read_file)
    runtime_path = m["runtime"]["path"]
    orig = {runtime_path: checker.read_file(source / runtime_path, m["runtime"]["sha256"])}
    orig.update({row["path"]: raw for row, raw, _ in old_rows})
    parent, unit_sha, manifest_sha = BASIS[sha(orig[runtime_path])]
    assert sha(old_manifest_bytes) == manifest_sha
    unit_row = next(row for row in m["units"] if row["name"] == UNIT)
    upath = unit_row["path"]
    assert sha(orig[upath]) == unit_sha
    patched = {k: transform(v, inc, k == upath) if k in {runtime_path, upath} else v for k, v in orig.items()}
    assert {k for k in orig if orig[k] != patched[k]} == {runtime_path, upath}
    for path in (runtime_path, upath):
        assert transform(patched[path], inc, path == upath, reverse=True) == orig[path]
    m["name"] = "opt_in_joint_where_same_shape_cpu"
    m.pop("expected_native_sha256")
    unit_row["sha256"] = sha(patched[upath])
    unit_row.pop("expected_object_sha256")
    runtime = checker.parse_source(patched[runtime_path])
    units = checker.parse_source(patched[upath], unit=True)
    old_runtime = checker.parse_source(old_runtime)
    old_defs = old_runtime["definitions"]
    assert set(runtime["definitions"]) - set(old_defs) == set(NAMES)
    assert set(old_defs) - set(runtime["definitions"]) == set()
    assert {k for k in old_defs if old_defs[k] != runtime["definitions"][k]} == {"entry"}
    assert all(runtime["signatures"][k] == old_runtime["signatures"][k] for k in old_defs)
    store_body = runtime["definitions"]["typed_store"]
    assert b"compare.lt(v10,v8)" in store_body and b"jump b1(v0,v4,v2,v3,v5)" in store_body
    new_body = runtime["definitions"]["where_self_tensor"]
    assert b'width:u64 = call(rt) {"function":"dtype_width"}' in new_body
    assert b'call(ts,addr,byte_width,bits) {"function":"typed_store"}' in new_body
    m["runtime"].update(sha256=sha(patched[runtime_path]), **{k: runtime[k] for k in ("functions", "blocks", "instructions")})
    def memory_bind(path, expected):
        raw = patched[path.relative_to(source).as_posix()]
        if sha(raw) != expected:
            raise ValueError("in-memory projected source mismatch")
        return raw
    checker.check_sources(m, source, memory_bind)
    if dst is not None:
        dst.mkdir()
        (dst / "units").mkdir()
        for k, v in patched.items():
            (dst / k).write_bytes(v)
        (dst / "manifest.json").write_bytes(checker.json_bytes(m))
        checker.check_sources(m, dst, checker.read_file)
    return {
        "basis_git": parent, "basis_runtime_sha256": sha(orig[runtime_path]), "basis_unit_sha256": unit_sha,
        "runtime_sha256": sha(patched[runtime_path]), "unit_sha256": sha(patched[upath]),
        "manifest_sha256": sha(checker.json_bytes(m)),
        "changed": ["entry"], "added": list(NAMES), "opcode_candidate": 129,
        "unit_counts": {k: units[k] for k in ("functions", "blocks", "instructions")},
        "unit_bytes": len(patched[upath]), "check_sources": "PASS", "reversal": "byte_exact",
        "native_or_torch_or_build_executed": False,
    }

def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--source", type=Path, required=True)
    args.add_argument("--checker", type=Path, required=True)
    args.add_argument("--materialize-later", type=Path)
    opts = args.parse_args()
    inc = Path(__file__).with_name("where.hepta.inc").read_bytes()
    print(json.dumps(validate(opts.source.resolve(strict=True), load_tool(opts.checker), inc, opts.materialize_later), sort_keys=True, indent=2))

if __name__ == "__main__":
    main()
