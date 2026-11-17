"""Apply the frozen Bool-output HIA patch; source construction only, no Tensor evaluation."""
import argparse
import hashlib
import json
import os
import re
import stat
import traceback
from pathlib import Path

PATCH_SHA256 = "3fd037205b00fd6341408b2b5a12799d87893f2fae892038de9cfcb8c0ed045b"
PARENT_SHA256 = "afd54444849ed10f7b4af53231c9db6542854b70f33e3927f8b05dfbb23ed1e3"
FUNCTION = re.compile(
    rb"^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)",
    re.M | re.S,
)
CALL = re.compile(rb'\{"function":"([A-Za-z_][A-Za-z_0-9]*)"\}')
MODIFIED = {
    "state_header", "init_state", "so_state_init", "so_state_header",
    "to_command", "si_dispatch_sum",
}
ADDED = {"sb_sum_bool"}
PRESERVED = {
    "si_plan", "si_sum_integral", "sd_axis_mask", "sd_sum_tensor",
    "sd_execute_plan", "sd_backward", "typed_descriptor",
    "scalar_previous_typed_descriptor", "dtype_width", "tget", "taddr",
    "typed_load", "typed_store", "alloc", "so_slot_new",
    "gradient_accumulate", "incoming_describe", "partial_apply_node",
    "publish_tensor", "shape_command_finish", "entry",
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read_bound(path, expected_sha=None, expected_bytes=None, expected_mode=None):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), str(path)
    before = path.stat()
    assert stat.S_ISREG(before.st_mode)
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda s: (
        s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
        stat.S_IMODE(s.st_mode),
    )
    assert identity(before) == identity(after), str(path)
    actual = {
        "path": str(path.resolve()), "bytes": len(raw), "sha256": sha(raw),
        "mode": stat.S_IMODE(after.st_mode),
    }
    if expected_sha is not None:
        assert actual["sha256"] == expected_sha, (str(path), actual["sha256"])
    if expected_bytes is not None:
        assert len(raw) == expected_bytes, (str(path), len(raw))
    if expected_mode is not None:
        assert actual["mode"] == expected_mode, (str(path), actual["mode"])
    return raw, actual

def parse_source(raw):
    matches = list(FUNCTION.finditer(raw))
    assert matches
    names = [m[2].decode("ascii") for m in matches]
    assert len(names) == len(set(names))
    functions = {m[2].decode("ascii"): m[1] for m in matches}
    header = raw[:matches[0].start()]
    assert header + b"".join(functions.values()) == raw
    return header, functions

def write_new(path, raw):
    assert isinstance(raw, bytes)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    metadata = {
        "bytes": len(raw), "sha256": sha(raw),
        "mode": stat.S_IMODE(Path(path).stat().st_mode),
    }
    assert metadata["mode"] == 0o600 and Path(path).read_bytes() == raw
    return metadata

def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("ascii")

def apply_exact(parent_raw, patch):
    assert patch["schema"] == "heptatorch.sum-bool.exact-source-body-patch.v1"
    assert patch["status"] == "SOURCE_ONLY_NOT_APPLIED"
    assert patch["parent"]["source_sha256"] == PARENT_SHA256
    assert len(parent_raw) == patch["parent"]["bytes"] == 2312858
    assert sha(parent_raw) == PARENT_SHA256
    parent_header, parent = parse_source(parent_raw)
    assert parent_header == patch["parent_header"].encode("ascii")
    assert len(parent) == patch["parent"]["functions"] == 699
    assert list(parent) == patch["parent_function_order"]
    assert set(parent) == set(patch["parent_body_sha256"])
    assert {name: sha(body) for name, body in parent.items()} == patch["parent_body_sha256"]
    changes = patch["changed_functions"]
    additions = patch["added_functions"]
    assert len(changes) == 6 and {x["name"] for x in changes} == MODIFIED
    assert len(additions) == 1 and {x["name"] for x in additions} == ADDED
    assert ADDED.isdisjoint(parent)
    child = dict(parent)
    for row in changes:
        name = row["name"]
        before = row["before_body"].encode("ascii")
        after = row["after_body"].encode("ascii")
        assert parent[name] == before and sha(before) == row["before_sha256"]
        assert sha(after) == row["after_sha256"] and before != after
        _, parsed = parse_source(after)
        assert list(parsed) == [name] and parsed[name] == after
        child[name] = after
    for row in additions:
        body = row["body"].encode("ascii")
        assert sha(body) == row["sha256"]
        _, parsed = parse_source(body)
        assert list(parsed) == [row["name"]]
        assert parsed[row["name"]] == body
        child[row["name"]] = body
    unchanged = set(parent) - MODIFIED
    assert len(unchanged) == 693
    assert patch["unchanged_functions"] == [
        {"name": name, "sha256": sha(parent[name])}
        for name in parent if name in unchanged
    ]
    assert all(child[name] == parent[name] for name in unchanged)
    assert PRESERVED <= unchanged
    assert list(child) == patch["child_function_order"]
    assert len(child) == patch["prospective"]["functions"] == 700
    assert patch["prospective"]["descriptor_bytes"] == 672
    assert patch["prospective"]["state_format"] == "HTSINT02"
    assert patch["prospective"]["backing_format"] == "HTOSIN02"
    for name in MODIFIED | ADDED:
        dependencies = {m[1].decode("ascii") for m in CALL.finditer(child[name])}
        assert dependencies <= set(child), (name, sorted(dependencies - set(child)))
    child_header = patch["child_header"].encode("ascii")
    assert child_header == b"hia-asm.v1 hepta.hia.program.v8 sum_integral_bool_runtime\n"
    runtime = child_header + b"".join(child.values())
    actual_header, actual = parse_source(runtime)
    assert actual_header == child_header and actual == child
    restored = dict(actual)
    for row in changes:
        assert restored[row["name"]] == row["after_body"].encode("ascii")
        restored[row["name"]] = row["before_body"].encode("ascii")
    for row in additions:
        assert restored[row["name"]] == row["body"].encode("ascii")
        del restored[row["name"]]
    recovered = parent_header + b"".join(restored[name] for name in parent)
    assert recovered == parent_raw and sha(recovered) == PARENT_SHA256
    forwarded = dict(parent)
    for row in changes:
        forwarded[row["name"]] = row["after_body"].encode("ascii")
    for row in additions:
        forwarded[row["name"]] = row["body"].encode("ascii")
    assert child_header + b"".join(forwarded.values()) == runtime
    inverse = {
        "schema": "heptatorch.sum-bool.whole-runtime-inverse.v1",
        "status": "SOURCE_APPLIED_NOT_COMPILED",
        "parent_source_sha256": PARENT_SHA256,
        "child_source_sha256": sha(runtime),
        "parent_header": parent_header.decode("ascii"),
        "child_header": child_header.decode("ascii"),
        "parent_function_order": list(parent), "child_function_order": list(child),
        "changed_functions": changes, "added_functions": additions,
        "unchanged_functions": patch["unchanged_functions"],
        "whole_forward_exact": True, "whole_inverse_exact": True,
        "old_count": 699, "current_count": 700, "unchanged_count": 693,
        "modified_count": 6, "added_count": 1,
        "compiler_executed": False, "native_executed": False, "torch_executed": False,
    }
    return runtime, inverse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-runtime", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), "Output must be a new directory"
    package = args.package.resolve()
    manifest_raw, manifest_info = read_bound(package / "source-manifest.json")
    manifest = json.loads(manifest_raw)
    assert manifest["schema"] == "heptatorch.sum-bool.source-package.v1"
    assert manifest["status"] == "SOURCE_ONLY_NOT_APPLIED"
    assert manifest["parent_source_sha256"] == PARENT_SHA256
    declared = manifest["files"]
    assert manifest["self_exclusions"] == ["source-manifest.json"]
    files = list(package.rglob("*"))
    assert not any(p.is_symlink() for p in files)
    actual_names = {str(p.relative_to(package)) for p in files if p.is_file()}
    assert actual_names == set(declared) | {"source-manifest.json"}
    input_bindings = {manifest_info["path"]: manifest_info}
    payloads = {}
    for name, item in declared.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        raw, metadata = read_bound(package / relative, item["sha256"], item["bytes"], item["mode"])
        payloads[name] = raw
        input_bindings[metadata["path"]] = metadata
    assert manifest["payload_count"] == len(payloads)
    assert manifest["payload_bytes"] == sum(map(len, payloads.values()))
    assert (package / "apply_source.py").resolve() == Path(__file__).resolve()
    assert sha(payloads["body-patch.json"]) == PATCH_SHA256
    patch = json.loads(payloads["body-patch.json"])
    parent_raw, parent_info = read_bound(
        args.parent_runtime, PARENT_SHA256, 2312858, 0o600,
    )
    input_bindings[parent_info["path"]] = parent_info
    runtime, inverse = apply_exact(parent_raw, patch)
    for path, item in input_bindings.items():
        read_bound(path, item["sha256"], item["bytes"], item["mode"])
    args.output.mkdir(parents=False, exist_ok=False)
    emitted = {}
    primary_error = None
    try:
        emitted["runtime.hepta"] = write_new(args.output / "runtime.hepta", runtime)
        emitted["integral01-to-bool01-body-inverse.json"] = write_new(
            args.output / "integral01-to-bool01-body-inverse.json", json_bytes(inverse),
        )
        emitted["inputs-before.json"] = write_new(
            args.output / "inputs-before.json", json_bytes(input_bindings),
        )
        after = {}
        for path, item in input_bindings.items():
            _, after[path] = read_bound(path, item["sha256"], item["bytes"], item["mode"])
        assert after == input_bindings
        emitted["inputs-after.json"] = write_new(
            args.output / "inputs-after.json", json_bytes(after),
        )
    except BaseException:
        primary_error = traceback.format_exc()
    report = {
        "schema": "heptatorch.sum-bool.source-application.v1",
        "status": "FAIL" if primary_error else "PASS_SOURCE_APPLICATION_ONLY",
        "parent_source_sha256": PARENT_SHA256,
        "package_manifest": manifest_info, "patch_sha256": PATCH_SHA256,
        "source_sha256": sha(runtime), "functions": 700, "descriptor_bytes": 672,
        "state_format": "HTSINT02", "backing_format": "HTOSIN02",
        "changed_functions": sorted(MODIFIED), "added_functions": sorted(ADDED),
        "unchanged_parent_count": 693,
        "whole_forward_exact": True, "whole_inverse_exact": True,
        "input_bindings": input_bindings,
        "all_inputs_unchanged": primary_error is None,
        "files": emitted, "primary_error": primary_error,
        "unit_generation_executed": False, "ast_executed": False,
        "compiler_executed": False, "native_executed": False, "torch_executed": False,
        "numeric_expectations_computed_by_host": False,
        "runtime_qualified": False, "schema_closure": False, "full_project_status": "HOLD",
    }
    write_new(args.output / "report.json", json_bytes(report))
    return 1 if primary_error else 0

if __name__ == "__main__":
    raise SystemExit(main())
