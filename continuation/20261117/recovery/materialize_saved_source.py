#!/usr/bin/env python3
"""Materialize the exact saved Sum03 source delta; no source import or execution."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import stat
import sys
import traceback

PIN = "d4f4dee66b04a9548265d83ca614e661211a66f2733c4b6300322f0c5a8af0b8"
SIZE = 143906
ROOT = Path("/home/qian-qi/heptatorch-continuation-20261116/runs")
EXPECTED = {
    "source01": ("sum-source03-original76-recorded-source-01",
                 "5be6207769494ace8716c65f2a8890f5e5c9d2c21913c63ba07e2131851e5639"),
    "source02": ("sum-source03-original76-recorded-source-02",
                 "b12e09e3a2896d223852aad0a6d483e58a66fa9eee28742a16a957f2aaa54b58"),
}
BOUND = {}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def need(test, rule):
    if not test:
        raise AssertionError(rule)

def audit(event, args):
    if event in {"subprocess.Popen", "os.system", "os.exec", "os.fork", "os.forkpty",
                 "os.posix_spawn", "socket.connect", "socket.bind", "ctypes.dlopen",
                 "resource.prlimit", "resource.setrlimit"}:
        raise AssertionError(("source-recovery-no-process-network-native", event))

sys.addaudithook(audit)
sys.dont_write_bytecode = True

def fingerprint(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
            st.st_ctime_ns, stat.S_IMODE(st.st_mode))

def read(path, pin, size=None, mode=None):
    path = Path(path).absolute()
    need(path.resolve() == path and not path.is_symlink(), "canonical-no-symlink-input")
    before = path.stat()
    need(stat.S_ISREG(before.st_mode), "regular-input")
    raw = path.read_bytes()
    need(fingerprint(before) == fingerprint(path.stat()), "stable-complete-read")
    need(sha(raw) == pin and (size is None or len(raw) == size), "full-input-SHA-and-size")
    actual_mode = stat.S_IMODE(before.st_mode)
    need(mode is None or actual_mode == mode, "original-input-mode")
    row = {"bytes": len(raw), "sha256": pin, "mode": actual_mode}
    need(str(path) not in BOUND or BOUND[str(path)] == row, "unchanged-input-on-reread")
    BOUND[str(path)] = row
    return raw

def name(value):
    need(isinstance(value, str) and value and "\\" not in value, "relative-source-name")
    path = PurePosixPath(value)
    need(not path.is_absolute() and str(path) == value
         and all(part not in {".", "..", ""} for part in path.parts), "safe-source-name")
    return value

def transformed(parent, edits):
    value = parent.decode("utf-8")
    for edit in edits:
        need(set(edit) == {"before", "after", "count"} and type(edit["count"]) is int
             and edit["count"] > 0 and value.count(edit["before"]) == edit["count"],
             "exact-forward-edit-occurrence")
        value = value.replace(edit["before"], edit["after"], edit["count"])
    inverse = value
    for edit in reversed(edits):
        need(inverse.count(edit["after"]) == edit["count"], "exact-inverse-edit-occurrence")
        inverse = inverse.replace(edit["after"], edit["before"], edit["count"])
    need(inverse.encode() == parent, "whole-text-inverse")
    return value.encode()

def sources(bundle):
    need(bundle["schema"] == "heptatorch.sum03.original76.complete-authored-delta-recovery.v1",
         "exact-source-delta-schema")
    result = {}
    for revision in ("source01", "source02"):
        record = bundle[revision]
        subdir, manifest_pin = EXPECTED[revision]
        need(record["intended_root"] == str(ROOT / subdir)
             and record["source_manifest_sha256"] == manifest_pin, "fixed-source-destination")
        manifest_raw = record["source_manifest_text"].encode()
        need(sha(manifest_raw) == manifest_pin, "exact-source-manifest-text")
        manifest = json.loads(manifest_raw)
        need(set(record["files"]) == set(manifest["files"]), "entire-source-payload-set")
        current = {}
        for relative, entry in record["files"].items():
            name(relative)
            expected = manifest["files"][relative]
            need(all(entry[key] == expected[key] for key in ("bytes", "sha256", "mode")),
                 "full-source-member-binding")
            forms = set(entry) - {"bytes", "sha256", "mode"}
            need(len(forms) == 1, "exactly-one-source-recovery-form")
            form = next(iter(forms))
            if form == "text":
                raw = entry["text"].encode()
            elif form == "source_ref":
                need(revision == "source01", "only-original-declared-source-references")
                raw = read(entry["source_ref"], entry["sha256"], entry["bytes"], entry["mode"])
            elif form == "from_source01":
                raw = result["source01"][entry["from_source01"]]
            elif form == "from_source01_manifest":
                need(entry[form] is True, "exact-parent-manifest-reference")
                raw = result["source01"]["source-manifest.json"]
            elif form == "transform":
                item = entry[form]
                need(set(item) == {"from_source01", "edits"}, "exact-transform-record")
                raw = transformed(result["source01"][item["from_source01"]], item["edits"])
            elif form == "append":
                item = entry[form]
                need(set(item) == {"from_source01", "text"}, "exact-append-record")
                raw = result["source01"][item["from_source01"]] + item["text"].encode()
            else:
                raise AssertionError(("unknown-source-recovery-form", form))
            need(len(raw) == entry["bytes"] and sha(raw) == entry["sha256"],
                 "complete-reconstructed-source-hash")
            current[relative] = raw
        need(len(current) == manifest["payload_count"]
             and sum(map(len, current.values())) == manifest["payload_bytes"],
             "complete-reconstructed-source-counts")
        current["source-manifest.json"] = manifest_raw
        result[revision] = current
    return result

def check_directory(directory, wanted, *, complete):
    need(directory.absolute() == directory.resolve() and not directory.is_symlink(),
         "canonical-package-directory")
    if not directory.exists():
        need(not complete, "complete-package-directory-exists")
        return
    need(directory.is_dir(), "package-root-directory")
    wanted_dirs = set()
    for relative in wanted:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            wanted_dirs.add(str(parent))
            parent = parent.parent
    observed = set()
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        metadata = path.lstat()
        need(not stat.S_ISLNK(metadata.st_mode), "no-source-package-symlink")
        if stat.S_ISDIR(metadata.st_mode):
            need(relative in wanted_dirs, "no-unmanifested-source-directory")
        else:
            need(stat.S_ISREG(metadata.st_mode) and relative in wanted,
                 "no-unmanifested-source-file")
            raw = wanted[relative]
            read(path, sha(raw), len(raw), 0o660)
            observed.add(relative)
    need(not complete or observed == set(wanted), "complete-source-package-member-set")

def publish(path, raw, mode):
    need(path.absolute() == path.resolve() and not path.is_symlink(), "canonical-target")
    if path.exists():
        need(read(path, sha(raw), len(raw), mode) == raw, "existing-target-exact-reuse-only")
        return "REUSED_EXACT"
    path.parent.mkdir(parents=True, exist_ok=True)
    need(path.parent.resolve() == path.parent.absolute(), "no-parent-symlink-after-mkdir")
    # xb refuses an existing final path. A failed partial file is retained and
    # must fail later reconciliation; it is never silently removed or replaced.
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    need(read(path, sha(raw), len(raw), mode) == raw, "complete-created-source-readback")
    return "CREATED_EXACT"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-json", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.absolute()
    need(report_path.resolve() == report_path and not report_path.exists(),
         "create-only-recovery-report")
    need(not any(report_path.is_relative_to(ROOT / row[0]) for row in EXPECTED.values()),
         "report-outside-frozen-source-packages")
    report = {"schema": "heptatorch.sum03.source-delta-materialization.v1",
              "status": "PREPARING", "source_blob_sha1": "c858980307ddfcf581a44687e4d9f5f3d1e22d86",
              "delta_sha256": PIN, "files": [], "source_imported_or_executed": False,
              "native_Torch_compiler_executions": 0, "Git_mutations": 0}
    try:
        read(Path(__file__), args.expected_source_sha256)
        bundle = json.loads(read(args.delta_json, PIN, SIZE))
        reconstructed = sources(bundle)
        # All external originals and all full reconstructed hashes are verified
        # before the first source package write. Existing unknown writes are
        # reconciled in both namespaces before any missing member is created.
        for revision in ("source01", "source02"):
            check_directory(ROOT / EXPECTED[revision][0], reconstructed[revision], complete=False)
        for revision in ("source01", "source02"):
            directory = ROOT / EXPECTED[revision][0]
            for relative, raw in reconstructed[revision].items():
                mode = 0o660
                action = publish(directory / name(relative), raw, mode)
                report["files"].append({"revision": revision, "path": str(directory / relative),
                                        "bytes": len(raw), "sha256": sha(raw), "mode": mode,
                                        "action": action})
        for revision in ("source01", "source02"):
            check_directory(ROOT / EXPECTED[revision][0], reconstructed[revision], complete=True)
        for path, expected in list(BOUND.items()):
            read(path, expected["sha256"], expected["bytes"], expected["mode"])
        report.update(status="PASS_SOURCE_MATERIALIZED_ONLY", all_inputs_unchanged=True,
                      original_unknown_writes_reconciled_by_complete_bytes_and_modes=True)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    report["bound_inputs"] = BOUND
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "files": len(report["files"])}))
    return 0 if report["status"] == "PASS_SOURCE_MATERIALIZED_ONLY" else 1

if __name__ == "__main__":
    raise SystemExit(main())
