#!/usr/bin/env python3
"""Run the frozen Integral24 byte reader in an isolated original-path layout.

The recovered files live below --mirror-root/home. A private Linux user/mount
namespace binds that directory read-only and noexec at /home, preserving the
original paths inside every frozen report. No reader source or report field is
rewritten. Only the original Python sealing/replay entry is executed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import traceback

PACKAGE = "0fd26b7e199fba02cb0eac502865ed5bdea17f4dc6b2b84436da30f85337af94"
REPORT = "5f8f9a7b4dbf6455c628a78b392c79d06ff83beecb9eca0caa940f91b85e700b"
RECEIPTS = "71b02aa9d4528a67209c7826c2b7d8b0fa4a412d56f48069b808a4341daaa9f7"
ORIGINAL_ROOT = Path("/home/qian-qi/heptatorch-continuation-20261116")
SOURCE_RELATIVE = Path("continuation/20261117/source-packages/sum-integral-pure-admission-source-01")
RECEIPTS_RELATIVE = Path("continuation/20261117/evidence/native24-terminal-and-EOF.json")
JOB = "1a1e8fcdd64-392-410d"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_record(path):
    path = Path(path).absolute()
    before = path.stat()
    require(path.resolve() == path and not path.is_symlink()
            and stat.S_ISREG(before.st_mode), ("regular-canonical-file", str(path)))
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
    require(all(getattr(before, k) == getattr(after, k) for k in fields),
            ("stable-file", str(path)))
    return {"path": str(path), "bytes": after.st_size, "sha256": h.hexdigest(),
            "mode": stat.S_IMODE(after.st_mode)}


def write_json(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return file_record(path)


def restore_reader(source, output):
    manifest_raw = (source / "source-manifest.json").read_bytes()
    require(sha(manifest_raw) == PACKAGE, "original-pure-reader-manifest")
    manifest = json.loads(manifest_raw)
    expected = set(manifest["files"]) | {"source-manifest.json"}
    require({p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()} == expected,
            "complete-original-reader-file-set")
    output.mkdir(exist_ok=False)
    for name, pin in manifest["files"].items():
        path = source / name
        require(path.resolve() == path.absolute() and not path.is_symlink(),
                ("regular-reader-source", name))
        raw = path.read_bytes()
        require(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"],
                ("whole-original-reader-source", name))
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
        destination.chmod(pin["mode"])
    with (output / "source-manifest.json").open("xb") as stream:
        stream.write(manifest_raw)
    (output / "source-manifest.json").chmod(0o660)
    return manifest


def receipt_object(receipts, key):
    item = receipts[key]
    require(item["status"] == "fulfilled" and not item["value"].get("isError"),
            ("actual-original-receipt", key))
    content = item["value"]["content"]
    require(len(content) == 1 and content[0]["type"] == "text", ("complete-receipt", key))
    return json.loads(content[0]["text"])


def verify_receipts(path, profile):
    raw = path.read_bytes()
    require(sha(raw) == RECEIPTS, "entire-original-execution-receipt-bytes")
    receipts = json.loads(raw)
    start = receipt_object(receipts, "original_first_start")["job"]
    terminal = receipt_object(receipts, "original_terminal")
    logs = receipt_object(receipts, "original_full_logs")
    info = receipt_object(receipts, "original_report_file_info")
    request = receipts["actual_exact_request"]
    require(start["id"] == terminal["job_id"] == logs["job_id"] == JOB,
            "same-original-native24-job")
    for key in ("args", "cwd", "executable", "host", "idempotency_key"):
        require(start[key] == request[key], ("original-request-start-identity", key))
    require(request["host"] == "rog" and request["args"][-1] == "--execute-native"
            and receipts["rerun_permitted_as_recovery"] is False,
            "readback-of-completed-job-not-new-native-execution")
    require(terminal["state"] == "succeeded" and terminal["exit_code"] == 0
            and terminal["output_complete"] is True and terminal["output_durable"] is True
            and terminal["terminal_publication_confirmed"] is True
            and not terminal["cancelled"] and not terminal["timed_out"],
            "actual-complete-original-terminal")
    for name in ("stdout", "stderr"):
        stream = logs[name]
        require(stream["eof"] is True and stream["offset"] == stream["next_offset"] == stream["size"] == 0
                and stream["bytes_hex"] == stream["text"] == ""
                and terminal["log_bytes"][name] == {"seen": 0, "stored": 0}
                and terminal["log_synced"][name] is True
                and terminal["log_truncated"][name] is False
                and terminal["log_errors"][name] is None,
                ("complete-original-empty-stream-and-EOF", name))
    expected = profile["actual_native_report"]
    require(info["path"] == expected["path"] and info["size"] == expected["bytes"]
            and info["sha256"] == expected["sha256"] == REPORT and info["mode"] == expected["mode"],
            "original-file-info-matches-whole-native-report")
    return {"role": "original-request-start-terminal-dual-EOF", **file_record(path)}


def namespace_main(argv):
    mirror, entry, expected_entry, original_namespace = argv[:4]
    mirror, entry = Path(mirror), Path(entry)
    require(os.geteuid() == 0 and os.readlink("/proc/self/ns/mnt") != original_namespace,
            "fresh-private-user-and-mount-namespace")
    home = mirror / "home"
    require(home.is_dir() and home.resolve() == home and not home.is_symlink(),
            "explicit-mirror-home-directory")
    require(file_record(entry)["sha256"] == expected_entry, "exact-frozen-entry-before-exec")
    subprocess.run(["/usr/bin/mount", "--bind", str(home), "/home"], check=True)
    subprocess.run(["/usr/bin/mount", "-o", "remount,bind,ro,nosuid,nodev,noexec", "/home"], check=True)
    require(os.stat(home).st_ino == os.stat("/home").st_ino
            and os.stat(home).st_dev == os.stat("/home").st_dev
            and Path("/home").resolve() == Path("/home"), "exact-read-only-original-path-view")
    # A fresh interpreter removes this wrapper's imports before the frozen
    # reader installs its subprocess, Torch, network and /proc audit hook.
    os.execv("/usr/bin/python3", ["/usr/bin/python3", "-I", "-S", "-B", str(entry), *argv[4:]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("seal", "prepare", "admit"), required=True)
    parser.add_argument("--evidence-binding", type=Path)
    parser.add_argument("--evidence-binding-sha256")
    args = parser.parse_args()
    require(not sys.flags.optimize, "all-original-assertions-enabled")
    mirror, repo, output = (p.resolve() for p in (args.mirror_root, args.repo, args.output))
    require((mirror / "home").is_dir() and not output.is_relative_to(mirror),
            "separate-recovered-inputs-and-new-output")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "heptatorch.integral24.portable-recorded.v1", "status": "PREPARING",
              "phase": args.phase, "native_executed": False, "Torch_executed": False,
              "compiler_executed": False, "frozen_reader_sha256": PACKAGE,
              "original_report_sha256": REPORT, "schema_closure": False,
              "full_project_status": "HOLD"}
    code = 1
    try:
        reader = output / "reader"
        manifest = restore_reader(repo / SOURCE_RELATIVE, reader)
        profile = json.loads((reader / "profile.json").read_bytes())
        report_pin = profile["actual_native_report"]
        logical = Path(report_pin["path"])
        recovered_report = mirror / logical.relative_to("/")
        restored_pin = file_record(recovered_report)
        require(all(restored_pin[k] == report_pin[k] for k in ("bytes", "sha256", "mode")),
                "whole-original-report-bytes-and-mode")
        report = json.loads(recovered_report.read_bytes())
        for path, pin in report["checked_inputs"].items():
            original = Path(path)
            require(original.is_relative_to(ORIGINAL_ROOT) and pin["path"] == path,
                    "bounded-original-input-prefix")
            observed = file_record(mirror / original.relative_to("/"))
            require(all(observed[k] == pin[k] for k in ("bytes", "sha256", "mode")),
                    ("complete-original-input-bytes-and-mode", path))
        entry_name = "seal_raw.py" if args.phase == "seal" else "admit_recorded_integral24.py"
        entry = reader / entry_name
        argv = ["--package-sha256", PACKAGE, "--output", str(output / "result")]
        if args.phase == "seal":
            receipt = verify_receipts(repo / RECEIPTS_RELATIVE, profile)
            plan = {"schema": "heptatorch.sum-integral24.raw-seal-plan.v1",
                    "status": "ACTUAL_NATIVE_RUN_BOUND",
                    "profile_sha256": sha((reader / "profile.json").read_bytes()),
                    "native_output": str(logical.parent), "native_report": report_pin,
                    "execution_evidence": [receipt]}
            plan_record = write_json(output / "seal-plan.json", plan)
            argv += ["--plan", plan_record["path"], "--plan-sha256", plan_record["sha256"]]
            result["seal_plan"] = plan_record
        else:
            require(args.evidence_binding is not None and args.evidence_binding_sha256 is not None,
                    "actual-sealed-evidence-binding-required")
            binding = file_record(args.evidence_binding)
            require(binding["sha256"] == args.evidence_binding_sha256, "actual-evidence-binding-pin")
            argv += ["--evidence-binding", binding["path"], "--evidence-binding-sha256", binding["sha256"],
                     "--prepare-only" if args.phase == "prepare" else "--admit-recorded"]
            result["evidence_binding"] = binding
        command = ["/usr/bin/unshare", "--user", "--map-root-user", "--mount", "--propagation", "private",
                   "/usr/bin/python3", "-I", "-S", "-B", str(Path(__file__).resolve()), "_namespace",
                   str(mirror), str(entry), manifest["files"][entry_name]["sha256"],
                   os.readlink("/proc/self/ns/mnt"), *argv]
        result.update(original_input_count=len(report["checked_inputs"]), command=command,
                      original_paths_preserved_by="private-read-only-noexec-home-bind")
        write_json(output / "request.json", result)
        with (output / "stdout.bin").open("xb") as stdout, (output / "stderr.bin").open("xb") as stderr:
            completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=False)
        code = completed.returncode
        result.update(exit_code=code, stdout=file_record(output / "stdout.bin"),
                      stderr=file_record(output / "stderr.bin"))
        if (output / "result/report.json").exists():
            original = json.loads((output / "result/report.json").read_bytes())
            result.update(reader_report=file_record(output / "result/report.json"), reader_status=original["status"])
            if code == 0:
                wanted = {"seal": "PASS_CUSTODY_ONLY", "prepare": "PREPARED_RECORDED_INPUTS_NOT_REPLAYED",
                          "admit": "PASS_PURE_RECORDED_ADMISSION"}[args.phase]
                require(original["status"] == wanted and original["native_executed"] is False
                        and original["Torch_executed"] is False and original["compiler_executed"] is False
                        and not any(original["forbidden_execution_attempts"].values()),
                        "successful-complete-pure-reader-without-forbidden-execution")
                if args.phase == "admit":
                    result.update(cases_recomputed=original["cases_recomputed"],
                                  reference_events_recomputed=original["reference_events_recomputed"],
                                  bounded_controls_recomputed=original["bounded_controls_recomputed"])
        else:
            require(code != 0, "successful-reader-must-write-its-original-result")
        result["status"] = "PASS" if code == 0 else "FAIL"
    except BaseException as error:
        code = 1
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    write_json(output / "report.json", result)
    print(json.dumps({k: result[k] for k in ("status", "phase", "native_executed", "Torch_executed")}, sort_keys=True))
    return code


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_namespace":
        namespace_main(sys.argv[2:])
    else:
        raise SystemExit(main())
