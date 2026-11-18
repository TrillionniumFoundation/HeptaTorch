#!/usr/bin/env python3
"""Run the original 76 view/scalar cases against the complete Bool700 image."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import time


ROOT = Path(__file__).resolve().parents[2]
INTENT = ROOT / "continuation/20261117/evidence/root-sum-integral-original76-reviewed-intent-v1.json"
INTENT_SHA = "2fb7e1f6a95e5cd6d0fe80468e414d65731f7b9d87852a33ec3c1812fbff07e8"
ARCHIVE = ROOT / "validation/current-runtime76-reference/complete.tar.xz"
ARCHIVE_SHA = "ce4aff6daaf5bb727fa6a520cbecf98dd8e09e67ffffa6cb60171a210b82e490"
SOURCE_SHA = "fac3d413c6c0f6874d224cbea312108ee2c8baa027102d41890d85094a5faa38"
NATIVE_SHA = "5b7aac203d2c913d43781f0da45d40e9bac2037c2bc074ec30f8dd663710a805"
COMPILER_SHA = "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"
LINKER_SHA = "761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e"
BUILDER_SHA = "9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d"
DEPENDENCIES = {
    "protocol.py": "tests/differentiable_view_native_review/protocol.py",
    "owner_client.py": "tests/differentiable_view_native_review/owner_client.py",
    "replay_direct_view14.py": "tests/differentiable_view_native_review/replay_direct_view14.py",
    "native_observer_shape02.py": "tests/differentiable_view_shape_review/native_observer_shape02.py",
    "identity_epochs.py": "tests/differentiable_view_shape_review/identity_epochs.py",
    "replay_shape26_r3.py": "tests/differentiable_view_shape_review/replay_shape26_r3.py",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def frozen_sources(out, bind):
    intent = json.loads(bind(INTENT, INTENT_SHA))
    original = intent["source_manifest"]
    manifest_raw = original["text"].encode()
    require(len(manifest_raw) == original["size"] and sha(manifest_raw) == original["sha256"],
            "original source manifest mismatch")
    manifest = json.loads(manifest_raw)
    require(set(manifest["files"]) == set(intent["source_payloads"]), "incomplete original source payloads")
    frozen = out / "frozen"
    frozen.mkdir()
    for name, row in intent["source_payloads"].items():
        require(Path(name).name == name, "invalid original payload name")
        raw = row["text"].encode()
        require(len(raw) == row["size"] == manifest["files"][name]["bytes"]
                and sha(raw) == row["sha256"] == manifest["files"][name]["sha256"], "original payload mismatch: " + name)
        (frozen / name).write_bytes(raw)
    (frozen / "source-manifest.json").write_bytes(manifest_raw)
    configuration = json.loads((frozen / "configuration.json").read_bytes())
    require([r["name"] for r in configuration["suites"]] == ["view26", "scalar32", "view14", "input4"],
            "original suite order changed")
    require([r["cases"] for r in configuration["suites"]] == [26, 32, 14, 4], "original cases changed")
    dependencies = {}
    for name, relative in DEPENDENCIES.items():
        path = ROOT / relative
        bind(path, configuration["fixed_dependencies"][name]["sha256"])
        dependencies[name] = path
    observer_raw = (frozen / "native_observer_integral_legacy.py").read_bytes()
    changed = observer_raw
    for old, new in ((b"HTSINT01", b"HTSINT02"), (b"HTOSIN01", b"HTOSIN02")):
        require(changed.count(old) == 1 and new not in changed, "observer format edit is not unique")
        changed = changed.replace(old, new, 1)
    require(changed.replace(b"HTSINT02", b"HTSINT01").replace(b"HTOSIN02", b"HTOSIN01") == observer_raw,
            "observer inverse differs from frozen source")
    observer = out / "native_observer_bool700_legacy.py"
    observer.write_bytes(changed)
    dump(out / "source-bindings.json", {
        "original_intent_sha256": INTENT_SHA, "original_manifest_sha256": original["sha256"],
        "all_original_payloads_byte_identical": True,
        "comparison_readers_modified": False, "assertions_or_expectations_modified": False,
        "observer_original_sha256": sha(observer_raw), "observer_sha256": sha(changed),
        "observer_changes": {"HTSINT01": "HTSINT02", "HTOSIN01": "HTOSIN02"},
        "observer_whole_inverse_exact": True, "old_pass_transferred": False,
    })
    return configuration, frozen, dependencies, observer


def reference_inputs(configuration, args, out, bind):
    result = {}
    if args.references:
        for suite in configuration["suites"]:
            directory = args.references.resolve() / suite["name"]
            for kind, basename in (("reference", "actual.json"), ("probe", "probe.py")):
                path = directory / basename
                raw = bind(path, suite[kind]["sha256"])
                require(len(raw) == suite[kind]["bytes"], "original reference size mismatch")
                result[suite["name"], kind] = path
        return result
    archive = bind(args.reference_archive.resolve(), ARCHIVE_SHA)
    wanted = {suite[kind]["sha256"]: (suite["name"], kind, suite[kind]["bytes"])
              for suite in configuration["suites"] for kind in ("reference", "probe")}
    sizes = {row[2] for row in wanted.values()}
    matches = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as packet:
        for member in packet:
            if not member.isfile() or member.size not in sizes:
                continue
            raw = packet.extractfile(member).read()
            digest = sha(raw)
            if digest not in wanted:
                continue
            suite, kind, count = wanted[digest]
            require((suite, kind) not in result and len(raw) == count, "duplicate reference member")
            directory = out / "reference" / suite
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ("actual.json" if kind == "reference" else "probe.py")
            path.write_bytes(raw)
            bind(path, digest)
            result[suite, kind] = path
            matches.append({"suite": suite, "kind": kind, "member": member.name, "sha256": digest})
    require(len(result) == 8, "not all four original references and probes were recovered")
    dump(out / "reference-bindings.json", {"archive_sha256": ARCHIVE_SHA, "matches": matches})
    return result


def build_binding(directory, expected, out, bind):
    original = bind(directory / "build.json", expected)
    built = json.loads(original)
    require(built["schema"] == "heptatorch.complete-runtime-build.v1"
            and built["status"] == "PASS_NATIVE_BUILD_ONLY", "expected an actual complete-runtime cold build")
    require(built["source_sha256"] == SOURCE_SHA and built["native_sha256"] == NATIVE_SHA
            and built["functions"] == 712 and built["units_planned"] == 15, "unexpected current runtime profile")
    require(built["compiler_sha256"] == COMPILER_SHA and built["linker_sha256"] == LINKER_SHA
            and built["link_exit"] == 0 and built["all_inputs_unchanged"] is True,
            "native tool or successful build binding mismatch")
    require(built["objects_reused"] is False and built["runtime_qualified"] is False,
            "build receipt cannot transfer runtime qualification")
    builder = ROOT / "tools/current_runtime/build.py"
    bind(builder, BUILDER_SHA)
    require(any(Path(path).name == "build.py" and row["sha256"] == BUILDER_SHA for path, row in built["inputs"].items()),
            "actual build does not bind the fixed current builder")
    require(len(built["units"]) == 15 and len(built["processes"]) == 16, "incomplete cold build")
    require([p["name"] for p in built["processes"]] == [u["name"] for u in built["units"]] + ["link"],
            "cold build process order mismatch")
    require(all(p["started"] is True and p["exit_code"] == 0 and p["stdout_EOF"] is True
                and p["stderr_EOF"] is True and p["stderr_bytes"] == 0 for p in built["processes"]),
            "cold build process incomplete or failed")
    profile = out / "profile"
    profile.mkdir()
    image = bind(directory / "heptatorch", NATIVE_SHA)
    source = bind(directory / "runtime.hepta", SOURCE_SHA)
    (profile / "heptatorch").write_bytes(image); (profile / "heptatorch").chmod(0o500)
    (profile / "runtime.hepta").write_bytes(source)
    (profile / "actual-build.json").write_bytes(original)
    objects = []
    for unit, process in zip(built["units"], built["processes"]):
        name = unit["name"]
        require(Path(name).name == name and unit["compiler_exit"] == 0
                and unit["compiler_stderr_bytes"] == 0, "invalid native unit")
        require(process["input_sha256"] == unit["source_sha256"]
                and process["stdout_sha256"] == unit["object_sha256"], "unit process binding mismatch")
        for suffix, key in ((".hepta", "source_sha256"), (".o", "object_sha256")):
            raw = bind(directory / (name + suffix), unit[key])
            (profile / (name + suffix)).write_bytes(raw)
            if suffix == ".o":
                objects.append(raw)
        require(not bind(directory / (name + ".stderr")), "compiler stderr is not empty")
    request = b"HIACGS01" + struct.pack("<3Q", 15, 4, 0) + b"step"
    request += b"".join(struct.pack("<Q", len(obj)) + obj for obj in objects)
    actual_request = bind(directory / "link.request.bin", built["processes"][-1]["input_sha256"])
    require(actual_request == request and not bind(directory / "link.stderr"), "native link request/stderr mismatch")
    require(built["processes"][-1]["stdout_sha256"] == NATIVE_SHA, "native link output mismatch")
    # A separate, explicitly named metadata bridge leaves the actual receipt untouched.
    # The 672-byte descriptor and two tags belong to this exact source/image profile.
    bridge = {**built, "schema": "heptatorch.current-runtime.legacy76-build-binding.v1",
              "descriptor_bytes": 672, "state_format": "HTSINT02", "backing_format": "HTOSIN02",
              "original_build_schema": built["schema"], "original_build_sha256": sha(original),
              "bridge_adds_format_metadata_only": True, "legacy_runtime_pass_transferred": False}
    dump(profile / "build.json", bridge)
    return profile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-directory", type=Path, required=True)
    ap.add_argument("--expected-build-sha256", required=True)
    ap.add_argument("--references", type=Path, help="Already recovered original references: suite/actual.json and suite/probe.py.")
    ap.add_argument("--reference-archive", type=Path, default=ARCHIVE)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    require(not sys.flags.optimize, "original assertions must remain enabled")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    inputs, rows = {}, []
    report = {"schema": "heptatorch.current-runtime.original76.v1", "status": "PREPARING",
              "source_sha256": SOURCE_SHA, "native_sha256": NATIVE_SHA,
              "expected_cases": 76, "functions": 712, "native_units": 15,
              "state_format": "HTSINT02", "backing_format": "HTOSIN02", "descriptor_bytes": 672,
              "actual_build_sha256": args.expected_build_sha256,
              "old_pass_transferred": False, "torch_executed": False, "native_suites": rows,
              "assertions_or_expectations_modified": False, "whole_pytorch_rewrite_complete": False}

    def bind(path, pin=None):
        path = path.resolve()
        raw = path.read_bytes()
        require(pin is None or sha(raw) == pin, "input SHA mismatch: " + str(path))
        require(path not in inputs or inputs[path] == raw, "input changed: " + str(path))
        inputs[path] = raw
        return raw

    def save():
        dump(out / "report.json", report)

    try:
        bind(Path(__file__).resolve())
        configuration, frozen, dependencies, observer = frozen_sources(out, bind)
        references = reference_inputs(configuration, args, out, bind)
        profile = build_binding(args.build_directory.resolve(), args.expected_build_sha256, out, bind)
        for path in [*frozen.iterdir(), observer, *profile.iterdir()]:
            if path.is_file():
                bind(path)
        report["status"] = "RUNNING_PREPARATION" if args.prepare_only else "RUNNING_NATIVE"
        save()
        for suite in configuration["suites"]:
            name = suite["name"]
            command = [sys.executable, "-I", "-S", "-B", str(frozen / suite["reader"]),
                "--binary", str(profile / "heptatorch"), "--source", str(profile / "runtime.hepta"),
                "--build-report", str(profile / "build.json"), "--protocol", str(dependencies["protocol.py"]),
                "--observer", str(observer), "--reference", str(references[name, "reference"]),
                "--probe", str(references[name, "probe"]), "--expected-source-sha256", SOURCE_SHA,
                "--expected-native-sha256", NATIVE_SHA, "--expected-observer-sha256", sha(observer.read_bytes()),
                "--expected-functions", "712", "--expected-native-units", "15", "--output", str(out / name)]
            if name != "view14":
                command += ["--support-reader", str(dependencies["replay_direct_view14.py"])]
            if name in {"view26", "input4"}:
                command += ["--expected-reference-sha256", suite["reference"]["sha256"]]
            if name == "view26":
                command += ["--identity-adapter", str(dependencies["identity_epochs.py"]),
                            "--parent-reader", str(dependencies["replay_shape26_r3.py"])]
            if args.prepare_only:
                command.append("--prepare-only")
            row = {"name": name, "expected_cases": suite["cases"], "argv": command, "status": "RUNNING"}
            rows.append(row); save()
            print(json.dumps({"suite": name, "status": row["status"]}), flush=True)
            started = time.monotonic()
            with (out / (name + ".stdout")).open("wb") as stdout, (out / (name + ".stderr")).open("wb") as stderr:
                child = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=7200)
            stdout = (out / (name + ".stdout")).read_bytes(); stderr = (out / (name + ".stderr")).read_bytes()
            row.update(exit_code=child.returncode, stdout_EOF=True, stderr_EOF=True,
                       stdout_sha256=sha(stdout), stderr_sha256=sha(stderr), stderr_bytes=len(stderr),
                       elapsed_seconds=time.monotonic() - started)
            path = out / name / "report.json"
            if path.is_file():
                raw = path.read_bytes(); child_report = json.loads(raw)
                row["report_sha256"] = sha(raw)
                row["status"] = child_report["status"]
                for key in ("cases", "passed_cases", "failed_cases", "assertions", "frames", "exact_words", "full_rng_bytes", "rollback_bytes"):
                    if key in child_report:
                        row[key] = child_report[key]
                expected_status = "PREPARED_NOT_RUN" if args.prepare_only else "PASS"
                row["qualified"] = child.returncode == 0 and not stderr and child_report["status"] == expected_status and child_report["all_inputs_unchanged"] is True
                if not args.prepare_only:
                    row["qualified"] &= child_report["cases"] == suite["cases"] and child_report["passed_cases"] == suite["cases"]
                    row["qualified"] &= child_report.get("all_sessions_eof_exit_zero", child_report.get("all_executed_sessions_eof_exit_zero", False)) is True
            else:
                row.update(status="FAIL_NO_REPORT", qualified=False)
            save(); print(json.dumps({"suite": name, "status": row["status"], "qualified": row["qualified"]}), flush=True)
        unchanged = all(path.read_bytes() == raw for path, raw in inputs.items())
        passed = len(rows) == 4 and all(row["qualified"] for row in rows) and unchanged
        report.update(status=("PREPARED_NOT_RUN" if args.prepare_only else "PASS") if passed else "FAIL",
                      native_executed=not args.prepare_only, all_inputs_unchanged=unchanged,
                      cases=sum(row.get("cases", 0) for row in rows),
                      inputs={str(path): {"bytes": len(raw), "sha256": sha(raw)} for path, raw in inputs.items()})
        for key in ("passed_cases", "failed_cases", "assertions", "frames", "exact_words", "full_rng_bytes", "rollback_bytes"):
            report[key] = sum(row.get(key, 0) for row in rows)
        save(); print(json.dumps({k: report[k] for k in ("status", "cases", "passed_cases", "frames", "assertions")}))
        return 0 if passed else 1
    except Exception as error:
        report.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        save(); print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
