#!/usr/bin/env python3
"""Replay all original F32 Sum32 observations on the current native image.

The frozen SumCase and reference admission retain all original assertions. This
entry point binds the new cold build and explicitly relocates old input paths;
it does not evaluate tensor arithmetic or transfer a historical PASS.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
LEGACY_RUNNER_SHA = "e4f3c6118ce66bcec7489533a876652075674bca86ecb05cb4183b0a1780cb42"
ORIGINAL_SHA = "4676870b32dd341f9e6c30d170e8c5a4abbc401668ed3cdae492d9712233e138"
PARENT_SHA = "f13a204f5071c07b34ab35ee1b4a6dc90163137b0762fb57bd8c34abb0c8b277"
INTEGRAL_SHA = "6eee1fcd26367af3fba97488118e419788d64f41f90901b37cc594fe38154a9f"
INTEGRAL_INTENT = ROOT / "continuation/20261117/evidence/root-sum-integral-native24-reviewed-intent-v2.json"
INTEGRAL_INTENT_SHA = "4050bf41236f33080d590d0b65bbf1359f9834d4001eb64883a62cbef2153efc"
REFERENCE_SHA = "c14e4b18f657429ad5dac5498c3abcae303b6a7e7550db7d15a3c9179bdcd728"
REFERENCE_MANIFEST_SHA = "7db1e860e6387a0d48b98769eb9f124211a90b92bfb62a07e55eb294ab4ef780"
CASE_SHA = "191bdc7d1f69679912345ddca8efcf4e06bc1f29659ec7d65c1b349a17b68a54"
REFERENCE_ADMITTER_SHA = "7121e1ff7f856eb2e4cb1fcee197c3092b08f10eafb0744f201629639e3f0ec8"
CONSTRUCTOR_SHA = "3ed7a51cb5ee0879851b47890f9fa62ec88bef246fd3f37f222150f990571bc5"
OLD_RUNS = Path("/home/qian-qi/heptatorch-continuation-20261116/runs")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class Inputs:
    def __init__(self):
        self.rows = {}

    def bind(self, path, expected=None):
        requested = Path(path)
        require(not requested.is_symlink(), "input must not be a symlink: " + str(requested))
        path = requested.resolve()
        before = path.stat()
        require(stat.S_ISREG(before.st_mode), "input must be a regular file: " + str(path))
        raw = path.read_bytes()
        after = path.stat()
        fields = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(fields(before) == fields(after), "input changed during read: " + str(path))
        row = {"path": str(path), "bytes": len(raw), "sha256": sha(raw), "mode": stat.S_IMODE(after.st_mode)}
        require(expected is None or row["sha256"] == expected, "input hash mismatch: " + str(path))
        require(str(path) not in self.rows or self.rows[str(path)] == row, "input changed: " + str(path))
        self.rows[str(path)] = row
        return raw

    def recheck(self):
        errors = []
        for path, row in list(self.rows.items()):
            try:
                self.bind(path, row["sha256"])
            except BaseException as error:
                errors.append({"path": path, "error": repr(error)})
        return errors


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package(root, digest, inputs):
    manifest = json.loads(inputs.bind(root / "source-manifest.json", digest))
    require(isinstance(manifest["files"], dict), "expected a complete original package manifest")
    files = manifest["files"]
    require(len(files) == manifest["file_count"], "original package file count differs")
    require({str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} == set(files) | {"source-manifest.json"}
            and not any(p.is_symlink() for p in root.rglob("*")), "original package contains extra or symbolic files")
    total = 0
    for name, pin in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name,
                "unsafe original package member")
        path = root / name
        raw = inputs.bind(path, pin["sha256"])
        require({k: inputs.rows[str(path.resolve())][k] for k in ("bytes", "mode", "sha256")} == pin,
                "original package bytes/mode mismatch: " + name)
        total += len(raw)
    require(total == manifest["payload_bytes"], "original package payload size differs")
    return manifest


def recover_integral(out, inputs):
    intent = json.loads(inputs.bind(INTEGRAL_INTENT, INTEGRAL_INTENT_SHA))
    item = intent["source_manifest"]
    raw = item["text"].encode()
    require(item["eof"] is True and sha(raw) == item["sha256"] == INTEGRAL_SHA and len(raw) == item["size"],
            "frozen Integral24 manifest differs")
    manifest = json.loads(raw)
    require(set(manifest["files"]) == set(intent["source_payloads"]), "incomplete Integral24 frozen package")
    directory = out / "frozen-integral24"
    directory.mkdir()
    (directory / "source-manifest.json").write_bytes(raw)
    (directory / "source-manifest.json").chmod(0o660)
    for name, item in intent["source_payloads"].items():
        require(Path(name).name == name, "unexpected Integral24 member path")
        raw = item["text"].encode()
        require(item["eof"] is True and sha(raw) == item["sha256"] == manifest["files"][name]["sha256"]
                and len(raw) == item["size"] == manifest["files"][name]["bytes"], "frozen Integral24 member differs")
        path = directory / name
        path.write_bytes(raw)
        path.chmod(manifest["files"][name]["mode"])
    package(directory, INTEGRAL_SHA, inputs)
    return directory


def derived(original, edits, path, inputs):
    changed = original
    for before, after, count in edits:
        require(changed.count(before) == count and after not in changed, "derived source edit is not exact")
        changed = changed.replace(before, after)
    inverse = changed
    for before, after, count in reversed(edits):
        require(inverse.count(after) == count, "derived source inverse count differs")
        inverse = inverse.replace(after, before)
    require(inverse == original, "derived source inverse differs from entire original")
    path.write_bytes(changed)
    inputs.bind(path, sha(changed))
    return {"original_sha256": sha(original), "derived_sha256": sha(changed),
            "original_bytes": len(original), "derived_bytes": len(changed), "whole_inverse_exact": True,
            "edits": [{"before": a.decode(), "after": b.decode(), "count": n} for a, b, n in edits]}


def admit_sources(args, out, inputs, functions):
    original = args.original_source.resolve()
    parent = args.parent_source.resolve()
    package(original, ORIGINAL_SHA, inputs)
    package(parent, PARENT_SHA, inputs)
    integral = recover_integral(out, inputs)
    adaptation = out / "adapted"
    adaptation.mkdir()
    sys.path[:0] = [str(original), str(integral)]
    reference_reader = load(original / "admit_reference32.py", "admit_reference32")
    inputs.bind(original / "admit_reference32.py", REFERENCE_ADMITTER_SHA)
    raw = inputs.bind(original / "replay_original_sum32.py")
    # Preserve the original package/path bindings while allowing the verified
    # packages to live at new physical locations. No SumCase predicate changes.
    edits = [
        (b'cfg["source_package"]["path"]==str(integral_root/"source-manifest.json")',
         b'cfg["source_package"]["path"]==manifest["integral_native_package"]["path"]', 1),
        (b'cfg["parent_package"]["path"]==str(parent_root/"source-manifest.json")',
         b'cfg["parent_package"]["path"]==manifest["parent_source_package"]["path"]', 1),
    ]
    reader_path = adaptation / "replay_sum32_bool700.py"
    proofs = {"source_location_adaptation": derived(raw, edits, reader_path, inputs)}
    start, end = b"\ndef case_class(", b"\ndef unchanged_inputs("
    body = raw[raw.index(start):raw.index(end)]
    changed = reader_path.read_bytes()
    require(body == changed[changed.index(start):changed.index(end)] and len(body) == 27544 and sha(body) == CASE_SHA,
            "the complete original SumCase changed")
    replay = load(reader_path, "current_runtime_original_sum32")
    original_inputs = {}
    replay.admit_regression_package(original, ORIGINAL_SHA, parent, integral, original_inputs)
    integral_replay = load(integral / "replay_integral24.py", "current_runtime_integral_source_admission")
    integral_replay.admit_integral_package(integral, INTEGRAL_SHA, parent, original_inputs)
    for path, row in original_inputs.items():
        inputs.bind(path, row["record"]["sha256"])
        require(inputs.rows[path] == row["record"], "original source admission records differ")
    owner_raw = inputs.bind(integral / "owner_integral.py")
    observer_raw = inputs.bind(integral / "native_observer_integral.py")
    parent_observer = inputs.bind(parent / "native_observer_sum02.py")
    def constructor(source):
        base = source.index(b"class Native(ParentNative):")
        at = source.index(b"    def __init__(", base)
        return source[at:source.index(b"\n    def ", at + 1)]
    require(constructor(observer_raw) == constructor(parent_observer) and len(constructor(observer_raw)) == 2514
            and sha(constructor(observer_raw)) == CONSTRUCTOR_SHA, "the complete original Native constructor differs")
    tags = [(b"HTSINT01", b"HTSINT02", 2), (b"HTOSIN01", b"HTOSIN02", 2)]
    owner_edits = tags + [(b"functions == 699", ("functions == %d" % functions).encode(), 1),
        (b"sum-integral-source01-UNBOUND-699", ("current-runtime-UNBOUND-%d" % functions).encode(), 1),
        (b"sum-integral-source01-%d-preserved47-48", b"sum-bool-source01-%d-preserved47-48", 1),
        (b"heptatorch.sum-integral-owner-source01.v1", b"heptatorch.sum-bool-owner.v1", 1)]
    owner_edits = [edit for edit in owner_edits if edit[0] != edit[1]]
    owner_path, observer_path = adaptation / "owner_current.py", adaptation / "observer_current.py"
    proofs["owner"] = derived(owner_raw, owner_edits, owner_path, inputs)
    proofs["observer"] = derived(observer_raw, [(a, b, 1) for a, b, _ in tags], observer_path, inputs)
    require(constructor(observer_path.read_bytes()) == constructor(observer_raw), "derived Native constructor changed")
    proofs.update(whole_SumCase_unchanged=True, whole_SumCase_bytes=len(body), whole_SumCase_sha256=sha(body),
                  whole_reference_admitter_unchanged=True, reference_admitter_sha256=REFERENCE_ADMITTER_SHA,
                  original_Native_constructor_unchanged=True, original_Native_constructor_sha256=CONSTRUCTOR_SHA,
                  original_source_manifest_sha256=ORIGINAL_SHA, parent_source_manifest_sha256=PARENT_SHA,
                  integral_source_manifest_sha256=INTEGRAL_SHA, old_pass_transferred=False,
                  original_comparison_assertions_or_expected_values_modified=False)
    dump(out / "source-bindings.json", proofs)
    return replay, reference_reader, load(owner_path, "current_sum32_owner"), load(observer_path, "current_sum32_observer")


def admit_reference(args, out, inputs, reader):
    reference_path, physical_root = args.reference.resolve(), args.reference_source.resolve()
    raw = inputs.bind(reference_path, REFERENCE_SHA)
    require(len(raw) == 39045926, "original reference byte count differs")
    data = json.loads(raw)
    manifest = json.loads(inputs.bind(physical_root / "source-manifest.json", REFERENCE_MANIFEST_SHA))
    original_root = OLD_RUNS / "sum-dims-f32-reference32-source-01"
    mapping = {str(original_root / "source-manifest.json"): physical_root / "source-manifest.json"}
    require(len(manifest["files"]) == 38, "original reference source payload count differs")
    for item in manifest["files"]:
        name = item["path"]
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and str(Path(name)) == name,
                "unsafe reference source member")
        mapping[str(original_root / name)] = physical_root / name
    actual_records = {row["path"]: row for row in data["inputs_before"]}
    require(len(actual_records) == 41 and set(mapping) <= set(actual_records), "original reference input paths differ")
    external = set(actual_records) - set(mapping)
    known_external = {reader.PIP_SHA: "inputs/pytorch-install.json", reader.INVENTORY_SHA: "inputs/upstream-lock.json"}
    require(len(external) == 2 and {actual_records[p]["sha256"] for p in external} == set(known_external),
            "unexpected external original reference inputs")
    copies = out / "reference-external"
    copies.mkdir()
    for index, old in enumerate(sorted(external)):
        record = actual_records[old]
        source = physical_root / known_external[record["sha256"]]
        payload = inputs.bind(source, record["sha256"])
        require(len(payload) == record["bytes"], "external reference input bytes differ")
        local = copies / ("input-%d.json" % index)
        local.write_bytes(payload)
        local.chmod(record["mode"])
        mapping[old] = local
    base_read = reader.file_bytes
    relocations = []
    for old, physical in sorted(mapping.items()):
        inputs.bind(physical, actual_records[old]["sha256"])
        local = inputs.rows[str(physical.resolve())]
        require({**local, "path": old} == actual_records[old], "relocated original input bytes/mode differ")
        relocations.append({"original_path": old, "physical_path": str(physical.resolve()),
                            "bytes": local["bytes"], "mode": local["mode"], "sha256": local["sha256"]})
    def relocated_file_bytes(path, expected=None, admitted=None):
        requested = str(Path(path).resolve())
        if requested == str(reference_path):
            physical, logical = reference_path, requested
        else:
            require(requested in mapping, "reference reader requested an unmapped path")
            physical, logical = mapping[requested], requested
        payload, record = base_read(physical, expected, admitted)
        inputs.bind(physical, record["sha256"])
        return payload, {**record, "path": logical}
    reader.file_bytes = relocated_file_bytes
    try:
        reference, contract, admission = reader.admit_reference(reference_path, original_root, {})
    finally:
        reader.file_bytes = base_read
    require(sha(reference_path.read_bytes()) == REFERENCE_SHA, "original reference changed during admission")
    dump(out / "reference-bindings.json", {"reference_sha256": REFERENCE_SHA,
         "original_reference_json_unchanged": True, "original_admission_code_unchanged": True,
         "adaptation": "Read verified physical files; preserve all original canonical path, SHA, byte and mode comparisons.",
         "relocated_input_records": relocations, "actual_Torch_executed_here": False})
    return reference, contract, admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--expected-build-sha256", required=True)
    parser.add_argument("--original-source", type=Path, required=True)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(not sys.flags.optimize, "all original assertions must remain enabled")
    sys.dont_write_bytecode = True
    out = args.output.resolve()
    for folder in (args.build_directory, args.original_source, args.parent_source, args.reference_source):
        require(out != folder.resolve() and not out.is_relative_to(folder.resolve()), "output must be independent of inputs")
    out.mkdir(parents=True, exist_ok=False)
    inputs, reports = Inputs(), []
    started = time.monotonic()
    result = {"schema": "heptatorch.current-runtime.original-sum32-comparison.v1", "status": "PREPARING",
              "profile": "current/HTSINT02/HTOSIN02", "descriptor_bytes": 672,
              "state_format": "HTSINT02", "backing_format": "HTOSIN02", "case_count": 32, "event_count": 626,
              "reference_sha256": REFERENCE_SHA, "native_executed": False, "Torch_executed": False,
              "compiler_executed": False, "cases": reports, "old_PASS_transferred": False,
              "schema_closure": False, "full_project_status": "HOLD",
              "all32_execution_index": [{"index": i, "status": "NOT_RUN"} for i in range(32)]}
    exit_code = 1
    try:
        inputs.bind(Path(__file__).resolve())
        helper_path = Path(__file__).with_name("run.py")
        inputs.bind(helper_path, LEGACY_RUNNER_SHA)
        helper = load(helper_path, "current_runtime_build_admission")
        profile = helper.build_binding(args.build_directory.resolve(), args.expected_build_sha256, out, inputs.bind)
        inputs.bind(profile / "heptatorch", helper.NATIVE_SHA)
        inputs.bind(profile / "runtime.hepta", helper.SOURCE_SHA)
        inputs.bind(profile / "actual-build.json", args.expected_build_sha256)
        result.update(source_sha256=helper.SOURCE_SHA, native_sha256=helper.NATIVE_SHA,
                      functions=helper.FUNCTIONS, units=helper.UNITS,
                      actual_build_sha256=args.expected_build_sha256,
                      build_admission_code_sha256=LEGACY_RUNNER_SHA)
        replay, reference_reader, owner, observer = admit_sources(args, out, inputs, helper.FUNCTIONS)
        reference, contract, admission = admit_reference(args, out, inputs, reference_reader)
        result["reference_admission"] = admission
        if args.prepare_only:
            result["status"], exit_code = "PREPARED_NOT_RUN", 0
        else:
            parent_root = args.parent_source.resolve()
            parent = load(parent_root / "vendor/replay_direct_view14.py", "current_sum32_original_case")
            observer.configure(owner, helper.SOURCE_SHA, helper.NATIVE_SHA, helper.FUNCTIONS)
            Case = replay.case_class(parent, owner)
            run_args = argparse.Namespace(binary=profile / "heptatorch", source=profile / "runtime.hepta",
                                          protocol=parent_root / "vendor/protocol.py")
            result["native_executed"] = True
            for spec, row in zip(contract["cases"], reference["rows"]):
                directory = out / ("case-%02d" % spec["index"])
                try:
                    case = Case(run_args, observer, reference, row, spec, directory)
                    report = case.run()
                except BaseException as error:
                    report = {"index": spec["index"], "name": spec["name"], "status": "FAIL",
                              "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
                              "completed_reference_events": [], "reference_event_count": len(row["events"])}
                    directory.mkdir(exist_ok=True)
                    if not (directory / "report.json").exists():
                        dump(directory / "report.json", report)
                reports.append(report)
                result["all32_execution_index"][spec["index"]] = {"index": spec["index"], "status": report["status"]}
                print(json.dumps({"case": spec["index"], "status": report["status"],
                                  "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
            require(len(reports) == 32 and [r["index"] for r in reports] == list(range(32)), "not all original cases ran")
            result.update(passed=sum(r["status"] == "PASS" for r in reports),
                          blocked=sum(r["status"] == "BLOCKED" for r in reports),
                          failed=sum(r["status"] == "FAIL" for r in reports),
                          completed_reference_events=sum(len(r["completed_reference_events"]) for r in reports))
            for key in ("frames", "assertions", "exact_words", "full_rng_bytes", "rollback_bytes", "complete_frame_count"):
                result[key] = sum(r.get(key, 0) for r in reports)
            transports = []
            for index in range(32):
                path = out / ("case-%02d" % index) / "raw/transport.json"
                if path.is_file():
                    row = json.loads(path.read_bytes())
                    transports.append({k: row[k] for k in ("exit_code", "stdout_eof", "stderr_eof", "stderr_bytes",
                                       "extra_stdout_bytes", "forced_cleanup", "cleanup_errors", "frames", "executable_sha256")})
            result["transports"] = transports
            result["all_native_EOF"] = len(transports) == 32 and all(
                t["stdout_eof"] is True and t["stderr_eof"] is True and t["exit_code"] == 0
                and t["stderr_bytes"] == t["extra_stdout_bytes"] == 0 and not t["forced_cleanup"]
                and t["cleanup_errors"] == [] and t["executable_sha256"] == helper.NATIVE_SHA for t in transports)
            complete = result["passed"] == 32 and result["completed_reference_events"] == 626 and result["all_native_EOF"]
            result["status"] = "PASS" if complete else "FAIL" if result["failed"] else "BLOCKED"
            exit_code = 0 if complete else 1
    except BaseException as error:
        result.update(status="FAIL", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    finally:
        errors = inputs.recheck()
        result.update(input_recheck_errors=errors, all_inputs_unchanged=not errors,
                      checked_inputs=inputs.rows, elapsed_seconds=time.monotonic() - started)
        if errors:
            result["status"], exit_code = "FAIL", 1
        dump(out / "report.json", result)
    print(json.dumps({"status": result["status"], "report": str(out / "report.json")}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
