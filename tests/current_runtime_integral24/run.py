#!/usr/bin/env python3
"""Execute every original Integral24 comparison on the cold-built Bool700 image.

The original Torch observations, comparison class and five historical dtype
controls are loaded byte-for-byte. Source paths are relocated explicitly;
the current build and owner tags use separately checked admission bridges.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
INTENT = "continuation/20261117/evidence/root-sum-integral-native24-reviewed-intent-v2.json"
INTENT_SHA = "4050bf41236f33080d590d0b65bbf1359f9834d4001eb64883a62cbef2153efc"
PACKAGE_SHA = "6eee1fcd26367af3fba97488118e419788d64f41f90901b37cc594fe38154a9f"
REFERENCE_SHA = "6c9df4b579664c89c200b9d796367dd6d41f93f8260e8ff012f17d5041d2278b"
REFERENCE_BINDING_SHA = "40098ea257abf0cf170b3d4625fd7118416c685322e6ab9086272a56749c7dff"
SOURCE_SHA = "b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4"
IMAGE_SHA = "c6afeedc14ba8cc408a62d6fbef0abb5b78dff0039b8099f3449d813e290a90e"
BOOL_SUPPORT = {
    "cases.py": "cb489ce173d6cf3984d86c796b516c490b07c2a5d48639a3cdded5cc58187508",
    "common.py": "073bfd1c717587e31dacb71d050ca19b0e128ef456a0696204d0dd079b5c5eba",
    "controls.py": "cf883730b6ba116f81fc3f8222d76a6b196425717f73756e9caea4d496fba287",
    "native.py": "7c5a2ead8dd9c6a39835991095880a6c61e6ab18fb034172163b2f440c486b2d",
    "owner_adapter.py": "5a32e6995a8e34f81b14e24c271de496e48d62b23f85a3f16fcb1989febe16fa",
    "reference.py": "5ed8900f8cde38b6491db834c3a233e4a7621d9b2ccd808c5dd822cb78b9736c",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, rule, **facts):
    if not condition:
        raise AssertionError(dict(rule=rule, **facts))


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Inputs:
    def __init__(self):
        self.rows = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        require(not path.is_symlink(), "regular-input-no-symlink", path=str(path))
        before = path.stat()
        require(stat.S_ISREG(before.st_mode), "regular-input", path=str(path))
        raw = path.read_bytes()
        after = path.stat()
        fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)
        require(fields(before) == fields(after), "stable-input-read", path=str(path))
        record = dict(path=str(path), bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(after.st_mode))
        require(expected is None or record["sha256"] == expected, "input-sha256", path=str(path))
        require(str(path) not in self.rows or self.rows[str(path)] == record,
                "input-same-on-reread", path=str(path))
        self.rows[str(path)] = record
        return raw

    def unchanged(self):
        original = dict(self.rows)
        for name, record in original.items():
            self.bind(name, record["sha256"])
        require(original == self.rows, "all-input-bytes-and-modes-unchanged")
        return True


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bool_support(inputs):
    folder = ROOT / "tests/sum_bool_review"
    for name, digest in BOOL_SUPPORT.items():
        inputs.bind(folder / name, digest)
    sys.path.insert(0, str(folder))
    native = importlib.import_module("native")
    adapter = importlib.import_module("owner_adapter")
    return native, adapter


def recover_original_package(out, inputs):
    intent = json.loads(inputs.bind(ROOT / INTENT, INTENT_SHA))
    entry = intent["source_manifest"]
    raw = entry["text"].encode()
    require(entry["eof"] is True and len(raw) == entry["size"] and
            sha(raw) == entry["sha256"] == PACKAGE_SHA, "complete-original-native24-manifest")
    manifest = json.loads(raw)
    require(set(manifest["files"]) == set(intent["source_payloads"]) and
            manifest["file_count"] == len(manifest["files"]) == 19,
            "complete-nineteen-original-payloads")
    frozen = out / "frozen"
    frozen.mkdir()
    total = 0
    proof = []
    for name, row in intent["source_payloads"].items():
        require(Path(name).name == name, "safe-frozen-payload-name")
        value, pin = row["text"].encode(), manifest["files"][name]
        require(row["eof"] is True and len(value) == row["size"] == pin["bytes"] and
                sha(value) == row["sha256"] == pin["sha256"], "whole-original-payload", name=name)
        path = frozen / name
        path.write_bytes(value)
        path.chmod(pin["mode"])
        inputs.bind(path, pin["sha256"])
        total += len(value)
        proof.append(dict(name=name, **pin, original_bytes_equal=True, source_edits=[]))
    require(total == manifest["payload_bytes"], "complete-original-payload-byte-count")
    path = frozen / "source-manifest.json"
    path.write_bytes(raw)
    path.chmod(0o660)
    inputs.bind(path, PACKAGE_SHA)
    dump(out / "original-source-preservation.json", dict(
        schema="heptatorch.current-runtime.integral24-source-preservation.v1",
        original_intent_sha256=INTENT_SHA, original_manifest_sha256=PACKAGE_SHA,
        complete_original_files=19, original_payload_bytes=total, files=proof,
        original_reference_validator_source_unchanged=True, read_path_bridge_installed=True,
        original_comparison_class_modified=False,
        original_historical_five_controls_modified=False, old_native_pass_transferred=False))
    return frozen


class ReferencePaths:
    """Relocate reads while preserving every original path/byte/mode assertion.

The original file_bytes function still performs each real file read. The
returned logical path is the immutable reference's original path. Physical
paths and their exact metadata are independently retained for final recheck.
"""
    def __init__(self, module, binding, args, inputs):
        self.original_read = module.file_bytes
        self.inputs = inputs
        self.physical_inputs = {}
        self.rows = {}
        self.roots = [
            (Path(binding["reference_source_manifest"]["path"]).parent, args.reference24_source),
            (Path(binding["parent_reference_manifest"]["path"]).parent, args.reference32_source),
            (Path(binding["integral_source_manifest"]["path"]).parent, args.integral_source),
        ]
        self.files = {Path(binding["actual"]["path"]): args.reference}
        for pin in binding["external_repo_files"]:
            if pin["sha256"] == module.PIP_SHA:
                self.files[Path(pin["path"])] = args.pip_report
            elif pin["sha256"] == module.INVENTORY_SHA:
                self.files[Path(pin["path"])] = args.upstream_lock
            else:
                raise AssertionError("unmapped original external source")
        physical = [x[1] for x in self.roots] + list(self.files.values())
        require(len(physical) == len(set(physical)), "distinct-relocated-reference-roots")

    def paths(self, requested):
        requested = Path(requested).absolute()
        for logical, actual in self.files.items():
            if requested in (logical, actual):
                return logical, actual
        for logical, actual in self.roots:
            if requested == logical or requested.is_relative_to(logical):
                return requested, actual / requested.relative_to(logical)
            if requested == actual or requested.is_relative_to(actual):
                return logical / requested.relative_to(actual), requested
        require(not str(requested).startswith("/home/qian-qi/heptatorch-continuation-20261116/"),
                "all-original-reference-paths-explicitly-relocated", path=str(requested))
        return requested, requested

    def read(self, path, expected=None, inputs=None):
        logical, actual = self.paths(path)
        raw, physical = self.original_read(actual, expected, self.physical_inputs)
        require(self.inputs.bind(actual, expected) == raw, "same-physical-reference-read")
        record = {**physical, "path": str(logical)}
        key = str(actual.resolve())
        if inputs is not None:
            entry = dict(raw=raw, record=record)
            require(key not in inputs or inputs[key] == entry, "same-relocated-input-reread", path=key)
            inputs[key] = entry
        row = dict(logical_path=str(logical), physical=physical)
        require(key not in self.rows or self.rows[key] == row, "stable-path-relocation", path=key)
        self.rows[key] = row
        return raw, record


def execute_cases(replay, parent, observer, owner, run_args, reference, contract, directory):
    directory.mkdir()
    Case = replay.case_class(parent, owner)
    reports = []
    for spec, row in zip(contract["cases"], reference["rows"]):
        target = directory / ("case-%02d" % spec["id"])
        try:
            case = Case(run_args, observer, reference, row, spec, target)
            result = case.run()
        except BaseException as failure:
            result = dict(index=spec["id"], name=spec["label"], status="FAIL",
                error=dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc()),
                reference_event_count=len(row["events"]), completed_reference_events=[],
                unattempted_reference_events=[dict(index=i, label=e["label"]) for i, e in enumerate(row["events"])],
                schema_closure=False, full_project_status="HOLD")
            target.mkdir(exist_ok=True)
            if not (target / "report.json").exists():
                dump(target / "report.json", result)
        reports.append(result)
        print(json.dumps(dict(index=spec["id"], name=spec["label"], status=result["status"])), flush=True)
    require(len(reports) == 24 and [r["index"] for r in reports] == list(range(24)),
            "every-original24-case-executed-in-order")
    return reports


def classify_original_controls(result):
    """Retain the original failures caused by the two explicitly added APIs."""
    require(result["status"] == "FAIL" and result["case_count"] == 5 and
            result["passed"] == 3 and result["failed"] == 2, "original-five-recorded-with-new-contract")
    require([r["index"] for r in result["cases"]] == list(range(5)), "all-five-original-control-indices")
    changed = []
    retained = []
    for index, row in enumerate(result["cases"]):
        require(row["exit_code"] == 0, "original-control-native-exit", index=index)
        if index in (0, 2):
            require(row["status"] == "FAIL" and row["observations"]["frame_status"] == 0 and
                    row["observations"]["record_count"] == 1 and len(row["checks"]) == 1 and
                    row["checks"][0] == dict(label="native-cause7", actual=0, expected=7, check=False),
                    "only-original-refusal-expectation-superseded", index=index)
            error = row["error"]
            require(error["type"] == "AssertionError" and error["message"] == str(dict(
                    reference_rule="bounded-api-control", control=row["name"], label="native-cause7"))
                    and "EOF_failure" not in error,
                    "specific-original-control-failure", index=index)
            command = row["observations"]["command"]
            require(command[0] == 115 and command[5] == 1,
                    "explicit-Bool-success-is-intended-extension", index=index)
            changed.append(index)
        else:
            require(row["status"] == "PASS" and row["error"] is None and
                    row["observations"]["frame_status"] == 7 and
                    row["observations"]["record_count"] == 0 and
                    all(c["check"] is True for c in row["checks"]),
                    "original-scope-refusal-retained", index=index)
            retained.append(index)
    return dict(original_controls_contract_changed=len(changed),
                original_controls_changed_indices=changed, retained_refusals=len(retained),
                retained_refusal_indices=retained, original_five_all_pass=False,
                current_dtype_contract_qualified_here=False,
                current_dtype_gate="tests/sum_bool_review/native.py separate seven-control qualification")


def transport_summary(directory):
    paths = sorted(directory.rglob("transport.json"))
    require(len(paths) == 29, "all-twenty-nine-native-transport-receipts", actual=len(paths))
    rows = []
    for path in paths:
        raw = path.read_bytes()
        row = json.loads(raw)
        require(row["exit_code"] == 0 and row["stdout_eof"] is True and row["stderr_eof"] is True
                and row["stderr_bytes"] == row["extra_stdout_bytes"] == 0 and
                row["responses_before_stdin_eof"] is True and row["forced_cleanup"] is False and
                row["cleanup_errors"] == [] and row["executable_sha256"] == IMAGE_SHA,
                "native-exit-and-complete-streams", path=str(path))
        require(row["frames"] == row["complete_owner_boundaries"] == len(row["rows"]),
                "every-frame-complete-owner-boundaries", path=str(path))
        rows.append(dict(path=str(path.relative_to(directory)), sha256=sha(raw), frames=row["frames"],
                         exit_code=0, stdout_eof=True, stderr_eof=True))
    return dict(children=29, frames=sum(r["frames"] for r in rows), all_exit_zero=True,
                all_stdout_eof=True, all_stderr_eof=True, complete_owner_boundaries=True, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-directory", "reference", "reference24-source", "reference32-source",
                 "integral-source", "parent-native-package", "pip-report", "upstream-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-build-sha256", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(not sys.flags.optimize, "all-original-assertions-enabled")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.absolute())
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, logical_inputs = Inputs(), {}
    report = dict(schema="heptatorch.current-runtime.integral24.v1", status="PREPARING",
        source_sha256=SOURCE_SHA, native_sha256=IMAGE_SHA, reference_sha256=REFERENCE_SHA,
        expected_cases=24, original_event_count=260, native_executed=False, Torch_executed=False,
        compiler_executed=False, old_native_pass_transferred=False, cases=[], schema_closure=False,
        full_transition_recomputed=False, full_project_status="HOLD", error=None)
    paths = replay = None
    exit_code = 1
    try:
        inputs.bind(Path(__file__).resolve())
        bool_native, adapter = bool_support(inputs)
        build_args = argparse.Namespace(repository=ROOT, source=args.build_directory / "runtime.hepta",
            binary=args.build_directory / "heptatorch", build_report=args.build_directory / "build.json",
            expected_source_sha256=SOURCE_SHA, expected_native_sha256=IMAGE_SHA,
            expected_build_sha256=args.expected_build_sha256)
        build = bool_native.admit_build(build_args, inputs)
        report.update(actual_build_sha256=args.expected_build_sha256, functions=build["functions"],
                      native_units=build["units_planned"], state_format="HTSINT02", backing_format="HTOSIN02",
                      descriptor_bytes=672, build_admission="same pinned complete cold build as Bool27")
        frozen = recover_original_package(args.output, inputs)
        reference_module = load_module(frozen / "admit_reference24.py", "admit_reference24")
        binding_raw = inputs.bind(frozen / "actual-reference-binding.json", REFERENCE_BINDING_SHA)
        paths = ReferencePaths(reference_module, json.loads(binding_raw), args, inputs)
        reference_module.file_bytes = paths.read
        sys.path.insert(0, str(frozen))
        replay = load_module(frozen / "replay_integral24.py", "original_integral24_replay")
        replay.admit_integral_package(frozen, PACKAGE_SHA, args.parent_native_package, logical_inputs)
        reference, contract, admission = replay.admit_reference(
            frozen / "actual-reference-binding.json", REFERENCE_BINDING_SHA, logical_inputs)
        require(admission["case_count"] == 24 and admission["operation_totals"]["attempted"] == 260,
                "original24-and260-reference-denominators")
        report["reference_admission"] = admission
        dump(args.output / "reference-paths.json", dict(
            schema="heptatorch.current-runtime.integral24-paths.v1", original_validator_bytes_unchanged=True,
            original_file_bytes_called_for_every_read=True, original_metadata_assertions_unchanged=True,
            relocations=list(paths.rows.values())))
        observer, owner, proof = adapter.make_observer(ROOT, inputs, SOURCE_SHA, IMAGE_SHA)
        report["owner_adaptation"] = proof
        report["original24_comparison_source_sha256"] = sha(inputs.bind(frozen / "replay_integral24.py"))
        report["original5_control_source_sha256"] = sha(inputs.bind(frozen / "typed_scope_controls.py"))
        if args.prepare_only:
            report.update(status="PREPARED_NOT_RUN", original_controls_contract_changed=2,
                          original_controls_changed_indices=[0, 2], retained_refusals=3,
                          historical_control_status="NOT_RUN", completed_reference_events=0)
            exit_code = 0
        else:
            parent = load_module(args.parent_native_package / "vendor/replay_direct_view14.py", "original_integral24_parent")
            run_args = argparse.Namespace(binary=build_args.binary, source=build_args.source,
                                          protocol=args.parent_native_package / "vendor/protocol.py")
            directory = args.output / "native"
            report["native_executed"] = True
            report["cases"] = execute_cases(replay, parent, observer, owner, run_args,
                                             reference, contract, directory)
            reports = report["cases"]
            report.update(passed=sum(r["status"] == "PASS" for r in reports),
                          failed=sum(r["status"] == "FAIL" for r in reports),
                          blocked=sum(r["status"] == "BLOCKED" for r in reports),
                          completed_reference_events=sum(len(r["completed_reference_events"]) for r in reports))
            controls = load_module(frozen / "typed_scope_controls.py", "original_integral24_controls")
            historical = controls.run_controls(observer, owner, run_args, directory / "historical-controls")
            report["historical_controls"] = historical
            report.update(classify_original_controls(historical))
            report["transport"] = transport_summary(directory)
            complete = (report["passed"] == 24 and report["failed"] == report["blocked"] == 0 and
                report["completed_reference_events"] == report["original_event_count"] == 260)
            report["status"] = "PASS" if complete else "FAIL"
            exit_code = 0 if complete else 1
    except BaseException as failure:
        report.update(status="FAIL", error=dict(type=type(failure).__name__, message=str(failure),
                                                traceback=traceback.format_exc()))
    finally:
        try:
            report["original_input_recheck_errors"] = replay.unchanged_inputs(logical_inputs) if replay else []
            require(not report["original_input_recheck_errors"], "original-input-final-recheck")
            report["all_inputs_unchanged"] = inputs.unchanged()
        except BaseException as failure:
            report.update(status="FAIL", final_recheck_error=dict(type=type(failure).__name__, message=str(failure)))
            exit_code = 1
        report["inputs"] = inputs.rows
        report["original_logical_inputs"] = {name: row["record"] for name, row in logical_inputs.items()}
        dump(args.output / "report.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "passed", "completed_reference_events",
        "retained_refusals", "original_controls_contract_changed", "native_executed", "error")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
