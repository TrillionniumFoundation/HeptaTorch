#!/usr/bin/env python3
"""Seal one completed Integral24 native run as complete original bytes and modes.

This entry creates a new archive and binding. It does not execute the comparison,
a Tensor image, Torch, the compiler, or the later recorded semantic admission.
The original producer status is retained even when it is FAIL.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import traceback

HERE = Path(__file__).resolve().parent
_COMMON_SHA = "a4ac713abf15756548d6b0be15c1d8287f73e6f4dee40a812415c1daf160f207"
_common_path = HERE/"admission_common.py"
if hashlib.sha256(_common_path.read_bytes()).hexdigest() != _COMMON_SHA:
    raise AssertionError("exact-pure-common-source-required-before-import")
_spec = importlib.util.spec_from_file_location("admission_common", _common_path)
_common = importlib.util.module_from_spec(_spec)
sys.modules["admission_common"] = _common
_spec.loader.exec_module(_common)
from admission_common import SHA, BOUND, AUDIT, need, read, data, write, same, load, recheck_bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(not sys.flags.optimize, "source-assertions-enabled")
    output = args.output.resolve()
    need(output != HERE and not output.is_relative_to(HERE), "independent-create-only-seal-output")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "heptatorch.sum-integral24.raw-sealing.v1", "status": "PREPARING",
              "native_admission_performed": False, "native_executed": False,
              "Torch_executed": False, "compiler_executed": False, "errors": [],
              "all_original_files_unchanged": False}
    exit_code, binding, manifest, archive = 1, None, None, None
    try:
        package_data = data(HERE/"source-manifest.json", args.package_sha256)
        sc = load(HERE/"source_custody.py", "source_custody",
                  package_data["files"]["source_custody.py"]["sha256"])
        package, profile, profile_meta = sc.self_package(HERE, args.package_sha256)
        archive = load(HERE/"archive_custody.py", "archive_custody",
                       package["files"]["archive_custody.py"]["sha256"])
        plan_raw = read(args.plan, args.plan_sha256)
        plan = json.loads(plan_raw)
        need(set(plan) == {"schema", "status", "profile_sha256", "native_output",
             "native_report", "execution_evidence"} and
             plan["schema"] == "heptatorch.sum-integral24.raw-seal-plan.v1" and
             plan["status"] == "ACTUAL_NATIVE_RUN_BOUND" and
             plan["profile_sha256"] == profile_meta["sha256"],
             "actual-terminal-seal-plan-not-placeholder")
        native_root = Path(plan["native_output"]).resolve()
        need(output != native_root and not output.is_relative_to(native_root)
             and not native_root.is_relative_to(output), "seal-cannot-modify-native-output")
        report_pin = plan["native_report"]
        need(set(report_pin) == {"path", "bytes", "sha256", "mode"}
             and Path(report_pin["path"]).resolve() == native_root/"report.json",
             "explicit-original-top-report-member")
        same(report_pin, profile["actual_native_report"], "exact-completed-native24-original-report")
        report_raw = read(report_pin["path"], report_pin["sha256"], report_pin["bytes"], report_pin["mode"])
        native = json.loads(report_raw)
        need(native["schema"] == profile["native_schema"]
             and native["source_revision"] == profile["source_revision"]
             and native["source_sha256"] == profile["runtime"]["sha256"]
             and native["native_sha256"] == profile["binary"]["sha256"]
             and native["package_sha256"] == profile["native_package"]["manifest"]["sha256"]
             and native["actual_build_binding_sha256"] == profile["actual_build_binding"]["sha256"],
             "actual-run-and-integral699-build-identity")
        need(native["native_executed"] is True and native["Torch_executed"] is False
             and native["compiler_executed"] is False and native["case_count"] == 24
             and native["reference_case_count"] == 24 and len(native["cases"]) == 24
             and [case["index"] for case in native["cases"]] == list(range(24)),
             "original-full24-run-report-not-preparation")
        need(native["original_event_count"] == 260 and native["status"] in ("PASS", "FAIL", "BLOCKED"),
             "original-case-event-denominators-retained")
        for key in ("functions", "blocks", "instructions", "units", "descriptor_bytes",
                    "state_format", "backing_format"):
            need(native[key] == profile["profile"][key], "exact-actual-profile-field", field=key)
        for case in native["cases"]:
            same(data(native_root/("case-%02d" % case["index"])/"report.json"), case,
                 "whole-case-report-in-original-top")
        case_summaries = [{"index": case["index"], "name": case["name"], "status": case["status"],
            "reference_events": case["reference_event_count"],
            "completed_events": len(case["completed_reference_events"]),
            "unattempted_events": len(case["unattempted_reference_events"]),
            "frames": case.get("frames"), "assertions": case.get("assertions"),
            "exit_code": case.get("exit_code"),
            "mismatch_count": len(case.get("mismatches", [])),
            "blocker_count": len(case.get("blockers", [])),
            "error_present": case.get("error") is not None} for case in native["cases"]]
        need(sum(case["reference_events"] for case in case_summaries) == native["original_event_count"] == 260
             and sum(case["completed_events"] for case in case_summaries) == native["completed_reference_events"],
             "bounded-summary-keeps-entire260-event-denominator")
        for status, field in (("PASS", "passed"), ("FAIL", "failed"), ("BLOCKED", "blocked")):
            need(sum(case["status"] == status for case in case_summaries) == native[field],
                 "bounded-summary-recomputes-all24-original-status-counts", status=status)
        controls = native["bounded_api_controls"]
        need(controls["schema"] == "heptatorch.sum-integral.bounded-dtype-controls.v1"
             and controls["case_count"] == len(controls["cases"]) == 5
             and [case["index"] for case in controls["cases"]] == list(range(5))
             and controls["Torch_parity_claim"] is False
             and controls["native_executed"] is True and controls["Torch_executed"] is False
             and controls["compiler_executed"] is False,
             "five-original-bounded-controls-have-separate-denominator")
        same(data(native_root/"bounded-api-controls"/"report.json"), controls,
             "entire-original-controls-report-in-top-report")
        control_summaries = []
        for case in controls["cases"]:
            same(data(native_root/"bounded-api-controls"/("control-%02d" % case["index"])/"report.json"),
                 case, "whole-original-control-report-in-control-summary")
            need(case["reference_case"] is None and case["Torch_parity_claim"] is False,
                 "no-control-counted-as-a-Torch-reference-case")
            control_summaries.append({
                "index": case["index"], "name": case["name"], "status": case["status"],
                "frames": case["frames"], "checks": len(case["checks"]),
                "exit_code": case["exit_code"], "error_present": case["error"] is not None})
        need(sum(case["status"] == "PASS" for case in controls["cases"]) == controls["passed"]
             and sum(case["status"] != "PASS" for case in controls["cases"]) == controls["failed"],
             "all-five-original-control-status-counts")
        original_files, original_dirs = archive.inventory(native_root)
        files = {"native-output/"+name: entry for name, entry in original_files.items()}
        directories = {"native-output"+("/"+name if name else ""): mode
                       for name, mode in original_dirs.items()}
        synthetic = set()

        def add_file(name, path, expected=None):
            from admission_common import safe_path, digest_file
            safe_path(output, name)
            need(name not in files, "unique-sealed-member", path=name)
            pin = digest_file(path)
            if expected is not None:
                same(pin, {key: expected[key] for key in ("bytes", "sha256", "mode")},
                     "declared-source-input-byte-mode")
            files[name] = {**pin, "source_path": str(Path(path).resolve())}
            parent = Path(name).parent
            while str(parent) != ".":
                label = parent.as_posix()
                if label not in directories:
                    directories[label] = 0o700
                    synthetic.add(label)
                parent = parent.parent

        declared_inputs = native["checked_inputs"]
        need(type(declared_inputs) is dict and declared_inputs,
             "nonempty-original-native-input-ledger")
        for i, (path, item) in enumerate(sorted(declared_inputs.items())):
            need(set(item) == {"path", "bytes", "sha256", "mode"} and item["path"] == path,
                 "entire-original-input-record")
            add_file("inputs/%04d-%s" % (i, Path(path).name), path, item)
        evidence = plan["execution_evidence"]
        need(type(evidence) is list and evidence, "original-execution-receipt-files-required")
        roles = set()
        for i, item in enumerate(evidence):
            need(set(item) == {"role", "path", "bytes", "sha256", "mode"}
                 and type(item["role"]) is str and item["role"] and item["role"] not in roles,
                 "explicit-unique-original-execution-evidence-role")
            roles.add(item["role"])
            add_file("execution/%04d-%s" % (i, Path(item["path"]).name), item["path"], item)
        add_file("execution/seal-plan.json", args.plan)
        for name, item in package["files"].items():
            add_file("pure-reader-source/"+name, HERE/name, item)
        add_file("pure-reader-source/source-manifest.json", HERE/"source-manifest.json")
        manifest = {"schema": archive.MEMBER_SCHEMA, "native_output": str(native_root),
            "native_report_sha256": report_pin["sha256"],
            "native_original_status": native["status"], "files": files, "directories": directories,
            "synthetic_container_directories": sorted(synthetic),
            "file_count": len(files), "directory_count": len(directories),
            "payload_bytes": sum(item["bytes"] for item in files.values()),
            "input_bindings": declared_inputs, "execution_evidence": evidence,
            "source_profile_sha256": profile_meta["sha256"],
            "mode_policy": "Every file retains its real source mode. Native directory modes are exact; explicitly synthetic container directories use0700.",
            "manifest_is_outside_archive": True,
            "native_Torch_compiler_or_recorded_admission_executed": False}
        manifest_meta = write(output/"file-manifest.json", manifest)
        same(data(output/"file-manifest.json", manifest_meta["sha256"],
                  manifest_meta["bytes"], manifest_meta["mode"]), manifest,
             "full-sealed-member-manifest-readback")
        archive_meta = archive.create_archive(output/"complete.tar.xz", manifest)
        archive_check = archive.verify_archive(output/"complete.tar.xz", manifest, archive_meta)
        original_check = archive.verify_originals(manifest)
        need(native["all_inputs_unchanged"] is True and native["input_recheck_errors"] == [],
             "original-native-input-recheck-preserved")
        result.update(status="PASS_CUSTODY_ONLY", native_original_status=native["status"],
            original_report_sha256=report_pin["sha256"],
            original_case_count=24, case_summaries=case_summaries,
            original_bounded_control_count=5, bounded_control_summaries=control_summaries,
            original_input_count=len(declared_inputs), original_native_inputs_unchanged=True,
            original_case_counts={
                "reference_cases": {"case_count": 24, **{key: native[key] for key in (
                    "passed", "failed", "blocked", "original_event_count", "completed_reference_events")}},
                "bounded_controls": {key: controls[key] for key in (
                    "case_count", "passed", "failed", "status", "Torch_parity_claim")}},
            file_manifest={**manifest_meta, "path": str(output/"file-manifest.json")},
            archive={**archive_meta, "path": str(output/"complete.tar.xz")},
            archive_verification=archive_check, original_verification=original_check,
            all_original_files_unchanged=True, native_admission_eligible=native["status"] == "PASS")
        binding = {"schema": archive.BINDING_SCHEMA, "status": "RAW_BYTES_SEALED",
            "profile_sha256": profile_meta["sha256"],
            "member_manifest": result["file_manifest"], "archive": result["archive"],
            "native_report": report_pin, "native_output": str(native_root),
            "native_status": native["status"], "native_case_counts": result["original_case_counts"],
            "execution_evidence": evidence, "input_bindings": declared_inputs,
            "sealing_report": None, "no_new_native_Torch_or_compiler_execution": True}
        exit_code = 0
    except BaseException as error:
        result["status"] = "FAIL"
        result["errors"].append({"type": type(error).__name__, "message": str(error),
                                 "traceback": traceback.format_exc()})
    finally:
        errors = recheck_bound()
        result.update(input_recheck_errors=errors, checked_inputs=BOUND,
                      forbidden_execution_attempts=dict(AUDIT))
        if errors or any(AUDIT.values()):
            result["status"], exit_code = "FAIL", 1
        report_meta = write(output/"report.json", result)
        if exit_code == 0:
            binding["sealing_report"] = {**report_meta, "path": str(output/"report.json")}
            write(output/"evidence-binding.json", binding)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
