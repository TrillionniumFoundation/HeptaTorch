#!/usr/bin/env python3
"""Reapply the immutable Integral24 Case and five controls to recorded native bytes.

This is a pure admission entry. Its only response/owner source is the sealed
original record. No Tensor executable, Torch, compiler, subprocess or /proc
read is permitted. The original native status is reconstructed, not relabelled.
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


def original_top(profile, context, cases, controls):
    build, binding = context["build"], context["binding"]
    reference, contract = context["reference"], context["contract"]
    result = {"schema":"heptatorch.sum-integral.native24-comparison.v1",
        "status":"PREPARING", "case_count":24, "reference_case_count":24,
        "reference_sha256":None, "case_contract_sha256":profile["reference"]["case_contract_sha256"],
        "source_revision":"sum-integral-source01", "source_sha256":None,
        "native_sha256":None, "functions":None, "units":None,
        "descriptor_bytes":672, "state_format":"HTSINT01", "backing_format":"HTOSIN01",
        "native_executed":False, "Torch_executed":False, "compiler_executed":False,
        "schema_closure":False, "full_project_status":"HOLD", "cases":cases,
        "all_observations_supported":False,
        "all24_execution_index":[{"index":i,"status":"NOT_RUN"} for i in range(24)],
        "bounded_api_controls":{"status":"NOT_RUN","Torch_parity_claim":False}}
    result.update(reference_admission=context["reference_admission"],
        build_admission=context["build_admission"],
        reference_sha256=context["reference_admission"]["reference_sha256"],
        source_sha256=binding["runtime"]["sha256"], native_sha256=binding["binary"]["sha256"],
        functions=build["functions"], blocks=build["blocks"], instructions=build["instructions"],
        units=binding["profile"]["units"], package_sha256=profile["native_package"]["manifest"]["sha256"],
        actual_build_binding_sha256=profile["actual_build_binding"]["sha256"],
        actual_reference_binding_sha256=profile["reference_binding"]["sha256"])
    # This object reconstructs the original producer. The containing pure report
    # separately declares that no native process or new Tensor observation ran.
    result["native_executed"] = True
    need(len(cases) == 24 and [row["index"] for row in cases] == list(range(24)),
         "all24-original-case-results-reconstructed")
    for row in cases:
        result["all24_execution_index"][row["index"]] = {"index":row["index"],"status":row["status"]}
    result["passed"] = sum(row["status"] == "PASS" for row in cases)
    result["blocked"] = sum(row["status"] == "BLOCKED" for row in cases)
    result["failed"] = sum(row["status"] == "FAIL" for row in cases)
    result["original_event_count"] = sum(len(row["events"]) for row in reference["rows"])
    result["completed_reference_events"] = sum(len(row["completed_reference_events"]) for row in cases)
    need(result["passed"]+result["blocked"]+result["failed"] == 24 and
         result["original_event_count"] == reference["operation_totals"]["attempted"] == 260,
         "complete24-and-actual260-event-denominators")
    result["bounded_api_controls"] = controls
    complete = (result["passed"] == 24 and
                result["completed_reference_events"] == result["original_event_count"] and
                result["bounded_api_controls"]["status"] == "PASS_BOUNDED_API_CONTROLS_ONLY")
    result["status"] = "FAIL" if result["failed"] or result["bounded_api_controls"]["status"] != "PASS_BOUNDED_API_CONTROLS_ONLY" else "PASS" if complete else "BLOCKED"
    result["all_observations_supported"] = complete
    result["exact_Python_diagnostic_text_generated_by_native"] = False
    result["exception_scope"] = "Original exact text retained; real native acceptance/cause/payload/full owner rollback compared. No diagnostic text is synthesized."
    errors = context["replay"].unchanged_inputs(context["inputs"])
    result["input_recheck_errors"] = errors
    result["all_inputs_unchanged"] = not errors
    result["checked_inputs"] = {name:value["record"] for name,value in context["inputs"].items()}
    if errors:
        result["status"] = "FAIL"
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--evidence-binding", type=Path, required=True)
    parser.add_argument("--evidence-binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--admit-recorded", action="store_true")
    args = parser.parse_args()
    need(not sys.flags.optimize, "original-assertions-enabled")
    output = args.output.resolve()
    need(output != HERE and not output.is_relative_to(HERE), "independent-create-only-pure-output")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "heptatorch.sum-integral24.complete-recorded-admission.v1",
              "status": "PREPARING", "native_executed": False, "Torch_executed": False,
              "compiler_executed": False, "case_count": 24, "original_event_count": 260,
              "cases": [], "all24_admission_index": [
                  {"index": i, "status": "NOT_ADMITTED"} for i in range(24)],
              "bounded_control_count": 5, "bounded_controls": {"status": "NOT_ADMITTED",
                  "reference_cases_added": 0, "Torch_parity_claim": False},
              "full_general_transition_recomputed": False, "schema_closure": False,
              "full_project_status": "HOLD"}
    exit_code, archive, member_manifest = 1, None, None
    try:
        raw_manifest = data(HERE/"source-manifest.json", args.package_sha256)
        sc = load(HERE/"source_custody.py", "source_custody",
                  raw_manifest["files"]["source_custody.py"]["sha256"])
        manifest, profile, profile_meta = sc.self_package(HERE, args.package_sha256)
        for name in ("archive_custody", "recorded_wire", "complete_owners", "control_owners",
                     "recorded_observer", "recorded_controls"):
            load(HERE/(name+".py"), name, manifest["files"][name+".py"]["sha256"])
        archive = sys.modules["archive_custody"]
        evidence, member_manifest, archive_result = archive.read_binding(
            args.evidence_binding, args.evidence_binding_sha256, profile_meta)
        native_root = Path(evidence["native_output"])
        need(not output.is_relative_to(native_root) and not native_root.is_relative_to(output),
             "pure-output-is-disjoint-from-original-native-evidence")
        same(evidence["native_report"], profile["actual_native_report"],
             "this-complete-original5f8-native24-report-not-a-substitute")
        original_raw = read(evidence["native_report"]["path"], evidence["native_report"]["sha256"],
                            evidence["native_report"]["bytes"], evidence["native_report"]["mode"])
        native_report = json.loads(original_raw)
        need(native_report["schema"] == profile["native_schema"] and
             native_report["source_sha256"] == profile["runtime"]["sha256"] and
             native_report["native_sha256"] == profile["binary"]["sha256"] and
             native_report["source_revision"] == profile["source_revision"],
             "original-native-report-is-this-actual-source-and-image")
        need(evidence["native_status"] == native_report["status"], "sealed-original-native-status")
        original_controls = native_report["bounded_api_controls"]
        same(evidence["native_case_counts"], {
             "reference_cases": {"case_count": 24, **{key: native_report[key] for key in (
                 "passed", "failed", "blocked", "original_event_count", "completed_reference_events")}},
             "bounded_controls": {key: original_controls[key] for key in (
                 "case_count", "passed", "failed", "status", "Torch_parity_claim")}},
             "sealed-original24-and5-denominators-remain-separate")
        need(original_controls["case_count"] == len(original_controls["cases"]) == 5
             and [row["index"] for row in original_controls["cases"]] == list(range(5))
             and original_controls["Torch_parity_claim"] is False,
             "all-five-original-bounded-control-reports")
        same(json.loads(read(native_root/"bounded-api-controls"/"report.json")), original_controls,
             "whole-original-control-summary-in-native-top-report")
        for row in original_controls["cases"]:
            same(json.loads(read(native_root/"bounded-api-controls"/("control-%02d" % row["index"])/"report.json")),
                 row, "whole-original-control-report-in-summary")
        context = sc.admit_original_inputs(profile, sc.native_sources(profile))
        expected_inputs = {path: value["record"] for path, value in context["inputs"].items()}
        same(native_report["checked_inputs"], expected_inputs,
             "entire-native-input-ledger-recomputed-from-original-admission")
        same(evidence["input_bindings"], expected_inputs, "seal-input-set-is-entire-original-input-set")
        need(native_report["all_inputs_unchanged"] is True and native_report["input_recheck_errors"] == [],
             "actual-native-execution-input-recheck-was-complete")
        need(native_report["case_count"] == native_report["reference_case_count"] == 24
             and len(native_report["cases"]) == 24 and
             [row["index"] for row in native_report["cases"]] == list(range(24)),
             "all-original24-native-case-reports")
        for spec, observed, row in zip(context["contract"]["cases"], context["reference"]["rows"],
                                      native_report["cases"]):
            need(row["index"] == spec["id"] and
                 row["name"] == spec["label"] == observed["name"] and
                 row["reference_event_count"] == len(observed["events"]),
                 "each-original-case-name-and-full-event-count")
            case_raw = read(native_root/("case-%02d" % row["index"])/"report.json")
            same(json.loads(case_raw), row, "top-report-embeds-entire-original-case-report")
        result.update(original_native_status=native_report["status"],
                      original_native_report_sha256=SHA(original_raw),
                      evidence_binding_sha256=args.evidence_binding_sha256,
                      source_package_sha256=args.package_sha256, custody=archive_result,
                      source_sha256=profile["runtime"]["sha256"],
                      native_sha256=profile["binary"]["sha256"],
                      actual_build_sha256=profile["build_report"]["sha256"],
                      source02_failure_history=profile["source02_failure_history"],
                      original_input_count=len(expected_inputs))
        if args.prepare_only:
            result["status"] = "PREPARED_RECORDED_INPUTS_NOT_REPLAYED"
            result["prepared_case_records"] = [
                {"index": row["index"], "name": row["name"], "status": "NOT_ADMITTED",
                 "original_native_status": row["status"],
                 "original_event_count": row["reference_event_count"]}
                for row in native_report["cases"]]
            result["prepared_bounded_controls"] = [
                {"index": row["index"], "name": row["name"], "status": "NOT_ADMITTED",
                 "original_native_status": row["status"], "reference_case": None,
                 "Torch_parity_claim": False} for row in original_controls["cases"]]
            exit_code = 0
        else:
            observer, owner = context["observer"], context["owner"]
            observer.configure(owner, profile["runtime"]["sha256"], profile["binary"]["sha256"],
                               profile["profile"]["functions"])
            factory = sys.modules["recorded_observer"].RecordedObserver(
                observer, owner, context["protocol_path"],
                profile["runtime"]["path"], profile["binary"]["path"])
            Case = context["replay"].case_class(context["parent"], owner)
            case_args = argparse.Namespace(binary=Path(profile["binary"]["path"]),
                source=Path(profile["runtime"]["path"]),
                protocol=context["protocol_path"])
            replayed = []
            for spec, observed, original in zip(context["contract"]["cases"], context["reference"]["rows"],
                                               native_report["cases"]):
                index = original["index"]
                destination = output/("case-%02d" % index)
                facade = factory.begin(native_root/("case-%02d" % index), original, destination)
                case = Case(case_args, facade, context["reference"], observed, spec, destination)
                current = case.run()
                same(current, original, "case-%02d-entire-original-report-recomputed" % index)
                need((destination/"report.json").read_bytes() ==
                     read(native_root/("case-%02d" % index)/"report.json"),
                     "whole-original-case-report-byte-identical", index=index)
                details = factory.completed()
                replayed.append(current)
                result["cases"].append({"index": index, "name": original["name"],
                    "status": "RECOMPUTED", "original_native_status": original["status"],
                    "frames": current["frames"], "assertions": current["assertions"],
                    "exact_words": current["exact_words"], "full_rng_bytes": current["full_rng_bytes"],
                    "rollback_bytes": current["rollback_bytes"],
                    "completed_reference_events": len(current["completed_reference_events"]), **details})
                result["all24_admission_index"][index] = {"index": index, "status": "RECOMPUTED"}
            controls, control_proof = sys.modules["recorded_controls"].replay_controls(
                context, factory, owner, case_args, native_root,
                output/"bounded-api-controls", original_controls)
            result["bounded_controls"] = {"status": "RECOMPUTED", **control_proof}
            reconstructed = original_top(profile, context, replayed, controls)
            same(reconstructed, native_report, "entire-original-top-report-with-no-field-projection")
            reconstructed_path = output/"reconstructed-original-report.json"
            context["replay"].write_json(reconstructed_path, reconstructed)
            need(reconstructed_path.read_bytes() == original_raw,
                 "whole-original-top-report-byte-identical-including-all-case-fields")
            need(native_report["status"] == "PASS" and native_report["passed"] == 24 and
                 native_report["failed"] == native_report["blocked"] == 0 and
                 native_report["original_event_count"] == native_report["completed_reference_events"] == 260,
                 "new-actual-complete24-success-not-inferred-from-historical-PASS")
            need(controls["status"] == "PASS_BOUNDED_API_CONTROLS_ONLY"
                 and controls["case_count"] == controls["passed"] == 5 and controls["failed"] == 0
                 and controls["Torch_parity_claim"] is False,
                 "all-five-native-only-controls-recomputed-with-no-added-reference-case")
            result.update(status="PASS_PURE_RECORDED_ADMISSION", cases_recomputed=24,
                          reference_events_recomputed=260, bounded_controls_recomputed=5,
                          original_top_report_reconstructed_bytes=len(original_raw),
                          original_top_report_reconstructed_sha256=SHA(original_raw),
                          all_original_fields_compared=True)
            exit_code = 0
    except BaseException as error:
        result.update(status="FAIL", error={"type": type(error).__name__, "message": str(error),
                                           "traceback": traceback.format_exc()})
    finally:
        if archive is not None and member_manifest is not None:
            try:
                result["original_member_final_recheck"] = archive.verify_originals(member_manifest)
            except BaseException as error:
                result["member_recheck_error"] = repr(error)
                result["status"], exit_code = "FAIL", 1
        errors = recheck_bound()
        result.update(input_recheck_errors=errors, all_inputs_unchanged=not errors,
                      checked_inputs=BOUND, forbidden_execution_attempts=dict(AUDIT))
        if errors or any(AUDIT.values()):
            result["status"], exit_code = "FAIL", 1
        write(output/"report.json", result)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
