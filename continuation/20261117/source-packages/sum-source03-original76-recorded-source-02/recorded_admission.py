"""Recorded-only admission of all four original696 suites.

Every Case predicate and native event mapper is loaded from the exact executed
source copy. The frozen RecordedSession, Owners and RecordedCase implementation
has only five profile constants changed; its complete function bodies are intact.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import traceback
from admission_common import SHA, BOUND, AUDIT, need, read, data, write, safe_path, same, digest_file, load

HERE = Path(__file__).resolve().parent
STAGE_NAMES = {
    "support": "support.py", "protocol": "protocol.py", "client": "owner_client.py",
    "observer": "native_observer.py", "shape26": "shape26.py", "scalar32": "scalar32.py",
    "direct14": "direct14.py", "input4": "input4.py",
    "identity_epochs": "identity_epochs.py", "parent26": "parent26.py",
}

def stage_bytes(path, raw, expected_sha256, mode):
    need(SHA(raw) == expected_sha256 and not path.exists(), "exact-create-only-module-copy")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fchmod(stream.fileno(), mode)
    read(path, expected_sha256, len(raw), mode)
    return path

def modules(config, suites, selected, output):
    directory = output / "frozen-modules"
    directory.mkdir(mode=0o700, exist_ok=False)
    need(set(config["modules"]) == set(STAGE_NAMES), "all-and-only-explicit-frozen-modules")
    staged, receipt = {}, {}
    for name, pin in config["modules"].items():
        original = suites[pin["suite"]]
        actual_copy = original["custody"][pin["original_path"]]
        same(actual_copy, {"copy": pin["copy"], "bytes": pin["bytes"], "sha256": pin["sha256"]},
             "exact-original-native-input-module")
        source = safe_path(original["base"], pin["copy"])
        raw = read(source, pin["sha256"], pin["bytes"])
        mode = BOUND[str(source)]["mode"]
        target = stage_bytes(directory / STAGE_NAMES[name], raw, pin["sha256"], mode)
        staged[name] = target
        receipt[name] = {"original_source_path": pin["original_path"], "recorded_copy": str(source),
                         "staged_path": str(target), "bytes": len(raw), "sha256": pin["sha256"],
                         "original_recorded_copy_mode": mode, "staged_mode": mode}
    # The unchanged protocol imports owner_client.py by adjacent filename.
    # Both bytes were checked before any protocol import.
    loaded = {}
    for name in STAGE_NAMES:
        if name != "client":
            loaded[name] = load(staged[name], "sum696_frozen_" + name, config["modules"][name]["sha256"])
    owner_pin = config["owner"]
    owner_raw = selected[owner_pin["archive_member"]]
    need(len(owner_raw) == owner_pin["bytes"] and SHA(owner_raw) == owner_pin["sha256"],
         "owner-is-exact-original-shared-member")
    owner_path = stage_bytes(directory / "owner_sum47_48.py", owner_raw,
                             owner_pin["sha256"], owner_pin["original_mode"])
    owner = load(owner_path, "sum696_strict84Q_owner", owner_pin["sha256"])
    need(owner.SOURCE is None and owner.IMAGE is None and owner.DESCRIPTOR == 672,
         "strict-owner-original-explicit-profile-API")
    # Call the unchanged owner's real binding API so its count, pin-format and
    # rebinding guards execute, and its original PROFILE label stays exact.
    owner.bind_profile(config["profile"]["source_sha256"],
                       config["profile"]["native_sha256"], config["profile"]["functions"])
    need(owner.PROFILE == "sum-dims-source03-696-op47-op48",
         "original-owner-profile-label")
    loaded["owner"] = owner
    receipt["owner"] = {"archive_member": owner_pin["archive_member"],
                        "sha256": owner_pin["sha256"], "bytes": len(owner_raw),
                        "staged_path": str(owner_path), "staged_mode": owner_pin["original_mode"],
                        "runtime_profile_binding_only": {"SOURCE": owner.SOURCE, "IMAGE": owner.IMAGE,
                                                         "PROFILE": owner.PROFILE},
                        "owner_function_bodies_modified": False}
    write(output / "module-bindings.json", receipt)
    return loaded, receipt

def reference(family, suite, loaded, support):
    refpin, probepin = suite["spec"]["reference"], suite["spec"]["probe"]
    raw = read(suite["reference"], refpin["sha256"], refpin["bytes"])
    probe = read(suite["probe"], probepin["sha256"], probepin["bytes"])
    if family == "26":
        loaded["shape26"].admit_reference(raw, probe, refpin["sha256"], support)
        original, plan = loaded["identity_epochs"].admit(raw, probe)
        plan["copied_negative_controls"] = loaded["identity_epochs"].negative_controls(json.loads(raw), plan)
        proof = read(suite["base"] / "identity-epoch-plan.json",
                     suite["aggregate"]["identity_epoch_admission"]["proof_sha256"])
        need((json.dumps(plan, indent=2, allow_nan=False) + "\n").encode() == proof,
             "whole-original-identity-epoch-proof-recomputed")
        need(plan["retirements"] == 2 and plan["unaffected_rows"] == 24
             and plan["identity_annotations"] == 208 and len(plan["copied_negative_controls"]) == 6,
             "fixed-proven-released-object-lifetime-epoch-boundaries")
        same(loaded["identity_epochs"].inverse(original), json.loads(raw),
             "exact-original-reference-recoverable-by-epoch-inverse")
        Case = loaded["shape26"].case_class(support)
    elif family == "32":
        original = loaded["scalar32"].admit_reference(raw, probe, support)
        original = {**original, "states": original["full_rng_states"]}
        Case = loaded["scalar32"].case_class(support)
    elif family == "14":
        original = loaded["direct14"].admit_reference(raw, probe)
        Case = loaded["direct14"].Case
    elif family == "4":
        original = loaded["input4"].admit_reference(raw, probe, refpin["sha256"], support)
        Case = loaded["input4"].case_class(support)
    else:
        raise AssertionError(("unknown-recorded-family", family))
    return original, Case

def admit(config, suites, selected, output):
    bridge = load(HERE / config["recorded_bridge"]["path"], "sum696_recorded_frozen_bridge",
                  config["recorded_bridge"]["sha256"])
    need(bridge.SOURCE == config["profile"]["source_sha256"]
         and bridge.IMAGE == config["profile"]["native_sha256"]
         and bridge.BUILD == config["profile"]["build_sha256"]
         and bridge.OBSERVER == config["modules"]["observer"]["sha256"]
         and bridge.OWNER == config["owner"]["sha256"],
         "exact-five-recorded-profile-bindings")
    loaded, module_receipt = modules(config, suites, selected, output)
    support, wire, observer, owner = (loaded[n] for n in ("support", "protocol", "observer", "owner"))
    args = SimpleNamespace(wire=wire)
    results, suite_results = [], []
    for spec in config["suites"]:
        family, suite = spec["family"], suites[spec["name"]]
        observed, NativeCase = reference(family, suite, loaded, support)
        aggregate = suite["aggregate"]
        need(len(observed["rows"]) == len(aggregate["rows"]) == spec["cases"],
             "full-reference-and-original-case-denominators")
        Case = bridge.recorded_case_class(NativeCase, support)
        suite_output = output / ("suite-" + family)
        suite_output.mkdir(mode=0o700, exist_ok=False)
        these = []
        for row, summary in zip(observed["rows"], aggregate["rows"]):
            original_directory = suite["base"] / ("case-%02d" % row["index"])
            destination = suite_output / ("case-%02d" % row["index"])
            result = None
            try:
                original = data(original_directory / "report.json")
                for key, value in summary.items():
                    same(original[key], value, "whole-original-case-summary/" + key)
                need(original["index"] == row["index"] and original["name"] == row["name"]
                     and original["status"] == "PASS" and original["error"] is None
                     and original["mismatches"] == [] and original["exit_code"] == 0,
                     "actual-original-native-case-success")
                need(original["reference_event_count"] == len(original["operations"]) == len(row["events"]),
                     "every-original-reference-event-exported")
                result = Case(args, observer, observed, row, destination, original_directory,
                              original, owner, family).run_admission()
            except BaseException as error:
                destination.mkdir(parents=True, exist_ok=True)
                result = {"schema": "heptatorch.sum03.original76.recorded-case-failure.v1",
                          "status": "FAIL", "suite": family, "index": row["index"], "name": row["name"],
                          "error": repr(error), "traceback": traceback.format_exc(),
                          "original_report_path": str(original_directory / "report.json"),
                          "new_native_executions": 0, "new_Torch_executions": 0,
                          "original_reports_modified": False}
                path = destination / ("attempt-failure.json" if (destination / "report.json").exists()
                                      else "report.json")
                write(path, result)
            results.append(result)
            these.append(result)
        suite_result = {"suite": family, "name": spec["name"], "cases": spec["cases"],
                        "passed_cases": sum(r["status"] == "PASS_PURE_RECORDED_EVIDENCE" for r in these),
                        "failed_cases": sum(r["status"] != "PASS_PURE_RECORDED_EVIDENCE" for r in these),
                        "original_report_sha256": spec["report"]["sha256"],
                        "original_input_copy_count": len(suite["custody"])}
        write(suite_output / "summary.json", suite_result)
        suite_results.append(suite_result)
    write(output / "recorded-case-results.json", {"cases": results, "suites": suite_results})
    need(len(results) == 76 and all(row["status"] == "PASS_PURE_RECORDED_EVIDENCE" for row in results),
         "all76-recorded-cases-no-status-transfer",
         failed=[{"suite": r["suite"], "index": r["index"], "error": r.get("error")}
                 for r in results if r["status"] != "PASS_PURE_RECORDED_EVIDENCE"])
    total_keys = (
        "pure_comparisons", "exact_words_recomputed", "full_rng_bytes_recomputed",
        "all_raw_frames_generated_and_matched", "exported_snapshots_compared",
        "physical_Tensor_reads_from_full_owner", "full_owner_blobs_invariant_checked",
        "full_owner_records", "directly_exported_adjacent_rollback_pairs",
        "directly_exported_adjacent_rollback_bytes", "original_refusal_bytes_source_bound",
        "refusals_without_exported_immediate_before_owner",
    )
    totals = {key: sum(row[key] for row in results) for key in total_keys}
    wanted = config["expected_totals"]
    for key, expected in (("exact_words_recomputed", wanted["exact_words"]),
                          ("full_rng_bytes_recomputed", wanted["full_rng_bytes"]),
                          ("all_raw_frames_generated_and_matched", wanted["frames"]),
                          ("original_refusal_bytes_source_bound", wanted["source_bound_refusal_bytes"])):
        same(totals[key], expected, "all-original-expected-counts/" + key)
    same(sum(row["original_assertions"] for row in results), wanted["original_assertions"],
         "original-assertion-count-preserved-not-all-claimed-recomputed")
    bridge.recheck_bound()
    for path, expected in bridge.BOUND.items():
        digest_file(path, expected["sha256"], expected["bytes"], expected["mode"])
    need(not any(bridge.AUDIT.values()) and not any(AUDIT.values()),
         "no-process-or-live-memory-in-recorded-replay")
    return {
        "status": "PASS_PURE_RECORDED_76", "source_sha256": config["profile"]["source_sha256"],
        "native_sha256": config["profile"]["native_sha256"],
        "original_semantic_reader_sha256": config["recorded_bridge"]["original_sha256"],
        "profile_bound_semantic_reader_sha256": config["recorded_bridge"]["sha256"],
        "original_native_qualification_transferred": False,
        "module_bindings": module_receipt, "suites": suite_results, "cases": results, **totals,
        "new_native_executions": 0, "new_Torch_executions": 0, "new_compiler_executions": 0,
        "unexported_intermediate_memory_recreated": False,
        "general_ViewInfo_transition_checker_invoked": False,
        "all_original_Case_comparator_function_bodies_unchanged": True,
        "all_RecordedSession_Owners_RecordedCase_function_bodies_unchanged": True,
        "existing_pure32_reexecuted": False, "whole_pytorch_gate": "HOLD", "closed_schema_count": 0,
    }
