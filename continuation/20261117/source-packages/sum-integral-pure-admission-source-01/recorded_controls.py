"""Run the five immutable bounded controls over their own original tapes.

This module does not replace run_controls or any check inside it. Its Native
factory selects one separately archived control directory before the original
constructor runs. No empty Case owner ledger, expected Tensor or process is made.
"""
from pathlib import Path
from types import SimpleNamespace
import json

from admission_common import SHA, need, read, same


def replay_controls(context, factory, owner, case_args, native_root, output, expected):
    controls = context["controls"]
    original_root = Path(native_root)/"bounded-api-controls"
    output = Path(output)
    same(json.loads(read(original_root/"report.json")), expected,
         "entire-original-control-summary-is-in-native-report")
    need(expected["schema"] == "heptatorch.sum-integral.bounded-dtype-controls.v1"
         and expected["case_count"] == len(expected["cases"]) == len(controls.CONTROLS) == 5
         and [row["index"] for row in expected["cases"]] == list(range(5))
         and expected["Torch_parity_claim"] is False
         and expected["schema_closure"] is False and expected["full_project_status"] == "HOLD",
         "five-bounded-controls-remain-separate-from24-reference-cases")
    for index, ((name, dtype, raw, target_dtype), report) in enumerate(
            zip(controls.CONTROLS, expected["cases"])):
        need(report["name"] == name and report["reference_case"] is None
             and report["Torch_parity_claim"] is False,
             "original-control-source-input-and-report-identity", index=index)
        same(json.loads(read(original_root/("control-%02d" % index)/"report.json")),
             report, "whole-original-control-report-embedded-in-summary")
    started, details = [], []

    def recorded_control_native(binary, source, protocol, directory):
        index = len(started)
        need(index < 5 and Path(directory) == output/("control-%02d" % index),
             "all-five-controls-in-original-order-with-no-extra-constructor")
        if index:
            details.append(factory.completed())
        facade = factory.begin(original_root/("control-%02d" % index),
                               expected["cases"][index], Path(directory))
        started.append(index)
        return facade.Native(binary, source, protocol, directory)

    values = dict(vars(factory.facade))
    values["Native"] = recorded_control_native
    control_facade = SimpleNamespace(**values)
    current = controls.run_controls(control_facade, owner, case_args, output)
    need(started == list(range(5)), "all-five-original-controls-actually-recomputed")
    details.append(factory.completed())
    need(len(details) == 5, "five-complete-original-control-EOF-admissions")
    same(current, expected, "entire-original-controls-report-recomputed-without-projection")
    for index in range(5):
        relative = Path("control-%02d" % index)/"report.json"
        need((output/relative).read_bytes() == read(original_root/relative),
             "whole-original-control-report-bytes-reconstructed", index=index)
    summary_raw = read(original_root/"report.json")
    need((output/"report.json").read_bytes() == summary_raw,
         "whole-original-five-control-summary-bytes-reconstructed")
    need(current["status"] == "PASS_BOUNDED_API_CONTROLS_ONLY"
         and current["passed"] == 5 and current["failed"] == 0,
         "each-control-recomputed-success-does-not-claim-Torch-parity")
    return current, {
        "case_count": 5, "controls_recomputed": len(details),
        "reference_cases_added": 0, "Torch_parity_claim": False,
        "complete_owner_admissions": details,
        "control_report_bytes": len(summary_raw), "control_report_sha256": SHA(summary_raw),
        "all_original_fields_and_report_bytes_compared": True,
        "native_executed_here": False, "full_general_transition_recomputed": False}
