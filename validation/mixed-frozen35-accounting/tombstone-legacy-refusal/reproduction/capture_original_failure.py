"""Read-only trace capture of the frozen test's primary failure.

The original source is executed unchanged. Its broken Session.stop cleanup
can prevent report.json from being written; this driver preserves the live
local report, then separately closes its own native child with real EOF.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import runpy
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("script", "binary", "source", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    target = args.script.resolve()
    raw = target.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "2dc6d1d6444264fec06c3429926adcf778e48b6b8b80226b690bc31667dcd6cc"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    captured = {}
    live = []
    events = []

    def trace(frame, event, argument):
        if frame.f_code.co_filename == str(target) and frame.f_code.co_name == "main":
            report = frame.f_locals.get("report")
            if report is not None and report.get("status") == "FAIL":
                captured.clear()
                captured.update(copy.deepcopy(report))
            session = frame.f_locals.get("session")
            if session is not None and not live:
                live.append(session)
            if event == "exception":
                events.append({"line": frame.f_lineno, "type": argument[0].__name__,
                               "error": repr(argument[1])})
        return trace

    sys.argv = [str(target), "--binary", str(args.binary), "--source", str(args.source),
                "--protocol", str(args.protocol), "--output", str(output / "original-run")]
    sys.settrace(trace)
    escaping = None
    try:
        runpy.run_path(str(target), run_name="__main__")
    except BaseException as error:
        escaping = {"error": repr(error), "traceback": traceback.format_exc()}
    finally:
        sys.settrace(None)
    cleanup = {"attempted": False}
    if live:
        cleanup["attempted"] = True
        try:
            live[0].finish()
            cleanup.update(status="PASS", EOF_exit_code=live[0].exit_code,
                           frames=live[0].frames)
        except BaseException as error:
            cleanup.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    custody = target.read_bytes() == raw
    (output / "captured-primary-report.json").write_text(json.dumps(captured, indent=2) + "\n")
    receipt = {"schema": "heptatorch.frozen-test-local-report-capture.v1",
               "original_source_sha256": hashlib.sha256(raw).hexdigest(),
               "original_source_unchanged": custody,
               "captured_by": "read-only Python trace; no source or live-local mutation",
               "original_execution_escaped": escaping, "exception_events": events,
               "separate_driver_EOF_cleanup": cleanup,
               "original_suite_status": captured.get("status"),
               "native_qualification_claim": False, "whole_pytorch_gate": "HOLD"}
    (output / "capture-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert custody and captured["status"] == "FAIL" and captured["native_executed"]
    assert captured["error"] == "AssertionError('Expected original6ea observer mismatch was not reproduced')"
    assert escaping["error"] == "AttributeError(\"'Session' object has no attribute 'stop'\")"
    assert cleanup["status"] == "PASS"
    print(json.dumps({"status": "ORIGINAL_FAILURE_CAPTURED", "report": str(output / "captured-primary-report.json"),
                      "primary_error": captured["error"], "frames": cleanup["frames"]}))


if __name__ == "__main__":
    main()
