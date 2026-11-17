"""Integral control endpoints; the original Case owner reader stays unchanged.

Controls do not export Case.owner_records/owners archives. This class validates
their actual complete frame journal plus the two explicitly exported manual
capture_owner records. It does not invent an empty Case ledger or omit a page.
"""
from collections import OrderedDict
import json
from pathlib import Path
import re

from admission_common import SHA, need, read, same
from complete_owners import CompleteOwners


class CompleteControlOwners(CompleteOwners):
    def __init__(self, directory, report, session, owner_module):
        self.directory, self.report = Path(directory), report
        self.session, self.module = session, owner_module
        self.blobs = report["complete_frame_owner_blobs"]
        same(session.transport["owner_blobs"], self.blobs,
             "transport-and-case-complete-owner-blob-ledger")
        raw = read(self.directory / "owner-frames.jsonl")
        need(raw.endswith(b"\n") and raw, "complete-nonempty-owner-journal-EOF")
        lines = raw.splitlines()
        need(all(lines), "no-empty-owner-journal-record")
        self.journal = [json.loads(line) for line in lines]
        need(len(self.journal) == report["complete_frame_count"] == len(session.requests),
             "one-original-complete-owner-pair-per-frame")
        self.cache, self.capture_cursor = OrderedDict(), 0
        self.seen_digests, self.current_reads = set(), 0
        self.captured_endpoints = 0
        self.frame_refusals, self.partial_payload_errors = 0, 0
        self.rollback_bytes = 0
        referenced = set()
        previous = None
        for index, row in enumerate(self.journal, 1):
            need(set(row) == {"frame", "request_sha256", "request_bytes", "commands", "before",
                 "response_sha256", "response_bytes", "after", "frame_status", "record_count", "checks"},
                 "complete-successful-observer-journal-fields", frame=index)
            request, response = session.requests[index-1], session.responses[index-1]
            status, records, commands = session.decoded[index-1]
            need(row["frame"] == index and row["request_sha256"] == SHA(request)
                 and row["request_bytes"] == len(request) and row["response_sha256"] == SHA(response)
                 and row["response_bytes"] == len(response), "raw-wire-journal-binding", frame=index)
            same(row["commands"], [list(command) for command in commands],
                 "exact-journal-command-words")
            need(row["frame_status"] == status and row["record_count"] == len(records),
                 "actual-wire-header-status-and-record-count", frame=index)
            for label, frame in (("before", index-1), ("after", index)):
                ref = row[label]
                need(set(ref) == {"label", "frame", "pointer", "length", "sha256", "verified_summary"},
                     "complete-original-endpoint-fields", frame=index, phase=label)
                need(ref["label"] == label and ref["frame"] == frame
                     and type(ref["pointer"]) is int and ref["pointer"] >= 0
                     and type(ref["length"]) is int and 0 <= ref["length"] <= 64*(1 << 20)
                     and (ref["pointer"] == 0) == (ref["length"] == 0),
                     "exact-endpoint-state-and-length", frame=index, phase=label)
                need(ref["sha256"] in self.blobs, "declared-complete-endpoint-blob")
                entry = self.blobs[ref["sha256"]]
                need(set(entry) == {"path", "bytes", "sha256", "compressed_bytes", "compressed_sha256", "mode"}
                     and entry["sha256"] == ref["sha256"] and entry["bytes"] == ref["length"]
                     and entry["path"] == "complete-owner/"+ref["sha256"]+".bin.xz",
                     "whole-owner-ledger-and-canonical-name")
                referenced.add(ref["sha256"])
            if previous is not None:
                need(all(previous[key] == row["before"][key]
                         for key in ("pointer", "length", "sha256")),
                     "after-N-equals-before-N-plus-one-complete-owner", frame=index)
                # SHA binding is independently checked against the complete bytes
                # when every capture is consumed; neither endpoint is synthesized.
            previous = row["after"]
            if status:
                need(all(row["before"][key] == row["after"][key]
                         for key in ("pointer", "length", "sha256")),
                     "original-header-error-whole-owner-rollback", frame=index)
                self.frame_refusals += 1
                self.rollback_bytes += row["before"]["length"]
            for record in records:
                if record["opcode"] == 58 and record["value"] != 0:
                    need(status == 0, "partial-backward-payload-is-not-a-frame-error", frame=index)
                    self.partial_payload_errors += 1
        need(referenced == set(self.blobs), "no-missing-or-unreferenced-complete-owner-member")
        need(report["full_transition_recomputed"] is False,
             "original-general-transition-scope-preserved")
        need(set(report) == {"index", "name", "status", "scope", "Torch_parity_claim",
             "reference_case", "checks", "observations", "error", "frames", "exit_code",
             "complete_frame_count", "complete_frame_owner_blobs", "full_transition_recomputed",
             "schema_closure", "full_project_status"}, "exact-original-bounded-control-report")
        need(type(report["index"]) is int and 0 <= report["index"] < 5
             and report["scope"] == "native bounded dtype dispatch only"
             and report["Torch_parity_claim"] is False and report["reference_case"] is None
             and report["schema_closure"] is False and report["full_project_status"] == "HOLD",
             "five-native-controls-have-no-invented-Torch-or-Case-ledger")
        observations = report["observations"]
        need(set(observations) == {"command", "frame_status", "record_count",
             "request_hex", "request_sha256", "response_hex", "response_sha256",
             "before_owner", "after_owner"}, "complete-control-observation-fields")
        before, after = observations["before_owner"], observations["after_owner"]
        need(type(before["frame"]) is int and after["frame"] == before["frame"]+1
             and 0 <= before["frame"] < len(self.journal),
             "one-real-wire-command-between-manual-control-endpoints")
        row = self.journal[before["frame"]]
        need(row["frame"] == after["frame"] and row["commands"] == [observations["command"]]
             and observations["command"][0] == 115,
             "control-manual-endpoints-bind-the-actual-command115-frame")
        need(observations["frame_status"] == row["frame_status"]
             and observations["record_count"] == row["record_count"],
             "control-refusal-status-from-original-frame")
        request = session.requests[before["frame"]]
        response = session.responses[before["frame"]]
        need(observations["request_hex"] == request.hex()
             and observations["request_sha256"] == SHA(request)
             and observations["response_hex"] == response.hex()
             and observations["response_sha256"] == SHA(response),
             "complete-control-wire-bytes-not-summary-values")
        self.manual_references = {
            "bounded-dtype-before": before,
            "bounded-dtype-after": after}
        self.manual_capture_cursor = 0
        for phase, label in (("before", "bounded-dtype-before"),
                             ("after", "bounded-dtype-after")):
            reference = self.manual_references[label]
            need(set(reference) == {"label", "frame", "pointer", "length", "sha256", "verified_summary"}
                 and reference["label"] == label, "original-manual-control-capture-fields")
            endpoint = self.endpoint(reference["frame"])
            same({key: reference[key] for key in reference if key != "label"},
                 {key: endpoint[key] for key in endpoint if key != "label"},
                 "manual-control-owner-is-complete-exported-frame-endpoint")
            same({key: reference[key] for key in reference if key != "label"},
                 {key: row[phase][key] for key in row[phase] if key != "label"},
                 "manual-control-owner-is-the-correct-command-side")

    def capture(self, label, frame):
        if label in ("before", "after"):
            return super().capture(label, frame)
        order = ("bounded-dtype-before", "bounded-dtype-after")
        need(self.manual_capture_cursor < 2 and label == order[self.manual_capture_cursor]
             and self.capture_cursor == 2*frame,
             "exact-original-manual-control-capture-order", label=label, frame=frame)
        expected = self.manual_references[label]
        need(frame == expected["frame"], "manual-capture-at-original-recorded-frame")
        raw_owner = self.current(frame)
        raw = raw_owner["bytes"]
        reference = {"label": label, "frame": frame, "pointer": raw_owner["pointer"],
                     "length": raw_owner["length"], "sha256": SHA(raw)}
        snapshot = self.module.decode_owner(raw, raw_owner["pointer"])
        reference["verified_summary"] = self.module.verify_snapshot(snapshot)
        same(reference, expected, "entire-manual-control-owner-observation-recomputed")
        self.manual_capture_cursor += 1
        self.seen_digests.add(reference["sha256"])
        return snapshot, reference, raw

    def finish(self, frames):
        need(frames == len(self.journal) and self.capture_cursor == 2*frames
             and self.seen_digests == set(self.blobs),
             "all-original-control-frame-endpoints-consumed")
        need(self.manual_capture_cursor == 2, "both-original-manual-control-endpoints-consumed")
        return {"frames": frames, "complete_endpoints": self.captured_endpoints,
                "complete_owner_members": len(self.blobs),
                "manual_control_captures": self.manual_capture_cursor,
                "Case_owner_archive_claim": False,
                "recorded_memory_reads": self.current_reads,
                "frame_header_refusals": self.frame_refusals,
                "whole_owner_rollback_bytes": self.rollback_bytes,
                "partial_backward_payload_errors": self.partial_payload_errors,
                "full_general_transition_recomputed": False}


def owners_for_record(directory, report, session, owner_module):
    directory = Path(directory)
    if re.fullmatch(r"case-[0-9]{2}", directory.name):
        need("owner_records" in report and "owner_blobs" in report
             and "reference_case" not in report,
             "original-reference-Case-keeps-complete-Case-owner-contract")
        return CompleteOwners(directory, report, session, owner_module)
    need(directory.parent.name == "bounded-api-controls"
         and re.fullmatch(r"control-0[0-4]", directory.name)
         and int(directory.name[-2:]) == report["index"]
         and "owner_records" not in report and "owner_blobs" not in report,
         "separate-original-control-directory-and-schema")
    return CompleteControlOwners(directory, report, session, owner_module)
