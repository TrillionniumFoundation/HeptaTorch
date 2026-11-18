"""Complete original owner endpoints for pure Sum32 replay.

Every memory observation is read from a raw endpoint exported by the original
native observer. Adjacent endpoints must match exactly. No /proc, live child,
Tensor arithmetic, snapshot projection, or descriptor-page exclusion is used.
"""
from collections import OrderedDict
import copy
import json
from pathlib import Path
from admission_common import SHA, need, read, same, safe_path, single_xz

class CompleteOwners:
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
        self.case_records = report["owner_records"]
        self.case_blobs = report["owner_blobs"]
        case_references = set()
        for item in self.case_records:
            need(set(item) == {"label", "frame", "pointer", "length", "sha256"}
                 and type(item["frame"]) is int and 0 <= item["frame"] <= len(self.journal),
                 "original-Case-owner-record-shape")
            ref = self.endpoint(item["frame"])
            need(all(item[key] == ref[key] for key in ("frame", "pointer", "length", "sha256")),
                 "every-Case-owner-record-is-real-complete-endpoint", label=item["label"])
            need(item["sha256"] in self.case_blobs, "Case-owner-member-present")
            entry = self.case_blobs[item["sha256"]]
            need(set(entry) == {"path", "bytes", "sha256", "compressed_bytes", "compressed_sha256"}
                 and entry["sha256"] == item["sha256"] and entry["bytes"] == item["length"]
                 and entry["path"] == "owners/"+item["sha256"]+".bin.xz",
                 "original-Case-owner-blob-shape")
            case_references.add(item["sha256"])
        need(case_references == set(self.case_blobs), "all-Case-owner-members-referenced")

    def endpoint(self, frame):
        need(type(frame) is int and 0 <= frame <= len(self.journal),
             "recorded-owner-requested-at-real-frame", frame=frame)
        return self.journal[0]["before"] if frame == 0 else self.journal[frame-1]["after"]

    def raw(self, digest):
        if digest in self.cache:
            self.cache.move_to_end(digest)
            return self.cache[digest]
        entry = self.blobs[digest]
        path = safe_path(self.directory, entry["path"])
        read(path, entry["compressed_sha256"], entry["compressed_bytes"], entry["mode"])
        raw = single_xz(path, entry)
        self.cache[digest] = raw
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return raw

    def current(self, frame):
        ref = self.endpoint(frame)
        raw = self.raw(ref["sha256"])
        need(len(raw) == ref["length"], "full-current-owner-byte-length")
        self.current_reads += 1
        return {"pointer": ref["pointer"], "length": ref["length"], "bytes": raw, "live": 1}

    def capture(self, label, frame):
        index, phase = divmod(self.capture_cursor, 2)
        need(index < len(self.journal) and label == ("before" if phase == 0 else "after")
             and frame == index+phase, "original-owner-capture-order", label=label, frame=frame)
        expected = self.journal[index][label]
        raw_owner = self.current(frame)
        digest = SHA(raw_owner["bytes"])
        reference = {"label": label, "frame": frame, "pointer": raw_owner["pointer"],
                     "length": raw_owner["length"], "sha256": digest}
        snapshot = self.module.decode_owner(raw_owner["bytes"], raw_owner["pointer"])
        reference["verified_summary"] = self.module.verify_snapshot(snapshot)
        same(reference, expected, "original-complete-endpoint-recomputed-without-projection")
        self.capture_cursor += 1
        self.captured_endpoints += 1
        self.seen_digests.add(digest)
        return snapshot, reference, raw_owner["bytes"]

    def finish(self, frames):
        need(frames == len(self.journal) and self.capture_cursor == 2*frames
             and self.seen_digests == set(self.blobs), "all-original-complete-endpoints-consumed")
        for digest, entry in self.case_blobs.items():
            case_raw = single_xz(safe_path(self.directory, entry["path"]), entry)
            need(case_raw == self.raw(digest), "Case-and-frame-entire-owner-bytes-identical")
        return {"frames": frames, "complete_endpoints": self.captured_endpoints,
                "complete_owner_members": len(self.blobs), "Case_owner_members": len(self.case_blobs),
                "recorded_memory_reads": self.current_reads, "frame_header_refusals": self.frame_refusals,
                "whole_owner_rollback_bytes": self.rollback_bytes,
                "partial_backward_payload_errors": self.partial_payload_errors,
                "full_general_transition_recomputed": False}
