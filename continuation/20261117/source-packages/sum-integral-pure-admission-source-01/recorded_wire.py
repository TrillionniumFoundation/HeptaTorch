"""Exact recorded-wire transport for the complete Sum32 owner journals.

The original R4 RecordedSession body is retained through an explicit four-edit
whole-block inverse. No native executable is opened as code or started.
"""
from admission_common import need, read, data, same, unpack_frames
import struct

IMAGE = "fec4f5cd7f80591fee98378cb5a8a85b6e71c5e4a7f0a00a739dad16afc24c28"

class RecordedSession:
    """Transport only: exact request comparison and original response bytes."""
    tensor_results = {1, 2, 6, 17, 18, 19, 38, 55, 56, 91, 97, 107,
                      108, 109, 110, 111, 112, 113, 114, 115, 116, 117}

    def __init__(self, directory, wire, expected_frames):
        self.wire = wire
        self.requests = unpack_frames(read(directory / "requests.frames.bin"), b"HTOWNR01")
        self.responses = unpack_frames(read(directory / "responses.frames.bin"), b"HTOWNS01")
        self.transport = data(directory / "transport.json")
        t = self.transport
        need(set(t) == {"pid", "frames", "executable_sha256", "rows", "exit_code",
             "responses_before_stdin_eof", "stdout_eof", "stderr_eof", "stderr_bytes",
             "stderr_sha256", "extra_stdout_bytes", "extra_stdout_sha256",
             "complete_owner_boundaries", "owner_blobs", "full_transition_recomputed",
             "forced_cleanup", "cleanup_errors"}, "complete-Sum-transport-fields")
        need(t["stdout_eof"] is True and t["stderr_eof"] is True
             and t["forced_cleanup"] is False and t["cleanup_errors"] == []
             and t["full_transition_recomputed"] is False,
             "original-dual-EOF-no-forced-cleanup-no-transition-overclaim")
        need(t["complete_owner_boundaries"] == expected_frames,
             "one-complete-owner-pair-per-original-frame")
        stderr = read(directory / "stderr.bin", t["stderr_sha256"], t["stderr_bytes"])
        extra = read(directory / "extra-stdout-after-frames.bin",
                     t["extra_stdout_sha256"], t["extra_stdout_bytes"])
        need(stderr == b"" and extra == b"", "exact-original-empty-EOF-streams")
        self.owner_source = None
        self.pid, self.binary_sha256 = t["pid"], t["executable_sha256"]
        need(t["frames"] == len(self.requests) == len(self.responses) == len(t["rows"]) == expected_frames,
             "all-raw-frames-present")
        need(t["exit_code"] == 0 and t["responses_before_stdin_eof"] is True
             and t["executable_sha256"] == IMAGE, "recorded-native-EOF-and-image")
        self.decoded = []
        for i, (request, response, tr) in enumerate(zip(self.requests, self.responses, t["rows"]), 1):
            same(tr, {"frame": i, "pid": t["pid"], "response_bytes": len(response), "stdin_open": True},
                 "transport-frame-row")
            tag, count, size, reserved = struct.unpack_from("<8s3Q", request)
            need(tag == b"HTORCH03" and size == len(request) and reserved == 0
                 and 32 + 64 * count <= len(request), "request-v3-header")
            commands = [struct.unpack_from("<8Q", request, 32 + 64*j) for j in range(count)]
            status, records = wire.parse(response)
            if status:
                need(records == [], "frame-error-no-records")
            else:
                need(len(records) <= len(commands), "response-record-count")
                same([r["opcode"] for r in records], [q[0] for q in commands[:len(records)]],
                     "response-command-prefix")
                if len(records) < len(commands):
                    need(records and records[-1]["opcode"] == 58 and records[-1]["value"] != 0,
                         "partial-backward-stops-suffix")
                for record in records:
                    if record["opcode"] == 58:
                        need(len(record["payload"]) == 32, "all-backward-four-Q-payloads")
                        need(struct.unpack("<4Q", record["payload"])[0] == record["value"],
                             "backward-status-payload-agrees")
            self.decoded.append((status, records, commands))
        self.frames = 0
        self.exit_code = None
        self.public = set()
        self.weak = set()
        self.hold = set()
        self.generators = set()

    def exchange(self, body):
        need(self.frames < len(self.requests), "no-invented-extra-request")
        need(body == self.requests[self.frames], "generated-request-equals-original-raw",
             frame=self.frames + 1)
        status, records, commands = self.decoded[self.frames]
        self.frames += 1
        if status == 0:
            for command, record in zip(commands, records):
                op, value = command[0], record["value"]
                if op in self.tensor_results and value:
                    need((value >> 24) & 255 == 1, "returned-public-Tensor-kind", frame=self.frames)
                    self.public.add(value)
                elif op == 41:
                    need(command[1] in self.public, "drop-known-public-capability", frame=self.frames)
                    self.public.remove(command[1])
                elif op == 42:
                    need(value not in self.weak, "new-Weak-generation", frame=self.frames)
                    self.weak.add(value)
                elif op == 44:
                    need(command[1] in self.weak, "drop-known-Weak", frame=self.frames)
                    self.weak.remove(command[1])
                elif op == 43:
                    need(value not in self.hold, "new-Hold-generation", frame=self.frames)
                    self.hold.add(value)
                elif op == 45:
                    need(command[1] in self.hold, "drop-known-Hold", frame=self.frames)
                    self.hold.remove(command[1])
                elif op == 60:
                    self.generators.add(value)
                elif op == 66:
                    need(command[1] in self.generators, "drop-known-generator", frame=self.frames)
                    self.generators.remove(command[1])
        return self.responses[self.frames - 1]

    def run(self, commands):
        status, records = self.wire.parse(self.exchange(self.wire.packet(commands)))
        need(status == 0 and len(records) == len(commands), "successful-observer-command", frame=self.frames)
        same([r["opcode"] for r in records], [c[0] for c in commands], "observer-command-order")
        return records

    def owner(self, *args, **kwargs):
        need(not args and not kwargs, "only-original-owner-read-interface")
        need(self.owner_source is not None, "complete-owner-journal-bound-before-any-read")
        return self.owner_source.current(self.frames)

    def finish(self, *args, **kwargs):
        need(self.frames == len(self.requests), "all-original-wire-requests-consumed",
             consumed=self.frames, total=len(self.requests))
        need(not self.public and not self.weak and not self.hold and not self.generators,
             "recorded-observer-capability-cleanup")
        need(self.owner_source is not None, "complete-owner-source-at-recorded-EOF")
        self.owner_finish_summary = self.owner_source.finish(self.frames)
        self.exit_code = self.transport["exit_code"]
        need(self.exit_code == 0, "recorded-final-native-exit")
        return self.exit_code
