#!/usr/bin/env python3
"""Pure admission of the four frozen scalar07b native sessions.

No executable, compiler, Torch, numerical model or /proc reader is run.
The original event mappings generate packets into a recorded-wire session.
Original check/tensor/identity/compare_node predicates are inherited unchanged.
Exported observations are compared again to the original references; committed
owners are independently decoded. Unexported intermediate /proc reads are not
invented. In particular original rollback assertions and directly recoverable
adjacent-before/after rollback byte pairs are reported separately.
"""
from pathlib import Path, PurePosixPath
from collections import OrderedDict
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import lzma
import struct
import sys
import traceback

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
SOURCE = "be841bf3468b4ce6ce66daead602d540839dadb1f8ceca2fd3b5940ff2e6d545"
IMAGE = "cff287705bc2c9cfc9eb53f0c3f68b287c30e6e93c2b44cfe8954047fc386234"
BUILD = "b2d9dd9eff9b9f1887a9d3268fe5692024b65804e73bd319539ce9644df3ea6f"
SUPPORT = "52c3b3242a73fb412102c83c5ba47cf8d95a4eb3c01e56f41bfe593066506d05"
OBSERVER = "61b02ca4d3d9757aaed35f08a377e315f41e1bf7cfe2b002796e5c4e637956d8"
OWNER = "0eadfed89c33fb8d8615dd2f976b08b66ef6bc309ddbeca3685f25517634449d"
BOUND = {}
FORBIDDEN = ("subprocess.Popen", "os.system", "os.exec", "os.fork",
             "os.forkpty", "os.posix_spawn", "socket.connect")
AUDIT = {"process_attempts": 0, "proc_memory_attempts": 0}

def no_execution(event, args):
    if event in FORBIDDEN:
        AUDIT["process_attempts"] += 1
        raise AssertionError(("pure-admission-forbids-process-or-network", event))
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        path = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if path.startswith("/proc/"):
            AUDIT["proc_memory_attempts"] += 1
            raise AssertionError(("unexported-proc-observation-forbidden", path))

sys.addaudithook(no_execution)

def need(condition, assertion_label, **facts):
    if not condition:
        raise AssertionError({"raw_admission_rule": assertion_label, **facts})

def read(path, pin=None, size=None):
    p = Path(path)
    need(p.is_file() and not p.is_symlink(), "regular-input-file", path=str(p))
    raw = p.read_bytes()
    row = {"sha256": SHA(raw), "bytes": len(raw), "mode": p.stat().st_mode & 0o777}
    if pin is not None:
        need(row["sha256"] == pin, "input-SHA256", path=str(p), got=row["sha256"], expected=pin)
    if size is not None:
        need(len(raw) == size, "input-byte-length", path=str(p))
    if str(p) in BOUND:
        need(BOUND[str(p)] == row, "input-unchanged-on-reread", path=str(p))
    BOUND[str(p)] = row
    return raw

def data(path, pin=None):
    return json.loads(read(path, pin))

def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with p.open("xb") as f:
        f.write(raw)
    return SHA(raw)

def safe_path(root, rel):
    q = PurePosixPath(rel)
    need(bool(rel) and not q.is_absolute() and ".." not in q.parts, "relative-input-path", path=rel)
    p = root / rel
    need(p.is_relative_to(root), "contained-input-path", path=rel)
    return p

def same(a, b, label):
    # Python json retains exact u64 integers; no JS Number or float conversion.
    need(json.loads(json.dumps(a)) == json.loads(json.dumps(b)), label)

def load(path, name, pin):
    read(path, pin)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def restore_value(value, wire):
    if isinstance(value, dict) and set(value) == {"wire_payload"}:
        d = value["wire_payload"]
        raw = bytes.fromhex(d["hex"])
        need(len(raw) == d["bytes"], "recorded-payload-length")
        return wire.Payload(raw, d["alignment"])
    if isinstance(value, dict) and set(value) == {"bytes", "hex"}:
        raw = bytes.fromhex(value["hex"])
        need(len(raw) == value["bytes"], "recorded-byte-length")
        return raw
    if isinstance(value, list):
        return [restore_value(x, wire) for x in value]
    if isinstance(value, dict):
        return {k: restore_value(x, wire) for k, x in value.items()}
    return value

def single_xz(path, expected):
    packed = read(path, expected["compressed_sha256"], expected["compressed_bytes"])
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    raw = decoder.decompress(packed)
    need(decoder.eof and not decoder.unused_data, "one-complete-owner-XZ", path=str(path))
    need(len(raw) == expected["bytes"] and SHA(raw) == expected["sha256"],
         "decompressed-whole-owner", path=str(path))
    return raw

def unpack_frames(raw, magic):
    rows, at = [], 0
    while at < len(raw):
        need(at + 16 <= len(raw), "whole-wire-frame-header")
        tag, size = struct.unpack_from("<8sQ", raw, at)
        at += 16
        need(tag == magic and 32 <= size <= 1 << 20 and at + size <= len(raw),
             "whole-wire-frame-bound")
        rows.append(raw[at:at+size])
        at += size
    need(at == len(raw), "wire-complete-EOF")
    return rows

class RecordedSession:
    """Transport only: exact request comparison and original response bytes."""
    tensor_results = {1, 2, 6, 17, 18, 19, 38, 55, 56, 91, 97, 107,
                      108, 109, 110, 111, 112, 113, 114}

    def __init__(self, directory, wire, expected_frames):
        self.wire = wire
        self.requests = unpack_frames(read(directory / "requests.frames.bin"), b"HTOWNR01")
        self.responses = unpack_frames(read(directory / "responses.frames.bin"), b"HTOWNS01")
        self.transport = data(directory / "transport.json")
        t = self.transport
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
        raise AssertionError("An intermediate /proc read was not exported; cannot synthesize it")

    def finish(self, *args, **kwargs):
        need(self.frames == len(self.requests), "all-original-wire-requests-consumed",
             consumed=self.frames, total=len(self.requests))
        need(not self.public and not self.weak and not self.hold and not self.generators,
             "recorded-observer-capability-cleanup")
        self.exit_code = self.transport["exit_code"]
        need(self.exit_code == 0, "recorded-final-native-exit")
        return self.exit_code

class Owners:
    """Decode every exported full owner, with bounded in-memory caching."""
    def __init__(self, directory, report, observer, owner_module, frames):
        self.directory, self.report = directory, report
        self.observer, self.module, self.frames = observer, owner_module, frames
        self.cache = OrderedDict()
        self.blobs = report["owner_blobs"]
        self.records = report["owner_records"]
        self.by_label_frame = {}
        self.by_frame = {}
        self.snapshot_checks = []
        self.adjacent_rollback_pairs = 0
        self.adjacent_rollback_bytes = 0
        self.unexported_before_rollback_records = 0
        for digest, entry in self.blobs.items():
            need(entry["sha256"] == digest, "owner-digest-key")
            raw = self.raw(digest)
            decoded = owner_module.decode_owner(raw)
            checked = owner_module.verify_snapshot(decoded)
            self.snapshot_checks.append({"sha256": digest, "bytes": len(raw),
                "decoded_snapshot_sha256": SHA(owner_module.canonical(decoded)),
                "invariant_result_sha256": SHA(owner_module.canonical(checked))})
        for entry in self.records:
            need(entry["sha256"] in self.blobs and entry["length"] == self.blobs[entry["sha256"]]["bytes"],
                 "owner-record-blob-binding")
            need(0 < entry["frame"] <= len(frames.requests) and entry["pointer"] > 0,
                 "owner-record-frame-and-pointer")
            key = (entry["label"], entry["frame"])
            need(key not in self.by_label_frame, "unique-owner-label-frame")
            self.by_label_frame[key] = entry
            self.by_frame.setdefault(entry["frame"], []).append(entry)
        need(set(self.blobs) == {x["sha256"] for x in self.records}, "no-unreferenced-owner-blob")
        refused = [x for x in self.records if x["label"].endswith("/frame_refusal")]
        for after in refused:
            f = after["frame"]
            need(frames.decoded[f-1][0] != 0, "refusal-owner-attached-to-real-error", frame=f)
            preceding = self.by_frame.get(f-1, [])
            if preceding:
                before = preceding[-1]
                need(before["pointer"] == after["pointer"] and before["length"] == after["length"]
                     and self.raw(before["sha256"]) == self.raw(after["sha256"]),
                     "directly-exported-whole-owner-rollback", frame=f)
                self.adjacent_rollback_pairs += 1
                self.adjacent_rollback_bytes += after["length"]
            else:
                self.unexported_before_rollback_records += 1
        need(sum(x["length"] for x in refused) == report["rollback_bytes"],
             "original-rollback-counter-bound-to-exported-refusals")
        self.seen_records = set()

    def raw(self, digest):
        if digest in self.cache:
            self.cache.move_to_end(digest)
            return self.cache[digest]
        entry = self.blobs[digest]
        raw = single_xz(safe_path(self.directory, entry["path"]), entry)
        self.cache[digest] = raw
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return raw

    def record(self, label, frame):
        key = (label, frame)
        need(key in self.by_label_frame, "exported-owner-label-at-exact-frame", label=label, frame=frame)
        self.seen_records.add(key)
        row = self.by_label_frame[key]
        return {"pointer": row["pointer"], "length": row["length"], "bytes": self.raw(row["sha256"]),
                "live": 1}

    def snapshot(self, captured):
        key = (captured["label"], captured["frame"])
        need(key in self.by_label_frame, "snapshot-owner-present", label=captured["label"])
        row = self.by_label_frame[key]
        need(row["sha256"] == captured["owner_sha256"], "snapshot-owner-digest")
        return self.observer.Memory(self.raw(row["sha256"]))

def recorded_case_class(NativeCase, support):
    class RecordedCase(NativeCase):
        def __init__(self, args, observer, reference, row, directory, original_directory,
                     original_report, owner_module, family):
            self.args, self.observer, self.reference, self.row = args, observer, reference, row
            self.directory = directory
            directory.mkdir(parents=True, exist_ok=False)
            self.original_directory, self.original_report = original_directory, original_report
            self.family = family
            self.live, self.weak, self.tmap, self.treverse = {}, {}, {}, {}
            self.smap, self.sreverse = {}, {}
            self.failures, self.operations, self.snapshots, self.owner_records = [], [], [], []
            self.checks = self.words = self.rng_bytes = self.rollback_bytes = 0
            self.owner_blobs = {}
            self.current_path = "initialization"
            self._snapshots = iter(original_report["snapshots"])
            self._op_active = False
            self._current_operation = None
            self.physical_checks = 0
            self.raw_graph_checks = 0
            self.getter_checks = 0
            self.n = object.__new__(observer.Native)
            self.n.w = args.wire
            self.n.s = RecordedSession(original_directory / "raw", args.wire, original_report["frames"])
            self.owners = Owners(original_directory, original_report, observer, owner_module, self.n.s)
            self.gen = self.n.one(60, 1729)["value"]
            initial = bytes.fromhex(reference["states"][row["state_before_setup"]]["hex"])
            self.n.one(64, self.gen, self.n.w.Payload(initial), len(initial))
            self.check("initial_rng", self.n.one(63, self.gen)["payload"], initial)
            self.rng_bytes += len(initial)

        def owner(self, label):
            result = self.owners.record(label, self.n.s.frames)
            self.owner_records.append({"label": label, "frame": self.n.s.frames,
                "pointer": result["pointer"], "length": result["length"], "sha256": SHA(result["bytes"])})
            return result

        def mem(self):
            # No fallback to a fabricated intermediate owner is permitted.
            raise AssertionError("Intermediate memory read was not exported")

        def issue(self, command, label, *, partial=False):
            status, records = self.n.w.parse(self.n.s.exchange(self.n.w.packet([command])))
            if status:
                self.check(label + "/no_records_on_frame_error", records, [])
                # This is the original exported after-refusal owner, not a made-up
                # before image. Owners separately checks every adjacent exported pair.
                after = self.owner(label + "/frame_refusal")
                self.rollback_bytes += len(after["bytes"])
                return status, None
            self.check(label + "/record_count", len(records), 1, True)
            self.check(label + "/record_opcode", records[0]["opcode"], command[0], True)
            record = copy.deepcopy(records[0])
            if partial:
                self.check(label + "/partial_payload_bytes", len(record["payload"]), 32, True)
                fields = struct.unpack("<4Q", record["payload"])
                self.check(label + "/partial_status_consistent", fields[0], record["value"])
                record["partial_fields_recorded_not_Torch_oracle_counters"] = {
                    "status": fields[0], "completed_nodes": fields[1],
                    "completed_leaves": fields[2], "failed_opcode": fields[3]}
                return record["value"], record
            return 0, record

        def raw_tensor(self, token=None, *, index=None):
            need(self._op_active and index is None, "only-exported-operation-Tensor-read")
            actual = self._current_operation.get("returned_tensor")
            need(actual is not None, "exported-returned-Tensor-present")
            memory = self.owners.snapshot(self._operation_after_snapshot)
            need(memory.node(actual["native_node"])[24] == token,
                 "returned-token-matches-exported-real-T-row")
            current = memory.tensor(actual["native_node"])
            same(current, actual, "returned-Tensor-physical-owner")
            self.physical_checks += 1
            return copy.deepcopy(actual)

        def physical(self, value, memory, label):
            if value is None:
                return
            current = memory.tensor(value["native_node"])
            # requires/grad_fn/is_leaf can be lazily refreshed after the exported
            # before-getter observation. Only immutable geometry/bytes/identity
            # fields are corroborated by this later committed owner.
            for key in ("native_incarnation", "dtype", "element_size", "shape", "strides",
                        "numel", "storage_bytes", "storage_offset_elements", "version",
                        "logical_bytes_hex", "storage_bytes_hex", "native_storage_id",
                        "native_version_id", "native_base_incarnation", "is_view"):
                same(current[key], value[key], label + "/" + key)
            self.physical_checks += 1

        def physical_node(self, value, memory, label):
            if value is None:
                return
            if "variable" in value:
                self.physical(value["variable"], memory, label + "/Accum-variable")
            for i, edge in enumerate(value.get("next_edges", [])):
                self.physical_node(edge["node"], memory, label + "/edge%d" % i)

        def owner_node_info(self, memory, index, depth=4):
            if not index:
                return None
            node = memory.node(index)
            if node[5] == 0:
                target = (node[76] & 65535) or index
                result = {"kind": "AccumulateGrad", "variable": memory.tensor(target)}
                if depth:
                    result["next_edges"] = []
                return result
            kinds = {3: "MulBackward0", 10: "TBackward0", 16: "ViewBackward0",
                     17: "CloneBackward0", 43: "ToCopyBackward0", 44: "AsStridedBackward0",
                     45: "ReshapeAliasBackward0", 46: "UnsafeViewBackward0"}
            need(node[5] in kinds, "raw-current-AD-opcode", opcode=node[5])
            result = {"kind": kinds[node[5]]}
            if depth:
                count = 2 if node[5] == 3 else 1
                result["next_edges"] = [{"output_nr": 0,
                    "node": self.owner_node_info(memory, node[6+i], depth-1) if node[9+i] else None}
                    for i in range(count)]
            return result

        def metadata(self, token, raw, requires):
            meta = self.n.w.meta(self.n.one(13, token))
            expected = {"rank": len(raw["shape"]), "numel": raw["numel"],
                "requires_grad": int(requires), "storage_id": raw["native_storage_id"],
                "version_id": raw["native_version_id"], "version": raw["version"],
                "storage_offset": raw["storage_offset_elements"] * raw["element_size"],
                "storage_bytes": raw["storage_bytes"]}
            for key, value in expected.items():
                self.check("raw13/" + key, meta[key], value)
            self.check("raw13/shape", meta["shape"][:meta["rank"]], raw["shape"])
            self.check("raw13/strides", meta["strides"][:meta["rank"]], raw["strides"])
            self.getter_checks += 1

        def gradient_observation(self, token, expected, actual, fn, retained, memory):
            presence = self.n.one(59, token)["value"]
            public_before = set(self.n.s.public)
            got = self.n.one(91, token)["value"]
            self.check("gradient_presence_public", bool(presence), expected["tensor"] is not None)
            self.check("gradient_getter_public", bool(got), expected["tensor"] is not None)
            self.check("gradient_exported_presence", bool(got), actual is not None)
            pindex = memory.public_node(token)
            parent = memory.node(pindex)
            gindex = parent[48] if parent[45] else 0
            self.check("gradient_final_owner_binding", bool(got), bool(gindex))
            if actual is not None:
                self.check("gradient_node_binding", actual["native_node"], gindex)
                self.check("gradient_public_token", got, memory.node(gindex)[24])
            self.tensor(expected["tensor"], actual, "gradient_tensor")
            self.physical(actual, memory, self.current_path + "/gradient")
            self.check("gradient_getter_accepted", expected["accepted"], True)
            self.check("gradient_retained_probe", bool(self.n.one(94, token)["value"]), retained)
            warning = bool(expected["warnings"])
            self.check("nonleaf_warning_condition", bool(fn) and not retained, warning)
            if warning:
                self.check("warning_count", len(expected["warnings"]), 1)
                self.check("warning_type", expected["warnings"][0]["type"], "UserWarning")
            if got and expected["tensor"] is not None:
                key = "observed_grad_storage_" + expected["tensor"]["storage_identity"]
                if key not in self.weak:
                    self.add_weak(key, got)
            self.close_temporary(got, got in public_before)
            self.getter_checks += 3

        def saved_observation(self, token, expected, actual, memory):
            node = memory.node(memory.public_node(token))
            op = node[5]
            keys = (("_saved_size", "_saved_storage_offset", "_saved_stride") if op == 44 else
                    ("_saved_self_sym_sizes",) if op in (16, 45, 46) else
                    ("_saved_other", "_saved_self") if op == 3 else ())
            self.check("Saved_attribute_names", list(keys), list(expected))
            self.check("Saved_export_names", list(actual), list(expected))
            for name in keys:
                event, observed = expected[name], actual[name]
                self.check(name + "/warnings", event["warnings"], [])
                if op == 3:
                    before = set(self.n.s.public)
                    slot = 1 if name == "_saved_other" else 0
                    status, record = self.issue((107, token, slot), name)
                    self.check(name + "/normalized_status", status, support.expected_status(event))
                    self.check(name + "/accepted", observed["accepted"], event["accepted"])
                    if status:
                        self.check(name + "/wire_error", observed["normalized_wire_status"], status)
                    else:
                        got = record["value"]
                        value = event["value"]
                        expected_tensor = value.get("tensor") if isinstance(value, dict) else None
                        observed_tensor = observed["value"].get("tensor") if isinstance(observed["value"], dict) else None
                        self.check(name + "/exported_presence", bool(got), observed_tensor is not None)
                        self.tensor(expected_tensor, observed_tensor, name + "/original_saved_tensor")
                        if observed_tensor is not None:
                            self.check(name + "/real_original_token", got, memory.node(observed_tensor["native_node"])[24])
                            self.physical(observed_tensor, memory, self.current_path + "/" + name)
                        else:
                            self.check(name + "/None_value", observed["value"], value)
                        self.close_temporary(got, got in before)
                    self.getter_checks += 1
                else:
                    if op in (16, 45, 46):
                        parent = memory.node(node[6])
                        value = list(parent[12:12+parent[0]])
                    else:
                        value = (list(node[12:12+node[0]]) if name == "_saved_size"
                            else list(node[16:16+node[0]]) if name == "_saved_stride"
                            else node[29] // {0: 4, 1: 1, 2: 8}[node[72]])
                    self.check(name + "/accepted", event["accepted"], True)
                    self.check(name + "/value", value, event["value"])
                    same(observed, support.event_normal(True, True, value), "raw-owner-primitive-Saved")

        def check_public_owner(self, memory):
            public = {memory.node(i)[24] for i in range(1, memory.q(8))
                      if memory.node(i)[38] and memory.node(i)[25]}
            same(sorted(public), sorted(self.n.s.public), "recorded-public-capabilities-vs-whole-owner")
            for kind, observed in ((4, self.n.s.weak), (6, self.n.s.hold), (7, self.n.s.generators)):
                actual = set()
                for i in range(1, memory.slots + 1):
                    at = memory.start - 16 - 552*memory.slots + ((kind-1)*memory.slots+i-1)*64
                    # memory.start = BASE+144+552*slots; records start at BASE+128.
                    row = struct.unpack_from("<8Q", memory.raw, at)
                    if row[1] and (kind != 6 or row[3] == 0):
                        actual.add((row[0] << 32) | (kind << 24) | i)
                same(sorted(actual), sorted(observed), "recorded-capability-kind-%d" % kind)

        def snapshot(self, expected, label):
            captured = next(self._snapshots)
            need(captured["label"] == label, "captured-snapshot-order")
            memory = self.owners.snapshot(captured)
            self.current_path = label
            self.check("getter_order", list(self.live), expected["tensor_getter_order"], True)
            self.check("captured_getter_order", captured["tensor_getter_order"], expected["tensor_getter_order"], True)
            self.check("captured_Tensor_names", list(captured["tensors"]), list(self.live), True)
            for name in expected["tensor_getter_order"]:
                self.current_path = label + "/" + name
                token = self.live[name]
                et, at = expected["tensors"][name], captured["tensors"][name]
                raw = at["raw"] if self.family == "32" else at["before_grad_fn_getter"]
                eraw = et["raw"] if self.family == "32" else et["before_grad_fn_getter"]
                self.check("live_Tensor_capability", memory.node(raw["native_node"])[24], token)
                self.tensor(eraw, raw, "raw_before_getters")
                self.physical(raw, memory, self.current_path + "/physical")
                req = raw["requires_grad"] if self.family == "32" else at["requires_after_getter"]
                self.metadata(token, raw, req)
                efn = et["grad_fn"] if self.family == "32" else et["grad_fn_getter"]
                self.check("grad_fn_getter_accepted", efn["accepted"], True)
                self.check("grad_fn_getter_warnings", efn["warnings"], [])
                self.compare_node(efn["node"], at["grad_fn"], "grad_fn_node")
                self.physical_node(at["grad_fn"], memory, self.current_path + "/grad_fn")
                if self.family == "32":
                    self.saved_observation(token, efn["saved"], at["saved"], memory)
                    self.check("leaf_getter_accepted", et["is_leaf"]["accepted"], True)
                    self.check("leaf_getter_warnings", et["is_leaf"]["warnings"], [])
                    self.check("leaf_value", at["is_leaf"], et["is_leaf"]["value"])
                    self.check("retains_grad_getter_accepted", et["retains_grad"]["accepted"], True)
                    self.check("retains_grad_getter_warnings", et["retains_grad"]["warnings"], [])
                    self.check("retains_grad_value", at["retains_grad"], et["retains_grad"]["value"])
                else:
                    self.check("requires_after_getter", at["requires_after_getter"], et["requires_grad_after_grad_fn_getter"])
                    self.check("leaf_getter_accepted", et["is_leaf_after_grad_fn_getter"]["accepted"], True)
                    self.check("leaf_after_getter", at["is_leaf_after_getter"], et["is_leaf_after_grad_fn_getter"]["value"])
                    self.check("retains_grad_accepted", et["retains_grad"]["accepted"], True)
                    self.check("retains_grad", at["retains_grad"], et["retains_grad"]["value"])
                self.check("raw_retains_grad", bool(self.n.one(94, token)["value"]), at["retains_grad"])
                if self.family != "32":
                    self.check("logical_public_vs_real_Storage",
                        self.n.one(98, token, 0, raw["numel"])["payload"].hex(), raw["logical_bytes_hex"])
                self.gradient_observation(token, et["gradient"], at["gradient"],
                                          at["grad_fn"], at["retains_grad"], memory)
                if self.family == "32":
                    self.check("logical_public_vs_real_Storage",
                        self.n.one(98, token, 0, raw["numel"])["payload"].hex(), raw["logical_bytes_hex"])
                self.getter_checks += 2
            self.current_path = label
            if self.family != "32":
                self.check("all_live_identity_relations", captured["relations"], expected["relations"])
                # Independently reconstruct live identity relations from exported
                # real incarnation/S tokens, never from expected relation booleans.
                relations = {}
                names = list(self.live)
                for i, left in enumerate(names):
                    for right in names[i+1:]:
                        a = captured["tensors"][left]["before_grad_fn_getter"]
                        b = captured["tensors"][right]["before_grad_fn_getter"]
                        relations[left + "|" + right] = {
                            "same_Python_object": a["native_incarnation"] == b["native_incarnation"],
                            "same_Tensor": a["native_incarnation"] == b["native_incarnation"],
                            "same_Storage": a["native_storage_id"] == b["native_storage_id"],
                            "same_current_Version_value": a["version"] == b["version"]}
                same(relations, captured["relations"], "exported-live-relations-from-raw-T-S")
                current = {}
                for name, token in self.live.items():
                    raw = captured["tensors"][name]["before_grad_fn_getter"]
                    self.metadata(token, raw, captured["tensors"][name]["requires_after_getter"])
                    index = memory.public_node(token)
                    current[name] = index if memory.node(index)[5] else None
                    self.check("edge_getter_accepted/" + name, expected["edge_getters"][name]["accepted"], True)
                    self.check("edge_getter_warnings/" + name, expected["edge_getters"][name]["warnings"], [])
                for name, index in current.items():
                    self.current_path = label + "/Saved_and_edges/" + name
                    self.saved_observation(self.live[name],
                        expected["saved_getters_after_tensor_getters"][name], captured["saved"][name], memory)
                    children = []
                    if index:
                        node = memory.node(index)
                        for i in range(2 if node[5] == 3 else 1):
                            child = node[6+i] if node[9+i] else None
                            children.append({"output_nr": 0,
                                "node": self.owner_node_info(memory, child),
                                "matches_live_current_grad_fn": [other for other, value in current.items()
                                    if child is not None and child == value]})
                    same(children, captured["edges"][name], "exported-final-edges-vs-raw-owner")
                    self.raw_graph_checks += len(children)
                    want = expected["edges_after_all_tensor_getters"][name]
                    self.check("edge_count", len(children), len(want))
                    for i, (a, e) in enumerate(zip(children, want)):
                        self.check("output_nr/" + str(i), a["output_nr"], e["output_nr"])
                        self.check("live_grad_fn_identity/" + str(i),
                                   a["matches_live_current_grad_fn"], e["matches_live_current_grad_fn"])
                        self.compare_node(e["node"], a["node"], "edge_node/" + str(i))
                self.current_path = label
                self.check("exported_Weak_expiry", self.check_expired(expected["expired"]), captured["expired"])
            state = self.n.one(63, self.gen)["payload"]
            wanted = bytes.fromhex(self.reference["states"][expected["state"]]["hex"])
            self.check("complete_5056_MT", state, wanted)
            self.check("exported_rng_digest", captured["rng_sha256"], SHA(state))
            self.rng_bytes += len(wanted)
            self.check("native_audit", self.n.one(53)["value"], 1)
            owner = self.owner(label)
            self.check("snapshot_frame", self.n.s.frames, captured["frame"], True)
            self.check("exported_owner_digest", SHA(owner["bytes"]), captured["owner_sha256"], True)
            self.check_public_owner(memory)
            self.snapshots.append(copy.deepcopy(captured))

        def execute(self, event, event_index):
            self._current_operation = self.original_report["operations"][event_index]
            self._op_active = True
            wanted_after_label = "event%02d/%s/after" % (event_index, event["label"])
            matches = [s for s in self.original_report["snapshots"] if s["label"] == wanted_after_label]
            need(len(matches) == 1, "unique-operation-after-snapshot")
            self._operation_after_snapshot = matches[0]
            # Exact frozen event body constructs actual requests. No event formula
            # or numeric expectation is substituted by this admission program.
            NativeCase.execute(self, event, event_index)
            self._op_active = False
            same(support.safe(self.operations[-1]), self._current_operation,
                 "original-event-mapping-and-record-vs-exported-operation")

        def run_admission(self):
            final_done = False
            for i, event in enumerate(self.row["events"]):
                if self.family != "32" and event["label"].startswith("final_drop_") and not final_done:
                    self.snapshot(self.row["before_final_drops"], "before_final_drops")
                    final_done = True
                self.execute(event, i)
            if self.family == "32":
                self.snapshot(self.row["final"], "final_observed_snapshot")
                self.current_path = "explicit_root_cleanup"
                want = bytes.fromhex(self.reference["states"][self.row["state_after"]]["hex"])
                self.check("state_after_operations", self.n.one(63, self.gen)["payload"], want)
                self.rng_bytes += len(want)
                for token in list(self.live.values()):
                    self.n.one(41, token)
                self.live.clear()
                want = bytes.fromhex(self.reference["states"][self.row["state_after_explicit_roots_cleared"]]["hex"])
                self.check("state_after_explicit_roots_cleared", self.n.one(63, self.gen)["payload"], want)
                self.rng_bytes += len(want)
                self.check("cleanup_native_audit", self.n.one(53)["value"], 1)
                self.owner("after_explicit_root_cleanup")
            else:
                if not final_done:
                    self.snapshot(self.row["before_final_drops"], "before_final_drops")
                self.current_path = "final"
                self.check("all_public_names_dropped", self.live, {})
                self.check_expired(self.row["final_expired"])
                want = bytes.fromhex(self.reference["states"][self.row["state_after"]]["hex"])
                self.check("final_complete_MT", self.n.one(63, self.gen)["payload"], want)
                self.rng_bytes += len(want)
                self.check("final_native_audit", self.n.one(53)["value"], 1)
                self.owner("final_before_observer_release")
            for token in self.weak.values():
                self.n.one(44, token)
            self.n.one(66, self.gen)
            self.n.finish()
            try:
                next(self._snapshots)
            except StopIteration:
                pass
            else:
                raise AssertionError("unconsumed exported snapshot")
            self.check("all_original_numeric_word_comparisons_recomputed",
                       self.words, self.original_report["exact_words"])
            self.check("all_original_full_rng_byte_comparisons_recomputed",
                       self.rng_bytes, self.original_report["full_rng_bytes"])
            self.check("original_refusal_counter_accounted", self.rollback_bytes,
                       self.original_report["rollback_bytes"])
            same(self.owner_records, self.original_report["owner_records"], "all-exported-owner-records-in-order")
            need(self.owners.seen_records == set(self.owners.by_label_frame), "all-owner-records-consumed")
            need(not self.failures, "recomputed-comparison-failures", first=self.failures[:10])
            result = {
                "schema": "heptatorch.scalar07b.raw-case-admission.v1",
                "status": "PASS_PURE_RECORDED_EVIDENCE",
                "suite": self.family, "index": self.row["index"], "name": self.row["name"],
                "original_report_sha256": SHA(read(self.original_directory / "report.json")),
                "original_assertions": self.original_report["assertions"],
                "pure_comparisons": self.checks, "exact_words_recomputed": self.words,
                "full_rng_bytes_recomputed": self.rng_bytes,
                "all_raw_frames_generated_and_matched": self.n.s.frames,
                "recorded_native_exit_code": self.n.s.exit_code,
                "exported_snapshots_compared": len(self.snapshots),
                "physical_Tensor_reads_from_full_owner": self.physical_checks,
                "final_edge_records_read_from_owner": self.raw_graph_checks,
                "raw_getter_groups_bound": self.getter_checks,
                "full_owner_blobs_invariant_checked": len(self.owners.blobs),
                "full_owner_records": len(self.owner_records),
                "owner_checks": self.owners.snapshot_checks,
                "directly_exported_adjacent_rollback_pairs": self.owners.adjacent_rollback_pairs,
                "directly_exported_adjacent_rollback_bytes": self.owners.adjacent_rollback_bytes,
                "original_refusal_bytes_source_bound": self.rollback_bytes,
                "refusals_without_exported_immediate_before_owner":
                    self.owners.unexported_before_rollback_records,
                "unexported_intermediate_memory_reads_recreated": False,
                "general_ViewInfo_transition_checker_invoked": False,
                "new_native_executions": 0, "new_Torch_executions": 0,
            }
            write(self.directory / "report.json", result)
            return result
    return RecordedCase

def archive_admission(root, config):
    import tarfile
    p = safe_path(root, config["package_directory"])
    package_raw = read(p / "package-manifest.json", config["package_manifest_sha256"])
    package = json.loads(package_raw)
    need(package["schema"] == "heptatorch.scalar07b.native-four-suite-handoff.v1"
         and package["payload_count"] == len(package["payload"]) == 9, "frozen-outer-package")
    need(package["native_source_sha256"] == SOURCE and package["native_image_sha256"] == IMAGE,
         "frozen-package-image")
    for entry in package["payload"]:
        target = safe_path(p, entry["path"])
        read(target, entry["sha256"], entry["bytes"])
        need(target.stat().st_mode & 0o777 == entry["mode"], "outer-Git-copy-mode")
    mraw = read(p / "file-manifest.json", config["member_manifest_sha256"])
    manifest = json.loads(mraw)
    need(manifest["schema"] == "heptatorch.scalar07b.native-four-suite-members.v1"
         and manifest["manifest_itself_is_additional_archive_member"] is True,
         "frozen-member-manifest")
    rows = manifest["members"]
    need(len(rows) == manifest["member_count"] == 3199
         and sum(x["bytes"] for x in rows) == manifest["payload_bytes"] == 169145041,
         "original-member-count-and-bytes")
    by_name = {}
    for row in rows:
        safe_path(p, row["path"])
        need(row["path"] not in by_name, "unique-archive-member")
        by_name[row["path"]] = row
    packed = read(p / "complete.tar.xz", config["archive_sha256"], config["archive_bytes"])
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    at = 0
    # Consume the full XZ checksum and EOF, not only the earlier tar EOF.
    # Limit each decompressed chunk; high compression cannot grow one temporary
    # buffer to the size of the complete evidence archive.
    while not decoder.eof:
        block = packed[at:at+65536] if decoder.needs_input else b""
        at += len(block)
        need(block or not decoder.needs_input, "not-truncated-XZ")
        decoder.decompress(block, max_length=1 << 20)
    need(not decoder.unused_data and at == len(packed), "single-XZ-complete-trailer")
    seen = set()
    with tarfile.open(p / "complete.tar.xz", "r:xz") as tar:
        for member in tar:
            need(member.isfile() and member.name not in seen, "regular-unique-tar-member")
            seen.add(member.name)
            stream = tar.extractfile(member)
            need(stream is not None, "tar-file-stream")
            if member.name == "file-manifest.json":
                need(stream.read() == mraw and member.mode == 0o644, "embedded-member-manifest-exact")
            else:
                need(member.name in by_name, "declared-tar-member", name=member.name)
                entry = by_name[member.name]
                need(member.size == entry["bytes"] and member.mode == entry["mode"],
                     "member-size-and-original-mode", name=member.name)
                digest = hashlib.sha256()
                count = 0
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
                    count += len(block)
                need(count == entry["bytes"] and digest.hexdigest() == entry["sha256"],
                     "complete-member-SHA256", name=member.name)
    need(seen == set(by_name) | {"file-manifest.json"}, "complete-tar-member-coverage")
    # Independently bind the still-present original working files. --root may
    # point to a restored original-runs layout; original absolute paths remain
    # descriptive custody labels, not executable commands.
    oldroot = Path(config["original_runs_root"])
    mapped = {}
    for row in rows:
        original = Path(row["source_path"])
        need(original.is_relative_to(oldroot), "original-source-below-runs")
        target = safe_path(root, original.relative_to(oldroot).as_posix())
        read(target, row["sha256"], row["bytes"])
        need(target.stat().st_mode & 0o777 == row["mode"], "original-member-mode", path=str(target))
        mapped[str(target)] = row
    return {"outer_payloads": 9, "archive_members": 3200,
            "original_payload_bytes": manifest["payload_bytes"],
            "archive_sha256": config["archive_sha256"],
            "single_complete_XZ_stream": True,
            "all_member_SHA_size_modes_recomputed": True}, mapped

def prepare(root, config, output):
    archive, archive_rows = archive_admission(root, config)
    module_paths = {}
    for name, entry in config["modules"].items():
        path = safe_path(root, entry["path"])
        raw = read(path, entry["sha256"], entry["bytes"])
        ast.parse(raw)
        module_paths[name] = path
    support = load(module_paths["support"], "pure_frozen_case_support", SUPPORT)
    wire = load(module_paths["protocol"], "pure_frozen_protocol", config["modules"]["protocol"]["sha256"])
    observer = load(module_paths["observer"], "pure_frozen_observer", OBSERVER)
    owner = load(module_paths["owner"], "pure_frozen_owner", OWNER)
    shape = load(module_paths["shape26"], "pure_frozen_shape26", config["modules"]["shape26"]["sha256"])
    scalar = load(module_paths["scalar32"], "pure_frozen_scalar32", config["modules"]["scalar32"]["sha256"])
    original14 = load(module_paths["direct14"], "pure_frozen_direct14", config["modules"]["direct14"]["sha256"])
    input4 = load(module_paths["input4"], "pure_frozen_input4", config["modules"]["input4"]["sha256"])
    epoch = load(module_paths["identity_epochs"], "pure_frozen_epochs", config["modules"]["identity_epochs"]["sha256"])
    inverse = data(safe_path(root, config["owner_inverse"]["path"]), config["owner_inverse"]["sha256"])
    entry = inverse["files"]["owner_view05.py"]
    need(entry["after_sha256"] == OWNER and entry["whole_byte_inverse"] is True
         and len(entry["edits"]) == 3, "owner-explicit-profile-only-inverse")
    after = read(module_paths["owner"])
    restored = after
    for edit in reversed(entry["edits"]):
        a, b = edit["after"].encode(), edit["before"].encode()
        need(restored.count(a) == edit["count"] == 1, "owner-inverse-occurrence")
        restored = restored.replace(a, b, 1)
    parent_owner = read(module_paths["owner06"], entry["before_sha256"])
    need(restored == parent_owner and SHA(parent_owner) == config["modules"]["owner06"]["sha256"],
         "owner-whole-byte-inverse")
    funcs = lambda raw: {node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(raw).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    same(funcs(after), funcs(parent_owner), "all-owner-function-predicates-unchanged")
    need(owner.SOURCE == SOURCE and owner.IMAGE == IMAGE and owner.DESCRIPTOR == 672,
         "strict-owner-final-profile")

    native_dir = safe_path(root, config["native_directory"])
    source = read(native_dir / "runtime.hepta", SOURCE)
    read(native_dir / "tensor-shape-scalar-session", IMAGE)
    build = data(native_dir / "build.json", BUILD)
    need(build["status"] == "PASS" and build["source_sha256"] == SOURCE and build["native_sha256"] == IMAGE,
         "actual-compiled-profile")
    need(build["functions"] == 658 and build["blocks"] == 8661 and build["instructions"] == 30980
         and build["descriptor_bytes"] == 672 and len(build["units"]) == 13,
         "actual-source-and-unit-counts")
    need(len(support.FUNCTIONS.findall(source)) == 658 and build["link_exit"] == 0,
         "complete-HIA-function-count-and-link-exit")
    for unit in build["units"]:
        need(unit["compiler_exit"] == 0, "real-unit-compiler-exit")
        for suffix, key in ((".hepta", "source_sha256"), (".o", "object_sha256")):
            read(native_dir / (unit["name"] + suffix), unit[key])

    suites, prepared_rows = [], []
    for spec in config["suites"]:
        family = spec["family"]
        directory = safe_path(root, spec["directory"])
        aggregate = data(directory / "report.json", spec["report_sha256"])
        count = int(family)
        need(aggregate["status"] == "PASS" and aggregate["cases"] == aggregate["passed_cases"] == count
             and aggregate["failed_cases"] == 0, "complete-original-suite-PASS", suite=family)
        need(aggregate["source_sha256"] == SOURCE and aggregate["native_sha256"] == IMAGE
             and aggregate["observer_sha256"] == OBSERVER and aggregate["functions"] == 658
             and aggregate["native_units"] == 13 and aggregate["descriptor_bytes"] == 672,
             "suite-exact-final-profile", suite=family)
        need(aggregate["all_inputs_unchanged"] is True and aggregate["native_executed"] is True
             and aggregate["torch_executed_here"] is False
             and aggregate["full_pytorch_rewrite_complete"] is False
             and aggregate["closed_schema_count"] == 0 and aggregate["pytorch_api_denominator"] == 2590,
             "original-suite-scope", suite=family)
        need(aggregate.get("all_sessions_eof_exit_zero",
                           aggregate.get("all_executed_sessions_eof_exit_zero")) is True,
             "suite-all-recorded-EOF", suite=family)
        if family == "26":
            need(aggregate["blocked_cases"] == 0 and aggregate["all_reference_cases_executed"] is True
                 and aggregate["selected_case_indices"] == list(range(26)),
                 "complete26-not-historical24-plus2")
        custody = data(directory / "input-custody.json")
        need(set(custody) == set(aggregate["inputs_sha256"]), "complete-child-input-copies", suite=family)
        copied_hashes = set()
        for original, record in custody.items():
            copied = read(safe_path(directory, record["copy"]), record["sha256"], record["bytes"])
            need(aggregate["inputs_sha256"][original] == SHA(copied), "report-copy-custody")
            copied_hashes.add(SHA(copied))
        ref_raw = read(safe_path(root, spec["reference"]["path"]), spec["reference"]["sha256"])
        probe = read(safe_path(root, spec["probe"]["path"]), spec["probe"]["sha256"])
        need(SHA(ref_raw) == aggregate["reference_sha256"] and SHA(probe) == aggregate["probe_sha256"]
             and SHA(ref_raw) in copied_hashes and SHA(probe) in copied_hashes,
             "actual-reference-is-the-original-native-input")
        if family == "26":
            shape.admit_reference(ref_raw, probe, spec["reference"]["sha256"], support)
            reference, plan = epoch.admit(ref_raw, probe)
            plan["copied_negative_controls"] = epoch.negative_controls(json.loads(ref_raw), plan)
            original_plan = read(directory / "identity-epoch-plan.json",
                                 aggregate["identity_epoch_admission"]["proof_sha256"])
            need((json.dumps(plan, indent=2, allow_nan=False) + "\n").encode() == original_plan,
                 "identity-epoch-proof-recomputed-byte-exact")
            need(plan["retirements"] == 2 and plan["unaffected_rows"] == 24
                 and plan["identity_annotations"] == 208 and len(plan["copied_negative_controls"]) == 6,
                 "bounded-lifetime-adapter-scope")
            need(epoch.inverse(reference) == json.loads(ref_raw), "all-original-reference-fields-recoverable")
            NativeCase = shape.case_class(support)
        elif family == "32":
            reference = scalar.admit_reference(ref_raw, probe, support)
            reference = {**reference, "states": reference["full_rng_states"]}
            NativeCase = scalar.case_class(support)
        elif family == "14":
            reference = original14.admit_reference(ref_raw, probe)
            NativeCase = original14.Case
        else:
            reference = input4.admit_reference(ref_raw, probe, spec["reference"]["sha256"], support)
            NativeCase = input4.case_class(support)
        need(len(reference["rows"]) == len(aggregate["rows"]) == count, "complete-reference-case-list")
        for key in ("assertions", "exact_words", "full_rng_bytes", "rollback_bytes", "frames"):
            need(sum(row[key] for row in aggregate["rows"]) == aggregate[key], "original-suite-total/" + key)
        execution = data(directory.parent / "execution.json")
        intent = data(safe_path(root, spec["intent_path"]), execution["intent_sha256"])
        need(execution["status"] == "PASS" and execution["returncode"] == 0
             and execution["child_started"] and execution["child_completed"]
             and execution["all_inputs_unchanged"] and execution["request_file_unchanged"]
             and execution["child_report_sha256"] == spec["report_sha256"]
             and execution["child_status"] == "PASS" and execution["error"] is None
             and execution["stderr_bytes"] == 0, "original-custody-terminal")
        same(execution["argv"], intent["argv"], "exact-native-request-argv")
        need(intent["native_requested"] is True and intent["torch_requested"] is False
             and intent["execution_owner"] == "/root/pytorch_scope", "original-native-execution-ownership")
        read(directory.parent / "stdout.bin", execution["stdout_sha256"], execution["stdout_bytes"])
        read(directory.parent / "stderr.bin", execution["stderr_sha256"], 0)
        for entry in intent["inputs"]:
            need(entry["path"] in execution["inputs_after"], "outer-input-before-after-coverage")
            same(execution["inputs_after"][entry["path"]],
                 {"sha256": entry["sha256"], "bytes": entry["bytes"]}, "outer-input-unchanged")
        need(set(execution["inputs_after"]) == {x["path"] for x in intent["inputs"]}, "outer-custody-input-set")
        prepared_rows.append({"family": family, "cases": count, "report_sha256": spec["report_sha256"],
            "native_input_copies": len(custody), "outer_inputs": len(intent["inputs"]),
            "reference_sha256": SHA(ref_raw), "probe_sha256": SHA(probe),
            "reference_events": sum(len(row["events"]) for row in reference["rows"])})
        suites.append((spec, aggregate, reference, NativeCase))
    need([x[0]["family"] for x in suites] == ["26", "32", "14", "4"], "four-suite-order")
    prepare_report = {
        "schema": "heptatorch.scalar07b.raw-admission-prepare.v1",
        "status": "PREPARED_PURE_CASE_ADMISSION_NOT_EXECUTED",
        "source_sha256": SOURCE, "native_sha256": IMAGE, "build_sha256": BUILD,
        "functions": 658, "native_units": 13, "archive": archive, "suites": prepared_rows,
        "original_case_comparison_predicates": SUPPORT,
        "owner_observer_sha256": OWNER,
        "owner_predicates_whole_byte_inverse": True,
        "identity_epoch_all_nonidentity_fields_unchanged": True,
        "source_methods_reused_without_edit": ["check", "bind_storage", "bind_tensor", "tensor",
            "compare_node", "add_weak", "close_temporary", "check_expired", "view", "make_base",
            "published", "original family execute body", "observer.Native.create", "observer.Native.one"],
        "I_O_adaptation": "No Native/Session constructor. Source-bound event packets are matched to recorded frames. The issue adapter preserves status/record/partial checks, while unexported before-owner rollback memory is expressly not recreated.",
        "new_native_executions": 0, "new_Torch_executions": 0, "new_compiler_executions": 0,
        "audit_guard": dict(AUDIT),
        "qualification_scope": "Preparation verifies original byte custody and references, not a new pure case admission PASS.",
    }
    write(output / "prepare.json", prepare_report)
    return suites, support, wire, observer, owner

def recheck_bound():
    for path, expected in list(BOUND.items()):
        p = Path(path)
        need(p.is_file() and not p.is_symlink(), "input-still-regular", path=path)
        raw = p.read_bytes()
        need({"sha256": SHA(raw), "bytes": len(raw), "mode": p.stat().st_mode & 0o777} == expected,
             "all-inputs-byte-mode-unchanged", path=path)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--configuration", required=True, type=Path)
    ap.add_argument("--expected-configuration-sha256", required=True)
    ap.add_argument("--expected-reader-sha256", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "heptatorch.scalar07b.raw-four-suite-admission.v1", "status": "STARTING",
              "new_native_executions": 0, "new_Torch_executions": 0, "new_compiler_executions": 0,
              "cases": [], "error": None}
    try:
        read(Path(__file__), args.expected_reader_sha256)
        config = data(args.configuration, args.expected_configuration_sha256)
        need(config["schema"] == "heptatorch.scalar07b.raw-admission-configuration.v1", "configuration-schema")
        suites, support, wire, observer, owner = prepare(args.root, config, output)
        args.wire = wire
        if args.prepare_only:
            report["status"] = "PREPARED_NOT_ADMITTED"
        else:
            for spec, aggregate, reference, NativeCase in suites:
                directory = safe_path(args.root, spec["directory"])
                family = spec["family"]
                Case = recorded_case_class(NativeCase, support)
                for row, summary in zip(reference["rows"], aggregate["rows"]):
                    original_directory = directory / ("case-%02d" % row["index"])
                    original = data(original_directory / "report.json")
                    for key, value in summary.items():
                        same(original[key], value, "original-case-summary/" + key)
                    need(original["status"] == "PASS" and original["error"] is None
                         and original["mismatches"] == [] and original["exit_code"] == 0,
                         "original-case-terminal")
                    need(original["reference_event_count"] == len(original["operations"]) == len(row["events"]),
                         "all-original-events-present")
                    case_output = output / ("suite-" + family) / ("case-%02d" % row["index"])
                    case = Case(args, observer, reference, row, case_output,
                                original_directory, original, owner, family)
                    result = case.run_admission()
                    report["cases"].append(result)
                    print(json.dumps({"suite": family, "case": row["index"], "status": result["status"],
                                      "raw_frames": result["all_raw_frames_generated_and_matched"]}), flush=True)
            need(len(report["cases"]) == 76, "complete-four-suite-admission")
            report["status"] = "PASS_PURE_RECORDED_EVIDENCE"
            for key in ("pure_comparisons", "exact_words_recomputed", "full_rng_bytes_recomputed",
                        "all_raw_frames_generated_and_matched", "exported_snapshots_compared",
                        "physical_Tensor_reads_from_full_owner", "full_owner_blobs_invariant_checked",
                        "full_owner_records", "directly_exported_adjacent_rollback_pairs",
                        "directly_exported_adjacent_rollback_bytes", "original_refusal_bytes_source_bound",
                        "refusals_without_exported_immediate_before_owner"):
                report[key] = sum(row[key] for row in report["cases"])
            need(report["exact_words_recomputed"] == 66165
                 and report["full_rng_bytes_recomputed"] == 11295104
                 and report["all_raw_frames_generated_and_matched"] == 82503
                 and report["original_refusal_bytes_source_bound"] == 2143906608,
                 "same-complete-original-word-state-frame-coverage")
        recheck_bound()
        report["all_inputs_unchanged"] = True
    except BaseException as failure:
        report["status"] = "FAIL_PURE_ADMISSION"
        report["error"] = {"type": type(failure).__name__, "message": str(failure),
                           "traceback": traceback.format_exc()}
        try:
            recheck_bound()
            report["all_inputs_unchanged"] = True
        except BaseException as nested:
            report["all_inputs_unchanged"] = False
            report["custody_error"] = repr(nested)
    report.update(source_sha256=SOURCE, native_sha256=IMAGE, build_sha256=BUILD,
        functions=658, native_units=13, owner_observer_sha256=OWNER, audit_guard=dict(AUDIT),
        original_full_assertions=481580, unexported_intermediate_memory_recreated=False,
        general_ViewInfo_transition_checker_invoked=False,
        comparison_scope="Every exported reference comparison and original event packet is recomputed from frozen inputs. Complete frames/recorded EOF and every exported full owner are verified. Directly exported adjacent rollback pairs are compared byte-for-byte; other original rollback checks remain source-bound executed assertions, never fabricated before bytes.",
        whole_pytorch_gate="HOLD", closed_schema_count=0, pytorch_api_denominator=2590)
    write(output / "inputs.json", BOUND)
    write(output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "source_sha256", "native_sha256",
        "all_inputs_unchanged", "audit_guard", "error")}), flush=True)
    return 0 if report["status"] in ("PREPARED_NOT_ADMITTED", "PASS_PURE_RECORDED_EVIDENCE") else 1

if __name__ == "__main__":
    raise SystemExit(main())
