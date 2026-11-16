"""Read-only source-pinned accounting for the unchanged frozen Tensor oracle.

No numeric reference or native allocation is executed here. The observer reads
the already-committed owner of its own child process. Typed Tensor identities,
real graph edges, gradient bindings and the monotonic incarnation allocator are
checked independently of the old public-client ordinal ledger.
"""
from pathlib import Path
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
import struct
import time

SCHEMA = "heptatorch.private-node-observer.v1"
BASE = 139264
DESC_BYTES = 576
TAG = 1 << 63
U64 = struct.Struct("<Q")
PROFILES = {
    "be18ab877f38a4027a527783db80a1418a137c7d6da91e5bc39e2af8c044cb6b": {
        "name": "gradient-objects-355",
        "source_sha256": "1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5",
        "state": "HTGRAD02", "backing": "HTOGRAD2",
    },
    "6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe": {
        "name": "gradient-adam-distributions-428",
        "source_sha256": "1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae",
        "state": "HTGAD002", "backing": "HTOGAD02",
    },
}
SOURCE_PINS = {
    "bridge.py": "c2888687f8876ad3505f0e13c9aecddc45695eff22bfdd9fa16d760f9aa3dac5",
    "lifecycle.py": "4a195d07772123342ac3a33aa878aab1663ce0d0d1dbd8ea5ddb110438fd278c",
    "pytorch_tensor_objects_oracle.py": "149b41c455444edeb8d25d0ad98045e29683e780f37f7e210e7985242d359597",
    "pytorch_owned_tensor_oracle.py": "ee3cf72154ac9dfa31055557b3facab8fdd3e14aad03ef6000ced55ccd75ed88",
    "pytorch_session_oracle.py": "35fa5fa3e1b740f901c11d46dca484079b77805004586c03ae42b35be2265577",
    "pytorch_oracle.py": "ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae",
}
FUNCTION_PINS = {
    "descriptor": "3ea0866cdc43a6eda40d576863b84268f1e05d69a3efbd60bac682a94da524df",
    "graph_reset": "e7d463c2e8b82126357059abb4fcd932320f5b92f7da2565ebae74316f858e4b",
    "graph_node_release": "7f7c3f7bccc8fbbee23ef14c5c38a505cb18011dd195c8add68584102b8fcfbd",
    "gradient_new": "622e5631fb34db0a2a21014e42ab004ee8b419106fdad794da9e4ef79c5adada",
    "gradient_audit": "8e28d47f567955c5a7ba40d5aa7f26bdb7797c5ef84862e1dc3fce334eb347e5",
}
SESSIONS = []

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def need(condition, rule, **facts):
    if not condition:
        raise AssertionError({"observer_rule": rule, **facts})

def word(node, offset):
    return node[offset // 8]

def identity(kind, slot, row):
    return (row[0] << 32) | (kind << 24) | slot

def pin_profile(binary, source):
    raw, native = Path(source).read_bytes(), Path(binary).read_bytes()
    image_sha = sha(native)
    need(image_sha in PROFILES, "unapproved-native-image", sha256=image_sha)
    profile = dict(PROFILES[image_sha])
    need(sha(raw) == profile["source_sha256"], "native-source-pin")
    import re
    bodies = {name.decode(): body for body, name in re.findall(
        rb"^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)",
        raw, re.M | re.S)}
    for name, expected in FUNCTION_PINS.items():
        need(name in bodies and sha(bodies[name]) == expected, "native-function-pin", function=name)
    profile["native_sha256"] = image_sha
    profile["function_pins"] = dict(FUNCTION_PINS)
    return profile

class OwnerReader:
    """Only rb access to the caller's own child; never writes native memory."""
    def __init__(self, process, profile):
        self.process, self.profile, self.control = process, profile, None

    def settled_control(self):
        deadline = time.monotonic() + 10
        while True:
            need(self.process.poll() is None, "native-exited-before-observation")
            fields = Path(f"/proc/{self.process.pid}/syscall").read_text().split()
            if fields and fields[0] not in ("running", "-1"):
                values = [int(v, 0) for v in fields]
                if values[0] == 0 and values[1] == 0 and values[3] == 16:
                    control = values[2]
                    if self.control is None:
                        self.control = control
                    need(control == self.control, "wrapper-control-moved")
                    return control
            need(time.monotonic() < deadline, "native-not-at-frame-read")
            time.sleep(.001)

    def _read(self, fd, address, size):
        need(address >= 0 and size >= 0, "negative-read-range")
        data = os.pread(fd, size, address)
        need(len(data) == size, "short-owner-read", requested=size, received=len(data))
        return data

    def snapshot(self, full_digest=False):
        control = self.settled_control()
        with open(f"/proc/{self.process.pid}/mem", "rb", buffering=0) as memory:
            fd = memory.fileno()
            area = self._read(fd, control, 200)
            pointer, length = struct.unpack_from("<QQ", area, 32)
            need(U64.unpack_from(area, 192)[0] == 1, "wrapper-no-live-owner")
            need(area[128:144] == bytes(16) and area[160:192] == bytes(32),
                 "wrapper-left-live-call-slots")
            need(bool(pointer) == bool(length), "empty-owner-pair")
            if not length:
                return {
                    "schema": SCHEMA, "profile": self.profile["name"], "pointer": 0, "length": 0,
                    "header": {"next": 1, "incarnation": 0, "epoch": 0, "keep": 0,
                               "slots": 0, "pool": 0, "resize_epoch": 0},
                    "nodes": {}, "tables": {}, "edges": {}, "free_extents": [],
                    "full_sha256": sha(b"") if full_digest else None,
                }
            header = self._read(fd, pointer, 8192)
            h = lambda at: U64.unpack_from(header, at)[0]
            need(header[64:72] == self.profile["state"].encode(), "wrong-state-tag")
            for field in (0, 48, 56, 80, 88, 96, 104, 112, 3968, 3976, 3984, 3992, 4000, 4008):
                need(h(field) == 0, "committed-header-transient", field=field, value=h(field))
            object_header = self._read(fd, pointer + BASE, 128)
            oh = lambda at: U64.unpack_from(object_header, at)[0]
            need(object_header[:8] == self.profile["backing"].encode(), "wrong-backing-tag")
            need(oh(8) == 2 and oh(48) == 1, "wrong-backing-schema")
            slots, pool = oh(16), oh(24)
            need(0 < slots <= 65536 and slots & (slots - 1) == 0, "invalid-slot-capacity")
            need(0 < pool <= 1 << 30 and pool & (pool - 1) == 0, "invalid-pool-capacity")
            pool_base = BASE + 144 + 552 * slots
            need(length == pool_base + pool, "owner-size-formula")
            need(1 <= h(8) <= 65536 and h(40) < h(8), "invalid-node-high-water")
            need(oh(32) <= slots + 1, "free-extent-count")
            tables_raw = self._read(fd, pointer + BASE + 128, 8 * slots * 64)
            tables = {}
            for kind in range(1, 9):
                tables[str(kind)] = [list(struct.unpack_from("<8Q", tables_raw,
                    ((kind - 1) * slots + i) * 64)) for i in range(slots)]
            def lookup(kind, token):
                need(token >> 24 & 255 == kind and 0 < (token & 0xffffff) <= slots
                     and token >> 32 > 0, "typed-owner-record", kind=kind, token=token)
                row = tables[str(kind)][(token & 0xffffff) - 1]
                need(row[0] == token >> 32 and row[1] == 1,
                     "stale-owner-record", kind=kind, token=token)
                return row
            def pool_read(address, size):
                need(address & TAG and 0 <= size <= pool, "not-pool-relative-address")
                offset = address & (TAG - 1)
                need(offset <= pool and size <= pool - offset, "pool-read-out-of-bounds")
                return self._read(fd, pointer + pool_base + offset, size)
            nodes, edges = {}, {}
            for page in range(256):
                address, hold = h(4096 + page * 8), h(6144 + page * 8)
                if not address:
                    need(hold == 0, "page-hold-without-page", page=page)
                    need((page << 8) >= h(8) or page == 0 and h(8) == 1,
                         "missing-live-descriptor-page", page=page)
                    continue
                hp = lookup(6, hold)
                need(hp[3] == 1, "descriptor-page-not-private")
                sp = lookup(2, hp[2])
                need(sp[3] == DESC_BYTES * 256 and address == TAG + sp[2],
                     "descriptor-page-extent")
                page_raw = pool_read(address, DESC_BYTES * 256)
                for offset in range(256):
                    index = (page << 8) | offset
                    if 0 < index < h(8):
                        node = list(struct.unpack_from("<72Q", page_raw, offset * DESC_BYTES))
                        nodes[str(index)] = node
            for key, node in nodes.items():
                if not word(node, 304):
                    continue
                op = word(node, 40)
                if op == 0:
                    pairs = []
                elif op in (37, 38):
                    address = word(node, 184)
                    count, = struct.unpack("<Q", pool_read(address, 8))
                    need(count <= 16777216 and 8 + count * 16 <= pool, "parent-list-bound")
                    raw = pool_read(address + 8, count * 16)
                    pairs = [list(pair) for pair in struct.iter_unpack("<QQ", raw)]
                else:
                    pairs = [[word(node, 48), word(node, 72)], [word(node, 56), word(node, 80)]]
                edges[key] = pairs
            raw = self._read(fd, pointer + BASE + 128 + slots * 512, oh(32) * 16)
            extents = [list(pair) for pair in struct.iter_unpack("<QQ", raw)]
            whole_sha = None
            if full_digest:
                digest = hashlib.sha256()
                for offset in range(0, length, 65536):
                    digest.update(self._read(fd, pointer + offset, min(65536, length - offset)))
                whole_sha = digest.hexdigest()
            result = {
                "schema": SCHEMA, "profile": self.profile["name"], "pointer": pointer, "length": length,
                "header": {"next": h(8), "incarnation": h(120), "epoch": h(32), "keep": h(40),
                           "slots": slots, "pool": pool, "resize_epoch": oh(40)},
                "nodes": nodes, "tables": tables, "edges": edges, "free_extents": extents,
                "full_sha256": whole_sha,
            }
        self.settled_control()
        verify_snapshot(result)
        return result

def semantic_snapshot(snapshot):
    result = {key: value for key, value in snapshot.items() if key not in ("pointer", "full_sha256")}
    return canonical(result)

def verify_snapshot(snapshot):
    """Pure checker over a copied snapshot; useful for independent negative tests."""
    need(snapshot["schema"] == SCHEMA, "snapshot-schema")
    header, nodes, tables = snapshot["header"], snapshot["nodes"], snapshot["tables"]
    if not snapshot["length"]:
        need(header == {"next": 1, "incarnation": 0, "epoch": 0, "keep": 0,
                        "slots": 0, "pool": 0, "resize_epoch": 0}
             and not nodes and not tables and not snapshot["edges"], "empty-snapshot")
        return {"live_nodes": 0, "public_nodes": [], "gradient_nodes": [],
                "gradient_incarnations": [], "stats": [0] * 12}
    need(set(nodes) == {str(i) for i in range(1, header["next"])}, "node-index-coverage")
    slots = header["slots"]
    need(set(tables) == {str(k) for k in range(1, 9)}, "record-kind-coverage")
    for kind in range(1, 9):
        rows = tables[str(kind)]
        need(len(rows) == slots, "record-slot-coverage", kind=kind)
        for slot, row in enumerate(rows, 1):
            need(len(row) == 8 and 0 <= row[0] <= 0xffffffff and row[1] in (0, 1),
                 "record-header", kind=kind, slot=slot)
            if row[1] == 0:
                need(not any(row[2:]), "dead-record-payload", kind=kind, slot=slot)
            else:
                need(row[0] > 0, "zero-record-generation", kind=kind, slot=slot)
    # The fixed35 suite does not submit Generator or owned Optimizer commands.
    need(not any(r[1] for k in ("7", "8") for r in tables[k]), "unexpected-owned-object-in-fixed35")
    def lookup(kind, token):
        need(token >> 24 & 255 == kind and 0 < (token & 0xffffff) <= slots
             and token >> 32 > 0, "typed-record", kind=kind, token=token)
        row = tables[str(kind)][(token & 0xffffff) - 1]
        need(row[0] == token >> 32 and row[1] == 1, "stale-record", kind=kind, token=token)
        return row
    live, public, incarnations = {}, set(), set()
    graph_refs, leaf_refs, grad_refs = Counter(), Counter(), Counter()
    gradients = set()
    for key, node in nodes.items():
        index = int(key)
        need(len(node) == 72, "descriptor-width")
        for field in (336, 368, 376, 416, 424, 432, 440, *range(448, 576, 8)):
            need(word(node, field) == 0, "descriptor-transient", node=index, field=field)
        need(word(node, 304) in (0, 1) and word(node, 200) in (0, 1)
             and word(node, 240) in (0, 1), "descriptor-live-state", node=index)
        if word(node, 304) == 0:
            need(all(word(node, f) == 0 for f in (200, 208, 240, 88, 312, 384, 392, 400)),
                 "dead-node-roots", node=index)
            continue
        live[index] = node
        incarnation = word(node, 320)
        need(0 < incarnation <= header["incarnation"] and incarnation not in incarnations,
             "node-incarnation-unique", node=index, incarnation=incarnation)
        incarnations.add(incarnation)
        need(word(node, 312) == 0, "unexpected-private-optimizer-root", node=index)
        need(word(node, 24) == word(node, 248) == 0, "obsolete-gradient-buffer", node=index)
        need(word(node, 360) in (0, 1), "gradient-presence-bit", node=index)
        if word(node, 200):
            public.add(index)
            need(word(node, 240) == 1, "public-without-T-row", node=index)
        if word(node, 240):
            token = word(node, 192)
            row = lookup(1, token)
            need(row[6] == index and row[7] == 0, "T-row-node-mapping", node=index)
            need(row[2] == word(node, 216) and row[3] == word(node, 224)
                 and row[4] == word(node, 232), "T-row-storage-version", node=index)
            lookup(2, row[2]); lookup(3, row[3])
        else:
            need(not any(word(node, f) for f in (88, 312, 384, 392, 400)),
                 "rowless-strong-T-root", node=index)
    need(set(snapshot["edges"]) == {str(i) for i in live}, "edge-node-coverage")
    for index, node in live.items():
        pairs = snapshot["edges"][str(index)]
        op = word(node, 40)
        need((not pairs if op == 0 else len(pairs) == 2 if op not in (37, 38) else True),
             "edge-arity", node=index)
        for parent, enabled in pairs:
            need(enabled in (0, 1) and 0 <= parent < index, "graph-edge-order", node=index)
            if parent:
                need(parent in live, "graph-edge-to-dead-node", node=index, parent=parent)
                graph_refs[parent] += 1
                if enabled and word(live[parent], 40) == 0:
                    leaf_refs[parent] += 1
            else:
                need(enabled == 0, "enabled-null-edge", node=index)
        g = word(node, 384)
        if g == 0:
            need(word(node, 392) == word(node, 360) == 0, "empty-gradient-binding", node=index)
        else:
            need(g in live, "gradient-to-dead-node", node=index, gradient=g)
            q = live[g]
            need(word(q, 320) == word(node, 392), "gradient-incarnation", node=index, gradient=g)
            need(word(node, 360) == 1 and word(q, 240) == 1,
                 "gradient-not-defined", node=index)
            need(word(q, 40) == word(q, 32) == word(q, 200) == 0,
                 "unexpected-public-or-nonleaf-gradient", gradient=g)
            need(word(node, 0) == word(q, 0) and word(node, 8) == word(q, 8)
                 and node[12:12 + word(node, 0)] == q[12:12 + word(q, 0)],
                 "gradient-shape", node=index, gradient=g)
            grad_refs[g] += 1
            gradients.add(g)
    for index, node in live.items():
        need(word(node, 208) == graph_refs[index] + word(node, 312) + grad_refs[index],
             "node-reference-count", node=index, observed=word(node, 208),
             expected=graph_refs[index] + word(node, 312) + grad_refs[index])
        need(word(node, 88) == leaf_refs[index], "accumulator-reference-count", node=index)
        need(word(node, 400) == grad_refs[index], "gradient-reference-count", node=index)
    for slot, row in enumerate(tables["1"], 1):
        if row[1]:
            index = row[6]
            need(index in live and word(live[index], 240) == 1
                 and word(live[index], 192) == identity(1, slot, row),
                 "unmapped-live-T-row", slot=slot)
    reachable, pending = set(), list(public)
    while pending:
        index = pending.pop()
        if index in reachable:
            continue
        need(index in live, "root-to-dead-node")
        reachable.add(index)
        pending.extend(parent for parent, _ in snapshot["edges"][str(index)] if parent)
        if word(live[index], 384):
            pending.append(word(live[index], 384))
    need(reachable == set(live), "anonymous-live-node",
         missing=sorted(set(live) - reachable)[:16])
    # Count actual Storage and Version owners, not merely descriptor aliases.
    strong, weak, version = Counter(), Counter(), Counter()
    for kind in (1, 5, 6, 4):
        for row in tables[str(kind)]:
            if not row[1]:
                continue
            lookup(2, row[2])
            if kind == 4:
                weak[row[2]] += 1
            else:
                strong[row[2]] += 1
            if kind in (1, 5):
                lookup(3, row[3])
                version[row[3]] += 1
    for slot, row in enumerate(tables["2"], 1):
        token = identity(2, slot, row)
        if row[1]:
            count = strong[token]
            need(row[5] == count and row[6] == weak[token] + int(count > 0),
                 "storage-reference-count", slot=slot)
            need(row[7] == int(count == 0), "storage-payload-liveness", slot=slot)
    for slot, row in enumerate(tables["3"], 1):
        if row[1]:
            need(row[3] == version[identity(3, slot, row)] and row[3] > 0,
                 "version-reference-count", slot=slot)
    live_counts = [sum(row[1] for row in tables[str(k)]) for k in range(1, 7)]
    stats = [header["slots"], header["pool"], header["resize_epoch"], snapshot["length"],
             sum(size for _, size in snapshot["free_extents"]), *live_counts, header["next"] - 1]
    return {"live_nodes": len(live), "public_nodes": sorted(public),
            "gradient_nodes": sorted(gradients),
            "gradient_incarnations": sorted(word(live[g], 320) for g in gradients),
            "stats": stats}

def verify_transition(before, after, tickets, commands, status=0):
    """Pure source-bound allocation/root checker. Never fills unexplained gaps."""
    previous, current = verify_snapshot(before), verify_snapshot(after)
    if status:
        need(before["pointer"] == after["pointer"] and semantic_snapshot(before) == semantic_snapshot(after),
             "failed-frame-structural-rollback")
        need(before["full_sha256"] is not None
             and before["full_sha256"] == after["full_sha256"], "failed-frame-full-owner-rollback")
        return {"status": "PASS", "failed_frame": True, "created_program_nodes": 0, "created_private_G": 0}
    old, new = before["header"], after["header"]
    inc_delta = new["incarnation"] - old["incarnation"]
    need(inc_delta >= 0, "incarnation-counter-regressed")
    counts = []
    for ticket in tickets:
        ordinal, count = ticket["record"], ticket["count"]
        need(0 <= ordinal < len(commands) and count > 0
             and ticket["opcode"] == commands[ordinal][0], "invalid-creation-ticket")
        counts.append(count)
    need(len({t["record"] for t in tickets}) == len(tickets), "duplicate-creation-ticket")
    new_gradients = [inc for inc in current["gradient_incarnations"] if inc > old["incarnation"]]
    need(len(new_gradients) == len(set(new_gradients)), "duplicate-gradient-incarnation")
    need(not new_gradients or any(c[0] in (7, 29) for c in commands),
         "gradient-created-without-backward")
    need(inc_delta == sum(counts) + len(new_gradients), "unexplained-incarnation-delta",
         actual=inc_delta, program=sum(counts), private_gradients=len(new_gradients))
    resets = [i for i, c in enumerate(commands) if c[0] == 33]
    need(len(resets) <= 1, "observer-scope-multiple-reset")
    if resets:
        ordinal = resets[0]
        need(not any(c[0] in (7, 29) for c in commands), "observer-scope-backward-with-reset")
        need(not any(t["record"] > ordinal for t in tickets), "observer-scope-create-after-reset")
        need(new["epoch"] == old["epoch"] + 1 and new["keep"] == commands[ordinal][1],
             "reset-epoch-and-prefix")
        highest = max([new["keep"], *[int(i) for i, n in after["nodes"].items() if word(n, 304)]])
        need(new["next"] == highest + 1, "reset-high-water", observed=new["next"], expected=highest + 1)
        # No new node may survive above keep merely through an unexplained root.
        need(set(current["public_nodes"]) == set(range(1, new["keep"] + 1)),
             "reset-public-prefix")
        if not tickets:
            expected, pending = set(), list(range(1, new["keep"] + 1))
            while pending:
                index = pending.pop()
                if index in expected:
                    continue
                node = before["nodes"].get(str(index))
                need(node is not None and word(node, 304), "reset-prefix-was-not-live")
                expected.add(index)
                pending.extend(p for p, _ in before["edges"][str(index)] if p)
                if word(node, 384):
                    pending.append(word(node, 384))
            actual = {int(i) for i, n in after["nodes"].items() if word(n, 304)}
            need(actual == expected, "reset-exact-root-closure")
            for i in actual:
                need(word(before["nodes"][str(i)], 320) == word(after["nodes"][str(i)], 320),
                     "reset-retained-incarnation")
    else:
        need(new["epoch"] == old["epoch"] and new["keep"] == old["keep"], "unrequested-reset")
        need(new["next"] - old["next"] == inc_delta, "node-sequence-delta")
        for i in range(1, old["next"]):
            need(word(before["nodes"][str(i)], 320) == word(after["nodes"][str(i)], 320),
                 "existing-slot-incarnation-changed", node=i)
        actual_new = sorted(word(after["nodes"][str(i)], 320) for i in range(old["next"], new["next"]))
        need(actual_new == list(range(old["incarnation"] + 1, new["incarnation"] + 1)),
             "new-node-incarnation-interval")
    return {"status": "PASS", "failed_frame": False, "created_program_nodes": sum(counts),
            "created_private_G": len(new_gradients), "private_G_incarnations": new_gradients,
            "native_incarnation_delta": inc_delta, "native_node_high_water": new["next"] - 1,
            "live_nodes": current["live_nodes"], "live_gradient_nodes": len(current["gradient_nodes"]),
            "reset_count": len(resets)}

class Journal:
    def __init__(self, directory, case):
        self.directory = Path(directory) / sha(case.encode())[:16]
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = gzip.open(self.directory / "snapshots.jsonl.gz", "wb", compresslevel=1)
        self.saved, self.events, self.full_rollback_bytes = set(), [], 0

    def snapshot(self, value):
        raw = canonical(value)
        digest = sha(raw)
        if digest not in self.saved:
            self.stream.write(canonical({"sha256": digest, "snapshot": value}) + b"\n")
            self.stream.flush()
            self.saved.add(digest)
        return digest

    def event(self, before, after, tickets, commands, status, result=None, error=None):
        row = {"before": self.snapshot(before), "after": self.snapshot(after),
               "tickets": tickets, "commands": commands, "expected_status": status,
               "result": result, "error": error}
        self.events.append(row)
        (self.directory / "events.json").write_text(json.dumps(self.events, indent=2) + "\n")
        if status and result:
            self.full_rollback_bytes += before["length"]

    def close(self):
        self.stream.close()

def install(original_install, scope, definitions, owned_program, base_program, executable,
            trace_directory, profile, observer_directory):
    Program, Session = original_install(scope, definitions, owned_program, base_program, executable, trace_directory)
    OriginalSession = Session.__mro__[1]
    class AccountedProgram(Program):
        def __init__(self):
            super().__init__()
            self.account_starts, self.creation_tickets = {}, []

        def command(self, code, *args):
            before = self.next_id
            ordinal = super().command(code, *args)
            self.account_starts[ordinal] = before
            return ordinal

        def handle(self, slot):
            logical = super().handle(slot)
            ordinal = len(self.commands) - 1
            count = self.next_id - self.account_starts[ordinal]
            need(count > 0, "logical-creation-ticket-not-positive")
            self.creation_tickets.append({"record": ordinal, "opcode": self.commands[ordinal][0],
                                          "count": count, "logical": logical})
            return logical

    class AccountedSession(Session):
        def __init__(self, row):
            super().__init__(row)
            self.reader = OwnerReader(self.process, profile)
            self.journal = Journal(observer_directory, row["case"])
            self.observer_rows, self._expected_status = [], 0
            row["private_node_observer"] = {
                "schema": SCHEMA, "profile": profile["name"], "status": "RUNNING",
                "frames_checked": 0, "program_created_nodes": 0, "private_G_created_nodes": 0,
                "eof_exit_assertions_executed": False, "post_close_parameter_version_assertions_executed": False,
                "snapshot_directory": str(self.journal.directory)}
            SESSIONS.append(self)

        def _v3_begin(self, program, body):
            super()._v3_begin(program, body)
            self._before_owner = self.reader.snapshot(full_digest=bool(self._expected_status))

        def _account(self, program, status):
            after = self.reader.snapshot(full_digest=bool(status))
            try:
                result = verify_transition(self._before_owner, after, program.creation_tickets,
                                           program.commands, status)
            except Exception as error:
                self.journal.event(self._before_owner, after, program.creation_tickets,
                                   program.commands, status, error=repr(error))
                self.row["private_node_observer"]["status"] = "FAIL"
                raise
            self.journal.event(self._before_owner, after, program.creation_tickets,
                               program.commands, status, result=result)
            self._last_owner = after
            self.observer_rows.append(result)
            report = self.row["private_node_observer"]
            report["frames_checked"] += 1
            report["program_created_nodes"] += result["created_program_nodes"]
            report["private_G_created_nodes"] += result["created_private_G"]

        def _v3_commit(self, program):
            super()._v3_commit(program)
            self._account(program, 0)

        def exchange(self, program, expected=None, status=0):
            self._expected_status = status
            result = super().exchange(program, expected, status)
            if status:
                self._account(program, status)
            return result

        def close(self):
            program = self.program()
            index = program.command(54)
            record = self.exchange(program)[index]
            need(record[1] == 1 and len(record[2]) == 96, "actual-stats-reply")
            observed = list(struct.unpack("<12Q", record[2]))
            exact = verify_snapshot(self._last_owner)["stats"]
            need(observed == exact, "actual-stats-owner-exact-equality", observed=observed, expected=exact)
            self.row["v3_bridge"]["final_native_stats"] = observed
            self.row["v3_bridge"]["native_node_sequence_matches_logical_creation_count"] = (
                observed[11] == self.next_id - 1)
            self.row["v3_bridge"]["node_sequence_contract"] = (
                "Internal node high-water is checked against actual owner and every allocation; "
                "legacy public-client ordinal equality is preserved only as a diagnostic observation.")
            report = self.row["private_node_observer"]
            report["final_logical_next_id"] = self.next_id
            report["final_native_high_water"] = observed[11]
            report["final_native_incarnation"] = self._last_owner["header"]["incarnation"]
            report["full_owner_rollback_bytes"] = self.journal.full_rollback_bytes
            # This is the unmodified frozen Session.close body: EOF, exit0 and empty pipes.
            OriginalSession.close(self)
            report["eof_exit_assertions_executed"] = True
            report["status"] = "PASS"

        def stop(self):
            try:
                super().stop()
            finally:
                self.journal.close()

    scope["Program"], scope["Session"] = AccountedProgram, AccountedSession
    original_version_gate = scope["_v2_assert_versions"]
    def checked_version_gate(session):
        original_version_gate(session)
        session.row["private_node_observer"]["post_close_parameter_version_assertions_executed"] = True
    scope["_v2_assert_versions"] = checked_version_gate
    return AccountedProgram, AccountedSession
