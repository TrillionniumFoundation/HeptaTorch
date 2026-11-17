"""Independent read-only accounting for the pinned 672-byte dtype/history owner.

This is a test observer, not a numerical runtime or arbitrary-machine-code
verifier.  It copies the committed owner of its own child, decodes real records,
and derives ownership counts.  The old 355/428/446 observer is neither imported
nor modified by this module.  Unsupported transition shapes fail explicitly.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import re
import struct
import time

SCHEMA = "heptatorch.tensor-view-owner05.v1"
BASE = 139264
DESCRIPTOR = 672
TAG = 1 << 63
MAX_U64 = (1 << 64) - 1
SOURCE = "3c2c0a125bcac6d47075bf58fc3f78c0e7b9a0b1370757981b99dcd3793b21c6"
IMAGE = "7841b19051960b6fa13cb98841dbb8e4d6ead97f03e2ad65af415e1a35b03221"
PROFILE = "differentiable-view-r3-627"
TRANSIENTS = (336, 368, 376, 416, 424, 432, 440, *range(448, 576, 8), 616)
HEADER_TRANSIENTS = (0, 48, 56, 80, 88, 96, 104, 112, 3968, 3976, 3984, 3992, 4000, 4008)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def need(condition, rule, **facts):
    if not condition:
        raise AssertionError({"observer_rule": rule, **facts})


def w(node, offset):
    return node[offset // 8]


def token(kind, slot, row):
    return (row[0] << 32) | (kind << 24) | slot


def pin_profile(binary, source):
    native, authored = Path(binary).read_bytes(), Path(source).read_bytes()
    need(sha(native) == IMAGE and sha(authored) == SOURCE, "native-profile-pin")
    return {"name": PROFILE, "source_sha256": SOURCE, "native_sha256": IMAGE,
            "state": "HTVAD003", "backing": "HTOVAD03", "descriptor_bytes": DESCRIPTOR}


def decode_owner(raw, pointer=0):
    """Decode only bytes copied from the actual committed, relative-offset owner."""
    empty = {"next": 1, "incarnation": 0, "epoch": 0, "keep": 0,
             "slots": 0, "pool": 0, "resize_epoch": 0}
    result = {"schema": SCHEMA, "profile": PROFILE, "source_sha256": SOURCE,
              "native_sha256": IMAGE, "pointer": pointer, "length": len(raw),
              "header": empty, "nodes": {}, "tables": {}, "edges": {},
              "pages": [], "free_extents": [], "hold_regions": {},
              "optimizer_blobs": {}, "generator_states": {}, "saved_layouts": {}, "binary_saved_roots": {},
              "full_sha256": sha(raw)}
    if not raw:
        need(pointer == 0, "empty-owner-pointer")
        return result
    need(len(raw) >= BASE + 128, "owner-too-small")
    q = lambda at: struct.unpack_from("<Q", raw, at)[0]
    need(raw[64:72] == b"HTVAD003" and raw[BASE:BASE + 8] == b"HTOVAD03", "owner-tags")
    for field in HEADER_TRANSIENTS:
        need(q(field) == 0, "committed-header-transient", field=field)
    need(q(BASE + 8) == 2 and q(BASE + 48) == 1, "backing-schema")
    slots, pool = q(BASE + 16), q(BASE + 24)
    need(0 < slots <= 65536 and slots & (slots - 1) == 0, "slot-capacity")
    need(0 < pool <= 1 << 30 and pool & (pool - 1) == 0, "pool-capacity")
    start = BASE + 144 + 552 * slots
    need(len(raw) == start + pool, "owner-size-formula")
    need(1 <= q(8) <= 65536 and q(40) < q(8), "node-high-water")
    result["header"] = {"next": q(8), "incarnation": q(120), "epoch": q(32),
                        "keep": q(40), "slots": slots, "pool": pool,
                        "resize_epoch": q(BASE + 40)}
    tables = result["tables"]
    for kind in range(1, 9):
        tables[str(kind)] = [list(struct.unpack_from("<8Q", raw,
            BASE + 128 + ((kind - 1) * slots + i) * 64)) for i in range(slots)]

    def row(kind, handle):
        need(handle >> 24 & 255 == kind and 0 < (handle & 0xffffff) <= slots
             and 0 < handle >> 32 <= 0xffffffff, "decode-typed-record")
        value = tables[str(kind)][(handle & 0xffffff) - 1]
        need(value[1] == 1 and value[0] == handle >> 32, "decode-stale-record")
        return value

    def region(address, size):
        need(address & TAG and 0 <= size <= pool, "decode-relative-address")
        offset = address & (TAG - 1)
        need(offset <= pool and size <= pool - offset, "decode-relative-bound")
        return raw[start + offset:start + offset + size]

    def hold(handle):
        hr = row(6, handle)
        sr = row(2, hr[2])
        need(hr[3] == 1 and sr[5] > 0 and sr[7] == 0, "decode-private-hold")
        return region(TAG + sr[2], sr[3])

    for page in range(256):
        address, handle = q(4096 + page * 8), q(6144 + page * 8)
        if not address:
            need(handle == 0, "page-hold-without-address")
            need((page << 8) >= q(8) or page == 0 and q(8) == 1, "missing-descriptor-page")
            continue
        hr = row(6, handle)
        sr = row(2, hr[2])
        need(hr[3] == 1 and sr[3] == DESCRIPTOR * 256 and address == TAG + sr[2],
             "descriptor-page-extent")
        page_raw = region(address, DESCRIPTOR * 256)
        result["pages"].append({"page": page, "address": address, "hold": handle})
        for index in range(page << 8, (page + 1) << 8):
            if 0 < index < q(8):
                result["nodes"][str(index)] = list(struct.unpack_from("<84Q", page_raw,
                    (index & 255) * DESCRIPTOR))
    for key, node in result["nodes"].items():
        if w(node, 304) == 0:
            continue
        op = w(node, 40)
        if op == 0:
            edges = []
        elif op in (37, 38):
            address = w(node, 184)
            count, = struct.unpack("<Q", region(address, 8))
            need(count <= 16777216 and count * 16 + 8 <= pool, "parent-list-bound")
            edges = [list(v) for v in struct.iter_unpack("<QQ", region(address + 8, count * 16))]
        else:
            edges = [[w(node, 48), w(node, 72)], [w(node, 56), w(node, 80)]]
        result["edges"][key] = edges
        if w(node, 584) > 65535:
            original_tensor_pairs = hold(w(node, 584))
            need(len(original_tensor_pairs) == 32, "decode-binary-Saved-Hold-size")
            result["binary_saved_roots"][key] = list(struct.unpack('<4Q', original_tensor_pairs))
    count = q(BASE + 32)
    need(count <= slots + 1, "free-extent-count")
    result["free_extents"] = [list(struct.unpack_from("<QQ", raw,
        BASE + 128 + slots * 512 + i * 16)) for i in range(count)]
    for i, hr in enumerate(tables["6"], 1):
        if hr[1] and hr[3]:
            value = hold(token(6, i, hr))
            result["hold_regions"][str(token(6, i, hr))] = {
                "storage": hr[2], "bytes": len(value), "sha256": sha(value)}
    for i, sr in enumerate(tables["5"], 1):
        if sr[1] and sr[7]:
            value = hold(sr[7])
            need(len(value) == 128, "saved-layout-size")
            result["saved_layouts"][str(token(5, i, sr))] = list(struct.unpack("<16Q", value))
    for i, gr in enumerate(tables["7"], 1):
        if gr[1]:
            result["generator_states"][str(token(7, i, gr))] = hold(gr[2]).hex()
    for i, op in enumerate(tables["8"], 1):
        if op[1]:
            value = hold(op[2])
            need(len(value) % 8 == 0, "optimizer-blob-alignment")
            result["optimizer_blobs"][str(token(8, i, op))] = list(struct.unpack("<" +
                "Q" * (len(value) // 8), value))
    return result


class OwnerReader:
    """Read-only observation at the frozen wrapper's next 16-byte frame read."""
    def __init__(self, process):
        self.process, self.control = process, None

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

    def snapshot(self):
        control = self.settled_control()
        with open(f"/proc/{self.process.pid}/mem", "rb", buffering=0) as stream:
            def read(at, size):
                need(at >= 0 and size >= 0, "negative-read-range")
                value = os.pread(stream.fileno(), size, at)
                need(len(value) == size, "short-owner-read")
                return value
            area = read(control, 200)
            pointer, length = struct.unpack_from("<QQ", area, 32)
            need(struct.unpack_from("<Q", area, 192)[0] == 1, "wrapper-no-live-owner")
            need(area[128:144] == bytes(16) and area[160:192] == bytes(32), "wrapper-live-call-slots")
            need(bool(pointer) == bool(length), "owner-pointer-pair")
            # Explicit observer resource boundary; never pretend a skipped owner
            # was verified. Production itself has a different, larger pool bound.
            need(length <= 128 << 20, "observer-owner-byte-budget", length=length)
            raw = b"".join(read(pointer + at, min(65536, length - at))
                           for at in range(0, length, 65536))
        self.settled_control()
        snapshot = decode_owner(raw, pointer)
        verify_snapshot(snapshot)
        return snapshot


def verify_snapshot(snapshot):
    """Pure invariant checker; negative controls use isolated copied JSON only."""
    need(snapshot["schema"] == SCHEMA and snapshot["profile"] == PROFILE
         and snapshot["source_sha256"] == SOURCE and snapshot["native_sha256"] == IMAGE, "snapshot-profile")
    h, tables, nodes = snapshot["header"], snapshot["tables"], snapshot["nodes"]
    if not snapshot["length"]:
        need(h == {"next": 1, "incarnation": 0, "epoch": 0, "keep": 0,
                   "slots": 0, "pool": 0, "resize_epoch": 0}
             and snapshot["pointer"] == 0 and all(not snapshot[k] for k in
                ("nodes", "tables", "edges", "pages", "free_extents", "hold_regions",
                 "optimizer_blobs", "generator_states", "saved_layouts", "binary_saved_roots")), "empty-owner")
        return {"live_nodes": 0, "public_nodes": [], "unrooted_strong_cycles": [], "stats": [0] * 12}
    slots, pool = h["slots"], h["pool"]
    need(0 < slots <= 65536 and slots & (slots - 1) == 0 and 0 < pool <= 1 << 30
         and pool & (pool - 1) == 0, "snapshot-capacity")
    need(snapshot["length"] == BASE + 144 + 552 * slots + pool, "snapshot-size-formula")
    need(1 <= h["next"] <= 65536 and h["keep"] < h["next"] and 0 <= h["incarnation"] <= MAX_U64,
         "snapshot-high-water")
    need(set(nodes) == {str(i) for i in range(1, h["next"])}, "node-index-coverage")
    need(set(tables) == {str(k) for k in range(1, 9)}, "record-kind-coverage")
    for kind in range(1, 9):
        need(len(tables[str(kind)]) == slots, "record-slot-coverage")
        for i, row in enumerate(tables[str(kind)], 1):
            need(len(row) == 8 and all(type(v) is int and 0 <= v <= MAX_U64 for v in row)
                 and row[0] <= 0xffffffff and row[1] in (0, 1), "record-header", kind=kind, slot=i)
            need((row[0] > 0 if row[1] else not any(row[2:])), "record-payload", kind=kind, slot=i)

    def lookup(kind, identity):
        need(type(identity) is int and 0 < identity <= MAX_U64 and identity >> 24 & 255 == kind
             and 0 < (identity & 0xffffff) <= slots and identity >> 32 > 0, "typed-record", kind=kind)
        row = tables[str(kind)][(identity & 0xffffff) - 1]
        need(row[0] == identity >> 32 and row[1] == 1, "stale-record", kind=kind, token=identity)
        return row

    def live_rows(kind):
        return [(token(kind, i, row), row) for i, row in enumerate(tables[str(kind)], 1) if row[1]]

    private_holds = {hid: row for hid, row in live_rows(6) if row[3] == 1}
    need(set(snapshot["hold_regions"]) == {str(hid) for hid in private_holds}, "private-hold-coverage")
    hold_roles = Counter()

    def private_hold(hid, role, size=None):
        hr = lookup(6, hid)
        sr = lookup(2, hr[2])
        need(hr[3] == 1 and hr[4] == 0 and sr[5] > 0 and sr[7] == 0, "private-hold-role", role=role)
        region = snapshot["hold_regions"].get(str(hid))
        need(region is not None and region["storage"] == hr[2] and region["bytes"] == sr[3],
             "private-hold-region", role=role)
        if size is not None:
            need(sr[3] == size, "private-hold-size", role=role, expected=size, observed=sr[3])
        hold_roles[hid] += 1
        return sr

    pages = {}
    for page in snapshot["pages"]:
        i = page["page"]
        need(0 <= i < 256 and i not in pages, "descriptor-page-index")
        sp = private_hold(page["hold"], "descriptor-page", DESCRIPTOR * 256)
        need(page["address"] == TAG + sp[2], "descriptor-page-pointer")
        pages[i] = page
    need(all((i >> 8) in pages for i in range(1, h["next"])), "descriptor-page-coverage")

    live, public, incs = {}, set(), set()
    storage_dtypes = {}
    for key, node in nodes.items():
        i = int(key)
        need(len(node) == 84 and all(type(v) is int and 0 <= v <= MAX_U64 for v in node), "descriptor-width")
        for field in TRANSIENTS:
            need(w(node, field) == 0, "descriptor-transient", node=i, field=field)
        need(all(w(node, f) in (0, 1) for f in (32, 200, 240, 304, 360, 408)), "descriptor-flags", node=i)
        if not w(node, 304):
            need(not any(w(node, f) for f in (200, 208, 240, 88, 312, 384, 392, 400,
                                             584, 592, 600, 608, 624, 632, 640, 648, 656, 664)), "dead-node-roots", node=i)
            continue
        live[i] = node
        incarnation = w(node, 320)
        need(0 < incarnation <= h["incarnation"] and incarnation not in incs, "node-incarnation", node=i)
        incs.add(incarnation)
        need(w(node, 24) == w(node, 248) == 0, "obsolete-gradient-buffer", node=i)
        dtype = w(node, 576)
        need(dtype in (0, 1, 2), "dtype-code", node=i)
        need(dtype == 0 or w(node, 32) == w(node, 384) == w(node, 360) == 0, "nonfloat-gradient-role", node=i)
        rank = w(node, 0)
        need(rank <= 4 and not any(node[12 + rank:16]), "tensor-rank-shape", node=i)
        numel = 1
        for dim in node[12:12 + rank]:
            numel *= dim
        need(numel == w(node, 8) and numel <= 16777216, "tensor-numel", node=i)
        need(w(node, 632) <= h["incarnation"], "history-order-future", node=i)
        if w(node, 200):
            public.add(i)
            need(w(node, 240) == 1, "public-without-T-row", node=i)
        if w(node, 240):
            tr = lookup(1, w(node, 192))
            need(tr[6] == i and tr[7] in (0, 1), "T-row-node-mapping", node=i)
            need(tr[2:5] == [w(node, 216), w(node, 224), w(node, 232)], "T-row-storage-version", node=i)
            sp = lookup(2, tr[2]); lookup(3, tr[3])
            storage_dtypes.setdefault(tr[2], set()).add(dtype)
            need(sp[5] > 0 and sp[7] == 0 and w(node, 16) == TAG + sp[2] + tr[4],
                 "Tensor-storage-address", node=i)
            remaining = sp[3] - tr[4] if tr[4] <= sp[3] else 0
            need(tr[5] == remaining, "Tensor-storage-remainder", node=i)
            if numel:
                width = (4, 1, 8)[dtype]
                last = tr[4] + sum((node[12 + d] - 1) * node[16 + d] for d in range(rank)) * width
                need(last <= sp[3] and width <= sp[3] - last, "Tensor-layout-storage-bound", node=i)
            if dtype != 0:
                need(tr[7] == 0, "nonfloat-retained-hook", node=i)
        else:
            need(w(node, 200) == w(node, 312) == w(node, 384) == w(node, 392) == w(node, 400) == 0,
                 "rowless-T-role", node=i)
        for field in (256, 264, 272):
            if w(node, field):
                private_hold(w(node, field), f"node{i}:{field}")

    def tensor(index, incarnation, rule):
        need(index in live and w(live[index], 240) == 1 and w(live[index], 320) == incarnation,
             rule, node=index, incarnation=incarnation)
        return live[index]

    graph_refs, leaf_refs, grad_refs, owned_refs, opt_refs = (Counter() for _ in range(5))
    strong_arcs = {i: [] for i in live}
    accumulator_arcs, weak_hooks, masks, integer_views, binary_saved_edges = [], [], [], [], []
    view_infos, disabled_copy_edges = [], []
    need(set(snapshot["binary_saved_roots"]) == {str(i) for i,n in live.items() if w(n,584)>65535},
         "binary-Saved-pair-coverage")
    need(set(snapshot["edges"]) == {str(i) for i in live}, "edge-node-coverage")
    for i, node in live.items():
        pairs = snapshot["edges"][str(i)]
        op = w(node, 40)
        need((not pairs if op == 0 else len(pairs) == 2 if op not in (37, 38) else len(pairs) <= 16777216),
             "edge-arity", node=i)
        if op == 43:
            need(w(node,576) == 0 and w(node,32) == 1
                 and not any(w(node,f) for f in (280,288,296,584,592)),
                 "ToCopy-no-Saved-Tensor", node=i)
            need(len(pairs) == 2 and pairs[0][0] != 0 and pairs[0][1] in (0,1)
                 and pairs[1] == [0,0], "ToCopy-real-parent-edge", node=i)
            if pairs[0][1] == 0:
                disabled_copy_edges.append([i, pairs[0][0]])
        if op == 44:
            need(w(node,576) == 0 and w(node,32) == 1
                 and not any(w(node,f) for f in (280,288,296,584,592)),
                 "AsStrided-no-Saved-Tensor", node=i)
            need(pairs[0][0] != 0 and pairs[1] == [0,0],
                 "AsStrided-real-base-edge", node=i)
        for parent, enabled in pairs:
            need(enabled in (0, 1) and 0 <= parent < h["next"], "graph-edge-index", node=i)
            if parent:
                need(parent in live, "graph-edge-dead", node=i, parent=parent)
                if enabled:
                    before = w(live[parent], 632) or w(live[parent], 320)
                    after = w(node, 632) or w(node, 320)
                    need(before < after, "enabled-history-order", node=i, parent=parent)
                graph_refs[parent] += 1
                if enabled and w(live[parent], 40) == 0:
                    leaf_refs[parent] += 1
                strong_arcs[i].append(parent)
            else:
                need(enabled == 0, "enabled-null-edge", node=i)
        g = w(node, 384)
        if g:
            gn = tensor(g, w(node, 392), "gradient-incarnation")
            need(w(node, 360) == 1 and w(gn, 576) == w(node, 576) == 0 and w(gn, 0) == w(node, 0)
                 and w(gn, 8) == w(node, 8) and gn[12:12 + w(node, 0)] == node[12:12 + w(node, 0)],
                 "gradient-shape-dtype", node=i)
            # Public, aliased and nonleaf user-assigned G are real supported roles.
            # Do not apply the old fixed35-only private/op0 restriction here.
            grad_refs[g] += 1; strong_arcs[i].append(g)
        else:
            need(w(node, 392) == w(node, 360) == 0, "empty-gradient-binding", node=i)
        marker = w(node, 664)
        if marker:
            need(marker in (1, 3), "ViewInfo-marker-bits", node=i)
            need(w(node, 240) == 1 and w(node, 408) == 1, "ViewInfo-live-Tensor-role", node=i)
            need(w(node, 576) == 0 or marker == 1, "ViewInfo-nonfloat-own-requires", node=i)
            base = w(node, 640)
            bn = tensor(base, w(node, 648), "typed-base-incarnation")
            need(w(bn, 664) == w(bn, 408) == 0 and w(bn, 576) == w(node, 576),
                 "typed-ultimate-base-role", node=i, base=base)
            need(w(node, 216) == w(bn, 216) and w(node, 224) == w(bn, 224),
                 "typed-view-storage-version", node=i, base=base)
            need(w(bn, 320) < w(node, 320), "typed-base-order", node=i, base=base)
            version_row = lookup(3, w(node, 224))
            need(w(node, 656) <= version_row[2], "ViewInfo-attr-version-future", node=i)
            requires = int(bool(marker & 2 or op != 0 or w(bn, 32)))
            need(w(node, 32) == requires, "ViewInfo-effective-requires", node=i)
            owned_refs[base] += 1; strong_arcs[i].append(base)
            integer_views.append([i, base])
            view_infos.append({'node': i, 'base': base, 'base_incarnation': w(node, 648),
                'attr_version': w(node, 656), 'shared_version': version_row[2],
                'own_requires': int(bool(marker & 2)), 'effective_requires': requires})
        else:
            need(w(node, 640) == w(node, 648) == w(node, 656) == 0,
                 "ViewInfo-absent-fields", node=i)
            need(not (w(node, 576) == 2 and w(node, 240) == 1 and w(node, 408) == 1),
                 "typed-view-missing-base", node=i)
        mask = w(node, 584)
        if op == 0 and w(node,408) == 1 and w(node,240) == 0:
            need(mask == 0 and w(node,592) == 0, "typed-rowless-view-base-retained", node=i)
        if mask > 65535:
            region = private_hold(mask, f"binary-Saved:{i}", 32)
            need(w(node, 592) == TAG + region[2], "binary-Saved-relative-address", node=i)
            need(w(node, 576) == 0 and w(node, 32) == 1 and op in (3,4)
                 and w(node, 328) == 0, "binary-Saved-owner-role", node=i)
            values = snapshot["binary_saved_roots"][str(i)]
            need(len(values) == 4 and all(type(x) is int and 0 <= x <= MAX_U64 for x in values),
                 "binary-Saved-pair-width", node=i)
            need(sha(struct.pack('<4Q',*values)) == snapshot["hold_regions"][str(mask)]["sha256"],
                 "binary-Saved-Hold-content", node=i)
            populated = 0
            for slot in range(2):
                target, incarnation = values[2*slot:2*slot+2]
                saved = w(node,280+8*slot)
                needed = w(node,80) if slot == 0 else (w(node,72)|w(node,80) if op==4 else w(node,72))
                need(bool(target) == bool(saved) == bool(needed), "binary-Saved-slot-presence", node=i, slot=slot)
                if target:
                    tn = tensor(target,incarnation,"binary-Saved-original-Tensor-incarnation")
                    need(incarnation < w(node,320), "binary-Saved-original-order", node=i, slot=slot)
                    sr = lookup(5,saved)
                    need(sr[2] == w(tn,216) and sr[3] == w(tn,224)
                         and sr[5] == w(tn,232) and sr[7] == 0,
                         "binary-Saved-original-S-V-offset", node=i, slot=slot)
                    owned_refs[target] += 1; strong_arcs[i].append(target)
                    binary_saved_edges.append([i,slot,target,incarnation,saved,mask])
                    populated += 1
                else:
                    need(incarnation == 0, "binary-Saved-absent-incarnation", node=i, slot=slot)
            need(populated > 0,"binary-Saved-empty-Hold",node=i)
        elif mask:
            mn = tensor(mask, w(node, 592), "Saved-mask-incarnation")
            need(op == 42 and w(node, 288) != 0, "Saved-mask-owner-role", node=i)
            sr = lookup(5, w(node, 288))
            need(sr[2] == w(mn, 216) and sr[3] == w(mn, 224), "Saved-mask-storage-version", node=i)
            need(w(mn, 8) == w(node, 8) and w(mn, 0) == w(node, 0)
                 and mn[12:12 + w(node, 0)] == node[12:12 + w(node, 0)], "Saved-mask-shape", node=i)
            need(w(mn, 576) == (1 if w(node, 8) else 0), "Saved-mask-dtype", node=i)
            owned_refs[mask] += 1; strong_arcs[i].append(mask); masks.append([i, mask])
        else:
            need(w(node, 592) == 0, "empty-Saved-mask-incarnation", node=i)
        role, target_inc = w(node, 608), w(node, 624)
        if w(node, 240):
            need(role >> 16 <= 2 and role & 65536 == 0, "Tensor-history-role-bits", node=i)
            cached = role & 65535
            if cached:
                need(cached in live and w(live[cached], 240) == 0 and w(live[cached], 320) == target_inc
                     and w(live[cached], 608) == i and w(live[cached], 624) == w(node, 320),
                     "cached-Accumulator-backlink", node=i)
            else:
                need(target_inc == 0, "empty-cached-Accumulator-incarnation", node=i)
        elif role >= 65536:
            need(role >> 16 == 1 and role & 65535 and op != 0 and 0 < target_inc <= h["incarnation"],
                 "weak-retained-hook-role", node=i)
            weak_hooks.append([i, role & 65535, target_inc])
            # The target may already be dead/reused. This is a weak-self hook,
            # not an additional owner or a requirement to keep Tensor storage.
        elif role:
            tensor(role, target_inc, "captured-Accumulator-target")
            need(op == 0 and w(node, 200) == w(node, 240) == 0 and w(node, 632) != 0,
                 "captured-Accumulator-role", node=i)
            owned_refs[role] += 1; strong_arcs[i].append(role); accumulator_arcs.append([i, role])
        else:
            need(target_inc == 0, "empty-history-target-incarnation", node=i)

    need(set(snapshot["optimizer_blobs"]) == {str(t) for t, _ in live_rows(8)}, "Optimizer-blob-coverage")
    opt_state_nodes = set()
    for oid, row in live_rows(8):
        algorithm, groups, count = row[7], row[3], row[4]
        need(algorithm in (0, 1, 2) and 0 < groups <= 1024 and count <= 4096,
             "Optimizer-header", optimizer=oid)
        head, gs, es = (64, 64, 48) if algorithm == 0 else (96, 80, 80)
        private_hold(row[2], "Optimizer-blob", head + groups * gs + count * es)
        data = snapshot["optimizer_blobs"][str(oid)]
        need(all(type(v) is int and 0 <= v <= MAX_U64 for v in data)
             and len(data) * 8 == head + groups * gs + count * es and data[:2] == [groups, count],
             "Optimizer-blob-shape", optimizer=oid)
        need(sha(struct.pack("<" + "Q" * len(data), *data))
             == snapshot["hold_regions"][str(row[2])]["sha256"], "Optimizer-region-hash", optimizer=oid)
        if algorithm:
            need(data[2:4] == [algorithm, 0], "Adam-blob-kind", optimizer=oid)
        def entry(index):
            base = (head + groups * gs + index * es) // 8
            return data[base:base + es // 8]
        first, duplicates, unique = 0, 0, set()
        for group in range(groups):
            at = (head + group * gs) // 8
            fields = data[at:at + gs // 8]
            start, amount = fields[-2:]
            need(start == first and amount <= count - first, "Optimizer-group-range", optimizer=oid)
            for index in range(first, first + amount):
                ep = entry(index)
                pn = tensor(ep[0], ep[1], "Optimizer-live-identity")
                need(w(pn, 576) == 0, "Optimizer-parameter-dtype", optimizer=oid)
                # Leaf qualification is required when adding a parameter, not
                # forever after a legal same-Tensor history transformation.
                opt_refs[ep[0]] += 1
                key = ep[2]
                need(first <= key <= index, "Optimizer-key-range", optimizer=oid)
                kp = entry(key)
                need(kp[:3] == [ep[0], ep[1], key], "Optimizer-key-identity", optimizer=oid)
                if key != index:
                    need(not any(ep[3:]), "Optimizer-duplicate-state", optimizer=oid)
                    duplicates += 1
                    continue
                identity_pair = (ep[0], ep[1])
                need(identity_pair not in unique, "Optimizer-duplicate-key", optimizer=oid)
                unique.add(identity_pair)
                present = ep[3]
                if algorithm == 0:
                    need(present in (0, 1), "SGD-state-mask", optimizer=oid)
                    states = [(ep[4], ep[5])] if present else []
                else:
                    need(present in (0, 14), "Adam-state-mask", optimizer=oid)
                    states = [(ep[4 + j * 2], ep[5 + j * 2]) for j in range(3)] if present else []
                if not present:
                    need(not any(ep[4:]), "uninitialized-Optimizer-state", optimizer=oid)
                else:
                    opt_refs[ep[0]] += 1  # actual state-dict key owns its parameter
                need(len({n for n, _ in states}) == len(states), "Optimizer-state-distinct", optimizer=oid)
                for ordinal, (state, incarnation) in enumerate(states):
                    sn = tensor(state, incarnation, "Optimizer-state-identity")
                    need(state != ep[0] and w(sn, 40) == 0 and w(sn, 576) == 0,
                         "Optimizer-state-role", optimizer=oid)
                    if algorithm and ordinal == 0:
                        need(w(sn, 0) == 0 and w(sn, 8) == 1, "Adam-step-shape", optimizer=oid)
                    else:
                        need(w(sn, 0) == w(pn, 0) and sn[12:12 + w(pn, 0)] == pn[12:12 + w(pn, 0)],
                             "Optimizer-state-shape", optimizer=oid)
                    opt_refs[state] += 1; opt_state_nodes.add(state)
            first += amount
        need(first == count and row[6] == duplicates and row[5] == count - duplicates,
             "Optimizer-occurrence-count", optimizer=oid)

    for i, node in live.items():
        need(w(node, 600) == owned_refs[i], "private-strong-Tensor-count", node=i,
             expected=owned_refs[i], observed=w(node, 600))
        need(w(node, 312) == opt_refs[i], "Optimizer-private-reference-count", node=i,
             expected=opt_refs[i], observed=w(node, 312))
        need(w(node, 400) == grad_refs[i], "gradient-reference-count", node=i)
        need(w(node, 88) == leaf_refs[i], "Accumulator-reference-count", node=i)
        expected = graph_refs[i] + opt_refs[i] + grad_refs[i] + owned_refs[i]
        need(w(node, 208) == expected, "node-reference-count", node=i, expected=expected, observed=w(node, 208))
    for tid, row in live_rows(1):
        need(row[6] in live and w(live[row[6]], 240) == 1 and w(live[row[6]], 192) == tid,
             "unmapped-T-row", token=tid)

    # A legitimate captured Accumulator can keep its Tensor in a real strong
    # cycle after public references disappear. Diagnose that exact SCC rather
    # than weakening reachability for every anonymous node.
    roots = public | set(opt_refs)
    reachable = _reach(roots, strong_arcs)
    components = _scc(strong_arcs)
    cycles = []
    for component in components:
        cyclic = len(component) > 1 or any(i in strong_arcs[i] for i in component)
        if cyclic:
            need(any(a in component and b in component for a, b in accumulator_arcs),
                 "unexplained-ownership-cycle", nodes=sorted(component))
            if not component & reachable:
                cycles.append(sorted(component))
    accounted = _reach(reachable | {n for c in cycles for n in c}, strong_arcs)
    need(accounted == set(live), "unrooted-acyclic-node", nodes=sorted(set(live) - accounted))

    strong, weak, version = Counter(), Counter(), Counter()
    for kind in (1, 4, 5, 6):
        for rid, row in live_rows(kind):
            lookup(2, row[2])
            (weak if kind == 4 else strong)[row[2]] += 1
            if kind in (1, 5):
                lookup(3, row[3]); version[row[3]] += 1
            if kind == 4:
                need(row[3] in (0, 1, 2) and not any(row[4:]), "Weak-dtype-fields", token=rid)
            if kind == 6:
                need(row[3] in (0, 1) and row[4] in (0, 1, 2) and not any(row[5:])
                     and (row[3] == 0 or row[4] == 0), "Hold-dtype-fields", token=rid)
    need(set(snapshot["saved_layouts"]) == {str(t) for t, row in live_rows(5) if row[7]}, "Saved-layout-coverage")
    for sid, row in live_rows(5):
        if row[7]:
            private_hold(row[7], "Saved-layout", 128)
            layout = snapshot["saved_layouts"][str(sid)]
            need(len(layout) == 16 and all(type(v) is int and 0 <= v <= MAX_U64 for v in layout)
                 and layout[0] <= 4 and layout[2] in (0, 1, 2)
                 and layout[3] == 0 and not any(layout[12:]), "Saved-layout-fields", token=sid)
            need(sha(struct.pack("<16Q", *layout))
                 == snapshot["hold_regions"][str(row[7])]["sha256"], "Saved-layout-region-hash", token=sid)
            storage_dtypes.setdefault(row[2], set()).add(layout[2])
            count = 1
            for dimension in layout[4:4 + layout[0]]:
                count *= dimension
            need(count == layout[1], "Saved-layout-numel", token=sid)
            if layout[2] == 2:
                sr = lookup(2, row[2])
                need(not any(layout[4 + layout[0]:8]), "I64-Saved-unused-dimensions", token=sid)
                if count:
                    last = row[5] + 8 * sum((layout[4 + d] - 1) * layout[8 + d]
                                           for d in range(layout[0]))
                    need(last <= sr[3] and 8 <= sr[3] - last,
                         "I64-Saved-layout-bound", token=sid)
                # A legitimate empty Saved may retain an out-of-range empty
                # byte offset; it grants no payload access, exactly as HIA.
                need(row[6] == (sr[3] - row[5] if row[5] <= sr[3] else 0),
                     "I64-Saved-remaining-bytes", token=sid)
    for sid, codes in storage_dtypes.items():
        need(len(codes) == 1, "Storage-alias-dtype", token=sid)
    for wid, row in live_rows(4):
        if row[2] in storage_dtypes:
            need(storage_dtypes[row[2]] == {row[3]}, "Weak-Storage-dtype", token=wid)
    for hid, row in live_rows(6):
        if row[3] == 0 and row[2] in storage_dtypes:
            need(storage_dtypes[row[2]] == {row[4]}, "Hold-Storage-dtype", token=hid)
    need(set(snapshot["generator_states"]) == {str(t) for t, _ in live_rows(7)}, "Generator-state-coverage")
    for gid, row in live_rows(7):
        need(not any(row[3:]), "Generator-record-fields", token=gid)
        private_hold(row[2], "Generator-state", 5056)
        state = bytes.fromhex(snapshot["generator_states"][str(gid)])
        _verify_mt_state(state)
        need(sha(state) == snapshot["hold_regions"][str(row[2])]["sha256"], "Generator-region-hash")
    for hid in private_holds:
        need(hold_roles[hid] == 1, "private-hold-role-count", token=hid, count=hold_roles[hid])
    intervals = []
    for sid, row in live_rows(2):
        refs = strong[sid]
        need(row[5] == refs and row[6] == weak[sid] + int(refs > 0), "Storage-reference-count", token=sid)
        need(row[7] == int(refs == 0), "Storage-payload-liveness", token=sid)
        if refs:
            need(row[4] == (row[3] + 15) & ~15 and row[2] % 16 == 0
                 and row[2] <= pool and row[4] <= pool - row[2], "Storage-extent", token=sid)
            if row[4]:
                intervals.append((row[2], row[2] + row[4], "Storage"))
        else:
            # so_storage_release clears offset/rounded/strong but retains
            # requested bytes in a same-generation Weak tombstone.
            need(row[2] == row[4] == 0, "expired-Storage-extent", token=sid)
            need(row[3] <= pool, "expired-Storage-requested-bound", token=sid)
    for vid, row in live_rows(3):
        need(row[3] == version[vid] and row[3] > 0 and not any(row[4:]), "Version-reference-count", token=vid)
    previous_end = -1
    for offset, amount in snapshot["free_extents"]:
        need(amount > 0 and offset % 16 == amount % 16 == 0 and offset > previous_end
             and offset <= pool and amount <= pool - offset, "free-extent-shape")
        previous_end = offset + amount
        intervals.append((offset, previous_end, "free"))
    cursor = 0
    for start, end, role in sorted(intervals):
        need(start == cursor and end > start, "pool-exact-partition", offset=start, expected=cursor, role=role)
        cursor = end
    need(cursor == pool, "pool-exact-partition-end")
    counts = [sum(row[1] for row in tables[str(k)]) for k in range(1, 7)]
    stats = [slots, pool, h["resize_epoch"], snapshot["length"],
             sum(n for _, n in snapshot["free_extents"]), *counts, h["next"] - 1]
    return {"live_nodes": len(live), "public_nodes": sorted(public),
            "gradient_nodes": sorted(grad_refs), "Optimizer_roots": sorted(opt_refs),
            "Optimizer_state_nodes": sorted(opt_state_nodes), "Saved_mask_edges": masks,
            "Accumulator_target_edges": accumulator_arcs, "weak_retained_hooks": weak_hooks,
            "typed_ultimate_base_edges": integer_views, "ViewInfo": view_infos,
            "disabled_ToCopy_metadata_edges": disabled_copy_edges,
            "binary_Saved_original_Tensor_edges": binary_saved_edges,
            "unrooted_strong_cycles": sorted(cycles), "stats": stats}


def _verify_mt_state(raw):
    need(len(raw) == 5056, "Generator-state-size")
    u32 = lambda at: struct.unpack_from("<I", raw, at)[0]
    u64 = lambda at: struct.unpack_from("<Q", raw, at)[0]
    left, index = u32(8), u64(16)
    need(u32(12) == 1 and 1 <= left <= 624 and index <= 624
         and ((left == 1 and index == 0) or index > 0 and left + index == 625), "Generator-position")
    need(u64(5016) == u64(5032) == u32(5044) == 0 and u32(5040) <= 1 and u32(5052) <= 1,
         "Generator-cache-header")
    need((u32(5040) != 0 or u64(5024) == 0) and (u32(5052) != 0 or u32(5048) == 0), "Generator-cache-padding")
    need(all(u64(24 + i * 8) <= 0xffffffff for i in range(624)), "Generator-word-width")


def _reach(roots, edges):
    seen, pending = set(), list(roots)
    while pending:
        i = pending.pop()
        if i in seen:
            continue
        need(i in edges, "root-to-dead-node")
        seen.add(i); pending.extend(edges[i])
    return seen


def _scc(edges):
    """Iterative Kosaraju; no Python recursion bound on the actual node graph."""
    seen, order = set(), []
    for start in edges:
        if start in seen:
            continue
        stack = [(start, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
            elif node not in seen:
                seen.add(node); stack.append((node, True))
                stack.extend((child, False) for child in reversed(edges[node]) if child not in seen)
    reverse = {i: [] for i in edges}
    for node, children in edges.items():
        for child in children:
            reverse[child].append(node)
    seen, result = set(), []
    for start in reversed(order):
        if start in seen:
            continue
        component, pending = set(), [start]
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node); component.add(node); pending.extend(reverse[node])
        result.append(component)
    return result


def semantic(snapshot):
    return canonical({k: v for k, v in snapshot.items() if k not in ("pointer", "full_sha256")})


def verify_isolated_transition(before, after, command, records, status=0):
    """Exact isolated-command node allocation/history ledger, not gap filling.

    Setup snapshots and unhandled multi-command/backward transactions use the
    full structural verifier separately; they are never labelled as having
    passed this stricter, explicitly bounded transition verifier.
    """
    verify_snapshot(before); verify_snapshot(after)
    need(not any(w(n,664) for s in (before,after) for n in s["nodes"].values()),
         "new-view-history-transition-not-qualified")
    for old, new in zip(before["tables"].get("2", []), after["tables"].get("2", [])):
        if old[1] and new[1] and old[0] == new[0]:
            need(old[3] == new[3], "same-generation-Storage-requested-bytes")
            need(not old[7] or new[7] == 1, "same-generation-Storage-payload-resurrection")
    if status:
        need(before["pointer"] == after["pointer"] and before["length"] == after["length"]
             and before["full_sha256"] == after["full_sha256"] and semantic(before) == semantic(after),
             "failed-frame-full-owner-rollback")
        return {"status": "PASS", "ordinary_rollback": True, "created_nodes": 0, "history_refreshes": 0}
    # The pinned wire.parse has already rejected every nonzero record status;
    # it returns opcode/value/payload, not a redundant `status` member.
    need(len(command) == 8 and len(records) == 1 and records[0]["opcode"] == command[0],
         "isolated-command-shape")
    code = command[0]
    bh, ah = before["header"], after["header"]
    need(bh["epoch"] == ah["epoch"] and bh["keep"] == ah["keep"], "unrequested-reset")
    allocated, refreshes, refresh_target = 0, 0, None

    def public_argument(handle):
        slots = bh["slots"]
        need(handle >> 24 & 255 == 1 and 0 < (handle & 0xffffff) <= slots, "command-T-handle")
        row = before["tables"]["1"][(handle & 0xffffff) - 1]
        need(row[0] == handle >> 32 and row[1] == 1, "command-T-generation")
        node = before["nodes"][str(row[6])]
        need(w(node, 200) == 1 and w(node, 304) == 1, "command-public-T")
        return row[6], node

    if code in (1, 2, 6, 17, 18, 19, 38, 55, 56, 69, 97):
        allocated = 1
    elif code in (103, 104):
        _, left = public_argument(command[2])
        _, right = public_argument(command[3])
        ca, cb = w(left, 576), w(right, 576)
        need(ca in (0, 1, 2) and cb in (0, 1, 2) and 2 in (ca, cb)
             and 1 <= command[1] <= 7 and (code == 103 or w(left, 32) == w(right, 32) == 0),
             "I64-operation-domain")
        common = 0 if ca == 0 or cb == 0 or command[1] == 7 else 2
        # integer_promote_operand emits one real descriptor for each input
        # whose dtype differs; integer_binary emits its output only for103.
        allocated = int(ca != common) + int(cb != common) + int(code == 103)
        if code == 104:
            need(ca == common == 2 and records[0]["value"] == command[2],
                 "I64-inplace-result-identity")
    elif code == 105:
        _, node = public_argument(command[1])
        need(w(node, 576) == 2 and command[4] in (0, 1, 2), "I64-dropout-domain")
        payload = records[0]["payload"]
        if isinstance(payload, str):
            payload = bytes.fromhex(payload)
        need(len(payload) == 16, "I64-dropout-result-width")
        error, mask = struct.unpack('<2Q', payload)
        allocated = 2 if w(node, 8) else 1
        if w(node, 8) and command[4] != 0:
            need(error == 12 and mask == records[0]["value"] == 0,
                 "I64-dropout-late-cast-result")
        else:
            need(error == 0 and mask != 0 and records[0]["value"] != 0,
                 "I64-dropout-result-identities")
    elif code == 107:
        _, node = public_argument(command[1])
        need(command[2] in (0,1) and not any(command[3:]), "Saved-accessor-command-shape")
        allocated = 0  # publish_tensor toggles an existing live T, no new descriptor
    elif code == 106:
        _, node = public_argument(command[1])
        need(w(node, 576) == 2 and records[0]["value"] == command[1],
             "I64-scalar-result-identity")
    elif code == 99:
        index, node = public_argument(command[1])
        if w(node, 8):
            allocated = 2  # actual output Tensor plus real bool mask Tensor
        else:
            allocated = 1  # float32 empty mask; output is the same Tensor
            if w(node, 32):
                if w(node, 40) == 0 and w(node, 608) & 65535:
                    refreshes, refresh_target = 1, index
                else:
                    allocated += 1  # real rowless prior-history descriptor
            need(records[0]["value"] == command[1], "empty-dropout-same-T")
    elif code == 101:
        _, node = public_argument(command[1])
        need(w(node, 408) == 0 and records[0]["value"] == command[1], "detach-view-or-identity")
        allocated = int(w(node, 40) != 0)
    elif code in (13, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 54,
                  59, 60, 61, 62, 63, 64, 65, 66, 70, 71, 73, 74, 75, 76, 77,
                  78, 79, 90, 91, 92, 93, 94, 95, 96, 98, 100, 102):
        allocated = 0
    else:
        need(False, "observer-transition-scope", opcode=code)
    need(ah["next"] - bh["next"] == allocated, "unexplained-node-allocation",
         opcode=code, expected=allocated, observed=ah["next"] - bh["next"])
    need(ah["incarnation"] - bh["incarnation"] == allocated + refreshes,
         "unexplained-history-counter", opcode=code, expected=allocated + refreshes,
         observed=ah["incarnation"] - bh["incarnation"])
    for i in range(1, bh["next"]):
        need(w(before["nodes"][str(i)], 320) == w(after["nodes"][str(i)], 320), "existing-incarnation-changed")
    increments = set(range(bh["incarnation"] + 1, ah["incarnation"] + 1))
    actual = [w(after["nodes"][str(i)], 320) for i in range(bh["next"], ah["next"])]
    if refresh_target is not None:
        node = after["nodes"][str(refresh_target)]
        # In the pinned empty99 branch the mask allocates first, then the
        # cached Accumulator refresh consumes the final incarnation token.
        need(w(node, 632) == ah["incarnation"] and w(node, 608) & 65535
             == w(before["nodes"][str(refresh_target)], 608) & 65535, "cached-history-refresh")
        actual.append(w(node, 632))
    need(len(actual) == len(set(actual)) and set(actual) == increments, "incarnation-token-attribution")
    return {"status": "PASS", "ordinary_rollback": False, "opcode": code,
            "created_nodes": allocated, "history_refreshes": refreshes,
            "refresh_target": refresh_target, "incarnation_tokens": sorted(actual)}
