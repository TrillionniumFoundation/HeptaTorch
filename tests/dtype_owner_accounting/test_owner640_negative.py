"""Pure copied-snapshot controls for the separately pinned 640-byte observer.

This program never starts a native child, imports Torch, reads /proc, or writes
an owner. Its positive input is the complete, hash-bound native-controls receipt
and its captured JSON snapshots. Mutations are independent JSON copies, not
new numerical or native cases.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import struct
import traceback


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("observer", "native-report", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--expected-observer-sha256", required=True)
    p.add_argument("--expected-native-report-sha256", required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    report = {"schema": "heptatorch.dtype640-copied-negative-controls.v1", "status": "RUNNING",
              "native_executed_this_invocation": False, "Torch_executed": False,
              "process_memory_accessed": False, "owner_mutated": False,
              "whole_pytorch_gate": "HOLD", "pytorch_api_denominator": 2590,
              "full_pytorch_parity": False, "positive_snapshots": [], "controls": []}

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def bind(path, expected=None):
        path = path.resolve(); raw = path.read_bytes()
        assert expected is None or sha(raw) == expected, (str(path), sha(raw), expected)
        inputs[path] = raw
        return raw

    try:
        self_bytes = bind(Path(__file__))
        module_bytes = bind(a.observer, a.expected_observer_sha256)
        raw_report = bind(a.native_report, a.expected_native_report_sha256)
        native = json.loads(raw_report)
        (out / "executed-test.py").write_bytes(self_bytes)
        (out / "observer-under-test.py").write_bytes(module_bytes)
        (out / "native-report.original.json").write_bytes(raw_report)
        spec = importlib.util.spec_from_file_location("copied_owner640_under_test", a.observer)
        observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)
        assert a.observer.read_bytes() == module_bytes
        names = ["bool-mask-Saved-Weak-Hold-and-ordinary-rollback",
                 "captured-Accumulator-refresh-and-real-unrooted-cycle",
                 "weak-captured-retain-hook-and-public-gradient",
                 "SGD-Adam-private-Tensor-roots-and-public-state-gradient-alias"]
        assert native["status"] == "PASS" and native["native_executed"] is True
        assert native["Torch_executed"] is False and native["source_custody_unchanged"] is True
        assert native["case_count"] == native["passed"] == 4 and native["failed"] == 0
        assert [r["name"] for r in native["cases"]] == names
        assert all(r["status"] == "PASS" and r["eof_exit_code"] == 0 for r in native["cases"])
        assert native["source_sha256"] == observer.SOURCE and native["native_sha256"] == observer.IMAGE
        assert native["observer_sha256"] == sha(module_bytes)
        assert native["old_observer_sha256"] == "281f1ea437046a9cd884de2c8da34f38649a3b6482ade23123ac4aa992faed64"
        assert native["frames_before_EOF"] == 68 and len(native["events"]) == 68
        assert native["isolated_transitions_checked"] == 64 and native["explicitly_snapshot_only_events"] == 4
        report.update(source_sha256=observer.SOURCE, native_sha256=observer.IMAGE,
                      observer_sha256=sha(module_bytes), native_report_sha256=sha(raw_report),
                      native_cases_referenced=4, native_frames_referenced=68,
                      parent_native_receipt_is_reused_not_reexecuted=True)
        snapshots = {}
        snapshot_dir = out / "positive-snapshots"; snapshot_dir.mkdir()
        for digest, meta in native["snapshots"].items():
            relative = Path(meta["path"])
            assert not relative.is_absolute() and ".." not in relative.parts
            path = a.native_report.parent / relative
            assert not path.is_symlink() and path.resolve().is_relative_to(a.native_report.parent.resolve())
            compressed = bind(path, meta["compressed_sha256"])
            raw = gzip.decompress(compressed)
            assert len(raw) == meta["bytes"] and sha(raw) == digest
            snapshot = json.loads(raw)
            assert observer.canonical(snapshot) == raw
            summary = observer.verify_snapshot(snapshot)
            snapshots[digest] = snapshot
            (snapshot_dir / (digest + ".json.gz")).write_bytes(compressed)
            report["positive_snapshots"].append({"sha256": digest, "bytes": len(raw),
                "compressed_sha256": sha(compressed), "summary": summary})
        assert snapshots and set(snapshots) == {e[k] for e in native["events"] for k in ("before", "after")}
        original_digest = sha(observer.canonical(snapshots))

        def parse_response(event):
            raw = bytes.fromhex(event["response_hex"])
            assert raw[:8] == b"HTORSP03" and len(raw) >= 32
            version, count, status = struct.unpack_from("<3Q", raw, 8)
            assert version == 3 and status == event["actual_status"]
            rows, position = [], 32
            for _ in range(count):
                opcode, error, value, size = struct.unpack_from("<4Q", raw, position)
                position += 32
                assert error == 0 and size <= len(raw) - position
                rows.append({"opcode": opcode, "value": value, "payload": raw[position:position + size]})
                position += size
            assert position == len(raw)
            return rows

        # Recheck the exact original event ledger, including the four explicitly
        # excluded setup operations. A skipped ledger cannot become a PASS here.
        ledgers = []
        for index, event in enumerate(native["events"]):
            before, after = snapshots[event["before"]], snapshots[event["after"]]
            assert event["snapshot_summary"] == observer.verify_snapshot(after)
            expected = event["ledger"]
            if expected["status"] == "SNAPSHOT_ONLY":
                assert event["actual_status"] == 0 and event["command_words"][0][0] in (58, 72)
                ledgers.append({"event": index, "status": "SNAPSHOT_ONLY"})
            else:
                result = observer.verify_isolated_transition(before, after, event["command_words"][0],
                    parse_response(event), event["actual_status"])
                assert result == expected
                ledgers.append({"event": index, "status": "PASS", "result": result})
        report["original_event_ledgers"] = ledgers
        ca, cb, cc, cd = native["cases"]
        A, B, C, D = [snapshots[c[k]] for c, k in (
            (ca, "negative_control_snapshot"), (cb, "negative_control_snapshot"),
            (cc, "gradient_snapshot"), (cd, "negative_control_snapshot"))]
        w = observer.w

        def node(snapshot, index):
            return snapshot["nodes"][str(index)]

        def put(snapshot, index, offset, value):
            node(snapshot, index)[offset // 8] = value

        def choose(snapshot, predicate):
            return next(int(k) for k, n in snapshot["nodes"].items() if w(n, 304) and predicate(n))

        bool_node = choose(A, lambda n: w(n, 576) == 1)
        nd_node = choose(A, lambda n: w(n, 584) != 0)
        captured = choose(B, lambda n: w(n, 240) == 0 and 0 < w(n, 608) < 65536)
        target = w(node(B, captured), 608)
        weak_hook = choose(C, lambda n: w(n, 240) == 0 and w(n, 608) >> 16 == 1)
        weak_target = w(node(C, weak_hook), 608) & 65535
        grad = w(node(D, choose(D, lambda n: w(n, 384) != 0 and w(n, 8) != 0)), 384)

        def mutation(name, base, rule, edit):
            altered = copy(base)
            edit(altered)
            assert observer.canonical(altered) != observer.canonical(base)
            raw = observer.canonical(altered)
            path = out / "mutations" / (f"{len(report['controls']) + 1:02d}-" + name + ".json.gz")
            path.parent.mkdir(exist_ok=True)
            with path.open("wb") as f:
                with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as z:
                    z.write(raw)
            result = {"name": name, "kind": "copied-snapshot", "expected_rule": rule,
                      "parent_snapshot_sha256": sha(observer.canonical(base)),
                      "mutated_snapshot_sha256": sha(raw), "path": str(path.relative_to(out)),
                      "compressed_sha256": sha(path.read_bytes()), "status": "RUNNING"}
            report["controls"].append(result); save()
            try:
                observer.verify_snapshot(altered)
            except AssertionError as exc:
                detail = exc.args[0]
                result["observed_error"] = detail
                assert isinstance(detail, dict) and detail.get("observer_rule") == rule, (name, rule, detail)
            else:
                raise AssertionError({"unexpected_acceptance": name, "expected_rule": rule})
            assert sha(observer.canonical(snapshots)) == original_digest
            result["status"] = "EXPECTED_REFUSAL"; save()

        mutation("invalid-dtype-code", A, "dtype-code", lambda s: put(s, bool_node, 576, 2))
        mutation("bool-gradient-role", A, "bool-gradient-role", lambda s: put(s, bool_node, 32, 1))
        mutation("Saved-mask-wrong-incarnation", A, "Saved-mask-incarnation",
                 lambda s: put(s, nd_node, 592, w(node(s, nd_node), 592) + 1))

        def wrong_saved_storage(s):
            handle = w(node(s, nd_node), 288)
            s["tables"]["5"][(handle & 0xffffff) - 1][2] = w(node(s, nd_node), 216)
        mutation("Saved-mask-storage-mismatch", A, "Saved-mask-storage-version", wrong_saved_storage)
        mutation("Saved-mask-with-wrong-operation", A, "Saved-mask-owner-role", lambda s: put(s, nd_node, 40, 3))
        mutation("mask-strong-reference-extra", A, "private-strong-Tensor-count",
                 lambda s: put(s, bool_node, 600, w(node(s, bool_node), 600) + 1))
        mutation("total-node-reference-extra", A, "node-reference-count",
                 lambda s: put(s, bool_node, 208, w(node(s, bool_node), 208) + 1))
        mutation("cached-Accumulator-incarnation", B, "cached-Accumulator-backlink",
                 lambda s: put(s, target, 624, w(node(s, target), 624) + 1))

        def wrong_accumulator_target(s):
            put(s, target, 608, 0); put(s, target, 624, 0)
            put(s, captured, 624, w(node(s, captured), 624) + 1)
        mutation("captured-Accumulator-target-incarnation", B, "captured-Accumulator-target", wrong_accumulator_target)
        mutation("Tensor-uses-weak-hook-role-bit", B, "Tensor-history-role-bits",
                 lambda s: put(s, target, 608, w(node(s, target), 608) | 65536))

        def counted_weak_as_strong(s):
            for field in (600, 208):
                put(s, weak_target, field, w(node(s, weak_target), field) + 1)
        mutation("weak-captured-hook-counted-as-owner", C, "private-strong-Tensor-count", counted_weak_as_strong)
        mutation("future-history-order", B, "history-order-future",
                 lambda s: put(s, target, 632, s["header"]["incarnation"] + 1))
        ordered_child = next(int(k) for k, pairs in B["edges"].items()
                             if any(parent and enabled for parent, enabled in pairs))
        enabled_parent = next(p for p, enabled in B["edges"][str(ordered_child)] if p and enabled)
        old_parent_order = w(node(B, enabled_parent), 632) or w(node(B, enabled_parent), 320)
        mutation("enabled-edge-nonincreasing-history", B, "enabled-history-order",
                 lambda s: put(s, ordered_child, 632, old_parent_order))

        def optimizer_parameter_incarnation(s):
            oid, row = next((observer.token(8, i, r), r) for i, r in enumerate(s["tables"]["8"], 1)
                            if r[1] and r[7] == 1)
            words = s["optimizer_blobs"][str(oid)]
            offset = (96 + row[3] * 80) // 8
            words[offset + 1] += 1
            s["hold_regions"][str(row[2])]["sha256"] = sha(struct.pack("<" + "Q" * len(words), *words))
        mutation("Optimizer-parameter-incarnation", D, "Optimizer-live-identity", optimizer_parameter_incarnation)
        optimizer_node = choose(D, lambda n: w(n, 312) > 0)
        mutation("Optimizer-private-count-extra", D, "Optimizer-private-reference-count",
                 lambda s: put(s, optimizer_node, 312, w(node(s, optimizer_node), 312) + 1))
        mutation("public-gradient-reference-extra", D, "gradient-reference-count",
                 lambda s: put(s, grad, 400, w(node(s, grad), 400) + 1))

        def wrong_weak_dtype(s):
            next(r for r in s["tables"]["4"] if r[1] and r[3] == 1)[3] = 0
        def wrong_hold_dtype(s):
            next(r for r in s["tables"]["6"] if r[1] and r[3] == 0 and r[4] == 1)[4] = 0
        mutation("bool-Weak-retyped-f32", A, "Weak-Storage-dtype", wrong_weak_dtype)
        mutation("bool-public-Hold-retyped-f32", A, "Hold-Storage-dtype", wrong_hold_dtype)

        def orphan_saved_layout(s):
            sid = next(iter(s["saved_layouts"]))
            s["tables"]["5"][(int(sid) & 0xffffff) - 1][7] = 0
            del s["saved_layouts"][sid]
        mutation("orphan-real-Saved-layout-Hold", A, "private-hold-role-count", orphan_saved_layout)
        mutation("persistent-audit-scratch", A, "descriptor-transient", lambda s: put(s, nd_node, 616, 1))
        mutation("persistent-Incoming-scratch", A, "descriptor-transient", lambda s: put(s, nd_node, 448, 1))
        orphan = choose(D, lambda n: w(n, 200) == 1 and w(n, 208) == 0 and w(n, 40) == 0)
        mutation("unrooted-acyclic-live-Tensor", D, "unrooted-acyclic-node", lambda s: put(s, orphan, 200, 0))

        def transition_control(name, event_index, expected_rule, alter):
            event = copy(native["events"][event_index])
            before, after = copy(snapshots[event["before"]]), copy(snapshots[event["after"]])
            alter(before, after)
            # These last two controls intentionally preserve all structural
            # invariants. Only the exact transition ledger must reject them.
            observer.verify_snapshot(before); observer.verify_snapshot(after)
            payload = {"event": event, "before": before, "after": after}
            raw = observer.canonical(payload)
            path = out / "mutations" / (f"{len(report['controls']) + 1:02d}-" + name + ".json.gz")
            with path.open("wb") as f:
                with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as z:
                    z.write(raw)
            result = {"name": name, "kind": "copied-transition", "expected_rule": expected_rule,
                      "original_event_index": event_index, "mutated_payload_sha256": sha(raw),
                      "path": str(path.relative_to(out)), "compressed_sha256": sha(path.read_bytes()),
                      "status": "RUNNING", "structural_checks_still_pass": True}
            report["controls"].append(result); save()
            try:
                observer.verify_isolated_transition(before, after, event["command_words"][0],
                    parse_response(event), event["actual_status"])
            except AssertionError as exc:
                detail = exc.args[0]; result["observed_error"] = detail
                assert isinstance(detail, dict) and detail.get("observer_rule") == expected_rule, (name, detail)
            else:
                raise AssertionError({"unexpected_transition_acceptance": name})
            assert sha(observer.canonical(snapshots)) == original_digest
            result["status"] = "EXPECTED_REFUSAL"; save()

        def extra_counter(_before, after):
            after["header"]["incarnation"] += 1
        transition_control("unattributed-incarnation-token", cb["refresh_event"],
                           "unexplained-history-counter", extra_counter)
        rollback_index = next(i for i, event in enumerate(native["events"])
                              if event["actual_status"] == 4)
        def different_owner_bytes(_before, after):
            after["full_sha256"] = "0" * 64 if after["full_sha256"] != "0" * 64 else "f" * 64
        transition_control("ordinary-error-changed-owner-bytes", rollback_index,
                           "failed-frame-full-owner-rollback", different_owner_bytes)
        assert len(report["controls"]) == 24
        assert all(r["status"] == "EXPECTED_REFUSAL" for r in report["controls"])
        report.update(status="PASS", positive_snapshot_count=len(snapshots),
                      original_isolated_ledgers_rechecked=64, original_setup_snapshot_only=4,
                      negative_control_count=24, expected_refusals=24, unexpected_acceptances=0)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        report["input_bytes_unchanged"] = all(path.read_bytes() == raw for path, raw in inputs.items())
        if not report["input_bytes_unchanged"]:
            report["status"] = "FAIL"
        report["inputs"] = {str(path): {"bytes": len(raw), "sha256": sha(raw)} for path, raw in inputs.items()}
        save()
    print(json.dumps({key: report.get(key) for key in ("status", "positive_snapshot_count",
        "negative_control_count", "expected_refusals", "error")}))
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
