"""Source-pinned native controls for the new 640-byte ownership observer.

No Torch runs here. Values select ownership states; this does not replace any
fixed numerical oracle. Every snapshot comes from the real native child.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import struct
import traceback

OLD_OBSERVER = "281f1ea437046a9cd884de2c8da34f38649a3b6482ade23123ac4aa992faed64"
WIRE = "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9"
CLIENT = "dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "observer", "protocol", "old-observer", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-observer-sha256", required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    inputs, sessions, events = {}, [], []
    report = {"schema": "heptatorch.dtype640-native-accounting-controls.v1", "status": "RUNNING",
              "native_executed": False, "Torch_executed": False, "observer_memory_writes": False,
              "whole_pytorch_gate": "HOLD", "pytorch_api_denominator": 2590,
              "full_pytorch_parity": False, "cases": [], "snapshots": {}, "events": events,
              "allocation_ledgers_scope": "Explicit isolated commands only; backward/step setup is labelled snapshot-only, never gap-accounted."}

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def bind(path, expected=None):
        path = path.resolve(); raw = path.read_bytes()
        assert expected is None or sha(raw) == expected, (str(path), sha(raw), expected)
        inputs[path] = raw
        return raw

    try:
        bind(Path(__file__))
        source = bind(a.observer, a.expected_observer_sha256)
        old_source = bind(a.old_observer, OLD_OBSERVER)
        bind(a.protocol, WIRE); bind(a.protocol.with_name("owner_client.py"), CLIENT)
        (out / "observer-under-test.py").write_bytes(source)
        (out / "old-observer-unchanged.py").write_bytes(old_source)
        (out / "executed-test.py").write_bytes(Path(__file__).read_bytes())
        observer = load("native_owner640_under_test", a.observer)
        wire = load("native_owner640_wire", a.protocol)
        build = json.loads(bind(a.build / "build.json"))
        runtime = a.build / "runtime.hepta"
        binary = a.build / "tensor-dtype-dropout-session"
        bind(runtime, observer.SOURCE); bind(binary, observer.IMAGE)
        profile = observer.pin_profile(binary, runtime)
        assert build["status"] == "PASS" and build["source_sha256"] == observer.SOURCE
        assert build["native_sha256"] == observer.IMAGE and build["descriptor_bytes"] == 640
        assert len(build["units"]) == 11
        for unit in build["units"]:
            assert unit["compiler_exit"] == 0
            bind(a.build / (unit["name"] + ".hepta"), unit["source_sha256"])
            bind(a.build / (unit["name"] + ".o"), unit["object_sha256"])
        report.update(profile=profile, observer_sha256=sha(source), old_observer_sha256=sha(old_source),
                      source_sha256=observer.SOURCE, native_sha256=observer.IMAGE)
        saved = set()

        def store(snapshot):
            raw = observer.canonical(snapshot); digest = sha(raw)
            if digest not in saved:
                directory = out / "snapshots"; directory.mkdir(exist_ok=True)
                path = directory / (digest + ".json.gz")
                with path.open("wb") as f:
                    with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as z:
                        z.write(raw)
                report["snapshots"][digest] = {"path": str(path.relative_to(out)), "bytes": len(raw),
                                              "compressed_sha256": sha(path.read_bytes())}
                saved.add(digest)
            return digest

        def begin(name):
            row = {"name": name, "status": "RUNNING", "event_start": len(events)}
            report["cases"].append(row)
            session = wire.Session(binary, directory=out / "transport" / name)
            sessions.append((session, row))
            report["native_executed"] = True
            session.reader = observer.OwnerReader(session)
            return session, row

        def run(session, commands, expected=0, ledger=True, label=""):
            before = session.reader.snapshot(); before_sha = store(before)
            body = wire.packet(commands)
            raw = session.exchange(body)
            status, rows = wire.parse(raw)
            after = session.reader.snapshot(); after_sha = store(after)
            event = {"case": sessions[-1][1]["name"], "label": label, "before": before_sha,
                     "after": after_sha, "command_words": [list(struct.unpack_from("<8Q", body, 32 + i * 64))
                        for i in range(len(commands))], "request_hex": body.hex(), "response_hex": raw.hex(),
                     "expected_status": expected, "actual_status": status, "ledger": None}
            events.append(event)
            assert status == expected and (not expected or not rows), (label, status, expected)
            if expected:
                event["ledger"] = observer.verify_isolated_transition(before, after, event["command_words"][0], rows, status)
            elif ledger:
                assert len(commands) == 1
                event["ledger"] = observer.verify_isolated_transition(before, after, event["command_words"][0], rows)
            else:
                event["ledger"] = {"status": "SNAPSHOT_ONLY", "reason": "Full structural proof executed; no per-command allocation claim for this setup operation."}
            event["snapshot_summary"] = observer.verify_snapshot(after)
            session.last_snapshot = after
            save()
            return rows

        def one(session, code, *args, **kw):
            return run(session, [(code, *args)], **kw)[0]

        def finish(session, row):
            rec = one(session, 54, label="actual public stats vs decoded owner")
            assert list(struct.unpack("<12Q", rec["payload"])) == observer.verify_snapshot(session.last_snapshot)["stats"]
            assert one(session, 53, label="production audit")['value'] == 1
            session.finish()
            # The frozen OwnerSession.finish itself checks empty stderr/stdout.
            assert session.exit_code == 0
            row.update(status="PASS", frame_count=session.frames, event_count=len(events) - row["event_start"],
                       eof_exit_code=session.exit_code)
            save()

        # A. Real 1-byte mask, its Saved Tensor role, public Saved layout and
        # Weak->Hold dtype. Mutation makes Saved stale, which is a valid state.
        s, row = begin("bool-mask-Saved-Weak-Hold-and-ordinary-rollback")
        x = run(s, [wire.full((2, 3), .5, 1)])[0]["value"]
        gen = one(s, 60, 1729)["value"]
        nd = one(s, 99, x, gen, 0x3fe0000000000000, 1, label="nonempty two actual Tensor allocations")
        mask, = struct.unpack("<Q", nd["payload"])
        saved_mask = one(s, 46, mask)["value"]
        weak = one(s, 42, mask)["value"]
        hold = one(s, 43, weak)["value"]
        one(s, 102, mask, label="bool Version mutation retains stale Saved role")
        run(s, [(102, mask), (999,)], expected=4, label="real write prefix then ordinary full-owner rollback")
        row["negative_control_snapshot"] = store(s.last_snapshot)
        one(s, 45, hold); one(s, 48, saved_mask); one(s, 44, weak)
        finish(s, row)

        # B. A real old child retains the first Accumulator across detach_.
        # Reusing that cache consumes one history token without a new node.
        s, row = begin("captured-Accumulator-refresh-and-real-unrooted-cycle")
        x = run(s, [wire.full((2, 0, 3), .25, 1)])[0]["value"]
        scalar = run(s, [wire.full((), 2)])[0]["value"]
        child = one(s, 2, 3, x, scalar)["value"]
        gen = one(s, 60, 19)["value"]
        first = one(s, 99, x, gen, 0x3fe0000000000000, 1, label="first empty capture creates a real Accumulator")
        first_mask, = struct.unpack("<Q", first["payload"])
        one(s, 101, x, label="same Tensor detach captures and releases previous graph")
        one(s, 90, x, 1)
        second = one(s, 99, x, gen, 0x3fe0000000000000, 1, label="reuse old Accumulator: one node plus one refresh")
        second_mask, = struct.unpack("<Q", second["payload"])
        assert events[-1]["ledger"]["created_nodes"] == events[-1]["ledger"]["history_refreshes"] == 1
        row["refresh_event"] = len(events) - 1
        row["negative_control_snapshot"] = store(s.last_snapshot)
        weak = one(s, 42, x)["value"]
        for t in (x, first_mask, second_mask, child, scalar):
            one(s, 41, t)
        summary = observer.verify_snapshot(s.last_snapshot)
        assert summary["unrooted_strong_cycles"], "The real native capture must demonstrate the retained cycle."
        row["cycle_snapshot"] = store(s.last_snapshot)
        holder = one(s, 43, weak)["value"]
        assert holder != 0
        one(s, 45, holder); one(s, 44, weak)
        finish(s, row)

        # C. Prior nonleaf retain hook is weak. Its old graph survives through
        # an older child, while the current Tensor's Storage can really expire.
        s, row = begin("weak-captured-retain-hook-and-public-gradient")
        x = run(s, [wire.full((2, 0, 3), .25, 1)])[0]["value"]
        scalar = run(s, [wire.full((), 2)])[0]["value"]
        q = one(s, 2, 3, x, scalar)["value"]
        one(s, 93, q)
        old_child = one(s, 2, 3, q, scalar)["value"]
        gen = one(s, 60, 29)["value"]
        result = one(s, 99, q, gen, 0x3fe0000000000000, 1)
        mask, = struct.unpack("<Q", result["payload"])
        seed = run(s, [wire.full((2, 0, 3), 1)])[0]["value"]
        assert one(s, 58, old_child, seed, 1, ledger=False, label="retained prehook setup; full ownership proof")['value'] == 0
        gradient = one(s, 91, q)["value"]
        assert gradient != 0
        row["gradient_snapshot"] = store(s.last_snapshot)
        weak = one(s, 42, q)["value"]
        one(s, 41, q); one(s, 41, mask)
        row["weak_stale_snapshot"] = store(s.last_snapshot)
        assert observer.verify_snapshot(s.last_snapshot)["weak_retained_hooks"]
        run(s, [(43, weak)], expected=6, label="weak captured self does not keep current Tensor Storage")
        assert one(s, 58, old_child, seed, 1, ledger=False, label="old graph executes with expired weak target")['value'] == 0
        one(s, 44, weak)
        finish(s, row)

        # D. Decode the actual Optimizer blobs, including duplicate parameter
        # occurrences and one state-key root. Never trust node312 by itself.
        s, row = begin("SGD-Adam-private-Tensor-roots-and-public-state-gradient-alias")

        def optimizer(params, algorithm):
            cfg = ([wire.bits(v) for v in (.125, .5, 0, 0)] + [0, 0] if algorithm == 0 else
                   [wire.bits(v) for v in (.125, 0, 0, 1, 0)] + [0, 0, 0])
            groups = [1, len(params), 0] + [0] * len(cfg) + list(params)
            return one(s, 71, algorithm, wire.Payload(struct.pack("<" + "Q" * len(cfg), *cfg)),
                       wire.Payload(struct.pack("<" + "Q" * len(groups), *groups)))["value"]

        x = run(s, [wire.full((2, 0, 3), .25, 1)])[0]["value"]
        g = run(s, [wire.full((2, 0, 3), 1)])[0]["value"]
        one(s, 92, x, g)
        sgd = optimizer([x, x], 0)
        one(s, 72, sgd, ledger=False, label="SGD state initialization; exact blob/reference accounting")
        gen = one(s, 60, 31)["value"]
        nd = one(s, 99, x, gen, 0x3fe0000000000000, 1)
        row["Optimizer_history_snapshot"] = store(s.last_snapshot)
        assert one(s, 53)["value"] == 1
        one(s, 101, x)
        aparam = run(s, [wire.full((2,), .25, 1)])[0]["value"]
        agrad = run(s, [wire.full((2,), 1)])[0]["value"]
        one(s, 92, aparam, agrad)
        adam = optimizer([aparam], 1)
        one(s, 72, adam, ledger=False, label="Adam three actual state Tensors; exact private references")
        avg = one(s, 77, adam, 0, 0, 2)["value"]
        one(s, 92, aparam, avg)
        row["negative_control_snapshot"] = store(s.last_snapshot)
        assert observer.verify_snapshot(s.last_snapshot)["Optimizer_state_nodes"]
        assert observer.verify_snapshot(s.last_snapshot)["gradient_nodes"]
        finish(s, row)
        report.update(status="PASS", case_count=4, passed=4, failed=0,
                      frames_before_EOF=sum(s.frames for s, _ in sessions),
                      full_structural_snapshots_checked=len(events) * 2,
                      isolated_transitions_checked=sum(e["ledger"]["status"] == "PASS" for e in events),
                      explicitly_snapshot_only_events=sum(e["ledger"]["status"] == "SNAPSHOT_ONLY" for e in events))
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        for session, row in sessions:
            if row["status"] != "PASS":
                row["status"] = "FAIL"
                try:
                    if session.output is not None:
                        session.finish()
                    row["cleanup_exit_code"] = session.exit_code
                except BaseException as error:
                    row["cleanup_error"] = repr(error)
        report["source_custody_unchanged"] = all(p.read_bytes() == raw for p, raw in inputs.items())
        if not report["source_custody_unchanged"]:
            report["status"] = "FAIL"
        report["inputs"] = {str(p): {"bytes": len(raw), "sha256": sha(raw)} for p, raw in inputs.items()}
        save()
    print(json.dumps({k: report.get(k) for k in ("status", "case_count", "frames_before_EOF",
        "isolated_transitions_checked", "explicitly_snapshot_only_events", "error")}))
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
