"""Pure replay of frozen640 snapshots/ledgers/24 malformed copies.

Historical native4/68 frames are reused as bytes, never represented as a new
native execution. No historical report or expected error is edited.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import struct
import tarfile
import traceback

ARCHIVE = "4517fc8a5d6a88b1645e54df4ae81cc3d8061c32552d317897f0c26fcca79564"
NATIVE = "73f48c3deeae5ee74f3fcdf0195e9ba5da4f3bf0b840d022997461c77b9ae554"
NEGATIVE = "44563bd4857a87e1503d8f30894ed9fd38c44ef7b2ef72057f86f8ca35b69e85"
OBSERVER = "f503df2d1dbce8242591e77211b96a7787201e991db8186c9f1cca6576244409"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--observer", type=Path, default=Path(__file__).with_name("owner640_storage_tombstone.py"))
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "native_executed": False, "Torch_executed": False,
              "historical_native_receipt_reused_not_reclassified": True,
              "archive_sha256": ARCHIVE, "observer_sha256": OBSERVER,
              "positive_snapshots": [], "events": [], "negative_rows": [],
              "whole_pytorch_gate": "HOLD", "full_pytorch_parity": False}
    try:
        assert sha(a.archive.read_bytes()) == ARCHIVE
        original_observer = a.observer.read_bytes(); assert sha(original_observer) == OBSERVER
        spec = importlib.util.spec_from_file_location("tombstone_structural_regression", a.observer)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tarfile.open(a.archive, "r:xz") as archive:
            def read(name, expected):
                m = archive.getmember(name); assert m.isfile()
                b = archive.extractfile(m).read(); assert sha(b) == expected
                return b
            native = json.loads(read("native-controls-02/observed/report.json", NATIVE))
            negative = json.loads(read("negative-controls-02/observed/report.json", NEGATIVE))
            assert native["status"] == negative["status"] == "PASS"
            assert native["observer_sha256"] == negative["observer_sha256"] == "6ea9e8b33164089034a2a71a03f4850e1ec18ed1a09155d7841f700ef311c833"
            assert native["case_count"] == native["passed"] == 4 and native["frames_before_EOF"] == 68
            assert native["failed"] == 0 and len(native["events"]) == 68
            snapshots = {}
            for digest, meta in native["snapshots"].items():
                packed = read("native-controls-02/observed/" + meta["path"], meta["compressed_sha256"])
                raw = gzip.decompress(packed); assert len(raw) == meta["bytes"] and sha(raw) == digest
                s = json.loads(raw); assert module.canonical(s) == raw
                snapshots[digest] = s
                report["positive_snapshots"].append({"sha256": digest, "summary": module.verify_snapshot(s)})
            assert len(snapshots) == 63
            assert set(snapshots) == {e[k] for e in native["events"] for k in ("before", "after")}

            def records(event):
                raw = bytes.fromhex(event["response_hex"])
                assert raw[:8] == b"HTORSP03"
                version, count, status = struct.unpack_from("<3Q", raw, 8)
                assert version == 3 and status == event["actual_status"]
                result, at = [], 32
                for _ in range(count):
                    op, error, value, size = struct.unpack_from("<4Q", raw, at); at += 32
                    assert error == 0 and at + size <= len(raw)
                    result.append({"opcode": op, "value": value, "payload": raw[at:at + size]}); at += size
                assert at == len(raw)
                return result

            for i, event in enumerate(native["events"]):
                before, after = snapshots[event["before"]], snapshots[event["after"]]
                assert module.verify_snapshot(after) == event["snapshot_summary"]
                if event["ledger"]["status"] == "SNAPSHOT_ONLY":
                    assert event["command_words"][0][0] in (58, 72)
                    report["events"].append({"index": i, "status": "SNAPSHOT_ONLY"})
                else:
                    got = module.verify_isolated_transition(before, after, event["command_words"][0], records(event), event["actual_status"])
                    assert got == event["ledger"]
                    report["events"].append({"index": i, "status": "PASS"})
            assert len(negative["controls"]) == 24
            for row in negative["controls"]:
                packed = read("negative-controls-02/observed/" + row["path"], row["compressed_sha256"])
                raw = gzip.decompress(packed)
                assert sha(raw) == row.get("mutated_snapshot_sha256", row.get("mutated_payload_sha256"))
                data = json.loads(raw)
                try:
                    if row["kind"] == "copied-snapshot":
                        module.verify_snapshot(data)
                    else:
                        assert row["kind"] == "copied-transition"
                        e = data["event"]
                        module.verify_isolated_transition(data["before"], data["after"], e["command_words"][0], records(e), e["actual_status"])
                except AssertionError as error:
                    assert error.args[0]["observer_rule"] == row["expected_rule"]
                    report["negative_rows"].append({"name": row["name"], "status": "EXPECTED_REFUSAL", "actual_rule": error.args[0]["observer_rule"]})
                else:
                    raise AssertionError((row["name"], "unexpected acceptance"))
        assert sum(e["status"] == "PASS" for e in report["events"]) == 64
        assert sum(e["status"] == "SNAPSHOT_ONLY" for e in report["events"]) == 4
        assert a.observer.read_bytes() == original_observer and sha(a.archive.read_bytes()) == ARCHIVE
        report.update(status="PASS", snapshot_count=63, isolated_ledgers=64, snapshot_only_setup_events=4,
                      copied_refusals=24, historical_native_report_sha256=NATIVE,
                      historical_negative_report_sha256=NEGATIVE, inputs_unchanged=True)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    (a.output / "executed-test.py").write_bytes(Path(__file__).read_bytes())
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "sha256": sha((a.output / "report.json").read_bytes()), "error": report.get("error")}))
    return 0 if report["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
