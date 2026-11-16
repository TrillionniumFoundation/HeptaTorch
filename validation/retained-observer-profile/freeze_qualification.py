"""Freeze the already-executed 446 observer controls without rerunning them."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import shutil
import tarfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def bound(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": sha(raw)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    profile, results, base, output = [x.resolve() for x in (a.profile, a.results, a.base, a.output)]
    output.mkdir(parents=True, exist_ok=False)
    proof = json.loads((profile / "inverse-proof.json").read_bytes())
    assert proof["status"] == "EXACT_REVERSIBLE_PROFILE_ONLY"
    for name, row in proof["files"].items():
        new = (profile / "tests/private_node_accounting" / name).read_bytes()
        old = (base / name).read_bytes()
        assert sha(new) == row["after_sha256"] and sha(old) == row["before_sha256"]
        inverse = new.decode()
        for pair in reversed(row["replacements"]):
            assert inverse.count(pair["after"]) == 1
            inverse = inverse.replace(pair["after"], pair["before"], 1)
        assert inverse.encode() == old
    native_file = results / "native-controls446-01/report.json"
    negative_file = results / "copied-controls446-01/report.json"
    native, negative = [json.loads(x.read_bytes()) for x in (native_file, negative_file)]
    assert native["status"] == "PASS" and native["cases"] == len(native["rows"]) == 3
    expected = [
        ("private-gradient-reset-keeps-real-high-slot-and-rollback", 6),
        ("same-frame-nongrad-create-reset-keeps-incarnation-counter", 3),
        ("original-Linear-and-MSE-count3-plus-three-private-gradients", 4),
    ]
    assert [(r["case"], r["frames"]) for r in native["rows"]] == expected
    assert native["frames"] == sum(r["frames"] for r in native["rows"]) == 13
    for row in native["rows"]:
        assert row["status"] == "PASS"
        item = row["private_node_observer"]
        assert item["status"] == "PASS"
        assert item["eof_exit_assertions_executed"] is True
        assert item["post_close_parameter_version_assertions_executed"] is True
    assert native["profile"]["name"] == proof["profile"]
    assert native["profile"]["source_sha256"] == proof["source_sha256"]
    assert native["profile"]["native_sha256"] == proof["native_sha256"]
    assert native["native_image_unchanged"] is True
    assert native["original_numeric_oracle_executed"] is False
    assert native["host_numeric_runtime_used"] is False
    assert negative["status"] == "PASS_OBSERVER_CONTROLS_ONLY"
    assert negative["positive_cases"] == len(negative["positive_rows"]) == 4
    assert negative["negative_cases"] == len(negative["negative_rows"]) == 18
    assert negative["failures"] == 0
    assert negative["expected_profile"] == proof["profile"]
    assert negative["observer_sha256"] == proof["files"]["owner_accounting.py"]["after_sha256"]
    assert negative["controls_sha256"] == proof["files"]["check_owner_snapshots.py"]["after_sha256"]
    assert {r["case"] for r in negative["positive_rows"]} == {
        "materialization", "compound-construction", "reset-with-private-gradient", "failed-frame-rollback"}
    assert all(r["status"] == "PASS" for r in negative["positive_rows"])
    assert all(r["status"] == "PASS_EXPECTED_REJECTION" and r["actual_rule"] == r["expected_rule"]
               for r in negative["negative_rows"])
    assert negative["original_journals_unchanged"] is negative["observer_source_unchanged"] is True
    assert negative["native_executed"] is negative["Torch_executed"] is False
    assert negative["native_memory_read"] is negative["native_memory_written"] is False
    snapshots = live_tensor_rows = 0
    for snapshot_file in sorted((results / "native-controls446-01/owners").glob("*/snapshots.jsonl.gz")):
        with gzip.open(snapshot_file, "rt") as f:
            for line in f:
                item = json.loads(line)
                snapshot = item["snapshot"]
                assert sha(canonical(snapshot)) == item["sha256"]
                assert snapshot["profile"] == proof["profile"]
                for row in snapshot["tables"].get("1", []):
                    if row[1] == 1:
                        assert row[7] == 0, "the original observer scope still forbids retained flags"
                        live_tensor_rows += 1
                snapshots += 1
    restore = json.loads((results / "restore-report.json").read_bytes())
    assert restore["status"] == "EXACT_PUBLISHED_BYTES_RESTORED_NOT_NEW_BUILD"
    assert restore["native_rebuilt"] is False and restore["profile"] == "446"
    assert len(restore["members"]) == 33
    assert restore["source_sha256"] == proof["source_sha256"] and restore["native_sha256"] == proof["native_sha256"]
    for item in restore["members"]:
        restored = results / "native446" / item["path"]
        raw = restored.read_bytes()
        assert len(raw) == item["bytes"] and sha(raw) == item["sha256"]
    # Archive raw executions exactly; absolute paths in their reports are never rewritten.
    archive = output / "raw-controls.tar.xz"
    raw_paths = [results / "restore-report.json"]
    for name in ("native-controls446-01", "copied-controls446-01"):
        raw_paths.extend(p for p in sorted((results / name).rglob("*")) if p.is_file())
    with tarfile.open(archive, "x:xz", preset=6) as t:
        for path in raw_paths:
            t.add(path, arcname=str(path.relative_to(results)), recursive=False)
    with tarfile.open(archive, "r:xz") as t:
        members = t.getmembers()
        assert len(members) == len(raw_paths)
        for member in members:
            assert member.isfile()
            assert t.extractfile(member).read() == (results / member.name).read_bytes()
    tests = output / "tests/private_node_accounting"
    tests.mkdir(parents=True)
    for name in proof["files"]:
        shutil.copyfile(profile / "tests/private_node_accounting" / name, tests / name)
    baseline = output / "baseline-af33"
    baseline.mkdir()
    for name, row in proof["files"].items():
        if row["changed"]:
            shutil.copyfile(base / name, baseline / name)
    shutil.copyfile(profile / "inverse-proof.json", output / "inverse-proof.json")
    for name in ("build_profile.py", "freeze_qualification.py"):
        shutil.copyfile(Path(__file__).resolve().parent / name, output / name)
    for original, name in ((native_file, "native-controls-report.json"), (negative_file, "copied-controls-report.json"),
                           (results / "restore-report.json", "restore-report.json")):
        shutil.copyfile(original, output / name)
    summary = {
        "schema": "heptatorch.retained446-observer-qualification.v1", "status": "PASS_NATIVE_AND_PURE_OBSERVER_CONTROLS",
        "profile": native["profile"], "inverse_proof": bound(output / "inverse-proof.json"),
        "native_controls": {"cases": 3, "frames": 13, "report": bound(output / "native-controls-report.json")},
        "copied_snapshot_controls": {"positive": 4, "negative": 18, "report": bound(output / "copied-controls-report.json")},
        "raw_archive": bound(archive), "raw_archive_members": len(raw_paths), "all_archived_members_verified_byte_exact": True,
        "committed_snapshots_checked": snapshots, "live_T_rows_with_exact_zero_flags": live_tensor_rows,
        "native_restoration": {"report": bound(output / "restore-report.json"), "members": 33, "new_build": False},
        "frozen_live32": "NOT_RUN_FOR_446", "frozen35": "NOT_RUN_FOR_446",
        "source_or_numeric_oracle_modifications": False, "workflow_modifications": False,
        "private_node_accounting_algorithm_modified": False, "retained_flag_observer_scope_widened": False,
        "native_memory_written": False, "full_rewrite_complete": False,
    }
    (output / "qualification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "raw_members": len(raw_paths), "snapshots": snapshots,
                      "zero_flag_T_observations": live_tensor_rows, "output": str(output)}))


if __name__ == "__main__":
    main()
