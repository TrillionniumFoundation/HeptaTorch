#!/usr/bin/env python3
"""Restore the fixed input4 observation packet without executing Torch or HIA."""
import argparse
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import traceback

PIN = {
    "complete.tar.xz": (2860876, "2b8225a7d7a2b0f56fe2caa84911da121f3fa5811e734edfcd990405b4f9d18a"),
    "manifest.json": (31399, "5ea2b8d14384b169985bfa0075144daede07bca23039900b6d846d6f2a672b1c"),
    "verification.json": (None, "43ea5201eab209731946221f54e80308e6822e22f4f02399b8cc67ea74218784"),
    "outer-custody.json": (None, "90f8c8c4af3314330a02e7d7a3c47c842cbf4fed305237b7be42f40fd4ebb891"),
}
ACTUAL = "f4abe6a750ecc791b3a7bc7aa507b4c77eb0a2fcccf45b0c6a828db0e01490d3"
PROBE = "cf569335f48624c1edf2f97a649200f2e792ebcef5b6f1df6bac73293ce02efc"
EXECUTION = "78ea18d6ab862d5d289e45b7b12e590b30ac5c6c816056477b23879b5e6aa6bb"
ADMISSION = "ffe90f023a412c7ac7afc7eab1d2dc8ca2c11a1f1f83ec6103c736d37906aca4"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        assert stat.S_ISREG(before.st_mode) and before.st_nlink == 1, str(path)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read()
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns",
                  "st_ctime_ns", "st_nlink")
        assert all(getattr(before, k) == getattr(after, k) for k in fields), str(path)
        assert len(raw) == before.st_size, str(path)
        return raw, {k: getattr(before, k) for k in fields}
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, default=Path(__file__).resolve().parents[2] /
                    "validation/differentiable-view-input4-reference")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    packet, output = args.packet.resolve(), args.output.resolve()
    assert not output.exists(), "output must be new; preserve failed attempts"
    assert output != packet and packet not in output.parents
    output.mkdir(parents=True, mode=0o700)
    report = {
        "schema": "heptatorch.differentiable-view-input4-portable-restore.v1",
        "status": "RUNNING", "packet": str(packet), "reference_sha256": ACTUAL,
        "probe_sha256": PROBE, "new_native_executions": 0, "new_torch_executions": 0,
        "reran_original_independent_admission": False,
        "native_qualified": False, "closed_schema_count": 0,
        "pytorch_api_denominator": 2590, "full_pytorch_rewrite_complete": False,
        "member_bytes_and_modes_verified": False,
    }
    inputs = {}
    members = {}
    try:
        own = Path(__file__).resolve()
        inputs[str(own)] = read(own)
        raw_inputs = {}
        for name, (size, digest) in PIN.items():
            path = packet / name
            raw, identity = read(path)
            assert (size is None or len(raw) == size) and sha(raw) == digest, name
            raw_inputs[name] = raw
            inputs[str(path)] = (raw, identity)
        manifest = json.loads(raw_inputs["manifest.json"])
        assert manifest["schema"] == "heptatorch.input4-original-reference-packet.v1"
        assert manifest["status"] == "OBSERVED_ONLY" and manifest["native_executed"] is False
        assert manifest["full_pytorch_rewrite_complete"] is False
        assert manifest["closed_schema_count"] == 0 and manifest["pytorch_api_denominator"] == 2590
        assert manifest["member_count"] == 42 and manifest["expanded_bytes"] == 16326386
        assert manifest["archive"]["bytes"] == len(raw_inputs["complete.tar.xz"])
        assert manifest["archive"]["sha256"] == PIN["complete.tar.xz"][1]
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
        raw_tar = decoder.decompress(raw_inputs["complete.tar.xz"])
        assert decoder.eof and not decoder.unused_data
        assert len(raw_tar) == manifest["decompressed_tar"]["bytes"] == 16363520
        assert sha(raw_tar) == manifest["decompressed_tar"]["sha256"] == "3b44670abee63201b5b90aaf5c88033a628e38a965d6d37705a68b930c1fe616"
        payload = output / "payload"
        payload.mkdir(mode=0o700)
        with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as tf:
            for member in tf:
                name = member.name
                rel = PurePosixPath(name)
                assert (member.isfile() and not rel.is_absolute() and rel.parts and
                        ".." not in rel.parts and name == str(rel) and
                        name not in members and name in manifest["files"]), name
                assert not member.pax_headers and not member.mode & ~0o777, name
                item = manifest["files"][name]
                raw = tf.extractfile(member).read()
                assert len(raw) == member.size == item["bytes"], name
                assert sha(raw) == item["sha256"] and member.mode == item["mode"], name
                dest = payload.joinpath(*rel.parts)
                dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, item["mode"])
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.fchmod(stream.fileno(), item["mode"])
                restored, identity = read(dest)
                assert restored == raw and stat.S_IMODE(identity["st_mode"]) == item["mode"], name
                members[name] = item
        assert set(members) == set(manifest["files"]) and len(members) == 42
        assert sum(v["bytes"] for v in members.values()) == 16326386
        for name, digest in {
            "actual/actual.json": ACTUAL,
            "source/pytorch_differentiable_view_input_boundaries_probe.py": PROBE,
            "execution/execution.json": EXECUTION,
            "admission/report.json": ADMISSION,
        }.items():
            assert members[name]["sha256"] == digest, name
        execution = json.loads(read(payload / "execution/execution.json")[0])
        admission = json.loads(read(payload / "admission/report.json")[0])
        actual = json.loads(read(payload / "actual/actual.json")[0])
        prepared = json.loads(read(payload / "source/prepare.json")[0])
        assert execution["status"] == "OBSERVED" and execution["exit_code"] == 0
        assert execution["child_started"] is True and execution["child_completed"] is True
        assert execution["inputs_before"] == execution["inputs_after"]
        bindings = manifest["actual_input_map"]
        assert set(bindings) == set(execution["inputs_before"]) and len(bindings) == 25
        assert len({row["archive_path"] for row in bindings.values()}) == 25
        for name, binding in bindings.items():
            assert {k: v for k, v in binding.items() if k != "archive_path"} == execution["inputs_before"][name]
            assert members[binding["archive_path"]] == {k: binding[k] for k in ("bytes", "mode", "sha256")}
        assert actual["status"] == "OBSERVED" and actual["cases"] == len(actual["rows"]) == 4
        assert actual["probe_errors"] == [] and actual["probe_error_count"] == 0
        assert actual["case_plan"] == prepared["case_plan"]
        assert actual["operation_totals"] == {"attempted": 103, "succeeded": 103, "raised": 0}
        assert actual["native_executed"] is False and actual["native_qualified"] is False
        assert actual["source_files_unchanged"] is True and actual["torch_executed"] is True
        assert actual["oracle_provenance"]["status"] == "PASS"
        checks = actual["oracle_provenance"]["checks"]
        assert len(checks) == 11 and all(v is True for v in checks.values())
        assert admission["status"] == "OBSERVATION_BYTE_INTEGRITY_VERIFIED"
        assert admission["primary_error"] is None
        assert admission["native_executions"] == admission["torch_executions"] == 0
        for key, expected in {"checks": 64059, "tensor_records": 2019,
                              "snapshots": 210, "logical_words": 10120,
                              "events": 103, "backward_events": 7}.items():
            assert admission["counts"][key] == expected, key
        for path, original in inputs.items():
            assert read(Path(path)) == original, path
        for name, item in members.items():
            raw, identity = read(payload / name)
            assert len(raw) == item["bytes"] and sha(raw) == item["sha256"], name
            assert stat.S_IMODE(identity["st_mode"]) == item["mode"], name
        report.update(
            status="PASS", member_bytes_and_modes_verified=True, all_inputs_unchanged=True,
            restored_payload_unchanged=True, single_xz_eof=True,
            complete_tar_sha256=sha(raw_tar), member_count=42, expanded_bytes=16326386,
            members=members, original_actual_inputs_verified=25,
            original_admission_sha256=ADMISSION, original_admission_counts=admission["counts"],
            actual_operation_totals=actual["operation_totals"],
            reference_path=str(payload / "actual/actual.json"),
            probe_path=str(payload / "source/pytorch_differentiable_view_input_boundaries_probe.py"),
            source_sha256=sha(inputs[str(own)][0]),
            delivery_note="Outer files are byte-pinned; original inner POSIX modes are restored exactly.")
    except BaseException:
        report.update(status="FAIL", error=traceback.format_exc())
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "new_native_executions",
                     "new_torch_executions", "member_bytes_and_modes_verified")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
