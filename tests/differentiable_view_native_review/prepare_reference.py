#!/usr/bin/env python3
"""Restore and independently admit the immutable actual PyTorch reference14."""
import argparse
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
import traceback

PIN = {
    "complete.tar.xz": (262788, "55c9718395e6007de32a43e93c60e90500ec1c2aea9b7e93dd831b25cefac345"),
    "manifest.json": (9989, "4542524a91d21f897fbb53b80b15f1f7f36c740a94b80d2b148729f76c8520c3"),
    "verification.json": (None, "bcefc9fa737482a8819bf12cd747321ce3cd21242c497af9b2d8655ed301b02c"),
    "outer-custody.json": (44599, "376909205644374ce7b3ef9ee5c914e127a1ac4aec5e7f6998416bd0c12faacb"),
}
READER = "0424b6b4d35c0354bbc92bf0b69f93ca8192f5d76881a7a8b4087ddb4165da2e"
ACTUAL = "2270d5ba925bcf91c56db94fac663695e69079c74edbf80aff8fa086e7c81665"
PROBE = "1ca74384675f4f5bbe1b477be827e1d405bf3ca04bc3eeaa22e4628dfe0cc1ac"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stable_read(path):
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
        return raw
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path(__file__).resolve().parents[2] /
                        "validation/differentiable-view-reference14")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet, output = args.packet.resolve(), args.output.resolve()
    assert not output.exists(), "output must be new; failed runs are retained"
    assert output != packet and packet not in output.parents
    output.mkdir(parents=True, mode=0o700)
    report = {
        "schema": "heptatorch.differentiable-view-reference14-portable-admission.v1",
        "status": "RUNNING", "packet": str(packet),
        "source_reader_sha256": READER, "reference_sha256": ACTUAL,
        "probe_sha256": PROBE, "new_native_executions": 0, "new_torch_executions": 0,
        "native_qualified": False, "closed_schema_count": 0,
        "pytorch_api_denominator": 2590, "full_pytorch_rewrite_complete": False,
        "member_bytes_and_modes_verified": False, "admission_started": False,
        "admission_completed": False,
    }
    inputs = {}
    try:
        for name, (size, digest) in PIN.items():
            raw = stable_read(packet / name)
            assert (size is None or len(raw) == size) and sha(raw) == digest, name
            inputs[name] = raw
        manifest = json.loads(inputs["manifest.json"])
        assert manifest["file_count"] == 52 and manifest["expanded_bytes"] == 7384090
        archive = manifest["archive"]
        assert archive["bytes"] == 262788 and archive["sha256"] == PIN["complete.tar.xz"][1]
        decoder = lzma.LZMADecompressor()
        raw_tar = decoder.decompress(inputs["complete.tar.xz"])
        assert decoder.eof and not decoder.unused_data
        assert len(raw_tar) == archive["tar_bytes"] == 7424000
        assert sha(raw_tar) == archive["tar_sha256"] == "870eca7389430b80b576f0e65f9691eb8f33c8fa1de1eb17c328c857c7d7aa1d"
        extracted = output / "payload"
        extracted.mkdir(mode=0o700)
        members = {}
        total = 0
        with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as tf:
            for member in tf.getmembers():
                name = member.name
                rel = PurePosixPath(name)
                assert (member.isfile() and not rel.is_absolute() and
                        ".." not in rel.parts and name == str(rel) and
                        name not in members and name in manifest["files"]), name
                item = manifest["files"][name]
                assert not member.mode & ~0o777, name
                assert member.mode == item["mode"] == 0o600, name
                raw = tf.extractfile(member).read()
                assert len(raw) == member.size == item["bytes"] and sha(raw) == item["sha256"], name
                dest = extracted.joinpath(*rel.parts)
                dest.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, item["mode"])
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.fchmod(stream.fileno(), item["mode"])
                assert stable_read(dest) == raw and stat.S_IMODE(dest.stat().st_mode) == item["mode"], name
                members[name] = {"bytes": len(raw), "sha256": sha(raw), "mode": item["mode"]}
                total += len(raw)
        assert set(members) == set(manifest["files"]) and len(members) == 52 and total == 7384090
        report.update(member_bytes_and_modes_verified=True, members=members,
                      member_count=len(members), expanded_bytes=total,
                      single_xz_eof=True, complete_tar_sha256=sha(raw_tar))
        reader = extracted / "admit_reference_remote_v1.py"
        assert sha(stable_read(reader)) == READER
        admission_path = output / "admission.json"
        argv = [sys.executable, "-I", "-S", "-B", str(reader),
                "--root", str(extracted / "original"), "--output", str(admission_path)]
        report.update(admission_argv=argv, admission_started=True)
        with (output / "admission.stdout").open("xb") as stdout, (output / "admission.stderr").open("xb") as stderr:
            child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env={"PATH": ""})
            report["admission_pid"] = child.pid
            try:
                result = child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                raise
        report.update(admission_completed=True, admission_exit_code=result)
        for name in ("admission.stdout", "admission.stderr", "admission.json"):
            raw = stable_read(output / name)
            report[name] = {"bytes": len(raw), "sha256": sha(raw)}
        assert result == 0 and report["admission.stderr"]["bytes"] == 0
        admitted = json.loads(stable_read(admission_path))
        assert admitted["status"] == "PASS" and admitted["reader_sha256"] == READER
        assert admitted["checks_passed"] == 41095
        assert admitted["observations_verified"] == {
            "snapshots": 432, "tensor_records": 2539, "binary32_words": 14126,
            "physical_bytes": 57144}
        assert admitted["new_native_executions"] == admitted["new_torch_executions"] == 0
        assert admitted["native_qualification"] is False and admitted["primary_error"] is None
        assert sha(stable_read(extracted / "original/actual.json")) == ACTUAL
        assert sha(stable_read(extracted / "original/inputs/differentiable-view14-probe-1ca74384.py")) == PROBE
        assert all(stable_read(packet / name) == raw for name, raw in inputs.items())
        for name, item in members.items():
            dest = extracted / name
            raw = stable_read(dest)
            assert len(raw) == item["bytes"] and sha(raw) == item["sha256"]
            assert stat.S_IMODE(dest.stat().st_mode) == item["mode"]
        report.update(status="PASS", all_inputs_unchanged=True,
                      restored_payload_unchanged=True, checks_passed=admitted["checks_passed"],
                      observations_verified=admitted["observations_verified"])
    except BaseException:
        report.update(status="FAIL", error=traceback.format_exc())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "new_native_executions",
                     "new_torch_executions", "member_bytes_and_modes_verified")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
