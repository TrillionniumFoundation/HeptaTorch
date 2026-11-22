#!/usr/bin/env python3
"""Run a bit-exact standalone abs differential with the official CPU wheel."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import struct
import sys
import tempfile
import types
import zipfile

HERE = Path(__file__).resolve().parent
LOCK_SHA = "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944"
ELF_SHA = "c26ae613ed861dfb13c42e54237a2eac4548ddaecfd528e36c6888ed5dbf355b"
FILES = {
    "abs_word.hepta": "df3c3b8d25cc1c6a27f9d323b68330202cc5ba2e877f74eaea63556ee4078ddd",
    "verify.py": "613895499ef9b78a3b04f2a9d7bf99b640ed78a8405cd947762a5481b04c17d1",
    "torch_oracle.py": "b4255a5ce01fd180fd91a33b6fbc9478dc83a22c5663b8a5c35d87b1836b37e9",
}


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def exact_source(name):
    path = HERE / name
    data = path.read_bytes()
    ensure(hashlib.sha256(data).hexdigest() == FILES[name], "wrong exact #60 source: " + name)
    return path, data


def load_verify():
    path, source = exact_source("verify.py")
    module = types.ModuleType("isolated_abs_verify")
    module.__file__ = str(path)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


verify = load_verify()


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def wheel_custody(wheel, lock, report):
    ensure(wheel.name == lock["wheel"]["filename"], "wrong wheel filename")
    ensure({"implementation": sys.implementation.name, "version": list(sys.version_info[:2])}
           == lock["python"] and {"sys_platform": sys.platform, "machine": platform.machine()}
           == lock["platform"] and sys.byteorder == "little", "wrong oracle platform")
    distribution = importlib.metadata.distribution("torch")
    ensure(distribution.version == lock["version"], "wrong installation version")
    ensure(verify.sha(distribution.read_text("METADATA").encode()) == lock["wheel"]["metadata_sha256"],
           "wrong installed metadata")
    records = [entry for entry in json.loads(report.read_bytes())["install"]
               if entry["metadata"]["name"].lower() == "torch"]
    ensure(len(records) == 1 and records[0]["requested"]
           and records[0]["metadata"]["version"] == lock["version"], "wrong pip report record")
    for artifact in (records[0]["download_info"], json.loads(distribution.read_text("direct_url.json"))):
        ensure(artifact["url"] == lock["wheel"]["url"]
               and artifact["archive_info"]["hashes"]["sha256"] == lock["wheel"]["sha256"],
               "wrong official URL or archive hash")
    root = Path(distribution.locate_file("")).resolve(strict=True)
    record = next(p for p in distribution.files if p.name == "RECORD" and p.parent.name.endswith(".dist-info"))
    files, size = 0, 0
    with wheel.open("rb") as archive_bytes:
        ensure(digest(archive_bytes) == lock["wheel"]["sha256"], "wrong official wheel bytes")
        archive_bytes.seek(0)
        with zipfile.ZipFile(archive_bytes) as archive:
            members = [item for item in archive.infolist() if not item.is_dir() and item.filename != str(record)]
            ensure(len(members) == len({item.filename for item in members}) == 12567, "wrong wheel membership")
            for member in members:
                path = root / member.filename
                ensure(path.resolve(strict=True) == path and path.is_relative_to(root), f"unexpected installed path: {member.filename}")
                with path.open("rb") as installed, archive.open(member) as packed:
                    st = os.fstat(installed.fileno())
                    ensure(stat.S_ISREG(st.st_mode) and st.st_size == member.file_size
                           and digest(installed) == digest(packed), f"installed member differs: {member.filename}")
                files += 1
                size += st.st_size
    return {"official_wheel_sha256": lock["wheel"]["sha256"],
            "wheel_metadata_sha256": lock["wheel"]["metadata_sha256"],
            "torch_version": lock["version"], "torch_git_commit": lock["git_commit"],
            "installed_immutable_members": files, "installed_immutable_bytes": size,
            "pip_install_report_sha256": verify.sha(report.read_bytes()), "cpu_capability": "AVX2"}


def float_class(word):
    magnitude = word & (verify.SIGN - 1)
    if magnitude > verify.INFINITY:
        return "quiet_nan" if magnitude & (1 << 51) else "signaling_nan"
    if magnitude == verify.INFINITY:
        return "infinity"
    if not magnitude:
        return "zero"
    return "subnormal" if magnitude < (1 << 52) else "finite_normal"


def corpora():
    i64, f64 = verify.cases()
    yield "original", i64, f64
    # Move signed qNaN/sNaN witnesses through vector lanes and scalar tails.
    payloads = [1, 2, 3, 1 << 31, (1 << 51) - 1]
    special = [sign | (0x7ff << 52) | quiet | payload
               for sign in (0, verify.SIGN) for quiet in (0, 1 << 51) for payload in payloads]
    mixed = [0, verify.SIGN, 1, verify.SIGN | 1, verify.INFINITY,
             verify.SIGN | verify.INFINITY, 0x3ff0000000000000]
    for offset in range(8):
        yield f"lane_{offset}", [verify.SIGN, verify.MASK64], mixed[:offset] + special + mixed


ORACLE_ENTRY = """
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

oracle, expected_payload, expected_source, *counts = sys.argv[1:]
expected = json.loads(expected_payload)
import torch
dist = importlib.metadata.distribution("torch")
if (str(torch.__version__) != expected["version"]
        or torch.version.git_version != expected["git_commit"]
        or {key: getattr(torch.version, key, "MISSING") for key in expected["backend_versions"]}
            != expected["backend_versions"]
        or Path(torch.__file__).resolve() != Path(dist.locate_file("torch/__init__.py")).resolve()):
    raise ValueError("wrong loaded Torch build or origin")
sys.argv = [oracle, *counts]
source = Path(oracle).read_bytes()
if hashlib.sha256(source).hexdigest() != expected_source:
    raise ValueError("wrong exact oracle source")
exec(compile(source, oracle, "exec"), {"__name__": "__main__", "__file__": oracle})
"""


def compare(image, cache, lock):
    stats, mismatches, transcripts = {}, [], {}
    frames = 0
    env = {"PATH": "", "ATEN_CPU_CAPABILITY": "avx2", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for name, ints, floats in corpora():
        groups = ((2, ints), (4, floats))
        words = ints + floats
        native, _, native_in, native_out, count = verify.native_cases(image, groups)
        inputs = struct.pack(f"<{len(words)}Q", *words)
        argv = [sys.executable, "-I", "-B", "-X", f"pycache_prefix={cache}", "-c", ORACLE_ENTRY,
                HERE / "torch_oracle.py", json.dumps(lock), FILES["torch_oracle.py"],
                str(len(ints)), str(len(floats))]
        reference = verify.succeeded(verify.run(argv, inputs, env, 90), "independent PyTorch")
        ensure(len(reference) == len(inputs), "wrong oracle extent")
        observed = struct.unpack(f"<{len(words)}Q", reference)
        counts = Counter()
        for index, (original, actual, expected) in enumerate(zip(words, native, observed, strict=True)):
            key = ("i64_min" if original == verify.SIGN else "i64_other") if index < len(ints) else (
                ("negative_" if original & verify.SIGN else "positive_") + float_class(original))
            counts[key] += 1
            if actual != expected:
                mismatches.append({"corpus": name, "index": index, "class": key,
                                   "input": f"{original:016x}", "native": f"{actual:016x}",
                                   "torch": f"{expected:016x}"})
        stats[name] = dict(sorted(counts.items()))
        transcripts[name] = {"native_request_sha256": native_in, "native_response_sha256": native_out,
                             "torch_input_sha256": verify.sha(inputs), "torch_response_sha256": verify.sha(reference)}
        frames += count
    return {"cases_by_corpus_and_class": stats, "transcripts": transcripts, "native_frames": frames,
            "comparison_count": sum(sum(corpus.values()) for corpus in stats.values()),
            "bit_mismatches": mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("binary", "wheel", "oracle-lock", "install-report", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    ensure(sys.flags.isolated and sys.dont_write_bytecode, "run the verifier with Python -I -B")
    lock_raw = args.oracle_lock.read_bytes()
    ensure(verify.sha(lock_raw) == LOCK_SHA, "wrong repository oracle lock")
    lock = json.loads(lock_raw)["oracle"]
    for name in FILES:
        exact_source(name)
    image = args.binary.resolve(strict=True)
    ensure(verify.sha(image.read_bytes()) == ELF_SHA, "wrong exact #60 executable")
    oracle = wheel_custody(args.wheel, lock, args.install_report)
    with tempfile.TemporaryDirectory(prefix="abs-clean-cache-", dir=args.output.resolve().parent) as cache:
        result = compare(image, cache, lock)
    ensure(verify.sha(image.read_bytes()) == ELF_SHA, "native executable changed")
    receipt = {"status": "FAIL_BITWISE" if result["bit_mismatches"] else "PASS_SCOPED_STRICT_TORCH",
               "scope": "standalone I64/F64 raw-word abs only; no public tensor dispatch",
               "native_image_sha256": ELF_SHA, "pinned_sources_sha256": FILES,
               "qualifier_sha256": verify.sha(Path(__file__).read_bytes()),
               "oracle_lock_sha256": LOCK_SHA, "oracle": oracle, **result}
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(receipt["status"], receipt["comparison_count"], receipt["native_frames"])
    return int(bool(result["bit_mismatches"]))


if __name__ == "__main__":
    raise SystemExit(main())
