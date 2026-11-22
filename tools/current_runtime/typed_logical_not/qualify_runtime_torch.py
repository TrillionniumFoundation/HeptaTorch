"""Strict, opt-in differential qualification of the exact opcode-127 ELF."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import struct
import sys

from test_component import WIDTHS, known_words
from test_runtime_candidate import NATIVE_SHA256, digest, factory, load_wire, require

LOCK_SHA256 = "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944"
DTYPES = {0: "float32", 1: "bool", 2: "int64", 4: "float64", 5: "float16", 6: "bfloat16"}


def load_oracle(lock_path, install_path):
    require(digest(lock_path) == LOCK_SHA256, "wrong exact repository oracle lock")
    expected = json.loads(lock_path.read_bytes())["oracle"]
    require({"implementation": sys.implementation.name, "version": list(sys.version_info[:2])}
            == expected["python"], "wrong Python")
    require({"sys_platform": sys.platform, "machine": platform.machine()} == expected["platform"]
            and sys.byteorder == "little", "wrong fixed-wheel platform")

    distribution = metadata.distribution(expected["distribution"])
    require(distribution.version == expected["version"], "wrong installed distribution version")
    wheel = expected["wheel"]
    require(sha256(distribution.read_text("METADATA").encode()).hexdigest() == wheel["metadata_sha256"],
            "wrong installed METADATA bytes")
    direct = json.loads(distribution.read_text("direct_url.json"))
    install = json.loads(install_path.read_bytes())
    records = [record for record in install["install"]
               if record.get("metadata", {}).get("name", "").lower() == "torch"]
    require(len(records) == 1, "pip report must describe exactly one torch install")
    record = records[0]
    require(record.get("requested") is True and record["metadata"]["version"] == expected["version"],
            "wrong pip torch record")
    for artifact in (direct, record["download_info"]):
        require(artifact.get("url") == wheel["url"] and
                artifact.get("archive_info", {}).get("hashes", {}).get("sha256") == wheel["sha256"],
                "wrong fixed-wheel URL or SHA-256")

    import torch
    require(str(torch.__version__) == expected["version"]
            and torch.version.git_version == expected["git_commit"]
            and {key: getattr(torch.version, key, "MISSING") for key in expected["backend_versions"]}
            == expected["backend_versions"], "wrong loaded torch build/backend")
    require(Path(torch.__file__).resolve() == Path(distribution.locate_file("torch/__init__.py")).resolve(),
            "loaded torch outside the verified distribution")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch, {"oracle_lock_sha256": LOCK_SHA256, "wheel_sha256": wheel["sha256"],
                   "wheel_metadata_sha256": wheel["metadata_sha256"],
                   "torch_version": str(torch.__version__), "torch_git_commit": torch.version.git_version,
                   "python_version": list(sys.version_info[:2]), "platform": expected["platform"]}


def run(wire, binary, torch):
    require(__debug__, "run with assertions enabled for the fixed protocol")
    require(digest(binary) == NATIVE_SHA256, "wrong exact opt-in candidate image")
    cases = Counter()
    values = Counter()
    with torch.inference_mode():
        session = wire.Session(binary)
        try:
            def one(*command):
                return session.run([command])[0]

            def inspect(handle):
                metadata = wire.meta(one(13, handle))
                dtype, width, _, _ = struct.unpack("<4Q", one(96, handle, 0)["payload"])
                data = one(98, handle, 0, metadata["numel"])
                require(data["value"] == metadata["numel"], "wrong typed logical read extent")
                return metadata, dtype, width, data["payload"]

            def compare(tag, words, shape, label, view=None):
                count = len(words)
                source_dtype = getattr(torch, DTYPES[tag])
                if count:
                    raw = bytearray().join(word.to_bytes(WIDTHS[tag], "little") for word in words)
                    torch_input = torch.frombuffer(raw, dtype=source_dtype).reshape(shape)
                else:
                    torch_input = torch.empty(shape, dtype=source_dtype, device="cpu")
                status, records = wire.parse(session.exchange(b"".join(factory(tag, shape, words))))
                require(status == 0 and len(records) == 1 and records[0]["opcode"] == 97,
                        f"{DTYPES[tag]} {label}: invalid typed factory reply")
                base = source = records[0]["value"]
                if view is not None:
                    dims, strides, offset = view
                    torch_input = torch_input.as_strided(dims, strides, offset)
                    payload = struct.pack("<10Q", len(dims), *dims, *([0] * (4 - len(dims))),
                                          *strides, *([0] * (4 - len(dims))), offset)
                    source = one(111, base, wire.Payload(payload), len(payload))["value"]
                torch_output = torch.logical_not(torch_input)
                require(torch_output.dtype == torch.bool and torch_output.device.type == "cpu",
                        "unexpected reference output dtype/device")
                input_before = inspect(source)
                output = one(127, source)["value"]
                actual, dtype, width, data = inspect(output)
                rank = actual["rank"]
                where = f"{DTYPES[tag]} {label}"
                require((dtype, width, data) == (1, 1, bytes(torch_output.flatten().tolist())),
                        f"{where}: native logical values differ from PyTorch")
                require(rank == torch_output.ndim and tuple(actual["shape"][:rank]) == tuple(torch_output.shape),
                        f"{where}: output shape differs from PyTorch")
                require(tuple(actual["strides"][:rank]) == torch_output.stride(),
                        (f"{where}: output strides differ from PyTorch", actual["strides"][:rank],
                         torch_output.stride()))
                require(actual["storage_offset"] == torch_output.storage_offset()
                        and actual["requires_grad"] == int(torch_output.requires_grad),
                        f"{where}: output offset or grad flag differs from PyTorch")
                require(output != source and actual["storage_id"] != input_before[0]["storage_id"]
                        and actual["version_id"] != input_before[0]["version_id"]
                        and inspect(source) == input_before, f"{where}: owner or input mutated")
                one(41, output)
                if source != base:
                    one(41, source)
                one(41, base)
                cases[label] += 1
                values[DTYPES[tag]] += torch_output.numel()

            for tag in DTYPES:
                words = known_words(tag)
                # The full raw-bit ensembles include every possible F16/BF16 word.
                compare(tag, words, (len(words),), "dense")
                compare(tag, [0], (), "scalar_zero")
                compare(tag, [1], (), "scalar_one_bit")
                compare(tag, [], (0, 3), "empty")
                if tag == 4:  # Opcode 111 is the frozen public F64 view path.
                    sample = [0, words[-1], 0, words[-2], words[-3], 0]
                    compare(tag, sample, (3, 2), "dense_transpose", ((2, 3), (1, 2), 0))
                    compare(tag, sample, (3, 2), "offset_slice", ((2, 2), (2, 1), 2))
                    compare(tag, sample, (3, 2), "holes", ((2, 2), (4, 1), 0))
                    compare(tag, sample, (3, 2), "broadcast", ((4, 2), (0, 1), 0))

            require(one(53)["value"] == 1, "final native owner audit failed")
        finally:
            session.finish()
    return {"cases_by_layout": dict(cases), "value_comparisons_by_dtype": dict(values),
            "cases": sum(cases.values()), "values": sum(values.values()), "native_frames": session.frames}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--oracle-lock", required=True, type=Path)
    p.add_argument("--install-report", required=True, type=Path)
    a = p.parse_args()
    require(digest(a.binary) == NATIVE_SHA256, "wrong exact opt-in candidate image")
    wire = load_wire(a.protocol)
    torch, provenance = load_oracle(a.oracle_lock, a.install_report)
    result = run(wire, a.binary.resolve(), torch)
    print(json.dumps({"status": "PASS_SCOPED_STRICT_TORCH",
                      "scope": "opt-in opcode 127 only; not full runtime or production activation",
                      "native_image_sha256": NATIVE_SHA256, "oracle": provenance, **result},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
