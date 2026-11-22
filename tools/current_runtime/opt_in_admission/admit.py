"""Offline identity admission for named current800 operator builds.

This verifies on-disk bytes and retained evidence bindings. It does not run
anything from the inputs, qualify a runtime, or update any canonical reader.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct


BASE_COMMIT = "f4f3bd5b44a14868e1af532e689b07c4ff197a89"
BASE_MANIFEST = "a5cf09f191db4887b4452001e0b72e75fffb2147f2476544cc3be49d8ffa2c85"
BASE_IMAGE = "083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5"
UNITS = ("rng03", *(f"runtime800_{index:02d}" for index in range(13)), "owners")
BUILD_TOOL_PINS = {
    "stack-module-compiler": ("compiler_sha256", "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"),
    "stack-linker-session": ("linker_sha256", "761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e"),
}
PROFILES = {
    "baseline": (BASE_MANIFEST, BASE_IMAGE, {}, {}),
    "logical_not": (
        "4538aaa9a5110e7dc9706d343fa9598d5b8c7b2826d1b86bc3249d3650a13041",
        "0cd10504717c5e5b01114455c55030e63a22b62bb1ba43f188355d399d6c158a",
        {
            "runtime800_10": "82d34309904925334f6d4168ae72f4dde69ef45487fdd942d2a8dae5f8d5e45e",
            "runtime800_12": "588beef6ef4573603a57dc30113fe81573d915796ff5863827a8329be919e833",
        },
        {
            "strict": ("a69b671914c1344e3a91144b27d5bfa8bf0c75a66efb7965cae3a7ba51a03323",
                       "native_image_sha256", "PASS_SCOPED_STRICT_TORCH"),
            "layout": ("0987873c1f5a821cccd0ef878a1438421a8b5032017022407143b0e29984a510",
                       "native_image_sha256", "PASS_SCOPED_LAYOUT"),
        },
    ),
    "dual": (
        "9f24d9903a9ffa96cd6121cbc7130cefe8536f2c3e5a425e6d669f72242de1f0",
        "5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72",
        {
            "runtime800_06": "91a4cb5a1bd45c0c3747c684427ece796a8ab60eab763f87189d7cd3caede87a",
            "runtime800_10": "82d34309904925334f6d4168ae72f4dde69ef45487fdd942d2a8dae5f8d5e45e",
            "runtime800_12": "588beef6ef4573603a57dc30113fe81573d915796ff5863827a8329be919e833",
        },
        {"dual_scoped": ("60c6238f52c8d505f915d4d9a942ce20f8807db04efcb90a133346d4b53f9d27", "native_image_sha256", "PASS_SCOPED_DUAL_ABS_LOGICAL_NOT")},
    ),
    "wide_abs": (
        "6e1d182d172bb2fe3f89eb6ae07b0d9841304e84b4fb7a20851b37e6bd8b5e98",
        "2adc665f1b042145725f8f92ded85e3f7f051f293180d704287847b16a95e005",
        {"runtime800_06": "91a4cb5a1bd45c0c3747c684427ece796a8ab60eab763f87189d7cd3caede87a"},
        {"strict": ("fb5330f3eb93aa40074ab4f939e6a22a01fb9d9762087573161787db8a671e29",
                    "candidate_image_sha256", "PASS_SCOPED_OPT_IN_WIDE_ABS")},
    ),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path, expected=None):
    # Refuse symlinks, devices, FIFOs and directories. No writes or execution.
    need(not path.is_symlink() and path.is_file(), f"not a regular input: {path}")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), "rb") as stream:
        need(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), f"not a regular input: {path}")
        raw = stream.read()
    need(expected is None or sha(raw) == expected, f"SHA-256 differs: {path}")
    return raw


def unpack_json(raw):
    def pairs(rows):
        unique = dict(rows)
        need(len(unique) == len(rows), "duplicate JSON keys")
        return unique
    return json.loads(raw, object_pairs_hook=pairs)


def admit(profile, baseline_manifest, build, evidence):
    manifest_sha, image_sha, deltas, evidence_pins = PROFILES[profile]
    need(not build.is_symlink() and build.is_dir(), "expected a real build directory")
    need(set(evidence) == set(evidence_pins), "scoped receipt labels differ from the chosen profile")
    baseline = unpack_json(read(baseline_manifest, BASE_MANIFEST))
    manifest = unpack_json(read(build / "manifest.json", manifest_sha))
    report_raw = read(build / "build.json")
    report = unpack_json(report_raw)
    for manifest_units in (baseline["units"], manifest["units"]):
        need(tuple(row["name"] for row in manifest_units) == UNITS, "partition roster differs")
    need(manifest["runtime"]["functions"] == 800, "native function census differs")
    need(manifest.get("expected_native_sha256", image_sha) == image_sha, "manifest executable differs")
    source_sha = manifest["runtime"]["sha256"]
    read(build / "runtime.hepta", source_sha)
    for key, value in {
        "schema": "heptatorch.complete-runtime-build.v1", "status": "PASS_NATIVE_BUILD_ONLY",
        "profile": manifest["name"], "manifest_sha256": manifest_sha, "source_sha256": source_sha,
        "native_basename": "heptatorch", "native_sha256": image_sha, "functions": 800,
        "units_planned": 15, "complete_core_bodies_matched": 800,
    }.items():
        need(report.get(key) == value, f"build report differs: {key}")
    for key, value in {
        "all_inputs_unchanged": True, "objects_reused": False, "runtime_qualified": False,
        "full_pytorch_rewrite_complete": False, "result_executable_executed": False,
        "historical_runtime_executed": False, "torch_executed": False,
    }.items():
        need(report.get(key) is value, f"build report differs: {key}")
    need(report["link_exit"] == 0, "build linker failed")
    need(not (build / "toolchain").is_symlink(), "linked toolchain directory not allowed")
    for name, (key, expected) in BUILD_TOOL_PINS.items():
        need(report[key] == expected, f"build tool identity differs: {name}")
        read(build / "toolchain" / name, expected)
    need(tuple(row["name"] for row in report["units"]) == UNITS, "build report omitted a unit")
    need(tuple(row["name"] for row in report["processes"]) == (*UNITS, "link"), "build process roster differs")
    for process in report["processes"]:
        need(process["started"] is True and process["exit_code"] == 0
             and process["stdout_EOF"] is True and process["stderr_EOF"] is True
             and process["stderr_bytes"] == 0 and process["stderr_sha256"] == sha(b""),
             f"build process evidence is incomplete: {process['name']}")
    links, objects, actual_deltas = [], {}, []
    for old, row, unit, process in zip(baseline["units"], manifest["units"],
                                       report["units"], report["processes"][:-1], strict=True):
        name = row["name"]
        unit_pin = row["sha256"]
        object_pin = deltas.get(name, old["expected_object_sha256"])
        if old["sha256"] != unit_pin or old["expected_object_sha256"] != object_pin:
            actual_deltas.append(name)
        need(row.get("expected_object_sha256", object_pin) == object_pin, f"manifest object differs: {name}")
        source = read(build / (name + ".hepta"), unit_pin)
        obj = read(build / (name + ".o"), object_pin)
        need(obj[:6] == b"\x7fELF\x02\x01" and struct.unpack_from("<2H", obj, 16) == (1, 62),
             f"not a Linux x86-64 ELF object: {name}")
        need(unit["source_sha256"] == unit_pin and unit["object_sha256"] == object_pin
             and unit["object_bytes"] == len(obj) and unit["compiler_exit"] == 0
             and unit["compiler_stderr_bytes"] == 0 and process["input_sha256"] == unit_pin
             and process["input_bytes"] == len(source)
             and process["stdout_sha256"] == object_pin and process["stdout_bytes"] == len(obj)
             and read(build / (name + ".stderr")) == b"", f"build output binding differs: {name}")
        links.append(struct.pack("<Q", len(obj)) + obj)
        objects[name] = object_pin
    need(tuple(actual_deltas) == tuple(deltas), "changed partition census differs")
    request = b"HIACGS01" + struct.pack("<3Q", len(UNITS), 4, 0) + b"step" + b"".join(links)
    image = read(build / "heptatorch", image_sha)
    link = report["processes"][-1]
    need(image[:6] == b"\x7fELF\x02\x01" and struct.unpack_from("<2H", image, 16) == (2, 62)
         and len(image) == report["native_bytes"], "not the exact Linux x86-64 executable")
    need(read(build / "link.request.bin") == request and read(build / "link.stderr") == b""
         and link["input_bytes"] == len(request) and link["input_sha256"] == sha(request)
         and link["stdout_bytes"] == len(image) and link["stdout_sha256"] == image_sha,
         "link input/output binding differs")
    receipts = {}
    for label, (pin, native_key, result) in evidence_pins.items():
        receipt = unpack_json(read(evidence[label], pin))
        need(receipt.get(native_key) == image_sha and receipt.get("status") == result,
             f"scoped receipt identity differs: {label}")
        if profile in ("wide_abs", "dual"):
            need(receipt["baseline_image_sha256"] == BASE_IMAGE
                 and receipt["canonical_runtime_qualification"] is False, "abs or dual scope differs")
        if profile == "dual":
            need(receipt["full_candidate_suite_qualification"] is False
                 and receipt["whole_pytorch_rewrite_complete"] is False, "dual scope differs")
        receipts[label] = dict(sha256=pin, status=result)
    return dict(schema="heptatorch.offline-opt-in-build-identity.v1", profile=profile,
                baseline_commit=BASE_COMMIT, manifest_sha256=manifest_sha, source_sha256=source_sha,
                native_image_sha256=image_sha, build_report_sha256=sha(report_raw), changed_units=actual_deltas,
                objects=objects, scoped_retained_receipts=receipts, native_executed=False,
                canonical_qualification_transferred=False, candidate_qualification_granted=False,
                production_activated=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True, choices=PROFILES)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--strict", type=Path)
    p.add_argument("--layout", type=Path)
    p.add_argument("--dual-scoped", type=Path)
    a = p.parse_args()
    evidence = {k: getattr(a, k) for k in ("strict", "layout", "dual_scoped") if getattr(a, k) is not None}
    try:
        result = admit(a.profile, a.baseline_manifest, a.build, evidence)
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        p.exit(1, f"ADMISSION REFUSED: {error}\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
