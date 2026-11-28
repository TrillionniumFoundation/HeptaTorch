#!/usr/bin/env python3
"""Cold-build and strictly replay the exact opt-in unary/copysign matrix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read_pin(path, pin):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    return data


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cold_signature(path, *, source, native, functions, evidence):
    report = json.loads((path / "build.json").read_bytes())
    expected = {
        "status": "PASS_NATIVE_BUILD_ONLY", "objects_reused": False,
        "source_sha256": source, "native_sha256": native, "functions": functions,
        "units_planned": 15, "compiler_sha256": evidence["compiler_sha256"],
        "linker_sha256": evidence["linker_sha256"], "all_inputs_unchanged": True,
    }
    if any(report.get(key) != val for key, val in expected.items()):
        raise ValueError(f"unexpected native build receipt: {path.name}")
    read_pin(path / "heptatorch", native)
    units = report["units"]
    if len(units) != 15 or len({u["name"] for u in units}) != 15:
        raise ValueError("unexpected object manifest")
    for unit in units:
        read_pin(path / (unit["name"] + ".o"), unit["object_sha256"])
    return [(u["name"], u["source_sha256"], u["object_sha256"]) for u in units]


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "compiler", "linker", "torch-python", "install-report", "output"):
        parser.add_argument("--" + name, type=lambda value: Path(value).absolute(), required=True)
    parser.add_argument("--inputs", type=Path, help="pre-staged original branch files; default reads pinned commits from local Git")
    args = parser.parse_args()
    runtime = args.repository / "tools/current_runtime"
    evidence = json.loads(read_pin(here / "evidence.json", "6260bcb9c7a47ae273826bb27fd2e1de5e7dc7a1a27d91c942dd4eff2c4be56f"))
    for file, pin in evidence["scripts_sha256"].items():
        read_pin(here / file, pin)
    for rel, pin in {
        "build.py": "9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d",
        "partition.py": "72dfb4337b84d810e28d5fa664198200001783d860a230e027906890e20685d4",
        "source/manifest.json": "348ed8e949256f985cdca0d9c612dc646ae738beee6013c27ab405d2df3e67c8",
    }.items():
        read_pin(runtime / rel, pin)
    oracle_files = {rel: read_pin(args.repository / rel, pin) for rel, pin in {
        "compatibility/pytorch-oracle-lock.json": "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944",
        "compatibility/upstream-lock.json": "d204c5a8268a0d0f0a1dd162b5e0ab184beee83310bcad643b084deb691f3f49",
        "tests/pytorch_oracle.py": "ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae",
        "tests/integer_tensor_review/wire/protocol.py": "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9",
        "tests/integer_tensor_review/wire/owner_client.py": "dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b",
        "tests/half_logical/__init__.py": "4c5fbdf0c7f8da54986ba8c0aef475abc92284941bcf215c0545a3528dc8b8fb",
        "tests/half_logical/native.py": "9c2cd19a5bb04eaa037f5905a52bce2dcc31042101f72d07e71205745160a14f",
        "tests/half_logical/reference.py": "5bb6a3804717968dba37f3174809a7c1eee5b225f0a3085b44fcec7656155528",
    }.items()}
    read_pin(args.compiler, evidence["compiler_sha256"])
    read_pin(args.linker, evidence["linker_sha256"])
    if not args.install_report.is_file():
        raise ValueError("pip install report must exist")
    lock = json.loads(read_pin(here / "replay-inputs.json",
                              "783e7e8028d88211939e704bcc4dbd16e4793d59d72115adbde98ef7cdbdd35a"))
    if lock["schema"] != 1:
        raise ValueError("unexpected input inventory schema")
    prepared = {}
    for row in lock["files"]:
        if args.inputs:
            raw = read_pin(args.inputs / row["file"], row["sha256"])
        else:
            raw = subprocess.check_output(
                ["git", "-C", str(args.repository), "show", row["commit"] + ":" + row["path"]],
                timeout=20,
            )
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"predecessor SHA-256 mismatch: {row['file']}")
        if row["file"] in prepared:
            raise ValueError("duplicate input")
        prepared[row["file"]] = raw
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    for name, data in prepared.items():
        dest = inputs / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    oracle_root = out / "oracle"
    # The pinned original oracle changes cwd to its repository and reads this
    # fixed relative path. Stage exact support bytes rather than edit either it
    # or the caller checkout to forward an arbitrary report path.
    for name, data in oracle_files.items():
        dest = oracle_root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    oracle_report = oracle_root / "build/pytorch-install.json"
    oracle_report.parent.mkdir(parents=True)
    oracle_report.write_bytes(args.install_report.read_bytes())
    env = dict(os.environ, HEPTA_PYTORCH_INSTALL_REPORT=str(oracle_report))

    def run(label, script, *params, python=sys.executable, cwd=args.repository):
        # -I disables inherited Python flags; the original qualifier requires asserts.
        with (out / (label + ".stdout")).open("wb") as so, (out / (label + ".stderr")).open("wb") as se:
            subprocess.run([str(python), "-I", "-B", str(script), *map(str, params)],
                           env=env, cwd=cwd, stdout=so, stderr=se,
                           check=True, timeout=420)

    def build(label, source):
        target = out / label
        run(label, runtime / "build.py", "--manifest", source / "manifest.json",
            "--source-root", source, "--compiler", args.compiler, "--linker", args.linker,
            "--output", target)
        return target

    def prepare(group, label, *params):
        target = out / label
        directory = here if group is None else inputs / group
        run(label, directory / "prepare.py", "--runtime-tools", runtime, "--output", target, *params)
        return target

    common = [arg for name in ("square", "reciprocal", "sign")
              for arg in ("--" + name + "-dir", inputs / name)]
    main_build = build("main-build", runtime / "source")
    cold_signature(main_build, source="0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43",
                   native=evidence["predecessor_suites"]["copysign32"]["parent_sha256"],
                   functions=775, evidence=evidence)
    c32 = prepare("copysign32", "f32-source")
    f32_build = build("f32-build", c32)
    cold_signature(f32_build, source=digest(c32 / "runtime.hepta"),
                   native=evidence["predecessor_suites"]["copysign64"]["parent_sha256"],
                   functions=776, evidence=evidence)
    c64 = prepare("copysign64", "f64-source", "--f32-source", c32 / "runtime.hepta")
    unary = prepare("unary", "unary-source", *common)
    combine_args = ["--unary-source", unary / "runtime.hepta", "--copysign-source", c64 / "runtime.hepta"]
    source_a = prepare(None, "combined-source-a", *combine_args)
    build_a = build("combined-build-a", source_a)
    source_b = prepare(None, "combined-source-b", *combine_args, "--expected-build", build_a / "build.json")
    build_b = build("combined-build-b", source_b)
    signatures = [cold_signature(path, source=evidence["source_sha256"],
                                 native=evidence["candidate_sha256"], functions=evidence["functions"],
                                 evidence=evidence) for path in (build_a, build_b)]
    if signatures[0] != signatures[1]:
        raise ValueError("cold object signatures differ")

    for build_dir, round_name in ((build_a, "a"), (build_b, "b")):
        qualification = out / ("qualification-" + round_name)
        run("qualify-" + round_name, here / "qualify.py", "--repository", oracle_root,
            "--parent", main_build / "heptatorch", "--f32-parent", f32_build / "heptatorch",
            "--candidate", build_dir / "heptatorch", "--output", qualification,
            "--unary-dir", inputs / "unary", "--copysign32-dir", inputs / "copysign32",
            "--copysign64-dir", inputs / "copysign64", *common, python=args.torch_python, cwd=oracle_root)
        read_pin(qualification / "qualification.json", evidence["qualification_sha256"])

    gate = {
        "status": "SCOPED_PASS", "canonical_credit": False,
        "qualification_sha256": evidence["qualification_sha256"],
        "candidate_sha256": evidence["candidate_sha256"],
        "cross_family_session": evidence["cross_family_session"],
        "focused_totals": evidence["focused_totals"],
        "two_cold_builds_byte_identical": True, "two_full_replays_report_byte_identical": True,
        "object_sha256": {name: obj for name, _, obj in signatures[0]},
    }
    (out / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
