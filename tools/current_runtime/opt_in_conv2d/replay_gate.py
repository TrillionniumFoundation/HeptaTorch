"""Rebuild and replay the fixed #78 convolution differential qualification."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
POOL = RUNTIME / "opt_in_adaptive_avg_pool2d"
COMPILER = "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"
LINKER = "761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e"
WHEEL = "0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3"
BASE_SOURCE = "0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43"
AVG_BUILD = "be63305774cb9964348d50c2b7dfdc2e74663da597cf98e20c37fcf513ce61f2"
CONV_BUILD = "9724d9bd92d30d039767316289e374c7ce7adff6d2cef72f716b4b44dc2162c1"
CONV_REPORT = "a5de924eea32ff3afb5c0a7ae61e2e6a68b662a87c69452aa9912106ce715061"
UNIT_FIELDS = ("name", "source_sha256", "object_sha256", "object_bytes",
               "functions", "blocks", "instructions", "compiler_exit",
               "compiler_stderr_bytes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_pin(path, expected, label):
    require(sha(path) == expected, label + " SHA-256 mismatch")


def read_locked(path, expected, label):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, label + " SHA-256 mismatch")
    return json.loads(raw)


def command(executable, script, args, out, label):
    with (out / (label + ".stdout")).open("xb") as stdout, (
        out / (label + ".stderr")
    ).open("xb") as stderr:
        result = subprocess.run([str(executable), "-I", "-B", str(script),
                                 *map(str, args)], cwd=RUNTIME.parents[1],
                                stdout=stdout, stderr=stderr, timeout=300)
    require(result.returncode == 0, f"{label} failed ({result.returncode}); see {out}")


def compare_build(observed, expected):
    require(observed["status"] == "PASS_NATIVE_BUILD_ONLY", "native build did not pass")
    require(observed["all_inputs_unchanged"] is True and observed["objects_reused"] is False,
            "build did not use unchanged inputs and fresh objects")
    for name in ("compiler_sha256", "linker_sha256", "stack_inspector_sha256",
                 "source_sha256", "manifest_sha256", "native_sha256"):
        require(observed[name] == expected[name], "build differs on " + name)
    if "stack" in expected:
        require(observed["stack"] == expected["stack"], "stack budget report differs")

    def pins(units):
        return [tuple(row[k] for k in UNIT_FIELDS) for row in units]

    require(pins(observed["units"]) == pins(expected["units"]), "native object table differs")


def build(out, name, prefix, runtime, compiler, linker, expected):
    source, target = out / (prefix + "-source"), out / (prefix + "-build")
    command(sys.executable, RUNTIME / "partition.py", (
        "--runtime", runtime, "--name", name, "--prefix", prefix, "--output", source,
    ), out, prefix + "-partition")
    command(sys.executable, RUNTIME / "build.py", (
        "--manifest", source / "manifest.json", "--source-root", source,
        "--compiler", compiler, "--linker", linker, "--output", target,
    ), out, prefix + "-build")
    compare_build(json.loads((target / "build.json").read_bytes()), expected)
    image = target / "heptatorch"
    check_pin(image, expected["native_sha256"], prefix + " native image")
    return image


def compare_report(observed, expected):
    # A fresh install's pip report records its own source path. The qualifier
    # must still bind that report to the wheel, and all tested component bytes
    # and result digests must match the published observation.
    def stable(report):
        return {**report, "reference": {
            key: value for key, value in report["reference"].items()
            if key != "install_report_sha256"
        }}

    require(stable(observed) == stable(expected), "differential report differs from #78")
    require(observed["status"] == "PASS" and observed["full_pytorch_rewrite_complete"] is False,
            "wrong qualification status or scope")
    counts = Counter(sum(row["gradients"]) for row in observed["cases"])
    require(counts == {0: 8, 1: 16, 2: 8}, "wrong gradient coverage")


def replay(args):
    require(sys.version_info >= (3, 11), "requires driver Python 3.11 or newer")
    require(sys.platform == "linux" and platform.machine() == "x86_64", "requires Linux x86-64")
    out = args.output.absolute()
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    compiler, linker, wheel, install = [
        path.absolute() for path in (args.compiler, args.linker, args.wheel, args.install)
    ]
    # Do not resolve the Torch executable: its invocation path selects the venv.
    python = args.torch_python.absolute()
    for path, expected, label in ((compiler, COMPILER, "compiler"), (linker, LINKER, "linker"),
                                  (wheel, WHEEL, "Torch wheel"),
                                  (RUNTIME / "source/runtime.hepta", BASE_SOURCE, "base source")):
        check_pin(path, expected, label)
    avg = read_locked(POOL / "build-receipt.json", AVG_BUILD, "average-pool build receipt")
    conv = read_locked(HERE / "build-receipt.json", CONV_BUILD, "conv2d build receipt")
    expected = read_locked(HERE / "qualification.json", CONV_REPORT, "conv2d differential receipt")
    avg_src, conv_src = out / "avg.hepta", out / "conv.hepta"
    command(sys.executable, POOL / "project_pool.py", (
        "--parent", RUNTIME / "source/runtime.hepta", "--output", avg_src,
    ), out, "project-pool")
    command(sys.executable, HERE / "project_conv.py", (
        "--parent", avg_src, "--output", conv_src,
    ), out, "project-conv")
    parent = build(out, "narrow-avg-pool-no-grad777", "avg_pool777", avg_src, compiler, linker, avg)
    candidate = build(out, "narrow-conv-no-grad780", "conv780", conv_src, compiler, linker,
                      conv["builds"][0])
    command(python, HERE / "qualify.py", (
        "--binary", candidate, "--parent", parent, "--wheel", wheel, "--install", install,
    ), out, "differential")
    actual = json.loads((out / "differential.stdout").read_bytes())
    compare_report(actual, expected)
    check_pin(install, actual["reference"]["install_report_sha256"], "replayed install report")
    summary = {
        "schema": "heptatorch.opt-in-conv2d-replay-gate.v1",
        "status": "PASS_SCOPED_REPLAY",
        "candidate_sha256": sha(candidate),
        "parent_sha256": sha(parent),
        "differential_sha256": sha(out / "differential.stdout"),
        "comparisons": len(actual["cases"]),
        "candidate_conv_frames": sum(row["native_frames"] for row in actual["cases"]),
        "parent_no_grad_refusals": sum(row["parent_status"] == 1 for row in actual["cases"]),
        "channel_refusals": len(actual["channel_refusals"]),
        "native_chain_frames": actual["no_grad_chain"]["frames"],
        "full_pytorch_rewrite_complete": False,
    }
    with (out / "gate.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, sort_keys=True, indent=2)
        f.write("\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler", "linker", "torch-python", "wheel", "install", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    try:
        print(json.dumps(replay(p.parse_args()), sort_keys=True))
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        p.exit(1, "conv2d replay gate: " + str(exc) + "\n")
