#!/usr/bin/env python3
"""Run the frozen logical147 suite on exactly the joint opt-in current800 build."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
import types

NATIVE = "5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72"
REFERENCE = "e2874986d52e62725d54767ea5824a76d066e948b5d54a99e3b24f9e32c0e425"
SHA256 = {
    "tools/current_runtime/opt_in_admission/admit.py": "51b32b1835fe8a294f1f42b98ca3b87b10a5096dec08d34840ac7acab61f672d",
    "tools/current_runtime/qualification.py": "f4be6d41c29659c3132f1f64014ad9d8ebcf612db3a9cb556ee5ddd178a540b6",
    "tests/logical_reductions/qualify.py": "e1d713f00f5a71920ee94b0fd617a8cb91209fa342849c1ac4ea8a96d4e4a25c",
    "tests/logical_reductions/native.py": "f6d5d2a3112df5193c3df4e3f78ce734ad69f6340068d8ecc63c48ec23575355",
    "tests/logical_reductions/cases.py": "347bb0d518fc34d589a6018a49729a18f62ce0b29cc2cc028952da38c64b57a5",
    "tests/logical_reductions/reference.py": "fcdae0181548b0de2e13399d453d3319c2e0e2d4c7dcbcf20f1dcad2ad9b145f",
    "tests/differentiable_view_native_review/protocol.py": "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9",
    "tests/differentiable_view_native_review/owner_client.py": "dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b",
}
OBSERVER = "tests/logical_reductions/qualify.py"
OLD_CALL = b"built=profile.admit_build(build,inputs.bind)"
NEW_CALL = b"built=admit_dual_observer(profile,build,inputs.bind)"
LEGACY_DENIAL = "build used another source manifest"


def require(condition, why):
    if not condition:
        raise ValueError(why)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and path.resolve() == path, "ordinary input required: " + str(path))
    data = path.read_bytes()
    info2 = path.lstat()
    key = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(key(info) == key(info2) and len(data) == info2.st_size, "input changed: " + str(path))
    return data


def load(name, path, source, extra=None):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(extra or {})
    exec(compile(source, str(path), "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


def clean_caches(repo):
    # The unchanged controller starts a Python child that uses ordinary
    # package imports for cases and transport. -B does not ignore existing caches.
    for parent in {((repo / p).parent) for p in SHA256 if p.startswith("tests/")} | {repo / "tools/current_runtime"}:
        require(not (parent / "__pycache__").exists(), "use a clean checkout without bytecode caches: " + str(parent))


def run(repo, build, scoped, output):
    clean_caches(repo)
    sources = {p: read(repo / p) for p in SHA256}
    require(all(digest(data) == SHA256[p] for p, data in sources.items()), "fixed observer or admission source differs")
    base_manifest = repo / "tools/current_runtime/source/manifest.json"
    admit_path = "tools/current_runtime/opt_in_admission/admit.py"
    admit = load("frozen_opt_in_admission", repo / admit_path, sources[admit_path]).admit
    original = sources[OBSERVER]
    require(original.count(OLD_CALL) == 1 and NEW_CALL not in original, "retained launcher anchor differs")
    derived = original.replace(OLD_CALL, NEW_CALL, 1)
    require(derived.replace(NEW_CALL, OLD_CALL, 1) == original, "retained launcher inverse differs")
    evidence = {"observer_derived_sha256": digest(derived), "observer_single_call_inverse_exact": True}
    build_preflight = admit("dual", base_manifest, build, {"dual_scoped": scoped})

    def admit_dual_observer(legacy, chosen, bind):
        require(chosen == build and "dual_admission" not in evidence, "unexpected retained launcher invocation")
        try:
            legacy.admit_build(chosen, bind)
        except ValueError as error:
            require(str(error) == LEGACY_DENIAL, "unexpected original admission refusal")
        else:
            raise ValueError("original admission unexpectedly accepted an opt-in candidate")
        admission = admit("dual", base_manifest, build, {"dual_scoped": scoped})
        require(admission == build_preflight and admission["native_image_sha256"] == NATIVE, "dual admission changed")
        bind(build / "heptatorch", NATIVE)
        bind(build / "runtime.hepta", admission["source_sha256"])
        bind(base_manifest, admit.__globals__["BASE_MANIFEST"])
        evidence.update(original_canonical_admission_refusal=LEGACY_DENIAL, dual_admission=admission)
        return json.loads(bind(build / "build.json", admission["build_report_sha256"]))

    module = load("joint_logical147_retained", repo / OBSERVER, derived, {"admit_dual_observer": admit_dual_observer})
    argv = sys.argv[:]
    try:
        sys.argv = [str(repo / OBSERVER), "--build-directory", str(build), "--output", str(output / "observer")]
        code = module.main()
    finally:
        sys.argv = argv
    raw = read(output / "observer/report.json")
    result = json.loads(raw)
    evidence.update(observer_exit_code=code, observer_report_sha256=digest(raw), observer_report_status=result["status"],
                    native_started=result["native_executed"], reference_source="frozen original PyTorch observation")
    if result["status"] == "PASS":
        require(code == 0 and result["native_executed"] is True and result["torch_executed"] is False
                and result["all_native_EOF"] is True and result["all_inputs_unchanged"] is True
                and result["native_sha256"] == NATIVE and result["source_sha256"] == build_preflight["source_sha256"]
                and result["actual_build_sha256"] == build_preflight["build_report_sha256"]
                and result["reference_sha256"] == REFERENCE and result["cases"] == 147
                and result["packet_controls"] == 18 and result["allocation_controls"] == 2
                and result["native_children"] == 150, "retained observer evidence incomplete")
        runner = result["runner"]
        require(runner["started"] is True and runner["exit_code"] == 0
                and runner["stdout_EOF"] is True and runner["stderr_EOF"] is True
                and runner["stderr_bytes"] == 0 and not runner.get("timed_out")
                and not runner.get("forced_cleanup"), "complete actual logical runner required")
        native_raw = read(output / "observer/native/report.json")
        native = json.loads(native_raw)
        require(digest(native_raw) == result["native_report_sha256"] and native["status"] == "PASS"
                and native["cases"] == native["expected_cases"] == len(native["rows"]) == 147
                and all(row["status"] == "PASS" for row in native["rows"])
                and native["source_sha256"] == build_preflight["source_sha256"]
                and native["native_sha256"] == NATIVE, "all actual native logical comparisons required")
        raw_reference = read(output / "observer/reference.json")
        require(digest(raw_reference) == REFERENCE and json.loads(raw_reference)["status"] == "OBSERVED"
                and result["full_pytorch_rewrite_complete"] is False and result["schema_closure"] is False,
                "independent logical reference and finite scope required")
        evidence["observed"] = {k:result[k] for k in ("cases", "frames", "case_checks", "total_checks",
              "packet_controls", "allocation_controls", "native_children", "reference_sha256", "native_report_sha256")}
    else:
        evidence["observed_error"] = result.get("error")
    require(all(read(repo / p) == data for p, data in sources.items()), "observer sources changed during replay")
    require(admit("dual", base_manifest, build, {"dual_scoped": scoped}) == build_preflight,
            "actual native build changed during replay")
    return result["status"] == "PASS", evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, required=True, help="exact #63 checkout (canonical files still exact #61)")
    p.add_argument("--dual-build", type=Path, required=True)
    p.add_argument("--dual-scoped", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(sys.flags.isolated and not sys.flags.optimize and sys.dont_write_bytecode, "run with python3 -I -B")
    require(not a.dual_build.is_symlink() and not a.dual_scoped.is_symlink() and not a.output.is_symlink(),
            "symlinked build, receipt, or output")
    out = a.output.resolve()
    out.mkdir()
    receipt = dict(schema="heptatorch.opt-in.logical147-replay.v1", status="FAIL",
                   retained_suite="logical147", mandatory_suite_count=35,
                   full_candidate_suite_qualification=False, canonical_qualification_transferred=False,
                   production_activated=False, observer_sources_sha256=SHA256,
                   replay_source_sha256=digest(read(Path(__file__).resolve())))
    try:
        passed, evidence = run(a.repository.resolve(strict=True), a.dual_build.resolve(strict=True),
                               a.dual_scoped.resolve(strict=True), out)
        receipt["status"] = "PASS_RETAINED_LOGICAL147_ON_DUAL" if passed else "FAIL_RETAINED_LOGICAL147_ON_DUAL"
        receipt["evidence"] = evidence
    except (OSError, ValueError, AssertionError, KeyError, TypeError) as error:
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        passed = False
    with (out / "receipt.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(receipt["status"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
