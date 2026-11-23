"""Run offline admission controls using genuine previously built inputs."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import types


def load(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), path, "exec"), module.__dict__)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-reader", "baseline", "logical-not", "logical-strict",
                 "logical-layout", "wide-abs", "abs-strict", "dual", "dual-scoped"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    check = load(Path(__file__).resolve().with_name("admit.py"), "check")
    original_path = a.canonical_reader.resolve(strict=True)
    check.read(original_path, "f4be6d41c29659c3132f1f64014ad9d8ebcf612db3a9cb556ee5ddd178a540b6")
    original_reader = load(original_path, "canonical")
    baseline_manifest = a.baseline / "manifest.json"
    cases = []

    def accept(name, profile, build, evidence, units):
        row = check.admit(profile, baseline_manifest, build, evidence)
        check.need(row["changed_units"] == units and len(row["objects"]) == 15
                   and row["canonical_qualification_transferred"] is False
                   and row["candidate_qualification_granted"] is False
                   and row["native_executed"] is False
                   and row["production_activated"] is False, f"unexpected grant: {name}")
        cases.append(name)

    def reject(name, call, expected):
        try:
            call()
        except ValueError as error:
            check.need(expected in str(error), f"wrong refusal: {name}: {error}")
            cases.append(name)
            return
        raise ValueError(f"missing refusal: {name}")

    logical = {"strict": a.logical_strict, "layout": a.logical_layout}
    absolute = {"strict": a.abs_strict}
    dual = {"dual_scoped": a.dual_scoped}
    accept("exact_baseline", "baseline", a.baseline, {}, [])
    accept("exact_logical_image", "logical_not", a.logical_not, logical, ["runtime800_10", "runtime800_12"])
    accept("exact_abs_image", "wide_abs", a.wide_abs, absolute, ["runtime800_06"])
    accept("exact_dual_image", "dual", a.dual, dual, ["runtime800_06", "runtime800_10", "runtime800_12"])
    check.need(original_reader.admit_build(a.baseline, check.read)["status"] == "PASS_NATIVE_BUILD_ONLY",
               "the original reader no longer admits the baseline")
    cases.append("original_reader_baseline_unchanged")
    for label, directory in (("logical_not", a.logical_not), ("wide_abs", a.wide_abs), ("dual", a.dual)):
        reject("original_reader_retains_refusal_" + label,
               lambda: original_reader.admit_build(directory, check.read), "build used another source manifest")

    reject("wrong_manifest", lambda: check.admit("wide_abs", baseline_manifest, a.logical_not, absolute), "SHA-256 differs")
    reject("wrong_baseline_manifest",
           lambda: check.admit("wide_abs", a.logical_not / "manifest.json", a.wide_abs, absolute), "SHA-256 differs")
    reject("swapped_scoped_receipts", lambda: check.admit("wide_abs", baseline_manifest, a.wide_abs,
           {"strict": a.logical_strict}), "SHA-256 differs")
    reject("single_receipt_not_evidence_for_dual",
           lambda: check.admit("dual", baseline_manifest, a.dual, {"dual_scoped": a.abs_strict}), "SHA-256 differs")
    reject("missing_scoped_receipt", lambda: check.admit("logical_not", baseline_manifest, a.logical_not,
           {"strict": a.logical_strict}), "scoped receipt labels differ")
    reject("extra_scoped_receipt_on_baseline", lambda: check.admit(
           "baseline", baseline_manifest, a.baseline, absolute), "scoped receipt labels differ")
    with tempfile.TemporaryDirectory(prefix="heptatorch-opt-in-control-") as temp:
        root = Path(temp)
        staged = shutil.copytree(a.wide_abs, root / "build")

        def mutate(relative, payload, name, expected):
            path = staged / relative
            backup = path.read_bytes()
            # copytree preserves the genuine builder's read-only ELF mode.
            # Replace only the staged file; never chmod or mutate the originals.
            path.unlink()
            path.write_bytes(payload(backup))
            try:
                reject(name, lambda: check.admit("wide_abs", baseline_manifest, staged, absolute), expected)
            finally:
                path.write_bytes(backup)

        def flip(data):
            return data[:-1] + bytes([data[-1] ^ 1])

        mutate("runtime.hepta", flip, "changed_monolith", "SHA-256 differs")
        mutate("runtime800_06.hepta", flip, "changed_partition_source", "SHA-256 differs")
        mutate("runtime800_06.o", lambda _: (a.baseline / "runtime800_06.o").read_bytes(),
               "stale_canonical_object", "SHA-256 differs")
        mutate("heptatorch", lambda _: (a.baseline / "heptatorch").read_bytes(),
               "stale_canonical_executable", "SHA-256 differs")
        mutate("link.request.bin", flip, "edited_link_request", "link input/output binding differs")
        mutate("build.json", lambda v: json.dumps(dict(json.loads(v), runtime_qualified=True)).encode(),
               "no_fabricated_runtime_pass", "build report differs: runtime_qualified")
        mutate("build.json", lambda v: v[:-2] + b', "status":"PASS_NATIVE_BUILD_ONLY"}',
               "duplicate_report_status", "duplicate JSON keys")
        path = staged / "heptatorch"
        path.unlink()
        path.symlink_to((a.wide_abs / "heptatorch").resolve(strict=True))
        reject("no_symlinked_executable", lambda: check.admit(
            "wide_abs", baseline_manifest, staged, absolute), "not a regular input")
    print(json.dumps({"status": "PASS_OFFLINE_ADMISSION_CONTROLS", "controls": cases}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
