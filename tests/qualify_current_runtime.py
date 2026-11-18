#!/usr/bin/env python3
"""Execute all focused current-runtime suites and retain their original evidence.

This coordinator transports fixed observations and invokes independent native
readers and the fixed live Torch reference for original35. It implements no
Tensor arithmetic, edits no old report, and grants no whole-project qualification
from these finite suites.
"""

import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import current_runtime_fixtures as fixed_fixtures
from current_runtime_process import execute as execute_process
from current_runtime_admission import qualify_historical, qualify_f32, qualify_current35, qualify_current270
SOURCE = "82360ce87dae6f9a004ef50be4f991b6f83905a83f4b17ca8d6634827b7bd313"
IMAGE = "80445a8c433f20314d3bc7b00ebb68f6a70fa200ee2e6aba193cff21aabfbc17"
COMPILER = "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"
LINKER = "aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16"
BOOL_REFERENCE = "21dd75aeb86414818f7aa381f8b866d78b16f9bb3f5738e9a60d27ac3d15dd28"
BOOL_ARCHIVE = "353a8b2853773bc6678bb62fc9559d6f4244c571ba40c6b80f352b2649cd6edd"
LEGACY_ARCHIVE = "511dff328018a0a2d6a56a94fde4d2e0d64ff10ea66bb0cb71f89bb61ab2a3fb"
SUM32_REFERENCE = "c14e4b18f657429ad5dac5498c3abcae303b6a7e7550db7d15a3c9179bdcd728"
INTEGRAL_REFERENCE = "6c9df4b579664c89c200b9d796367dd6d41f93f8260e8ff012f17d5041d2278b"
SHARED_ARCHIVE = "validation/current-runtime-references/complete.tar.xz"
SHARED_ARCHIVE_SHA = "3a29420768a1bf7808976f04cb3ce3e6091db55ff6ae7858879f1b7ff70b69ea"
SHARED_MANIFEST_SHA = "666fa775586679cfd42606c95656ad261c8856332431931ed4b0fcfd8865057e"
GROUPS = ("bool27", "original76", "integral24", "sum_f32_17", "current35", "base270", "sum32")
SUITE_TIMEOUT_SECONDS = {"bool27": 600, "original76": 900, "integral24": 900,
                         "sum_f32_17": 600, "current35": 900, "base270": 900, "sum32": 3600}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


class Inputs:
    def __init__(self):
        self.records = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        require(not path.is_symlink(), "symlinked input: " + str(path))
        before = path.stat()
        require(stat.S_ISREG(before.st_mode), "nonregular input: " + str(path))
        raw = path.read_bytes()
        after = path.stat()
        fields = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(fields(before) == fields(after), "input changed while reading: " + str(path))
        row = {"bytes": len(raw), "sha256": sha(raw), "mode": stat.S_IMODE(after.st_mode)}
        require(expected is None or row["sha256"] == expected, "input SHA mismatch: " + str(path))
        require(str(path) not in self.records or self.records[str(path)] == row,
                "bound input changed: " + str(path))
        self.records[str(path)] = row
        return raw

    def recheck(self):
        errors = []
        for path, row in list(self.records.items()):
            try:
                self.bind(path, row["sha256"])
            except BaseException as error:
                errors.append({"path": path, "error": repr(error)})
        return errors


def build_binding(directory, inputs):
    raw = inputs.bind(directory / "build.json")
    built = json.loads(raw)
    require(built["schema"] == "heptatorch.complete-runtime-build.v1"
            and built["status"] == "PASS_NATIVE_BUILD_ONLY", "an actual cold build is required")
    require(built["functions"] == built["complete_core_bodies_matched"] == 701
            and built["units_planned"] == len(built["units"]) == 14
            and len(built["processes"]) == 15, "the entire current runtime must be built")
    require(built["objects_reused"] is False and built["all_inputs_unchanged"] is True,
            "cold-build input custody failed")
    require(built["source_sha256"] == SOURCE and built["native_sha256"] == IMAGE
            and built["compiler_sha256"] == COMPILER and built["linker_sha256"] == LINKER
            and built["link_exit"] == 0, "current source, image or tool identity changed")
    require(built["runtime_qualified"] is False and built["full_pytorch_rewrite_complete"] is False,
            "build receipt must not claim runtime or project qualification")
    inputs.bind(directory / "runtime.hepta", SOURCE)
    image = inputs.bind(directory / "heptatorch", IMAGE)
    require(len(image) == built["native_bytes"] and image.startswith(b"\x7fELF"),
            "current ELF byte count or header changed")
    return sha(raw)


class Fixtures:
    def __init__(self, output, inputs):
        self.output, self.inputs = output, inputs
        self.cache = {}

    def bool27(self):
        if "bool27" in self.cache:
            return self.cache["bool27"]
        original = ROOT / "validation/sum-bool27-reference"
        archive = self.inputs.bind(original / "report.json.xz", BOOL_ARCHIVE)
        raw = lzma.decompress(archive)
        require(len(archive) == 16332 and len(raw) == 3549617 and sha(raw) == BOOL_REFERENCE,
                "original Bool reference bytes changed")
        metadata = {}
        for name, pin in (
            ("manifest.json", "243b78143a0e008355ee3e31c77ac3713765e0b51af6463cef665c5b79ba90ba"),
            ("execution.json", "40ec5a8f5762e7aa5d2164e5eba5c41d09f6197e55fecceead78e20476ee36df"),
        ):
            metadata[name] = self.inputs.bind(original / name, pin)
        manifest = json.loads(metadata["manifest.json"])
        require(manifest["status"] == "OBSERVED" and manifest["original_torch_cases"] == 27
                and manifest["native_qualification"] is False, "Bool reference must remain an observation")
        require(manifest["source_report"]["sha256"] == BOOL_REFERENCE
                and manifest["source_report"]["bytes"] == len(raw)
                and manifest["source_report"]["mode"] == 0o600, "Bool source report identity changed")
        require(manifest["producer"]["exit_code"] == 0
                and manifest["producer"]["complete_stdout_and_stderr_EOF"] is True,
                "Bool reference execution was incomplete")
        output = self.output / "references/bool27"
        output.mkdir(parents=True, exist_ok=False)
        report = output / "report.json"
        report.write_bytes(raw); report.chmod(0o600)
        (output / "report.json.xz").write_bytes(archive)
        for name, payload in metadata.items():
            (output / name).write_bytes(payload)
        self.inputs.bind(report, BOOL_REFERENCE)
        self.cache["bool27"] = report
        return report

    def shared(self):
        if "shared" in self.cache:
            return self.cache["shared"]
        archive = ROOT / SHARED_ARCHIVE
        raw = self.inputs.bind(archive, SHARED_ARCHIVE_SHA)
        manifest_raw = self.inputs.bind(archive.with_name("manifest.json"), SHARED_MANIFEST_SHA)
        manifest = json.loads(manifest_raw)
        require(manifest["schema"] == "heptatorch.current-runtime.fixed-references.v1"
                and manifest["archive"] == {"path": archive.name, "bytes": len(raw), "sha256": sha(raw)}
                and len(raw) == 536900, "fixed reference archive identity changed")
        require(manifest["file_count"] == len(manifest["files"]) == 113
                and manifest["payload_bytes"] == 47548385, "fixed reference archive denominator changed")
        output = self.output / "references/original-sum-inputs"
        output.mkdir(parents=True, exist_ok=False)
        seen, rows, directories = set(), [], []
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as packet:
            for member in packet:
                name = PurePosixPath(member.name)
                require(not name.is_absolute() and ".." not in name.parts and "\\" not in member.name
                        and str(name) == member.name and name.parts, "unsafe fixture archive path")
                require(member.name not in seen, "duplicate fixture archive member")
                require(member.isdir() or member.isfile(), "fixture archive contains a link or special file")
                seen.add(member.name)
                path = output.joinpath(*name.parts)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                    directories.append((path, member.mode))
                    continue
                payload = packet.extractfile(member).read()
                require(len(payload) == member.size, "incomplete fixture archive member")
                require(member.name in manifest["files"], "unexpected fixture archive member")
                pin = manifest["files"][member.name]
                require(len(payload) == pin["bytes"] and sha(payload) == pin["sha256"]
                        and member.mode == pin["mode"], "original fixture bytes or mode changed")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(payload)
                path.chmod(member.mode)
                self.inputs.bind(path, sha(payload))
                rows.append({"path": member.name, "bytes": len(payload), "mode": member.mode,
                             "sha256": sha(payload)})
        for path, mode in reversed(directories):
            path.chmod(mode)
        require({row["path"] for row in rows} == set(manifest["files"])
                and sum(row["bytes"] for row in rows) == manifest["payload_bytes"],
                "not every original fixture member was restored")
        dump(self.output / "references/original-sum-inputs.json",
             {"archive": SHARED_ARCHIVE, "sha256": SHARED_ARCHIVE_SHA,
              "manifest_sha256": SHARED_MANIFEST_SHA, "files": rows})
        self.cache["shared"] = output
        return output


    def baseline(self):
        if "baseline" not in self.cache:
            self.cache["baseline"] = fixed_fixtures.restore_baseline(
                ROOT, self.output / "references/frozen700", self.inputs)
        return self.cache["baseline"]

    def legacy(self):
        if "legacy_proof" not in self.cache:
            self.cache["legacy_proof"] = fixed_fixtures.verify_legacy_sources(ROOT, self.baseline(), self.inputs)
        return self.cache["legacy_proof"]

    def f32_reference(self):
        if "f32_reference" not in self.cache:
            root = fixed_fixtures.restore_f32_reference(ROOT, self.output / "references/f32_17", self.inputs)
            self.cache["f32_reference"] = root / "actual/report.json"
        return self.cache["f32_reference"]

    def f32_reader(self):
        if "f32_reader" not in self.cache:
            self.cache["f32_reader"] = fixed_fixtures.prepare_f32_reader(
                ROOT, self.output / "f32-reader", self.baseline(), self.inputs)
        return self.cache["f32_reader"]

    def current270(self):
        if "current270" not in self.cache:
            self.cache["current270"] = fixed_fixtures.restore_current270(
                ROOT, self.output / "original270-source", self.inputs)
        return self.cache["current270"]



def runner_command(name, build, build_sha, output, fixtures, inputs, torch_python):
    common = ["--build-directory", str(build), "--expected-build-sha256", build_sha]
    python = sys.executable
    flags = ["-I", "-S", "-B"]
    if name in ("bool27", "integral24"):
        script = ROOT / "tests/current_runtime_historical/run.py"
        for path in sorted(script.parent.glob("*.py")):
            inputs.bind(path)
        args = common + ["--suite", name, "--repository", str(ROOT),
            "--bool-reference", str(fixtures.bool27()),
            "--references-directory", str(fixtures.shared()),
            "--owner-directory", str(fixtures.f32_reader())]
    elif name == "original76":
        fixtures.legacy()
        script = ROOT / "tests/current_runtime_legacy/run.py"
        inputs.bind(ROOT / "validation/view-reshape-r4-native76/complete.tar.xz", LEGACY_ARCHIVE)
        args = common
    elif name == "sum32":
        fixtures.legacy()
        script = ROOT / "tests/current_runtime_legacy/sum32.py"
        source = fixtures.shared()
        args = common + ["--original-source", str(source / "original32-source"),
            "--parent-source", str(source / "parent-native32-source"),
            "--reference-source", str(source / "reference32-source"),
            "--reference", str(source / "reference32/actual.json")]
    elif name == "sum_f32_17":
        script = fixtures.f32_reader() / "native.py"
        flags = ["-E", "-S", "-B"]
        args = ["--repository", str(ROOT), "--binary", str(build / "heptatorch"),
            "--source", str(build / "runtime.hepta"), "--build-report", str(build / "build.json"),
            "--reference", str(fixtures.f32_reference()), "--expected-source-sha256", SOURCE,
            "--expected-native-sha256", IMAGE, "--expected-build-sha256", build_sha,
            "--expected-reference-sha256", "c86b0a9384c788eaf7765480b022416075a0e3aa38455f48dcbd7f40476cecb0"]
    elif name == "current35":
        script = ROOT / "tests/current_runtime35.py"
        args = common + ["--torch-python", str(torch_python)]
    elif name == "base270":
        script = fixtures.current270() / "run.py"
        inputs.bind(torch_python.resolve())
        python, flags = str(torch_python), ["-B"]
        args = common + ["--repository", str(ROOT), "--binary", str(build / "heptatorch"),
            "--source", str(build / "runtime.hepta"), "--expected-source-sha256", SOURCE,
            "--expected-native-sha256", IMAGE]
    else:
        raise ValueError("unknown mandatory suite")
    inputs.bind(script)
    return [python, *flags, str(script), *args, "--output", str(output)]





def qualify_legacy(report, build_sha):
    require(report["schema"] == "heptatorch.current-runtime.original76.v1" and report["status"] == "PASS",
            "original76 native report did not pass")
    require(report["expected_cases"] == report["cases"] == report["passed_cases"] == 76
            and report["failed_cases"] == 0, "all 76 original cases must pass")
    require(report["native_executed"] is True and report["all_inputs_unchanged"] is True
            and report["actual_build_sha256"] == build_sha, "original76 execution or build binding incomplete")
    suites = report["native_suites"]
    require([(r["name"], r["expected_cases"]) for r in suites] ==
            [("view26", 26), ("scalar32", 32), ("view14", 14), ("input4", 4)],
            "original suite order or count changed")
    require(all(r["status"] == "PASS" and r["qualified"] is True and r["exit_code"] == 0
            and r["stdout_EOF"] is True and r["stderr_EOF"] is True
            and r["cases"] == r["passed_cases"] == r["expected_cases"] for r in suites),
            "all original suites must finish with complete streams")
    require(report["old_pass_transferred"] is False and report["whole_pytorch_rewrite_complete"] is False,
            "old qualification must not transfer to this image or project")
    return {"cases": 76, "frames": report["frames"], "assertions": report["assertions"]}


def qualify_sum32(report, build_sha):
    require(report["schema"] == "heptatorch.current-runtime.original-sum32-comparison.v1"
            and report["status"] == "PASS", "original Sum32 native report did not pass")
    require(report["case_count"] == report["passed"] == len(report["cases"]) == 32
            and report["failed"] == report["blocked"] == 0, "all original Sum32 cases must pass")
    require(report["event_count"] == report["completed_reference_events"] == 626
            and [r["index"] for r in report["cases"]] == list(range(32))
            and all(r["status"] == "PASS" for r in report["cases"]), "all original Sum32 events are required")
    require(report["native_executed"] is True and report["Torch_executed"] is False
            and report["compiler_executed"] is False and report["all_inputs_unchanged"] is True
            and report["input_recheck_errors"] == [] and report["all_native_EOF"] is True,
            "Sum32 execution or input custody incomplete")
    require(report["actual_build_sha256"] == build_sha and report["reference_sha256"] == SUM32_REFERENCE,
            "Sum32 actual build or fixed reference binding changed")
    transports = report["transports"]
    require(len(transports) == 32 and all(t["exit_code"] == 0 and t["stdout_eof"] is True
            and t["stderr_eof"] is True and t["stderr_bytes"] == t["extra_stdout_bytes"] == 0
            and t["forced_cleanup"] is False and t["cleanup_errors"] == []
            and t["executable_sha256"] == IMAGE for t in transports), "all 32 Sum32 native EOFs are required")
    require(report["old_PASS_transferred"] is False and report["schema_closure"] is False
            and report["full_project_status"] == "HOLD", "finite Sum32 qualification must preserve project status")
    return {"cases": 32, "events": 626, "frames": report["frames"], "assertions": report["assertions"]}





def execute(command, directory, timeout_seconds):
    return execute_process(command, directory, timeout_seconds, cwd=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    require(not sys.flags.optimize, "all original assertions must remain enabled")
    build, output = args.build_directory.resolve(), args.output.resolve()
    require(output != build and not output.is_relative_to(build), "qualification output must be separate from its build")
    output.mkdir(parents=True, exist_ok=False)
    inputs, suites, failures = Inputs(), [], []
    started = time.monotonic()
    result = {"schema": "heptatorch.current-runtime.qualification.v1", "status": "RUNNING",
              "source_sha256": SOURCE, "native_sha256": IMAGE, "functions": 701, "units": 14,
              "required_suites": list(GROUPS), "required_cases": {"bool27": 27, "original76": 76,
              "sum32": 32, "integral24": 24, "sum_f32_17": 17, "current35": 35, "base270": 270}, "required_cases_total": 481,
              "required_controls": {"historical_Bool_dtypes": 7, "historical_Integral_dtypes": 5,
                  "current_dtypes": 7, "F32_axes": 4, "entry_and_capability": 10,
                  "Bool_allocation": 1, "F32_allocation_stages": 2,
                  "original35_copied_positive": 4, "original35_copied_negative": 18,
                  "original270_actual_prefix_refusal_pairs": 14},
              "suite_timeout_seconds": SUITE_TIMEOUT_SECONDS,
              "suites": suites, "failures": failures, "full_pytorch_rewrite_complete": False,
              "schema_closure": False, "full_project_status": "HOLD", "Torch_executed": False}

    def save():
        dump(output / "qualification.json", result)

    try:
        inputs.bind(Path(__file__).resolve())
        for name in ("current_runtime_fixtures.py", "current_runtime_process.py", "current_runtime_admission.py",
                     "current_runtime_torch.py"):
            inputs.bind(ROOT / "tests" / name)
        build_sha = build_binding(build, inputs)
        result["actual_build_sha256"] = build_sha
        fixtures = Fixtures(output, inputs)
        for name in GROUPS:
            execution = output / (name + "-execution")
            execution.mkdir()
            row = {"name": name, "status": "PREPARING", "qualified": False}
            suites.append(row); save()
            print(json.dumps({"suite": name, "status": row["status"]}), flush=True)
            try:
                directory = output / name
                command = runner_command(name, build, build_sha, directory, fixtures, inputs, args.torch_python.absolute())
                receipt = execute(command, execution, SUITE_TIMEOUT_SECONDS[name])
                row["runner"] = receipt
                report = None
                if (directory / "report.json").is_file():
                    raw = inputs.bind(directory / "report.json")
                    report = json.loads(raw)
                    row["original_report"] = {"path": str((directory / "report.json").relative_to(output)),
                                              "bytes": len(raw), "sha256": sha(raw)}
                    row["original_report_status"] = report.get("status")
                    if name in ("current35", "base270") and result["Torch_executed"] is not True:
                        result["Torch_executed"] = report.get("Torch_executed")
                require(receipt["started"] is True and receipt["exit_code"] == 0
                        and receipt["stdout_EOF"] is True and receipt["stderr_EOF"] is True
                        and receipt["stderr_bytes"] == 0 and not receipt.get("timed_out")
                        and not receipt.get("forced_cleanup") and not receipt.get("cleanup_errors")
                        and not receipt.get("running_descendants_after_cleanup"),
                        "native suite runner failed or did not finish; report=" + str(row.get("original_report_status")))
                require(report is not None, "native suite produced no original report")
                require(report["source_sha256"] == SOURCE and report["native_sha256"] == IMAGE,
                        "suite source or native image identity changed")
                if name in ("bool27", "integral24"):
                    counts = qualify_historical(report, name, ROOT, build, build_sha, directory, inputs)
                elif name == "original76":
                    counts = qualify_legacy(report, build_sha)
                elif name == "sum32":
                    counts = qualify_sum32(report, build_sha)
                elif name == "sum_f32_17":
                    counts = qualify_f32(report, ROOT, build, build_sha, directory, inputs)
                elif name == "current35":
                    counts = qualify_current35(report, build_sha, directory, inputs)
                else:
                    counts = qualify_current270(report, build_sha, directory, inputs)
                row.update(status="PASS", qualified=True, **counts)
            except Exception as error:
                row.update(status="FAIL", error={"type": type(error).__name__, "message": str(error),
                                                "traceback": traceback.format_exc()})
                failures.append(name + ": " + str(error))
            save()
            print(json.dumps({"suite": name, "status": row["status"], "qualified": row["qualified"]}), flush=True)
    except Exception as error:
        failures.append("build/source admission: " + repr(error))
        result["admission_error"] = traceback.format_exc()
    finally:
        errors = inputs.recheck()
        if errors:
            failures.append("bound inputs changed during qualification")
        complete = len(suites) == len(GROUPS) and [row["name"] for row in suites] == list(GROUPS)
        passed = complete and all(row["qualified"] is True and row["status"] == "PASS" for row in suites) and not failures
        result.update(status="PASS_CURRENT_RUNTIME_FOCUSED" if passed else "FAIL",
                      all_suites_executed=complete and all(row.get("runner", {}).get("started") is True for row in suites),
                      all_inputs_unchanged=not errors, input_recheck_errors=errors,
                      checked_inputs=inputs.records, elapsed_seconds=time.monotonic() - started)
        save()
    print(json.dumps({"status": result["status"], "suites": {r["name"]: r["status"] for r in suites},
                      "failures": failures, "qualification": str(output / "qualification.json")}), flush=True)
    return 0 if result["status"] == "PASS_CURRENT_RUNTIME_FOCUSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
