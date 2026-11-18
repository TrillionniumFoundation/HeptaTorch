#!/usr/bin/env python3
"""Execute all focused current-runtime suites and retain their original evidence.

This coordinator transports fixed observations and invokes independent native
readers. It does not implement Tensor arithmetic, run Torch, edit an old report,
or grant whole-project qualification from these finite suites.
"""

import argparse
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import selectors
import signal
import stat
import subprocess
import sys
import tarfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4"
IMAGE = "c6afeedc14ba8cc408a62d6fbef0abb5b78dff0039b8099f3449d813e290a90e"
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
GROUPS = ("bool27", "original76", "integral24", "sum32")
SUITE_TIMEOUT_SECONDS = {"bool27": 600, "original76": 900, "integral24": 900, "sum32": 3600}


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
    require(built["functions"] == built["complete_core_bodies_matched"] == 700
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



def runner_command(name, build, build_sha, output, fixtures, inputs):
    common = ["--build-directory", str(build), "--expected-build-sha256", build_sha]
    if name == "bool27":
        script = ROOT / "tests/sum_bool_review/native.py"
        for path in sorted(script.parent.glob("*.py")):
            inputs.bind(path)
        args = ["--repository", str(ROOT), "--binary", str(build / "heptatorch"),
                "--source", str(build / "runtime.hepta"), "--build-report", str(build / "build.json"),
                "--reference", str(fixtures.bool27()), "--expected-source-sha256", SOURCE,
                "--expected-native-sha256", IMAGE, "--expected-build-sha256", build_sha,
                "--expected-reference-sha256", BOOL_REFERENCE]
        flags = ["-E", "-S", "-B"]  # Preserve verified sibling imports; disable site and environment inputs.
    elif name == "original76":
        script = ROOT / "tests/current_runtime_legacy/run.py"
        inputs.bind(ROOT / "validation/view-reshape-r4-native76/complete.tar.xz", LEGACY_ARCHIVE)
        args, flags = common, ["-I", "-S", "-B"]
    elif name == "sum32":
        script = ROOT / "tests/current_runtime_legacy/sum32.py"
        source = fixtures.shared()
        args = common + ["--original-source", str(source / "original32-source"),
                         "--parent-source", str(source / "parent-native32-source"),
                         "--reference-source", str(source / "reference32-source"),
                         "--reference", str(source / "reference32/actual.json")]
        flags = ["-I", "-S", "-B"]
    elif name == "integral24":
        script = ROOT / "tests/current_runtime_integral24/run.py"
        source = fixtures.shared()
        args = common + ["--reference", str(source / "reference24/actual.json"),
                         "--reference24-source", str(source / "reference24-source"),
                         "--reference32-source", str(source / "reference32-source"),
                         "--integral-source", str(source / "integral-source"),
                         "--parent-native-package", str(source / "parent-native32-source"),
                         "--pip-report", str(source / "reference-environment/build/pytorch-install.json"),
                         "--upstream-lock", str(source / "reference-environment/compatibility/upstream-lock.json")]
        flags = ["-E", "-S", "-B"]
    else:
        raise ValueError("unknown mandatory suite")
    inputs.bind(script)
    return [sys.executable, *flags, str(script), *args, "--output", str(output)]


def qualify_bool(report, build, build_sha):
    require(report["schema"] == "heptatorch.sum-bool.native27.v1" and report["status"] == "PASS",
            "Bool27 native report did not pass")
    require(report["expected_cases"] == report["passed"] == len(report["cases"]) == 27
            and report["failed"] == 0, "all 27 Bool cases must pass")
    require(len(report["dtype_controls"]) == 7 and len(report["entry_controls"]) == 1
            and len(report["allocation_controls"]) == 1, "all Bool control sessions are required")
    rows = report["cases"] + report["dtype_controls"] + report["entry_controls"] + report["allocation_controls"]
    require(len(rows) == 36 and all(r["status"] == "PASS" and r["exit_code"] == 0 and r["frames"] > 0
            and r["checks"] and all(c["passed"] is True for c in r["checks"]) for r in rows),
            "every Bool native session must execute and pass")
    require([r["name"] for r in report["dtype_controls"]] ==
            ["Bool_explicit_Bool", "I64_explicit_Bool", "Bool_explicit_F32", "I64_explicit_F32",
             "F32_explicit_I64", "F32_explicit_Bool", "wrapped_F64_explicit_Bool"],
            "all seven distinct dtype controls are required")
    entry = report["entry_controls"][0]["operations"]
    require([r["label"] for r in entry] ==
            ["invalid_public_capability", "invalid_keepdim", "unsupported_dtype_code",
             "reserved_word_six", "reserved_word_seven", "truncated_axis_payload",
             "None_nonzero_payload", "empty_axes_nonzero_payload", "axis_payload_before_payload_region",
             "stale-public-capability"] and [r["normalized_status"] for r in entry] == [1, 7, 7, 7, 7, 2, 7, 7, 2, 1],
            "all ten distinct entry and capability refusals are required")
    allocation = report["allocation_controls"][0]["operations"]
    require([r["label"] for r in allocation] ==
            ["Bool-result-Storage-allocation-failure", "same-Bool-operation-after-resource-recovery"]
            and [r["normalized_status"] for r in allocation] == [3, 0],
            "actual allocation failure and same-operation recovery are required")
    require(report["native_executed"] is True and report["Torch_executed"] is False
            and report["inputs_unchanged"] is True, "Bool execution or input custody incomplete")
    require(report["reference_sha256"] == BOOL_REFERENCE
            and report["inputs"][str(build / "build.json")]["sha256"] == build_sha,
            "Bool reference or actual build binding changed")
    require(report["schema_closure"] is False and report["full_project_status"] == "HOLD",
            "finite Bool qualification must preserve project status")
    return {"cases": 27, "native_sessions": 36, "frames": sum(r["frames"] for r in rows),
            "dtype_controls": 7, "entry_controls": 10, "allocation_controls": 1}


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


def qualify_integral24(report, build_sha):
    require(report["schema"] == "heptatorch.current-runtime.integral24.v1" and report["status"] == "PASS",
            "original Integral24 native comparisons did not pass")
    require(report["expected_cases"] == report["passed"] == len(report["cases"]) == 24
            and report["failed"] == report["blocked"] == 0
            and all(row["status"] == "PASS" for row in report["cases"]),
            "all 24 original Integral cases must pass")
    require(report["completed_reference_events"] == report["original_event_count"] == 260,
            "all original Integral24 reference events are required")
    require(report["native_executed"] is True and report["Torch_executed"] is False
            and report["compiler_executed"] is False and report["all_inputs_unchanged"] is True
            and report["original_input_recheck_errors"] == [], "Integral24 execution or input custody incomplete")
    require(report["actual_build_sha256"] == build_sha and report["reference_sha256"] == INTEGRAL_REFERENCE,
            "Integral24 actual build or fixed reference binding changed")
    require(report["original_controls_contract_changed"] == 2
            and report["original_controls_changed_indices"] == [0, 2]
            and report["retained_refusals"] == 3 and report["retained_refusal_indices"] == [1, 3, 4]
            and report["original_five_all_pass"] is False
            and report["current_dtype_contract_qualified_here"] is False,
            "the two explicitly extended APIs must not relabel the five original controls")
    historical = report["historical_controls"]
    require(historical["status"] == "FAIL" and historical["case_count"] == 5
            and historical["passed"] == 3 and historical["failed"] == 2
            and [row["index"] for row in historical["cases"]] == list(range(5)),
            "all five original control results must remain recorded")
    for index, row in enumerate(historical["cases"]):
        require(row["exit_code"] == 0, "historical control native child failed")
        if index in (0, 2):
            require(row["status"] == "FAIL" and row["observations"]["frame_status"] == 0
                    and row["observations"]["record_count"] == 1 and row["checks"] ==
                    [{"label": "native-cause7", "actual": 0, "expected": 7, "check": False}]
                    and row["observations"]["command"][0] == 115
                    and row["observations"]["command"][5] == 1,
                    "only the original explicit-Bool refusal expectation may differ")
            require(row["error"]["type"] == "AssertionError" and "EOF_failure" not in row["error"],
                    "unexpected historical control failure")
        else:
            require(row["status"] == "PASS" and row["error"] is None
                    and row["observations"]["frame_status"] == 7 and row["observations"]["record_count"] == 0
                    and row["checks"] and all(c["check"] is True for c in row["checks"]),
                    "the other three original dtype refusals must still pass")
    transport = report["transport"]
    require(transport["children"] == len(transport["rows"]) == 29
            and transport["all_exit_zero"] is True and transport["all_stdout_eof"] is True
            and transport["all_stderr_eof"] is True and transport["complete_owner_boundaries"] is True
            and all(row["exit_code"] == 0 and row["stdout_eof"] is True and row["stderr_eof"] is True
                    and row["frames"] > 0 for row in transport["rows"]), "all Integral24/control native EOFs are required")
    require(report["old_native_pass_transferred"] is False and report["schema_closure"] is False
            and report["full_transition_recomputed"] is False and report["full_project_status"] == "HOLD",
            "finite Integral24 qualification must preserve project status")
    return {"cases": 24, "events": 260, "native_sessions": 29, "frames": transport["frames"],
            "historical_controls_status": "FAIL_ORIGINAL_CONTRACT", "retained_refusals": 3,
            "extended_Bool_control_indices": [0, 2], "current_dtype_qualification": "bool27/dtype_controls"}


def execute(command, directory, timeout_seconds):
    started = time.monotonic()
    receipt = {"argv": command, "timeout_seconds": timeout_seconds,
               "started": False, "stdout_EOF": False, "stderr_EOF": False}
    child = None
    selector = selectors.DefaultSelector()
    hashes = {name: hashlib.sha256() for name in ("stdout", "stderr")}
    sizes = {name: 0 for name in hashes}

    def stop():
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        with (directory / "runner.stdout").open("xb") as stdout, (directory / "runner.stderr").open("xb") as stderr:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, cwd=ROOT, start_new_session=True,
                                     env={"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
            receipt["started"] = True
            for name, pipe, destination, console in (("stdout", child.stdout, stdout, sys.stdout),
                                                      ("stderr", child.stderr, stderr, sys.stderr)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, (name, destination, console))
            deadline = started + timeout_seconds
            while selector.get_map():
                if time.monotonic() >= deadline:
                    if receipt.get("timed_out"):
                        raise TimeoutError("runner streams did not reach EOF after group termination")
                    receipt["timed_out"] = True
                    stop()
                    deadline = time.monotonic() + 10
                for key, _ in selector.select(timeout=max(0, min(1, deadline - time.monotonic()))):
                    name, destination, console = key.data
                    try:
                        raw = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not raw:
                        receipt[name + "_EOF"] = True
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    destination.write(raw); destination.flush()
                    hashes[name].update(raw); sizes[name] += len(raw)
                    if hasattr(console, "buffer"):
                        console.buffer.write(raw); console.buffer.flush()
                    else:
                        console.write(raw.decode("utf-8", errors="replace")); console.flush()
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                stop()
                child.wait(timeout=10)
            receipt["exit_code"] = child.returncode
    except BaseException as error:
        receipt["execution_error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if child is not None and child.poll() is None:
            stop()
            child.wait(timeout=10)
            receipt.update(forced_cleanup=True, exit_code=child.returncode)
        if child is not None:
            for pipe in (child.stdout, child.stderr):
                if pipe is not None:
                    pipe.close()
        selector.close()
        receipt.update(elapsed_seconds=time.monotonic() - started,
                       stdout_bytes=sizes["stdout"], stdout_sha256=hashes["stdout"].hexdigest(),
                       stderr_bytes=sizes["stderr"], stderr_sha256=hashes["stderr"].hexdigest())
        dump(directory / "runner.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not sys.flags.optimize, "all original assertions must remain enabled")
    build, output = args.build_directory.resolve(), args.output.resolve()
    require(output != build and not output.is_relative_to(build), "qualification output must be separate from its build")
    output.mkdir(parents=True, exist_ok=False)
    inputs, suites, failures = Inputs(), [], []
    started = time.monotonic()
    result = {"schema": "heptatorch.current-runtime.qualification.v1", "status": "RUNNING",
              "source_sha256": SOURCE, "native_sha256": IMAGE, "functions": 700, "units": 14,
              "required_suites": list(GROUPS), "required_cases": {"bool27": 27, "original76": 76,
              "sum32": 32, "integral24": 24}, "required_cases_total": 159,
              "required_bool_controls": {"dtypes": 7, "entry_and_capability": 10, "allocation": 1},
              "suite_timeout_seconds": SUITE_TIMEOUT_SECONDS,
              "suites": suites, "failures": failures, "full_pytorch_rewrite_complete": False,
              "schema_closure": False, "full_project_status": "HOLD", "Torch_executed": False}

    def save():
        dump(output / "qualification.json", result)

    try:
        inputs.bind(Path(__file__).resolve())
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
                command = runner_command(name, build, build_sha, directory, fixtures, inputs)
                receipt = execute(command, execution, SUITE_TIMEOUT_SECONDS[name])
                row["runner"] = receipt
                report = None
                if (directory / "report.json").is_file():
                    raw = inputs.bind(directory / "report.json")
                    report = json.loads(raw)
                    row["original_report"] = {"path": str((directory / "report.json").relative_to(output)),
                                              "bytes": len(raw), "sha256": sha(raw)}
                    row["original_report_status"] = report.get("status")
                require(receipt["started"] is True and receipt["exit_code"] == 0
                        and receipt["stdout_EOF"] is True and receipt["stderr_EOF"] is True
                        and receipt["stderr_bytes"] == 0 and not receipt.get("timed_out")
                        and not receipt.get("forced_cleanup"),
                        "native suite runner failed or did not finish; report=" + str(row.get("original_report_status")))
                require(report is not None, "native suite produced no original report")
                require(report["source_sha256"] == SOURCE and report["native_sha256"] == IMAGE,
                        "suite source or native image identity changed")
                if name == "bool27":
                    counts = qualify_bool(report, build, build_sha)
                elif name == "original76":
                    counts = qualify_legacy(report, build_sha)
                elif name == "sum32":
                    counts = qualify_sum32(report, build_sha)
                else:
                    counts = qualify_integral24(report, build_sha)
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
