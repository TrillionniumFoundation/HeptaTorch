#!/usr/bin/env python3
"""Execute all focused current-runtime suites and retain their original evidence.

This coordinator transports fixed observations and invokes independent native
readers and the fixed live Torch references for original35, base270, live32,
and the original Generator24 subset. Generator131 also builds and executes
separate current-object ABI and reordered-link probe images. It implements no
Tensor arithmetic, edits no old report, and grants no whole-project qualification
from these finite suites.
"""

import argparse
import hashlib
import io
import importlib.util
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
import current_runtime_generator_fixtures as generator_fixtures
import current_runtime_public_f64_logical192 as public_f64_logical192
import current_runtime_half_fixtures as half_fixtures
import current_runtime_f64_fixtures as f64_fixtures
from current_runtime_half_readback import qualify_half47, qualify_half12
from half_logical import reference as half_logical_reference
from half_logical.run import qualify as qualify_half_logical
from numeric_predicates.run import qualify as qualify_numeric_predicates
from bf16_copy.run import qualify as qualify_bf16_copy
from half_mul.run import qualify as qualify_half_mul, read_reference as read_half_mul_reference
from half_neg.run import qualify as qualify_half_neg, read_reference as read_half_neg_reference
from current_runtime_f64_readback import qualify_f64_48
from current_runtime_generator131 import qualify_generator131
from current_runtime_process import execute as execute_process
from current_runtime_admission import qualify_historical, qualify_f32, qualify_current35, qualify_current270
from current_runtime_live32 import qualify_live32
_asstrided_path = ROOT / "tests/current_runtime_asstrided/qualify.py"
_asstrided_spec = importlib.util.spec_from_file_location("current_runtime_asstrided_admission", _asstrided_path)
current_asstrided = importlib.util.module_from_spec(_asstrided_spec)
_asstrided_spec.loader.exec_module(current_asstrided)
qualify_asstrided16 = current_asstrided.qualify_asstrided16
_cat4_path = ROOT / "tests/current_runtime_cat4/qualify.py"
_cat4_spec = importlib.util.spec_from_file_location("current_runtime_cat4_admission", _cat4_path)
current_cat4 = importlib.util.module_from_spec(_cat4_spec)
_cat4_spec.loader.exec_module(current_cat4)
qualify_cat4 = current_cat4.qualify_cat4
_metadata10_path = ROOT / "tests/current_runtime_metadata10/qualify.py"
_metadata10_spec = importlib.util.spec_from_file_location("current_runtime_metadata10_admission", _metadata10_path)
current_metadata10 = importlib.util.module_from_spec(_metadata10_spec)
_metadata10_spec.loader.exec_module(current_metadata10)
qualify_metadata10 = current_metadata10.qualify_metadata10
_profile_path = ROOT / "tools/current_runtime/qualification.py"
_profile_spec = importlib.util.spec_from_file_location("current_runtime_build_admission", _profile_path)
current_profile = importlib.util.module_from_spec(_profile_spec)
_profile_spec.loader.exec_module(current_profile)
SOURCE, IMAGE = current_profile.SOURCE, current_profile.IMAGE
FUNCTIONS, UNITS = current_profile.FUNCTIONS, current_profile.UNITS
COMPILER = "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"
LINKER = "761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e"
FUSION_RESOURCE_SHA = 'e48c007916f64eb22e7b884d2d4d861e18fd488d2914c7fb3e4b7fe824186c1e'
BOOL_REFERENCE = "21dd75aeb86414818f7aa381f8b866d78b16f9bb3f5738e9a60d27ac3d15dd28"
BOOL_ARCHIVE = "353a8b2853773bc6678bb62fc9559d6f4244c571ba40c6b80f352b2649cd6edd"
SUM32_REFERENCE = "c14e4b18f657429ad5dac5498c3abcae303b6a7e7550db7d15a3c9179bdcd728"
INTEGRAL_REFERENCE = "6c9df4b579664c89c200b9d796367dd6d41f93f8260e8ff012f17d5041d2278b"
SHARED_ARCHIVE = "validation/current-runtime-references/complete.tar.xz"
SHARED_ARCHIVE_SHA = "3a29420768a1bf7808976f04cb3ce3e6091db55ff6ae7858879f1b7ff70b69ea"
SHARED_MANIFEST_SHA = "666fa775586679cfd42606c95656ad261c8856332431931ed4b0fcfd8865057e"
GROUPS = ("bool27", "original76", "integral24", "sum_f32_17", "current35", "base270", "sum32", "live32", "logical147", "resource7", "asstrided16", "cat4", "metadata10", "generator131", "public_f64_logical192", "f64_48", "half47", "half_logical264", "half_transactions12", "numeric_predicates", "bf16_copy", "half_mul", "half_neg")
SUITE_TIMEOUT_SECONDS = {"bool27": 600, "original76": 900, "integral24": 900,
                         "sum_f32_17": 900, "current35": 1800, "base270": 1800, "sum32": 3600,
                         "live32": 1800, "logical147": 1200, "resource7": 1200, "asstrided16": 1200, "cat4": 600, "metadata10": 1200, "generator131": 600, "public_f64_logical192": 1200,
                         "f64_48": 1200, "half47": 1200, "half_logical264": 1200, "half_transactions12": 600,
                         "numeric_predicates": 1200, "bf16_copy": 1200, "half_mul": 1200, "half_neg": 1200}


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
    current_profile.admit_build(directory, inputs.bind)
    return sha(inputs.bind(directory / "build.json"))


def cat_source_binding(directory, inputs):
    """Admit exact761 and three reviewed cat bodies; transfer no old result."""
    outer_path = ROOT / 'tools/current_runtime/fusion_source_transition.py'
    inputs.bind(outer_path, '992e666ca21e86bc2a95bdb2327aa2bf27f4b57c0208f1abe13ae48348e6c1d2')
    outer_spec = importlib.util.spec_from_file_location('current_runtime761_source_transition', outer_path)
    outer = importlib.util.module_from_spec(outer_spec)
    outer_spec.loader.exec_module(outer)
    historical700, outer_proof = outer.restore_source(inputs.bind(directory / 'runtime.hepta', SOURCE), inputs.bind)
    path = ROOT / 'tools/current_runtime/cat_source_transition.py'
    inputs.bind(path)
    spec = importlib.util.spec_from_file_location('current_runtime_cat_source_transition', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    restored, proof = module.restore_source(historical700, inputs.bind)
    require(module.SOURCE == outer.PARENT_SOURCE == sha(historical700)
            and FUNCTIONS == 761 and UNITS == 15 and sha(restored) == module.PARENT_SOURCE,
            'actual761 profile, exact historical700 and exact pre-cat source')
    return dict(schema='heptatorch.current761-cat.complete-source-inverse.v1',
        source_sha256=SOURCE, parent_source_sha256=module.PARENT_SOURCE,
        functions=761, historical_functions=700, outer_source_inverse=outer_proof,
        cat_transition=proof, runtime_qualification_transferred=False)

class Fixtures:
    def __init__(self, output, inputs, torch_python):
        self.output, self.inputs = output, inputs
        self.torch_python = Path(torch_python).absolute()
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

    def reference76(self):
        if "reference76" not in self.cache:
            self.cache["reference76"] = fixed_fixtures.restore_original76(
                ROOT, self.output / "references/original76", self.inputs)
        return self.cache["reference76"]

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


    def generator(self):
        if "generator" not in self.cache:
            self.cache["generator"] = {
                "original26": generator_fixtures.restore_original26(
                    ROOT, self.output / "references/generator-original26", self.inputs),
                "candidate": generator_fixtures.restore_candidate(
                    ROOT, self.output / "references/generator-candidate", self.inputs),
                "package": generator_fixtures.restore_qualifier(
                    ROOT, self.output / "generator131-source", self.inputs),
            }
        return self.cache["generator"]


    def public_f64_logical192(self):
        if "public_f64_logical192" not in self.cache:
            self.cache["public_f64_logical192"] = public_f64_logical192.restore_reference(
                ROOT, self.output / "references/public-f64-logical192", self.inputs)
        return self.cache["public_f64_logical192"]


    def half_source(self):
        if "half_source" not in self.cache:
            self.cache["half_source"] = half_fixtures.restore_source(
                ROOT, self.output / "half47-source", self.inputs)
        return self.cache["half_source"]

    def half47(self):
        if "half47" not in self.cache:
            self.cache["half47"] = half_fixtures.restore_reference(
                ROOT, self.output / "references/half47", self.half_source(), self.inputs)
        return self.cache["half47"]

    def f64_48(self):
        if "f64_48" not in self.cache:
            source = f64_fixtures.restore_source(ROOT, self.output / "f64-48-source", self.inputs)
            self.cache["f64_48"] = f64_fixtures.restore_reference(
                ROOT, self.output / "references/f64-48", source, self.inputs)
        return self.cache["f64_48"]

    def half_logical(self):
        if "half_logical" not in self.cache:
            self.cache["half_logical"] = half_logical_reference.restore(
                ROOT, self.output / "references/half-logical264", self.inputs.bind)
        return self.cache["half_logical"]

    def half_mul(self):
        """Produce one fresh Torch observation shared by the exact22 transition and312 suite."""
        if "half_mul" not in self.cache:
            reference = self.output / "references/half-mul312"
            execution = self.output / "references/half-mul312-execution"
            execution.mkdir(parents=True, exist_ok=False)
            self.inputs.bind(self.torch_python.resolve(strict=True))
            script = ROOT / "tests/half_mul/reference.py"
            self.inputs.bind(script)
            process = execute_process([str(self.torch_python), "-B", str(script),
                "--output", str(reference)], execution, 300, cwd=ROOT)
            dump(execution / "process.json", process)
            observed = read_half_mul_reference(reference, execution, self.inputs)
            require(Path(observed["report"]["actual_process"]["python"]["path"])
                    == self.torch_python.resolve(strict=True),
                    "the shared multiplication reference used the selected physical Torch interpreter")
            self.cache["half_mul"] = {"root": reference, "execution": execution}
        return self.cache["half_mul"]

    def half_neg(self):
        """Produce one fresh Torch observation shared by the exact2 transition and371 suite."""
        if "half_neg" not in self.cache:
            reference = self.output / "references/half-neg371"
            execution = self.output / "references/half-neg371-execution"
            execution.mkdir(parents=True, exist_ok=False)
            self.inputs.bind(self.torch_python.resolve(strict=True))
            script = ROOT / "tests/half_neg/reference.py"
            self.inputs.bind(script)
            process = execute_process([str(self.torch_python), "-B", str(script),
                "--output", str(reference)], execution, 300, cwd=ROOT)
            dump(execution / "process.json", process)
            observed = read_half_neg_reference(reference, execution, self.inputs)
            require(Path(observed["report"]["actual_process"]["python"]["path"])
                    == self.torch_python.resolve(strict=True),
                    "the shared negation reference used the selected physical Torch interpreter")
            self.cache["half_neg"] = {"root": reference, "execution": execution}
        return self.cache["half_neg"]


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
        args = common + ["--references", str(fixtures.reference76())]
    elif name == "sum32":
        fixtures.legacy()
        script = ROOT / "tests/current_runtime_legacy/sum32.py"
        source = fixtures.shared()
        args = common + ["--original-source", str(source / "original32-source"),
            "--parent-source", str(source / "parent-native32-source"),
            "--reference-source", str(source / "reference32-source"),
            "--reference", str(source / "reference32/actual.json")]
    elif name == "sum_f32_17":
        script = ROOT / "tests/sum_f32_review/fusion.py"
        reader = fixtures.f32_reader()
        for path in sorted(script.parent.glob("*.py")):
            inputs.bind(path)
        flags = ["-E", "-S", "-B"]
        args = ["--repository", str(ROOT), "--binary", str(build / "heptatorch"),
            "--source", str(build / "runtime.hepta"), "--build-report", str(build / "build.json"),
            "--reference", str(fixtures.f32_reference()), "--expected-source-sha256", SOURCE,
            "--expected-native-sha256", IMAGE, "--expected-build-sha256", build_sha,
            "--expected-reference-sha256", "c86b0a9384c788eaf7765480b022416075a0e3aa38455f48dcbd7f40476cecb0",
            "--reader-directory", str(reader)]
    elif name == "current35":
        script = ROOT / "tests/current_runtime35.py"
        args = common + ["--torch-python", str(torch_python)]
    elif name == "base270":
        script = ROOT / "tests/current_runtime270.py"
        inputs.bind(torch_python.resolve())
        python, flags = str(torch_python), ["-B"]
        args = common
    elif name == "live32":
        script = ROOT / "tests/current_runtime_live32.py"
        args = common + ["--torch-python", str(torch_python)]
    elif name == "logical147":
        script = ROOT / "tests/logical_reductions/qualify.py"
        for path in sorted(script.parent.glob("*.py")):
            inputs.bind(path)
        args = ["--build-directory", str(build)]
    elif name == "resource7":
        script = ROOT / "tests/current_runtime_resources/qualify.py"
        for path in sorted(script.parent.iterdir()):
            if path.is_file():
                inputs.bind(path)
        args = common + ["--repository", str(ROOT)]
    elif name == "asstrided16":
        script = ROOT / "tests/current_runtime_asstrided/qualify.py"
        for path in sorted(script.parent.iterdir()):
            if path.is_file():
                inputs.bind(path)
        args = common + ["--repository", str(ROOT)]
    elif name == "cat4":
        script = ROOT / "tests/current_runtime_cat4/run.py"
        for path in sorted(script.parent.iterdir()):
            if path.is_file():
                inputs.bind(path)
        args = common + ["--repository", str(ROOT)]
    elif name == "metadata10":
        script = ROOT / "tests/current_runtime_metadata10/worker.py"
        for path in sorted(script.parent.rglob("*")):
            if path.is_file():
                inputs.bind(path)
        args = common + ["--repository", str(ROOT)]
    elif name == "generator131":
        sources = fixtures.generator()
        script = sources["package"] / "run.py"
        tool_build, stack_build = fixtures.cache["generator_tool_paths"]
        inputs.bind(torch_python.resolve())
        python, flags = str(torch_python), ["-B"]
        tool_sha = sha(inputs.bind(tool_build / "build.json"))
        args = common + ["--repository", str(ROOT), "--candidate", str(sources["candidate"]),
            "--original-package-root", str(sources["original26"]),
            "--tool-build", str(tool_build), "--stack-build", str(stack_build),
            "--expected-native-sha256", IMAGE, "--expected-tool-build-sha256", tool_sha,
            "--group", "all"]
    elif name == "public_f64_logical192":
        source = fixtures.public_f64_logical192()
        script = ROOT / "tests/public_f64_logical/qualify736.py"
        for filename, expected in {**public_f64_logical192.READER_FILES, **public_f64_logical192.WRAPPER_FILES}.items():
            inputs.bind(script.parent / filename, expected)
        args = public_f64_logical192.reader_cli(ROOT, build, build_sha, source)
    elif name == "f64_48":
        source = fixtures.f64_48()
        script = ROOT / "tests/current_runtime_f64_fixtures.py"
        args = f64_fixtures.reader_cli(ROOT, build, build_sha, source)
    elif name == "half47":
        source = fixtures.half47()
        script = ROOT / "tests/current_runtime_half_fixtures.py"
        args = half_fixtures.reader_cli(ROOT, build, build_sha, source, fixtures.half_mul(), fixtures.half_neg())
    elif name in ("half_logical264", "half_transactions12"):
        script = ROOT / "tests/half_logical/run.py"
        for path in sorted(script.parent.glob("*.py")):
            inputs.bind(path)
        source = fixtures.half_logical()
        args = common + ["--group", name, "--reference-directory", str(source["root"])]
    elif name in ("numeric_predicates", "bf16_copy", "half_mul", "half_neg"):
        script = ROOT / "tests" / name / "run.py"
        for path in sorted(script.parent.glob("*.py")):
            inputs.bind(path)
        inputs.bind(torch_python.resolve())
        args = common + ["--torch-python", str(torch_python.absolute())]
        if name in ("half_mul", "half_neg"):
            reference = fixtures.half_mul() if name == "half_mul" else fixtures.half_neg()
            args += ["--reference-directory", str(reference["root"]),
                     "--reference-execution", str(reference["execution"])]
    else:
        raise ValueError("unknown mandatory suite")
    inputs.bind(script)
    # Bind the selected entity before execution while retaining argv[0],
    # including the original Torch virtual-environment invocation path.
    inputs.bind(Path(python).resolve(strict=True))
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


def qualify_logical(report, build_sha, directory, inputs):
    require(report['schema'] == 'heptatorch.logical-reductions.qualification.v1'
            and report['status'] == 'PASS' and report['actual_build_sha256'] == build_sha
            and report['source_sha256'] == SOURCE and report['native_sha256'] == IMAGE,
            'logical reductions must qualify the same complete current build')
    require(report['cases'] == 147 and report['packet_controls'] == 18
            and report['allocation_controls'] == 2 and report['native_children'] == 150
            and report['native_executed'] is True and report['torch_executed'] is False
            and report['all_native_EOF'] is True and report['all_inputs_unchanged'] is True,
            'all logical cases, packet controls, real allocation failures and children are required')
    runner = report['runner']
    require(runner['started'] is True and runner['exit_code'] == 0
            and runner['stdout_EOF'] is True and runner['stderr_EOF'] is True
            and runner['stderr_bytes'] == 0 and not runner.get('timed_out')
            and not runner.get('forced_cleanup'), 'complete actual logical runner process')
    raw = inputs.bind(directory / 'native/report.json', report['native_report_sha256'])
    native = json.loads(raw)
    require(native['status'] == 'PASS' and native['cases'] == native['expected_cases'] == 147
            and len(native['rows']) == 147 and all(row['status'] == 'PASS' for row in native['rows'])
            and native['native_sha256'] == IMAGE and native['source_sha256'] == SOURCE,
            'all actual native logical comparisons')
    reference = inputs.bind(directory / 'reference.json', report['reference_sha256'])
    require(json.loads(reference)['status'] == 'OBSERVED' and report['full_pytorch_rewrite_complete'] is False
            and report['schema_closure'] is False, 'independent logical reference and finite scope')
    return dict(cases=147, frames=report['frames'], assertions=report['total_checks'],
                native_sessions=150, packet_controls=18, allocation_controls=2)


def qualify_resource7(report, build_sha, directory, inputs):
    expected = {
        'r4': ['entry_guards_and_relative_tokens', 'saved_scalar_and_temporary_lifetime',
               'result_allocation_oom_recovery'],
        'backward': ['backward_private_preserve_allocation_oom_recovery'],
        'r3': ['tail_release_reuse', 'preserve_capacity_oom_recovery', 'tail_then_saved_error_prefix'],
    }
    require(report['schema'] == 'heptatorch.current761.resource7.v1'
            and report['status'] == 'PASS_CURRENT_RESOURCE7'
            and report['actual_build_sha256'] == build_sha
            and report['source_sha256'] == SOURCE and report['native_sha256'] == IMAGE
            and report['functions'] == FUNCTIONS and report['units'] == UNITS,
            'resource7 must qualify the same complete actual cold build')
    require(report['selected_suites'] == list(expected) and report['required_cases'] == expected
            and report['required_case_count'] == report['cases'] == 7
            and report['prepare_only'] is False and report['failures'] == []
            and report['native_executed'] is True and report['all_inputs_unchanged'] is True
            and report['Torch_executed'] is False and report['compiler_executed'] is False,
            'all seven original resource recipes must actually execute')
    require(report['old_pass_transferred'] is False and report['schema_closure'] is False
            and report['full_pytorch_rewrite_complete'] is False,
            'finite resource controls do not close a whole schema or project')
    require([row['suite'] for row in report['suites']] == list(expected), 'all three resource readers')
    helper_path = ROOT / 'tools/current_runtime/fusion_resource_transition.py'
    inputs.bind(helper_path, FUSION_RESOURCE_SHA)
    helper_spec = importlib.util.spec_from_file_location('qualified761_resource_source_paths', helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    expected_source_paths = helper.prove(inputs.bind(ROOT / 'tools/current_runtime/source/runtime.hepta', SOURCE), inputs.bind)
    frames = checks = children = restored_limits = 0
    for row in report['suites']:
        name = row['suite']
        folder = directory / name
        runner = row['process']
        require(row['status'] == 'PASS' and row['case_names'] == expected[name]
                and runner['started'] is True and runner['exit_code'] == 0
                and runner['stdout_EOF'] is True and runner['stderr_EOF'] is True
                and runner['stderr_bytes'] == 0 and not runner.get('timed_out')
                and not runner.get('forced_cleanup') and not runner.get('execution_error')
                and not runner.get('error') and not runner.get('cleanup_errors')
                and not runner.get('running_descendants_after_cleanup'), 'complete actual resource worker')
        for stream in ('stdout', 'stderr'):
            raw = inputs.bind(folder / ('runner.' + stream), runner[stream + '_sha256'])
            require(len(raw) == runner[stream + '_bytes'], 'complete original resource worker stream')
        actual = json.loads(inputs.bind(folder / 'actual/report.json', row['report_sha256']))
        proof = json.loads(inputs.bind(folder / 'adaptation/source-bindings.json'))
        require(proof['fusion_source_path_proof'] == expected_source_paths,
                'outer independent actual761 resource path proof differs')
        require(actual['status'] == 'PASS' and actual['native_executed'] is True
                and actual['source_sha256'] == SOURCE and actual['native_sha256'] == IMAGE
                and actual['all_inputs_unchanged'] is True
                and actual['case_count'] == actual['passed'] == row['cases'] == len(expected[name])
                and [case['name'] for case in actual['cases']] == expected[name], 'all original resource cases')
        require(proof['actual_build_sha256'] == build_sha and proof['source_sha256'] == SOURCE
                and proof['native_sha256'] == IMAGE and proof['functions'] == FUNCTIONS
                and proof['units'] == UNITS and proof['all_original_inputs_unchanged'] is True
                and proof['whole_file_inverse_exact'] is True and proof['original_pass_transferred'] is False
                and proof['numeric_expectations_modified'] is False
                and proof['resource_thresholds_modified'] is False
                and proof['owner_equations_modified'] is False, 'exact resource source and allocation obligations')
        transports = row['native_transports']
        require([item['case'] for item in transports] == expected[name], 'one native child per original recipe')
        for case, transport in zip(actual['cases'], transports):
            # The fixed original OwnerSession.finish requires waitpid exit 0,
            # then reads both output and error streams and asserts literal EOF.
            # Only after these checks does the original case set EOF_exit_code.
            require(case['status'] == 'PASS' and case['EOF_exit_code'] == transport['exit_code'] == 0,
                    'original native finish and both EOF checks must succeed')
            child = folder / 'actual' / case['name'] / 'transport'
            ledger = json.loads(inputs.bind(child / 'transport.json'))
            require(ledger['executable_sha256'] == IMAGE and ledger['exit_code'] == 0
                    and ledger['responses_before_stdin_eof'] is True
                    and ledger['frames'] == len(ledger['rows']) == case['frames_before_EOF'] == transport['frames'],
                    'actual native resource frame ledger')
            inputs.bind(child / 'requests.frames.bin', transport['requests_sha256'])
            inputs.bind(child / 'responses.frames.bin', transport['responses_sha256'])
            if 'oom' in case['name']:
                require(case['original_limit_restored'] is True, 'actual process allocation limit restored')
                restored_limits += 1
            children += 1
        require(row['frames'] == actual['frames_before_EOF'] and row['checks'] == actual['checks'],
                'original resource reader totals')
        frames += row['frames']
        checks += row['checks']
    require(children == 7 and restored_limits == 3 and report['frames'] == frames and report['checks'] == checks,
            'complete resource comparisons and all three actual limit recoveries')
    return dict(cases=7, frames=frames, assertions=checks, native_sessions=7,
                worker_sessions=3, allocation_controls=3)





def execute(command, directory, timeout_seconds):
    return execute_process(command, directory, timeout_seconds, cwd=ROOT)


def observer_intent(command, execution, inputs):
    interpreter = Path(command[0]).resolve(strict=True)
    inputs.bind(interpreter)
    value = dict(argv=command, observer_python=dict(path=str(interpreter),
                 **inputs.records[str(interpreter)]))
    path = execution / "intent.json"
    require(not path.exists(), "new pre-execution observer intent only")
    dump(path, value)
    raw = inputs.bind(path)
    return value, dict(path=execution.name + "/intent.json", bytes=len(raw), sha256=sha(raw))


def check_observer_intent(expected, receipt, execution, inputs):
    actual = json.loads(inputs.bind(execution / "intent.json"))
    require(actual == expected and actual["argv"] == receipt["argv"],
            "unchanged complete pre-execution argv and observer identity")
    interpreter = Path(actual["observer_python"]["path"])
    require(interpreter == Path(receipt["argv"][0]).resolve(strict=True),
            "actual requested observer route still resolves to the selected entity")
    inputs.bind(interpreter, actual["observer_python"]["sha256"])
    require(actual["observer_python"] == dict(path=str(interpreter), **inputs.records[str(interpreter)]),
            "complete pre-execution observer bytes and mode unchanged")


def save_physical_qualification(name, proof, build_sha, directory, inputs):
    expected = {"f64_48": (48, "PASS_PHYSICAL_F64_48"),
                "half47": (47, "PASS_PHYSICAL_HALF47"),
                "half_refusals12": (12, "PASS_PHYSICAL_HALF12")}
    count, status = expected[name]
    require(proof["status"] == status and proof["cases"] == count
            and proof["source_sha256"] == SOURCE and proof["native_sha256"] == IMAGE
            and proof["actual_build_sha256"] == build_sha
            and proof["all_physical_inputs_unchanged"] is True,
            "complete original physical qualification and current build binding")
    require(all(proof[key] is False for key in ("native_executed_by_qualifier",
                "Torch_executed_by_qualifier", "compiler_executed_by_qualifier")),
            "the physical qualifier cannot claim a second native or reference execution")
    successor = {}
    if name == "half47":
        successor = dict(original_entry_requests=239, original_refusals=239,
                         retained_refusals=215, accepted_multiplication_successors=22,
                         accepted_negation_successors=2)
        require(all(proof.get(key) == value for key, value in successor.items()),
                "the exact original239 entry requests require215 refusals,22 multiplication and2 negation numerical successors")
    path = directory / "physical-qualification.json"
    require(not path.exists(), "new physical qualification; original report remains unchanged")
    dump(path, proof)
    raw = inputs.bind(path)
    return dict(cases=count, frames=proof["frames"], assertions=proof["checks"], **successor,
                native_sessions=count, physical_qualification=dict(
                    path=name + "/physical-qualification.json", bytes=len(raw),
                    sha256=sha(raw), status=status))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--tool-build", type=Path, required=True)
    parser.add_argument("--stack-build", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not sys.flags.optimize, "all original assertions must remain enabled")
    build, output = args.build_directory.resolve(), args.output.resolve()
    require(output != build and not output.is_relative_to(build), "qualification output must be separate from its build")
    tool_build, stack_build = args.tool_build.resolve(), args.stack_build.resolve()
    require(tool_build.is_dir() and stack_build.is_dir(), "physical entropy and stack tool builds are required")
    require(all(output != path and not output.is_relative_to(path)
                for path in (tool_build, stack_build)), "qualification output must be separate from tool inputs")
    output.mkdir(parents=True, exist_ok=False)
    inputs, suites, failures = Inputs(), [], []
    started = time.monotonic()
    result = {"schema": "heptatorch.current-runtime.qualification.v1", "status": "RUNNING",
              "source_sha256": SOURCE, "native_sha256": IMAGE, "functions": FUNCTIONS, "units": UNITS,
              "required_suites": list(GROUPS), "required_cases": {"bool27": 27, "original76": 76,
              "sum32": 32, "integral24": 24, "sum_f32_17": 17, "current35": 35, "base270": 270,
              "live32": 32, "logical147": 147, "resource7": 7, "asstrided16": 16, "cat4": 4, "metadata10": 10, "generator131": 131, "public_f64_logical192": 192,
              "f64_48": 48, "half47": 47, "half_logical264": 264, "half_transactions12": 12,
              "numeric_predicates": 485, "bf16_copy": 280, "half_mul": 312, "half_neg": 371}, "required_cases_total": 2839,
              "generator_subgroups": {"oracle24": 24, "native11": 11, "entry59": 59, "dispatch37": 37},
              "image_scope": "All thirteen inherited groups and Generator production cases bind the current main image; "
                  "Generator131 also requires separately linked current-object ABI and reordered probe images. "
                  "Public F64 logical192, F64 48, Half47, Half logical264, Half transactions12 and numeric predicates "
                  "and BF16 copy280, Half multiplication312 and Half negation371 use the same current main image.",
              "half_mul_scope": {"actual_reference_cases": 312, "opcode": 2,
                  "cpu_capability": "AVX2", "requested_cpu_capability": "avx2", "torch_threads": 1,
                  "exact_raw_bytes_required": True, "physical_replay_required": True,
                  "fresh_pinned_Torch_reference_required": True,
                  "original_half47_refusals_with_explicit_successor": 22,
                  "original_half47_refusals_before_negation": 217, "historical239_evidence_rewritten": False},
              "half_neg_scope": {"actual_reference_cases": 371, "opcode": 3, "operation": 6,
                  "cpu_capability": "AVX2", "requested_cpu_capability": "avx2", "torch_threads": 1,
                  "exact_raw_bytes_required": True, "physical_replay_required": True,
                  "fresh_pinned_Torch_reference_required": True,
                  "original_half47_refusals_with_explicit_successor": 2,
                  "original_half47_other_refusals": 215, "historical239_evidence_rewritten": False},
              "bf16_copy_scope": {"actual_reference_cases": 280, "opcodes": [108, 109],
                  "memory_formats": [0, 1], "exact_raw_bytes_required": True,
                  "fresh_pinned_Torch_reference_required": True, "physical_replay_required": True,
                  "historical_predicate_and_Half_results_rewritten": False},
              "public_f64_scope": {"dense_no_grad_cases": 192, "native_boundary_controls": 2,
                  "outside_native_scope_torch_observations": 6,
                  "boundary_controls_counted_as_torch_compatibility": False, "original147_replaced": False},
              "f64_48_scope": {"numeric_groups": 33, "boundary_cases": 12, "entry_groups": 1,
                  "allocation_controls": 2, "entry_refusals": 87,
                  "malformed_Half_factory_expected_statuses_changed": 2,
                  "other85_expected_statuses_unchanged": True,
                  "malformed_controls_counted_as_successful_torch_compatibility": False},
              "half47_scope": {"numeric_groups": 26, "boundary_cases": 16, "entry_groups": 1,
                  "allocation_controls": 4, "input_values": 908216, "entry_refusals": 215,
                  "entry_multiplication_successors": 22, "entry_negation_successors": 2, "original_entry_control_count": 239,
                  "historical239_evidence_rewritten": False,
                  "ordinary_half_AD_supported": False},
              "half_logical264_scope": {"actual_reference_cases": 264, "accepted": 222, "observed_errors": 42,
                  "exhaustive_encoding_cases": 6, "ordinary_dense_no_grad_only": True,
                  "additional_packet_controls": 52, "additional_allocation_recoveries": 6},
              "half_transactions12_scope": {"direct_successes": 6, "failed_FE_transactions": 6,
                  "successful_prefix_checked_before_invalid_suffix": True,
                  "historical_refusal12_report_rewritten": False, "original47_and239_replaced": False},
              "numeric_predicates_scope": {"actual_reference_cases": 485,
                  "operations": ["isnan", "isinf", "isfinite", "isposinf", "isneginf"],
                  "source_dtype_codes": [0, 1, 2, 3, 4, 5, 6],
                  "complete_f16_bf16_encoding_enumeration": True,
                  "original_dispatcher_body_preserved": True,
                  "additional_packet_controls": 56, "successful_FE_transactions": 5,
                  "failed_FE_transactions": 5, "actual_allocation_recoveries": 5},
              "required_controls": {"historical_Bool_dtypes": 7, "historical_Integral_dtypes": 5,
                  "current_dtypes": 7, "F32_axes": 4, "entry_and_capability": 10,
                  "Bool_allocation": 1, "F32_allocation_stages": 2,
                  "original35_copied_positive": 4, "original35_copied_negative": 18,
                  "original270_actual_prefix_refusal_pairs": 14,
                  "logical_packet_controls": 18, "logical_allocation_stages": 2,
                  "inherited_resource_allocation_recoveries": 3, "cat_expanded_capacity": 1,
                  "public_f64_packet_controls": 18, "public_f64_allocation_stages": 2,
                  "public_f64_native_scope_controls": 2,
                  "f64_48_entry_refusals": 87, "f64_48_allocation_recoveries": 2,
                  "half47_entry_refusals": 215, "half47_entry_multiplication_successors": 22,
                  "half47_entry_negation_successors": 2,
                  "half47_allocation_recoveries": 4,
                  "half_logical_packet_controls": 52, "half_logical_allocation_recoveries": 6,
                  "half_direct_successes": 6, "half_failed_FE_transactions": 6,
                  "predicate_packet_controls": 56, "predicate_successful_FE_transactions": 5,
                  "predicate_failed_FE_transactions": 5, "predicate_allocation_recoveries": 5},
              "suite_timeout_seconds": SUITE_TIMEOUT_SECONDS,
              "suites": suites, "failures": failures, "full_pytorch_rewrite_complete": False,
              "schema_closure": False, "full_project_status": "HOLD", "Torch_executed": False}

    def save():
        dump(output / "qualification.json", result)

    try:
        inputs.bind(Path(__file__).resolve())
        inputs.bind(Path(sys.executable).resolve(strict=True))
        for name in ("current_runtime_fixtures.py", "current_runtime_process.py", "current_runtime_admission.py",
                     "current_runtime_torch.py", "current_runtime_live32.py", "current_runtime_live32_readback.py",
                     "current_runtime_cat4/qualify.py", "current_runtime_metadata10/qualify.py",
                     "current_runtime_metadata10/common.py", "current_runtime_metadata10/worker.py",
                     "current_runtime_metadata10/readback.py", "current_runtime_metadata10/source.json",
                     "current_runtime_metadata10/frozen/native_v2.py",
                     "current_runtime_generator131.py", "current_runtime_generator_fixtures.py",
                     "current_runtime_generator_binding.py", "current_runtime_generator_binding.json",
                     "current_runtime_public_f64_logical192.py",
                     "current_runtime_half_fixtures.py", "current_runtime_half_readback.py",
                     "current_runtime_half_refusals12.py", "current_runtime_half_mul_successor.py", "current_runtime_half_neg_successor.py", "current_runtime_f64_fixtures.py",
                     "current_runtime_f64_readback.py",
                     "half_logical/__init__.py", "half_logical/reference.py", "half_logical/native.py", "half_logical/run.py",
                     "numeric_predicates/__init__.py", "numeric_predicates/cases.py", "numeric_predicates/reference.py",
                     "numeric_predicates/evidence.py", "numeric_predicates/native.py", "numeric_predicates/run.py",
                     "bf16_copy/__init__.py", "bf16_copy/cases.py", "bf16_copy/reference.py", "bf16_copy/run.py",
                     "half_mul/__init__.py", "half_mul/cases.py", "half_mul/legacy_controls.py",
                     "half_mul/reference.py", "half_mul/run.py",
                     "half_neg/__init__.py", "half_neg/cases.py", "half_neg/reference.py", "half_neg/run.py",
                     "pytorch_cpu_generator_oracle.py"):
            inputs.bind(ROOT / "tests" / name)
        build_sha = build_binding(build, inputs)
        result["actual_build_sha256"] = build_sha
        result["cat_source_transition"] = cat_source_binding(build, inputs)
        tool_raw = inputs.bind(tool_build / "build.json")
        inputs.bind(tool_build / "generator-session-linker", LINKER)
        result["entropy_tool_build"] = {"path": str(tool_build / "build.json"),
            "bytes": len(tool_raw), "sha256": sha(tool_raw)}
        result["stack_build"] = str(stack_build)
        fixtures = Fixtures(output, inputs, args.torch_python)
        fixtures.cache["generator_tool_paths"] = (tool_build, stack_build)
        for name in GROUPS:
            execution = output / (name + "-execution")
            execution.mkdir()
            row = {"name": name, "status": "PREPARING", "qualified": False}
            suites.append(row); save()
            print(json.dumps({"suite": name, "status": row["status"]}), flush=True)
            try:
                directory = output / name
                command = runner_command(name, build, build_sha, directory, fixtures, inputs, args.torch_python.absolute())
                intent = None
                if name in ("f64_48", "half47", "half_logical264", "half_transactions12", "numeric_predicates", "bf16_copy", "half_mul", "half_neg"):
                    intent, row["execution_intent"] = observer_intent(command, execution, inputs)
                receipt = execute(command, execution, SUITE_TIMEOUT_SECONDS[name])
                row["runner"] = receipt
                report = None
                if (directory / "report.json").is_file():
                    raw = inputs.bind(directory / "report.json")
                    report = json.loads(raw)
                    row["original_report"] = {"path": str((directory / "report.json").relative_to(output)),
                                              "bytes": len(raw), "sha256": sha(raw)}
                    row["original_report_status"] = report.get("status")
                    if name in ("current35", "base270", "live32") and result["Torch_executed"] is not True:
                        result["Torch_executed"] = report.get("Torch_executed")
                    if name == "generator131" and result["Torch_executed"] is not True:
                        result["Torch_executed"] = report.get("Torch_executed")
                    if name in ("numeric_predicates", "bf16_copy", "half_mul", "half_neg") and result["Torch_executed"] is not True:
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
                if intent is not None:
                    check_observer_intent(intent, receipt, execution, inputs)
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
                elif name == "base270":
                    counts = qualify_current270(report, build_sha, directory, inputs)
                elif name == "live32":
                    counts = qualify_live32(report, build_sha, directory, inputs)
                elif name == "logical147":
                    counts = qualify_logical(report, build_sha, directory, inputs)
                elif name == "resource7":
                    counts = qualify_resource7(report, build_sha, directory, inputs)
                elif name == "asstrided16":
                    counts = qualify_asstrided16(report, build_sha, directory, inputs)
                elif name == "cat4":
                    counts = qualify_cat4(report, ROOT, build, build_sha, directory, inputs)
                elif name == "generator131":
                    sources = fixtures.generator()
                    counts = qualify_generator131(report, ROOT, build, build_sha, directory, sources["package"],
                        sources["candidate"], sources["original26"], tool_build, stack_build, inputs)
                elif name == "public_f64_logical192":
                    source = fixtures.public_f64_logical192()
                    counts = public_f64_logical192.qualify_public192(report, ROOT, build, build_sha,
                        directory, source["root"], inputs, execution)
                elif name == "f64_48":
                    source = fixtures.f64_48()
                    proof = qualify_f64_48(report, ROOT, build, build_sha, directory,
                        source["source"], source["root"], inputs, execution)
                    counts = save_physical_qualification(name, proof, build_sha, directory, inputs)
                elif name == "half47":
                    source = fixtures.half47()
                    proof = qualify_half47(report, ROOT, build, build_sha, directory,
                        source["source"], source["root"], inputs, execution)
                    counts = save_physical_qualification(name, proof, build_sha, directory, inputs)
                elif name in ("half_logical264", "half_transactions12"):
                    counts = qualify_half_logical(report, ROOT, build, build_sha, directory, inputs)
                    physical = inputs.bind(directory / "physical-qualification.json")
                    require(json.loads(physical) == counts, "complete independent physical Half readback agrees")
                    counts["physical_qualification"] = dict(path=name + "/physical-qualification.json",
                        bytes=len(physical), sha256=sha(physical), status="PASS_PHYSICAL_HALF_LOGICAL")
                elif name == "numeric_predicates":
                    counts = qualify_numeric_predicates(report, ROOT, build, build_sha, directory, inputs)
                    require(counts["cases"] == 485, "all fixed independent predicate comparisons are mandatory")
                    physical = inputs.bind(directory / "physical-qualification.json")
                    require(json.loads(physical) == counts, "complete independent predicate frame and owner readback agrees")
                    counts["physical_qualification"] = dict(path=name + "/physical-qualification.json",
                        bytes=len(physical), sha256=sha(physical), status="PASS_PHYSICAL_NUMERIC_PREDICATES")
                elif name == "bf16_copy":
                    counts = qualify_bf16_copy(report, ROOT, build, build_sha, directory, inputs)
                    require(counts["cases"] == 280 and counts["physical_replay"] is True,
                            "all independent BF16 conversion comparisons and physical replay are mandatory")
                    physical = inputs.bind(directory / "physical-qualification.json")
                    require(json.loads(physical) == counts, "complete BF16 byte and ownership readback agrees")
                    counts["physical_qualification"] = dict(path=name + "/physical-qualification.json",
                        bytes=len(physical), sha256=sha(physical), status="PASS_PHYSICAL_BF16_COPY")
                elif name == "half_mul":
                    counts = qualify_half_mul(report, ROOT, build, build_sha, directory, inputs)
                    require(counts["cases"] == 312 and counts["physical_replay"] is True,
                            "all independent Half multiplication comparisons and physical replay are mandatory")
                    physical = inputs.bind(directory / "physical-qualification.json")
                    require(json.loads(physical) == counts, "complete Half multiplication byte and ownership readback agrees")
                    counts["physical_qualification"] = dict(path=name + "/physical-qualification.json",
                        bytes=len(physical), sha256=sha(physical), status="PASS_PHYSICAL_HALF_MUL")
                elif name == "half_neg":
                    counts = qualify_half_neg(report, ROOT, build, build_sha, directory, inputs)
                    require(counts["cases"] == 371 and counts["physical_replay"] is True,
                            "all independent Half negation comparisons and physical replay are mandatory")
                    physical = inputs.bind(directory / "physical-qualification.json")
                    require(json.loads(physical) == counts, "complete Half negation byte and ownership readback agrees")
                    counts["physical_qualification"] = dict(path=name + "/physical-qualification.json",
                        bytes=len(physical), sha256=sha(physical), status="PASS_PHYSICAL_HALF_NEG")
                else:
                    counts = qualify_metadata10(report, ROOT, build, build_sha, directory, inputs, execution)
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
        counts_match = complete and all(row.get("cases") == result["required_cases"][row["name"]]
                                        for row in suites)
        observed_cases = sum(row.get("cases", 0) for row in suites)
        if complete and all(row["qualified"] is True for row in suites) and (
                not counts_match or observed_cases != result["required_cases_total"]):
            failures.append("the complete actual case denominator differs from the mandatory inventory")
        passed = complete and counts_match and observed_cases == result["required_cases_total"] and all(
            row["qualified"] is True and row["status"] == "PASS" for row in suites) and not failures
        result["executed_cases_total"] = observed_cases
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
