#!/usr/bin/env python3
"""Compare the fresh 700-function HIA image with the actual Bool reference.

Python transports commands and reads native-owned bytes. It computes no sum or
expected tensor values. Historical Integral24/5 sources remain unchanged.
"""
import argparse
import json
import re
import struct
import sys
import traceback
from pathlib import Path

from cases import CASES, CASE_SHA256, MASK, raw_input, product
from common import (Inputs, PROTOCOL_SHA, CLIENT_SHA, bind_sources, dump, require,
                    sha, source_functions)
from owner_adapter import make_observer

SOURCE = "b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4"
IMAGE = "c6afeedc14ba8cc408a62d6fbef0abb5b78dff0039b8099f3449d813e290a90e"
COMPILER = "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"
LINKER = "aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16"
TENSOR_FIELDS = ("dtype", "device", "element_size", "shape", "strides", "numel",
                 "storage_bytes", "storage_offset_elements", "storage_offset_bytes", "version",
                 "requires_grad", "is_leaf", "is_view", "grad_fn", "grad_is_none", "retains_grad",
                 "logical_bytes", "logical_bytes_hex", "logical_sha256", "storage_bytes_hex", "storage_sha256")
PROVENANCE_CHECKS = {"python_matches", "platform_matches", "version_backend_matches",
                     "git_matches_oracle_build_pin", "distribution_version_matches",
                     "direct_url_matches", "direct_url_archive_sha256_matches",
                     "pip_report_url_matches", "pip_report_archive_sha256_matches",
                     "pip_report_version_matches", "pip_report_requested"}


def normalized(event):
    require(event["warnings"] == [], "reference-no-unmapped-warning", event=event)
    if event["accepted"]:
        require(event["error"] is None, "reference-success-no-error")
        return 0
    kind, text = event["error"]["type"], event["error"]["message"].lower()
    if event["label"] == "sum_bool" and kind == "IndexError" and text.startswith(
            "dimension out of range (expected to be in range of ["):
        return 2
    if event["label"] == "sum_bool" and kind == "RuntimeError" and re.fullmatch(
            r"dim [0-9]+ appears multiple times in the list of dims", text):
        return 7
    if event["label"] == "result_requires_grad_true" and kind == "RuntimeError" and (
            "floating point" in text and "require gradients" in text):
        return 5
    if event["label"] == "backward_bool_result" and kind == "RuntimeError" and (
            "does not require grad" in text and "grad_fn" in text):
        return 5
    raise AssertionError(dict(rule="unmapped-reference-error", event=event))


def admit_build(args, inputs):
    source = inputs.bind(args.source, args.expected_source_sha256)
    require(sha(source) == SOURCE, "exact-Bool700-source")
    functions = source_functions(source)
    kernel = args.repository / "continuation/20261117/source-packages/sum-bool-source-01/hia/sb_sum_bool.hepta"
    require(functions["sb_sum_bool"] == inputs.bind(kernel), "original-Bool-kernel-complete-bytes")
    executable = inputs.bind(args.binary, args.expected_native_sha256)
    require(executable.startswith(b"\x7fELF") and sha(executable) == IMAGE, "exact-reproduced-Bool700-ELF")
    build = json.loads(inputs.bind(args.build_report, args.expected_build_sha256))
    require(build["schema"] == "heptatorch.complete-runtime-build.v1" and
            build["status"] == "PASS_NATIVE_BUILD_ONLY" and build["all_inputs_unchanged"] is True,
            "actual-complete-cold-build")
    require(build["objects_reused"] is False and build["complete_core_bodies_matched"] == 700,
            "all-seven-hundred-bodies-cold-compiled")
    require(build["source_sha256"] == SOURCE and build["native_sha256"] == sha(executable)
            and build["native_bytes"] == len(executable) and build["functions"] == 700
            and build["compiler_sha256"] == COMPILER and build["linker_sha256"] == LINKER
            and build["link_exit"] == 0, "actual-build-image-source-pins")
    require(len(build["units"]) == build["units_planned"] == 14 and len(build["processes"]) == 15,
            "cold-fourteen-units-fifteen-processes")
    root = args.build_report.parent
    require(args.source.parent == root and args.binary.parent == root and args.binary.name == build["native_basename"],
            "all-build-products-same-root")
    seen = set()
    for unit in build["units"]:
        name = unit["name"]
        require(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) and name not in seen, "unique-unit-name")
        seen.add(name)
        require(unit["compiler_exit"] == 0 and unit["compiler_stderr_bytes"] == 0,
                "actual-unit-compiler-success", unit=name)
        inputs.bind(root / (name + ".hepta"), unit["source_sha256"])
        obj = inputs.bind(root / (name + ".o"), unit["object_sha256"])
        require(obj.startswith(b"\x7fELF") and len(obj) == unit["object_bytes"], "actual-unit-object", unit=name)
        require(inputs.bind(root / (name + ".stderr")) == b"", "actual-unit-empty-stderr", unit=name)
    require(inputs.bind(root / "link.stderr") == b"", "actual-link-empty-stderr")
    link_request = inputs.bind(root / "link.request.bin")
    require({p["name"] for p in build["processes"]} == seen | {"link"},
            "every-actual-compiler-and-linker-process")
    for process in build["processes"]:
        require(process["started"] is True and process["exit_code"] == 0 and
                process["stdout_EOF"] is True and process["stderr_EOF"] is True and
                process["stderr_bytes"] == 0, "actual-process-complete-streams", process=process["name"])
        if process["name"] == "link":
            require(process["input_sha256"] == sha(link_request) and process["input_bytes"] == len(link_request)
                    and process["stdout_sha256"] == IMAGE and process["stdout_bytes"] == len(executable),
                    "actual-link-request-output-binding")
        else:
            unit = next(u for u in build["units"] if u["name"] == process["name"])
            require(process["input_sha256"] == unit["source_sha256"] and
                    process["stdout_sha256"] == unit["object_sha256"] and
                    process["stdout_bytes"] == unit["object_bytes"], "actual-compiler-input-output-binding")
    return build


def admit_reference(args, inputs):
    report = json.loads(inputs.bind(args.reference, args.expected_reference_sha256))
    require(report["schema"] == "heptatorch.sum-bool.reference27.v1" and report["status"] == "OBSERVED"
            and report["torch_executed"] is True and report["native_executed"] is False, "actual-Bool-reference")
    require(report["case_sha256"] == CASE_SHA256 and report["case_plan"] == list(CASES)
            and report["cases"] == report["expected_cases"] == len(report["rows"]) == 27,
            "complete-frozen-27-case-plan")
    require(report["source_sha256"] == sha(inputs.bind(Path(__file__).with_name("reference.py"))),
            "actual-reference-probe-source")
    provenance = report["oracle_provenance"]
    require(provenance["status"] == "PASS" and set(provenance["checks"]) == PROVENANCE_CHECKS
            and all(v is True for v in provenance["checks"].values()), "fixed-eleven-oracle-checks")
    lock = json.loads(inputs.bind(args.repository / "compatibility/pytorch-oracle-lock.json"))
    require(provenance["expected"] == lock["oracle"] and
            provenance["release_source_equivalence"] == lock["release_source_equivalence"] and
            provenance["lock_sha256"] == "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944",
            "same-original-oracle-wheel-lock")
    rng = bytes.fromhex(report["rng_hex"])
    require(len(rng) == 5056 and sha(rng) == report["rng_sha256"] and report["global_rng_unchanged"] is True
            and report["inputs_unchanged"] is True, "reference-source-and-full-RNG")
    for i, (spec, row) in enumerate(zip(CASES, report["rows"])):
        require(row["index"] == i and row["name"] == spec["name"] and row["spec"] == spec
                and row["status"] == "OBSERVED", "all-ordered-reference-cases")
        success = row["target_accepted"]
        operations = ["sum_bool"] + (["result_requires_grad_true", "backward_bool_result"] if success else []) + ["mutate_backing_zero"]
        marks = ["before_sum", "after_sum"] + (["after_requires_grad", "before_backward", "after_backward", "after_seed_drop"] if success else [])
        marks += ["after_input_mutation", "after_backing_drop", "after_input_drop"] + (["after_result_drop"] if success else [])
        require([r["label"] for r in row["operations"]] == operations and [r["label"] for r in row["marks"]] == marks,
                "complete-reference-operations-lifetimes", case=i)
        for event in row["operations"]:
            normalized(event)
        require(all(x["rng_hex"] == report["rng_hex"] and x["rng_sha256"] == report["rng_sha256"] for x in row["marks"]),
                "every-mark-full-RNG", case=i)
    return report


class Driver:
    def __init__(self, args, observer, owner, directory):
        self.args, self.owner, self.directory = args, owner, directory
        self.checks, self.marks, self.operations = [], [], []
        self.n = observer.Native(args.binary, args.source, args.protocol, directory)
        self.live, self.weak, self.tensor_targets = {}, {}, {}
        self.tensor_ids, self.storage_ids = {}, {}
        try:
            self.gen = self.n.one(60, 1729)["value"]
            self.rng = self.n.one(63, self.gen)["payload"]
            self.check("native-full-RNG-width", len(self.rng), 5056)
        except BaseException:
            self.n.finish()
            raise

    def check(self, label, actual, expected):
        row = dict(label=label, actual=actual, expected=expected, passed=actual == expected)
        self.checks.append(row)
        require(row["passed"], "native-comparison", label=label, actual=actual, expected=expected)

    def issue(self, command, label, *, partial=False):
        before, before_ref, before_raw = self.n.capture_owner(label + "/before")
        request = self.n.w.packet([command])
        response = self.n.s.exchange(request)
        status, records = self.n.w.parse(response)
        record = records[0] if records else None
        effective = status
        if not status and partial:
            self.check(label + "/partial-opcode", record["opcode"], 58)
            self.check(label + "/partial-payload-width", len(record["payload"]), 32)
            effective = struct.unpack("<4Q", record["payload"])[0]
        after, after_ref, after_raw = self.n.capture_owner(label + "/after")
        if effective:
            self.check(label + "/rollback-pointer", after["pointer"], before["pointer"])
            self.check(label + "/rollback-length", len(after_raw), len(before_raw))
            self.check(label + "/complete-owner-rollback", after_raw == before_raw, True)
            self.check(label + "/complete-semantic-rollback", self.owner.semantic(after) == self.owner.semantic(before), True)
        self.operations.append(dict(label=label, frame_status=status, normalized_status=effective,
                                    request_hex=request.hex(), response_hex=response.hex(),
                                    before_owner=before_ref, after_owner=after_ref))
        return effective, record

    def publish(self, name, token):
        self.check("new-role/" + name, name in self.live or name in self.weak, False)
        self.live[name] = token
        memory = self.n.memory()
        node = memory.public_node(token)
        self.tensor_targets[name] = (node, memory.node(node)[40])
        self.weak[name] = self.n.one(42, token)["value"]

    def view(self, source, shape, strides, offset):
        # Match Integral24's inherited Case.view / current Tensor.as_strided
        # entry. ParentNative.view is the older bounded command69 helper.
        cfg = struct.pack("<10Q", len(shape), *shape, *([0] * (4 - len(shape))),
                          *strides, *([0] * (4 - len(shape))), offset)
        return self.n.one(111, source, self.n.w.Payload(cfg), 80)["value"]

    def setup(self, spec):
        backing = self.n.create(1 if spec["dtype"] == "bool" else 2,
                                [len(spec["values"])], raw_input(spec), requires=0)
        self.publish("backing", backing)
        source = self.view(backing, spec["shape"], spec["strides"], spec["offset"])
        self.publish("input", source)

    def sum_command(self, spec, dtype=1):
        axes = spec["axes"]
        payload = 0 if not axes else self.n.w.Payload(struct.pack("<" + "q" * len(axes), *axes))
        return (115, self.live["input"], MASK if axes is None else len(axes), payload,
                int(spec["keepdim"]), dtype, 0, 0)

    def drop(self, name):
        self.n.one(41, self.live.pop(name))

    def tensor(self, token):
        row = self.n.tensor(token)
        presence = self.n.one(59, token)["value"]
        gradient = self.n.one(91, token)["value"]
        self.check("gradient-getter-and-presence", bool(gradient), bool(presence))
        self.check("no-invented-gradient-Tensor", gradient, 0)
        row["grad_is_none"] = gradient == 0
        row["retains_grad"] = bool(self.n.one(94, token)["value"])
        return row

    def compare_tensor(self, expected, actual, label):
        for field in TENSOR_FIELDS:
            self.check(label + "/" + field, actual[field], expected[field])
        key = (expected["tensor_identity"], expected["storage_identity"])
        value = actual["native_incarnation"]
        if key not in self.tensor_ids:
            self.check(label + "/injective-Tensor", value in self.tensor_ids.values(), False)
            self.tensor_ids[key] = value
        self.check(label + "/stable-Tensor", value, self.tensor_ids[key])
        storage = expected["storage_identity"]
        if storage not in self.storage_ids:
            self.check(label + "/injective-Storage", actual["native_storage_id"] in self.storage_ids.values(), False)
            self.storage_ids[storage] = actual["native_storage_id"]
        self.check(label + "/stable-Storage", actual["native_storage_id"], self.storage_ids[storage])
        base = expected["base_tensor_identity"]
        self.check(label + "/ultimate-base", actual["native_base_incarnation"],
                   None if base is None else self.tensor_ids[(base, storage)])
        if actual["dtype"] == "torch.bool":
            self.check(label + "/canonical-physical-Bool", set(bytes.fromhex(actual["storage_bytes_hex"])) <= {0, 1}, True)

    def mark(self, expected):
        label = expected["label"]
        self.check(label + "/live-roles", list(self.live), expected["live_roles"])
        tensors = {}
        for name, token in self.live.items():
            tensors[name] = self.tensor(token)
            self.compare_tensor(expected["tensors"][name], tensors[name], label + "/" + name)
        relations = {}
        names = list(tensors)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                a, b = tensors[left], tensors[right]
                relations[left + "|" + right] = dict(same_Tensor=a["native_incarnation"] == b["native_incarnation"],
                    same_Storage=a["native_storage_id"] == b["native_storage_id"],
                    same_current_Version_value=a["version"] == b["version"])
        self.check(label + "/all-relations", relations, expected["relations"])
        memory = self.n.memory()
        alive = {}
        for name, (index, incarnation) in self.tensor_targets.items():
            alive[name] = False
            if 0 < index < memory.q(8):
                node = memory.node(index)
                alive[name] = bool(node[38] and node[30] and node[40] == incarnation)
        self.check(label + "/actual-Tensor-weak-lifetime", alive, expected["tensor_weak_alive"])
        expired = {}
        for name, token in self.weak.items():
            status, record = self.issue((43, token), label + "/StorageWeak/" + name)
            self.check(label + "/StorageWeak-cause/" + name, status, 6 if expected["storage_weak_expired"][name] else 0)
            expired[name] = status == 6
            if status == 0:
                self.n.one(45, record["value"])
        self.check(label + "/actual-Storage-weak-lifetime", expired, expected["storage_weak_expired"])
        rng = self.n.one(63, self.gen)["payload"]
        self.check(label + "/complete-RNG-reference", rng.hex(), expected["rng_hex"])
        self.check(label + "/complete-RNG-native-stable", rng == self.rng, True)
        self.n.capture_owner(label + "/final")
        self.marks.append(dict(label=label, tensors=tensors, relations=relations,
                               tensor_weak_alive=alive, storage_weak_expired=expired, rng_sha256=sha(rng)))

    def finish_controls(self):
        self.check("final-native-owner-audit", self.n.one(53)["value"], 1)
        for token in self.weak.values():
            self.n.one(44, token)
        self.n.one(66, self.gen)
        self.check("after-observer-release-native-audit", self.n.one(53)["value"], 1)


def compare_case(driver, row):
    spec = row["spec"]
    marks = {x["label"]: x for x in row["marks"]}
    operations = {x["label"]: x for x in row["operations"]}
    driver.setup(spec)
    driver.mark(marks["before_sum"])
    status, record = driver.issue(driver.sum_command(spec), "sum_bool")
    driver.check("sum-actual-normalized-status", status, normalized(operations["sum_bool"]))
    if status == 0:
        driver.publish("result", record["value"])
    driver.mark(marks["after_sum"])
    if row["target_accepted"]:
        status, _ = driver.issue((90, driver.live["result"], 1), "result_requires_grad_true")
        driver.check("requires-actual-normalized-status", status, normalized(operations["result_requires_grad_true"]))
        driver.mark(marks["after_requires_grad"])
        shape = marks["before_backward"]["tensors"]["seed"]["shape"]
        seed = driver.n.create(1, shape, b"\x01" * product(shape), requires=0)
        driver.publish("seed", seed)
        driver.mark(marks["before_backward"])
        status, _ = driver.issue((58, driver.live["result"], driver.live["seed"], 0), "backward_bool_result", partial=True)
        driver.check("backward-actual-normalized-status", status, normalized(operations["backward_bool_result"]))
        driver.mark(marks["after_backward"])
        driver.drop("seed")
        driver.mark(marks["after_seed_drop"])
    status, _ = driver.issue((100, driver.live["backing"], 0, 0), "mutate_backing_zero")
    driver.check("mutation-actual-normalized-status", status, normalized(operations["mutate_backing_zero"]))
    driver.mark(marks["after_input_mutation"])
    driver.drop("backing")
    driver.mark(marks["after_backing_drop"])
    driver.drop("input")
    driver.mark(marks["after_input_drop"])
    if row["target_accepted"]:
        driver.drop("result")
        driver.mark(marks["after_result_drop"])
    driver.check("all-reference-marks-completed", [x["label"] for x in driver.marks], [x["label"] for x in row["marks"]])
    driver.check("all-strong-roles-dropped", driver.live, {})
    driver.finish_controls()


def run_one(args, observer, owner, directory, name, function):
    directory.mkdir(parents=True, exist_ok=False)
    driver, error = None, None
    try:
        driver = Driver(args, observer, owner, directory)
        function(driver)
    except BaseException as failure:
        error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
    finally:
        if driver is not None:
            try:
                driver.n.finish()
            except BaseException as failure:
                cleanup = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
                if error is None:
                    error = dict(type="EOF_FAILURE", cleanup=cleanup)
                else:
                    error["EOF_failure"] = cleanup
    result = dict(name=name, status="PASS" if error is None else "FAIL", error=error,
                  checks=driver.checks if driver else [], marks=driver.marks if driver else [],
                  operations=driver.operations if driver else [], frames=driver.n.s.frames if driver else 0,
                  exit_code=driver.n.s.exit_code if driver else None,
                  full_transition_recomputed=False, schema_closure=False, full_project_status="HOLD")
    dump(directory / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "binary", "source", "build-report", "reference", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("source", "native", "build", "reference"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(not sys.flags.optimize, "assertions-enabled")
    for name in ("repository", "binary", "source", "build_report", "reference", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.protocol = args.repository / "tests/differentiable_view_native_review/protocol.py"
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    bind_sources(inputs, Path(__file__).parent)
    report = dict(schema="heptatorch.sum-bool.native27.v1", status="PREPARED_NOT_RUN",
                  expected_cases=27, cases=[], dtype_controls=[], entry_controls=[], allocation_controls=[],
                  source_sha256=args.expected_source_sha256, native_sha256=args.expected_native_sha256,
                  reference_sha256=args.expected_reference_sha256, native_executed=False,
                  Torch_executed=False, original24_or_original5_modified=False,
                  error=None, schema_closure=False, full_project_status="HOLD")
    exit_code = 0
    try:
        build = admit_build(args, inputs)
        reference = admit_reference(args, inputs)
        inputs.bind(args.protocol, PROTOCOL_SHA)
        inputs.bind(args.protocol.with_name("owner_client.py"), CLIENT_SHA)
        observer, owner, proof = make_observer(args.repository, inputs, SOURCE, args.expected_native_sha256)
        report.update(build_functions=build["functions"], cold_units=len(build["units"]), observer_adaptation=proof)
        if not args.prepare_only:
            from controls import run_dtype_controls, run_entry_controls, run_allocation_control
            report["native_executed"] = True
            for row in reference["rows"]:
                result = run_one(args, observer, owner, args.output / ("case-%02d" % row["index"]),
                                 row["name"], lambda driver, row=row: compare_case(driver, row))
                report["cases"].append(result)
            report["dtype_controls"] = run_dtype_controls(args, observer, owner, reference, run_one)
            report["entry_controls"] = run_entry_controls(args, observer, owner, run_one)
            report["allocation_controls"] = run_allocation_control(args, observer, owner, run_one)
            all_rows = report["cases"] + report["dtype_controls"] + report["entry_controls"] + report["allocation_controls"]
            report.update(status="PASS" if all(x["status"] == "PASS" for x in all_rows) else "FAIL",
                          passed=sum(x["status"] == "PASS" for x in report["cases"]),
                          failed=sum(x["status"] != "PASS" for x in report["cases"]))
            exit_code = 0 if report["status"] == "PASS" else 1
        report["inputs_unchanged"] = inputs.unchanged()
    except BaseException as failure:
        report.update(status="FAIL", error=dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc()))
        exit_code = 1
    report["inputs"] = inputs.rows
    dump(args.output / "report.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "passed", "failed", "native_executed", "error")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
