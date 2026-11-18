#!/usr/bin/env python3
"""Compare the distinct 701-function HIA image with the actual F32 reference.

Python transports commands and reads native-owned bytes. It computes no sum or
expected tensor values. Historical Integral24/5 sources remain unchanged.
"""
import argparse
import json
import lzma
import re
import struct
import sys
import traceback
from pathlib import Path

from cases import AXIS_CONTROLS, CASES, CASE_SHA256, MASK, raw_input, product
from native_support import (Inputs, PROTOCOL_SHA, CLIENT_SHA, bind_sources, dump, require,
                            sha, source_functions, source_inverse)
from owner_adapter import make_observer

SOURCE = "82360ce87dae6f9a004ef50be4f991b6f83905a83f4b17ca8d6634827b7bd313"
IMAGE = "80445a8c433f20314d3bc7b00ebb68f6a70fa200ee2e6aba193cff21aabfbc17"
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
    require(event.get("warnings", []) == [], "reference-no-unmapped-warning", event=event)
    if event["accepted"]:
        require(event["error"] is None, "reference-success-no-error")
        return 0
    kind, message = event["error"]["type"], event["error"]["message"].lower()
    label = event.get("label", "")
    if label in {"sum_explicit_float32", "axis_control"}:
        if kind == "IndexError" and message.startswith("dimension out of range (expected to be in range of ["):
            return 2
        if kind == "RuntimeError" and re.fullmatch(r"dim [0-9]+ appears multiple times in the list of dims", message):
            return 7
    if label == "backward_without_requires" and kind == "RuntimeError" and (
            "does not require grad" in message and "grad_fn" in message):
        return 5
    raise AssertionError(dict(rule="unmapped-reference-error", event=event))


def admit_build(args, inputs):
    source = inputs.bind(args.source, args.expected_source_sha256)
    require(sha(source) == SOURCE, "exact-integral-F32-701-source")
    functions = source_functions(source)
    inverse = source_inverse(source, inputs)
    kernel = args.repository / "continuation/20261117/source-packages/sum-bool-source-01/hia/sb_sum_bool.hepta"
    require(functions["sb_sum_bool"] == inputs.bind(kernel), "original-Bool-kernel-complete-bytes")
    executable = inputs.bind(args.binary, args.expected_native_sha256)
    require(executable.startswith(b"\x7fELF") and sha(executable) == IMAGE, "exact-reproduced-integral-F32-701-ELF")
    build = json.loads(inputs.bind(args.build_report, args.expected_build_sha256))
    require(build["schema"] == "heptatorch.complete-runtime-build.v1" and
            build["status"] == "PASS_NATIVE_BUILD_ONLY" and build["all_inputs_unchanged"] is True,
            "actual-complete-cold-build")
    require(build["objects_reused"] is False and build["complete_core_bodies_matched"] == 701,
            "all-seven-hundred-one-bodies-cold-compiled")
    require(build["source_sha256"] == SOURCE and build["native_sha256"] == sha(executable)
            and build["native_bytes"] == len(executable) and build["functions"] == 701
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
    build["admitted_source_transition"] = inverse
    return build


def admit_reference(args, inputs):
    require(args.expected_reference_sha256 == "c86b0a9384c788eaf7765480b022416075a0e3aa38455f48dcbd7f40476cecb0",
            "exact-root-recovered-reference17")
    report = json.loads(inputs.bind(args.reference, args.expected_reference_sha256))
    require(report["schema"] == "heptatorch.sum-integral-f32.reference17.v1" and report["status"] == "OBSERVED"
            and report["torch_executed"] is True and report["native_executed"] is False, "actual-integral-F32-reference")
    require(report["case_sha256"] == CASE_SHA256 and report["case_plan"] == list(CASES)
            and report["cases"] == report["expected_cases"] == len(report["rows"]) == 17
            and report["accepted"] == report["cast_witness_matches"] == 17, "complete-frozen-17-case-plan")
    require(report["axis_control_plan"] == list(AXIS_CONTROLS) and report["expected_axis_controls"] ==
            len(report["axis_controls"]) == report["axis_controls_rejected"] == 4, "complete-four-axis-controls")
    require(report["cpu_capability"] == "AVX512" and report["num_threads"] == 1,
            "same-original-F32-single-thread-AVX512-reduction")
    source_pins = {
        "reference.py": "d00c56de5577b3d85433010a7163e125ab0d723432b8c014a0f4789811980fe7",
        "cases.py": "b43046f57884880be4c936826f9fd36e5d153ce19fb3f9b6f4355a56c01f9b16",
        "common.py": "7784ea8a57ea02adac975a69f0aeccb825b87a51989608dfac710e97e223f2c7",
    }
    for name, pin in source_pins.items():
        raw = inputs.bind(Path(__file__).with_name(name), pin)
        observed = [item for path, item in report["inputs"].items() if Path(path).name == name]
        require(len(observed) == 1 and observed[0]["sha256"] == pin and observed[0]["bytes"] == len(raw),
                "actual-reference-three-frozen-source-inputs", name=name)
    require(report["source_sha256"] == source_pins["reference.py"], "reference-probe-pin")
    provenance = report["oracle_provenance"]
    require(provenance["status"] == "PASS" and set(provenance["checks"]) == PROVENANCE_CHECKS
            and all(v is True for v in provenance["checks"].values()), "fixed-eleven-oracle-checks")
    lock = json.loads(inputs.bind(args.repository / "compatibility/pytorch-oracle-lock.json",
                                 "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944"))
    require(provenance["expected"] == lock["oracle"] and
            provenance["release_source_equivalence"] == lock["release_source_equivalence"] and
            provenance["lock_sha256"] == "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944",
            "same-original-oracle-wheel-lock")
    require(report["torch_version"] == "2.15.0+cpu" and report["torch_git_version"] ==
            "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e", "actual-fixed-official-CPU-wheel")
    rng = bytes.fromhex(report["rng_hex"])
    require(len(rng) == 5056 and sha(rng) == report["rng_sha256"] and report["global_rng_unchanged"] is True
            and report["inputs_unchanged"] is True, "reference-source-and-full-RNG")
    operation_names = ["sum_explicit_float32", "backward_without_requires", "result_requires_grad_true",
                       "backward_new_float32_leaf", "mutate_backing_zero"]
    mark_names = ["before_sum", "after_sum", "after_cast_witness_drop", "before_backward_without_requires",
                  "after_backward_without_requires", "after_requires_grad", "after_leaf_backward",
                  "after_grad_publish", "after_external_grad_drop", "after_seed_drop", "after_input_mutation",
                  "after_backing_drop", "after_input_drop", "after_result_drop"]
    for i, (spec, row) in enumerate(zip(CASES, report["rows"])):
        require(row["index"] == i and row["name"] == spec["name"] and row["spec"] == spec
                and row["status"] == "OBSERVED" and row["target_accepted"] is True, "ordered-reference-case", case=i)
        require([x["label"] for x in row["operations"]] == operation_names and
                [x["label"] for x in row["marks"]] == mark_names, "complete-reference-operation-and-mark-order", case=i)
        require([normalized(x) for x in row["operations"]] == [0, 5, 0, 0, 0], "observed-reference-operation-contract")
        witness = row["cast_witness"]
        require(all(witness[key] is True for key in ["matches_target", "cast_tensor_released",
                "cast_storage_released", "sum_tensor_released", "sum_storage_released"]), "real-cast-witness-and-release")
        require(witness["cast"]["dtype"] == witness["result"]["dtype"] == "torch.float32", "real-F32-cast-witness")
        require(all(x["rng_hex"] == report["rng_hex"] and x["rng_sha256"] == report["rng_sha256"] for x in row["marks"]),
                "every-mark-full-RNG", case=i)
        require(not any(row["final_tensor_weak_alive"].values()) and all(row["final_storage_weak_expired"].values()),
                "reference-final-Tensor-Storage-release", case=i)
    for spec, row in zip(AXIS_CONTROLS, report["axis_controls"]):
        require(row["spec"] == spec and row["name"] == spec["name"] and row["status"] == "OBSERVED"
                and row["source_spec"] == CASES[spec["case_index"]] and row["before"] == row["after"]
                and row["source_unchanged"] is True and row["warnings"] == []
                and row["rng_hex"] == report["rng_hex"] and row["rng_sha256"] == report["rng_sha256"],
                "exact-reference-axis-observation")
        require(normalized(dict(row["operation"], label="axis_control", warnings=[])) in (2, 7),
                "real-axis-error-classification")
    compressed = inputs.bind(args.repository / "validation/sum-bool27-reference/report.json.xz",
                             "353a8b2853773bc6678bb62fc9559d6f4244c571ba40c6b80f352b2649cd6edd")
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    raw_bool = decoder.decompress(compressed)
    require(decoder.eof and not decoder.unused_data and sha(raw_bool) ==
            "21dd75aeb86414818f7aa381f8b866d78b16f9bb3f5738e9a60d27ac3d15dd28", "unchanged-original-Bool-reference")
    bool_reference = json.loads(raw_bool)
    require(bool_reference["schema"] == "heptatorch.sum-bool.reference27.v1" and bool_reference["status"] == "OBSERVED"
            and bool_reference["cases"] == 27 and bool_reference["torch_executed"] is True,
            "actual-original-Bool-controls-reference")
    return report, bool_reference


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

    def sum_command(self, spec, dtype=0):
        axes = spec["axes"]
        payload = 0 if not axes else self.n.w.Payload(struct.pack("<" + "q" * len(axes), *axes))
        return (115, self.live["input"], MASK if axes is None else len(axes), payload,
                int(spec["keepdim"]), dtype, 0, 0)

    def drop(self, name):
        self.n.one(41, self.live.pop(name))

    def tensor(self, token):
        # The public getter is real. As in the original Sum32 observer, release
        # a newly published temporary getter result but keep an already-public
        # Tensor alive. This is required once the new F32 leaf owns a gradient.
        row = self.n.tensor(token)
        memory = self.n.memory()
        node = memory.node(memory.public_node(token))
        gindex = node[48] if node[45] else 0
        was_public = bool(memory.node(gindex)[25]) if gindex else False
        presence = self.n.one(59, token)["value"]
        gradient = self.n.one(91, token)["value"]
        self.check("gradient-getter-and-presence", bool(gradient), bool(presence))
        self.check("gradient-getter-and-real-private-owner", bool(gradient), bool(gindex))
        row["grad_is_none"] = gradient == 0
        if gradient:
            actual = self.n.tensor(gradient)
            self.check("gradient-getter-real-node", self.n.memory().public_node(gradient), gindex)
            row["observed_gradient"] = actual
            if not was_public:
                self.n.one(41, gradient)
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


def observe_cast_witness(driver, row):
    witness, spec = row["cast_witness"], row["spec"]
    # These short-lived Torch objects may have their addresses reused later.
    # Keep their identity namespace separate from long-lived case roles.
    old_tensor_ids, old_storage_ids = driver.tensor_ids, driver.storage_ids
    driver.tensor_ids, driver.storage_ids = {}, {}
    temporary = []
    try:
        cast = driver.n.one(108, driver.live["input"], 0, 0, 0)["value"]
        temporary.append(cast)
        actual = driver.tensor(cast)
        driver.compare_tensor(witness["cast"], actual, "real-cast-witness")
        source = driver.n.tensor(driver.live["input"])
        driver.check("cast-has-independent-Tensor", actual["native_incarnation"] != source["native_incarnation"], True)
        driver.check("cast-has-independent-Storage", actual["native_storage_id"] != source["native_storage_id"], True)
        cast_node, cast_inc = driver.n.memory().public_node(cast), actual["native_incarnation"]
        cast_weak = driver.n.one(42, cast)["value"]
        command = list(driver.sum_command(spec))
        command[1] = cast
        status, record = driver.issue(tuple(command), "real-cast-then-F32-sum")
        driver.check("cast-then-F32-sum-accepted", status, 0)
        summed = record["value"]
        temporary.append(summed)
        actual_sum = driver.tensor(summed)
        driver.compare_tensor(witness["result"], actual_sum, "real-cast-sum-witness")
        sum_node, sum_inc = driver.n.memory().public_node(summed), actual_sum["native_incarnation"]
        sum_weak = driver.n.one(42, summed)["value"]
        target = driver.n.tensor(driver.live["result"])
        for field in ("dtype", "shape", "strides", "logical_bytes_hex"):
            driver.check("native-target-versus-cast-witness/" + field, actual_sum[field], target[field])
        for token in reversed(temporary):
            driver.n.one(41, token)
        temporary.clear()
        memory = driver.n.memory()
        for name, index, incarnation, weak in [("cast", cast_node, cast_inc, cast_weak),
                                               ("sum", sum_node, sum_inc, sum_weak)]:
            node = memory.node(index)
            alive = bool(node[38] and node[30] and node[40] == incarnation)
            driver.check(name + "-witness-Tensor-released", not alive, witness[name + "_tensor_released"])
            status, record = driver.issue((43, weak), name + "-witness-StorageWeak")
            driver.check(name + "-witness-Storage-released", status == 6, witness[name + "_storage_released"])
            if status == 0:
                driver.n.one(45, record["value"])
            driver.n.one(44, weak)
    finally:
        driver.tensor_ids, driver.storage_ids = old_tensor_ids, old_storage_ids


def compare_case(driver, row):
    spec = row["spec"]
    marks = {x["label"]: x for x in row["marks"]}
    operations = {x["label"]: x for x in row["operations"]}
    driver.setup(spec)
    driver.mark(marks["before_sum"])
    status, record = driver.issue(driver.sum_command(spec), "sum_explicit_float32")
    driver.check("sum-actual-normalized-status", status, normalized(operations["sum_explicit_float32"]))
    driver.publish("result", record["value"])
    driver.mark(marks["after_sum"])
    observe_cast_witness(driver, row)
    driver.mark(marks["after_cast_witness_drop"])
    shape = marks["before_backward_without_requires"]["tensors"]["seed"]["shape"]
    seed = driver.n.create(0, shape, struct.pack("<f", 1.0) * product(shape), requires=0)
    driver.publish("seed", seed)
    driver.mark(marks["before_backward_without_requires"])
    status, _ = driver.issue((58, driver.live["result"], driver.live["seed"], 0),
                             "backward_without_requires", partial=True)
    driver.check("no-requires-backward-status", status, normalized(operations["backward_without_requires"]))
    driver.mark(marks["after_backward_without_requires"])
    status, _ = driver.issue((90, driver.live["result"], 1), "result_requires_grad_true")
    driver.check("new-F32-leaf-requires-status", status, normalized(operations["result_requires_grad_true"]))
    driver.mark(marks["after_requires_grad"])
    status, _ = driver.issue((58, driver.live["result"], driver.live["seed"], 0),
                             "backward_new_float32_leaf", partial=True)
    driver.check("new-F32-leaf-backward-status", status, normalized(operations["backward_new_float32_leaf"]))
    driver.mark(marks["after_leaf_backward"])
    gradient = driver.n.one(91, driver.live["result"])["value"]
    driver.check("real-result-gradient-present", bool(gradient), True)
    driver.publish("result_grad", gradient)
    driver.mark(marks["after_grad_publish"])
    driver.drop("result_grad")
    driver.mark(marks["after_external_grad_drop"])
    driver.drop("seed")
    driver.mark(marks["after_seed_drop"])
    status, _ = driver.issue((100, driver.live["backing"], 0, 0), "mutate_backing_zero")
    driver.check("mutation-actual-normalized-status", status, normalized(operations["mutate_backing_zero"]))
    driver.mark(marks["after_input_mutation"])
    driver.drop("backing")
    driver.mark(marks["after_backing_drop"])
    driver.drop("input")
    driver.mark(marks["after_input_drop"])
    driver.drop("result")
    driver.mark(marks["after_result_drop"])
    driver.check("all-reference-marks-completed", [x["label"] for x in driver.marks], [x["label"] for x in row["marks"]])
    driver.check("all-strong-roles-dropped", driver.live, {})
    driver.finish_controls()


def compare_axis_control(driver, row):
    driver.setup(row["source_spec"])
    # The axis producer records the source view and its ultimate base identity,
    # without a separate base snapshot. Anchor that identity to the actual
    # backing object created by the same recorded source construction.
    backing = driver.tensor(driver.live["backing"])
    base = row["before"]["base_tensor_identity"]
    require(base is not None, "axis-reference-source-has-real-base")
    driver.tensor_ids[(base, row["before"]["storage_identity"])] = backing["native_incarnation"]
    driver.storage_ids[row["before"]["storage_identity"]] = backing["native_storage_id"]
    driver.compare_tensor(row["before"], driver.tensor(driver.live["input"]), "axis-before")
    spec = {**row["source_spec"], "axes": row["spec"]["axes"], "keepdim": row["spec"]["keepdim"]}
    status, record = driver.issue(driver.sum_command(spec), row["name"])
    wanted = normalized(dict(row["operation"], label="axis_control", warnings=row["warnings"]))
    driver.check("actual-reference-axis-error", status, wanted)
    driver.check("axis-error-publishes-nothing", record, None)
    driver.compare_tensor(row["after"], driver.tensor(driver.live["input"]), "axis-after")
    driver.check("axis-complete-reference-RNG", driver.n.one(63, driver.gen)["payload"].hex(), row["rng_hex"])
    driver.drop("input")
    driver.drop("backing")
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
    report = dict(schema="heptatorch.sum-integral-f32.native17.v1", status="PREPARED_NOT_RUN",
                  expected_cases=17, expected_axis_controls=4, expected_dtype_controls=7,
                  expected_entry_checks=10, expected_allocation_controls=2,
                  cases=[], axis_controls=[], dtype_controls=[], entry_controls=[], allocation_controls=[],
                  source_sha256=args.expected_source_sha256, native_sha256=args.expected_native_sha256,
                  reference_sha256=args.expected_reference_sha256, native_executed=False,
                  Torch_executed=False, original700_or_original_controls_modified=False,
                  historical_Bool7_changed_indices=[2, 3], original_Bool7_unchanged_all_pass=False,
                  error=None, schema_closure=False, full_project_status="HOLD")
    exit_code = 0
    try:
        build = admit_build(args, inputs)
        reference, bool_reference = admit_reference(args, inputs)
        inputs.bind(args.protocol, PROTOCOL_SHA)
        inputs.bind(args.protocol.with_name("owner_client.py"), CLIENT_SHA)
        observer, owner, proof = make_observer(args.repository, inputs, SOURCE, args.expected_native_sha256)
        report.update(build_functions=build["functions"], cold_units=len(build["units"]), observer_adaptation=proof,
                      source_transition=build["admitted_source_transition"],
                      reference_cpu_capability=reference["cpu_capability"], reference_num_threads=reference["num_threads"])
        if not args.prepare_only:
            from controls import run_dtype_controls, run_entry_controls
            from allocation_controls import run_allocation_controls
            report["native_executed"] = True
            for row in reference["rows"]:
                report["cases"].append(run_one(args, observer, owner, args.output / ("case-%02d" % row["index"]),
                                               row["name"], lambda driver, row=row: compare_case(driver, row)))
            for i, row in enumerate(reference["axis_controls"]):
                report["axis_controls"].append(run_one(args, observer, owner, args.output / ("axis-control-%02d" % i),
                                                       row["name"], lambda driver, row=row: compare_axis_control(driver, row)))
            report["dtype_controls"] = run_dtype_controls(args, observer, owner, reference, bool_reference, run_one)
            report["entry_controls"] = run_entry_controls(args, observer, owner, run_one)
            report["allocation_controls"] = run_allocation_controls(args, observer, owner, run_one)
            require([len(report[key]) for key in ["cases", "axis_controls", "dtype_controls", "entry_controls", "allocation_controls"]]
                    == [17, 4, 7, 1, 2], "all-native-groups-mandatory-and-complete")
            all_rows = sum((report[key] for key in ["cases", "axis_controls", "dtype_controls", "entry_controls", "allocation_controls"]), [])
            report.update(status="PASS" if all(x["status"] == "PASS" for x in all_rows) else "FAIL",
                          passed=sum(x["status"] == "PASS" for x in report["cases"]),
                          failed=sum(x["status"] != "PASS" for x in report["cases"]),
                          children=len(all_rows), frames=sum(x["frames"] for x in all_rows))
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
