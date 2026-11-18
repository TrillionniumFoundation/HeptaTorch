#!/usr/bin/env python3
"""Observe integral-input, explicit-F32 sum on the exact pinned CPU wheel.

This producer imports neither an HIA image nor a host numerical expectation
model. It records a separate real Torch cast-then-sum witness, including cast
layout, alongside the target operation. The producer has a no-Torch prepare mode.
"""
import argparse
import gc
import json
import runpy
import sys
import traceback
import warnings
import weakref
from pathlib import Path

from cases import AXIS_CONTROLS, CASES, CASE_SHA256, UPSTREAM
from common import Inputs, bind_sources, dump, require, sha

SUPPORT_SHA = "ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae"
LOCK_SHA = "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944"
CHECKS = {"python_matches", "platform_matches", "version_backend_matches",
          "git_matches_oracle_build_pin", "distribution_version_matches",
          "direct_url_matches", "direct_url_archive_sha256_matches",
          "pip_report_url_matches", "pip_report_archive_sha256_matches",
          "pip_report_version_matches", "pip_report_requested"}


def tensor(torch, value):
    logical = bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
    physical = bytes(value.untyped_storage())
    return dict(dtype=str(value.dtype), device=str(value.device), element_size=value.element_size(),
                shape=list(value.shape), strides=list(value.stride()), numel=value.numel(),
                storage_bytes=value.untyped_storage().nbytes(),
                storage_offset_elements=value.storage_offset(),
                storage_offset_bytes=value.storage_offset() * value.element_size(),
                version=value._version, requires_grad=value.requires_grad, is_leaf=value.is_leaf,
                is_view=value._is_view(), grad_fn=None if value.grad_fn is None else type(value.grad_fn).__name__,
                grad_is_none=value.grad is None, retains_grad=value.retains_grad,
                logical_bytes=len(logical), logical_bytes_hex=logical.hex(), logical_sha256=sha(logical),
                storage_bytes_hex=physical.hex(), storage_sha256=sha(physical),
                tensor_identity=str(value._cdata), storage_identity=str(value.untyped_storage()._cdata),
                base_tensor_identity=None if value._base is None else str(value._base._cdata))


def observe(torch, spec, index):
    from torch.multiprocessing.reductions import StorageWeakRef
    live, tensor_refs, storage_refs = {}, {}, {}
    marks, operations = [], []

    def publish(name, value):
        require(name not in live and name not in tensor_refs, "new-reference-role", name=name)
        live[name] = value
        tensor_refs[name] = weakref.ref(value)
        storage_refs[name] = StorageWeakRef(value.untyped_storage())

    def mark(label):
        gc.collect()
        tensors = {name: tensor(torch, value) for name, value in live.items()}
        names = list(tensors)
        relations = {}
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                a, b = tensors[left], tensors[right]
                relations[left + "|" + right] = dict(
                    same_Tensor=a["tensor_identity"] == b["tensor_identity"],
                    same_Storage=a["storage_identity"] == b["storage_identity"],
                    same_current_Version_value=a["version"] == b["version"])
        rng = bytes(torch.get_rng_state().tolist())
        require(len(rng) == 5056, "full-reference-RNG-width")
        marks.append(dict(label=label, live_roles=names, tensors=tensors, relations=relations,
                          tensor_weak_alive={name: ref() is not None for name, ref in tensor_refs.items()},
                          storage_weak_expired={name: ref.expired() for name, ref in storage_refs.items()},
                          rng_hex=rng.hex(), rng_sha256=sha(rng)))

    def perform(label, function, publish_as=None):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value = function()
                if publish_as is not None:
                    publish(publish_as, value)
                row = dict(label=label, accepted=True, error=None)
                del value
            except Exception as error:
                row = dict(label=label, accepted=False,
                           error=dict(type=type(error).__name__, message=str(error)))
        row["warnings"] = [dict(type=type(x.message).__name__, message=str(x.message)) for x in caught]
        operations.append(row)
        return row["accepted"]

    publish("backing", torch.tensor(spec["values"], dtype=getattr(torch, spec["dtype"]), device="cpu"))
    publish("input", live["backing"].as_strided(spec["shape"], spec["strides"], spec["offset"]))
    mark("before_sum")
    accepted = perform("sum_explicit_float32", lambda: torch.ops.aten.sum.dim_IntList(
        live["input"], spec["axes"], keepdim=spec["keepdim"], dtype=torch.float32), "result")
    mark("after_sum")
    cast_witness = None
    if accepted:
        # This is an independent actual Torch observation, not an expected
        # number computed from Python integers. It also exposes preserve-format
        # conversion geometry for transposed, gapped, and overlapping inputs.
        cast = live["input"].to(dtype=torch.float32)
        cast_sum = torch.ops.aten.sum.dim_IntList(cast, spec["axes"],
                                                keepdim=spec["keepdim"], dtype=torch.float32)
        cast_weak, cast_storage_weak = weakref.ref(cast), StorageWeakRef(cast.untyped_storage())
        cast_sum_weak = weakref.ref(cast_sum)
        cast_sum_storage_weak = StorageWeakRef(cast_sum.untyped_storage())
        cast_witness = dict(cast=tensor(torch, cast), result=tensor(torch, cast_sum))
        target = tensor(torch, live["result"])
        keys = ("dtype", "shape", "strides", "logical_bytes_hex")
        cast_witness["matches_target"] = all(cast_witness["result"][key] == target[key] for key in keys)
        del cast, cast_sum, target
        gc.collect()
        cast_witness.update(cast_tensor_released=cast_weak() is None,
                            cast_storage_released=cast_storage_weak.expired(),
                            sum_tensor_released=cast_sum_weak() is None,
                            sum_storage_released=cast_sum_storage_weak.expired())
        mark("after_cast_witness_drop")
        publish("seed", torch.ones_like(live["result"]))
        mark("before_backward_without_requires")
        perform("backward_without_requires", lambda: torch.autograd.backward([live["result"]], [live["seed"]]))
        mark("after_backward_without_requires")
        perform("result_requires_grad_true", lambda: live["result"].requires_grad_(True))
        mark("after_requires_grad")
        backward_accepted = perform("backward_new_float32_leaf",
                                    lambda: torch.autograd.backward([live["result"]], [live["seed"]]))
        mark("after_leaf_backward")
        if backward_accepted and live["result"].grad is not None:
            publish("result_grad", live["result"].grad)
            mark("after_grad_publish")
            del live["result_grad"]
            mark("after_external_grad_drop")
        del live["seed"]
        mark("after_seed_drop")
    perform("mutate_backing_zero", lambda: live["backing"].fill_(0))
    mark("after_input_mutation")
    del live["backing"]
    mark("after_backing_drop")
    del live["input"]
    mark("after_input_drop")
    if accepted:
        del live["result"]
        mark("after_result_drop")
    require(not live, "reference-all-strong-roles-dropped")
    require(all(x["rng_hex"] == marks[0]["rng_hex"] for x in marks), "reference-RNG-unchanged")
    return dict(index=index, name=spec["name"], spec=spec, status="OBSERVED", operations=operations,
                marks=marks, target_accepted=accepted, cast_witness=cast_witness,
                final_tensor_weak_alive=marks[-1]["tensor_weak_alive"],
                final_storage_weak_expired=marks[-1]["storage_weak_expired"])


def observe_axis_control(torch, spec):
    base_spec = CASES[spec["case_index"]]
    backing = torch.tensor(base_spec["values"], dtype=getattr(torch, base_spec["dtype"]), device="cpu")
    value = backing.as_strided(base_spec["shape"], base_spec["strides"], base_spec["offset"])
    before = tensor(torch, value)
    rng_before = bytes(torch.get_rng_state().tolist())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = torch.ops.aten.sum.dim_IntList(value, spec["axes"],
                                                  keepdim=spec["keepdim"], dtype=torch.float32)
            operation = dict(accepted=True, error=None, result=tensor(torch, result))
            del result
        except Exception as error:
            operation = dict(accepted=False, error=dict(type=type(error).__name__, message=str(error)), result=None)
    after = tensor(torch, value)
    rng_after = bytes(torch.get_rng_state().tolist())
    require(before == after, "axis-control-source-unchanged")
    require(rng_before == rng_after and len(rng_before) == 5056, "axis-control-RNG-unchanged")
    return dict(name=spec["name"], spec=spec, source_spec=base_spec, status="OBSERVED",
                operation=operation, before=before, after=after, source_unchanged=True,
                warnings=[dict(type=type(x.message).__name__, message=str(x.message)) for x in caught],
                rng_hex=rng_before.hex(), rng_sha256=sha(rng_before))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(not sys.flags.optimize, "assertions-enabled")
    repo = args.repository.resolve()
    require(Path.cwd().resolve() == repo, "reference-cwd-is-pinned-repository")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    bind_sources(inputs, Path(__file__).parent)
    inputs.bind(repo / "tests/pytorch_oracle.py", SUPPORT_SHA)
    lock = json.loads(inputs.bind(repo / "compatibility/pytorch-oracle-lock.json", LOCK_SHA))
    inputs.bind(repo / "compatibility/upstream-lock.json")
    report = dict(schema="heptatorch.sum-integral-f32.reference17.v1", status="PREPARED_NOT_RUN",
                  case_sha256=CASE_SHA256, expected_cases=len(CASES), case_plan=CASES,
                  axis_control_plan=AXIS_CONTROLS, expected_axis_controls=len(AXIS_CONTROLS),
                  inventory_commit=UPSTREAM, oracle_build_commit=lock["oracle"]["git_commit"],
                  source_sha256=sha(Path(__file__).read_bytes()), rows=[], axis_controls=[], cases=0,
                  torch_executed=False, native_executed=False, schema_closure=False,
                  full_project_status="HOLD", error=None)
    exit_code = 0
    try:
        if not args.prepare_only:
            inputs.bind(repo / lock["oracle"]["install_report"])
            saved_argv = sys.argv
            try:
                sys.argv = [str(repo / "tests/pytorch_oracle.py"), str(out / "NO_NATIVE_EXECUTION"),
                            str(out / "oracle-provenance.json"), str(repo / "compatibility/pytorch-oracle-lock.json")]
                context = runpy.run_path(str(repo / "tests/pytorch_oracle.py"), run_name="sum_f32_reference_only")
            finally:
                sys.argv = saved_argv
            torch, provenance = context["torch"], context["PROVENANCE"]
            require(provenance["status"] == "PASS" and set(provenance["checks"]) == CHECKS
                    and all(x is True for x in provenance["checks"].values()), "fixed-Torch-provenance")
            report.update(torch_executed=True, oracle_provenance=provenance,
                          torch_version=str(torch.__version__), torch_git_version=torch.version.git_version,
                          cpu_capability=torch._C._get_cpu_capability(), num_threads=torch.get_num_threads())
            before = bytes(torch.get_rng_state().tolist())
            for i, spec in enumerate(CASES):
                row = observe(torch, spec, i)
                dump(out / ("case-%02d.json" % i), row)
                report["rows"].append(row)
            for i, spec in enumerate(AXIS_CONTROLS):
                row = observe_axis_control(torch, spec)
                dump(out / ("axis-control-%02d.json" % i), row)
                report["axis_controls"].append(row)
            require(before == bytes(torch.get_rng_state().tolist()), "reference-whole-RNG-unchanged")
            report.update(status="OBSERVED", cases=len(report["rows"]),
                          accepted=sum(x["target_accepted"] for x in report["rows"]),
                          cast_witness_matches=sum(bool(x["cast_witness"] and x["cast_witness"]["matches_target"])
                                                   for x in report["rows"]),
                          axis_controls_rejected=sum(not x["operation"]["accepted"] for x in report["axis_controls"]),
                          global_rng_unchanged=True, rng_hex=before.hex(), rng_sha256=sha(before))
        report["inputs_unchanged"] = inputs.unchanged()
    except BaseException as error:
        report.update(status="FAIL" if report["torch_executed"] else "NOT_RUN",
                      error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        exit_code = 1
    report["inputs"] = inputs.rows
    dump(out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "cases", "expected_cases", "torch_executed", "error")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
