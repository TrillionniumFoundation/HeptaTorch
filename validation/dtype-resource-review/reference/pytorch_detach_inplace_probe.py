#!/usr/bin/env python3
"""Observe pinned Torch detach_ view admission and Tensor/Storage/Version/grad identity.

Reference only. Every outcome comes from actual Torch. There is no HIA execution,
no numerical expected-result model, and no alteration of the older dropout probes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback
import warnings

SUPPORT_SHA = "ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae"
LOCK_SHA = "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944"
PARENT28_SHA = "bc5a490f1623a4078050829798f502b0be934722f371adbcbb5612aaeb1a563c"
PARENT9_SHA = "de7db2f985873be797bd7e167fb374f82b5690021e0a0541d2fb4f7a36b7ad83"
SOURCE_REF = "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457"
MANUAL_SOURCE_SHA = "3225c58e4eeec4520db8993245ab0458293480d9e76912826e0ce3b9e2a9034e"
CHECKS = {"python_matches", "platform_matches", "version_backend_matches",
          "git_matches_oracle_build_pin", "distribution_version_matches", "direct_url_matches",
          "direct_url_archive_sha256_matches", "pip_report_url_matches", "pip_report_archive_sha256_matches",
          "pip_report_version_matches", "pip_report_requested"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plan():
    rows = []
    for dtype in ("float32", "bool"):
        for construction in ("direct", "clone", "detach_alias", "view", "transpose"):
            for empty in (False, True):
                rows.append({"name": f"{dtype}-{construction}-{'empty' if empty else 'six'}",
                             "family": "view-admission", "dtype": dtype,
                             "construction": construction, "empty": empty})
    for construction in ("requires_leaf_existing_grad", "nonleaf_no_retain", "nonleaf_retained_existing_grad",
                         "nonleaf_retained_no_grad", "current_leaf_gradient", "current_assigned_strided_gradient"):
        rows.append({"name": construction.replace("_", "-"), "family": "autograd-identity",
                     "dtype": "float32", "construction": construction, "empty": False})
    assert len(rows) == 26 and len({r["name"] for r in rows}) == 26
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "lock", "repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), "Output is create-only; preserve every earlier observation"
    inputs = {Path(__file__).resolve(): Path(__file__).read_bytes(),
              args.reference.resolve(): args.reference.read_bytes(), args.lock.resolve(): args.lock.read_bytes()}
    assert sha(inputs[args.reference.resolve()]) == SUPPORT_SHA
    assert sha(inputs[args.lock.resolve()]) == LOCK_SHA
    planned = plan()
    common = {"schema": "heptatorch.pinned-torch-detach-inplace-probe.v1",
              "probe_sha256": sha(inputs[Path(__file__).resolve()]), "expected_cases": len(planned), "case_plan": planned,
              "unmodified_prior_observations": {"native_dropout28": PARENT28_SHA, "empty_identity9": PARENT9_SHA},
              "primary_source": {"ref": SOURCE_REF, "path": "torch/csrc/autograd/VariableTypeManual.cpp",
                                 "sha256": MANUAL_SOURCE_SHA,
                                 "url": f"https://github.com/pytorch/pytorch/blob/{SOURCE_REF}/torch/csrc/autograd/VariableTypeManual.cpp"},
              "comparison_policy": "OBSERVATION ONLY: acceptance, errors, view status and identities are not predicted or substituted.",
              "native_executed": False, "native_values_compared": 0,
              "full_pytorch_rewrite_complete": False, "pytorch_api_denominator": 2590}
    if args.prepare_only:
        result = {**common, "status": "PREPARED_NOT_RUN", "torch_executed": False, "cases": 0,
                  "inputs_sha256": {str(p): sha(b) for p, b in inputs.items()}}
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": result["status"], "expected_cases": len(planned), "probe_sha256": common["probe_sha256"]}))
        return 0

    assert Path.cwd().resolve() == args.repo.resolve(), "Use the locked repository's real pip report and inventory"
    for relative in ("compatibility/upstream-lock.json", "build/pytorch-install.json"):
        path = (args.repo / relative).resolve()
        inputs[path] = path.read_bytes()
    old_argv = sys.argv
    try:
        sys.argv = [str(args.reference), str(args.repo / "reference-only-no-native-execution"),
                    str(args.output.with_suffix(".provenance.json")), str(args.lock)]
        context = runpy.run_path(str(args.reference), run_name="detach_inplace_provenance_only")
    finally:
        sys.argv = old_argv
    torch, provenance = context["torch"], context["PROVENANCE"]
    assert provenance["status"] == "PASS" and set(provenance["checks"]) == CHECKS
    assert all(v is True for v in provenance["checks"].values())
    states, rows, errors = {}, [], []

    def capture(operation):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value = operation()
                event = {"accepted": True, "outcome": "SUCCEEDED", "error": None}
            except Exception as error:
                value = None
                event = {"accepted": False, "outcome": "RAISED",
                         "error": {"type": type(error).__name__, "message": str(error)}}
        event["warnings"] = [{"type": type(w.message).__name__, "message": str(w.message)} for w in caught]
        return value, event

    def state():
        raw = bytes(torch.get_rng_state().tolist())
        assert len(raw) == 5056
        digest = sha(raw)
        states.setdefault(digest, {"bytes": len(raw), "hex": raw.hex()})
        return digest

    def words(tensor):
        flat = tensor.detach().contiguous().reshape(-1)
        if tensor.dtype == torch.float32:
            return [int(v) & 0xffffffff for v in flat.view(torch.int32).tolist()]
        return [int(v) for v in flat.tolist()]

    def tensor_info(tensor):
        if tensor is None:
            return None
        base = tensor._base
        node = tensor.grad_fn
        return {"python_identity": str(id(tensor)), "tensor_identity": str(tensor._cdata),
                "storage_identity": str(tensor.untyped_storage()._cdata),
                "storage_bytes": tensor.untyped_storage().nbytes(), "element_size": tensor.element_size(),
                "dtype": str(tensor.dtype), "shape": list(tensor.shape), "strides": list(tensor.stride()),
                "storage_offset_elements": tensor.storage_offset(), "version": int(tensor._version),
                "requires_grad": bool(tensor.requires_grad), "is_leaf": bool(tensor.is_leaf),
                "retains_grad": bool(tensor.retains_grad), "is_view": bool(tensor._is_view()),
                "base_tensor_identity": None if base is None else str(base._cdata),
                "grad_fn": None if node is None else type(node).__name__,
                "output_nr": getattr(tensor, "output_nr", None), "binary_words": words(tensor)}

    def snapshot(tensor):
        gradient, event = capture(lambda: tensor.grad)
        return {"tensor": tensor_info(tensor), "gradient": tensor_info(gradient), "gradient_read": event}

    def base(dtype, shape):
        if any(d == 0 for d in shape):
            return torch.empty(shape, dtype=dtype)
        values = [[.25, -.5, 1.], [2., -4., 8.]] if dtype == torch.float32 else [[False, True, True], [False, True, False]]
        # A direct, owned Tensor, not arange(...).reshape(...) view metadata.
        if shape == (3, 2):
            values = [[values[0][0], values[0][1]], [values[0][2], values[1][0]], [values[1][1], values[1][2]]]
        return torch.tensor(values, dtype=dtype)

    def construct(spec):
        dtype = getattr(torch, spec["dtype"])
        form = spec["construction"]
        shape = (2, 0, 3) if spec["empty"] else (2, 3)
        holder = None
        if spec["family"] == "view-admission":
            if form == "transpose":
                source = base(dtype, (3, 0, 2) if spec["empty"] else (3, 2))
                target = source.transpose(0, 2 if spec["empty"] else 1)
            else:
                source = base(dtype, shape)
                if form == "direct":
                    target = source
                elif form == "clone":
                    target = source.clone()
                elif form == "detach_alias":
                    target = source.detach()
                elif form == "view":
                    target = source.view(shape)
                else:
                    raise AssertionError(form)
            return target, {"construction_source": source}, holder
        source = base(torch.float32, (2, 3)).requires_grad_(True)
        if form == "requires_leaf_existing_grad":
            source.sum().backward()
            target = source
        elif form == "nonleaf_no_retain":
            target = source * 2.
        elif form in ("nonleaf_retained_existing_grad", "nonleaf_retained_no_grad"):
            target = source * 2.
            target.retain_grad()
            if form == "nonleaf_retained_existing_grad":
                target.sum().backward(retain_graph=True)
        elif form == "current_leaf_gradient":
            (source * source).sum().backward()
            holder, target = source, source.grad
        elif form == "current_assigned_strided_gradient":
            source.grad = base(torch.float32, (3, 2)).transpose(0, 1)
            holder, target = source, source.grad
        else:
            raise AssertionError(form)
        return target, {"construction_source": source}, holder

    def observe(spec):
        target, aliases, holder = construct(spec)
        # Explicit retained references support identity comparison only. This
        # probe makes no weak-lifetime, final-free or reference-count claim.
        held_gradient, gradient_event = capture(lambda: target.grad)
        held_holder_gradient = None if holder is None else holder.grad
        witness = target.detach()
        before = snapshot(target)
        alias_before = {name: snapshot(value) for name, value in aliases.items()}
        holder_before = None if holder is None else snapshot(holder)
        witness_before = tensor_info(witness)
        before_rng = state()
        returned, event = capture(lambda: target.detach_())
        after_rng = state()
        after = snapshot(target)
        alias_after = {name: snapshot(value) for name, value in aliases.items()}
        holder_after = None if holder is None else snapshot(holder)
        current_gradient, current_gradient_event = capture(lambda: target.grad)
        result = {"before": before, "after_detach": after, "operation": event,
                  "returned_is_target_object": None if returned is None else returned is target,
                  "returned_tensor_identity": None if returned is None else str(returned._cdata),
                  "target_tensor_identity_unchanged": before["tensor"]["tensor_identity"] == after["tensor"]["tensor_identity"],
                  "target_storage_identity_unchanged": before["tensor"]["storage_identity"] == after["tensor"]["storage_identity"],
                  "target_version_value_unchanged": before["tensor"]["version"] == after["tensor"]["version"],
                  "target_grad_object_is_original": current_gradient is held_gradient,
                  "held_gradient_before_read": gradient_event, "current_gradient_after_read": current_gradient_event,
                  "held_gradient_after_detach": tensor_info(held_gradient),
                  "aliases_before": alias_before, "aliases_after_detach": alias_after,
                  "holder_before": holder_before, "holder_after_detach": holder_after,
                  "holder_grad_is_original_object": None if holder is None else holder.grad is held_holder_gradient,
                  "rng_before_detach": before_rng, "rng_after_detach": after_rng,
                  "rng_unchanged_by_detach": before_rng == after_rng,
                  "entire_snapshot_unchanged_after_rejection": None if event["accepted"] else before == after}
        # This distinct, recorded operation observes shared Version continuity:
        # witness was created BEFORE detach_ and has its own Tensor identity.
        # Version identities are not directly exposed; do not infer them merely
        # from equal version numbers before/after the detach operation.
        def mutate_witness():
            with torch.no_grad():
                if witness.dtype == torch.bool:
                    witness.logical_not_()
                else:
                    witness.add_(.25)
        _, witness_event = capture(mutate_witness)
        result["version_sharing_observation"] = {"witness_before_detach": witness_before,
            "operation_after_detach": witness_event, "witness_after_mutation": tensor_info(witness),
            "target_after_witness_mutation": snapshot(target),
            "aliases_after_witness_mutation": {name:snapshot(value) for name,value in aliases.items()},
            "holder_after_witness_mutation": None if holder is None else snapshot(holder),
            "direct_version_identity_exposed": False}
        return result

    with torch.random.fork_rng(devices=[]):
        for index, spec in enumerate(planned):
            try:
                torch.manual_seed(1729)
                before = state()
                observed = observe(spec)
                after = state()
                rows.append({"index":index, **spec, "status":"OBSERVED", "observation":observed,
                             "complete_state_before":before, "complete_state_after":after,
                             "whole_case_rng_unchanged":before==after})
            except BaseException as error:
                failure = {"index":index, "name":spec["name"], "error":repr(error), "traceback":traceback.format_exc()}
                errors.append(failure)
                rows.append({**failure, "status":"PROBE_ERROR"})
    assert all(path.read_bytes() == content for path, content in inputs.items())
    result = {**common, "status":"OBSERVED" if not errors else "PROBE_ERROR", "torch_executed":True,
              "oracle_provenance":provenance, "cases":len(rows), "rows":rows, "probe_errors":errors,
              "probe_error_count":len(errors), "states":states, "unique_complete_states":len(states),
              "source_files_unchanged":True, "inputs_sha256":{str(p):sha(b) for p,b in inputs.items()},
              "cpu_capability":torch.backends.cpu.get_cpu_capability(), "threads":torch.get_num_threads(),
              "scope":"26 direct observations. The separate post-detach witness write observes Version sharing; it is not part of detach_. Strong identity witnesses remain live, so no weak/final-free claim is made."}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status":result["status"], "cases":len(rows), "probe_error_count":len(errors),
                      "output_bytes":args.output.stat().st_size, "output_sha256":sha(args.output.read_bytes())}))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
