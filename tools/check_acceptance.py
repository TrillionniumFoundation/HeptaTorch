#!/usr/bin/env python3
"""Check immutable scope and source-owned full-contract qualification.

Current scoped kernels do not qualify complete PyTorch declarations. Future
closure must run the registered, independently implemented contract verifier;
neither a JSON status nor a finite-suite PASS can confer full qualification.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "heptatorch_acceptance_contracts", Path(__file__).with_name("acceptance_contracts.py"))
contracts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contracts)

PIN = "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457"
COUNT = 2590
SCHEMAS_SHA256 = "c2ebd09fea8de36e74301f48e7a1a84512a6ddbdf6b55138738712500b025718"
DERIVATIVE_COUNT = 688
DERIVATIVE_PATH = "tools/autograd/derivatives.yaml"
DERIVATIVE_BLOB = "d0ee6fb1e7b5261eb1e02eaf07cf08deff09a95f"
DERIVATIVE_SOURCE_SHA256 = "4a4d1ae1ddea605f57594de991da2c92e89ae11d6aa6edf348107589b548082c"
DERIVATIVE_SOURCE_BYTES = 188779
DERIVATIVES_SHA256 = "5e32bf19142454911112779474e2c52eab08a6789229b46b96a92246c578bac8"
# Pin the original subsystem obligations independently of the editable ledger.
GATE_REQUIREMENTS_SHA256 = "5e8c99421b69070f0197e19d9e9c03cdca1572726d070c6eff74d759712c584b"
GATES = frozenset((
    "module-abi", "storage-lifetime", "dtype-layout", "native-schemas",
    "views-mutation", "autodiff-transforms", "frontend-dispatch",
    "neural-networks", "optimizers", "rng-determinism", "serialization",
    "accelerators", "distributed", "compiler-export", "data-pipelines",
    "platform-packaging", "final-integration",
))


def qualified_records(root, rows, kind, field, open_statuses, closed_status, registry):
    verified, errors = {}, []
    for row in rows:
        name = row.get(field) if isinstance(row, dict) else None
        try:
            contracts.need(isinstance(row, dict) and isinstance(name, str),
                           "invalid contract record")
            status = row.get("status")
            contracts.need(status in open_statuses | {closed_status}, "unknown qualification status")
            if status == closed_status:
                contracts.need("qualification" in row, "full qualification evidence is missing")
                verified[name] = contracts.admit(root, kind, name, row["qualification"], registry)
            else:
                contracts.need("qualification" not in row,
                               "unqualified record must not carry a full qualification claim")
        except Exception as error:
            # A missing/failed/exceptional verifier always fails closed. It does
            # not convert unavailable evidence into an unimplemented success.
            errors.append({"kind": kind, "name": name,
                           "error": type(error).__name__ + ": " + str(error)})
    return verified, errors


def check(root):
    def read(path):
        return contracts.json_object((root / path).read_bytes())

    inventory = read("compatibility/upstream-schemas.json")
    derivative_inventory = read("compatibility/upstream-derivatives.json")
    upstream = read("compatibility/upstream-lock.json")
    capabilities = read("compatibility/capabilities.json")
    ledger = read("compatibility/full-acceptance.json")
    checks = {}
    schemas = inventory["schemas"]
    declarations = [row["schema"] for row in schemas]
    checks["upstream_identity"] = all(v == PIN for v in (
        inventory["upstream"]["commit"], upstream["commit"],
        capabilities["upstream_commit"], ledger["upstream_commit"],
        derivative_inventory["upstream"]["commit"]))
    checks["schema_count"] = all(type(v) is int and v == COUNT for v in (
        len(declarations), len(set(declarations)), inventory["count"],
        capabilities["native_schema_inventory_count"], ledger["native_schema_count"]))
    digest = hashlib.sha256(("\n".join(declarations) + "\n").encode()).hexdigest()
    checks["exact_schema_denominator"] = digest == ledger["native_schema_lines_sha256"] == SCHEMAS_SHA256
    # The original validator checked only native schemas. In particular, both
    # recorded derivative counts could be erased while every check still passed.
    # Retain the complete ordered names independently of those mutable counts.
    derivatives = derivative_inventory["derivatives"]
    derivative_names = [row["name"] for row in derivatives]
    derivative_sources = [row for row in upstream["source_files"]
                          if row.get("path") == DERIVATIVE_PATH]
    derivative_source = derivative_sources[0] if len(derivative_sources) == 1 else {}
    source = derivative_inventory["upstream"]
    checks["derivative_source_identity"] = (
        derivative_inventory["format"] == "heptatorch.upstream-derivative-inventory.v1"
        and source["repository"] == upstream["repository"] == "pytorch/pytorch"
        and source["version"] == upstream["version"] == "2.15.0"
        and source["tag"] == upstream["tag"] == "v2.15.0"
        and source["source_path"] == DERIVATIVE_PATH
        and source["source_git_blob_sha1"] == derivative_source.get("git_blob_sha1") == DERIVATIVE_BLOB
        and source["source_sha256"] == DERIVATIVE_SOURCE_SHA256
        and source["source_bytes"] == DERIVATIVE_SOURCE_BYTES
        and source["source_url"] == "https://github.com/pytorch/pytorch/blob/" + PIN + "/" + DERIVATIVE_PATH)
    counts = (len(derivative_names), len(set(derivative_names)),
              derivative_inventory["count"],
              capabilities["derivative_declaration_inventory_count"],
              ledger["derivative_declaration_count"],
              derivative_source.get("named_entry_count"))
    checks["derivative_declaration_count"] = all(
        type(value) is int and value == DERIVATIVE_COUNT for value in counts)
    derivative_digest = hashlib.sha256(("\n".join(derivative_names) + "\n").encode()).hexdigest()
    checks["exact_derivative_denominator"] = (
        derivative_digest == derivative_inventory["declaration_lines_sha256"]
        == ledger["derivative_declaration_lines_sha256"] == DERIVATIVES_SHA256)
    gates = ledger["gates"]
    ids = [row["id"] for row in gates]
    checks["all_subsystems_retained"] = len(ids) == len(GATES) and set(ids) == GATES
    gate_requirements = {row["id"]: row.get("required") for row in gates}
    requirements_bytes = json.dumps(gate_requirements, sort_keys=True,
                                    ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    requirements_digest = hashlib.sha256(requirements_bytes).hexdigest()
    checks["exact_subsystem_requirements"] = (
        checks["all_subsystems_retained"] and requirements_digest == GATE_REQUIREMENTS_SHA256)
    checks["continuation_plan_exists"] = (
        ledger["plan"] == "docs/continuation-plan.md"
        and (root / ledger["plan"]).is_file())

    # The registry comes only from the source-owned sibling module. Evidence
    # cannot introduce a callable, import path, command, target or scope change.
    registry = contracts.VERIFIERS
    targets = ({("native", name) for name in declarations}
               | {("derivative", name) for name in derivative_names}
               | {("subsystem", name) for name in GATES})
    checks["registered_verifier_targets_valid"] = set(registry) <= targets
    native, native_errors = qualified_records(
        root, schemas, "native", "schema", {"unimplemented", "in_progress"}, "qualified", registry)
    derivative, derivative_errors = qualified_records(
        root, derivatives, "derivative", "name", {"unimplemented", "in_progress"}, "qualified", registry)
    subsystem, subsystem_errors = qualified_records(
        root, gates, "subsystem", "id", {"OPEN", "IN_PROGRESS"}, "CLOSED", registry)
    errors = native_errors + derivative_errors + subsystem_errors
    checks["full_schema_qualification_not_inferred"] = (
        not native_errors
        and type(capabilities["full_upstream_schema_parity_count"]) is int
        and capabilities["full_upstream_schema_parity_count"] == len(native))
    checks["full_derivative_qualification_not_inferred"] = (
        not derivative_errors
        and type(capabilities["full_derivative_declaration_parity_count"]) is int
        and capabilities["full_derivative_declaration_parity_count"] == len(derivative))
    checks["no_unverified_subsystem_closure"] = not subsystem_errors and all(
        isinstance(row["required"], str) and bool(row["required"].strip())
        and isinstance(row["remaining"], str)
        and (row["remaining"] == "" if row["id"] in subsystem else bool(row["remaining"].strip()))
        for row in gates)
    # Capability prose is descriptive, never an alternate qualification route.
    # Its full-schema flag cannot be used while any native declaration is open.
    checks["capability_flags_do_not_infer_full_contracts"] = all(
        row["full_upstream_schema_qualified"] is False
        or (row["full_upstream_schema_qualified"] is True and len(native) == COUNT)
        for row in capabilities["capabilities"])
    complete = (all(checks.values()) and len(native) == COUNT
                and len(derivative) == DERIVATIVE_COUNT and set(subsystem) == GATES)
    checks["completion_flags_agree"] = (
        ledger["whole_pytorch_rewrite_complete"] is complete
        and capabilities["whole_pytorch_rewrite_complete"] is complete
        and ledger["status"] == ("PASS" if complete else "HOLD"))
    passed = all(checks.values())
    return {
        "schema": "heptatorch.acceptance-accounting.v1",
        "accounting_status": "PASS" if passed else "FAIL",
        "whole_project_status": "PASS" if passed and complete else "HOLD",
        "whole_pytorch_rewrite_complete": passed and complete,
        "full_upstream_schema_parity_count": len(native),
        "native_schema_inventory_count": len(schemas),
        "derivative_declaration_inventory_count": len(derivatives),
        "derivative_declaration_lines_sha256": derivative_digest,
        "full_derivative_declaration_parity_count": len(derivative),
        "open_full_subsystem_gates": len(GATES - set(subsystem)),
        "subsystem_requirements_sha256": requirements_digest,
        "registered_full_contract_verifiers": len(registry),
        "acceptance_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "admission_engine_sha256": hashlib.sha256(Path(contracts.__file__).read_bytes()).hexdigest(),
        "qualified_contracts": {"native": list(native.values()),
                                "derivative": list(derivative.values()),
                                "subsystem": list(subsystem.values())},
        "qualification_errors": errors,
        "checks": checks,
        "limitation": ("No production full-contract verifiers are registered."
                       if not registry else
                       "Only contracts accepted by their registered complete-contract verifier are counted."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        report = check(args.root.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"schema": "heptatorch.acceptance-accounting.v1",
                          "accounting_status": "FAIL", "whole_project_status": "HOLD",
                          "error": type(exc).__name__ + ": " + str(exc)}))
        return 1
    print(json.dumps(report, sort_keys=True))
    if report["accounting_status"] != "PASS":
        return 1
    return 2 if args.require_complete and not report["whole_pytorch_rewrite_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
