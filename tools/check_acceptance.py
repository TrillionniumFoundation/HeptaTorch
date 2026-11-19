#!/usr/bin/env python3
"""Check immutable target accounting; never promote scoped tests to full parity.

This ledger validator is a build/audit host tool, not a numerical runtime or a
subsystem qualification runner. Version 1 has no qualified full subsystem. A
future closure needs an implemented verifier for its complete evidence contract;
changing a JSON status alone must not turn this gate green.
"""
import argparse
import hashlib
import json
from pathlib import Path

PIN = "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457"
COUNT = 2590
SCHEMAS_SHA256 = "c2ebd09fea8de36e74301f48e7a1a84512a6ddbdf6b55138738712500b025718"
DERIVATIVE_COUNT = 688
DERIVATIVE_PATH = "tools/autograd/derivatives.yaml"
DERIVATIVE_BLOB = "d0ee6fb1e7b5261eb1e02eaf07cf08deff09a95f"
DERIVATIVE_SOURCE_SHA256 = "4a4d1ae1ddea605f57594de991da2c92e89ae11d6aa6edf348107589b548082c"
DERIVATIVE_SOURCE_BYTES = 188779
DERIVATIVES_SHA256 = "5e32bf19142454911112779474e2c52eab08a6789229b46b96a92246c578bac8"
GATES = frozenset((
    "module-abi", "storage-lifetime", "dtype-layout", "native-schemas",
    "views-mutation", "autodiff-transforms", "frontend-dispatch",
    "neural-networks", "optimizers", "rng-determinism", "serialization",
    "accelerators", "distributed", "compiler-export", "data-pipelines",
    "platform-packaging", "final-integration",
))


def check(root):
    def read(path):
        return json.loads((root / path).read_text())

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
    checks["schema_count"] = all(v == COUNT for v in (
        len(declarations), len(set(declarations)), inventory["count"],
        capabilities["native_schema_inventory_count"], ledger["native_schema_count"]))
    digest = hashlib.sha256(("\n".join(declarations) + "\n").encode()).hexdigest()
    checks["exact_schema_denominator"] = digest == ledger["native_schema_lines_sha256"] == SCHEMAS_SHA256
    checks["full_schema_qualification_not_inferred"] = (
        all(row["status"] == "unimplemented" for row in schemas)
        and capabilities["full_upstream_schema_parity_count"] == 0
        and all(row["full_upstream_schema_qualified"] is False
                for row in capabilities["capabilities"]))
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
              derivative_source.get("named_entry_count"))
    checks["derivative_declaration_count"] = all(
        type(value) is int and value == DERIVATIVE_COUNT for value in counts)
    derivative_digest = hashlib.sha256(("\n".join(derivative_names) + "\n").encode()).hexdigest()
    checks["exact_derivative_denominator"] = (
        derivative_digest == derivative_inventory["declaration_lines_sha256"] == DERIVATIVES_SHA256)
    checks["full_derivative_qualification_not_inferred"] = all(
        row["status"] == "unimplemented" for row in derivatives)
    gates = ledger["gates"]
    ids = [row["id"] for row in gates]
    checks["all_subsystems_retained"] = len(ids) == len(GATES) and set(ids) == GATES
    checks["no_unverified_subsystem_closure"] = all(
        row["status"] in ("OPEN", "IN_PROGRESS")
        and isinstance(row["required"], str) and row["required"].strip()
        and isinstance(row["remaining"], str) and row["remaining"].strip()
        for row in gates)
    checks["completion_flags_agree"] = (
        ledger["whole_pytorch_rewrite_complete"] is False
        and capabilities["whole_pytorch_rewrite_complete"] is False
        and ledger["status"] == "HOLD")
    checks["continuation_plan_exists"] = (
        ledger["plan"] == "docs/continuation-plan.md"
        and (root / ledger["plan"]).is_file())
    return {
        "schema": "heptatorch.acceptance-accounting.v1",
        "accounting_status": "PASS" if all(checks.values()) else "FAIL",
        "whole_project_status": "HOLD",
        "whole_pytorch_rewrite_complete": False,
        "full_upstream_schema_parity_count": 0,
        "native_schema_inventory_count": len(schemas),
        "derivative_declaration_inventory_count": len(derivatives),
        "derivative_declaration_lines_sha256": derivative_digest,
        "full_derivative_declaration_parity_count": 0,
        "open_full_subsystem_gates": len(gates),
        "checks": checks,
        "limitation": "Accounting consistency only; full subsystem qualification runners remain to be implemented alongside their native functionality.",
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
    return 2 if args.require_complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
