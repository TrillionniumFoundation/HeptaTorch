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
    upstream = read("compatibility/upstream-lock.json")
    capabilities = read("compatibility/capabilities.json")
    ledger = read("compatibility/full-acceptance.json")
    checks = {}
    schemas = inventory["schemas"]
    declarations = [row["schema"] for row in schemas]
    checks["upstream_identity"] = all(v == PIN for v in (
        inventory["upstream"]["commit"], upstream["commit"],
        capabilities["upstream_commit"], ledger["upstream_commit"]))
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
