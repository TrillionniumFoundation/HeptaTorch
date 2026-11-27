#!/usr/bin/env python3
"""Adversarial controls for accidentally deleting scope or declaring closure."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools/check_acceptance.py"
spec = importlib.util.spec_from_file_location("acceptance", VERIFIER)
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


def derivative_source(document):
    return next(row for row in document["source_files"]
                if row["path"] == "tools/autograd/derivatives.yaml")


def check_with_mutations(mutations):
    with tempfile.TemporaryDirectory(prefix="heptatorch-acceptance-") as temporary:
        root = Path(temporary)
        for path in ("compatibility/upstream-schemas.json", "compatibility/upstream-derivatives.json",
                     "compatibility/upstream-lock.json", "compatibility/capabilities.json",
                     "compatibility/full-acceptance.json", "docs/continuation-plan.md"):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, target)
        for filename, mutate in mutations:
            target = root / filename
            document = json.loads(target.read_text())
            mutate(document)
            target.write_text(json.dumps(document))
        return acceptance.check(root)


def reject_case(name, mutations, expected_check=None):
    report = check_with_mutations(mutations)
    failed_checks = [key for key, passed in report["checks"].items() if not passed]
    if report["accounting_status"] != "FAIL":
        raise SystemExit("Unsafe target alteration was accepted: " + name)
    if expected_check is not None and failed_checks != [expected_check]:
        raise SystemExit(f"{name} did not fail the expected check alone: {failed_checks}")
    return {"case": name, "rejected": True, "failed_checks": failed_checks}


def main():
    baseline = acceptance.check(ROOT)
    if baseline["accounting_status"] != "PASS" or baseline["whole_project_status"] != "HOLD":
        raise SystemExit("Current target must be internally consistent and remain HOLD")
    controls = (
        ("deleted-schema", "compatibility/upstream-schemas.json", lambda d: d["schemas"].pop()),
        ("changed-schema-preserving-count", "compatibility/upstream-schemas.json", lambda d: d["schemas"][0].update(schema="invented(Tensor x) -> Tensor")),
        ("declared-schema-qualification", "compatibility/upstream-schemas.json", lambda d: d["schemas"][0].update(status="qualified")),
        ("deleted-subsystem", "compatibility/full-acceptance.json", lambda d: d["gates"].pop()),
        ("duplicated-subsystem", "compatibility/full-acceptance.json", lambda d: d["gates"].append(d["gates"][0].copy())),
        ("unverified-subsystem-closure", "compatibility/full-acceptance.json", lambda d: d["gates"][0].update(status="CLOSED")),
        ("fabricated-whole-completion", "compatibility/full-acceptance.json", lambda d: d.update(whole_pytorch_rewrite_complete=True, status="PASS")),
        ("scoped-kernel-promoted-to-full-schema", "compatibility/capabilities.json", lambda d: d["capabilities"][0].update(full_upstream_schema_qualified=True)),
        ("changed-upstream-pin", "compatibility/upstream-lock.json", lambda d: d.update(commit="0" * 40)),
        ("erased-derivative-capability-count", "compatibility/capabilities.json", lambda d: d.update(derivative_declaration_inventory_count=0)),
        ("erased-derivative-ledger-count", "compatibility/full-acceptance.json", lambda d: d.update(derivative_declaration_count=0)),
        ("changed-derivative-ledger-digest", "compatibility/full-acceptance.json", lambda d: d.update(derivative_declaration_lines_sha256="0" * 64)),
        ("fabricated-native-qualified-count", "compatibility/capabilities.json", lambda d: d.update(full_upstream_schema_parity_count=1)),
        ("fabricated-derivative-qualified-count", "compatibility/capabilities.json", lambda d: d.update(full_derivative_declaration_parity_count=1)),
        ("erased-derivative-upstream-count", "compatibility/upstream-lock.json", lambda d: derivative_source(d).update(named_entry_count=0)),
        ("deleted-derivative-declaration", "compatibility/upstream-derivatives.json", lambda d: d["derivatives"].pop()),
        ("changed-derivative-preserving-count", "compatibility/upstream-derivatives.json", lambda d: d["derivatives"][0].update(name="invented(Tensor x) -> Tensor")),
        ("reordered-derivative-declarations", "compatibility/upstream-derivatives.json", lambda d: d["derivatives"].reverse()),
        ("duplicated-derivative-declaration", "compatibility/upstream-derivatives.json", lambda d: d["derivatives"][0].update(name=d["derivatives"][1]["name"])),
        ("declared-derivative-qualification", "compatibility/upstream-derivatives.json", lambda d: d["derivatives"][0].update(status="qualified")),
        ("changed-derivative-upstream-pin", "compatibility/upstream-derivatives.json", lambda d: d["upstream"].update(commit="0" * 40)),
        ("changed-derivative-source-path", "compatibility/upstream-derivatives.json", lambda d: d["upstream"].update(source_path="tools/autograd/other.yaml")),
        ("changed-derivative-source-blob", "compatibility/upstream-derivatives.json", lambda d: d["upstream"].update(source_git_blob_sha1="0" * 40)),
        ("changed-derivative-source-sha256", "compatibility/upstream-derivatives.json", lambda d: d["upstream"].update(source_sha256="0" * 64)),
        ("changed-derivative-list-sha256", "compatibility/upstream-derivatives.json", lambda d: d.update(declaration_lines_sha256="0" * 64)),
        ("changed-derivative-lock-blob", "compatibility/upstream-lock.json", lambda d: derivative_source(d).update(git_blob_sha1="0" * 40)),
        ("deleted-derivative-lock-record", "compatibility/upstream-lock.json", lambda d: d["source_files"].remove(derivative_source(d))),
        ("duplicated-derivative-lock-record", "compatibility/upstream-lock.json", lambda d: d["source_files"].append(derivative_source(d).copy())),
    )
    rows = [reject_case(name, ((filename, mutate),)) for name, filename, mutate in controls]
    ledger_path = "compatibility/full-acceptance.json"

    def swap_requirements(document):
        first, second = document["gates"][:2]
        first["required"], second["required"] = second["required"], first["required"]

    rows.extend((
        reject_case("changed-subsystem-requirement", ((ledger_path, lambda d: d["gates"][0].update(
            required="Independent native compilation")),), "exact_subsystem_requirements"),
        reject_case("swapped-subsystem-requirements", ((ledger_path, swap_requirements),),
                    "exact_subsystem_requirements"),
    ))

    def update_progress(document):
        document["gates"].reverse()
        document["gates"][0]["remaining"] += " Review this progress note."

    progressed = check_with_mutations(((ledger_path, update_progress),))
    if (progressed["accounting_status"] != "PASS"
            or progressed["whole_project_status"] != "HOLD"
            or progressed["subsystem_requirements_sha256"] != baseline["subsystem_requirements_sha256"]):
        raise SystemExit("Progress text or row ordering must not redefine the subsystem contracts")
    # Reproduce the observed defect: both existing count records said zero,
    # yet the previous native-schema-only validator still returned PASS.
    rows.append(reject_case("erased-both-original-derivative-counts", (
        ("compatibility/capabilities.json", lambda d: d.update(derivative_declaration_inventory_count=0)),
        ("compatibility/upstream-lock.json", lambda d: derivative_source(d).update(named_entry_count=0)),
    )))
    # Matching a shortened inventory's own count/digest must not redefine scope.
    def shrink_inventory(document):
        document["derivatives"].pop()
        document["count"] = len(document["derivatives"])
        names = [row["name"] for row in document["derivatives"]]
        document["declaration_lines_sha256"] = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
    rows.append(reject_case("coordinated-derivative-denominator-shrink", (
        ("compatibility/upstream-derivatives.json", shrink_inventory),
        ("compatibility/capabilities.json", lambda d: d.update(derivative_declaration_inventory_count=687)),
        ("compatibility/upstream-lock.json", lambda d: derivative_source(d).update(named_entry_count=687)),
    )))
    print(json.dumps({"schema": "heptatorch.acceptance-negative-controls.v1",
                      "status": "PASS", "cases": len(rows), "rows": rows,
                      "verifier_sha256": hashlib.sha256(VERIFIER.read_bytes()).hexdigest(),
                      "verifier_dependencies": {
                          "tools/acceptance_contracts.py": hashlib.sha256(
                              Path(acceptance.contracts.__file__).read_bytes()).hexdigest()},
                      "whole_project_status": "HOLD"}, sort_keys=True))


if __name__ == "__main__":
    main()
