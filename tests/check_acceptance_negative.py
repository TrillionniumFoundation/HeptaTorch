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
    )
    rows = []
    for name, filename, mutate in controls:
        with tempfile.TemporaryDirectory(prefix="heptatorch-acceptance-") as temporary:
            root = Path(temporary)
            for path in ("compatibility/upstream-schemas.json", "compatibility/upstream-lock.json",
                         "compatibility/capabilities.json", "compatibility/full-acceptance.json",
                         "docs/continuation-plan.md"):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / path, target)
            target = root / filename
            document = json.loads(target.read_text())
            mutate(document)
            target.write_text(json.dumps(document))
            report = acceptance.check(root)
            rejected = report["accounting_status"] == "FAIL"
            rows.append({"case": name, "rejected": rejected})
            if not rejected:
                raise SystemExit("Unsafe target alteration was accepted: " + name)
    print(json.dumps({"schema": "heptatorch.acceptance-negative-controls.v1",
                      "status": "PASS", "cases": len(rows), "rows": rows,
                      "verifier_sha256": hashlib.sha256(VERIFIER.read_bytes()).hexdigest(),
                      "whole_project_status": "HOLD"}, sort_keys=True))


if __name__ == "__main__":
    main()
