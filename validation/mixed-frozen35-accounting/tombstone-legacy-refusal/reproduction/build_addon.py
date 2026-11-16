"""Create an additive tombstone regression entry with an exact inverse.

No frozen source is edited. The private output mirrors the source package
solely to exercise the new file under the same repository-relative imports.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil

ORIGINAL = "2dc6d1d6444264fec06c3429926adcf778e48b6b8b80226b690bc31667dcd6cc"
LEGACY = "6ea9e8b33164089034a2a71a03f4850e1ec18ed1a09155d7841f700ef311c833"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): {"bytes": p.stat().st_size,
            "sha256": sha(p.read_bytes()), "mode": p.stat().st_mode & 0o777}
            for p in sorted(root.rglob("*")) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--legacy-refusal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_package = args.source_package.resolve()
    before = inventory(source_package)
    rel = Path("tests/mixed_frozen35_accounting/test_storage_tombstone.py")
    original = (source_package / rel).read_bytes()
    legacy = args.legacy_refusal.read_bytes()
    assert sha(original) == ORIGINAL and sha(legacy) == LEGACY
    text = original.decode()
    edits = []

    def edit(label, old, new):
        nonlocal text
        assert text.count(old) == 1, (label, text.count(old))
        text = text.replace(old, new)
        edits.append({"label": label, "before": old, "after": new, "count": 1})

    edit("explicit-frozen-defective-predicate",
         "import owner_accounting_mixed as observer\n",
         "import owner_accounting_mixed as observer\n\n"
         "# Deliberately separate the historical defective predicate from the\n"
         "# current structural checker (aa8 already contains the correction).\n"
         "LEGACY_REFUSAL = HERE.parent / \"dtype_owner_accounting\" / \"owner640.py\"\n"
         f"LEGACY_REFUSAL_SHA = \"{LEGACY}\"\n")
    edit("before-after-legacy-file-custody",
         '"protocol": a.protocol, "owner_client": a.protocol.with_name("owner_client.py")}',
         '"protocol": a.protocol, "owner_client": a.protocol.with_name("owner_client.py"),\n'
         '             "legacy_refusal_owner640": LEGACY_REFUSAL}')
    edit("load-only-SHA-verified-original6ea",
         '        original = load("tombstone_original640", observer.ORIGINAL_STRUCTURAL)\n',
         '        assert inputs["legacy_refusal_owner640"]["sha256"] == LEGACY_REFUSAL_SHA\n'
         '        original = load("tombstone_original640", LEGACY_REFUSAL)\n'
         '        assert observer.sha(LEGACY_REFUSAL.read_bytes()) == LEGACY_REFUSAL_SHA\n')
    edit("explain-four-identity-bindings",
         "        # file and every function body remain byte-identical; only these three\n",
         "        # file and every function body remain byte-identical; only these four\n")
    edit("bind-schema-identity-without-changing-predicate",
         "        original.SOURCE = observer.structural.SOURCE\n",
         '        historical_identity = {name: getattr(original, name)\n'
         '                               for name in ("SCHEMA", "SOURCE", "IMAGE", "PROFILE")}\n'
         "        original.SCHEMA = observer.structural.SCHEMA\n"
         "        original.SOURCE = observer.structural.SOURCE\n")
    edit("record-original-and-current-data-identities",
         '        report["original_predicate_profile_binding"] = {\n'
         '            "source_sha256": original.SOURCE, "native_sha256": original.IMAGE,\n',
         '        report["original_predicate_profile_binding"] = {\n'
         '            "predicate_file_sha256": LEGACY_REFUSAL_SHA,\n'
         '            "original_data_identity": historical_identity,\n'
         '            "changed_module_data_names": ["SCHEMA", "SOURCE", "IMAGE", "PROFILE"],\n'
         '            "schema": original.SCHEMA,\n'
         '            "source_sha256": original.SOURCE, "native_sha256": original.IMAGE,\n')
    edit("preserve-primary-error-and-use-real-EOF-cleanup",
         "        if session is not None and session.poll() is None:\n"
         "            session.stop()\n",
         "        if session is not None and session.poll() is None:\n"
         "            try:\n"
         "                session.finish()\n"
         "            except BaseException as cleanup_error:\n"
         '                report["cleanup_error"] = repr(cleanup_error)\n'
         '                report["cleanup_traceback"] = traceback.format_exc()\n'
         '                if report["status"] != "FAIL":\n'
         '                    report.update(status="FAIL", error=repr(cleanup_error),\n'
         '                                  traceback=traceback.format_exc())\n')
    inverse = text
    for item in reversed(edits):
        assert inverse.count(item["after"]) == 1
        inverse = inverse.replace(item["after"], item["before"])
    assert inverse.encode() == original
    original_assertions = [ast.dump(n, include_attributes=False)
                           for n in ast.walk(ast.parse(original)) if isinstance(n, ast.Assert)]
    new_assertions = [ast.dump(n, include_attributes=False)
                      for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Assert)]
    for assertion in original_assertions:
        assert new_assertions.count(assertion) >= original_assertions.count(assertion)
    output = args.output.resolve()
    shutil.copytree(source_package, output)
    new_path = output / rel.with_name("test_storage_tombstone_legacy_refusal.py")
    new_path.write_text(text)
    legacy_path = output / "tests/dtype_owner_accounting/owner640.py"
    legacy_path.parent.mkdir(parents=True, exist_ok=False)
    legacy_path.write_bytes(legacy)
    after = inventory(source_package)
    assert before == after
    proof = {"schema": "heptatorch.mixed-tombstone-legacy-refusal-adaptation.v1",
             "original_test_sha256": sha(original), "new_test_sha256": sha(text.encode()),
             "legacy_refusal_sha256": sha(legacy), "edits": edits,
             "inverse_entire_file_byte_identical": True,
             "all_original_assertion_AST_multiset_preserved": True,
             "original_assertion_count": len(original_assertions),
             "new_assertion_count": len(new_assertions),
             "source_package_before": before, "source_package_after": after,
             "source_package_unchanged": True,
             "native_production_changed": False, "numeric_expectations_changed": False}
    (output.parent / "adaptation.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"new_test": str(new_path), "sha256": sha(text.encode()),
                      "original_files_unchanged": len(before), "edits": len(edits)}))


if __name__ == "__main__":
    main()
