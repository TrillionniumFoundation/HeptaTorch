"""Make the exact 446 profile extension; preserve all accounting algorithms.

This authoring utility only copies pinned Python test code and changes explicit
native-profile constants / dependent source pins. It does not execute a native
program or manufacture oracle expectations.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json

PINS = {
    "owner_accounting.py": "e900466bff13bcaa5cd89a814dc411b4fc33320d84855ca6bd085179cd599c6a",
    "check_owner_snapshots.py": "c4873900abd44f2c74c2632d662d932ee0912501b7a127771c2bfecccc0f2db6",
    "check_frozen35_accounted.py": "5f21d4908d21f710e6139dd33a17151c43e4f00a56333c85cec90018077cdd6c",
    "restore_native_inputs.py": "a3bdac36dd4f45f03f1016c287737f51bab7204906a033b10ad45ee237cee811",
    "run_frozen35_accounted.py": "65c0d68725f0585b79f88ceb5fa106526a44805eca0b58535f32b5b29916cd12",
    "test_native_accounting.py": "55d57b8fea82556f6b5e03a5f2112ddaf689b8dfec8e93f53609b0637a337509",
    "collect_private_node_journals.py": "275d81e1bf316c8b091fb2591fb0e3246837143a0ac61909250ab204c2a7b941",
}
SOURCE = "666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b"
IMAGE = "7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5"
PROFILE = "retained-gradients-446"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    base = a.base.resolve()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    originals = {}
    for name, pin in PINS.items():
        raw = (base / name).read_bytes()
        assert sha(raw) == pin, (name, "af33 base source pin")
        originals[name] = raw
    paths = output / "tests" / "private_node_accounting"
    paths.mkdir(parents=True)
    changes = {}
    additions = {
        "owner_accounting.py": [(
            "}\nSOURCE_PINS = {",
            '    "' + IMAGE + '": {\n'
            '        "name": "' + PROFILE + '",\n'
            '        "source_sha256": "' + SOURCE + '",\n'
            '        "state": "HTRET001", "backing": "HTORET01",\n'
            '    },\n}\nSOURCE_PINS = {',
        )],
        "restore_native_inputs.py": [(
            "}\ndef sha(raw):",
            ' "446": ("validation/retained-gradients-root/root-cold-and-strict-01.tar.xz",\n'
            '         "0cc512e60904cf32812efdb1aba166f01ff190cea6966a75a1c50e888325aec5", "build/retained-root-cold-01/native/",\n'
            '         "' + SOURCE + '",\n'
            '         "tensor-retained-session", "' + IMAGE + '"),\n'
            '}\ndef sha(raw):',
        )],
        "check_frozen35_accounted.py": [(
            "}\nCHECKS=",
            ' "' + IMAGE + '":\n'
            '  ("' + PROFILE + '","' + SOURCE + '"),\n}\nCHECKS=',
        )],
    }
    new_sources = {}
    order = ["owner_accounting.py"] + [name for name in PINS if name != "owner_accounting.py"]
    for name in order:
        replacements = list(additions.get(name, []))
        if name == "check_owner_snapshots.py":
            replacements += [(
                "choices=['gradient-objects-355','gradient-adam-distributions-428']",
                "choices=['gradient-objects-355','gradient-adam-distributions-428','retained-gradients-446']",
            )]
        if name in ("check_owner_snapshots.py", "check_frozen35_accounted.py"):
            replacements += [(PINS["owner_accounting.py"], sha(new_sources["owner_accounting.py"]))]
        source = originals[name].decode("utf-8")
        for old, new in replacements:
            assert source.count(old) == 1, (name, "unique exact replacement", old)
            source = source.replace(old, new, 1)
        raw = source.encode("utf-8")
        compile(raw, str(paths / name), "exec", dont_inherit=True)
        inverse = source
        for old, new in reversed(replacements):
            assert inverse.count(new) == 1, (name, "unique exact inverse", new)
            inverse = inverse.replace(new, old, 1)
        assert inverse.encode("utf-8") == originals[name], (name, "byte-exact af33 inverse")
        (paths / name).write_bytes(raw)
        new_sources[name] = raw
        changes[name] = {
            "before_sha256": sha(originals[name]), "after_sha256": sha(raw),
            "before_bytes": len(originals[name]), "after_bytes": len(raw),
            "changed": raw != originals[name], "replacements": [
                {"before": old, "after": new} for old, new in replacements],
            "exact_inverse": True,
        }
    assert {name for name, row in changes.items() if row["changed"]} == {
        "owner_accounting.py", "check_owner_snapshots.py", "check_frozen35_accounted.py", "restore_native_inputs.py"}
    before_ast = ast.parse(originals["owner_accounting.py"])
    after_ast = ast.parse(new_sources["owner_accounting.py"])
    algorithm_rows = []
    originals_by_name = {n.name: n for n in before_ast.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    lines_old = originals["owner_accounting.py"].splitlines(keepends=True)
    lines_new = new_sources["owner_accounting.py"].splitlines(keepends=True)
    for node in after_ast.body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        old = originals_by_name.pop(node.name)
        old_raw = b"".join(lines_old[old.lineno - 1:old.end_lineno])
        new_raw = b"".join(lines_new[node.lineno - 1:node.end_lineno])
        assert old_raw == new_raw
        assert ast.dump(old, include_attributes=False) == ast.dump(node, include_attributes=False)
        algorithm_rows.append({"definition": node.name, "sha256": sha(old_raw), "bytes": len(old_raw)})
    assert not originals_by_name
    original_checker = ast.parse(originals["check_owner_snapshots.py"])
    updated_checker = ast.parse(new_sources["check_owner_snapshots.py"])
    # Only the exact dependent hash and argparse profile constant changed. The
    # full inverse above binds all 18 mutation bodies and every assertion.
    for node in ast.walk(updated_checker):
        if isinstance(node, ast.Constant):
            if node.value == sha(new_sources["owner_accounting.py"]):
                node.value = PINS["owner_accounting.py"]
        if isinstance(node, ast.List) and [getattr(n, "value", None) for n in node.elts] == [
            "gradient-objects-355", "gradient-adam-distributions-428", PROFILE]:
            node.elts.pop()
    assert ast.dump(updated_checker, include_attributes=False) == ast.dump(original_checker, include_attributes=False)
    before_profile = {}
    after_profile = {}
    for module_ast, dest in ((before_ast, before_profile), (after_ast, after_profile)):
        for node in module_ast.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("PROFILES", "SOURCE_PINS", "FUNCTION_PINS", "DESC_BYTES"):
                    dest[node.targets[0].id] = ast.literal_eval(node.value)
    assert after_profile["PROFILES"].pop(IMAGE)["name"] == PROFILE
    assert before_profile == after_profile
    for name in originals:
        assert (base / name).read_bytes() == originals[name], (name, "base changed during read-only build")
    receipt = {
        "schema": "heptatorch.retained-observer-profile-extension.v1", "status": "EXACT_REVERSIBLE_PROFILE_ONLY",
        "base_head": "af33d1065a3b097990e9536df5b22b1a3fb315cf", "base_tree": "88ed6bd6a41ca0e58e653b3d76718090b3bbba64",
        "profile": PROFILE, "source_sha256": SOURCE, "native_sha256": IMAGE,
        "files": changes, "byte_identical_accounting_definitions": algorithm_rows,
        "all_18_negative_mutations_and_assertions_unchanged": True,
        "old_source_pins_function_pins_profiles_and_descriptor_width_unchanged": True,
        "T_record_flags_remain_exact_zero": True,
        "original_launcher_and_native3_and_collector_byte_identical": True,
        "native_executed": False, "pytorch_executed": False,
        "scope": "Only admit the exact retained446 profile to unchanged read-only accounting. Does not qualify retain_grad calls or widen the observer input scope.",
    }
    (output / "inverse-proof.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "observer_sha256": sha(new_sources["owner_accounting.py"]), "files_changed": 4}))


if __name__ == "__main__":
    main()
