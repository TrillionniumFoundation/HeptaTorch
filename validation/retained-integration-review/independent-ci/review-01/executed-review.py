"""Read-only retained446 Make/CI/profile review; no native or Torch execution."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tarfile

import yaml

BASE = "af33d1065a3b097990e9536df5b22b1a3fb315cf"
TREE = "88ed6bd6a41ca0e58e653b3d76718090b3bbba64"
SOURCE = "666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b"
IMAGE = "7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5"
QUALIFIER = "c71d40ea111b9c80e9d0987f7cd42729cc3f47ecf5a86b7e5d4ecf0659ec0ee2"
ARCHIVE = "0cc512e60904cf32812efdb1aba166f01ff190cea6966a75a1c50e888325aec5"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--observer-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, package, output = (p.resolve() for p in (args.repo, args.observer_package, args.output))
    output.mkdir(parents=True, exist_ok=False)
    inputs, checks, helps = {}, [], []

    def read(path):
        path = path.resolve()
        assert path.is_file() and not path.is_symlink(), path
        raw = path.read_bytes()
        row = {"path": str(path), "bytes": len(raw), "sha256": sha(raw), "mode": path.stat().st_mode & 0o777}
        assert str(path) not in inputs or inputs[str(path)] == row
        inputs[str(path)] = row
        return raw

    def need(value, name, **facts):
        checks.append({"check": name, "passed": bool(value), **facts})
        assert value, (name, facts)

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(repo), *argv])

    def blocks(raw):
        matches = list(re.finditer(rb"^      - name: (.+)\n", raw, re.M))
        return [(m.group(1).decode(), raw[m.start(): matches[i + 1].start() if i + 1 < len(matches) else len(raw)])
                for i, m in enumerate(matches)]

    need(git("rev-parse", BASE + "^{tree}").decode().strip() == TREE, "immutable base tree")
    native_path, quick_path = ".github/workflows/native.yml", ".github/workflows/observer-accounting.yml"
    old_native, old_quick, old_make = (git("show", BASE + ":" + n) for n in (native_path, quick_path, "Makefile"))
    native, quick, make = (read(repo / n) for n in (native_path, quick_path, "Makefile"))
    old_steps, new_steps = blocks(old_native), blocks(native)
    old_map, new_map = dict(old_steps), dict(new_steps)
    added = [name for name, _ in new_steps if name not in old_map]
    changed = [name for name, raw in old_steps if new_map.get(name, b"").rstrip(b"\n") != raw.rstrip(b"\n")]
    need(len(old_steps) == 71 and len(new_steps) == 75, "main declared step counts", old=71, current=75)
    need([name for name, _ in new_steps if name in old_map] == [name for name, _ in old_steps], "old step order preserved")
    need(changed == ["Index original private-node reports and raw snapshots"], "70 old step bodies byte-identical", changed=changed)
    need(len(added) == 4, "four new main steps", names=added)
    reverted = native
    for name in added:
        need(reverted.count(new_map[name]) == 1, "unique inserted step", name=name)
        reverted = reverted.replace(new_map[name], b"", 1)
    needle = (b"              build/pytorch-gradient-adam-distributions-frozen35.json build/pytorch-gradient-adam-distributions-frozen35-original.json \\\n"
              b"              build/pytorch-retained-gradients-frozen35.json build/pytorch-retained-gradients-frozen35-original.json build/pytorch-install.json \\\n")
    restored = b"              build/pytorch-gradient-adam-distributions-frozen35.json build/pytorch-gradient-adam-distributions-frozen35-original.json build/pytorch-install.json \\\n"
    need(reverted.count(needle) == 1, "collector exactly two added report arguments")
    reverted = reverted.replace(needle, restored, 1)
    need(reverted == old_native + b"\n", "main full inverse bytes including one added final separator")
    quick_extra = (b"          - profile: '446'\n"
                   b"            profile_name: retained-gradients-446\n"
                   b"            image: tensor-retained-session\n"
                   b"            native_sha: " + IMAGE.encode() + b"\n"
                   b"            source_sha: " + SOURCE.encode() + b"\n")
    need(quick.count(quick_extra) == 1 and quick.replace(quick_extra, b"", 1) == old_quick, "quick only additional exact446 profile")
    ndoc, qdoc = yaml.safe_load(native), yaml.safe_load(quick)
    ns = ndoc["jobs"]["native"]["steps"]
    qs = qdoc["jobs"]["observer"]["steps"]
    need(len(ns) == 75 and len(qs) == 10, "independent YAML step parse")
    strategy = qdoc["jobs"]["observer"]["strategy"]
    need(strategy["fail-fast"] is False and [r["profile"] for r in strategy["matrix"]["include"]] == ["355", "428", "446"], "old independent matrix retained")
    need(strategy["matrix"]["include"][-1] == {"profile":"446", "profile_name":"retained-gradients-446", "image":"tensor-retained-session", "native_sha":IMAGE, "source_sha":SOURCE}, "quick source/image exact pins")
    envelopes = lambda raw: sorted(set(x.decode() for x in re.findall(rb"tools/emit_ci_report\.py (build/[A-Za-z0-9_.-]+\.json)(?:\s|$)", raw)))
    need(len(envelopes(old_native)) == 30 and len(envelopes(native)) == 33, "literal envelope counts are30and33")
    qualifier = read(repo / "tests/qualify_retained_gradients.py")
    need(sha(qualifier) == QUALIFIER, "imported c71 qualifier exact")
    ast.parse(qualifier)
    marker = b"\n\nRETAINED_GRADIENT_INPUTS :="
    start, end = make.index(marker), make.index(b"\n\ncheck-compiler:", make.index(marker))
    addition = make[start:end]
    need(make[:start] + make[end:] == old_make, "Make full inverse byte equality")
    build_recipe = addition.split(b"build/retained-gradients/build.json:", 1)[1].split(b"\ntest-retained-gradients:", 1)[0]
    test_recipe = addition.split(b"test-retained-gradients:", 1)[1]
    need(build_recipe.startswith(b" build/gradient-adam-distributions/build.json"), "build correct parent dependency")
    need(test_recipe.startswith(b" gradient-adam-distributions\n"), "qualification correct parent dependency")
    need(b"--producer-review" not in test_recipe and b"--repo \"$(CURDIR)\"" in test_recipe, "CI invokes fresh producer by default")
    copy_order = [test_recipe.index(s) for s in (b"|| status=$$?", b'cp "$$review/observed/report.json"', b'cp "$$review/observed/evidence-manifest.json"', b'cp "$$review/observed/raw-evidence.tar.xz"', b'test "$$status" -eq 0 || exit "$$status"', b'cp -a --remove-destination "$$review/observed/producer/native/."')]
    need(copy_order == sorted(copy_order), "Make preserves failure artifacts before returning nonzero and promotes native only on success")
    need(b"trap " not in test_recipe, "qualification temporary evidence retained")
    name_map = {step["name"]: step for step in ns}
    new_run = name_map[added[0]]["run"]
    need("|| status=$?" in new_run and new_run.rstrip().endswith('exit "$status"'), "main qualification does not swallow make failure")
    for name in added[1:3]:
        run = name_map[name]["run"]
        need(IMAGE in run and "|| status=$?" in run and run.rstrip().endswith('exit "$status"'), "new actual oracle source pinned and returns failure", name=name)
    need(SOURCE in name_map[added[1]]["run"], "live32 exact source pin")
    frozen_run = name_map[added[2]]["run"]
    need("--original-report build/pytorch-retained-gradients-frozen35-original.json" in frozen_run and 'if [ "$status" -eq 0 ]' in frozen_run, "frozen35 invokes EOF/Version/11-source checker on original success")
    for name in ("Index complete original CI reports and checkout identity", "Export original CI reports as a lossless artifact", "Index original private-node reports and raw snapshots", "Export original private-node snapshots and wire traces", added[-1]):
        need(name_map[name]["if"] == "always()", "artifact collection always runs", name=name)
    raw_artifact = name_map[added[-1]]
    need(raw_artifact["with"]["path"].splitlines() == ["build/retained-gradients-ci-raw.tar.xz", "build/retained-gradients-ci-raw-manifest.json", "build/hosted-retained-gradients-native.json"], "raw artifact contains archive manifest and separately bound final aggregate")
    for step in [s for s in ns if s["name"] in added] + qs:
        if "run" in step:
            result = subprocess.run(["bash", "-n"], input=step["run"].encode(), capture_output=True)
            need(result.returncode == 0, "shell syntax only", name=step["name"], stderr=result.stderr.decode())
    manifest = json.loads(read(package / "MANIFEST.json"))
    need(manifest["file_count"] == len(manifest["files"]) == 22, "observer frozen package22")
    for row in manifest["files"]:
        raw = read(package / row["path"])
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "observer frozen member", path=row["path"])
    proof = json.loads(read(package / "inverse-proof.json"))
    changed_files = []
    for name, row in proof["files"].items():
        root_bytes = read(repo / "tests/private_node_accounting" / name)
        package_bytes = read(package / "tests/private_node_accounting" / name)
        old = git("show", BASE + ":tests/private_node_accounting/" + name)
        need(root_bytes == package_bytes and sha(root_bytes) == row["after_sha256"] and sha(old) == row["before_sha256"], "observer actual root/current/base pins", name=name)
        inverse = root_bytes.decode()
        for replacement in reversed(row["replacements"]):
            need(inverse.count(replacement["after"]) == 1, "unique observer inverse", name=name)
            inverse = inverse.replace(replacement["after"], replacement["before"], 1)
        need(inverse.encode() == old, "recomputed observer exact byte inverse", name=name)
        if root_bytes != old:
            changed_files.append(name)
        ast.parse(root_bytes)
    need(set(changed_files) == {"owner_accounting.py", "check_owner_snapshots.py", "check_frozen35_accounted.py", "restore_native_inputs.py"}, "only four profile/dependency edits")
    owner_before = git("show", BASE + ":tests/private_node_accounting/owner_accounting.py")
    owner_after = read(repo / "tests/private_node_accounting/owner_accounting.py")
    def definitions(raw):
        lines = raw.splitlines(keepends=True)
        return {n.name:b"".join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    need(definitions(owner_before) == definitions(owner_after), "all accounting functions/classes byte-identical")
    archive_path = repo / "validation/retained-gradients-root/root-cold-and-strict-01.tar.xz"
    need(sha(read(archive_path)) == ARCHIVE, "quick446 archive actual installed pin")
    with tarfile.open(archive_path, "r:xz") as archive:
        expected = {"runtime.hepta":SOURCE, "tensor-retained-session":IMAGE}
        for relative, pin in expected.items():
            member = archive.getmember("build/retained-root-cold-01/native/" + relative)
            raw = archive.extractfile(member).read()
            need(member.isfile() and len(raw) == member.size and sha(raw) == pin, "quick446 archive member actual source/image", path=relative)
    owner_values = {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(owner_after).body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("PROFILES","FUNCTION_PINS","DESC_BYTES")}
    native_source = read(repo / "build/retained-root-cold-01/native/runtime.hepta")
    need(sha(native_source) == SOURCE, "actual cold source binds reviewed observer")
    functions = {name.decode():body for body,name in re.findall(rb"^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)", native_source,re.M|re.S)}
    for name, pin in owner_values["FUNCTION_PINS"].items():
        need(sha(functions[name]) == pin, "observer approved function matches actual446", function=name)
    need(owner_values["DESC_BYTES"] == 576 and owner_values["PROFILES"][IMAGE]["state"] == "HTRET001" and owner_values["PROFILES"][IMAGE]["backing"] == "HTORET01", "actual446 private layout profile")
    for relative in ("tests/qualify_retained_gradients.py", "tools/retained_gradients/build.py", "tests/private_node_accounting/restore_native_inputs.py"):
        proc = subprocess.run([sys.executable,"-I","-S","-B",str(repo/relative),"--help"],cwd=repo,capture_output=True)
        need(proc.returncode == 0, "real argparse help only", script=relative)
        helps.append({"script":relative,"returncode":proc.returncode,"stdout":proc.stdout.decode(),"stderr":proc.stderr.decode()})
    for row in list(inputs.values()):
        path=Path(row["path"])
        need(len(path.read_bytes()) == row["bytes"] and sha(path.read_bytes()) == row["sha256"] and path.stat().st_mode & 0o777 == row["mode"], "input custody after", path=str(path))
    report = {"schema":"heptatorch.retained446-ci-source-review.v1","status":"PASS_READ_ONLY_INTEGRATION_REVIEW","base_commit":BASE,"base_tree":TREE,"current_checkout":git("rev-parse","HEAD").decode().strip(),"qualifier_sha256":QUALIFIER,"checks":checks,"check_count":len(checks),"inputs":list(inputs.values()),"help_only_processes":helps,"yaml_parser_version":yaml.__version__,"step_counts":{"main_old":71,"main_current":75,"main_old_bodies_unchanged":70,"quick_matrix_profiles":3,"quick_declared_steps_each":10,"quick_expected_hosted_steps_each":13,"hosted_actual_counts_observed_by_this_review":False},"report_counts":{"literal_envelopes_old":30,"literal_envelopes_current":33,"new_root_json_paths":5,"expected_root_reports_if_previous58complete":63,"hosted_actual_reports_observed_by_this_review":False},"read_only_findings":["No integration blocker found in the reviewed pinned sources.","Failure report/raw copies precede the Make nonzero return; native promotion is success-only. GNU Make may expose its standard failure status2 while the qualifier original status remains in the preserved report.","The final aggregate is outside its own archive and is uploaded separately alongside the manifest; all child raw bytes are inside the coordinator archive.","Observer446 changes exact profile constants and dependent pins only. T flags remain zero for this old frozen35 workload; this does not widen observer retain_grad coverage.","Quick restoration is exact archived input restoration, not a fresh native build.","Main75 and quick10 are declared step counts. Automatic hosted steps and complete report counts require actual CI observation."],"native_programs_executed":0,"pytorch_executed":False,"fresh_coordinator_or_ci_executed":False,"root_worktree_modified":False,"full_rewrite_complete":False,"full_schema_parity":0,"pytorch_api_denominator":2590,"whole_project_admission":"HOLD"}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"executed-review.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status":report["status"],"checks":len(checks),"report_sha256":sha((output/"report.json").read_bytes())}))


if __name__ == "__main__":
    main()
