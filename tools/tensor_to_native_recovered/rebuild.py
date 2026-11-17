#!/usr/bin/env python3
"""Rebuild frozen Tensor-to-copy 09b HIA without the lost authoring environment."""
import argparse
import hashlib
import json
import os
import platform
import re
import stat
import struct
import subprocess
import sys
import traceback
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RECIPE = "tensor-to-native-recovered.09b.v1"
RUNTIME = "4f8f73f463f681b5eb29423d71a35a0b42d807fd92bf976bbf76c1970fbcee33"
IMAGE = "53fb43b869858f020c0b936a65f00f3b14b97d9e37038a74c357e1da471fe083"
ORIGINAL_BUILD = "9913ccb35c690f1a8795479ddd468531a966df71c375fcb91ed22d9f3c887557"
COMPILER = "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"
LINKER = "aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16"
RNG = "a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3"
OWNERS = "9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884"
INPUT_PINS = {
    "inputs/runtime.hepta": RUNTIME, "inputs/rng03.hepta": RNG,
    "inputs/owners.hepta": OWNERS, "inputs/original-build.json": ORIGINAL_BUILD,
}
FUNCTION = re.compile(rb"^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)", re.M | re.S)
CALL = re.compile(rb'\{"function":"(\w+)"\}')
EXTERNAL = {"rng_seed", "rng_restore", "rng_next", "rng_next64", "rng_unit32", "rng_unit64"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not a regular file: " + str(path))
    return stat.S_IMODE(mode)


def relative(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and ".." not in p.parts and str(p) == name,
            "noncanonical package path: " + name)
    return p


def file_record(path, name=None):
    mode = regular(path)
    raw = path.read_bytes()
    return {"path": str(path) if name is None else name, "bytes": len(raw),
            "sha256": sha(raw), "mode": mode}


def expected_profile(build):
    return {
        "schema": "heptatorch.tensor-to-native-recovered.profile.v1",
        "recipe_id": RECIPE, "runtime_sha256": RUNTIME, "native_sha256": IMAGE,
        "native_bytes": 3129344, "original_build_sha256": ORIGINAL_BUILD,
        "compiler_sha256": COMPILER, "linker_sha256": LINKER,
        "descriptor_bytes": 672, "state_format": "HTTOCP08", "backing_format": "HTOTOCP8",
        "functions": 614, "blocks": 8078, "instructions": 29033, "unit_count": 12,
        "grouping": {"instructions": 3600, "blocks": 1000, "functions": 65,
                     "definitions_plus_imports": 256, "calls": "sorted depth-first postorder"},
        "link": {"magic": "HIASTKS1", "entry": "step", "flags": 0},
        "units": build["units"],
        "qualification": {"whole_project_status": "HOLD", "pytorch_api_denominator": 2590,
                          "closed_api_schemas": 0, "full_pytorch_rewrite_complete": False,
                          "this_recipe_runs_semantic_suites": False},
        "original_5179_authoring_package_restored": False,
        "original_a6f91_coordinator_restored": False,
    }


def package():
    manifest_path = ROOT / "package-manifest.json"
    regular(manifest_path)
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    require(manifest.get("schema") == "heptatorch.tensor-to-native-recovered.package.v1",
            "package manifest schema")
    require(manifest.get("recipe_id") == RECIPE and isinstance(manifest.get("files"), dict)
            and bool(manifest["files"]), "empty or wrong package manifest")
    files = manifest["files"]
    require("package-manifest.json" not in files, "self-referential package manifest")
    found = set()
    for p in ROOT.rglob("*"):
        require(not p.is_symlink(), "package symlink: " + str(p))
        if p.is_file():
            found.add(p.relative_to(ROOT).as_posix())
        else:
            require(p.is_dir(), "package special file: " + str(p))
    require(found == set(files) | {"package-manifest.json"}, "package member set mismatch")
    data, before = {}, {}
    for name, expected in sorted(files.items()):
        relative(name)
        p = ROOT / name
        mode = regular(p)
        raw = p.read_bytes()
        got = {"bytes": len(raw), "sha256": sha(raw), "mode": mode}
        require(got == expected, "package member mismatch: " + name)
        data[name], before[name] = raw, got
    before["package-manifest.json"] = {
        "bytes": len(manifest_raw), "sha256": sha(manifest_raw), "mode": regular(manifest_path)}
    for name, pin in INPUT_PINS.items():
        require(name in data and sha(data[name]) == pin, "frozen input pin: " + name)
    build = json.loads(data["inputs/original-build.json"])
    expected_fields = {"status": "PASS", "source_sha256": RUNTIME, "native_sha256": IMAGE,
                       "native_bytes": 3129344, "functions": 614, "blocks": 8078,
                       "instructions": 29033, "compiler_sha256": COMPILER,
                       "linker_sha256": LINKER, "descriptor_bytes": 672,
                       "state_format": "HTTOCP08", "backing_format": "HTOTOCP8"}
    require(all(build.get(k) == v for k, v in expected_fields.items()), "original build identity")
    require(isinstance(build.get("units"), list) and len(build["units"]) == 12, "original units")
    profile = json.loads(data["profile.json"])
    require(profile == expected_profile(build), "profile disagrees with frozen original build")
    return data, before, build, profile


def functions(raw):
    pairs = FUNCTION.findall(raw)
    result = {name.decode(): body for body, name in pairs}
    require(len(pairs) == len(result), "duplicate function definition")
    return result


def reconstruct(data, build):
    raw = data["inputs/runtime.hepta"]
    core = functions(raw)
    rng = functions(data["inputs/rng03.hepta"])
    owners = functions(data["inputs/owners.hepta"])
    require(len(core) == 614 and len(rng) == 17 and len(owners) == 4, "function counts")
    require(raw == b"hia-asm.v1 hepta.hia.program.v8 tensor_to_copy_runtime\n" + b"".join(core.values()),
            "runtime contains unmatched bytes or omitted definitions")
    require(not (set(core) & EXTERNAL) and EXTERNAL <= set(rng), "external RNG interface set")
    require(sum(b.count(b" = ") for b in core.values()) == 29033, "instruction count")
    require(sum(len(re.findall(rb"^block ", b, re.M)) for b in core.values()) == 8078, "block count")
    ordered, done, active = [], set(), set()

    def visit(name):
        if name in EXTERNAL:
            return
        require(name in core and name not in active, "unknown callee or recursive cycle: " + name)
        if name in done:
            return
        active.add(name)
        for target in sorted({n.decode() for n in CALL.findall(core[name])}):
            visit(target)
        active.remove(name)
        done.add(name)
        ordered.append(name)

    for name in core:
        visit(name)
    require(len(ordered) == len(set(ordered)) == 614 and set(ordered) == set(core),
            "topological coverage")
    groups, group, instructions, blocks = [], [], 0, 0
    for name in ordered:
        body = core[name]
        ni, nb = body.count(b" = "), len(re.findall(rb"^block ", body, re.M))
        if group and (instructions + ni > 3600 or blocks + nb > 1000 or len(group) >= 65):
            groups.append(group)
            group, instructions, blocks = [], 0, 0
        group.append(name)
        instructions += ni
        blocks += nb
    if group:
        groups.append(group)
    require(len(groups) == 10, "core unit count")
    sources = [("rng03", data["inputs/rng03.hepta"])]
    relations = []
    for i, names in enumerate(groups):
        calls = {n.decode() for name in names for n in CALL.findall(core[name])}
        imports = sorted(calls - set(names))
        require(len(names) + len(imports) <= 256, "unit interface capacity")
        declarations = []
        for name in imports:
            body = rng[name] if name in EXTERNAL else core[name]
            declarations.append(re.sub(rb" entry\(\w+\)$", b" external()", body.split(b"\n")[0]) + b"\n")
        name = "to_copy_%02d" % i
        source = (("hia-asm.v1 hepta.hia.program.v8 %s\nexports(%s)\n" % (name, ",".join(names))).encode()
                  + b"".join(declarations) + b"".join(core[n] for n in names))
        sources.append((name, source))
        relations.append({"name": name, "exports": names, "imports": imports,
                          "body_sha256": {n: sha(core[n]) for n in names}})
    sources.append(("owners", data["inputs/owners.hepta"]))
    require([n for n, _ in sources] == [u["name"] for u in build["units"]], "unit order")
    unit_rows = []
    for (name, source), expected in zip(sources, build["units"]):
        defs = functions(source)
        row = {"name": name, "source_sha256": sha(source), "source_bytes": len(source),
               "functions": len(defs),
               "blocks": sum(len(re.findall(rb"^block ", b, re.M)) for b in defs.values()),
               "instructions": sum(b.count(b" = ") for b in defs.values()),
               "expected_object_sha256": expected["object_sha256"]}
        require(row["source_sha256"] == expected["source_sha256"], "unit source hash: " + name)
        require(all(row[k] == expected[k] for k in ("functions", "blocks", "instructions")),
                "unit source counts: " + name)
        if name.startswith("to_copy_"):
            require(row["functions"] <= 65 and row["blocks"] <= 1000 and row["instructions"] <= 3600,
                    "unit grouping bounds: " + name)
        unit_rows.append(row)
    return sources, {"schema": "heptatorch.tensor-to-native-recovered.source-check.v1",
                     "status": "SOURCE_ONLY_PASS", "recipe_id": RECIPE,
                     "runtime_sha256": RUNTIME, "original_build_sha256": ORIGINAL_BUILD,
                     "function_coverage": 614, "rng_functions": 17, "owner_functions": 4,
                     "unit_count": 12, "units": unit_rows, "core_relations": relations,
                     "all_unit_source_hashes_match_original": True,
                     "native_tool_processes_started": 0, "target_runtime_executed": False,
                     "Torch_executed": False, "native_admission_claimed": False}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_new(path, raw, mode=0o644):
    with path.open("xb") as f:
        f.write(raw)
    path.chmod(mode)


def save_report(out, report, check_only):
    name = "source-check.json" if check_only else "build.json"
    pending = out / (name + ".pending")
    pending.write_bytes(json_bytes(report))
    pending.chmod(0o644)
    os.replace(pending, out / name)


def unchanged(before):
    for name, expected in before.items():
        got = file_record(ROOT / name, name)
        require({k: got[k] for k in ("bytes", "sha256", "mode")} == expected,
                "package changed during invocation: " + name)


def tool(path, pin):
    p = path.resolve()
    got = file_record(p)
    require(got["sha256"] == pin and (got["mode"] & 0o111), "tool pin or executable mode: " + str(p))
    require(p.read_bytes()[:4] == b"\x7fELF", "tool is not an ELF: " + str(p))
    return p, got


def invoke(executable, request, stdout_path, stderr_path, row, save, timeout):
    row.update(argv=[str(executable)], started=False, completed=False, returncode=None)
    save()
    process, primary = None, None
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                       stdout=stdout, stderr=stderr, env={"PATH": ""})
            row.update(started=True, pid=process.pid)
            save()
            try:
                process.communicate(request, timeout=timeout)
            except subprocess.TimeoutExpired:
                row["timed_out"] = True
                process.kill()
                process.communicate()
            row.update(completed=True, returncode=process.returncode)
        stdout_path.chmod(0o644)
        stderr_path.chmod(0o644)
        row.update(stdout=file_record(stdout_path, stdout_path.name),
                   stderr=file_record(stderr_path, stderr_path.name))
    except BaseException as error:
        primary = error
        row["error"] = repr(error)
        if process is not None and process.poll() is None:
            try:
                process.kill()
                process.communicate()
                row.update(completed=True, returncode=process.returncode, killed_after_host_error=True)
            except BaseException as cleanup:
                row["process_cleanup_error"] = repr(cleanup)
        raise
    finally:
        try:
            save()
        except BaseException as secondary:
            if primary is None:
                raise
            print("Secondary receipt error: " + repr(secondary), file=sys.stderr)
    require(row["returncode"] == 0 and not row.get("timed_out"), "tool failed: " + str(executable))
    require(row["stderr"]["bytes"] == 0, "tool emitted stderr: " + str(executable))


def run(args, check_only):
    out = args.output.resolve()
    require(out != ROOT and ROOT not in out.parents, "output must be outside this frozen package")
    require(not out.exists() and not out.is_symlink(), "output already exists")
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "heptatorch.tensor-to-native-recovered.invocation.v1" if check_only
              else "heptatorch.tensor-to-native-recovered.build.v1",
              "recipe_id": RECIPE, "status": "RUNNING", "mode": "source-only" if check_only else "native-build",
              "runtime_sha256": RUNTIME, "source_sha256": RUNTIME,
              "expected_native_sha256": IMAGE, "original_build_sha256": ORIGINAL_BUILD,
              "native_tool_processes_started": 0, "native_tool_processes_completed": 0,
              "target_runtime_executed": False, "Torch_executed": False,
              "new_numerical_qualification": False, "whole_project_status": "HOLD",
              "pytorch_api_denominator": 2590, "closed_api_schemas": 0,
              "full_pytorch_rewrite_complete": False, "units": []}
    before, tool_before = None, []
    def save():
        rows = report["units"] + ([report["link"]] if "link" in report else [])
        report["native_tool_processes_started"] = sum(bool(r.get("started")) for r in rows)
        report["native_tool_processes_completed"] = sum(bool(r.get("completed")) for r in rows)
        save_report(out, report, check_only)
    save()
    try:
        data, before, original, profile = package()
        report["package_inputs"] = before
        report["profile"] = profile
        sources, source_check = reconstruct(data, original)
        for name, raw in sources:
            write_new(out / (name + ".hepta"), raw)
        write_new(out / "runtime.hepta", data["inputs/runtime.hepta"])
        write_new(out / "reference-original-build.json", data["inputs/original-build.json"])
        report["source_check"] = source_check
        if check_only:
            unchanged(before)
            report.update(status="SOURCE_ONLY_PASS", all_package_inputs_unchanged=True,
                          all_unit_source_hashes_match_original=True)
            save()
        else:
            require(sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64"),
                    "native build requires Linux x86-64; source-only check is portable")
            compiler, cr = tool(args.compiler, COMPILER)
            linker, lr = tool(args.linker, LINKER)
            tool_before = [(compiler, cr), (linker, lr)]
            report["tools"] = {"compiler": cr, "linker": lr}
            save()
            for (name, source), expected in zip(sources, original["units"]):
                row = {"name": name, "source_sha256": sha(source),
                       "expected_object_sha256": expected["object_sha256"],
                       "functions": expected["functions"], "blocks": expected["blocks"],
                       "instructions": expected["instructions"]}
                report["units"].append(row)
                invoke(compiler, source, out / (name + ".o"), out / (name + ".stderr"), row, save, args.timeout)
                object_raw = (out / (name + ".o")).read_bytes()
                require(object_raw.startswith(b"\x7fELF") and sha(object_raw) == expected["object_sha256"],
                        "compiled object mismatch: " + name)
                row.update(object_sha256=sha(object_raw), object_bytes=len(object_raw), matches_original=True)
                save()
            objects = [(out / (n + ".o")).read_bytes() for n, _ in sources]
            request = (b"HIASTKS1" + struct.pack("<3Q", len(objects), 4, 0) + b"step"
                       + b"".join(struct.pack("<Q", len(o)) + o for o in objects))
            write_new(out / "link.request.bin", request)
            report["link"] = {"request_sha256": sha(request), "entry": "step", "object_count": 12}
            invoke(linker, request, out / "link.stdout", out / "link.stderr", report["link"], save, args.timeout)
            image = (out / "link.stdout").read_bytes()
            require(len(image) == 3129344 and image.startswith(b"\x7fELF") and sha(image) == IMAGE,
                    "linked image mismatch")
            write_new(out / "tensor-to-copy-session", image, 0o500)
            unchanged(before)
            for path, expected in tool_before:
                require(file_record(path) == expected, "native tool changed during invocation")
            report.update(status="PASS", native_sha256=sha(image), native_bytes=len(image),
                          compiler_sha256=COMPILER, linker_sha256=LINKER, link_exit=report["link"]["returncode"],
                          functions=614, blocks=8078, instructions=29033, unit_count=12,
                          descriptor_bytes=672, state_format="HTTOCP08", backing_format="HTOTOCP8",
                          all_unit_source_hashes_match_original=True, all_unit_object_hashes_match_original=True,
                          linked_image_hash_matches_original=True, all_package_inputs_unchanged=True,
                          all_native_tools_unchanged=True, native_built_this_invocation=True)
            save()
        print(json.dumps({"status": report["status"], "recipe_id": RECIPE, "output": str(out),
                          "report_sha256": sha((out / ("source-check.json" if check_only else "build.json")).read_bytes()),
                          "unit_count": 12, "native_tool_processes_started": report["native_tool_processes_started"],
                          "target_runtime_executed": False}))
        return 0
    except BaseException as error:
        report.update(status="FAIL", primary_error=repr(error), traceback=traceback.format_exc())
        if before is not None:
            try:
                unchanged(before)
                report["all_package_inputs_unchanged"] = True
            except BaseException as secondary:
                report["all_package_inputs_unchanged"] = False
                report["custody_error"] = repr(secondary)
        try:
            report["partial_artifacts"] = [
                file_record(p, p.name) for p in sorted(out.iterdir())
                if p.is_file() and p.name not in {"build.json", "source-check.json"}
                and not p.name.endswith(".pending")]
            save()
        except BaseException as secondary:
            print("Could not save failure receipt: " + repr(secondary), file=sys.stderr)
        print("FAIL: " + repr(error), file=sys.stderr)
        return 1


def main(argv=None, check_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing paths are rejected")
    if not check_only:
        parser.add_argument("--compiler", type=Path, required=True, help="exact b1c851a0 module compiler")
        parser.add_argument("--linker", type=Path, required=True, help="exact aae650e8 session linker")
        parser.add_argument("--timeout", type=int, default=120, help="seconds per native tool process")
    args = parser.parse_args(argv)
    if not check_only and args.timeout <= 0:
        parser.error("--timeout must be positive")
    return run(args, check_only)


if __name__ == "__main__":
    raise SystemExit(main())
