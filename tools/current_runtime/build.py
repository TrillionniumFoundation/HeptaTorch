#!/usr/bin/env python3
"""Cold-build a complete HIA runtime without rebuilding its historical parents."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import sys
import time
import types


ROOT = Path(__file__).resolve().parents[2]
COMPILER = "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"
LINKER = "aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16"
INSPECTOR = "71b72260014417db72ffcebd80c86543e257c80d76cdd6c878b41feaa117cf92"
SUPPORT = {
    "rng03": "a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3",
    "owners": "9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884",
}
LIMITS = {"source_bytes": 1048576, "object_bytes": 1048576,
          "functions": 256, "blocks": 2560, "blocks_per_function": 256,
          "instructions": 8192, "parameters": 16, "results": 3,
          "objects": 64, "link_bytes": 8388608, "stack_path_bytes": 8384512,
          "stack_bytes": 8388608, "stack_reserve_bytes": 4096}
NAME = rb"[A-Za-z_][A-Za-z_0-9]*"
HEADER = re.compile(rb"fn (" + NAME + rb")\(([^\n]*)\) -> \(([^\n]*)\) "
                    rb"effects\(([^\n]*)\) (external\(\)|entry\(" + NAME + rb"\))\n")
CALL = re.compile(rb'\{"function"\s*:\s*"(' + NAME + rb')"\}')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "expected a lowercase SHA-256")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def relative(root, value):
    require(isinstance(value, str) and value and "\\" not in value, "invalid source path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and ".." not in path.parts,
            "source paths must be normalized and relative: " + value)
    result = root
    for part in path.parts:
        result /= part
        require(not result.is_symlink(), "symlinked source path: " + value)
    return result


def read_file(path, expected=None):
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                "regular file required: " + str(path))
        raw = stream.read()
        after = os.fstat(stream.fileno())
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "input changed during read: " + str(path))
    if expected is not None:
        require(sha(raw) == digest(expected), "input SHA-256 mismatch: " + str(path))
    return raw


def signature(match):
    parameters = match[2].split(b",") if match[2] else []
    require(all(re.fullmatch(NAME + rb":" + NAME, p) for p in parameters),
            "invalid function parameters")
    results = tuple(match[3].split(b",")) if match[3] else ()
    effects = tuple(match[4].split(b",")) if match[4] else ()
    require(len(parameters) <= LIMITS["parameters"] and len(results) <= LIMITS["results"],
            "function exceeds the pinned parameter/result limits")
    require(len(set(effects)) == len(effects), "duplicate effect")
    return (tuple(p.split(b":")[1] for p in parameters), results, tuple(sorted(effects)))


def parse_source(raw, unit=False):
    """Check the emitted source envelope; native tools still check HIA semantics."""
    first = re.match(rb"hia-asm.v1 hepta.hia.program.v8 " + NAME + rb"\n", raw)
    require(first is not None, "invalid HIA source header")
    cursor = first.end()
    exports = []
    if raw[cursor:].startswith(b"exports("):
        end = raw.find(b")\n", cursor)
        require(end >= 0, "unterminated exports")
        exports = raw[cursor + 8:end].decode("ascii").split(",")
        require(len(set(exports)) == len(exports) and all(re.fullmatch(NAME, n.encode()) for n in exports),
                "invalid or duplicate export")
        cursor = end + 2
    require(bool(exports) == unit, "unit exports required; aggregate runtime must have none")
    definitions, imports, signatures, calls = {}, {}, {}, {}
    while cursor < len(raw):
        match = HEADER.match(raw, cursor)
        require(match is not None, "unparsed HIA top-level bytes at " + str(cursor))
        name = match[1].decode("ascii")
        require(name not in signatures, "duplicate function: " + name)
        signatures[name] = signature(match)
        if match[5] == b"external()":
            imports[name] = signatures[name]
            cursor = match.end()
            continue
        end = raw.find(b"\nend\n", match.end())
        require(end >= 0, "unterminated function: " + name)
        body = raw[cursor:end + 5]
        block_headers = re.findall(rb"^block " + NAME + rb"\(([^\n]*)\)$", body, re.M)
        require(0 < len(block_headers) <= LIMITS["blocks_per_function"],
                "invalid block count: " + name)
        require(all(len(b.split(b",")) <= LIMITS["parameters"] for b in block_headers if b),
                "block exceeds parameter limit: " + name)
        targets = CALL.findall(body)
        require(len(targets) == body.count(b" = call("), "unparsed call: " + name)
        calls[name] = {target.decode("ascii") for target in targets}
        definitions[name] = body
        cursor = end + 5
    require(definitions, "empty HIA source")
    require(set(exports) <= definitions.keys(), "export is not locally defined")
    if not unit:
        require(not imports, "aggregate runtime contains an import declaration")
    result = {"definitions": definitions, "imports": imports, "signatures": signatures,
              "exports": exports, "calls": calls, "functions": len(definitions),
              "blocks": len(re.findall(rb"^block ", raw, re.M)), "instructions": raw.count(b" = ")}
    if unit:
        for key, actual in (("source_bytes", len(raw)), ("functions", len(signatures)),
                            ("blocks", result["blocks"]), ("instructions", result["instructions"])):
            require(actual <= LIMITS[key], "unit exceeds pinned " + key)
        for name, targets in calls.items():
            require(targets <= signatures.keys(), "undeclared callee in " + name)
    return result


def check_sources(manifest, source_root, bind):
    require(manifest.get("schema") == "heptatorch.complete-runtime-source.v1", "unsupported manifest schema")
    require(set(manifest) <= {"schema", "name", "runtime", "units", "expected_native_sha256"},
            "unknown manifest field")
    require(isinstance(manifest.get("name"), str) and manifest["name"], "missing profile name")
    source = manifest["runtime"]
    require(set(source) == {"path", "sha256", "functions", "blocks", "instructions"}, "invalid runtime record")
    raw = bind(relative(source_root, source["path"]), source["sha256"])
    runtime = parse_source(raw)
    for key in ("functions", "blocks", "instructions"):
        require(type(source[key]) is int and source[key] == runtime[key], "runtime count mismatch: " + key)
    entries = manifest["units"]
    require(isinstance(entries, list) and 3 <= len(entries) <= LIMITS["objects"], "invalid unit count")
    require(entries[0]["name"] == "rng03" and entries[-1]["name"] == "owners", "support unit order mismatch")
    units, names, paths, core, providers = [], set(), {source["path"]}, {}, {}
    for entry in entries:
        require(set(entry) <= {"name", "path", "sha256", "expected_object_sha256"}
                and {"name", "path", "sha256"} <= set(entry), "invalid unit record")
        name = entry["name"]
        require(isinstance(name, str) and re.fullmatch(NAME, name.encode()) and name not in names,
                "invalid or duplicate unit name")
        require(name not in {"runtime", "heptatorch", "build", "link", "manifest"}, "reserved unit name")
        require(entry["path"] not in paths, "duplicate source path")
        names.add(name); paths.add(entry["path"])
        unit_raw = bind(relative(source_root, entry["path"]), entry["sha256"])
        unit = parse_source(unit_raw, unit=True)
        if name in SUPPORT:
            require(sha(unit_raw) == SUPPORT[name], "changed pinned support unit: " + name)
        else:
            require(set(unit["exports"]) == unit["definitions"].keys(), "core unit must export all its functions")
            for function, body in unit["definitions"].items():
                require(function not in core, "core function defined in two units: " + function)
                require(runtime["definitions"].get(function) == body, "core function differs from runtime: " + function)
                core[function] = name
        for exported in unit["exports"]:
            require(exported not in providers, "duplicate global export: " + exported)
            providers[exported] = (name, unit["signatures"][exported])
        if "expected_object_sha256" in entry:
            digest(entry["expected_object_sha256"])
        units.append((entry, unit_raw, unit))
    require(core.keys() == runtime["definitions"].keys(), "units do not cover the complete runtime")
    dependencies = {}
    for entry, _, unit in units:
        deps = set()
        for imported, sig in unit["imports"].items():
            require(imported in providers, "unresolved external function: " + imported)
            provider, actual = providers[imported]
            require(actual == sig and provider != entry["name"], "external signature/effect mismatch: " + imported)
            deps.add(provider)
        dependencies[entry["name"]] = deps
    resolved = set()
    while len(resolved) < len(units):
        ready = {name for name, deps in dependencies.items() if name not in resolved and deps <= resolved}
        require(ready, "cyclic module imports")
        resolved |= ready
    if "expected_native_sha256" in manifest:
        digest(manifest["expected_native_sha256"])
    summary = {"source_sha256": sha(raw), "functions": runtime["functions"],
               "blocks": runtime["blocks"], "instructions": runtime["instructions"],
               "units_planned": len(units), "complete_core_bodies_matched": len(core),
               "global_exports": len(providers),
               "module_dependencies": {n: sorted(v) for n, v in dependencies.items()}}
    return raw, units, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--linker", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Check source identity and partitions; start no processes.")
    args = parser.parse_args(argv)
    require(not sys.flags.optimize, "run with Python assertions enabled for the fixed stack inspector")
    require(args.check_only or (args.compiler and args.linker), "compiler and linker required for a cold build")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    report = {"schema": "heptatorch.complete-runtime-build.v1", "status": "CHECKING_SOURCES",
              "compiler_sha256": COMPILER, "linker_sha256": LINKER, "limits": LIMITS,
              "objects_reused": False, "historical_runtime_executed": False,
              "result_executable_executed": False, "torch_executed": False,
              "runtime_qualified": False, "full_pytorch_rewrite_complete": False,
              "units": [], "processes": []}

    def bind(path, expected=None):
        raw = read_file(path, expected)
        require(path not in inputs or inputs[path] == raw, "bound input changed: " + str(path))
        inputs[path] = raw
        return raw

    def save():
        temporary = out / "build.json.tmp"
        temporary.write_bytes(json_bytes(report))
        temporary.replace(out / "build.json")

    def run(tool, payload, label, stdout_path):
        record = {"name": label, "input_sha256": sha(payload), "input_bytes": len(payload),
                  "started": False, "stdout_EOF": False, "stderr_EOF": False}
        report["processes"].append(record)
        save()
        started = time.monotonic()
        try:
            process = subprocess.run([str(tool)], input=payload, capture_output=True,
                                     cwd=out, env={"PATH": ""}, timeout=120)
            record.update(started=True, exit_code=process.returncode, stdout_EOF=True, stderr_EOF=True)
            stdout, stderr = process.stdout, process.stderr
        except subprocess.TimeoutExpired as error:
            record.update(started=True, timed_out=True, elapsed_seconds=time.monotonic() - started)
            stdout_path.write_bytes(error.stdout or b"")
            (out / (label + ".stderr")).write_bytes(error.stderr or b"")
            raise
        stdout_path.write_bytes(stdout)
        (out / (label + ".stderr")).write_bytes(stderr)
        record.update(elapsed_seconds=time.monotonic() - started, stdout_bytes=len(stdout),
                      stdout_sha256=sha(stdout), stderr_bytes=len(stderr), stderr_sha256=sha(stderr))
        save()
        require(process.returncode == 0 and not stderr, "native tool refused: " + label)
        require(len(stdout) >= 64 and stdout[:6] == b"\x7fELF\x02\x01"
                and struct.unpack_from("<H", stdout, 18)[0] == 62, "invalid x86-64 ELF: " + label)
        return stdout

    try:
        bind(Path(__file__).resolve())
        manifest_raw = bind(args.manifest.resolve())
        manifest = json.loads(manifest_raw, object_pairs_hook=unique_object)
        runtime, units, summary = check_sources(manifest, args.source_root.resolve(strict=True), bind)
        report.update(summary, profile=manifest["name"], manifest_sha256=sha(manifest_raw))
        (out / "manifest.json").write_bytes(manifest_raw)
        (out / "runtime.hepta").write_bytes(runtime)
        for entry, raw, _ in units:
            (out / (entry["name"] + ".hepta")).write_bytes(raw)
        if args.check_only:
            report["status"] = "SOURCE_CHECKED_NATIVE_NOT_RUN"
        else:
            inspector_raw = bind(ROOT / "tools/stack_budget/inspect_stack.py", INSPECTOR)
            inspector = types.ModuleType("fixed_stack_inspector")
            exec(compile(inspector_raw, "<fixed-stack-inspector>", "exec"), inspector.__dict__)
            tool_dir = out / "toolchain"
            tool_dir.mkdir()
            snapshots = []
            for path, name, pin in ((args.compiler, "stack-module-compiler", COMPILER),
                                    (args.linker, "stack-linker-session", LINKER)):
                raw = bind(path.resolve(), pin)
                snapshot = tool_dir / name
                snapshot.write_bytes(raw); snapshot.chmod(0o500)
                snapshots.append((snapshot, raw))
            compiler, linker = [path for path, _ in snapshots]
            objects, views = [], []
            report["status"] = "COMPILING"
            for entry, raw, parsed in units:
                name = entry["name"]
                obj = run(compiler, raw, name, out / (name + ".o"))
                require(len(obj) <= LIMITS["object_bytes"] and struct.unpack_from("<H", obj, 16)[0] == 1,
                        "invalid or oversized relocatable object: " + name)
                if "expected_object_sha256" in entry:
                    require(sha(obj) == entry["expected_object_sha256"], "reproduced object mismatch: " + name)
                view = inspector.inspect(raw, obj)
                views.append(view); objects.append(obj)
                report["units"].append({"name": name, "source_sha256": sha(raw), "object_sha256": sha(obj),
                    "object_bytes": len(obj), "functions": parsed["functions"], "blocks": parsed["blocks"],
                    "instructions": parsed["instructions"], "compiler_exit": 0, "compiler_stderr_bytes": 0})
                save()
            budget = inspector.longest(views)
            require(budget["maximum_path_bytes"] <= LIMITS["stack_path_bytes"], "native stack budget exceeded")
            report["stack"] = {k: v for k, v in budget.items() if k != "per_function_path_bytes"}
            request = b"HIASTKS1" + struct.pack("<3Q", len(objects), 4, 0) + b"step"
            request += b"".join(struct.pack("<Q", len(obj)) + obj for obj in objects)
            require(len(request) <= LIMITS["link_bytes"], "native linker input limit exceeded")
            (out / "link.request.bin").write_bytes(request)
            report["status"] = "LINKING"
            image = run(linker, request, "link", out / "link.stdout")
            require(len(image) <= LIMITS["link_bytes"] and struct.unpack_from("<H", image, 16)[0] == 2,
                    "invalid or oversized executable")
            if "expected_native_sha256" in manifest:
                require(sha(image) == manifest["expected_native_sha256"], "reproduced native image mismatch")
            for path, raw in snapshots:
                require(read_file(path) == raw, "tool snapshot changed: " + str(path))
            report.update(status="PASS_NATIVE_BUILD_ONLY", native_sha256=sha(image), native_bytes=len(image),
                          native_basename="heptatorch", link_exit=0, stack_inspector_sha256=INSPECTOR)
        for path, raw in inputs.items():
            require(read_file(path) == raw, "input changed during build: " + str(path))
        report["inputs"] = {str(path): {"bytes": len(raw), "sha256": sha(raw)} for path, raw in inputs.items()}
        report["all_inputs_unchanged"] = True
        if not args.check_only:
            (out / "link.stdout").rename(out / "heptatorch")
            (out / "heptatorch").chmod(0o500)
        save()
        print(json.dumps({k: report[k] for k in ("status", "source_sha256", "functions", "units_planned")}
                         | {"native_sha256": report.get("native_sha256"), "runtime_qualified": False}))
        return 0
    except Exception as error:
        report.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        save()
        print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
