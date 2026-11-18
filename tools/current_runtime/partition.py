#!/usr/bin/env python3
"""Partition complete HIA source into reproducible, independently compiled units."""

import argparse
import importlib.util
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("current_runtime_build", HERE / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def partition(runtime, rng, owners, prefix):
    core = build.parse_source(runtime)["definitions"]
    support = build.parse_source(rng, unit=True)
    external = set(support["exports"])
    ordered, done, active = [], set(), set()

    def visit(name):
        if name in external:
            return
        build.require(name in core and name not in active, "missing or cyclic function: " + name)
        if name in done:
            return
        active.add(name)
        for called in sorted(set(build.CALL.findall(core[name]))):
            visit(called.decode("ascii"))
        active.remove(name)
        done.add(name)
        ordered.append(name)

    for name in core:
        visit(name)
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
    units = [("rng03", rng)]
    for i, names in enumerate(groups):
        calls = {callee.decode("ascii") for name in names for callee in build.CALL.findall(core[name])}
        declarations = []
        for name in sorted(calls - set(names)):
            body = support["definitions"][name] if name in external else core[name]
            declarations.append(re.sub(rb" entry\(\w+\)$", b" external()", body.split(b"\n")[0]) + b"\n")
        module = f"{prefix}_{i:02d}"
        source = (f"hia-asm.v1 hepta.hia.program.v8 {module}\nexports({','.join(names)})\n".encode()
                  + b"".join(declarations) + b"".join(core[name] for name in names))
        units.append((module, source))
    units.append(("owners", owners))
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--rng", type=Path, default=HERE / "source/units/rng03.hepta")
    parser.add_argument("--owners", type=Path, default=HERE / "source/units/owners.hepta")
    parser.add_argument("--name", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--expected-build", type=Path, help="Require every old source/object/ELF pin when reproducing a known build.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build.require(re.fullmatch(build.NAME, args.prefix.encode("ascii")), "invalid module prefix")
    inputs = {p.resolve(): build.read_file(p.resolve())
              for p in (Path(__file__), HERE / "build.py", args.runtime, args.rng, args.owners)}
    runtime, rng, owners = [inputs[p.resolve()] for p in (args.runtime, args.rng, args.owners)]
    build.require(build.sha(rng) == build.SUPPORT["rng03"], "RNG support pin mismatch")
    build.require(build.sha(owners) == build.SUPPORT["owners"], "owner support pin mismatch")
    units = partition(runtime, rng, owners, args.prefix)
    parsed = build.parse_source(runtime)
    manifest = {"schema": "heptatorch.complete-runtime-source.v1", "name": args.name,
                "runtime": {"path": "runtime.hepta", "sha256": build.sha(runtime),
                            **{k: parsed[k] for k in ("functions", "blocks", "instructions")}},
                "units": [{"name": name, "path": f"units/{name}.hepta", "sha256": build.sha(raw)}
                          for name, raw in units]}
    if args.expected_build:
        old_raw = build.read_file(args.expected_build.resolve())
        inputs[args.expected_build.resolve()] = old_raw
        old = json.loads(old_raw, object_pairs_hook=build.unique_object)
        build.require(old["status"] in {"PASS", "PASS_NATIVE_BUILD_ONLY"} and old["compiler_sha256"] == build.COMPILER
                      and old["linker_sha256"] == build.LINKER and old["link_exit"] == 0,
                      "expected build must be a successful build with the same native tools")
        build.require(old["source_sha256"] == build.sha(runtime), "expected aggregate source mismatch")
        build.require(len(old["units"]) == len(units), "expected unit count mismatch")
        for expected, entry in zip(old["units"], manifest["units"]):
            build.require(expected["name"] == entry["name"] and expected["source_sha256"] == entry["sha256"],
                          "repartitioned source mismatch: " + entry["name"])
            entry["expected_object_sha256"] = build.digest(expected["object_sha256"])
        manifest["expected_native_sha256"] = build.digest(old["native_sha256"])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "units").mkdir()
    (out / "runtime.hepta").write_bytes(runtime)
    for name, raw in units:
        (out / "units" / (name + ".hepta")).write_bytes(raw)
    (out / "manifest.json").write_bytes(build.json_bytes(manifest))
    _, _, summary = build.check_sources(manifest, out, build.read_file)
    for path, raw in inputs.items():
        build.require(build.read_file(path) == raw, "partition input changed: " + str(path))
    report = {"schema": "heptatorch.complete-runtime-partition.v1", "status": "SOURCE_CHECKED_NATIVE_NOT_RUN",
              **summary, "manifest_sha256": build.sha((out / "manifest.json").read_bytes()),
              "compiler_executed": False, "runtime_executed": False,
              "original_unit_pins_reproduced": bool(args.expected_build),
              "inputs": {str(p): build.sha(raw) for p, raw in inputs.items()}}
    (out / "partition.json").write_bytes(build.json_bytes(report))
    print(json.dumps({k: report[k] for k in ("status", "source_sha256", "functions", "units_planned", "original_unit_pins_reproduced")}))


if __name__ == "__main__":
    main()
