#!/usr/bin/env python3
"""Portable finite boundary gate for the pinned expanded native compiler.

Fixtures are authored directly as HIA bytes; positive expectations are small
integer oracles. Native source compilation and program execution use only the
supplied ELF compiler/executable. Python owns fixture authoring and observation.
Only Python's standard library and Linux execution are needed for these16cases.
Original fixture bytes are pinned; no upstream checkout or Python modules are
loaded. This gate is finite evidence, not upstream/full-cold qualification.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import subprocess
import time

PIN = "c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6"
HEADER = ("hia-asm.v1 hepta.hia.program.v8 selfhost_exec\n"
          "fn f0(v0:readbuf,v1:mutbuf,v2:mutbuf) -> (u32) effects(read,write,trap) entry(b0)\n"
          "block b0(v0:readbuf,v1:mutbuf,v2:mutbuf)\n")
FIXTURE_HASHES = {
    "functions-129": "4d072c79f61710b251529574b9c83b0be651eebbe84585de4d6646ab6d5aac36",
    "functions-255": "dc600f65a8ca79a753653ccb6e314d19e920a91685268e206ba0e02213d149a6",
    "functions-256": "9566518ffa77efa127c554b49288e3df73862583223b3be25724cdaf24a4c50f",
    "functions-257": "c23e46830453b85af34cae7687a735d6f1641ae60d9992b053f919c3ee8ed568",
    "blocks-1537": "4c224ba9bc072763be8a7fafdb032808d824454699e26f2e38302b54d486c1fd",
    "blocks-2559": "c077fd1aa8beda22010624b7677f05a39418f116d248d0cb360263cb490451e3",
    "blocks-2560": "3144d53bcdcba80ce15e95424ff6d1834bb9450e60ab1ab73514e4880bce6e64",
    "blocks-2561": "c88252ab1bf98bd0c67e348ab98f96e7e263cee7cfeec49f30dd55e48de0d9d8",
    "high-index-cycle-reachable": "f6ad23c5433f22ff757d18fec0f78d6e698968ca5e6f70a9e4da2ce5dc476738",
    "high-index-cycle-unreachable": "0a61546f73fb5c46c6a603521b757397847105bffbac79926d400c5d15775b6f",
    "high-index-cycle-nontail-self": "9811f849da7d70e14bc6ff4fc5bf83b06b584ba062e2bd52e5e192b40e3a08a6",
    "blocks-per-function-257": "c633ca41eb8d547f56dafed4a3d8ce4a88a387b278c6724e1c9b5270e84e2717",
    "instructions-8192": "8f3de38202aa8e9573f15bca0a48e0b7600a2716a35af4788e16843f8bda7189",
    "instructions-8193": "6f984ca396d4667c1151873a68677c2becec77b398933db6d3632f3b1f8ae059",
    "ssa-names-2048": "ea69b22eed82e66a0c99d44d8949212b0d9a3d497987586511f27d6351df27f2",
    "ssa-names-2049": "b6d3cc1456829657717152d96349c616781850696fac84b95bcba238ab1505f1",
}


def sha(raw):
    return sha256(raw).hexdigest()


def output_word(value, serial):
    lines = []
    for index in range(4):
        current = value
        if index:
            shift, shifted = "v" + str(serial), "v" + str(serial + 1)
            serial += 2
            lines.extend([f'  {shift}:u32 = const() {{"value":{index * 8}}}',
                          f"  {shifted}:u32 = int.shr.checked({value},{shift})"])
            current = shifted
        position, byte = "v" + str(serial), "v" + str(serial + 1)
        serial += 2
        lines.extend([f'  {position}:usize = const() {{"value":{index}}}',
                      f'  {byte}:u8 = int.cast({current}) {{"mode":"wrap"}}',
                      f"  _ = buffer.store(v1,{position},{byte})"])
    result = "v" + str(serial)
    lines.extend([f'  {result}:u32 = const() {{"value":4}}', f"  ret({result})", "end"])
    return "\n".join(lines) + "\n"


def functions_source(count, *, cycle=None):
    body = HEADER + ('  v3:u32 = call() {"function":"f1"}\n'
                     '  v4:u32 = const() {"value":1}\n'
                     '  v5:u32 = int.add.wrap(v3,v4)\n') + output_word("v5", 6)
    for function in range(1, count):
        body += (f"fn f{function}() -> (u32) effects(read,write,trap) entry(b0)\n"
                 "block b0()\n")
        callee = function + 1 if function < count - 1 else None
        if cycle == "unreachable":
            callee = None
        if cycle in {"reachable", "unreachable"}:
            if function == count - 2:
                callee = count - 1
            elif function == count - 1:
                callee = count - 2
        if cycle == "nontail-self" and function == count - 1:
            callee = function
        if callee is None:
            body += '  v0:u32 = const() {"value":1}\n  ret(v0)\nend\n'
        else:
            body += (f'  v0:u32 = call() {{"function":"f{callee}"}}\n'
                     '  v1:u32 = const() {"value":1}\n'
                     '  v2:u32 = int.add.wrap(v0,v1)\n  ret(v2)\nend\n')
    return body.encode()


def blocks_source(total, *, per_function=256):
    counts, remaining = [], total - 1
    while remaining:
        n = min(per_function, remaining)
        counts.append(n)
        remaining -= n
    body = HEADER + ('  v3:u32 = const() {"value":0}\n'
                     '  v4:u32 = call(v3) {"function":"f1"}\n') + output_word("v4", 5)
    for function, count in enumerate(counts, 1):
        body += f"fn f{function}(v0:u32) -> (u32) effects(read,write,trap) entry(b0)\n"
        # Source order is ascending; actual control traverses0,last,...,1.
        # The late block near the total bound must therefore be replayed and
        # its predecessor merge validated by the ordinary resource worklist.
        for block in range(count):
            value, one, result = ("v" + str(3 * block + i) for i in range(3))
            body += (f"block b{block}({value}:u32)\n"
                     f'  {one}:u32 = const() {{"value":1}}\n'
                     f"  {result}:u32 = int.add.wrap({value},{one})\n")
            terminal = count == 1 or block == 1
            if not terminal:
                target = count - 1 if block == 0 else block - 1
                body += f"  jump b{target}({result})\n"
            elif function < len(counts):
                returned = "v" + str(3 * count)
                body += (f'  {returned}:u32 = call({result}) {{"function":"f{function + 1}"}}\n'
                         f"  ret({returned})\n")
            else:
                body += f"  ret({result})\n"
        body += "end\n"
    return body.encode()


def ssa_source(count):
    body = HEADER
    for serial in range(3, count):
        body += f'  v{serial}:u32 = const() {{"value":0}}\n'
    return (body + f"  ret(v{count - 1})\nend\n").encode()


def instruction_source(count):
    """Preserve the audited8192/8193 fixture bytes without importing Hepta."""
    if count not in (8192, 8193):
        raise ValueError("instruction fixture count is outside this exact gate")
    pieces = []
    function_count = (count + 1023) // 1024
    for function in range(function_count):
        operations = min(1024, count - 1024 * function)
        if function == 0:
            pieces.append(HEADER)
            pieces.append('  v3:u32 = const() {"value":0}\n')
            current, serial, used = 3, 4, 1
        else:
            pieces.append(f"fn f{function}(v0:u32) -> (u32) effects(read,write,trap) entry(b0)\nblock b0(v0:u32)\n")
            current, serial, used = 0, 1, 0
        one = serial
        pieces.append(f'  v{one}:u32 = const() {{"value":1}}\n')
        serial += 1
        used += 1
        has_next = function + 1 < function_count
        while used < operations - int(has_next):
            pieces.append(f"  v{serial}:u32 = int.add.wrap(v{current},v{one})\n")
            current = serial
            serial += 1
            used += 1
        if has_next:
            pieces.append(f'  v{serial}:u32 = call(v{current}) {{"function":"f{function + 1}"}}\n')
            current = serial
        pieces.append(f"  ret(v{current})\nend\n")
    return "".join(pieces).encode(), count - 2 * function_count


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    resource.setrlimit(resource.RLIMIT_STACK, (8 << 20, 8 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))


def execute(path, data):
    started = time.monotonic()
    try:
        result = subprocess.run([str(path)], input=data, capture_output=True, timeout=20,
                                env={"PATH": "", "LC_ALL": "C"}, preexec_fn=limits)
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess([str(path)], -1001, exc.stdout or b"", exc.stderr or b"")
    return result, time.monotonic() - started


def main(compiler, output):
    if not compiler.is_absolute() or not output.is_absolute():
        raise ValueError("--compiler and --output must be absolute paths")
    compiler, output = compiler.resolve(), output.resolve()
    raw_compiler = compiler.read_bytes()
    if sha(raw_compiler) != PIN:
        raise ValueError("expanded compiler hash mismatch")
    cases = []
    for count in (129, 255, 256, 257):
        cases.append(("functions-" + str(count), functions_source(count), 0 if count <= 256 else 2,
                      count.to_bytes(4, "little") if count <= 256 else None))
    for count in (1537, 2559, 2560, 2561):
        cases.append(("blocks-" + str(count), blocks_source(count), 0 if count <= 2560 else 2,
                      (count - 1).to_bytes(4, "little") if count <= 2560 else None))
    for cycle in ("reachable", "unreachable", "nontail-self"):
        cases.append(("high-index-cycle-" + cycle, functions_source(256, cycle=cycle), 2, None))
    cases.append(("blocks-per-function-257", blocks_source(258, per_function=257), 2, None))
    for count in (8192, 8193):
        raw, result = instruction_source(count)
        cases.append(("instructions-" + str(count), raw, 0 if count == 8192 else 2,
                      bytes(result) if count == 8192 else None))
    for count in (2048, 2049):
        cases.append(("ssa-names-" + str(count), ssa_source(count), 0 if count == 2048 else 2,
                      b"" if count == 2048 else None))

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, source, compile_status, expected in cases:
        if sha(source) != FIXTURE_HASHES[name]:
            raise ValueError("audited fixture bytes changed: " + name)
        directory = output / name
        directory.mkdir(exist_ok=True)
        (directory / "source.hepta").write_bytes(source)
        row = {"case": name, "source_bytes": len(source), "source_sha256": sha(source),
               "expected_compile_exit": compile_status}
        compiled, seconds = execute(compiler, source)
        (directory / "compiler.stdout.bin").write_bytes(compiled.stdout)
        (directory / "compiler.stderr.bin").write_bytes(compiled.stderr)
        row.update(compile_exit=compiled.returncode, compile_seconds=seconds,
                   produced_bytes=len(compiled.stdout), produced_sha256=sha(compiled.stdout))
        passed = compiled.returncode == compile_status and compiled.stderr == b""
        if compile_status == 0 and passed:
            executable = directory / "program"
            executable.write_bytes(compiled.stdout)
            executable.chmod(0o500)
            observed, elapsed = execute(executable, b"")
            (directory / "program.stdout.bin").write_bytes(observed.stdout)
            (directory / "program.stderr.bin").write_bytes(observed.stderr)
            (directory / "expected.stdout.bin").write_bytes(expected)
            row.update(execute_exit=observed.returncode, execute_seconds=elapsed,
                       expected_output_bytes=len(expected), expected_output_sha256=sha(expected),
                       observed_output_bytes=len(observed.stdout), observed_output_sha256=sha(observed.stdout))
            passed = passed and observed.returncode == 0 and observed.stderr == b"" and observed.stdout == expected
        elif compile_status != 0:
            passed = passed and compiled.stdout == b""
        row["pass"] = passed
        rows.append(row)
        (directory / "observation.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    report = {"schema": "heptatorch.compiler-capacity-boundaries.v1", "compiler_sha256": PIN,
              "compiler_unchanged": compiler.read_bytes() == raw_compiler,
              "scope": "finite local capacity/CFG/call-boundary compilation and Linux x86-64 execution",
              "full_cold_or_upstream_qualification": False,
              "cases": rows, "passed": sum(row["pass"] for row in rows), "total": len(rows),
              "resource_limits": {"cpu_seconds": 10, "wall_seconds": 20, "stack_bytes": 8 << 20,
                                  "address_bytes": 256 << 20}, "runtime_path": "empty",
              "upstream_imports_required": False,
              "original_audited_fixture_hashes_match": True,
              "checker_limits_modified": False}
    report["pass"] = report["compiler_unchanged"] and report["passed"] == report["total"]
    (output / "boundary-results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {"schema": "heptatorch.compiler-capacity-summary.v1", "pass": report["pass"],
               "compiler_sha256": PIN, "passed": report["passed"], "total": report["total"],
               "failed_cases": [row["case"] for row in rows if not row["pass"]],
               "fixture_hashes_match": True, "upstream_imports_required": False,
               "full_cold_or_upstream_qualification": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--compiler", type=Path, required=True)
    arguments.add_argument("--output", type=Path, required=True)
    args = arguments.parse_args()
    main(args.compiler, args.output)
