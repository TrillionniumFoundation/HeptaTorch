#!/usr/bin/env python3
"""Rebuild only the fixed HIA entropy session linker, never an application.

The exact tool-authoring prefix of the pinned original Generator builder is
executed without edits. Its later RNG/model/owner/application section is not
selected. Two real native tool invocations are permitted: compile linker.hepta
with the fixed module compiler, then link that object with the fixed tool linker.
"""
import argparse
import ast
import dis
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import time
import traceback
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ORIGINAL = "56cddf74330705c3058752f65858c0cb021ec2c3da905040dd0113ae986b2552"
COMPILER = "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"
TOOL_LINKER = "fa3065b5dc42bfe09fd29cf773c5adc5a9296c125a6a161daecf2fa885ea160b"
BASE_LINKER = "b0adabf3db5bdda3ee46c0c74630938c0f1532e97bc532d65529c3ea0af83ac7"
BASE_STARTUP = "1271a7805412930cf682d1f08f5aeed2955862f40747d2a4253784f4055fe8b0"
CONTRACT = "8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c"
NEW_AUTHORING = "771160ccfdde84bec0dc8fafd16de78bf4327add86b0913644dd13d05414b831"
NEW_SOURCE = "7bba866ece4f33fb6b1973e33475d868376f2b804d5ce81116cf206cfff31c34"
NEW_PREFIX = "75b74f8441acd38b84d5a4784ab3a37f48652e06caa1ca938595aa1324424a00"
NEW_LINKER = "761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def assignment_to(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def tool_prefix(original):
    """Select whole original AST statements; do not rewrite their contents."""
    text = original.decode("utf-8")
    tree = ast.parse(text)
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    require(len(mains) == 1, "one original main required")
    nodes = mains[0].body
    starts = [i for i, n in enumerate(nodes) if assignment_to(n, "contract_raw")]
    stops = [i for i, n in enumerate(nodes) if assignment_to(n, "frozen")]
    require(len(starts) == len(stops) == 1 and starts[0] < stops[0], "exact tool-only source boundary required")
    selected = nodes[starts[0]:stops[0]]
    lines = text.splitlines(keepends=True)
    segment = "".join(lines[selected[0].lineno-1:selected[-1].end_lineno])
    selected_dump = ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False)
    wrapper = ast.parse("def selected_original_tool_prefix():\n" + segment).body[0]
    require(ast.dump(ast.Module(body=wrapper.body, type_ignores=[]), include_attributes=False) == selected_dump,
            "selected text must exactly encode the original whole AST statements")
    # This bound excludes the entire RNG application authoring tail. In
    # particular no application source arguments or application ROOT are read.
    loaded = set()
    def globals_used(code):
        loaded.update(ins.argval for ins in dis.get_instructions(code)
                      if ins.opname in {"LOAD_NAME", "LOAD_GLOBAL"})
        for item in code.co_consts:
            if isinstance(item, types.CodeType):
                globals_used(item)
    globals_used(compile(ast.Module(body=selected, type_ignores=[]), "<selected-original-prefix-review>", "exec"))
    require(not ({"a", "ROOT", "own_inputs", "profile", "frozen", "bodies", "image"} & loaded),
            "tool-only prefix must not depend on application inputs")
    calls = [n for n in selected if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    last = calls[-1].value
    require(isinstance(last.func, ast.Name) and last.func.id == "link"
            and isinstance(last.args[0], ast.Name) and last.args[0].id == "tool_linker"
            and isinstance(last.args[1], ast.Constant) and last.args[1].value == b"HIASTKL1",
            "prefix must end by linking the compiler-produced linker object")
    return selected, segment.encode(), {
        "original_builder_sha256": ORIGINAL,
        "source_lines_in_original": [selected[0].lineno, selected[-1].end_lineno],
        "whole_original_ast_statements": len(selected),
        "selected_ast_sha256": sha(selected_dump.encode()),
        "selected_text_sha256": sha(segment.encode()),
        "original_statements_modified": False,
        "original_main_called": False,
        "old_application_section_selected": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--stack-build", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-only", action="store_true",
                        help="Check only the fixed source boundary; do not load Hepta or invoke native tools.")
    args = parser.parse_args(argv)
    require(not sys.flags.optimize, "original assertions require Python without -O")
    if not args.review_only:
        require(args.upstream is not None and args.stack_build is not None,
                "actual tool rebuilding requires --upstream and --stack-build")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    report = {
        "schema": "heptatorch.generator-entropy-linker.tool-only-build.v1",
        "status": "FAIL_TOOL_BUILD",
        "processes": [], "inputs": inputs,
        "compiler_executed": False, "tool_linker_executed": False,
        "generator_session_linker_executed": False,
        "old_generator_application_built": False,
        "old_generator_application_executed": False,
        "tensor_application_executed": False, "torch_executed": False,
        "runtime_qualified": False, "all_inputs_unchanged": False,
    }

    def save():
        write_json(out / "build.json", report)

    def bind(path, expected=None):
        path = Path(path)
        require(not path.is_symlink(), "symlinked input: " + str(path))
        path = path.resolve(strict=True)
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode), "regular input required: " + str(path))
            raw = stream.read()
            after = os.fstat(stream.fileno())
        require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_mode)
                == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_mode),
                "input changed during read: " + str(path))
        row = {"bytes": len(raw), "sha256": sha(raw), "mode": stat.S_IMODE(before.st_mode)}
        require(expected is None or row["sha256"] == expected, "input SHA mismatch: " + str(path))
        require(str(path) not in inputs or inputs[str(path)] == row, "input changed: " + str(path))
        inputs[str(path)] = row
        return raw

    try:
        bind(Path(__file__))
        original_path = HERE / "frozen_generator_build.py"
        original = bind(original_path, ORIGINAL)
        selected, source_prefix, proof = tool_prefix(original)
        (out / "tool-only-original-prefix.txt").write_bytes(source_prefix)
        report["source_proof"] = proof
        if args.review_only:
            report["status"] = "SOURCE_AUDITED_NATIVE_TOOLS_NOT_RUN"
        else:
            up = args.upstream.resolve(strict=True)
            base = args.stack_build.resolve(strict=True)
            compiler = base / "native/stack-module-compiler"
            tool_linker = base / "native/stack-linker-linker"
            bind(compiler, COMPILER)
            bind(tool_linker, TOOL_LINKER)
            base_linker = bind(base / "sources/linker-session/authoring.txt", BASE_LINKER)
            bind(base / "sources/linker-session/startup-authoring.txt", BASE_STARTUP)
            contract = json.loads(bind(up / "docs/HIA_SELFHOST_CORE_SUPPORT.json", CONTRACT))

            # Loading this exact original module only defines main; __main__ is
            # false. Nothing calls its application builder. Only the selected
            # original AST prefix below is subsequently executed.
            scope = {"__name__": "frozen_generator_tool_prefix", "__file__": str(original_path)}
            exec(compile(original, str(original_path), "exec"), scope)
            require(scope["COMPILER_SHA"] == COMPILER and scope["TOOL_LINKER_SHA"] == TOOL_LINKER
                    and scope["BASE_LINKER_SHA"] == BASE_LINKER and scope["BASE_STARTUP_SHA"] == BASE_STARTUP
                    and scope["CONTRACT_SHA"] == CONTRACT, "original tool constants changed")
            custody = {}
            for relative in scope["AUTHORING_FILES"]:
                wanted = contract["source_identity"]["files"][relative]
                custody[relative] = sha(bind(up / relative, wanted))
            report["authoring_inputs"] = custody
            report["authoring_contract_sha256"] = CONTRACT
            report["expected_upstream_revision"] = "f33d8da316de8fdf27b504371f1a5310997ff548"
            report["revision_checked_with_git"] = False
            report["all_authoring_sources_match_fixed_contract"] = True

            class ToolsOnly:
                """Record exact original calls and forbid any third executable."""
                def run(self, argv, **kwargs):
                    index = len(report["processes"])
                    require(index < 2, "only the two original tool bootstrap invocations are allowed")
                    tool = compiler if index == 0 else tool_linker
                    require(argv == [str(tool)] and set(kwargs) == {"input", "capture_output", "env", "timeout"}
                            and kwargs["capture_output"] is True and kwargs["env"] == {"PATH": ""}
                            and kwargs["timeout"] == 90, "original native invocation contract changed")
                    raw = kwargs["input"]
                    require(type(raw) is bytes, "real immutable tool input bytes required")
                    if index == 0:
                        require(sha(raw) == NEW_SOURCE and raw == (out / "linker.hepta").read_bytes(),
                                "only the exact fixed entropy linker HIA source may be compiled")
                        require(sha((out / "startup-prefix.bin").read_bytes()) == NEW_PREFIX,
                                "fixed genuine entropy startup prefix required")
                        require(sha((out / "linker-authoring.txt").read_bytes()) == NEW_AUTHORING,
                                "exact derived linker authoring required")
                    else:
                        obj = (out / "linker.o").read_bytes()
                        expected = b"HIASTKL1" + struct.pack("<3Q", 1, 2, 0) + b"f0" + struct.pack("<Q", len(obj)) + obj
                        require(raw == expected, "tool linker input must contain only the actual newly compiled object")
                    bind(tool, COMPILER if index == 0 else TOOL_LINKER)
                    label = "compile-entropy-linker" if index == 0 else "link-entropy-linker"
                    root = out / ("process-" + str(index + 1) + "-" + label)
                    root.mkdir()
                    (root / "input.bin").write_bytes(raw)
                    row = {"name": label, "argv": argv, "tool_sha256": sha(tool.read_bytes()),
                           "input_sha256": sha(raw), "input_bytes": len(raw), "started": False,
                           "stdout_EOF": False, "stderr_EOF": False,
                           "timed_out": False, "directory": str(root)}
                    report["processes"].append(row)
                    save()
                    started = time.monotonic()
                    stdout, stderr = b"", b""
                    try:
                        process = subprocess.run(argv, **kwargs)
                        row.update(started=True, exit_code=process.returncode,
                                   stdout_EOF=True, stderr_EOF=True)
                        stdout, stderr = process.stdout, process.stderr
                    except subprocess.TimeoutExpired as error:
                        stdout, stderr = error.stdout or b"", error.stderr or b""
                        row.update(started=True, timed_out=True, error_type=type(error).__name__)
                        raise
                    except BaseException as error:
                        row.update(error_type=type(error).__name__, error=str(error))
                        if not isinstance(error, OSError):
                            row["execution_outcome_unknown"] = True
                        raise
                    finally:
                        (root / "stdout.bin").write_bytes(stdout)
                        (root / "stderr.bin").write_bytes(stderr)
                        row.update(elapsed_seconds=time.monotonic()-started,
                                   stdout_bytes=len(stdout), stdout_sha256=sha(stdout),
                                   stderr_bytes=len(stderr), stderr_sha256=sha(stderr))
                        report["compiler_executed"] = bool(report["processes"][0]["started"])
                        report["tool_linker_executed"] = len(report["processes"]) == 2 and row["started"]
                        save()
                    return process

            scope.update(up=up, base=base, out=out, compiler=compiler,
                         tool_linker=tool_linker, subprocess=ToolsOnly())
            save()
            exec(compile(ast.Module(body=selected, type_ignores=[]),
                         str(original_path) + "[unchanged-tool-only-prefix]", "exec"), scope)
            require(len(report["processes"]) == 2 and all(
                p["started"] and p["exit_code"] == 0 and p["stdout_EOF"] and p["stderr_EOF"]
                and not p["timed_out"] and p["stderr_bytes"] == 0 for p in report["processes"]),
                "both actual native tools must finish successfully with complete stdout/stderr")
            products = {}
            for relative, wanted in (("linker.hepta", NEW_SOURCE),
                                     ("linker-authoring.txt", NEW_AUTHORING),
                                     ("startup-prefix.bin", NEW_PREFIX),
                                     ("generator-session-linker", NEW_LINKER)):
                path = out / relative
                raw = path.read_bytes()
                require(sha(raw) == wanted, "reproduced native tool identity mismatch: " + relative)
                products[relative] = {"bytes": len(raw), "sha256": sha(raw),
                                      "mode": stat.S_IMODE(path.stat().st_mode)}
            before_functions = {n.name: ast.dump(n, include_attributes=False)
                                for n in ast.parse(base_linker.decode()).body}
            after_functions = {n.name: ast.dump(n, include_attributes=False)
                               for n in ast.parse((out / "linker-authoring.txt").read_text()).body}
            require(before_functions.keys() == after_functions.keys() and len(before_functions) == 44,
                    "44 original linker authoring functions required")
            changed = [name for name in before_functions if before_functions[name] != after_functions[name]]
            require(set(changed) == {"link", "headers", "module_entry"},
                    "only the three original entropy linker adaptations may change")
            loaded = {}
            for name, module in tuple(sys.modules.items()):
                path = getattr(module, "__file__", None)
                if not path:
                    continue
                path = Path(path).resolve()
                if name == "hepta" or name.startswith("hepta.") or name == "tools" or name.startswith("tools."):
                    require(path.is_relative_to(up), "authoring import resolved outside the fixed upstream: " + name)
                if path.is_relative_to(up):
                    relative = str(path.relative_to(up))
                    require(relative in custody, "unbound upstream module loaded: " + relative)
                    loaded[relative] = sha(bind(path, custody[relative]))
            require({"tools/generate_hia_selfhost.py", "tools/generate_hia_elf_linker.py",
                     "hepta/hia_agent_assembly.py"} <= loaded.keys(), "required actual HIA authoring modules not loaded")
            report.update(status="PASS_NATIVE_ENTROPY_LINKER_BUILD_ONLY", products=products,
                          loaded_authoring_modules=loaded, loaded_modules_are_bound_subset=True,
                          linker_changed_authoring_functions=changed,
                          other41_linker_function_asts_unchanged=True,
                          compiler_sha256=COMPILER, tool_linker_sha256=TOOL_LINKER,
                          entropy_linker_sha256=NEW_LINKER,
                          entropy_linker_source_sha256=NEW_SOURCE, startup_prefix_sha256=NEW_PREFIX)
        for name, row in list(inputs.items()):
            bind(Path(name), row["sha256"])
        report["all_inputs_unchanged"] = True
    except BaseException as error:
        report.update(status="FAIL_TOOL_BUILD", error_type=type(error).__name__, error=str(error))
        (out / "failure.txt").write_text(traceback.format_exc())
        checks = {}
        for name, row in list(inputs.items()):
            try:
                bind(Path(name), row["sha256"])
                checks[name] = True
            except BaseException:
                checks[name] = False
        report["input_checks_after_failure"] = checks
        report["all_inputs_unchanged"] = bool(checks) and all(checks.values())
    save()
    print(json.dumps({key: report[key] for key in (
        "status", "compiler_executed", "tool_linker_executed", "old_generator_application_executed",
        "generator_session_linker_executed", "all_inputs_unchanged")}))
    return 0 if report["status"] in {"SOURCE_AUDITED_NATIVE_TOOLS_NOT_RUN", "PASS_NATIVE_ENTROPY_LINKER_BUILD_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
