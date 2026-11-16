"""Run the frozen35 entry with an explicit, read-only private-node observer.

The original entry, bridge, 28/21/159 sources and native image stay unchanged.
original-oracle.json is the exact report emitted by that entry during this
execution; this wrapper emits a separate report that identifies every runtime
observer override and binds the complete original report by SHA256.
"""
from pathlib import Path
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import owner_accounting523 as observer

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, text):
        self.file.write(text); self.file.flush()
        return self.stream.write(text)
    def flush(self):
        self.file.flush(); self.stream.flush()

def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--legacy-directory", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--observer-report", type=Path, required=True)
    args, original_args = parser.parse_known_args()
    if original_args and original_args[0] == "--":
        original_args = original_args[1:]
    observer.need(len(original_args) >= 2, "missing-original-entry-arguments")
    binary, inner_report = map(lambda p: Path(p).resolve(), original_args[:2])
    output = args.observer_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    legacy = args.legacy_directory.resolve()
    original_paths = {name: legacy / name for name in
                      ("bridge.py", "lifecycle.py", "pytorch_tensor_objects_oracle.py")}
    for name, path in original_paths.items():
        observer.need(observer.sha(path.read_bytes()) == observer.SOURCE_PINS[name],
                      "original-source-pin", source=name)
    for option, name in (("--owned-reference", "pytorch_owned_tensor_oracle.py"),
                         ("--session-reference", "pytorch_session_oracle.py"),
                         ("--reference", "pytorch_oracle.py")):
        observer.need(option in original_args, "missing-frozen-reference", option=option)
        path = Path(original_args[original_args.index(option) + 1]).resolve()
        observer.need(observer.sha(path.read_bytes()) == observer.SOURCE_PINS[name],
                      "original-reference-pin", source=name)
        original_paths[name] = path
    source = args.native_source.resolve()
    profile = observer.pin_profile(binary, source)
    if "--expected-native-sha256" in original_args:
        wanted = original_args[original_args.index("--expected-native-sha256") + 1]
        observer.need(wanted == profile["native_sha256"], "entry-image-argument-conflict")
    else:
        original_args.extend(["--expected-native-sha256", profile["native_sha256"]])
    paths = {**original_paths, "native_source": source, "native_image": binary,
             "observer": Path(observer.__file__).resolve(), "launcher": Path(__file__).resolve()}
    paths.update(observer.dependency_paths())
    custody = {name: {"path": str(path), "bytes": path.stat().st_size,
                     "sha256": observer.sha(path.read_bytes())} for name, path in paths.items()}
    invocation = {"argv": sys.argv, "original_entry_argv": [str(original_paths["pytorch_tensor_objects_oracle.py"]),
                  *original_args], "python": sys.version, "python_executable": sys.executable,
                  "cwd": str(Path.cwd()), "profile": profile, "sources": custody,
                  "native_rebuilt": False, "native_frozen_bytes_reused": True}
    (output / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    (output / "legacy-inverse-proof.json").write_text(json.dumps(observer.inverse_proof(), indent=2) + "\n")
    sys.path.insert(0, str(legacy))
    bridge = load_module("bridge", original_paths["bridge.py"])
    load_module("lifecycle", original_paths["lifecycle.py"])
    driver = load_module("frozen35_original_entry", original_paths["pytorch_tensor_objects_oracle.py"])
    original_install = bridge.install
    def install(scope, definitions, owned_program, base_program, executable, trace_directory=None):
        return observer.install(original_install, scope, definitions, owned_program, base_program,
                                executable, trace_directory, profile, output / "owners")
    original_bridge_ast = ast.dump(ast.parse(original_paths["bridge.py"].read_text()),
                                   include_attributes=False)
    original_install_code = original_install.__code__
    bridge.install = install
    overlay = {
        "schema": "heptatorch.frozen35-explicit-observer-overlay.v1",
        "original_bridge_sha256": custody["bridge.py"]["sha256"],
        "original_entry_sha256": custody["pytorch_tensor_objects_oracle.py"]["sha256"],
        "observer_sha256": custody["observer"]["sha256"],
        "launcher_sha256": custody["launcher"]["sha256"],
        "runtime_replacements": ["bridge.install"],
        "derived_program_overrides": ["command", "handle"],
        "derived_session_overrides": ["_v3_begin", "_v3_commit", "exchange", "close", "stop"],
        "original_bridge_close_assertion_replaced": "stats[11] == self.next_id-1",
        "replacement_contract": "Exact actual incarnation increments and source-derived constructor costs; full80-word owner under640 structural rules with explicit source-proved Storage tombstone correction plus inverse-proved original35 restrictions. New99/101 roles explicitly absent; exact stats/reset high-water, no inequality fallback.",
        "unchanged_frozen_numeric_inputs_assertions_tolerances_and_lifetime_sources": True,
        "original_Session_close_is_invoked_directly_after_new_stats_checks": True,
        "post_close_version_gate_runs_original_function_with_only_after-return_completion_marker": True,
        "no_native_memory_writes": True,
        "original_bridge_AST_sha256": observer.sha(original_bridge_ast.encode()),
        "inverse_runtime_restoration_checked_after_execution": False,
    }
    (output / "observer-overlay.json").write_text(json.dumps(overlay, indent=2) + "\n")
    started = time.monotonic()
    previous_argv = sys.argv
    result, caught = 1, None
    stdout_path, stderr_path = output / "original-entry.stdout", output / "original-entry.stderr"
    try:
        sys.argv = [str(original_paths["pytorch_tensor_objects_oracle.py"]), *original_args]
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            with contextlib.redirect_stdout(Tee(sys.stdout, out)), contextlib.redirect_stderr(Tee(sys.stderr, err)):
                result = driver.main()
    except BaseException as error:
        caught = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        sys.argv = previous_argv
        bridge.install = original_install
        overlay["inverse_runtime_restoration_checked_after_execution"] = (
            bridge.install is original_install and bridge.install.__code__ is original_install_code
            and ast.dump(ast.parse(original_paths["bridge.py"].read_text()),
                         include_attributes=False) == original_bridge_ast)
        (output / "observer-overlay.json").write_text(json.dumps(overlay, indent=2) + "\n")
        unchanged = {name: observer.sha(path.read_bytes()) == custody[name]["sha256"]
                     for name, path in paths.items()}
        inner = json.loads(inner_report.read_text()) if inner_report.exists() else None
        rows = inner.get("rows", []) if inner else []
        prepared = "--prepare-only" in original_args
        complete = (len(rows) == 35 and all(row.get("private_node_observer", {}).get("status") == "PASS"
                    and row["private_node_observer"].get("eof_exit_assertions_executed") is True
                    and row["private_node_observer"].get("post_close_parameter_version_assertions_executed") is True
                    for row in rows))
        success = (result == 0 and caught is None and all(unchanged.values())
                   and overlay["inverse_runtime_restoration_checked_after_execution"]
                   and (prepared and inner and inner["status"] == "NOT_RUN"
                        or not prepared and inner and inner["status"] == "PASS" and complete))
        report = {
            "schema": "heptatorch.frozen35-private-node-accounted-run.v1",
            "status": "PREPARED_NOT_RUN" if success and prepared else "PASS" if success else "FAIL",
            "original_entry_return_code": result, "exception": caught,
            "original_oracle_report": {"path": str(inner_report),
                "sha256": observer.sha(inner_report.read_bytes()) if inner_report.exists() else None,
                "bytes": inner_report.stat().st_size if inner_report.exists() else 0,
                "status": inner.get("status") if inner else None,
                "cases": inner.get("cases") if inner else None,
                "failed_cases": inner.get("failed_cases") if inner else None,
                "frames": inner.get("frames") if inner else None,
                "compared_values": inner.get("compared_values") if inner else None},
            "source_custody": custody, "source_files_unchanged_during_run": unchanged,
            "observer_overlay": overlay, "profile": profile,
            "all_original_EOF_and_post_close_Version_gates_executed": complete if not prepared else False,
            "numeric_oracle_rows_reclassified_by_wrapper": False,
            "scope": "Read-only fixed35 accounting for separately pinned523/640 owner, complete actual80-word snapshots and exact source-derived allocations. All35 original case implementations and actual Torch references unchanged. General99/101 ownership roles separately qualified; they are explicitly absent in this fixed35 command domain.",
            "observer_cases": [{"case": row["case"], **row.get("private_node_observer", {})} for row in rows],
            "full_rewrite_complete": False, "pytorch_api_denominator": 2590,
            "wall_seconds": time.monotonic() - started,
        }
        args.observer_report.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.observer_report.resolve().write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"observer_status": report["status"],
                      "original_oracle_status": report["original_oracle_report"]["status"],
                      "cases": report["original_oracle_report"]["cases"],
                      "failed_cases": report["original_oracle_report"]["failed_cases"],
                      "original_report_sha256": report["original_oracle_report"]["sha256"]}))
    return 0 if success else 1

if __name__ == "__main__":
    raise SystemExit(main())
