#!/usr/bin/env python3
"""Create-only original696 evidence sealing and recorded76 admission.

This entry never executes a Tensor engine, compiler, Torch, subprocess or live
/proc reader. Sealing proves custody only. Recorded comparison is a separate
explicit operation, with its own report and fresh output directory.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tarfile
import traceback

HERE = Path(__file__).resolve().parent
COMMON = "a4ac713abf15756548d6b0be15c1d8287f73e6f4dee40a812415c1daf160f207"
raw = (HERE / "admission_common.py").read_bytes()
if hashlib.sha256(raw).hexdigest() != COMMON:
    raise AssertionError("fixed admission_common source")
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("admission_common", HERE / "admission_common.py")
common = importlib.util.module_from_spec(spec)
sys.modules["admission_common"] = common
spec.loader.exec_module(common)
from admission_common import (
    SHA, BOUND, AUDIT, need, read, data, write, safe_path, same, digest_file,
    load, regular, stable_stat, recheck_bound,
)
EXTRA_AUDIT = {"resource_limit_attempts": 0}

def no_resource_change(event, args):
    if event in {"resource.prlimit", "resource.setrlimit"}:
        EXTRA_AUDIT["resource_limit_attempts"] += 1
        raise AssertionError(("recorded-only-no-resource-limit-change", event))

sys.addaudithook(no_resource_change)

def metadata(pin):
    return {key: pin[key] for key in ("bytes", "sha256", "mode")}

def pinned(path, pin):
    return read(path, pin["sha256"], pin["bytes"], pin["mode"])

def framed_inverse(before, after, record):
    need(SHA(before) == record["before_sha256"] and SHA(after) == record["after_sha256"],
         "whole-source-inverse-original-and-successor-SHA")
    value = before
    for edit in record["edits"]:
        need(set(edit) == {"before", "after", "count"} and type(edit["count"]) is int
             and edit["count"] > 0, "exact-wholetext-edit-format")
        old, new = edit["before"].encode(), edit["after"].encode()
        need(value.count(old) == edit["count"], "whole-source-forward-occurrences")
        value = value.replace(old, new, edit["count"])
    need(value == after, "whole-source-forward-exact")
    for edit in reversed(record["edits"]):
        old, new = edit["after"].encode(), edit["before"].encode()
        need(value.count(old) == edit["count"], "whole-source-inverse-occurrences")
        value = value.replace(old, new, edit["count"])
    need(value == before, "whole-source-inverse-exact")

def package(pin):
    manifest = data(HERE / "source-manifest.json", pin)
    need(manifest["schema"] == "heptatorch.sum03.original76.recorded-source.v1"
         and manifest["status"] == "SOURCE_ONLY_NOT_EXECUTED",
         "frozen-new-source-not-runtime-result")
    rows = manifest["files"]
    need(type(rows) is dict and rows and manifest["payload_count"] == len(rows)
         and manifest["payload_bytes"] == sum(row["bytes"] for row in rows.values()),
         "complete-new-source-manifest")
    observed = set()
    for parent, dirs, files in os.walk(HERE, followlinks=False):
        for name in dirs:
            need(not (Path(parent)/name).is_symlink(), "no-source-directory-symlink")
        for name in files:
            path = Path(parent)/name
            rel = path.relative_to(HERE).as_posix()
            if rel == "source-manifest.json":
                continue
            observed.add(rel)
            need(rel in rows, "no-unmanifested-source-file", path=rel)
            pinned(safe_path(HERE, rel), rows[rel])
    need(observed == set(rows), "exact-complete-new-source-tree")
    config_raw = read(HERE / "configuration.json")
    config = json.loads(config_raw)
    need(config["schema"] == "heptatorch.sum03.original76.supplement-config.v1"
         and config["closed_schema_count"] == 0 and config["full_project_status"] == "HOLD",
         "finite-scope-configuration")
    inverse = data(HERE / "recorded-profile-inverse.json")
    framed_inverse(read(HERE / inverse["before_path"]),
                   read(HERE / inverse["after_path"]), inverse)
    need(len(inverse["edits"]) == 5
         and all(edit["count"] == 1 and "\n" not in edit["before"]
                 and "\n" not in edit["after"] for edit in inverse["edits"]),
         "only-five-single-line-recorded-profile-constants")
    pinverse = data(HERE / "profile-api-inverse.json")
    framed_inverse(read(HERE / pinverse["before_path"]),
                   read(HERE / pinverse["after_path"]), pinverse)
    einverse = data(HERE / "source02-entry-inverse.json")
    framed_inverse(read(HERE / einverse["before_path"]),
                   read(HERE / einverse["after_path"]), einverse)
    cinverse = data(HERE / "archive-custody-inverse.json")
    framed_inverse(read(HERE / cinverse["before_path"]),
                   read(HERE / cinverse["after_path"]), cinverse)
    original = load(HERE / "archive_custody_original.py", "sum76_original_archive",
                    rows["archive_custody_original.py"]["sha256"])
    custody = load(HERE / "archive_custody_sum76.py", "sum76_supplement_archive",
                   rows["archive_custody_sum76.py"]["sha256"])
    return config, manifest, SHA(config_raw), original, custody

def take_members(archive, member_manifest, requested):
    need(len(requested) == len(set(requested)) and set(requested) <= set(member_manifest["files"]),
         "explicit-unique-shared-members")
    result = {}
    with tarfile.open(archive, "r:xz") as tar:
        for member in tar:
            if member.name not in requested:
                continue
            row = member_manifest["files"][member.name]
            need(member.isfile() and member.name not in result
                 and member.size == row["bytes"] and member.mode == row["mode"],
                 "selected-shared-original-member")
            stream = tar.extractfile(member)
            need(stream is not None, "selected-shared-member-stream")
            content = stream.read()
            need(len(content) == row["bytes"] and SHA(content) == row["sha256"],
                 "selected-shared-member-all-bytes")
            result[member.name] = content
    need(set(result) == set(requested), "all-explicit-shared-members-read")
    return result

def shared_carrier(runs, config, original):
    pins = config["shared"]
    archive_path = safe_path(runs, pins["archive"]["relative_path"])
    manifest_path = safe_path(runs, pins["manifest"]["relative_path"])
    binding_path = safe_path(runs, pins["binding"]["relative_path"])
    member_manifest = json.loads(pinned(manifest_path, pins["manifest"]))
    binding = json.loads(pinned(binding_path, pins["binding"]))
    need(binding["schema"] == original.BINDING_SCHEMA and binding["status"] == "RAW_BYTES_SEALED"
         and binding["no_new_native_Torch_or_compiler_execution"] is True,
         "original-shared-carrier-custody-binding")
    for field, expected_path in (("archive", archive_path), ("member_manifest", manifest_path)):
        need(binding[field]["path"] == str(expected_path), "shared-original-canonical-path")
        same(metadata(binding[field]), metadata(pins["archive" if field == "archive" else "manifest"]),
             "shared-original-custody-metadata")
    same(binding["input_bindings"], member_manifest["input_bindings"], "complete315-original-ledger")
    need(len(binding["input_bindings"]) == 315
         and binding["profile_sha256"] == member_manifest["source_profile_sha256"]
         and binding["native_report"]["sha256"] == config["pure32"]["reconstructed"]["sha256"]
         and binding["native_status"] == member_manifest["native_original_status"] == "PASS",
         "shared-native32-original-profile-and-report")
    archive_result = original.verify_archive(archive_path, member_manifest, pins["archive"])
    rolemap = json.loads(read(HERE / pins["role_map"]["path"],
                             pins["role_map"]["sha256"], pins["role_map"]["bytes"]))
    need(len(rolemap["input_roles"]) == 315 and len(rolemap["additional_roles"]) == 22,
         "complete-role-map-denominators")
    members = member_manifest["files"]
    observed = set()
    for role in rolemap["input_roles"] + rolemap["additional_roles"]:
        name = role["archive_member"]
        need(name not in observed and name in members, "unique-existing-role-member")
        observed.add(name)
        entry = members[name]
        same(entry, {"source_path": role["source_path"], "bytes": role["bytes"],
                     "sha256": role["sha256"], "mode": role["original_mode"]},
             "role-map-real-original-member-metadata")
    same(sorted(observed), sorted(name for name in members if not name.startswith("native-output/")),
         "all-shared-non-native-members-have-explicit-roles")
    selected = take_members(archive_path, member_manifest,
                            [pins["build_member"], pins["runtime_member"],
                             config["owner"]["archive_member"]])
    build_raw = selected[pins["build_member"]]
    need(SHA(build_raw) == config["profile"]["build_sha256"], "actual696-build-report")
    build = json.loads(build_raw)
    required = {"schema": config["profile"]["build_schema"], "source_revision": "04",
                "status": "PASS_NATIVE_BUILD_ONLY", "functions": 696, "blocks": 9216,
                "instructions": 32990, "units_planned": 14, "units_compiled": 14,
                "objects_reused": False, "all_inputs_unchanged": True,
                "link_exit": 0, "link_stderr_bytes": 0, "result_executable_executed": False,
                "torch_executed": False, "runtime_qualified": False, "schema_closure": False}
    for key, wanted in required.items():
        same(build[key], wanted, "fixed-actual-build/" + key)
    for key in ("source_sha256", "native_sha256", "descriptor_bytes", "state_format", "backing_format"):
        same(build[key], config["profile"][key], "same696-image-profile/" + key)
    need(SHA(selected[pins["runtime_member"]]) == config["profile"]["source_sha256"]
         and members[pins["image_member"]]["sha256"] == config["profile"]["native_sha256"]
         and members[pins["image_member"]]["bytes"] == build["native_bytes"],
         "source-and-ELF-actually-in-shared-carrier")
    checked = build["checked_inputs"]
    need(len(checked) == 177, "whole-original-cold-build-input-denominator")
    for path, expected in checked.items():
        need(path in binding["input_bindings"], "every-cold-input-in315", path=path)
        same(metadata(binding["input_bindings"][path]), metadata(expected),
             "whole-cold-input-byte-mode-binding")
    need(len(build["units"]) == 14 and len(build["processes"]) == 15,
         "all-original-native-units-and-link-process")
    origin = Path(members[pins["build_member"]]["source_path"]).parent
    by_source = {entry["source_path"]: entry for entry in members.values()}
    for unit in build["units"]:
        source = by_source[str(origin / (unit["name"] + ".hepta"))]
        obj = by_source[str(origin / (unit["name"] + ".o"))]
        need(source["sha256"] == unit["source_sha256"]
             and obj["sha256"] == unit["object_sha256"] and obj["bytes"] == unit["object_bytes"]
             and unit["compiler_exit"] == unit["compiler_stderr_bytes"] == 0,
             "every-actual-source-object-stays-exact", unit=unit["name"])
    for process in build["processes"]:
        need(process["started"] is True and process["completed"] is True
             and process["stdout_EOF"] is True and process["stderr_EOF"] is True
             and process["returncode"] == 0 and process["stderr_bytes"] == 0
             and process["cleanup_errors"] == [] and process["evidence_write_errors"] == [],
             "original-compile-process-complete-EOF")
        for stem in ("stdout", "stderr"):
            entry = by_source[str(origin / process[stem + "_path"])]
            need(entry["sha256"] == process[stem + "_sha256"]
                 and entry["bytes"] == process[stem + "_bytes"],
                 "compile-original-stream-bytes")
    report = {"archive": {"path": str(archive_path), **pins["archive"]},
              "manifest": {"path": str(manifest_path), **metadata(pins["manifest"])},
              "binding": {"path": str(binding_path), **metadata(pins["binding"])},
              "role_map_sha256": pins["role_map"]["sha256"],
              "input_count": 315, "additional_count": 22, "cold_input_count": 177,
              "native_units": 14, "compile_processes": 15,
              "archive_verification": archive_result,
              "archive_copied_or_recompressed": False,
              "runtime_semantics_reexecuted": False}
    return member_manifest, selected, report

def input_copies(base, aggregate, expected):
    custody_raw = pinned(base / "input-custody.json", expected)
    custody = json.loads(custody_raw)
    same(sorted(custody), sorted(aggregate["inputs_sha256"]), "complete-original-suite-input-custody")
    copies = {}
    for path, row in custody.items():
        need(set(row) == {"copy", "bytes", "sha256"}, "exact-original-copy-record")
        copy = safe_path(base, row["copy"])
        need(row["copy"].startswith("inputs/") and row["copy"] not in copies,
             "unique-source-copy-member")
        content = read(copy, row["sha256"], row["bytes"])
        need(SHA(content) == aggregate["inputs_sha256"][path], "original-suite-input-byte-identity")
        # No mode is invented: input-custody originally records only these three
        # fields. The supplement inventories each actual copy's original mode.
        copies[row["copy"]] = {"original_path": path, **BOUND[str(copy)]}
    actual = set()
    for parent, dirs, files in os.walk(base / "inputs", followlinks=False):
        for name in dirs:
            need(not (Path(parent)/name).is_symlink(), "input-copy-no-directory-symlink")
        for name in files:
            actual.add((Path(parent)/name).relative_to(base).as_posix())
    same(sorted(actual), sorted(copies), "all-and-only-original-suite-input-copies")
    return custody, copies

def suite_evidence(runs, config):
    result = {}
    totals = {key: 0 for key in ("cases", "assertions", "exact_words", "full_rng_bytes", "rollback_bytes", "frames")}
    for suite in config["suites"]:
        base = safe_path(runs, suite["root"])
        raw = pinned(base / "report.json", suite["report"])
        aggregate = json.loads(raw)
        required = {"status": "PASS", "cases": suite["cases"], "passed_cases": suite["cases"],
                    "failed_cases": 0, "source_sha256": config["profile"]["source_sha256"],
                    "native_sha256": config["profile"]["native_sha256"],
                    "observer_sha256": config["modules"]["observer"]["sha256"],
                    "functions": 696, "native_units": 14, "descriptor_bytes": 672,
                    "all_inputs_unchanged": True, "native_executed": True,
                    "torch_executed_here": False, "full_pytorch_rewrite_complete": False,
                    "closed_schema_count": 0, "pytorch_api_denominator": 2590}
        for key, wanted in required.items():
            same(aggregate[key], wanted, "original-suite-header/" + suite["name"] + "/" + key)
        need(aggregate.get("all_sessions_eof_exit_zero",
                           aggregate.get("all_executed_sessions_eof_exit_zero")) is True,
             "original-all-session-EOF0")
        if suite["family"] == "26":
            need(aggregate["blocked_cases"] == 0 and aggregate["all_reference_cases_executed"] is True
                 and aggregate["selected_case_indices"] == list(range(26)),
                 "original-all26-no-filter-no-blocker")
        custody, copies = input_copies(base, aggregate, suite["input_custody"])
        reference_paths = {}
        for label in ("reference", "probe"):
            expected = suite[label]
            path = expected["original_path"]
            need(path in custody and custody[path]["sha256"] == expected["sha256"]
                 and custody[path]["bytes"] == expected["bytes"]
                 and aggregate[label + "_sha256"] == expected["sha256"],
                 "exact-original-reference-input-not-model", label=label)
            reference_paths[label] = safe_path(base, custody[path]["copy"])
        need(len(aggregate["rows"]) == suite["cases"]
             and [row["index"] for row in aggregate["rows"]] == list(range(suite["cases"])),
             "original-case-denominator-order")
        for key in ("assertions", "exact_words", "full_rng_bytes", "rollback_bytes", "frames"):
            same(sum(row[key] for row in aggregate["rows"]), aggregate[key],
                 "actual-original-aggregate/" + key)
            totals[key] += aggregate[key]
        totals["cases"] += suite["cases"]
        result[suite["name"]] = {"spec": suite, "base": base, "aggregate": aggregate,
                                "custody": custody, "copies": copies, **reference_paths}
    wanted = config["expected_totals"]
    same(totals, {"cases": wanted["cases"], "assertions": wanted["original_assertions"],
                  "exact_words": wanted["exact_words"], "full_rng_bytes": wanted["full_rng_bytes"],
                  "rollback_bytes": wanted["source_bound_refusal_bytes"], "frames": wanted["frames"]},
         "complete-original76-original-observation-totals")
    return result, totals

def pure32_evidence(runs, config):
    base = safe_path(runs, config["pure_root"])
    original = json.loads(pinned(base / "report.json", config["pure32"]["report"]))
    reconstructed = digest_file(base / "reconstructed-original-report.json",
                                config["pure32"]["reconstructed"]["sha256"],
                                config["pure32"]["reconstructed"]["bytes"],
                                config["pure32"]["reconstructed"]["mode"])
    need(original["schema"] == "heptatorch.sum-native32.complete-recorded-admission.v1"
         and original["status"] == config["pure32"]["status"]
         and original["native_executed"] is False and original["Torch_executed"] is False
         and original["compiler_executed"] is False
         and original["case_count"] == 32 and original["original_event_count"] == 626
         and len(original["cases"]) == 32,
         "existing-original-recorded32-complete-result")
    need([row["index"] for row in original["cases"]] == list(range(32))
         and all(row["status"] == "RECOMPUTED" and row["original_native_status"] == "PASS"
                 for row in original["cases"]), "existing32-all-original-recomputed-results")
    return {"report": {"path": str(base / "report.json"), **config["pure32"]["report"]},
            "reconstructed": {"path": str(base / "reconstructed-original-report.json"), **reconstructed},
            "original_case_count": 32, "original_event_count": 626,
            "new_recorded32_execution": False, "qualification_transferred_to76": False}

def unwrap_tool(response):
    if response.get("structuredContent") is not None:
        return response["structuredContent"]
    text = [row["text"] for row in response["content"] if row["type"] == "text"]
    need(len(text) == 1, "one-original-tool-JSON-result")
    return json.loads(text[0])

def operator_receipts(runs, config, suites):
    pin = config["operator_receipt"]
    path = safe_path(runs, pin["relative_path"])
    original = json.loads(pinned(path, pin))
    need(original["schema"] == "heptatorch.root.sum03-original76-and-recorded32-operator-receipts.v1"
         and original["new_processes_started"] == 0,
         "original-status-log-recovery-not-new-execution")
    rows = original["original76"]
    need([row["name"] for row in rows] == [s["name"] for s in config["suites"]],
         "operator-four-suite-order")
    results = []
    for row in rows:
        request = row["exact_operator_request"]
        first = unwrap_tool(row["original_first_start_tool_result"])
        terminal = unwrap_tool(row["fresh_read_original_terminal_tool_result"])
        logs = unwrap_tool(row["fresh_read_complete_original_log_tool_result"])
        job = first["job"]
        need(first["reused"] is False and first["launch"]["ok"] is True
             and job["id"] == terminal["job_id"] == logs["job_id"] == row["job_id"],
             "original-start-terminal-log-same-actual-job")
        for field in ("host", "executable", "args", "cwd", "idempotency_key"):
            same(job[field], request[field], "original-request-exact-job/" + field)
        need(request["host"] == "rog" and request["executable"] == "/usr/bin/python3"
             and terminal["state"] == "succeeded" and terminal["exit_code"] == 0
             and terminal["automatic_retry"] is False and terminal["cancelled"] is False
             and terminal["timed_out"] is False and terminal["output_complete"] is True
             and terminal["output_durable"] is True
             and terminal["terminal_publication_confirmed"] is True,
             "authoritative-original-success-not-report-inference")
        for stream in ("stdout", "stderr"):
            observed = logs[stream]
            raw = bytes.fromhex(observed["bytes_hex"])
            need(observed["eof"] is True and observed["offset"] == 0
                 and observed["next_offset"] == observed["size"] == len(raw)
                 and raw.decode("utf-8") == observed["text"]
                 and terminal["log_bytes"][stream]["seen"] == len(raw)
                 and terminal["log_bytes"][stream]["stored"] == len(raw)
                 and terminal["log_synced"][stream] is True
                 and terminal["log_truncated"][stream] is False
                 and terminal["log_errors"][stream] is None,
                 "whole-original-operator-stream-and-EOF", stream=stream)
            if stream == "stderr":
                need(raw == b"", "original-operator-stderr-empty")
        argv = request["args"]
        prefix = ["-B"]
        same(argv[:len(prefix)], prefix, "original-interpreter-arguments")
        script = argv[len(prefix)]
        options = argv[len(prefix)+1:]
        need(len(options) % 2 == 0 and all(options[i].startswith("--")
             for i in range(0, len(options), 2)), "original-paired-explicit-CLI")
        opts = dict(zip(options[::2], options[1::2]))
        need(len(opts) * 2 == len(options), "no-duplicate-original-CLI-option")
        suite = suites[row["name"]]
        spec = suite["spec"]
        wanted_options = {
            "--binary": str(Path(config["original_runs_root"]) / "sum-dims-f32-native-build-source04-01/candidate-sum-dims/tensor-sum-dims-session"),
            "--source": str(Path(config["original_runs_root"]) / "sum-dims-f32-native-build-source04-01/candidate-sum-dims/runtime.hepta"),
            "--build-report": str(Path(config["original_runs_root"]) / "sum-dims-f32-native-build-source04-01/candidate-sum-dims/build.json"),
            "--protocol": config["modules"]["protocol"]["original_path"],
            "--observer": config["modules"]["observer"]["original_path"],
            "--reference": spec["reference"]["original_path"],
            "--probe": spec["probe"]["original_path"],
            "--expected-source-sha256": config["profile"]["source_sha256"],
            "--expected-native-sha256": config["profile"]["native_sha256"],
            "--expected-observer-sha256": config["modules"]["observer"]["sha256"],
            "--expected-functions": "696", "--expected-native-units": "14",
            "--output": str(suite["base"]),
        }
        if spec["family"] != "14":
            wanted_options["--support-reader"] = config["modules"]["support"]["original_path"]
        if spec["family"] in {"26", "4"}:
            wanted_options["--expected-reference-sha256"] = spec["reference"]["sha256"]
        if spec["family"] == "26":
            wanted_options["--identity-adapter"] = config["modules"]["identity_epochs"]["original_path"]
            wanted_options["--parent-reader"] = config["modules"]["parent26"]["original_path"]
        same(opts, wanted_options, "entire-original-native-command-identity")
        need(script in suite["custody"]
             and suite["custody"][script]["sha256"] == spec["reader_sha256"],
             "executed-reader-is-original-frozen-reader")
        same(row["original_report"], {"path": str(suite["base"] / "report.json"), **spec["report"]},
             "operator-receipt-binds-whole-native-report")
        results.append({"name": row["name"], "job_id": row["job_id"], "exit_code": 0,
                        "full_stdout_bytes": logs["stdout"]["size"], "full_stderr_bytes": 0,
                        "both_EOF": True, "original_request_exact": True})
    return {"path": str(path), **metadata(pin)}, results

# The pure32 flag-only CLI differs from all native readers. This helper validates
# it independently without changing or reconstructing its saved request bytes.
def pure_operator_receipt(runs, config):
    pin = config["operator_receipt"]
    original = json.loads(pinned(safe_path(runs, pin["relative_path"]), pin))
    row = original["recorded32"]
    request = row["exact_operator_request"]
    first = unwrap_tool(row["original_first_start_tool_result"])
    terminal = unwrap_tool(row["original_terminal_tool_result"])
    logs = unwrap_tool(row["original_complete_log_tool_result"])
    need(first["reused"] is False and first["launch"]["ok"] is True
         and first["job"]["id"] == terminal["job_id"] == logs["job_id"] == row["job_id"],
         "original-pure32-job-identity")
    for key in ("host", "executable", "args", "cwd", "idempotency_key"):
        same(first["job"][key], request[key], "pure32-exact-original-request")
    wanted = ["-I", "-S", "-B",
              str(runs / "sum-native32-pure-admission-source-02/admit_recorded_sum32.py"),
              "--package-sha256", "a0968c1dfdf00d9f5a2c67d181d15dc9b9b3ea6bfd671ac1c672dd2badcd2dd3",
              "--evidence-binding", str(runs / config["shared"]["binding"]["relative_path"]),
              "--evidence-binding-sha256", config["shared"]["binding"]["sha256"],
              "--output", str(runs / config["pure_root"]), "--admit-recorded"]
    same(request["args"], wanted, "entire-original-pure32-argv")
    need(request["host"] == "rog" and request["executable"] == "/usr/bin/python3"
         and terminal["state"] == "succeeded" and terminal["exit_code"] == 0
         and terminal["automatic_retry"] is False and terminal["cancelled"] is False
         and terminal["timed_out"] is False and terminal["output_complete"] is True
         and terminal["output_durable"] is True
         and terminal["terminal_publication_confirmed"] is True,
         "original-pure32-authoritative-terminal")
    for stream in ("stdout", "stderr"):
        observed = logs[stream]
        raw = bytes.fromhex(observed["bytes_hex"])
        need(observed["offset"] == 0 and observed["eof"] is True
             and observed["size"] == observed["next_offset"] == len(raw)
             and raw.decode("utf-8") == observed["text"]
             and terminal["log_bytes"][stream]["seen"] == len(raw)
             and terminal["log_bytes"][stream]["stored"] == len(raw)
             and terminal["log_synced"][stream] is True
             and terminal["log_truncated"][stream] is False
             and terminal["log_errors"][stream] is None,
             "pure32-original-full-stream-EOF")
    need(logs["stderr"]["size"] == 0, "pure32-original-empty-stderr")
    same(row["report"], {"path": str(runs / config["pure_root"] / "report.json"),
                        **config["pure32"]["report"]}, "pure32-original-whole-report-binding")
    return {"name": row["name"], "job_id": row["job_id"], "exit_code": 0,
            "both_EOF": True, "full_stdout_bytes": logs["stdout"]["size"],
            "full_stderr_bytes": logs["stderr"]["size"], "original_request_exact": True}


def new_manifest(runs, output, config, package_manifest, config_pin, custody,
                 native_totals, pure_evidence, receipt, shared):
    roots = (("native-output", safe_path(runs, config["native_root"])),
             ("pure32-output", safe_path(runs, config["pure_root"])))
    files, directories = {}, {"inputs": 0o700, "execution": 0o700}
    for prefix, root in roots:
        observed_files, observed_dirs = custody.inventory(root)
        for rel, row in observed_files.items():
            files[prefix + "/" + rel] = row
        for rel, mode in observed_dirs.items():
            directories[prefix + ("/" + rel if rel else "")] = mode
    inputs = {}
    for rel, pin in package_manifest["files"].items():
        path = safe_path(HERE, rel)
        inputs[str(path)] = {"path": str(path), **digest_file(path, pin["sha256"], pin["bytes"], pin["mode"])}
    self_manifest = HERE / "source-manifest.json"
    inputs[str(self_manifest)] = {"path": str(self_manifest), **digest_file(self_manifest)}
    for row in config["additional_input_sources"]:
        path = Path(row["path"])
        need(str(path) not in inputs, "unique-additional-original-source")
        inputs[str(path)] = {"path": str(path), **digest_file(path, row["sha256"], row["bytes"], row["mode"])}
    for index, (path, row) in enumerate(sorted(inputs.items())):
        name = "inputs/%04d-%s" % (index, Path(path).name)
        files[name] = {"source_path": path, **metadata(row)}
    receipts = [receipt]
    for index, row in enumerate(receipts):
        name = "execution/%04d-%s" % (index, Path(row["path"]).name)
        files[name] = {"source_path": row["path"], **metadata(row)}
    plan = {"schema": "heptatorch.sum03.original76.actual-seal-plan.v1",
            "source_package_sha256": SHA(read(self_manifest)), "configuration_sha256": config_pin,
            "original_runs_root": str(runs), "native_root": str(roots[0][1]),
            "pure_root": str(roots[1][1]), "shared_carrier": shared,
            "original76_totals": native_totals, "pure32_original": pure_evidence,
            "operator_receipt": receipt, "additional_source_count": len(config["additional_input_sources"]),
            "complete_roots_no_file_filters": True, "new_native_Torch_compiler_executions": 0,
            "new_recorded_comparisons": 0, "original_statuses_not_rewritten": True}
    plan_path = output / "seal-plan.json"
    plan_meta = write(plan_path, plan)
    files["execution/seal-plan.json"] = {"source_path": str(plan_path), **plan_meta}
    return {
        "schema": custody.MEMBER_SCHEMA, "native_output": str(roots[0][1]),
        "pure_output": str(roots[1][1]),
        "native_report_pins": {s["name"]: {"path": str(safe_path(runs, s["root"]) / "report.json"),
                                         **s["report"]} for s in config["suites"]},
        "native_original_statuses": {s["name"]: "PASS" for s in config["suites"]},
        "pure_report_sha256": config["pure32"]["report"]["sha256"],
        "files": files, "directories": directories,
        "synthetic_container_directories": {"inputs": 0o700, "execution": 0o700},
        "file_count": len(files), "directory_count": len(directories),
        "payload_bytes": sum(row["bytes"] for row in files.values()),
        "input_bindings": inputs, "execution_evidence": receipts,
        "source_profile_sha256": config_pin,
        "mode_policy": "Every original file and directory mode retained; two explicit synthetic containers only.",
        "manifest_is_outside_archive": True,
        "native_Torch_compiler_or_recorded_admission_executed": False,
        "shared_carrier": shared,
    }

def seal(runs, output, config, package_manifest, config_pin, custody, totals,
         pure_evidence, receipt, operator_results, shared):
    manifest = new_manifest(runs, output, config, package_manifest, config_pin, custody,
                            totals, pure_evidence, receipt, shared)
    manifest_path = output / "file-manifest.json"
    manifest_meta = write(manifest_path, manifest)
    archive_path = output / "complete.tar.xz"
    archive_meta = custody.create_archive(archive_path, manifest)
    archived = custody.verify_archive(archive_path, manifest, archive_meta)
    originals = custody.verify_originals(manifest)
    errors = recheck_bound()
    need(not errors, "all-sealed-original-inputs-unchanged", errors=errors)
    report = {
        "schema": "heptatorch.sum03.original76.supplement-seal.v1", "status": "PASS_CUSTODY_ONLY",
        "source_package_sha256": SHA(read(HERE / "source-manifest.json")),
        "configuration_sha256": config_pin,
        "archive": {"path": str(archive_path), **archive_meta},
        "file_manifest": {"path": str(manifest_path), **manifest_meta},
        "shared_carrier": shared, "original76_totals": totals,
        "native_report_pins": manifest["native_report_pins"],
        "native_original_statuses": manifest["native_original_statuses"],
        "pure32_original": pure_evidence, "operator_receipt": receipt,
        "operator_jobs": operator_results, "archive_verification": archived,
        "originals_verification": originals, "all_original_files_unchanged": True,
        "input_bindings": dict(BOUND), "native_admission_performed": False,
        "recorded_comparison_performed": False, "new_native_executions": 0,
        "new_Torch_executions": 0, "new_compiler_executions": 0,
        "whole_pytorch_gate": "HOLD", "closed_schema_count": 0,
        "audit": {**AUDIT, **EXTRA_AUDIT},
    }
    report_path = output / "report.json"
    report_meta = write(report_path, report)
    binding = {
        "schema": custody.BINDING_SCHEMA, "status": "RAW_BYTES_SEALED",
        "source_package_sha256": report["source_package_sha256"], "configuration_sha256": config_pin,
        "member_manifest": report["file_manifest"], "archive": report["archive"],
        "shared_carrier": shared, "native_output": manifest["native_output"],
        "pure_output": manifest["pure_output"], "native_report_pins": manifest["native_report_pins"],
        "native_original_statuses": manifest["native_original_statuses"],
        "original76_totals": totals, "pure32_original": pure_evidence,
        "execution_evidence": manifest["execution_evidence"],
        "input_bindings": manifest["input_bindings"],
        "sealing_report": {"path": str(report_path), **report_meta},
        "new_native_Torch_compiler_or_recorded_comparison_executions": 0,
    }
    write(output / "evidence-binding.json", binding)
    return report

def verify_supplement(binding_path, binding_pin, config, config_pin, custody, shared,
                      native_totals, pure_evidence, receipt):
    binding = data(binding_path, binding_pin)
    need(set(binding) == {"schema", "status", "source_package_sha256", "configuration_sha256",
         "member_manifest", "archive", "shared_carrier", "native_output", "pure_output",
         "native_report_pins", "native_original_statuses", "original76_totals",
         "pure32_original", "execution_evidence", "input_bindings", "sealing_report",
         "new_native_Torch_compiler_or_recorded_comparison_executions"},
         "exact-supplement-binding-fields")
    need(binding["schema"] == custody.BINDING_SCHEMA and binding["status"] == "RAW_BYTES_SEALED"
         and binding["source_package_sha256"] == SHA(read(HERE / "source-manifest.json"))
         and binding["configuration_sha256"] == config_pin
         and binding["new_native_Torch_compiler_or_recorded_comparison_executions"] == 0,
         "this-exact-source-and-custody-only-binding")
    row = binding["member_manifest"]
    manifest = json.loads(pinned(row["path"], row))
    need(binding["native_output"] == manifest["native_output"]
         == str(Path(config["original_runs_root"]) / config["native_root"])
         and binding["pure_output"] == manifest["pure_output"]
         == str(Path(config["original_runs_root"]) / config["pure_root"])
         and manifest["source_profile_sha256"] == config_pin
         and manifest["pure_report_sha256"] == config["pure32"]["report"]["sha256"],
         "whole-supplement-original-roots-and-pure32")
    for key in ("native_report_pins", "native_original_statuses", "input_bindings", "execution_evidence"):
        same(binding[key], manifest[key], "whole-supplement-binding/" + key)
    same(binding["shared_carrier"], shared, "exact-verified-shared-carrier")
    same(manifest["shared_carrier"], shared, "manifest-exact-shared-carrier")
    same(binding["original76_totals"], native_totals, "actual-original76-counts")
    same(binding["pure32_original"], pure_evidence, "whole-existing-recorded32-result")
    same(binding["execution_evidence"], [receipt], "original-operator-receipt-file")
    expected_reports = {row["name"]: {"path": str(Path(config["original_runs_root"]) / row["root"] / "report.json"),
                                     **row["report"]} for row in config["suites"]}
    same(binding["native_report_pins"], expected_reports, "all-four-fixed-original-report-pins")
    same(binding["native_original_statuses"], {row["name"]: "PASS" for row in config["suites"]},
         "all-four-original-statuses-preserved")
    archive = binding["archive"]
    checked = custody.verify_archive(archive["path"], manifest, archive)
    originals = custody.verify_originals(manifest)
    sealrow = binding["sealing_report"]
    report = json.loads(pinned(sealrow["path"], sealrow))
    need(report["status"] == "PASS_CUSTODY_ONLY"
         and report["native_admission_performed"] is False
         and report["recorded_comparison_performed"] is False
         and report["all_original_files_unchanged"] is True,
         "actual-sealing-receipt-only-claims-custody")
    for key in ("archive", "shared_carrier", "native_report_pins", "native_original_statuses",
                "original76_totals", "pure32_original"):
        same(report[key], binding[key], "sealer-whole-binding/" + key)
    same(report["file_manifest"], binding["member_manifest"], "sealer-manifest-byte-binding")
    return binding, manifest, {"archive": checked, "originals": originals}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seal", action="store_true")
    mode.add_argument("--admit-recorded", action="store_true")
    parser.add_argument("--evidence-binding", type=Path)
    parser.add_argument("--expected-binding-sha256")
    args = parser.parse_args()
    runs, output = args.runs_root.absolute(), args.output.absolute()
    need(runs.resolve() == runs and output.resolve() == output and runs.is_dir(),
         "canonical-real-input-output-paths")
    need(not output.exists() and output != HERE and not output.is_relative_to(HERE)
         and not HERE.is_relative_to(output), "new-disjoint-output-and-frozen-source")
    # Current source is for the already recorded ROG roots. A future portable
    # relocation must supply a separate explicit mapping and inverse.
    for protected in (runs / "root-sum-source03-original76-actual-01",
                      runs / "root-sum-native32-source03-recorded-admission-01",
                      runs / "root-sum-native32-source03-raw-seal-01"):
        need(not output.is_relative_to(protected) and not protected.is_relative_to(output),
             "output-disjoint-from-all-original-carriers")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.umask(0o077)
    report = None
    try:
        config, source_manifest, config_pin, original, custody = package(args.package_sha256)
        need(str(runs) == config["original_runs_root"], "fixed-original-run-root-no-implicit-remap")
        shared_manifest, selected, shared = shared_carrier(runs, config, original)
        suites, totals = suite_evidence(runs, config)
        pure = pure32_evidence(runs, config)
        receipt, jobs = operator_receipts(runs, config, suites)
        jobs.append(pure_operator_receipt(runs, config))
        need(not any(AUDIT.values()) and not any(EXTRA_AUDIT.values()), "zero-forbidden-execution-attempts")
        if args.seal:
            need(args.evidence_binding is None and args.expected_binding_sha256 is None,
                 "seal-has-no-preexisting-binding")
            report = seal(runs, output, config, source_manifest, config_pin, custody,
                          totals, pure, receipt, jobs, shared)
        else:
            need(args.evidence_binding is not None and args.expected_binding_sha256 is not None,
                 "recorded-admission-requires-completed-supplement-binding")
            binding, member_manifest, custody_result = verify_supplement(
                args.evidence_binding, args.expected_binding_sha256, config, config_pin,
                custody, shared, totals, pure, receipt)
            adapter = load(HERE / "recorded_admission.py", "sum76_recorded_adapter",
                           source_manifest["files"]["recorded_admission.py"]["sha256"])
            compared = adapter.admit(config, suites, selected, output)
            errors = recheck_bound()
            need(not errors, "all-raw-admission-inputs-unchanged", errors=errors)
            need(not any(AUDIT.values()) and not any(EXTRA_AUDIT.values()),
                 "zero-execution-attempts-through-complete-admission")
            report = {
                "schema": "heptatorch.sum03.original76.complete-recorded-admission.v1",
                "status": "PASS_PURE_RECORDED_76", "configuration_sha256": config_pin,
                "source_package_sha256": args.package_sha256, "source_profile": config["profile"],
                "supplement_binding": {"path": str(args.evidence_binding), **digest_file(args.evidence_binding)},
                "shared_carrier": shared, "custody": custody_result,
                "operator_receipt": receipt, "operator_jobs": jobs,
                "original76_totals": totals, "recorded76": compared,
                "existing_recorded32_preserved": pure, "new_recorded32_execution": False,
                "all_inputs_unchanged": True, "input_bindings": dict(BOUND),
                "native_executed": False, "Torch_executed": False, "compiler_executed": False,
                "closed_schema_count": 0, "whole_pytorch_gate": "HOLD",
                "audit": {**AUDIT, **EXTRA_AUDIT},
            }
            write(output / "report.json", report)
        print(json.dumps({"status": report["status"], "output": str(output),
                          "native_Torch_compiler_executions": 0}, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {"schema": "heptatorch.sum03.original76.supplement-attempt.v1",
                   "status": "FAIL", "operation": "seal" if args.seal else "admit-recorded",
                   "error": repr(error), "traceback": traceback.format_exc(),
                   "input_bindings": dict(BOUND), "audit": {**AUDIT, **EXTRA_AUDIT},
                   "native_executed": False, "Torch_executed": False, "compiler_executed": False,
                   "prior_original_reports_modified": False,
                   "closed_schema_count": 0, "whole_pytorch_gate": "HOLD"}
        # A final binding-write error must not overwrite a completed custody-only
        # report; its enclosing attempt is still an explicit failure.
        destination = output / ("attempt-failure.json" if (output / "report.json").exists() else "report.json")
        write(destination, failure)
        print(json.dumps({"status": "FAIL", "error": repr(error), "output": str(output)}, sort_keys=True))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
