#!/usr/bin/env python3
"""Verify exact frozen Current HIA Runtime evidence; never executes archive content."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

HEAD = "e9444902df9878060ab389a70ee957c1b7f4a887"
ZIP = (645877944, "af19c74130dd3ab06d251324eecb50fd553959c81efc860fbf160f99c1b432da")
CI = (76650, "a92cf94efaa4745168d4ab82239b1af60b4819942e6f89c82a794777004a9902")
CUSTODY = (103458, "0ea8a3a22f7d4c2541590f18f98b3631d51603c70aa79ef8330714bc676172a9")
ARCHIVE = (618919504, "c6b27d1917e7e9f44882c3168ced4eb0f6c33e22b197802967f2a2be1ddc9b86")
Q = (59884812, "016d3436d05ac12691f39f9b50f82712cde3b9484b92128ee46858534fd4864e")
BUILD = "c1b4cb1c483705d935b453f3b9ab1f82b20b59ea658048bc5750cc33bf9b0d33"
MANIFEST = "348ed8e949256f985cdca0d9c612dc646ae738beee6013c27ab405d2df3e67c8"
SOURCE = "0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43"
ELF = "ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e"
SUITES = {
    "bool27": 27, "original76": 76, "integral24": 24, "sum_f32_17": 17,
    "current35": 35, "base270": 270, "sum32": 32, "live32": 32, "logical147": 147,
    "resource7": 7, "asstrided16": 16, "cat4": 4, "metadata10": 10,
    "generator131": 131, "public_f64_logical192": 192, "f64_48": 48, "half47": 47,
    "half_logical264": 264, "half_transactions12": 12, "numeric_predicates": 485,
    "bf16_copy": 280, "half_mul": 312, "half_neg": 371, "half_addsub": 1006,
    "wide_neg": 405, "half_div": 1196, "public_f64_arithmetic": 2541,
    "same_f64_arithmetic": 5088, "typed_relu": 776,
}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def record(value, expected, label):
    for key, item in expected.items():
        need(value.get(key) == item, f"{label} {key} differs")


def pin(raw, fingerprint, label):
    need((len(raw), hashlib.sha256(raw).hexdigest()) == fingerprint, f"{label} byte pin differs")
    return json.loads(raw)


def check(path):
    need(path.stat().st_size == ZIP[0], "wrong official original ZIP length")
    with path.open("rb") as stream:
        need(hashlib.file_digest(stream, "sha256").hexdigest() == ZIP[1], "wrong official original ZIP digest")
    selected = {}
    with zipfile.ZipFile(path) as z:
        need(len(z.infolist()) == len(set(z.namelist())) == 8, "wrong ZIP member inventory")
        need(z.testzip() is None, "ZIP member CRC failure")
        ci = pin(z.read("heptatorch-current/qualification.json"), CI, "CI summary")
        custody = pin(z.read("heptatorch-current-evidence.custody.json"), CUSTODY, "CI custody")
        record(ci, {"status": "PASS_CURRENT_RUNTIME_FOCUSED", "commit": HEAD, "failures": [],
                    "full_project_status": "HOLD", "full_pytorch_rewrite_complete": False}, "CI")
        record(custody, {
            "source_commit": HEAD, "status": "PASS", "original_qualification_status": ci["status"],
            "complete_single_XZ_EOF": True, "complete_TAR_zero_padding": True, "all_selected_files_unchanged": True,
            "all_original_member_modes_preserved": True, "errors": [], "missing": [],
            "member_count": 277502, "member_bytes": 3003744583, "declared_input_count": 231793,
            "whole_pytorch_gate": "HOLD", "full_pytorch_rewrite_complete": False,
        }, "custody")
        record(custody["archive"], dict(zip(("bytes", "sha256"), ARCHIVE, strict=True)), "inner archive")
        record(ci["qualification"], {"bytes": Q[0], "sha256": Q[1], "path": "review/qualification.json"},
               "actual qualification binding")
        paths = {
            "current-runtime/review/qualification.json": Q[1],
            "current-runtime/build/build.json": BUILD,
            "current-runtime/build/manifest.json": MANIFEST,
            "current-runtime/build/heptatorch": ELF,
            "current-runtime/build/runtime.hepta": SOURCE,
        }
        with z.open("heptatorch-current-evidence.tar.xz") as inner, tarfile.open(fileobj=inner, mode="r|xz") as t:
            for member in t:
                if member.name not in paths:
                    continue
                need(member.name not in selected and member.isfile() and 0 < member.size <= Q[0],
                     "unsafe or duplicate selected member")
                with t.extractfile(member) as stream:
                    data = stream.read()
                need(hashlib.sha256(data).hexdigest() == paths[member.name], f"wrong native member: {member.name}")
                if member.name.endswith("/qualification.json"):
                    q = pin(data, Q, "actual qualification")
                selected[member.name] = paths[member.name]
                if set(selected) == set(paths):
                    break
    need(set(selected) == set(paths), "required native member missing")
    record(q, {"status": "PASS_CURRENT_RUNTIME_FOCUSED", "all_suites_executed": True,
               "all_inputs_unchanged": True, "Torch_executed": True, "failures": [], "input_recheck_errors": [],
               "native_sha256": ELF, "source_sha256": SOURCE, "actual_build_sha256": BUILD, "functions": 775,
               "units": 15, "required_cases_total": 13851, "executed_cases_total": 13851,
               "required_cases": SUITES, "required_suites": list(SUITES), "schema_closure": False,
               "full_project_status": "HOLD", "full_pytorch_rewrite_complete": False,
               "generator_subgroups": {"dispatch37": 37, "entry59": 59, "native11": 11, "oracle24": 24}}, "actual")
    need([(x["name"], x["cases"], x["status"], x["qualified"]) for x in q["suites"]]
         == [(name, count, "PASS", True) for name, count in SUITES.items()], "suite order/count/status differs")
    return {
        "schema": "heptatorch.current-runtime775.exact-main-workflow-evidence.v1",
        "scope": "Frozen Linux x86-64 Current HIA Runtime focused suites only; no other head or subsystem qualified.",
        "status": q["status"], "source_commit": HEAD, "workflow_run": 44912529511, "run_attempt": 1,
        "workflow_job": 136871176361, "event": "push", "artifact": {"id": 14713706649, "bytes": ZIP[0],
            "sha256": ZIP[1], "zip_member_crc_count": 8, "expiry": "2026-12-21T15:43:37Z"},
        "sealed_archive_reported_by_custody": {"sha256": ARCHIVE[1], "bytes": ARCHIVE[0],
            "members": custody["member_count"], "member_bytes": custody["member_bytes"],
            "declared_inputs": custody["declared_input_count"], "custody_report_sha256": CUSTODY[1]},
        "workflow_summary_sha256": CI[1], "archived_member_sha256": selected,
        "executed_cases": q["executed_cases_total"], "groups": len(q["suites"]),
        "passing_suites": [{"name": x["name"], "cases": x["cases"]} for x in q["suites"]],
        "compiled_functions": q["functions"], "compiled_units": q["units"],
        "generator_subgroups": q["generator_subgroups"], "verified_archive_member_payloads": len(selected),
        "full_project_status": q["full_project_status"], "full_pytorch_rewrite_complete": False,
        "full_schema_or_subsystem_credit": False, "native_or_Torch_rerun_by_this_reader": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_zip", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.original_zip), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
