"""Run the separately pinned 640-byte owner accounting qualification.

The default path really executes four native controls, then 24 pure copied
negative controls. It does not rebuild a producer or execute Torch. All build
inputs and the unchanged legacy observer are checked before and after use.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import traceback

PROFILE_SHA = "2b17470180ce3658bd3bd38ee3be36d67646e107e6e6c90fb97692b03c956660"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    repo, build, here = a.repo.resolve(), a.build.resolve(), Path(__file__).resolve().parent
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    report = {"schema": "heptatorch.dtype640-owner-qualification.v1", "status": "RUNNING",
              "qualification_role": "Integration of explicit native controls and independent copied-JSON refusals.",
              "Torch_executed": False, "whole_pytorch_gate": "HOLD", "full_pytorch_parity": False,
              "pytorch_api_denominator": 2590, "children": [], "profile_sha256": PROFILE_SHA,
              "executable_rebuilt_this_invocation": False}

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def bind(path, relative, expected=None, size=None):
        path = path.resolve()
        assert path.is_file(), str(path)
        raw = path.read_bytes(); mode = stat.S_IMODE(path.stat().st_mode)
        assert expected is None or sha(raw) == expected, (str(path), sha(raw), expected)
        assert size is None or len(raw) == size
        relative = Path(relative)
        assert not relative.is_absolute() and ".." not in relative.parts
        target = out / "inputs" / relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw); target.chmod(mode)
        inputs[path] = {"bytes": len(raw), "sha256": sha(raw), "mode": mode,
                        "copied_to": str(target.relative_to(out))}
        return raw

    def execute(name, args):
        directory = out / name; directory.mkdir(exist_ok=False)
        argv = [sys.executable, "-I", "-B", *map(str, args), "--output", str(directory / "observed")]
        command = json.dumps(argv, indent=2).encode() + b"\n"
        (directory / "command.json").write_bytes(command)
        child = {"name": name, "argv": argv, "command_sha256": sha(command), "status": "RUNNING"}
        report["children"].append(child); save()
        with (directory / "stdout.txt").open("wb") as stdout, (directory / "stderr.txt").open("wb") as stderr:
            process = subprocess.run(argv, cwd=repo, stdout=stdout, stderr=stderr)
        child["exit_code"] = process.returncode
        for stream in ("stdout", "stderr"):
            value = (directory / (stream + ".txt")).read_bytes()
            child[stream] = {"path": str((directory / (stream + ".txt")).relative_to(out)),
                             "bytes": len(value), "sha256": sha(value)}
        rp = directory / "observed/report.json"
        if rp.is_file():
            raw = rp.read_bytes(); observed = json.loads(raw)
            child.update(report={"path": str(rp.relative_to(out)), "bytes": len(raw), "sha256": sha(raw)},
                         observed=observed)
        else:
            observed = {}
        child["status"] = "PASS" if process.returncode == 0 and observed.get("status") == "PASS" else "FAIL"
        save(); assert child["status"] == "PASS", (name, process.returncode, observed.get("error"))
        return observed, rp

    try:
        bind(Path(__file__), "profile/qualify.py")
        profile = json.loads(bind(here / "profile.json", "profile/profile.json", PROFILE_SHA))
        for name, info in profile["files"].items():
            bind(here / name, "profile/" + name, info["sha256"], info["bytes"])
        for name, info in profile["repository_dependencies"].items():
            bind(repo / name, "repository/" + name, info["sha256"])
        br = json.loads(bind(build / "build.json", "native/build.json"))
        assert br["status"] == "PASS" and br["source_sha256"] == profile["source_sha256"]
        assert br["native_sha256"] == profile["native_sha256"]
        assert br["descriptor_bytes"] == profile["descriptor_bytes"] and len(br["units"]) == profile["module_count"]
        bind(build / "runtime.hepta", "native/runtime.hepta", profile["source_sha256"])
        bind(build / profile["native_filename"], "native/" + profile["native_filename"], profile["native_sha256"])
        unit_names = set()
        for unit in br["units"]:
            assert unit["compiler_exit"] == 0 and unit["name"] not in unit_names
            unit_names.add(unit["name"])
            for suffix, key in ((".hepta", "source_sha256"), (".o", "object_sha256")):
                name = unit["name"] + suffix
                assert Path(name).name == name
                bind(build / name, "native/" + name, unit[key])
        report.update(profile=profile, source_sha256=profile["source_sha256"],
                      native_sha256=profile["native_sha256"], unit_count=len(unit_names))
        native, native_report = execute("native-controls", [here / "test_owner640.py", "--build", build,
            "--observer", here / "owner640.py", "--expected-observer-sha256", profile["files"]["owner640.py"]["sha256"],
            "--protocol", repo / "tests/retained_resource_review/protocol.py", "--old-observer",
            repo / "tests/private_node_accounting/owner_accounting.py"])
        expected = profile["native_controls"]
        assert native["case_count"] == native["passed"] == expected["case_count"] and native["failed"] == 0
        assert native["frames_before_EOF"] == expected["frames"]
        assert native["full_structural_snapshots_checked"] == expected["full_structural_snapshots"]
        assert native["isolated_transitions_checked"] == expected["isolated_transitions"]
        assert native["explicitly_snapshot_only_events"] == expected["snapshot_only_events"]
        assert native["source_custody_unchanged"] is True and native["native_executed"] is True
        copied, _ = execute("copied-controls", [here / "test_owner640_negative.py", "--observer", here / "owner640.py",
            "--expected-observer-sha256", profile["files"]["owner640.py"]["sha256"], "--native-report", native_report,
            "--expected-native-report-sha256", sha(native_report.read_bytes())])
        assert copied["negative_control_count"] == copied["expected_refusals"] == profile["copied_controls"]["expected_refusals"]
        assert copied["unexpected_acceptances"] == 0 and copied["input_bytes_unchanged"] is True
        assert [{"name": r["name"], "expected_rule": r["expected_rule"]} for r in copied["controls"]] == profile["copied_controls"]["names_and_rules"]
        assert all(r["status"] == "EXPECTED_REFUSAL" for r in copied["controls"])
        report.update(status="PASS", native_case_count=4, native_frames=68,
                      copied_negative_controls=24, native_executed_this_invocation=True,
                      unsupported_allocation_ledgers_not_claimed=True,
                      frozen_355_428_446_observer_unchanged=True)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        unchanged = all(path.is_file() and path.stat().st_size == meta["bytes"]
            and stat.S_IMODE(path.stat().st_mode) == meta["mode"] and sha(path.read_bytes()) == meta["sha256"]
            for path, meta in inputs.items())
        report["inputs_unchanged"] = unchanged
        if not unchanged:
            report["status"] = "FAIL"
        report["inputs"] = {str(path): meta for path, meta in inputs.items()}
        save()
    print(json.dumps({k: report.get(k) for k in ("status", "native_case_count", "native_frames",
        "copied_negative_controls", "inputs_unchanged", "error")}))
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
