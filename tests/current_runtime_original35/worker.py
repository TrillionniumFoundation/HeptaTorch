#!/usr/bin/env python3
"""Execute original35 and the original18 observer controls on one current image."""
import argparse
import ast
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from binding import ROOT, Inputs, bind_build, dump, require, sha
import observer as adapter


def invoke(function, argv, output):
    saved = sys.argv
    try:
        sys.argv = argv
        with output.open("w") as stream, contextlib.redirect_stdout(stream):
            result = function()
        require(result in (None, 0), "original validator returned failure")
    finally:
        sys.argv = saved


def check_original_report(original, wrapped, profile, out, inputs):
    inputs.bind(adapter.CHECKER, adapter.CHECKER_SHA)
    checker = adapter.load(adapter.CHECKER, "current_runtime_original35_report_checker")
    before = dict(checker.PROFILES)
    checker.PROFILES = {profile["native_sha256"]: (profile["name"], profile["source_sha256"])}
    try:
        invoke(checker.main, [str(adapter.CHECKER), "--observer-report", str(wrapped),
                             "--original-report", str(original)], out / "original-report-check.stdout")
    finally:
        checker.PROFILES = before
    return dict(original_checker_sha256=adapter.CHECKER_SHA,
                runtime_rebindings=["PROFILES exact native/source/name identity table"],
                original_main_body_unchanged=True, all_original_assertions_executed=True)


def copied_controls(build, bodies, build_directory, journals, out, inputs):
    """Keep all four positive and eighteen negative predicates byte-for-byte.

    The original loader still reads/checks the original observer source. Its
    freshly imported module receives the same explicit current identity adapter.
    The only AST edit selects the current profile name in the CLI choices.
    """
    raw = inputs.bind(adapter.CONTROLS, adapter.CONTROLS_SHA)
    controls = adapter.load(adapter.CONTROLS, "current_runtime_original18_controls")
    original = next(node for node in ast.parse(raw).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
    text = ast.get_source_segment(raw.decode(), original)
    old, new = repr(adapter.OLD_NAME), repr(adapter.NAME)
    require(text.count(old) == 1 and new not in text, "unique original profile CLI choice")
    changed = text.replace(old, new, 1)
    require(ast.dump(ast.parse(changed.replace(new, old, 1)).body[0], include_attributes=False)
            == ast.dump(original, include_attributes=False), "entire control function inverse AST")
    exec(compile(changed, str(adapter.CONTROLS) + "::<current-profile-name>", "exec"), controls.__dict__)
    original_spec = importlib.util.spec_from_file_location

    def current_spec(name, location, *args, **kwargs):
        spec = original_spec(name, location, *args, **kwargs)
        if name != "copied_snapshot_observer_under_test":
            return spec
        require(Path(location).resolve() == adapter.PARENT, "original control observer source path")
        original_loader = spec.loader

        class ConfiguredLoader:
            def create_module(self, inner_spec):
                return original_loader.create_module(inner_spec)

            def exec_module(self, module):
                original_loader.exec_module(module)
                adapter.configure(build, bodies, build_directory, inputs, module=module)

        spec.loader = ConfiguredLoader()
        return spec

    importlib.util.spec_from_file_location = current_spec
    try:
        invoke(controls.main, [str(adapter.CONTROLS), "--observer", str(adapter.PARENT),
                              "--journals", str(journals), "--output", str(out / "copied-controls"),
                              "--expected-profile", adapter.NAME], out / "copied-controls.stdout")
    finally:
        importlib.util.spec_from_file_location = original_spec
    report = json.loads((out / "copied-controls/report.json").read_bytes())
    require(report["status"] == "PASS_OBSERVER_CONTROLS_ONLY"
            and report["positive_cases"] == 4 and report["negative_cases"] == 18,
            "complete original observer controls")
    return dict(original_controls_sha256=adapter.CONTROLS_SHA,
                profile_choice_edit=dict(before=adapter.OLD_NAME, after=adapter.NAME),
                control_entire_inverse_AST_equals_original=True,
                runtime_rebindings=["observer loader applies explicit current identity adapter"],
                original_positive_and_negative_predicates_unchanged=True,
                positive_cases=4, negative_cases=18,
                report_sha256=sha((out / "copied-controls/report.json").read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--expected-build-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("native", "strict", "controls"), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(not sys.flags.optimize, "original assertions must remain enabled")
    directory, out = args.build_directory.resolve(), args.output.resolve()
    require(out.is_dir(), "current35 coordinator must create its fresh output directory")
    inputs = Inputs()
    report = dict(schema="heptatorch.current-runtime.original35-stage.v1", status="RUNNING",
                  phase=args.phase, actual_build_sha256=args.expected_build_sha256,
                  full_project_status="HOLD", full_pytorch_rewrite_complete=False,
                  old_pass_transferred=False, reference_or_numeric_assertions_changed=False)
    result = 1
    try:
        for path in sorted(HERE.glob("*.py")):
            inputs.bind(path)
        build, bodies = bind_build(directory, args.expected_build_sha256, inputs)
        owner, adaptation = adapter.configure(build, bodies, directory, inputs)
        report.update(source_sha256=build["source_sha256"], native_sha256=build["native_sha256"],
                      functions=build["functions"], native_units=build["units_planned"])
        identity = out / "identity-adapter.json"
        if args.phase == "native":
            require(not identity.exists(), "fresh actual identity adapter report")
            dump(identity, adaptation)
        else:
            require(json.loads(inputs.bind(identity)) == adaptation,
                    "every stage must use the same explicit current identity and ViewInfo adapter")
        original, wrapped = out / "original-report.json", out / "observer-report.json"
        if args.phase == "native":
            require(not original.exists() and not wrapped.exists(), "fresh original35 reports")
            sys.path.insert(0, str(adapter.PARENT.parent))
            launcher = adapter.load(adapter.LAUNCHER, "current_runtime_original35_launcher")
            launcher.observer = owner
            argv = [str(adapter.LAUNCHER), "--legacy-directory", str(ROOT / "tests/tensor_object_oracle"),
                    "--native-source", str(directory / "runtime.hepta"),
                    "--observer-output", str(out / "observer"), "--observer-report", str(wrapped),
                    "--", str(directory / "heptatorch"), str(original),
                    "--owned-reference", str(ROOT / "tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py"),
                    "--session-reference", str(ROOT / "tests/pytorch_session_oracle.py"),
                    "--reference", str(ROOT / "tests/pytorch_oracle.py"),
                    "--lock", str(ROOT / "compatibility/pytorch-oracle-lock.json"), "--repo", str(ROOT),
                    "--trace-directory", str(out / "wire"), "--expected-native-sha256", build["native_sha256"]]
            if args.prepare_only:
                argv.append("--prepare-only")
            saved = sys.argv
            try:
                sys.argv = argv
                returned = launcher.main()
            finally:
                sys.argv = saved
            report["original_launcher_return_code"] = returned
            inner_raw = inputs.bind(original)
            inner = json.loads(inner_raw)
            report["original_report"] = dict(path=str(original), sha256=sha(inner_raw),
                status=inner["status"], cases=inner["cases"], failed_cases=inner.get("failed_cases", 0),
                failed_rows=[dict(case=row["case"], error=row.get("error"), frames=row.get("frames"))
                             for row in inner.get("rows", []) if row["status"] != "PASS"])
            # Preserve the original wrapper's complete bytes before adding the
            # independently checked cold-build and adapter receipt. Neither the
            # original numeric report nor any observer assertion is rewritten.
            wrapper_raw = wrapped.read_bytes()
            raw_copy = out / "observer-report.before-build-binding.json"
            with raw_copy.open("xb") as stream:
                stream.write(wrapper_raw)
            inputs.bind(raw_copy, sha(wrapper_raw))
            wrapper = json.loads(wrapper_raw)
            require("cold_build_admission" not in wrapper, "original wrapper has no implicit build claim")
            wrapper["cold_build_admission"] = dict(
                schema="heptatorch.current-runtime.cold-build-admission.v1",
                status="ADMITTED_CURRENT_COLD_BUILD", build_sha256=args.expected_build_sha256,
                source_sha256=build["source_sha256"], native_sha256=build["native_sha256"],
                functions=build["functions"], units=build["units_planned"],
                native_executed_by_admission=False, Torch_executed_by_admission=False,
                runtime_qualification_inferred=False, files=dict(inputs.records))
            wrapper["current_adapter_binding"] = dict(
                original_wrapper_sha256=sha(wrapper_raw), identity_adapter_sha256=sha(identity.read_bytes()),
                original_wrapper_fields_unchanged=True, original_numeric_report_unchanged=True,
                inputs=dict(inputs.records))
            dump(wrapped, wrapper)
            require(returned == 0, "original35 launcher failure; see original_report.failed_rows")
            if args.prepare_only:
                require(inner["status"] == "NOT_RUN" and inner["cases"] == 0,
                        "preparation cannot claim native or Torch execution")
            else:
                require(inner["status"] == "PASS" and inner["cases"] == 35,
                        "all original35 cases must execute successfully")
        elif args.phase == "strict":
            inputs.bind(original)
            wrapper = json.loads(inputs.bind(wrapped))
            require(wrapper["cold_build_admission"]["build_sha256"] == args.expected_build_sha256,
                    "strict stage actual build binding")
            profile = owner.pin_profile(directory / "heptatorch", directory / "runtime.hepta")
            report["original_report_validation"] = check_original_report(original, wrapped, profile, out, inputs)
            # The parent supervisor parses exactly the unchanged strict
            # checker's original JSON on stdout, with no adapter relabeling.
            sys.stdout.write((out / "original-report-check.stdout").read_text())
            sys.stdout.flush()
        else:
            require(not args.prepare_only, "copied controls require actual native journals")
            report["original_observer_controls"] = copied_controls(
                build, bodies, directory, out / "observer/owners", out, inputs)
        report["status"] = "PREPARED_NOT_RUN" if args.prepare_only else "PASS"
        result = 0
    except BaseException as error:
        report.update(status="FAIL", error=dict(type=type(error).__name__, message=str(error),
                                                 traceback=traceback.format_exc()))
        traceback.print_exc(file=sys.stderr)
    finally:
        report["inputs"] = inputs.records
        report["changed_inputs"] = inputs.recheck()
        if report["changed_inputs"]:
            report["status"] = "FAIL"
            result = 1
        dump(out / (args.phase + "-stage.json"), report)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
