"""Real nonempty Weak tombstone plus copied adversarial observer controls.

This is ownership admission, not an additional PyTorch numerical API case.
The first6ea refusal is recorded as an observer defect against actual source.
"""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import struct
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import owner_accounting_int64 as observer

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "source", "protocol", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    paths = {**observer.dependency_paths(), "observer": Path(observer.__file__),
             "executed_test": Path(__file__), "source": a.source, "native": a.binary,
             "protocol": a.protocol, "owner_client": a.protocol.with_name("owner_client.py")}
    inputs = {key: {"path": str(path.resolve()), "sha256": observer.sha(path.read_bytes()),
                   "bytes": path.stat().st_size} for key, path in paths.items()}
    report = {"status": "RUNNING", "schema": "heptatorch.Storage-tombstone-observer-regression.v1",
              "native_executed": False, "Torch_executed": False, "numeric_API_cases": 0,
              "inputs": inputs, "negative_rows": [], "snapshot_files": {},
              "whole_pytorch_gate": "HOLD", "full_pytorch_parity": False}
    session = None
    try:
        report["profile"] = observer.pin_profile(a.binary, a.source)
        assert inputs["protocol"]["sha256"] == "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9"
        assert inputs["owner_client"]["sha256"] == "dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b"
        wire = load("tombstone_wire", a.protocol)
        original = load("tombstone_original640", observer.ORIGINAL_STRUCTURAL)
        # Pure old-predicate replay under the new snapshot identity. The old
        # file and every function body remain byte-identical; only these three
        # module data bindings identify this profile's copied JSON snapshot.
        original.SOURCE = observer.structural.SOURCE
        original.IMAGE = observer.structural.IMAGE
        original.PROFILE = observer.structural.PROFILE
        report["original_predicate_profile_binding"] = {
            "source_sha256": original.SOURCE, "native_sha256": original.IMAGE,
            "profile": original.PROFILE, "function_code_changed": False,
            "purpose": "copied JSON predicate replay only; old native profile not relabelled"}
        session = wire.Session(a.binary, directory=out / "transport")
        report["native_executed"] = True

        def snapshot(label):
            actual = session.owner()
            value = observer.structural.decode_owner(actual["bytes"], actual["pointer"])
            data = observer.canonical(value)
            path = out / (label + ".json.gz")
            with path.open("wb") as f:
                with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as z:
                    z.write(data)
            report["snapshot_files"][label] = {"path": path.name, "bytes": len(data),
                "sha256": observer.sha(data), "compressed_sha256": observer.sha(path.read_bytes())}
            observer.verify_snapshot(value)
            return value

        x = session.run([wire.tensor((3,), [.25, .5, 1])])[0]["value"]
        weak = session.run([(42, x)])[0]["value"]
        before = snapshot("before-last-strong-drop")
        wr = before["tables"]["4"][(weak & 0xffffff) - 1]
        sid = wr[2]; slot = (sid & 0xffffff) - 1
        prior = before["tables"]["2"][slot]
        assert prior[3:8] == [12, 16, 1, 2, 0], prior
        session.run([(41, x)])
        after = snapshot("after-last-strong-drop")
        current = after["tables"]["2"][slot]
        assert current == [prior[0], 1, 0, 12, 0, 0, 1, 1], current
        assert sum(n for _, n in after["free_extents"]) == sum(n for _, n in before["free_extents"]) + 16
        commands = [[41, x, 0, 0, 0, 0, 0, 0]]
        report["transition"] = observer.verify_transition(before, after, [], commands)
        try:
            original.verify_snapshot(after)
        except AssertionError as error:
            assert error.args[0]["observer_rule"] == "expired-Storage-extent"
            report["preserved_original_observer_refusal"] = error.args[0]
        else:
            raise AssertionError("Expected original6ea observer mismatch was not reproduced")

        def reject(name, mutate, wanted):
            changed = copy.deepcopy(after)
            mutate(changed["tables"]["2"][slot])
            data = observer.canonical(changed)
            with gzip.open(out / (name + ".json.gz"), "wb") as f:
                f.write(data)
            try:
                observer.verify_transition(before, changed, [], commands)
            except AssertionError as error:
                assert error.args[0]["observer_rule"] == wanted, repr(error)
                report["negative_rows"].append({"case": name, "status": "PASS_EXPECTED_REJECTION",
                    "expected_rule": wanted, "actual_rule": error.args[0]["observer_rule"],
                    "input_sha256": observer.sha(data)})
            else:
                raise AssertionError((name, "incorrectly accepted"))

        reject("expired-offset-nonzero", lambda r: r.__setitem__(2, 16), "expired-Storage-extent")
        reject("expired-rounded-nonzero", lambda r: r.__setitem__(4, 16), "expired-Storage-extent")
        reject("expired-requested-outside-pool", lambda r: r.__setitem__(3, after["header"]["pool"] + 1), "expired-Storage-requested-bound")
        reject("expired-requested-changed", lambda r: r.__setitem__(3, 8), "expired-Storage-requested-preserved")
        report["weak_lock_rollback_bytes"] = session.error([(43, weak)], 6)
        assert observer.canonical(snapshot("after-expired-weak-lock")) == observer.canonical(after)
        session.run([(44, weak)])
        dropped = snapshot("after-last-weak-drop")
        final = dropped["tables"]["2"][slot]
        assert final[1:] == [0] * 7 and final[0] == prior[0]
        assert session.run([(53,)])[0]["value"] == 1
        session.finish(); assert session.exit_code == 0
        report.update(status="PASS", native_cases=1, copied_negative_cases=4,
                      frames=session.frames, EOF_exit_code=0, before_storage=prior, expired_storage=current)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        if session is not None and session.poll() is None:
            session.stop()
        report["inputs_after_unchanged"] = all(path.stat().st_size == inputs[key]["bytes"]
            and observer.sha(path.read_bytes()) == inputs[key]["sha256"] for key, path in paths.items())
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "executed-test.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": report["status"], "report": str(out / "report.json"),
                      "sha256": observer.sha((out / "report.json").read_bytes()),
                      "error": report.get("error")}))
    return 0 if report["status"] == "PASS" and report["inputs_after_unchanged"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
