"""Small actual native controls for the private-node observer; no Torch import.

Uses the unchanged source-verified Program/Session definitions as serializers and
executes the frozen native images. All assertions concern allocation identities,
real references, transport, reset and rollback. Numerical oracle qualification
is deliberately left to the separate pinned-Torch35 run.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import owner_accounting as observer

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, binary, source, out = (p.resolve() for p in (args.repository, args.binary, args.source, args.output))
    out.mkdir(parents=True, exist_ok=False)
    profile = observer.pin_profile(binary, source)
    legacy_dir = repo / "tests/tensor_object_oracle"
    legacy = module("frozen_native_control_bridge", legacy_dir / "bridge.py")
    need_paths = {
        "bridge.py": legacy_dir / "bridge.py",
        "pytorch_owned_tensor_oracle.py": repo / "tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py",
        "pytorch_session_oracle.py": repo / "tests/pytorch_session_oracle.py",
        "pytorch_oracle.py": repo / "tests/pytorch_oracle.py",
    }
    pins = {name: observer.sha(path.read_bytes()) for name, path in need_paths.items()}
    for name, pin in pins.items():
        observer.need(pin == observer.SOURCE_PINS[name], "frozen-control-source-pin", source=name)
    scope, definitions, registrations, owned_program, support_nodes, audit = legacy.prepare_sources(
        need_paths["pytorch_owned_tensor_oracle.py"], need_paths["pytorch_session_oracle.py"],
        need_paths["pytorch_oracle.py"])
    support = {"struct": struct, "__name__": "wire_definitions_only"}
    exec(compile(ast.fix_missing_locations(ast.Module(body=support_nodes, type_ignores=[])),
                 "<unchanged-serializer-definitions>", "exec"), support)
    scope["bits"], scope["f32"] = support["bits"], support["f32"]
    Program, Session = observer.install(legacy.install, scope, definitions, owned_program,
        support["Program"], binary, out / "wire", profile, out / "owners")
    rows = []
    def case(name, body):
        row = {"case": name, "status": "PASS", "rtol": 0., "atol": 0., "frames": 0,
               "request_bytes": 0, "response_bytes": 0, "compared_values": 0,
               "metadata_checks": 0, "max_absolute_error": 0.}
        session = None
        try:
            session = Session(row)
            body(session)
            session.close()
            scope["_v2_assert_versions"](session)
        except Exception as error:
            row.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        finally:
            if session is not None:
                session.stop()
        rows.append(row)
        (out / "report.json").write_text(json.dumps({
            "schema": "heptatorch.private-node-observer-native-controls.v1",
            "status": "RUNNING", "profile": profile, "rows": rows}, indent=2) + "\n")

    def private_gradient_reset_holes(s):
        p = s.program()
        x = p.full((), 2., True)
        factor = p.full((), 3.)
        product = p.binary(3, x, factor)
        loss = p.reduce(7, product)
        s.exchange(p)
        p = s.program(); p.backward(loss); s.exchange(p)
        before = s._last_owner
        g = observer.word(before["nodes"]["1"], 384)
        observer.need(g > 2 and observer.word(before["nodes"][str(g)], 400) == 1,
                      "control-real-private-G")
        p = s.program(); p.reset_graph(2); s.exchange(p)
        observer.need(s._last_owner["header"]["next"] == g + 1 and s.next_id == 3,
                      "control-native-versus-logical-high-water")
        p = s.program(); p.full((), .25); s.exchange(p)
        p = s.program(); p.command(999); s.exchange(p, status=4)
        s.row["explicit_private_G_node"] = g

    def same_frame_create_and_reset(s):
        for _ in range(2):
            p = s.program()
            x = p.full((3,), 1.)
            p.read(x)
            p.reset_graph(0)
            s.exchange(p)
        observer.need(s._last_owner["header"]["incarnation"] == 2
                      and s._last_owner["header"]["next"] == 1,
                      "control-incarnation-survives-reset")

    def compound_original_count(s):
        p = s.program()
        x = p.full((1, 1), 2., True)
        w = p.full((1, 1), 3., True)
        b = p.full((1,), 1., True)
        target = p.full((1, 1), 0.)
        output = p.linear(x, w, b)
        loss = p.mse(output, target)
        observer.need([t["count"] for t in p.creation_tickets] == [1, 1, 1, 1, 3, 3],
                      "control-original-compound-counts")
        s.exchange(p)
        p = s.program(); p.backward(loss); s.exchange(p)
        observer.need(s.observer_rows[-1]["created_private_G"] == 3, "control-three-real-G")
        p = s.program(); p.reset_graph(0); s.exchange(p)
        observer.need(s._last_owner["header"]["next"] == 1, "control-final-graph-disposal")

    case("private-gradient-reset-keeps-real-high-slot-and-rollback", private_gradient_reset_holes)
    case("same-frame-nongrad-create-reset-keeps-incarnation-counter", same_frame_create_and_reset)
    case("original-Linear-and-MSE-count3-plus-three-private-gradients", compound_original_count)
    good = all(row["status"] == "PASS" for row in rows)
    report = {
        "schema": "heptatorch.private-node-observer-native-controls.v1",
        "status": "PASS" if good else "FAIL", "cases": len(rows), "profile": profile,
        "frames": sum(row["frames"] for row in rows), "rows": rows,
        "source_custody": {**pins, "observer": observer.sha(Path(observer.__file__).read_bytes()),
                           "test": observer.sha(Path(__file__).read_bytes())},
        "original_numeric_oracle_executed": False, "host_numeric_runtime_used": False,
        "native_image_unchanged": observer.sha(binary.read_bytes()) == profile["native_sha256"],
        "frozen_source_overlay_audit": audit, "full_rewrite_complete": False,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("schema", "status", "cases", "frames")}))
    return 0 if good else 1

if __name__ == "__main__":
    raise SystemExit(main())
