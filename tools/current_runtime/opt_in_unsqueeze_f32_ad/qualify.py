#!/usr/bin/env python3
"""Differentiate real pinned Torch and exact native opt-in F32 unsqueeze."""
import argparse
import importlib.util
import math
import json
import os
from pathlib import Path
import runpy
import sys

PARENT = "dea9c05d594d54e42c5f2d7cf48b3e0a731b3272859402efd3a57b74dfb3da7a"
IMAGE = "9a40e0951189760a0d6de41bec2fe35630eeb381e1af2e7fa89eb77243c98474"
SOURCE = "8ee5db3dd34f56596c13d8c214f554ada057f04e8cfb8cfc8cdd7ce6d91013a7"
SHAPES = ((), (1,), (7,), (2, 3), (1, 3, 5), (2, 1, 9), (2, 3, 4))


def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("repository", "parent", "candidate", "candidate-source", "output", "graph-free-qualifier"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    assert not sys.flags.optimize
    repo, out, parent, candidate, source, base_driver = [x.resolve() for x in (
        a.repository, a.output, a.parent, a.candidate, a.candidate_source, a.graph_free_qualifier)]
    base = module("unsqueeze_graph_free", base_driver)
    sha, one = base.sha, base.one
    assert sha(parent.read_bytes()) == PARENT and sha(candidate.read_bytes()) == IMAGE
    assert sha(source.read_bytes()) == SOURCE
    out.mkdir(parents=True, exist_ok=False)
    wire = module("unsqueeze_ad_wire", repo / "tests/integer_tensor_review/wire/protocol.py")
    sys.path.insert(0, str(repo / "tests"))
    from half_logical.native import Memory
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    os.chdir(repo)
    argv = sys.argv
    try:
        sys.argv = [str(repo / "tests/pytorch_oracle.py"), str(out / "NO_NATIVE"),
                    str(out / "oracle-error.json"), str(repo / "compatibility/pytorch-oracle-lock.json")]
        ctx = runpy.run_path(sys.argv[0], run_name="heptatorch_unsqueeze_ad_reference")
    finally:
        sys.argv = argv
    torch, prov = ctx["torch"], ctx["PROVENANCE"]
    assert prov["status"] == "PASS" and torch.get_num_threads() == 1
    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
    old = wire.Session(parent, directory=out / "parent")
    new = wire.Session(candidate, directory=out / "candidate")
    rows = []
    try:
        x = one(new, *wire.tensor((1,), (0.0,)))["value"]
        one(new, 41, x)
        for shape in SHAPES:
            count = math.prod(shape)
            vals = tuple((i - 8) * 0.5 for i in range(count))
            for axis in range(-len(shape)-1, len(shape)+1):
                before = Memory(new.owner()["bytes"]).live_counts()
                p_old = one(old, *wire.tensor(shape, vals, requires=1))["value"]
                old.error([(129, p_old, axis)], 5)
                one(old, 41, p_old)
                x = one(new, *wire.tensor(shape, vals, requires=1))["value"]
                tx = torch.tensor(vals, dtype=torch.float32).reshape(shape).clone().requires_grad_()
                rounds = []
                for round_index in range(2):
                    y = one(new, 129, x, axis)["value"]
                    ty = torch.ops.aten.unsqueeze.default(tx, axis)
                    old_input, view = (wire.meta(r) for r in new.run([(13, x), (13, y)]))
                    assert (view["storage_id"], view["version_id"]) == (old_input["storage_id"], old_input["version_id"])
                    assert ty.untyped_storage()._cdata == tx.untyped_storage()._cdata
                    assert ty.requires_grad and not ty.is_leaf
                    assert (view["rank"], tuple(view["shape"][:ty.ndim]), tuple(view["strides"][:ty.ndim])) == (
                        ty.ndim, tuple(ty.shape), ty.stride())
                    assert view["storage_offset"] == view["version"] == 0 and view["requires_grad"] == 1
                    m = Memory(new.owner()["bytes"])
                    nx, ny = m.node(x), m.node(y)
                    assert nx[27:29] == ny[27:29] and ny[4:6] == (1, 45)
                    assert ny[6] == m.row(1, x)[6]
                    observed = one(new, 39, y, 0, count)
                    assert observed["payload"] == base.raw_tensor(ty)
                    seeds = tuple((2 + i % 9) * (0.125 if round_index == 0 else -0.25) for i in range(count))
                    g = one(new, *wire.tensor(tuple(ty.shape), seeds))["value"]
                    tg = torch.tensor(seeds, dtype=torch.float32).reshape(ty.shape)
                    ty.backward(tg)
                    assert one(new, 29, y, g)["value"] == 1
                    grad = one(new, 11, x)
                    expected = base.raw_tensor(tx.grad)
                    assert grad["payload"] == expected and grad["value"] == count
                    rounds.append({"forward_sha256": sha(observed["payload"]), "accumulated_gradient_sha256": sha(expected)})
                    new.run([(41, y), (41, g)])
                assert one(new, 39, x, 0, count)["payload"] == base.raw_tensor(tx)
                one(new, 41, x)
                assert Memory(new.owner()["bytes"]).live_counts() == before
                rows.append({"shape": shape, "axis": axis, "elements": count, "rounds": rounds})

        edge = one(new, *wire.tensor((3,), (1, 2, 3), requires=1))["value"]
        view = one(new, 129, edge, 0)["value"]
        new.error([(129, view, 0)], 5)
        empty = one(new, *wire.tensor((0,), (), requires=1))["value"]
        new.error([(129, empty, 0)], 5)
        new.error([(129, edge, 2)], 2)
        new.run([(41, view), (41, edge), (41, empty)])
    finally:
        old.finish()
        new.finish()

    baseline = wire.Session(parent, directory=out / "graph-free-parent")
    experiment = wire.Session(candidate, directory=out / "graph-free-candidate")
    equal_frames = 0
    def replay(cmds):
        nonlocal equal_frames
        p = wire.packet(cmds)
        left, right = baseline.exchange(p), experiment.exchange(p)
        assert left == right and wire.parse(left)[0] == 0
        equal_frames += 1
        return wire.parse(left)[1]
    try:
        for shape, layout in base.CASES:
            count = math.prod(shape)
            for axis in range(-len(shape if layout is None else layout[0])-1, len(shape if layout is None else layout[0])+1):
                x = replay([wire.tensor(shape, range(count))])[0]["value"]
                z = x if layout is None else replay([(111, x, base.view_config(wire, layout), 80)])[0]["value"]
                y = replay([(129, z, axis)])[0]["value"]
                replay([(95, x, wire.bits(1.0)), (13, y)])
                replay(([(41, z)] if z != x else []) + [(41, x), (13, y)])
                replay([(41, y)])
    finally:
        baseline.finish()
        experiment.finish()

    report = {"status": "SCOPED_PASS", "parent_sha256": PARENT, "candidate_sha256": IMAGE,
              "candidate_source_sha256": SOURCE, "oracle_lock_sha256": prov["lock_sha256"],
              "native_executed": True, "pytorch_executed": True, "canonical_credit": False,
              "scoped_cases": len(rows), "seeded_accumulation_rounds": 2 * len(rows),
              "forward_and_grad_values_each": 2 * sum(r["elements"] for r in rows),
              "old_grad_refusals": len(rows), "candidate_scope_refusals": 3,
              "graph_free_byte_exact_frames": equal_frames, "rows": rows}
    (out / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
