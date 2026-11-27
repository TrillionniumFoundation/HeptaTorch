#!/usr/bin/env python3
"""Run pinned PyTorch and separate canonical/candidate native owner processes."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import struct
import sys

PARENT = "ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e"
CANDIDATE = "dea9c05d594d54e42c5f2d7cf48b3e0a731b3272859402efd3a57b74dfb3da7a"
SOURCE = "2336ca87ac4fce0b46380c3a3d8539d230ff5c765c1c6ba4069d08617d614177"
CASES = (
    ((), None),
    ((0,), None),
    ((1,), None),
    ((5,), None),
    ((2, 3), None),
    ((2, 0, 3), None),
    ((2, 3, 4), None),
    ((14,), ((3, 2), (1, 4), 2)),
    ((14,), ((4, 3), (0, 1), 4)),
    ((14,), ((1, 3, 2), (200, 2, 1), 1)),
    ((14,), ((0, 3), (4, 1), 2)),
    ((14,), ((), (), 8)),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def one(session, *command):
    return session.run([command])[0]


def raw_tensor(t):
    if t.numel() == 0:
        return b""
    return bytes(t.contiguous().clone().untyped_storage())


def view_config(wire, layout):
    shape, strides, offset = layout
    pad = (0,) * (4 - len(shape))
    data = struct.pack("<10Q", len(shape), *shape, *pad, *strides, *pad, offset)
    return wire.Payload(data)


def observe(session, wire, decoder, token, torch_tensor):
    meta = wire.meta(one(session, 13, token))
    words = len(torch_tensor.shape)
    assert (meta["rank"], tuple(meta["shape"][:words]), tuple(meta["strides"][:words])) == (
        words, tuple(torch_tensor.shape), tuple(torch_tensor.stride())), (meta, torch_tensor.shape, torch_tensor.stride())
    assert meta["numel"] == torch_tensor.numel()
    assert meta["storage_offset"] == 4 * torch_tensor.storage_offset(), (meta, torch_tensor.storage_offset())
    assert meta["requires_grad"] == 0 and meta["version"] == torch_tensor._version
    if torch_tensor.numel():
        record = one(session, 39, token, 0, torch_tensor.numel())
        assert record["value"] == torch_tensor.numel() and record["payload"] == raw_tensor(torch_tensor)
    owner = decoder(session.owner()["bytes"])
    node = owner.node(token)
    storage, versions = owner.row(2, node[27]), owner.row(3, node[28])
    assert (node[27], node[28], node[29], storage[3], versions[2]) == (
        meta["storage_id"], meta["version_id"], meta["storage_offset"], meta["storage_bytes"], meta["version"])
    start = owner.start + storage[2]
    native = owner.raw[start:start + storage[3]]
    physical = bytes(torch_tensor.untyped_storage())
    assert native == physical
    return meta, owner


def construct(session, wire, torch, shape, layout):
    count = math.prod(shape)
    vals = tuple((i - 9) * 0.25 for i in range(count))
    root = one(session, *wire.tensor(shape, vals))["value"]
    t_root = torch.tensor(vals, dtype=torch.float32).reshape(shape).clone()
    if layout is None:
        return root, root, t_root, t_root
    source = one(session, 111, root, view_config(wire, layout), 80)["value"]
    t_input = torch.as_strided(t_root, *layout)
    return root, source, t_root, t_input


def exercise(session, wire, decoder, torch, shape, layout, axis):
    before = decoder(session.owner()["bytes"]).live_counts()
    root, source, t_root, t_input = construct(session, wire, torch, shape, layout)
    t_view = torch.ops.aten.unsqueeze.default(t_input, axis)
    y = one(session, 129, source, axis)["value"]
    assert y not in {root, source}
    m_source, _ = observe(session, wire, decoder, source, t_input)
    m_y, owner = observe(session, wire, decoder, y, t_view)
    assert t_input.untyped_storage()._cdata == t_view.untyped_storage()._cdata
    assert (m_source["storage_id"], m_source["version_id"]) == (m_y["storage_id"], m_y["version_id"])
    assert owner.node(source)[27:29] == owner.node(y)[27:29]
    # Read real result after a base mutation to prove shared physical bytes and version.
    one(session, 95, root, wire.bits(1.0))
    t_root.add_(1.0)
    updated_input, _ = observe(session, wire, decoder, source, t_input)
    updated_y, _ = observe(session, wire, decoder, y, t_view)
    assert updated_input["version"] == updated_y["version"] == 1
    assert updated_y["storage_id"] == m_y["storage_id"] and updated_y["version_id"] == m_y["version_id"]
    if source != root:
        one(session, 41, source)
    one(session, 41, root)
    session.error([(129, source, 0)], 1)
    del t_input, t_root
    gc.collect()
    m_retained, _ = observe(session, wire, decoder, y, t_view)
    assert m_retained == updated_y
    one(session, 41, y)
    after = decoder(session.owner()["bytes"]).live_counts()
    assert after == before, (shape, layout, axis, before, after)
    return {
        "base_shape": shape, "source_geometry": layout, "axis": axis,
        "result_shape": tuple(t_view.shape), "result_stride": t_view.stride(),
        "storage_offset": t_view.storage_offset(), "values": t_view.numel(),
        "physical_storage_sha256_after_mutation": sha(bytes(t_view.untyped_storage())),
        "data_sha256_after_mutation": sha(raw_tensor(t_view)),
        "physical_alias_and_version": True, "base_release_survival": True, "final_live_counts_restored": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("repository", "parent", "candidate", "candidate-source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    assert not sys.flags.optimize
    repo, out, parent, candidate, source = (
        a.repository.resolve(), a.output.resolve(), a.parent.resolve(), a.candidate.resolve(), a.candidate_source.resolve())
    assert sha(parent.read_bytes()) == PARENT
    assert sha(candidate.read_bytes()) == CANDIDATE
    assert sha(source.read_bytes()) == SOURCE
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("unsqueeze_wire", repo / "tests/integer_tensor_review/wire/protocol.py")
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    sys.path.insert(0, str(repo / "tests"))
    from half_logical.native import Memory
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    os.chdir(repo)
    argv = sys.argv
    try:
        sys.argv = [str(repo / "tests/pytorch_oracle.py"), str(out / "NO_NATIVE"),
                    str(out / "oracle-error.json"), str(repo / "compatibility/pytorch-oracle-lock.json")]
        ctx = runpy.run_path(sys.argv[0], run_name="heptatorch_unsqueeze_reference")
    finally:
        sys.argv = argv
    torch, prov = ctx["torch"], ctx["PROVENANCE"]
    assert prov["status"] == "PASS" and torch.get_num_threads() == 1
    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
    sessions = []
    try:
        old = wire.Session(parent, directory=out / "parent")
        sessions.append(old)
        new = wire.Session(candidate, directory=out / "candidate")
        sessions.append(new)
        warm = one(new, *wire.tensor((1,), (0.0,)))["value"]
        one(new, 41, warm)
        rows = []
        parent_refusals = 0
        for shape, layout in CASES:
            rank = len(shape if layout is None else layout[0])
            for axis in range(-rank - 1, rank + 1):
                rows.append(exercise(new, wire, Memory, torch, shape, layout, axis))
                root, token, _, _ = construct(old, wire, torch, shape, layout)
                old.error([(129, token, axis)], 4)
                parent_refusals += 1
                if token != root:
                    one(old, 41, token)
                one(old, 41, root)
        x = one(new, *wire.tensor((2, 3), range(6)))["value"]
        for axis in (-4, 3, -(1 << 63), (1 << 63) - 1):
            new.error([(129, x, axis)], 2)
            try:
                torch.ops.aten.unsqueeze.default(torch.empty((2, 3)), axis)
            except IndexError:
                pass
            else:
                raise AssertionError("Torch accepted invalid dimension")
        new.error([(129, x, 1, 99)], 7)
        new.error([(95, x, wire.bits(1.0)), (129, x, 3)], 2)
        other_dtype = one(new, 108, x, 5, 0, 0)["value"]
        new.error([(129, other_dtype, 0)], 7)
        one(new, 41, other_dtype)
        one(new, 41, x)
        grad = one(new, *wire.tensor((1,), (1.0,), requires=1))["value"]
        new.error([(129, grad, 0)], 5)
        one(new, 41, grad)
        max_rank = one(new, *wire.tensor((1, 1, 1, 1), (1.0,)))["value"]
        new.error([(129, max_rank, 0)], 7)
        one(new, 41, max_rank)
    finally:
        for session in sessions:
            session.finish()
    regressions = 0
    old = wire.Session(parent, directory=out / "regression-parent")
    new = wire.Session(candidate, directory=out / "regression-candidate")
    def replay(commands):
        nonlocal regressions
        frame = wire.packet(commands)
        left, right = old.exchange(frame), new.exchange(frame)
        assert left == right and wire.parse(left)[0] == 0
        regressions += 1
        return wire.parse(left)[1]

    try:
        x = replay([wire.tensor((8,), range(8))])[0]["value"]
        y = replay([(111, x, view_config(wire, ((2, 2), (1, 3), 1)), 80)])[0]["value"]
        replay([(13, x), (13, y), (39, y, 0, 4)])
        replay([(95, x, wire.bits(1.0)), (13, x), (13, y), (39, y, 0, 4)])
        replay([(41, x), (13, y), (39, y, 0, 4)])
        replay([(41, y)])
    finally:
        old.finish()
        new.finish()
    receipt = {
        "status": "SCOPED_PASS", "canonical_credit": False, "candidate_source_sha256": SOURCE,
        "candidate_sha256": CANDIDATE, "parent_sha256": PARENT, "oracle_lock_sha256": prov["lock_sha256"],
        "oracle_checks": prov["checks"], "native_executed": True, "pytorch_executed": True,
        "cases": len(rows), "parent_unknown_op_refusals": parent_refusals,
        "invalid_or_rollback_frames": 9, "stale_rejections": len(rows), "byte_exact_regression_frames": regressions, "elements": sum(r["values"] for r in rows), "rows": rows,
    }
    (out / "qualification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
