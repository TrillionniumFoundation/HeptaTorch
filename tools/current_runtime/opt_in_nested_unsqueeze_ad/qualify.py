"""Compare nested native unsqueeze and two seeded adjoints with pinned Torch."""
import argparse
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys

from project import SOURCE, sha

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARENT = "970ee71344f863f3d473f3c626d4cbdb2672ea0971506ebaa7f6f2ca4205c76b"
ELF = "92c807d3d2affb2da2fdf3af4a39509f940ab986cf9e928cc53e0f261f502607"
WIRE = "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9"
SHAPES = ((), (1,), (5,), (11,), (1, 4), (2, 3), (3, 1), (3, 5))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for cap in ("binary", "parent", "source", "wheel", "install", "output"):
        p.add_argument("--" + cap, type=Path, required=True)
    a = p.parse_args()
    assert not sys.flags.optimize
    assert sha(a.binary.read_bytes()) == ELF and sha(a.parent.read_bytes()) == PARENT
    assert sha(a.source.read_bytes()) == SOURCE
    proto = REPO / "tests/integer_tensor_review/wire/protocol.py"
    assert sha(proto.read_bytes()) == WIRE
    w = load("nested_wire", proto)
    r = load("nested_custody", HERE.parent / "opt_in_conv2d_bias_rounding/qualify.py")
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.__version__ == "2.15.0+cpu" and torch.backends.cpu.get_cpu_capability() == "AVX2"
    custody = r.custody.verify_torch_custody(a.wheel, a.install, torch)
    sys.path.insert(0, str(REPO / "tests"))
    from half_logical.native import Memory
    a.output.mkdir(parents=True, exist_ok=False)
    parent = w.Session(a.parent.resolve(), directory=a.output / "parent")
    native = w.Session(a.binary.resolve(), directory=a.output / "candidate")
    frames = 0
    rows, controls = [], []

    def op(s, *command):
        nonlocal frames
        frames += 1
        return s.run([command])[0]

    def val(s, *command):
        return op(s, *command)["value"]

    def memory(s):
        return Memory(s.owner()["bytes"])

    def non_pages(s):
        m = memory(s)
        count = m.live_counts()
        stores, holds = set(), set()
        for i in range(256):
            addr, hold = m.q(4096 + 8*i), m.q(6144 + 8*i)
            assert bool(addr) == bool(hold)
            if hold:
                storage = m.row(6, hold)[2]
                row = m.row(2, storage)
                assert addr == (1 << 63) + row[2] and row[3] == row[4] == 172032
                stores.add(storage)
                holds.add(hold)
        assert len(stores) == len(holds)
        count[1] -= len(stores)
        count[5] -= len(holds)
        return count, len(holds)

    def raw(t):
        return bytes(t.detach().contiguous().clone().untyped_storage())

    def check(h, t):
        meta = w.meta(op(native, 13, h))
        rank = t.ndim
        assert meta["rank"] == rank and meta["numel"] == t.numel()
        assert tuple(meta["shape"][:rank]) == tuple(t.shape)
        assert tuple(meta["strides"][:rank]) == t.stride()
        assert meta["requires_grad"] == int(t.requires_grad)
        assert meta["storage_offset"] == meta["version"] == 0
        assert op(native, 39, h, 0, t.numel())["payload"] == raw(t)

    def tensor(s, shape, vals, grad=False):
        return val(s, *w.tensor(shape, vals, requires=int(grad)))

    def drop(s, *handles):
        for h in handles:
            op(s, 41, h)

    try:
        for s in (native, parent):
            drop(s, tensor(s, (1,), [0]))
            assert non_pages(s)[0] == [0]*8
        for shape in SHAPES:
            n = math.prod(shape)
            src = [((k*7+3)%31-15)*0.125 for k in range(n)]
            rank = len(shape)
            old_x = tensor(parent, shape, src, True)
            old_y = val(parent, 129, old_x, -1)
            parent.error([(129, old_y, 0)], 5)
            drop(parent, old_y, old_x)
            for a1, a2 in itertools.product(range(-rank-1, rank+1), range(-rank-2, rank+2)):
                tx = torch.tensor(src, dtype=torch.float32).reshape(shape).requires_grad_()
                ty = torch.unsqueeze(tx, a1)
                tz = torch.unsqueeze(ty, a2)
                x = tensor(native, shape, src, True)
                y = val(native, 129, x, a1)
                z = val(native, 129, y, a2)
                for h, t in ((x, tx), (y, ty), (z, tz)):
                    check(h,t)
                m = memory(native)
                nx, ny, nz = (m.node(h) for h in (x,y,z))
                assert ny[4:6] == nz[4:6] == (1,45)
                assert ny[6] == m.row(1,x)[6] and nz[6] == m.row(1,y)[6]
                assert nx[27:29] == ny[27:29] == nz[27:29]
                assert tx.untyped_storage()._cdata == ty.untyped_storage()._cdata == tz.untyped_storage()._cdata
                drop(native, y)
                check(z,tz)
                grads = []
                for factor in (0.25,-0.125):
                    seed = [((i*3+4)%23+1)*factor for i in range(n)]
                    g = tensor(native, tuple(tz.shape), seed)
                    tz.backward(torch.tensor(seed, dtype=torch.float32).reshape(tz.shape), retain_graph=True)
                    assert val(native, 29, z, g) == 1
                    record = op(native, 11, x)
                    assert record["value"] == n and record["payload"] == raw(tx.grad)
                    grads.append(sha(record["payload"]))
                    drop(native, g)
                check(x,tx)
                drop(native,z,x)
                rows.append({"shape":shape, "axis_pair":[a1,a2], "elements":n,
                             "forward_sha256":sha(raw(tz)), "gradients_sha256":grads})
            assert non_pages(native)[0] == non_pages(parent)[0] == [0]*8

        x = tensor(native, (), [2], True)
        y = val(native, 129, x, 0)
        z = val(native, 129, y, 0)
        native.error([(129,z,0)], 5)
        controls.append("third-ad-edge-refused")
        native.error([(129,y,2)], 2)
        controls.append("invalid-nested-axis-refused")
        drop(native,z,y,x)
        x = tensor(native, (2,), [2,3], True)
        y = val(native, 17, x, 2, 1, 2, 0, 0)
        native.error([(129,y,0)], 5)
        controls.append("unrelated-kind16-reshape-refused")
        z = val(native, 3, 16, x)
        native.error([(129,z,0)], 5)
        controls.append("unrelated-square-nonleaf-refused")
        drop(native,z,y,x)
        empty = tensor(native, (0,), [], True)
        native.error([(129,empty,0)], 5)
        controls.append("empty-grad-leaf-refused")
        drop(native,empty)
        final_live, pages = non_pages(native)
        assert final_live == [0]*8 and val(native,53) == val(parent,53) == 1
    finally:
        native.finish()
        parent.finish()

    result = {"status":"SCOPED_PASS", "source_sha256":SOURCE, "native_sha256":ELF,
              "predecessor_sha256":PARENT, "oracle":custody,
              "candidate_cases":len(rows), "seeded_rounds":2*len(rows),
              "forward_words":sum(r["elements"] for r in rows),
              "reverse_words":2*sum(r["elements"] for r in rows),
              "old_refusal_cases":len(SHAPES), "full_snapshot_controls":controls,
              "native_test_data_commands_excluding_refusals":frames,
              "mixed_owner_descriptor_pages":pages, "final_non_page_live_counts":final_live,
              "rows":rows, "canonical_credit":False}
    (a.output/"qualification.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"rows","oracle"}},sort_keys=True))


if __name__ == "__main__":
    main()
