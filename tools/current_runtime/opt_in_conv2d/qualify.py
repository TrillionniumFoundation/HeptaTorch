"""Check native wire conv2d and existing AD against the pinned Torch wheel."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_POOL = Path(__file__).resolve().parents[1] / "opt_in_adaptive_avg_pool2d"
sys.path.insert(0, str(_POOL))
import adaptive_pool as wire

_custodian_spec = importlib.util.spec_from_file_location("spatial_oracle_custody", _POOL / "qualify.py")
_custodian = importlib.util.module_from_spec(_custodian_spec)
_custodian_spec.loader.exec_module(_custodian)
sys.path.pop(0)

ELF_SHA = "37b2c11cdf50fb136dab2273063a75489ef05ef7c4fd582725f7e9550382a464"
# input, weight (no bias), stride, padding, dilation, groups.
FIXTURES = (
    ((1, 1, 1, 1), (1, 1, 1, 1), (1, 1), (0, 0), (1, 1), 1),
    ((1, 1, 3, 3), (1, 1, 2, 2), (1, 1), (0, 0), (1, 1), 1),
    ((1, 2, 5, 6), (3, 2, 3, 2), (2, 1), (1, 0), (1, 1), 1),
    ((2, 2, 5, 7), (2, 2, 3, 2), (1, 2), (0, 1), (1, 2), 1),
    ((2, 4, 7, 9), (6, 2, 2, 3), (2, 1), (1, 2), (2, 2), 2),
    ((1, 3, 4, 5), (3, 1, 3, 3), (1, 1), (1, 1), (1, 1), 3),
    ((1, 4, 4, 5), (8, 1, 2, 3), (2, 2), (2, 1), (1, 1), 4),
    ((2, 3, 6, 7), (5, 3, 1, 1), (2, 3), (0, 0), (1, 1), 1),
)
GRADIENTS = ((False, False), (True, False), (False, True), (True, True))


def pair(vals):
    return vals[0] << 32 | vals[1]


def pack_values(shape, offset):
    count = math.prod(shape)
    # Dyadic operands and reductions smaller than the exact F32 precision bound.
    return struct.pack(f"<{count}f", *(((i * 13 + offset) % 29 - 14) / 16 for i in range(count)))


def run_native(exe, geometry, raws, gradients, seed, wanted, *, refusal=None):
    xs, ws, stride, pad, dilation, groups = geometry
    with wire.Session(exe) as s:
        handles = [wire.factory(s, shape, raw, grad) for shape, raw, grad in zip((xs, ws), raws, gradients)]
        snapshots = [wire.read_tensor(s, h, shape, grad) for h, shape, grad in zip(handles, (xs, ws), gradients)]
        wire.need([snap[1] for snap in snapshots] == list(raws), "input constructor changed values")
        try:
            y, _ = s.call(22, *handles, pair(stride), pair(pad), pair(dilation), groups)
        except wire.NativeStatus as exc:
            if (exc.status, exc.opcode) != (refusal, 22):
                raise
            wire.need([wire.read_tensor(s, h, shape, grad)
                       for h, shape, grad in zip(handles, (xs, ws), gradients)] == snapshots,
                      "refusal changed existing inputs")
            for h in handles:
                s.call(41, h)
            wire.need(s.call(53)[0] == 1, "refusal damaged owner invariants")
            return {"status": exc.status, "frames": s.frames}
        wire.need(refusal is None, "expected native refusal did not occur")
        output = wire.read_tensor(s, y, wanted, any(gradients))
        live = [i[0] for i in snapshots] + [output[0]]
        seed_handle = None
        pullbacks = [None, None]
        if any(gradients):
            seed_handle = wire.factory(s, wanted, seed, False)
            before_seed = wire.read_tensor(s, seed_handle, wanted, False)
            wire.need(before_seed[1] == seed, "seed constructor changed values")
            live.append(before_seed[0])
            s.call(29, y, seed_handle)
            for i, (h, shape, want) in enumerate(zip(handles, (xs, ws), gradients)):
                if want:
                    n, buf = s.call(11, h)
                    wire.need(n == math.prod(shape) and len(buf) == 4*n, "gradient extent mismatch")
                    pullbacks[i] = buf
            wire.need(wire.read_tensor(s, seed_handle, wanted, False) == before_seed, "seed changed")
            s.call(41, seed_handle)
        wire.need(all(len(set(k)) == len(live) for k in zip(*live)), "live tensors share output ownership")
        for h, shape, grad, saved in zip(handles, (xs, ws), gradients, snapshots):
            wire.need(wire.read_tensor(s, h, shape, grad) == saved, "operation changed input")
            s.call(41, h)
        wire.need(wire.read_tensor(s, y, wanted, any(gradients)) == output, "output invalid after input release")
        s.call(41, y)
        wire.need(s.call(53)[0] == 1, "owner invariant audit failed")
        return {"status": 0, "frames": s.frames, "output": output[1], "pullbacks": pullbacks}



def run_no_grad_chain(exe, torch):
    # A single actual wire session: no detach, autograd, or host-computed values.
    xs, ws = (1, 1, 5, 5), (2, 1, 2, 2)
    raw_x, raw_w = pack_values(xs, 13), pack_values(ws, 7)
    x = torch.frombuffer(bytearray(raw_x), dtype=torch.float32).reshape(xs)
    w = torch.frombuffer(bytearray(raw_w), dtype=torch.float32).reshape(ws)
    conv = torch.nn.functional.conv2d(x, w)
    relu = torch.relu(conv)
    pooled = torch.nn.functional.avg_pool2d(relu, 2)
    wire.need(bool((conv < 0).any()) and bool((conv > 0).any()), "degenerate ReLU control")
    with wire.Session(exe) as s:
        hx, hw = wire.factory(s, xs, raw_x, False), wire.factory(s, ws, raw_w, False)
        ox, ow = wire.read_tensor(s, hx, xs, False), wire.read_tensor(s, hw, ws, False)
        hc, _ = s.call(22, hx, hw, pair((1, 1)), 0, pair((1, 1)), 1)
        hr, _ = s.call(3, 5, hc)
        hy, _ = s.call(23, 23, hr, pair((2, 2)), pair((2, 2)), 0, 0)
        stages = [(hc, conv), (hr, relu), (hy, pooled)]
        actual = [wire.read_tensor(s, handle, tuple(expected.shape), False)
                  for handle, expected in stages]
        wire.need(all(result[1] == _custodian.values(expected)
                      for (_, expected), result in zip(stages, actual)), "chain-stage Torch mismatch")
        wire.need(all(len(set(fields)) == 5 for fields in zip(ox[0], ow[0], *(item[0] for item in actual))),
                  "chain output aliases another live owner")
        wire.need(wire.read_tensor(s, hx, xs, False) == ox and wire.read_tensor(s, hw, ws, False) == ow,
                  "chain inputs modified")
        for handle in (hx, hw, hc, hr):
            s.call(41, handle)
        wire.need(wire.read_tensor(s, hy, tuple(pooled.shape), False) == actual[-1],
                  "pool result changed after prior owners released")
        s.call(41, hy)
        wire.need(s.call(53)[0] == 1, "chain owner invariant audit failed")
        return {"frames": s.frames, "operations": [22,3,23], "input_sha256": [sha(raw_x), sha(raw_w)],
                "stage_sha256": [sha(result[1]) for result in actual]}


def sha(buf):
    return None if buf is None else _custodian.sha(buf)


def run(binary, parent, wheel, install):
    import torch
    if torch.__version__ != "2.15.0+cpu":
        raise RuntimeError("requires locked Torch 2.15.0+cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise RuntimeError("requires locked AVX2 eager path")
    custody = _custodian.verify_torch_custody(wheel, install, torch)
    rows = []
    with tempfile.TemporaryDirectory(prefix="heptatorch-conv-") as root:
        cpath, ppath = (Path(root)/k for k in ("candidate", "parent"))
        cpath.mkdir()
        ppath.mkdir()
        candidate = wire.stage_executable(cpath, wire.pinned_bytes(binary, ELF_SHA, wire.ELF_CAP))
        comparison = wire.stage_executable(ppath, wire.pinned_bytes(parent, wire.ELF_SHA, wire.ELF_CAP))
        for i, geometry in enumerate(FIXTURES):
            xs, ws, stride, pad, dilation, groups = geometry
            raws = (pack_values(xs, i+4), pack_values(ws, i+11))
            for gradients in GRADIENTS:
                x, w = (torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).requires_grad_(grad)
                        for raw, shape, grad in zip(raws, (xs,ws), gradients))
                y = torch.nn.functional.conv2d(x, w, None, stride, pad, dilation, groups)
                wanted = tuple(y.shape)
                seed = pack_values(wanted, i+20)
                wire.need(y.stride() == tuple(wire.contiguous_strides(wanted)), "eager output not contiguous")
                wire.need(y.data_ptr() not in (x.data_ptr(), w.data_ptr()), "eager result unexpectedly aliases")
                if any(gradients):
                    g = torch.frombuffer(bytearray(seed), dtype=torch.float32).reshape(wanted)
                    y.backward(g)
                    wire.need(_custodian.values(g) == seed, "Torch changed gradient seed")
                wire.need((_custodian.values(x), _custodian.values(w)) == raws, "Torch changed inputs")
                eager = (_custodian.values(y), [_custodian.values(z.grad) if z.grad is not None else None for z in (x,w)])
                candidate_row = run_native(candidate, geometry, raws, gradients, seed, wanted)
                legacy_row = run_native(comparison, geometry, raws, gradients, seed, wanted,
                                        refusal=None if any(gradients) else 1)
                got = candidate_row["output"], candidate_row["pullbacks"]
                wire.need(got == eager, f"native/Torch bits differ on case {i}, grad={gradients}")
                if any(gradients):
                    wire.need((legacy_row["output"], legacy_row["pullbacks"]) == got,
                              "existing native backward behavior changed")
                rows.append({
                    "geometry": geometry, "gradients": gradients, "shape": wanted,
                    "input_sha256": list(map(sha, raws)),
                    "seed_sha256": sha(seed) if any(gradients) else None,
                    "forward_sha256": sha(got[0]), "backward_sha256": list(map(sha, got[1])),
                    "native_frames": candidate_row["frames"], "parent_frames": legacy_row["frames"],
                    "parent_status": legacy_row["status"],
                })
        chain = run_no_grad_chain(candidate, torch)
        # Ordinary nonsensical channel geometry stays refused before calculation.
        fixture = ((1,2,4,4), (1,1,1,1), (1,1), (0,0), (1,1), 1)
        bad_rows = [run_native(exe, fixture, tuple(pack_values(x, 4) for x in fixture[:2]),
                               (False,False), None, None, refusal=2) for exe in (candidate, comparison)]
    return {
        "schema":"heptatorch.opt-in-conv2d-qualification.v1", "status":"PASS",
        "source_sha256":"115d01ac9377070f6841d619c1d9adf4a52cba809f676eb8b1522adefe88fcf1",
        "native_sha256": ELF_SHA, "parent_sha256": wire.ELF_SHA,
        "reference": {"torch":str(torch.__version__), "cpu":"AVX2", "threads":1, **custody},
        "cases": rows, "channel_refusals": bad_rows, "no_grad_chain": chain, "full_pytorch_rewrite_complete": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "parent", "wheel", "install"):
        p.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(run(**vars(p.parse_args())), sort_keys=True, indent=2))
