"""Compare owned conv/ReLU/average pooling chains to pinned eager Torch."""
import argparse
import json
from pathlib import Path
import tempfile

# Import the earlier qualifier before Torch to establish the same CPU environment.
from qualify import make_values, sha, values, verify_torch_custody
import adaptive_pool as pool
import project_pool as projection

PUBLISHED_BASE = "951eab0503ba8e2d76f14567946894729529765a"
FIXTURES = (
    ((1, 1, 5, 5), (1, 1, 2, 2), (1, 1), (0, 0), (1, 1), 1, (2, 2)),
    ((2, 2, 5, 7), (3, 2, 2, 2), (1, 1), (0, 0), (1, 1), 1, (2, 2)),
    ((1, 4, 6, 6), (4, 2, 3, 3), (1, 1), (0, 0), (1, 1), 2, (2, 2)),
    ((1, 2, 8, 8), (2, 1, 2, 2), (2, 2), (1, 1), (2, 2), 2, (2, 2)),
)


def pair(value):
    return (value[0] << 32) | value[1]


def require(ok, context):
    if not ok:
        raise ValueError(context)


def full_read(session, handle, expected, label):
    auth, payload = pool.read_tensor(session, handle, tuple(expected.shape), expected.requires_grad)
    require(payload == values(expected), f"{label}: actual native/eager bits differ")
    return auth, payload


def distinct(owners):
    return all(len({item[i] for item in owners}) == len(owners) for i in range(3))


def run_fixture(binary, spec, index, torch, mode):
    from torch.nn import functional as functional
    xs, ws, stride, padding, dilation, groups, pooling = spec
    x_raw, w_raw = make_values(xs, index+3), make_values(ws, index+9)
    tracking = mode != "plain"
    x = torch.frombuffer(bytearray(x_raw), dtype=torch.float32).reshape(xs).requires_grad_(tracking)
    w = torch.frombuffer(bytearray(w_raw), dtype=torch.float32).reshape(ws).requires_grad_(tracking)
    c = functional.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups)
    r = torch.relu(c)
    if mode == "detach":
        feed = r.detach()
        require(feed.data_ptr() == r.data_ptr() and not feed.requires_grad, "eager detach unexpectedly copied")
    else:
        feed = r
    y = functional.avg_pool2d(feed, pooling, pooling)
    with pool.Session(binary) as s:
        hx = pool.factory(s, xs, x_raw, tracking)
        hw = pool.factory(s, ws, w_raw, tracking)
        ox, ow = full_read(s, hx, x, "input"), full_read(s, hw, w, "weight")
        require(distinct([ox[0],ow[0]]), "root alias")
        try:
            hc, _ = s.call(22, hx, hw, pair(stride), pair(padding), pair(dilation), groups)
        except pool.NativeStatus as error:
            require(mode == "plain" and (error.opcode, error.status) == (22,1), "unexpected conv2d refusal")
            require(full_read(s,hx,x,"refusal root") == ox and full_read(s,hw,w,"refusal weight") == ow,
                    "refusal modified a root")
            s.call(41,hw)
            s.call(41,hx)
            require(s.call(53)[0] == 1, "refusal owner audit failed")
            return dict(index=index, mode=mode, status="HELD_CONV2D", refusal=[22,1],
                        later_native_stages=0, frames=s.frames)
        require(mode != "plain", "plain no-grad convolution was not refused; update frozen gate")
        oc = full_read(s, hc, c, "convolution")
        hr, _ = s.call(3, 5, hc)
        orelu = full_read(s, hr, r, "relu")
        hfeed = hr
        if mode == "detach":
            hfeed, _ = s.call(19, hr)
            od = full_read(s, hfeed, feed, "detached activation")
            require(od[0][0] != hr and od[0][1:] == orelu[0][1:], "detach did not share Storage/Version")
        hy, _ = s.call(23, 23, hfeed, pair(pooling), pair(pooling), 0, 0)
        oy = full_read(s, hy, y, "average pooling")
        require(distinct([z[0] for z in (ox,ow,oc,orelu,oy)]), "new stage owner aliases prior owner")

        gradient_receipt = {}
        if mode == "backward":
            seed_raw = make_values(y.shape, index + 19)
            seed = torch.frombuffer(bytearray(seed_raw), dtype=torch.float32).reshape(y.shape)
            y.backward(seed)
            hs = pool.factory(s, tuple(seed.shape), seed_raw, False)
            os = full_read(s, hs, seed, "seed")
            require(distinct([z[0] for z in (ox,ow,oc,orelu,oy,os)]), "seed aliases another owner")
            s.call(29,hy,hs)
            for name, handle, expected in (("x",hx,x.grad),("w",hw,w.grad)):
                count, raw = s.call(11, handle)
                eager = values(expected)
                require(count == expected.numel() and raw == eager, f"{name} gradient mismatch")
                gradient_receipt[name] = dict(words=count, sha256=sha(raw))
            require(full_read(s,hs,seed,"retained seed") == os, "backward modified seed")
            s.call(41, hs)

        require(full_read(s,hx,x,"retained input") == ox and full_read(s,hw,w,"retained weight") == ow,
                "input versions or bytes changed")
        require(full_read(s,hc,c,"retained conv") == oc and full_read(s,hr,r,"retained relu") == orelu,
                "intermediate versions or bytes changed")
        if hfeed != hr:
            s.call(41, hfeed)
        release = [hx,hw,hc,hr]
        for handle in release if index % 2 == 0 else reversed(release):
            s.call(41, handle)
        require(full_read(s,hy,y,"surviving output") == oy, "output modified after releasing intermediates")
        s.call(41,hy)
        require(s.call(53)[0] == 1, "owner audit failed after release")
        return dict(index=index, mode=mode, status="PASS_SCOPED", frames=s.frames,
                    geometry=dict(input=xs, weight=ws, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, pooling=pooling),
                    stages={name: dict(shape=tuple(expected.shape), words=len(observation[1])//4,
                                       sha256=sha(observation[1])) for name,expected,observation in
                            (("conv",c,oc),("relu",r,orelu),("pool",y,oy))},
                    relu_positive=int((r>0).sum().item()), relu_elements=r.numel(),
                    leaf_gradients=gradient_receipt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for field in ("binary", "source", "wheel", "install"):
        ap.add_argument("--" + field, required=True, type=Path)
    args=ap.parse_args()
    require(verify_source(args.source), "wrong exact #75 source")
    import torch
    require(torch.__version__ == "2.15.0+cpu", "expected pinned Torch 2.15.0+cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.backends.cpu.get_cpu_capability()=="AVX2", "expected AVX2")
    custody=verify_torch_custody(args.wheel,args.install,torch)
    with tempfile.TemporaryDirectory(prefix="heptatorch-spatial-chain-") as tmp:
        binary=pool.stage_executable(tmp,pool.pinned_bytes(args.binary,pool.ELF_SHA,pool.ELF_CAP))
        rows=[run_fixture(binary,spec,index,torch,mode)
              for index,spec in enumerate(FIXTURES) for mode in ("plain","detach","backward")]
    print(json.dumps(dict(schema="heptatorch.opt-in-spatial-chain.v1",
                          status="PASS_SCOPED_WITH_PLAIN_NO_GRAD_HELD",
                          published_source_base=PUBLISHED_BASE, native_source_sha256=projection.RESULT,
                          native_elf_sha256=pool.ELF_SHA, reference_custody=custody, torch_cpu="AVX2",
                          candidate_frames=sum(row["frames"] for row in rows), cases=rows,
                          production_qualification=False, full_pytorch_rewrite_complete=False),
                     sort_keys=True,indent=2))


def verify_source(path):
    return sha(path.read_bytes()) == projection.RESULT


if __name__ == "__main__":
    main()
