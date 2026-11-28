"""Exercise spatial and reverse-AD routes in one persistent native session."""
import argparse
import hashlib
import importlib.util
from itertools import product
import json
import math
import os
from pathlib import Path
import struct
import sys

from project import SOURCE

ELF = "970ee71344f863f3d473f3c626d4cbdb2672ea0971506ebaa7f6f2ca4205c76b"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARENT = "41d17a95bdc89b37bd9050c63372a244497c32114b7ef9c2366f5340fc3e380c"
WIRE = "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9"
WORDS = (0, 0x80000000, 1, 0x80000001, 0x3f800000, 0xbf800000,
         0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff, 0x7f812345,
         0x7f812346, 0xff812345, 0x7fc123ab, 0x7fc123de, 0xffc123ab)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def raw(tensor):
    return bytes(tensor.detach().clone(memory_format=__import__("torch").contiguous_format).untyped_storage())


def pack(words):
    return struct.pack("<" + "I" * len(words), *words)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "parent", "source", "wheel", "install", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    assert not sys.flags.optimize
    assert sha(a.binary.read_bytes()) == ELF and sha(a.source.read_bytes()) == SOURCE
    assert sha(a.parent.read_bytes()) == PARENT
    proto = REPO / "tests/integer_tensor_review/wire/protocol.py"
    assert sha(proto.read_bytes()) == WIRE
    w = load("spatial_wire", proto)
    r = load("conv_rounding", HERE.parent / "opt_in_conv2d_bias_rounding" / "qualify.py")
    pair = r.base.previous.pair
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.__version__ == "2.15.0+cpu" and torch.backends.cpu.get_cpu_capability() == "AVX2"
    assert torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled
    oracle = r.custody.verify_torch_custody(a.wheel, a.install, torch)
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(REPO / "tests"))
    from half_logical.native import Memory
    s = w.Session(a.binary.resolve(), directory=a.output / "same-session")
    spatial, ad, special = [], [], []
    frames = 0

    def one(*command):
        nonlocal frames
        frames += 1
        return s.run([command])[0]

    def factory(shape, payload, grad=0):
        return one(1, len(shape), grad, *shape, *([0] * (4-len(shape))), w.Payload(payload,4))["value"]

    def meta(h):
        return w.meta(one(13,h))

    def check(h, want, grad):
        info = meta(h)
        n = want.numel()
        assert info["requires_grad"] == grad
        assert info["rank"] == want.ndim and info["numel"] == n
        assert tuple(info["shape"][:want.ndim]) == want.shape
        if one(39,h,0,n)["payload"] != raw(want):
            raise ValueError(f"wrong native forward words for {tuple(want.shape)}")
        return (info["storage_id"], info["version_id"])

    def drop(*hs):
        for h in hs:
            one(41,h)

    def conv(n, case):
        label, first, last, bias, _, pre, post = case
        xs, ws = (n,2,4,4), (2,2,3,3)
        x = torch.ones(xs,dtype=torch.float32)
        weight = torch.zeros(ws,dtype=torch.float32)
        weight[:,0,0,0] = r.value(first)
        weight[:,1,2,2] = r.value(last)
        bias = torch.full((2,),r.value(bias),dtype=torch.float32)
        p = [factory(tuple(t.shape),raw(t)) for t in (x,weight,bias)]
        owners = [check(h,t,0) for h,t in zip(p,(x,weight,bias),strict=True)]
        native = [one(op,*p,pair((1,1)),0,pair((1,1)),1)["value"] for op in (125,126)]
        with torch.backends.mkldnn.flags(enabled=False):
            slow, slow_name = r.observe(torch,x,weight,bias)
        with torch.backends.mkldnn.flags(enabled=True):
            direct, direct_name = r.observe(torch,x,weight,bias)
        assert slow_name == "aten::_slow_conv2d_forward"
        assert direct_name == ("aten::_slow_conv2d_forward" if n == 1 else "aten::mkldnn_convolution")
        assert raw(slow) == bytes.fromhex(pre) * slow.numel()
        assert one(39,native[0],0,slow.numel())["payload"] == bytes.fromhex(post) * slow.numel()
        assert one(39,native[1],0,slow.numel())["payload"] == raw(slow)
        assert one(39,native[0 if n == 2 else 1],0,slow.numel())["payload"] == raw(direct)
        for h,t,own in zip(p,(x,weight,bias),owners,strict=True):
            assert check(h,t,0) == own
        drop(*p)
        root = native[0 if n==2 else 1]
        q = one(3,16,root)["value"]
        target = torch.square(direct)
        square_owners = check(q,target,0)
        rmeta = meta(root)
        assert square_owners[0] != rmeta["storage_id"] and square_owners[1] != rmeta["version_id"]
        pooled = one(23,23,q,pair((2,2)),pair((2,2)),0,0)["value"]
        eager_pool = torch.nn.functional.avg_pool2d(target,2)
        check(pooled,eager_pool,0)
        drop(pooled,q,*native)
        return {"n":n,"case":label,"words_each_bias_order":slow.numel(),
                "enabled_kernel":direct_name,"slow_word":pre,"post_word":post,
                "square_sha256":sha(raw(target)),"pool_sha256":sha(raw(eager_pool))}

    def grad_chain(shape, axis):
        n = math.prod(shape)
        vals = torch.arange(n,dtype=torch.float32).reshape(shape) / 4 - 1
        leaf = vals.clone().requires_grad_()
        x = factory(shape,raw(leaf),1)
        u = one(129,x,axis)["value"]
        ut = torch.unsqueeze(leaf,axis)
        owners = check(u,ut,1)
        assert owners == check(x,leaf,1)
        assert tuple(meta(u)["strides"][:ut.ndim]) == ut.stride()
        y = one(3,16,u)["value"]
        yt = torch.square(ut)
        assert all(a != b for a,b in zip(check(y,yt,1),owners))
        drop(u)
        result = []
        for sign in (1,-2):
            seed = (torch.arange(n,dtype=torch.float32)+1).reshape(ut.shape) * (sign/8)
            g = factory(tuple(ut.shape),raw(seed))
            yt.backward(seed,retain_graph=True)
            assert one(29,y,g)["value"] == 1
            got = one(11,x)
            assert got["value"] == n and got["payload"] == raw(leaf.grad)
            result.append(sha(got["payload"]))
            drop(g)
        assert owners == check(x,leaf,1)
        drop(y,x)
        return {"shape":shape,"axis":axis,"elements":n,"gradient_rounds":result}

    def exceptional(xs, gs, shape):
        xt = torch.frombuffer(bytearray(pack(xs)),dtype=torch.float32).reshape(shape).requires_grad_()
        gt = torch.frombuffer(bytearray(pack(gs)),dtype=torch.float32).reshape(shape)
        yt = torch.square(xt)
        yt.backward(gt)
        x,g = factory(shape,pack(xs),1),factory(shape,pack(gs))
        y = one(3,16,x)["value"]
        check(y,yt,1)
        assert one(29,y,g)["value"] == 1
        actual = one(11,x)
        assert actual["value"] == len(xs) and actual["payload"] == raw(xt.grad)
        drop(y,g,x)
        return sha(actual["payload"])

    def object_counts(session):
        m = Memory(session.owner()["bytes"])
        # The runtime intentionally retains descriptor pages until process
        # close. Verify that each counted page has its real typed Hold and
        # Storage backing, and exclude these exact identities only.
        page_storages = set()
        page_holds = set()
        for i in range(256):
            addr = m.q(4096+8*i)
            hold = m.q(6144+8*i)
            assert bool(addr) == bool(hold)
            if addr:
                storage = m.row(6,hold)[2]
                row = m.row(2,storage)
                assert addr == (1 << 63) + row[2] and row[3] == row[4] == 172032
                page_storages.add(storage)
                page_holds.add(hold)
        assert len(page_storages) == len(page_holds)
        actual = m.live_counts()
        actual[1] -= len(page_storages)
        actual[5] -= len(page_holds)
        return actual, len(page_holds)

    def release_replay(exe, label):
        trial = w.Session(exe.resolve(), directory=a.output / label)
        report = []
        try:
            prime = trial.run([w.tensor((1,), (0,))])[0]["value"]
            trial.run([(41,prime)])
            good = (1,2,4,4), (2,2,3,3), (2,)
            for op in (22,125,126):
                start = object_counts(trial)[0]
                bits = []
                for _ in range(4):
                    args = tuple(good[:2] if op==22 else good)
                    handles = [trial.run([w.tensor(shape, [0.5]*math.prod(shape))])[0]["value"]
                               for shape in args]
                    cmd = (op,*handles,pair((1,1)),0,pair((1,1)),1)
                    h = trial.run([cmd])[0]["value"]
                    bits.append(trial.run([(39,h,0,8)])[0]["payload"])
                    trial.run([(41,k) for k in (h,*handles)])
                finish = object_counts(trial)[0]
                assert len(set(bits)) == 1
                report.append({"opcode":op,"iterations":4,
                               "owner_count_delta":[b-a for a,b in zip(start,finish,strict=True)],
                               "output_sha256":sha(bits[0])})
            assert trial.run([(53,)])[0]["value"] == 1
        finally:
            trial.finish()
        return report

    def conv_ad():
        x = torch.ones((1,2,4,4),dtype=torch.float32).requires_grad_()
        k = (torch.arange(36,dtype=torch.float32)/8).reshape(2,2,3,3).requires_grad_()
        u,v = factory(tuple(x.shape),raw(x),1),factory(tuple(k.shape),raw(k),1)
        args = (pair((1,1)),0,pair((1,1)),1)
        h = one(22,u,v,*args)["value"]
        eager = torch.nn.functional.conv2d(x,k)
        check(h,eager,1)
        q = torch.full_like(eager,0.25)
        eager.backward(q)
        seed = factory(tuple(q.shape),raw(q))
        assert one(29,h,seed)["value"] == 1
        for actual,expected in ((u,x.grad),(v,k.grad)):
            record = one(11,actual)
            assert record["value"] == expected.numel() and record["payload"] == raw(expected)
        drop(seed,h,u,v)
        return {"forward_sha256":sha(raw(eager)), "input_adjoint_sha256":sha(raw(x.grad)),
                "weight_adjoint_sha256":sha(raw(k.grad))}

    def empty_output_controls():
        dims = ((1,2,2,2),(2,2,3,3),(2,))
        x,k,b = [factory(shape,pack((0,)*math.prod(shape))) for shape in dims]
        args=(pair((1,1)),0,pair((1,1)),1)
        for cmd in ((22,x,k,*args),(125,x,k,b,*args),(126,x,k,b,*args)):
            s.error([cmd],2)  # spatial_extent fails after scratch allocation
        drop(x,k,b)

    def work_cap_controls():
        dims = ((1,128,4,4),(2,128,1,1),(2,))
        x,k,b = [factory(shape,pack((0,)*math.prod(shape))) for shape in dims]
        args=(pair((1,1)),pair((200,200)),pair((1,1)),1)
        for cmd in ((22,x,k,*args),(125,x,k,b,*args),(126,x,k,b,*args)):
            s.error([cmd],3)  # output allocated but work cap forbids reduction
        drop(x,k,b)

    try:
        # Prime the owner table before asserting that each mixed phase has no
        # new live objects. Every group uses this same native process.
        drop(factory((1,),pack((0,))))
        initial = object_counts(s)[0]
        for shape in ((),(3,),(2,3),(1,2,3)):
            for axis in range(-len(shape)-1,len(shape)+1):
                if len(spatial) < len(r.CASES)*2:
                    j=len(spatial)
                    spatial.append(conv(1+j%2,r.CASES[j//2]))
                ad.append(grad_chain(shape,axis))
                assert object_counts(s)[0] == initial
        pairs = tuple(product(WORDS,repeat=2))
        special.append(exceptional(*zip(*pairs,strict=True),(16,16)))
        for x,g in pairs:
            special.append(exceptional((x,),(g,),()))
        assert object_counts(s)[0] == initial
        ad_conv = conv_ad()
        empty_output_controls()
        work_cap_controls()
        assert object_counts(s)[0] == initial
        remaining, pages = object_counts(s)
        assert one(53)["value"] == 1
    finally:
        s.finish()
    before = release_replay(a.parent,"pinned-parent-replay")
    after = release_replay(a.binary,"candidate-replay")
    assert all(x["output_sha256"] == y["output_sha256"] for x,y in zip(before,after,strict=True))
    assert all(x["owner_count_delta"] == [0,4,0,0,0,4,0,0] for x in before)
    assert all(x["owner_count_delta"] == [0]*8 for x in after)
    result = {"status":"SCOPED_PASS","candidate_elf_sha256":ELF,
              "source_sha256":SOURCE,"oracle":oracle,"native_data_frames":frames,
              "single_native_session":True,"spatial_cases":spatial,"ad_cases":ad,
              "exceptional_gradient_sha256":special,"exceptional_words_each":512,
              "conv_bias_free_ad":ad_conv,"final_non_page_live_counts":remaining,"descriptor_pages_retained":pages,"empty_output_full_owner_snapshot_guard_count":3,"permitted_output_work_cap_full_owner_snapshot_guard_count":3,
              "pinned_parent_replay":before,"candidate_replay":after,
              "canonical_credit":False}
    (a.output/"qualification.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"frames":frames,"spatial":len(spatial),"ad":len(ad),"exceptional":len(special)}))


if __name__ == "__main__":
    main()
