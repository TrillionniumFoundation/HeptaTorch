"""Compare the restricted native CPU selector to actual locked Torch dispatch."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

_FIRST = Path(__file__).resolve().parents[1] / "opt_in_conv2d_bias_first"
sys.path.insert(0, str(_FIRST))
import qualify as previous
sys.path.pop(0)
rounding, base, wire, custody = previous.rounding, previous.base, previous.wire, previous.custody
ELF = "e0ef7fb173c69c0c3d68bfb3d9e0b931b2a6afd6507602e27f6f6528fedc7e94"
SOURCE = "4cd14a332afc3cad5c67f9f12412591985021fa51af84836a558242e4339f8db"


def measure(exe, geometry, raws, output, *, op=127, **kw):
    original = base.OP
    try:
        base.OP = op
        return base.measure(exe, geometry, raws, output, **kw)
    finally:
        base.OP = original


def compare(binary,parent,wheel,install):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    wire.need(torch.__version__=="2.15.0+cpu" and torch.backends.cpu.get_cpu_capability()=="AVX2",
              "requires locked Torch reference")
    wire.need(torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled,
              "requires default enabled oneDNN availability")
    binding = torch.backends.mkldnn.enabled
    reference = custody.verify_torch_custody(wheel, install, torch)
    with tempfile.TemporaryDirectory(prefix="heptatorch-cpu-route-") as tmp:
        path = Path(tmp)
        qa, qb = path/"candidate", path/"parent"
        qa.mkdir()
        qb.mkdir()
        native = wire.stage_executable(qa, wire.pinned_bytes(binary,ELF,wire.ELF_CAP))
        explicit = wire.stage_executable(qb, wire.pinned_bytes(parent,previous.ELF,wire.ELF_CAP))
        rows, controls, old = [], [], []
        for n in (1,2):
            xs, ws, stride, pad, dil, groups = ((n,2,4,4),(2,2,3,3),(1,1),(0,0),(1,1),1)
            shape = (n,2,2,2)
            cases = []
            for label, first, last, bval, _, preword, postword in rounding.CASES:
                x = torch.ones(xs, dtype=torch.float32)
                w = torch.zeros(ws, dtype=torch.float32)
                w[:,0,0,0] = rounding.value(first)
                w[:,1,2,2] = rounding.value(last)
                b = torch.full((2,), rounding.value(bval), dtype=torch.float32)
                cases.append((label,x,w,b,preword,postword))
            for seed in range(3):
                tensors = [torch.frombuffer(bytearray(base.previous.pack_values(s,3*seed+j)),
                                            dtype=torch.float32).reshape(s)
                           for j,s in enumerate((xs,ws,(2,)))]
                cases.append((f"dense-dyadic-{seed}",*tensors,None,None))
            for label,x,w,b,pre,post in cases:
                raw = tuple(custody.values(z) for z in (x,w,b))
                fixed = (xs,ws,stride,pad,dil,groups)
                pre_out = measure(native,fixed,raw,shape,op=126)
                post_out = measure(native,fixed,raw,shape,op=125)
                for opcode,acc in ((126,pre_out),(125,post_out)):
                    wire.need(measure(explicit,fixed,raw,shape,op=opcode)==acc,
                              "new command changed explicit parent route")
                if pre is not None:
                    wire.need(pre_out["output"]==bytes.fromhex(pre)*(n*8) and
                              post_out["output"]==bytes.fromhex(post)*(n*8), "old discriminator changed")
                for enabled in (False,True):
                    fixed = (xs,ws,stride,pad,dil,int(enabled))
                    try:
                        torch.backends.mkldnn.enabled=enabled
                        direct, family = rounding.observe(torch,x,w,b)
                    finally:
                        torch.backends.mkldnn.enabled=binding
                    expect_post = n==2 and enabled
                    expected_family = "aten::mkldnn_convolution" if expect_post else "aten::_slow_conv2d_forward"
                    wire.need(family==expected_family, f"observed kernel {family}, expected {expected_family}")
                    auto = measure(native,fixed,raw,shape)
                    oracle = custody.values(direct)
                    accepted = post_out if expect_post else pre_out
                    rejected = pre_out if expect_post else post_out
                    wire.need(auto["output"]==oracle==accepted["output"], "auto output not bit-exact")
                    wire.need(tuple(custody.values(z) for z in (x,w,b)) == raw, "oracle mutated inputs")
                    rows.append(dict(case=label,batch=n,mkldnn_enabled=enabled,kernel=family,
                                     explicit_opcode=125 if expect_post else 126,words=n*8,
                                     output_sha256=custody.sha(oracle),native_frames=auto["native_frames"],
                                     unselected_would_disagree=oracle!=rejected["output"]))
            data = tuple(base.previous.pack_values(s,4) for s in (xs,ws,(2,)))
            old.append(measure(explicit,(xs,ws,stride,pad,dil,1),data,None,refusal=4))
        x,w,b=(1,2,4,4),(2,2,3,3),(2,)
        geom=(x,w,(1,1),(0,0),(1,1),1)
        variants=[
            ("flag-2",(*geom[:5],2),b,{},10),
            ("flag-max",(*geom[:5],2**64-1),b,{},10),
            ("stride",(x,w,(2,1),(0,0),(1,1),1),b,{},10),
            ("padding",(x,w,(1,1),(1,0),(1,1),1),b,{},10),
            ("dilation",(x,w,(1,1),(0,0),(2,1),1),b,{},10),
            ("batch-3",((3,*x[1:]),*geom[1:]),b,{},10),
            ("input-rank",((2,4,4),*geom[1:]),b,{},10),
            ("weight-height",(x,(2,2,2,3),*geom[2:]),b,{},10),
            ("bias-length",geom,(3,),{},10),
            ("bias-rank",geom,(1,2),{},10),
            ("missing-owner",geom,b,{"handle":0},1),
        ]
        for batch,enabled in ((1,False),(2,True)):
            tested=((batch,*x[1:]),*geom[1:5],int(enabled))
            for i, which in enumerate(("input","weight","bias")):
                variants.append((f"grad-{batch}-{which}",tested,b,
                                 {"grad":tuple(j==i for j in range(3))},7))
        for name, g, bs, extra, refusal in variants:
            data = tuple(base.previous.pack_values(s,4) for s in (*g[:2],bs))
            controls.append(dict(case=name,**measure(native,g,data,None,bias_shape=bs,
                                                     refusal=refusal,**extra)))
    return dict(schema="heptatorch.native-conv2d-cpu-bias-order-route.v1",status="PASS_SCOPED",
                source_sha256=SOURCE,native_sha256=ELF,parent_sha256=previous.ELF,new_opcode=127,
                reference=dict(torch=str(torch.__version__),cpu="AVX2",threads=1,**reference),
                rows=rows,output_words=sum(r["words"] for r in rows),
                route_discriminators=sum(r["unselected_would_disagree"] for r in rows),
                negative_controls=controls,exact_explicit_parent_refusals=old,
                reduction_order_generalized=False,full_pytorch_qualification=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","parent","wheel","install"):
        p.add_argument("--"+name,type=Path,required=True)
    print(json.dumps(compare(**vars(p.parse_args())),sort_keys=True,indent=2))
