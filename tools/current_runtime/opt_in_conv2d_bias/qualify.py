"""Qualify the bounded native conv2d-with-bias operation against locked Torch."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile

_PARENT = Path(__file__).resolve().parents[1] / "opt_in_conv2d" / "qualify.py"
_spec = importlib.util.spec_from_file_location("conv_no_grad_qualification", _PARENT)
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
wire, custody = previous.wire, previous._custodian
ELF = "5591987579976a9f0df1b5381eccc295b92dab324bbd6c33e2a2071b9b6714b1"
SRC = "96ccc2c837b25a07acb2b66b089d4f55889c40a84c50dbf765eb887f6da9aee2"
OP = 125


def factory(s, shape, raw, grad=False):
    dims = (*shape, *((0,) * (4-len(shape))))
    config = struct.pack("<8Q", len(shape), int(grad), *dims, 160, len(raw))
    return s.exchange(wire.request((97, 0, 96), config + raw), 97)[0]


def read(s, h, shape, grad=False):
    rank, n, g, dims, strides, storage, vid, ver, off, size = wire.unpack_meta(s.call(13, h)[1])
    wire.need((rank, n, g, dims, strides) ==
              (len(shape), math.prod(shape), int(grad), list(shape), wire.contiguous_strides(shape)),
              "shape, stride, or autograd flag changed")
    wire.need(storage and vid and (ver, off, size)==(0,0,n*4), "not fresh physical F32 storage")
    elems, data = s.call(98, h, 0, n)
    wire.need(elems == n and len(data) == size, "wrong physical buffer extent")
    return ((h,storage,vid), data)


def measure(exe, fixture, data, wanted, *, grad=None, refusal=None, bias_shape=None, handle=None,
            chain=None):
    xs, ws, stride, pad, dilation, groups = fixture
    bs = (ws[0],) if bias_shape is None else bias_shape
    grad = (False,False,False) if grad is None else grad
    with wire.Session(exe) as s:
        shapes = (xs,ws,bs)
        inputs = [factory(s, shape, raw, g) for shape, raw, g in zip(shapes,data,grad)]
        snapshots = [read(s,h,shape,g) for h,shape,g in zip(inputs,shapes,grad)]
        wire.need(tuple(x[1] for x in snapshots) == tuple(data), "input construction mismatch")
        bias = inputs[2] if handle is None else handle
        try:
            out, _ = s.call(OP, *inputs[:2], bias, previous.pair(stride), previous.pair(pad),
                            previous.pair(dilation), groups)
        except wire.NativeStatus as e:
            wire.need((e.status,e.opcode)==(refusal,OP), f"expected status {refusal}: {e}")
            for h, shape, g, snapshot in zip(inputs, shapes, grad, snapshots):
                wire.need(read(s,h,shape,g)==snapshot, "refusal mutated input")
                s.call(41,h)
            wire.need(s.call(53)[0] == 1,"refusal ownership failed")
            return dict(native_frames=s.frames,status=e.status)
        wire.need(refusal is None, "missing native refusal")
        display = read(s,out,wanted)
        wire.need(all(len(set(ls)) == 4 for ls in zip(*(e[0] for e in (*snapshots,display)))),
                  "result aliases another live data or version owner")
        chained = None
        if chain is not None:
            relu, _ = s.call(3,5,out)
            pooled, _ = s.call(23,23,relu,previous.pair((2,2)),previous.pair((2,2)),0,0)
            observed = (read(s,relu,wanted),read(s,pooled,chain[1]))
            wire.need(tuple(x[1] for x in observed) == (chain[0],chain[2]), "downstream stage mismatch")
            wire.need(all(len(set(ls)) == 6 for ls in zip(*(e[0] for e in (*snapshots,display,*observed)))),
                      "downstream stage aliases another live owner")
            for h in (pooled,relu):
                s.call(41,h)
            chained = [custody.sha(x[1]) for x in observed]
        for h, shape, g, snapshot in zip(inputs, shapes, grad, snapshots):
            wire.need(read(s,h,shape,g)==snapshot, "operation mutated input")
            s.call(41,h)
        wire.need(read(s,out,wanted)==display, "input release affected result")
        s.call(41,out)
        wire.need(s.call(53)[0] == 1,"success ownership failed")
        return dict(native_frames=s.frames,status=0,output=display[1],chain_sha256=chained)


def run(binary, parent, wheel, install):
    import torch
    if torch.__version__!="2.15.0+cpu" or torch.backends.cpu.get_cpu_capability()!="AVX2":
        raise RuntimeError("requires pinned Torch 2.15.0+cpu AVX2")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = custody.verify_torch_custody(wheel,install,torch)
    fixtures = previous.FIXTURES + (
        ((1,2,3,4),(4,1,1,1),(2,2),(2,2),(1,1),2),
        ((2,2,3,3),(2,2,3,3),(1,1),(2,2),(2,2),1),
    )
    cases, guards, preserved = [], [], []
    with tempfile.TemporaryDirectory(prefix="heptatorch-conv-bias-") as tmp:
        p,c=Path(tmp)/"parent",Path(tmp)/"candidate"
        p.mkdir()
        c.mkdir()
        old=wire.stage_executable(p,wire.pinned_bytes(parent,previous.ELF_SHA,wire.ELF_CAP))
        new=wire.stage_executable(c,wire.pinned_bytes(binary,ELF,wire.ELF_CAP))
        for i, fixture in enumerate(fixtures):
            xs,ws,stride,pad,dilation,groups=fixture
            for j in range(3):
                shapes=(xs,ws,(ws[0],))
                data=tuple(previous.pack_values(shape,7*i+11*j+k+3)
                           for k,shape in enumerate(shapes))
                x,w,b=(torch.frombuffer(bytearray(raw),dtype=torch.float32).reshape(shape)
                       for raw,shape in zip(data,shapes))
                eager=torch.nn.functional.conv2d(x,w,b,stride,pad,dilation,groups)
                none=torch.nn.functional.conv2d(x,w,None,stride,pad,dilation,groups)
                want=tuple(eager.shape)
                # Include a real all-padding region, plus grouped/depthwise and dilation.
                if i == 8:
                    wire.need(torch.equal(eager[:,:,0,0],b.unsqueeze(0)),"padded region was not bias-only")
                chain = None
                if i == 5 and j == 0:
                    relu = eager.relu()
                    pooled = torch.nn.functional.avg_pool2d(relu, 2)
                    chain = custody.values(relu),tuple(pooled.shape),custody.values(pooled)
                actual=measure(new,fixture,data,want,chain=chain)
                wire.need(actual.pop("output")==custody.values(eager),f"Torch bits differ ({i},{j})")
                wire.need(tuple(custody.values(t) for t in (x,w,b))==data,"eager input changed")
                wire.need(eager.data_ptr() not in {t.data_ptr() for t in (x,w,b)},"eager alias")
                old_reply=measure(old,fixture,data,want,refusal=4)
                cases.append(dict(geometry=fixture,values_sha256=[custody.sha(raw) for raw in data],
                                  result_sha256=custody.sha(custody.values(eager)),shape=want,
                                  forward_words=eager.numel(),bias_changes_output=not torch.equal(eager,none),
                                  previous=old_reply,**actual))
            # Re-run all four bias-free modes for each original geometry; include the real seeded wire AD.
            if i<len(previous.FIXTURES):
                for req in previous.GRADIENTS:
                    seed=previous.pack_values(want,i+20)
                    new_row=previous.run_native(new,fixture,data[:2],req,seed,want)
                    old_row=previous.run_native(old,fixture,data[:2],req,seed,want)
                    wire.need(new_row==old_row,"existing no-bias #78 path changed")
                    preserved.append(dict(geometry=fixture,requires_grad=req,native_frames=new_row["frames"],
                                          output_sha256=custody.sha(new_row["output"]),
                                          gradients_sha256=[previous.sha(g) for g in new_row["pullbacks"]]))
        f=fixtures[3]
        xs,ws=f[:2]
        source=(previous.pack_values(xs,4),previous.pack_values(ws,5))
        b=(ws[0],)
        for label,opts,shape,status in [
            ("bias-length",{},(b[0]+1,),2),
            ("bias-rank",{},(1,b[0]),2),
            ("input-grad",dict(grad=(True,False,False)),b,7),
            ("weight-grad",dict(grad=(False,True,False)),b,7),
            ("bias-grad",dict(grad=(False,False,True)),b,7),
            ("null-bias",dict(handle=0),b,1),
            ("unknown-bias",dict(handle=2**63+3),b,1),
        ]:
            data=(*source,previous.pack_values(shape,6))
            guards.append(dict(case=label,**measure(new,f,data,None,bias_shape=shape,refusal=status,**opts)))
    wire.need(any(row["bias_changes_output"] for row in cases),"degenerate bias fixtures")
    return dict(schema="heptatorch.opt-in-conv2d-f32-bias.v1",status="PASS_SCOPED",operator=OP,
                native_sha256=ELF,parent_sha256=previous.ELF_SHA,source_sha256=SRC,
                reference=dict(torch=str(torch.__version__), threads=torch.get_num_threads(),cpu="AVX2",**reference),
                forward_cases=cases,negative_controls=guards,preserved_bias_free_cases=preserved,
                whole_pytorch_qualification=False,production_qualification=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("binary","parent","wheel","install"):
        parser.add_argument("--"+arg,required=True,type=Path)
    print(json.dumps(run(**vars(parser.parse_args())),sort_keys=True,indent=2))
