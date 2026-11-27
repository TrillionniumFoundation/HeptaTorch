"""Compare three real seeded HIA pullbacks with pinned Torch eager."""
import argparse
import importlib.util
import json
from itertools import product
from pathlib import Path
import lower

_previous=Path(__file__).resolve().parents[1]/"opt_in_conv2d"/"qualify.py"
_spec=importlib.util.spec_from_file_location("conv_no_grad_qualification",_previous)
previous=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
custody=previous._custodian


def run(binary,wheel,install):
    import torch
    if torch.__version__!="2.15.0+cpu" or torch.backends.cpu.get_cpu_capability()!="AVX2":
        raise RuntimeError("requires exact locked Torch and AVX2 eager")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference=custody.verify_torch_custody(wheel,install,torch)
    rows=[]
    accumulated=[]
    for i,fixture in enumerate(previous.FIXTURES):
        xs,ws,st,pad,dil,grp=fixture
        options=dict(stride=st,padding=pad,dilation=dil,groups=grp)
        sizes=(xs,ws,(ws[0],))
        raws=tuple(previous.pack_values(shape,7*i+11*j+3) for j,shape in enumerate(sizes))
        target=lower.geometry(xs,ws,st,pad,dil,grp)
        seed=previous.pack_values(target,i+21)
        for req in product((False,True),repeat=3):
            if not any(req):
                continue
            leaves=[torch.frombuffer(bytearray(raw),dtype=torch.float32).reshape(shape).requires_grad_(g)
                    for raw,shape,g in zip(raws,sizes,req)]
            y=torch.nn.functional.conv2d(*leaves,st,pad,dil,grp)
            probe=torch.frombuffer(bytearray(seed),dtype=torch.float32).reshape(target)
            y.backward(probe,retain_graph=i in (3,5))
            assert tuple(y.shape)==target
            assert tuple(custody.values(z) for z in leaves)==raws and custody.values(probe)==seed
            eager_out=custody.values(y)
            eager_pullbacks=tuple(custody.values(z.grad) if z.grad is not None else None for z in leaves)
            shp,out,pullbacks,frames=lower.conv2d_bias_seeded(binary,xs,ws,*raws,seed,
                                                      **options,requires_grad=req)
            if (shp,out,pullbacks)!=(target,eager_out,eager_pullbacks):
                raise ValueError(f"native/PyTorch mismatch in geometry {i} and gradient flags {req}")
            rows.append(dict(geometry=fixture,requires_grad=req,shape=shp,
                             input_sha256=list(map(custody.sha,raws)),seed_sha256=custody.sha(seed),
                             output_sha256=custody.sha(out),output_words=len(out)//4,
                             gradient_sha256=[None if r is None else custody.sha(r) for r in pullbacks],
                             gradient_words=[0 if r is None else len(r)//4 for r in pullbacks],frames=frames))
            if i in (3,5):
                seed2=previous.pack_values(target,i+12)
                probe2=torch.frombuffer(bytearray(seed2),dtype=torch.float32).reshape(target)
                y.backward(probe2)
                _,repeat_out,repeat_pullbacks,repeat_frames=lower.conv2d_bias_seeded(
                    binary,xs,ws,*raws,seed,**options,requires_grad=req,accumulate_seed=seed2)
                eager_totals=tuple(custody.values(z.grad) if z.grad is not None else None for z in leaves)
                if repeat_out!=eager_out or repeat_pullbacks!=eager_totals:
                    raise ValueError(f"native seeded accumulation mismatch in geometry {i}, flags {req}")
                if custody.values(probe2)!=seed2 or eager_totals==eager_pullbacks:
                    raise ValueError("invalid second seed control")
                accumulated.append(dict(geometry=fixture,requires_grad=req,first_seed_sha256=custody.sha(seed),
                                        second_seed_sha256=custody.sha(seed2),frames=repeat_frames,
                                        gradient_words=[0 if r is None else len(r)//4 for r in repeat_pullbacks],
                                        gradient_sha256=[None if r is None else custody.sha(r) for r in repeat_pullbacks]))

    xs,ws,st,pad,dil,grp=previous.FIXTURES[1]
    x,w,b=(previous.pack_values(shape,4) for shape in (xs,ws,(ws[0],)))
    target=lower.geometry(xs,ws,st,pad,dil,grp)
    seed=previous.pack_values(target,4)
    negatives=[]
    for name,arguments,flags in (
        ("wrong bias extent",(xs,ws,x,w,b+b,seed),{}),
        ("wrong seed extent",(xs,ws,x,w,b,seed[:4]),{}),
        ("non-finite bias",(xs,ws,x,w,bytes.fromhex("0000807f"),seed),{}),
        ("outside dyadic grid",(xs,ws,x,w,bytes.fromhex("cdcccc3d"),seed),{}),
        ("no gradient",(xs,ws,x,w,b,seed),{"requires_grad":(False,False,False)}),
        ("incompatible groups",(xs,ws,x,w,b,seed),{"groups":2}),
    ):
        # A missing executable proves each refusal is before native I/O.
        try:
            lower.conv2d_bias_seeded(binary.parent/"nonexistent-fail-closed",*arguments,**flags)
        except ValueError as e:
            negatives.append(dict(case=name,error=str(e)))
        else:
            raise ValueError("missing preflight refusal: "+name)
    return dict(schema="heptatorch.native-composed-conv2d-bias-seeded.v1",status="PASS_SCOPED",
                native_sha256=lower.ELF,reference=dict(torch=str(torch.__version__),cpu="AVX2",threads=1,**reference),
                native_forward_opcodes=[22,17,2],native_backward_opcode=29,wire_lowering_only=True,
                cases=rows,accumulation_cases=accumulated,preflight_refusals=negatives,whole_pytorch_qualification=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("binary","wheel","install"):
        p.add_argument("--"+arg,required=True,type=Path)
    print(json.dumps(run(**vars(p.parse_args())),sort_keys=True,indent=2))
