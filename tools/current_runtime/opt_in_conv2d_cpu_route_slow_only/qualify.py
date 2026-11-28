"""Qualify the fail-closed native CPU route against paired eager dispatch."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import random
import tempfile

_path = Path(__file__).resolve().parents[1] / "opt_in_conv2d_cpu_route_n3/qualify.py"
_spec = importlib.util.spec_from_file_location("conv_cpu_route_n3_qualification", _path)
upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream)
prior, wire, custody, rounding, base = (upstream.prior, upstream.wire, upstream.custody,
                                       upstream.rounding, upstream.base)
ELF = "6d4f452a6fb1d5db9bab6840ff23aa8a6e5ba624d44abf0cfb63b71e658d72e7"
SOURCE = "07eab895d403e780eb52737720c36f6fd8bd7e2bb5ee7db1e9f64087c4d89084"


def run(binary,parent,wheel,install):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    wire.need(torch.__version__=="2.15.0+cpu" and torch.backends.cpu.get_cpu_capability()=="AVX2",
              "requires locked Torch reference")
    wire.need(torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled,
              "requires default oneDNN availability and preference")
    binding=torch.backends.mkldnn.enabled
    reference=custody.verify_torch_custody(wheel,install,torch)
    rng=random.Random(507521)
    rows, totals = [], Counter()
    with tempfile.TemporaryDirectory(prefix="heptatorch-slow-only-route-") as td:
        r=Path(td)
        for label in ("cur","prev"):
            (r/label).mkdir()
        candidate=wire.stage_executable(r/"cur",wire.pinned_bytes(binary,ELF,wire.ELF_CAP))
        prev=wire.stage_executable(r/"prev",wire.pinned_bytes(parent,upstream.ELF,wire.ELF_CAP))
        for n in (1,2,3):
            shapes=((n,2,4,4),(2,2,3,3),(2,))
            outs=(n,2,2,2)
            fixed=(*shapes[:2],(1,1),(0,0),(1,1))
            for i in range(39):
                if i<32:
                    vals=tuple(torch.tensor([rng.uniform(-3,3) for _ in range(k)],dtype=torch.float32)
                               .reshape(sh) for sh,k in zip(shapes,(n*32,36,2)))
                else:
                    _,a,z,b,*_ = rounding.CASES[i-32]
                    x=torch.ones(shapes[0],dtype=torch.float32)
                    kernel=torch.zeros(shapes[1],dtype=torch.float32)
                    kernel[:,0,0,0]=rounding.value(a)
                    kernel[:,1,2,2]=rounding.value(z)
                    bias=torch.full(shapes[2],rounding.value(b),dtype=torch.float32)
                    vals=x,kernel,bias
                raw=tuple(custody.values(z) for z in vals)
                ex={}
                for op in (125,126):
                    result=prior.measure(candidate,(*fixed,1),raw,outs,op=op)
                    wire.need(result==prior.measure(prev,(*fixed,1),raw,outs,op=op),
                              "explicit native order changed")
                    ex[op]=result
                    totals["unchanged_explicit_op_comparisons"]+=1
                for enabled in (False,True):
                    try:
                        torch.backends.mkldnn.enabled=enabled
                        actual,profile=rounding.observe(torch,*vals)
                    finally:
                        torch.backends.mkldnn.enabled=binding
                    oracle=custody.values(actual)
                    forbidden=enabled and n>1
                    expected="aten::mkldnn_convolution" if forbidden else "aten::_slow_conv2d_forward"
                    wire.need(expected==profile,"unexpected locked eager route")
                    args=(*fixed,int(enabled))
                    if forbidden:
                        hit=prior.measure(candidate,args,raw,None,refusal=10)
                        totals[f"N{n}_forbidden_refusals"]+=1
                        if n==2:
                            old=prior.measure(prev,args,raw,outs)
                            wire.need(old["output"]==ex[125]["output"],"wrong earlier experimental order")
                        else:
                            prior.measure(prev,args,raw,None,refusal=10)
                        if ex[125]["output"]!=oracle:
                            totals[f"N{n}_post_sum_counterexamples"]+=1
                    else:
                        hit=prior.measure(candidate,args,raw,outs)
                        wire.need(hit==prior.measure(prev,args,raw,outs),
                                  "existing slow route or physical owners changed")
                        wire.need(oracle==hit["output"]==ex[126]["output"],"accepted slow lane was not bit-exact")
                        totals["accepted_slow_observations"]+=1
                        totals["accepted_slow_words"]+=len(oracle)//4
                    rows.append(dict(batch=n,case=i,enabled=enabled,input_sha256=list(map(custody.sha,raw)),
                                     kernel=profile,oracle_sha256=custody.sha(oracle),status=hit["status"],
                                     native_frames=hit["native_frames"]))
        guard=[]
        for n,enabled,grad,status in ((1,0,True,7),(1,1,True,7),(2,0,True,7),(2,1,True,10),
                                      (3,0,True,7),(3,1,True,10),(4,0,False,10)):
            x,w,b=(n,2,4,4),(2,2,3,3),(2,)
            vals=tuple(base.previous.pack_values(s,7) for s in (x,w,b))
            r=prior.measure(candidate,(x,w,(1,1),(0,0),(1,1),enabled),vals,None,
                            grad=(False,False,grad),refusal=status)
            guard.append(dict(batch=n,enabled=enabled,bias_grad=grad,**r))
    wire.need(totals["N2_post_sum_counterexamples"]>0 and totals["N3_post_sum_counterexamples"]>0,
              "oneDNN boundary no longer discriminates")
    return dict(schema="heptatorch.native-conv2d-cpu-slow-only.v1",status="PASS_SCOPED",
                source_sha256=SOURCE,native_sha256=ELF,parent_sha256=upstream.ELF,
                reference=dict(torch=str(torch.__version__),cpu="AVX2",threads=1,**reference),
                totals=dict(totals),row_count=len(rows),
                row_stream_sha256=custody.sha(json.dumps(rows,sort_keys=True).encode()),
                sample_rows=[x for x in rows if x["case"] in (0,33)],
                additional_negative_controls=guard,full_pytorch_qualification=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","parent","wheel","install"):
        p.add_argument("--"+name,type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args())),sort_keys=True,indent=2))
