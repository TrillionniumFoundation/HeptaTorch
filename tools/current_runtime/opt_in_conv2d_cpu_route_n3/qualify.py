"""Prove the N=3 slow lane while preserving and measuring the oneDNN boundary."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

_ROOT = Path(__file__).resolve().parents[1]
_QUAL = _ROOT / "opt_in_conv2d_cpu_route/qualify.py"
_spec = importlib.util.spec_from_file_location("cpu_route_n2_qualification", _QUAL)
prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prior)
wire, custody, rounding, base = prior.wire, prior.custody, prior.rounding, prior.base

SOURCE = "bf90089e0cbd96f05852b6ba8488b5f3416156298822609d65bc6a4df1fbd79f"
ELF = "82c87f5987dd0d918b41c482d5df6b610262f004750ed0bae155311dba7e0ba3"
PREVIOUS = prior.ELF


def require_prior_suite(binary, parent, wheel, install):
    expected = _QUAL.with_name("qualification.json").read_bytes()
    cmd = (str(sys.executable), "-s", "-E", "-B", str(Path(__file__).resolve()),
           "--binary", str(binary), "--parent", str(parent), "--wheel", str(wheel),
           "--install", str(install), "--prior-only")
    previous = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, timeout=120).stdout
    wire.need(previous==expected,"N1/N2 primary qualification altered beyond candidate pin")
    return custody.sha(previous)


def bits_differ(a,b):
    wire.need(len(a)==len(b) and len(a)%4==0,"invalid output extent")
    return [[i//4,a[i:i+4].hex(),b[i:i+4].hex()]
            for i in range(0,len(a),4) if a[i:i+4]!=b[i:i+4]]


def run(binary,parent,explicit,wheel,install):
    import torch
    replay = require_prior_suite(binary, explicit, wheel, install)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    wire.need(torch.__version__=="2.15.0+cpu" and torch.backends.cpu.get_cpu_capability()=="AVX2",
              "requires locked Torch reference")
    wire.need(torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled,
              "requires default enabled oneDNN availability")
    binding = torch.backends.mkldnn.enabled
    reference = custody.verify_torch_custody(wheel, install, torch)
    rng = random.Random(127019)
    rows, controls = [], []
    totals = Counter()
    with tempfile.TemporaryDirectory(prefix="heptatorch-n3-route-") as tmp:
        rt=Path(tmp)
        ca, cb = rt/"new", rt/"old"
        ca.mkdir()
        cb.mkdir()
        current=wire.stage_executable(ca,wire.pinned_bytes(binary,ELF,wire.ELF_CAP))
        older=wire.stage_executable(cb,wire.pinned_bytes(parent,PREVIOUS,wire.ELF_CAP))
        for n,h,w in ((2,4,4),(3,4,4),(1,4,5)):
            xshape,wshape,bshape=(n,2,h,w),(2,2,3,3),(2,)
            shapes=(xshape,wshape,bshape)
            target=(n,2,h-2,w-2)
            fixed=(xshape,wshape,(1,1),(0,0),(1,1))
            cases=[]
            for k in range(32):
                vals=[torch.tensor([rng.uniform(-3,3) for _ in range(count)],dtype=torch.float32).reshape(shape)
                      for count,shape in zip((n*2*h*w,36,2),shapes)]
                cases.append((f"dense-{k:02}",*vals))
            if n==3:
                for label, fir, las, bias, _, _pre, _post in rounding.CASES:
                    x=torch.ones(xshape,dtype=torch.float32)
                    kernels=torch.zeros(wshape,dtype=torch.float32)
                    kernels[:,0,0,0]=rounding.value(fir)
                    kernels[:,1,2,2]=rounding.value(las)
                    b=torch.full(bshape,rounding.value(bias),dtype=torch.float32)
                    cases.append((label,x,kernels,b))
            for label,x,kernels,b in cases:
                raw=tuple(custody.values(t) for t in (x,kernels,b))
                same_args=(*fixed,1)
                path_pre=prior.measure(current,same_args,raw,target,op=126)
                path_post=prior.measure(current,same_args,raw,target,op=125)
                observations=[]
                for enabled in (False,True):
                    try:
                        torch.backends.mkldnn.enabled=enabled
                        direct, observed=rounding.observe(torch,x,kernels,b)
                    finally:
                        torch.backends.mkldnn.enabled=binding
                    oracle=custody.values(direct)
                    wire.need(tuple(custody.values(v) for v in (x,kernels,b))==raw,"oracle mutated inputs")
                    post=enabled and n>1
                    family="aten::mkldnn_convolution" if post else "aten::_slow_conv2d_forward"
                    wire.need(observed==family,"unexpected real oracle kernel")
                    explicit_result=path_post if post else path_pre
                    mismatches=bits_differ(explicit_result["output"],oracle)
                    auto_args=(*fixed,int(enabled))
                    total=f"N{n}-{h}x{w}-enabled{int(enabled)}"
                    totals[total+"-observations"]+=1
                    totals[total+"-explicit-mismatch-cases"]+=bool(mismatches)
                    totals[total+"-explicit-mismatch-words"]+=len(mismatches)
                    if n==2:
                        selected=prior.measure(current,auto_args,raw,target)
                        wire.need(selected==prior.measure(older,auto_args,raw,target),
                                  "old N2 auto changed in expanded dense suite")
                        wire.need(selected["output"]==explicit_result["output"],"unchanged N2 route disagrees")
                    elif n==3 and not enabled:
                        wire.need(not mismatches,"N3 slow was not exact")
                        selected=prior.measure(current,auto_args,raw,target)
                        wire.need(selected["output"]==oracle,"N3 native auto was not exact")
                        totals["N3-slow-qualified-words"]+=len(oracle)//4
                    else:
                        selected=prior.measure(current,auto_args,raw,None,refusal=10)
                    if (n,h,w)!=(2,4,4):
                        prior.measure(older,auto_args,raw,None,refusal=10)
                    observations.append(dict(enabled=enabled,kernel=observed,words=len(oracle)//4,
                                             eager_sha256=custody.sha(oracle),
                                             explicit_sha256=custody.sha(explicit_result["output"]),
                                             different_word_count=len(mismatches),
                                             first_different_word=(mismatches[0] if mismatches else None),
                                             auto_status=selected["status"],auto_frames=selected["native_frames"]))
                if n==2 and observations[1]["different_word_count"]:
                    totals["old-N2-on-dense-counterexamples"]+=1
                if n==3 and observations[1]["different_word_count"]:
                    totals["N3-on-counterexamples"]+=1
                if n==3 and path_pre["output"]!=path_post["output"]:
                    totals["N3-order-discriminating-cases"]+=1
                rows.append(dict(case=label,shape=xshape,source_sha256=list(map(custody.sha,raw)),
                                 observations=observations))
        for lab,n,h,w,grad in (("batch4",4,4,4,(False,False,False)),
                ("width5",3,4,5,(False,False,False)),
                ("input-grad",3,4,4,(True,False,False)),
                ("weight-grad",3,4,4,(False,True,False)),
                ("bias-grad",3,4,4,(False,False,True))):
            xs, ws, bs=(n,2,h,w),(2,2,3,3),(2,)
            raw=tuple(base.previous.pack_values(s,4) for s in (xs,ws,bs))
            item=prior.measure(current,(xs,ws,(1,1),(0,0),(1,1),0),raw,None,
                               grad=grad,refusal=7 if any(grad) else 10)
            controls.append(dict(case=lab,**item))
    wire.need(totals["old-N2-on-dense-counterexamples"]>0 and totals["N3-on-counterexamples"]>0
              and totals["N1-4x5-enabled0-explicit-mismatch-cases"]>0
              and totals["N3-order-discriminating-cases"]>0,
              "boundary cases no longer discriminate")
    return dict(schema="heptatorch.native-conv2d-bias-order-n3-slow.v1",status="PASS_SCOPED",
                source_sha256=SOURCE,native_sha256=ELF,parent_sha256=PREVIOUS,
                preserved_primary_report_sha256=replay,
                reference=dict(torch=str(torch.__version__),cpu="AVX2",threads=1,**reference),
                totals=dict(totals),raw_case_count=len(rows),
                row_stream_sha256=custody.sha(json.dumps(rows,sort_keys=True).encode()),
                representative_rows=[v for v in rows if v["case"] in ("dense-00","decimal-cancellation")],
                negative_controls=controls,
                full_pytorch_qualification=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","parent","wheel","install"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--explicit",type=Path)
    p.add_argument("--prior-only",action="store_true")
    args=vars(p.parse_args())
    mode=args.pop("prior_only")
    if mode:
        orig_id,orig_source = prior.ELF,prior.SOURCE
        prior.ELF,prior.SOURCE = ELF,SOURCE
        args.pop("explicit")
        result=prior.compare(**args)
        result.update(native_sha256=orig_id,source_sha256=orig_source)
    else:
        result=run(**args)
    print(json.dumps(result,sort_keys=True,indent=2))
