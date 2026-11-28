"""Qualify explicit native HIA FMA order on two pinned oneDNN implementations."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile

_path=Path(__file__).resolve().parents[1]/"opt_in_conv2d_cpu_route_slow_only/qualify.py"
_spec=importlib.util.spec_from_file_location("conv_slow_only_qualification",_path)
up=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(up)
base,prior,wire,custody,rounding=up.base,up.prior,up.wire,up.custody,up.rounding
ELF="55bce48df60b81b99fb36707c3a2dbc72760427edf0f5422d5dd401e3b850790"
SOURCE="139eff54155623e29b8ad31f8d13167d66e1b607acf60cd6db39290c9fdd71b3"
VERSION="oneDNN v3.13.3 (commit b38f97eb6f2173513112f15a0481750b8a66e901)"


def lock(torch):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    wire.need(torch.__version__=="2.15.0+cpu" and torch.backends.cpu.get_cpu_capability()=="AVX2",
              "requires locked Torch reference")
    wire.need(torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled,
              "requires default enabled oneDNN")


def verbose_kernel(expected):
    env=dict(os.environ,ONEDNN_VERBOSE="1")
    cmd=(str(sys.executable),"-s","-B",str(Path(__file__).resolve()),"--describe-kernel")
    out=subprocess.run(cmd,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                       timeout=30).stdout.decode("utf-8")
    info=[line for line in out.splitlines() if VERSION in line]
    kernels=[line for line in out.splitlines() if ",exec,cpu,convolution," in line]
    wire.need(len(info)==1 and len(kernels)==2 and all(f",convolution,{expected}," in k for k in kernels),
              "oneDNN implementation does not match the tested explicit mode")
    wire.need(any("mb2_ic2oc2_ih4oh2kh3" in s for s in kernels)
              and any("mb3_ic2oc2_ih4oh2kh3" in s for s in kernels),"missing active scoped shapes")
    return {"oneDNN_version":VERSION,"jit_primitive":expected}


def samples(torch,n,rng):
    x=(n,2,4,4)
    w=(2,2,3,3)
    b=(2,)
    shapes=(x,w,b)
    for i in range(32):
        items=[]
        for shape,count in zip(shapes,(n*32,36,2)):
            items.append(torch.tensor([rng.uniform(-3,3) for _ in range(count)],dtype=torch.float32)
                         .reshape(shape))
        yield f"dense-{i:02}",tuple(items)
    for i in range(8):
        palette=(0.,-0.,2**-24,-2**-24,2**-23,-2**-23,2**-12,-2**-12,1.,-1.,8.,-8.)
        items=[]
        for shape,count in zip(shapes,(n*32,36,2)):
            items.append(torch.tensor([rng.choice(palette) for _ in range(count)],dtype=torch.float32)
                         .reshape(shape))
        yield f"signed-domain-edge-{i}",tuple(items)
    for label,aa,zz,bb,*_ in rounding.CASES:
        vals=tuple(rounding.value(z) for z in (aa,zz,bb))
        if any(v and not 2**-24<=abs(v)<=8. for v in vals):
            continue
        ts=torch.ones(x,dtype=torch.float32)
        tw=torch.zeros(w,dtype=torch.float32)
        tw[:,0,0,0]=vals[0]
        tw[:,1,2,2]=vals[1]
        tb=torch.full(b,vals[2],dtype=torch.float32)
        yield label,(ts,tw,tb)
    for i,sign in enumerate((0.,-0.)):
        yield f"all-same-zero-{i}",tuple(torch.full(shape,sign,dtype=torch.float32) for shape in shapes)


def run(binary,parent,baseline,wheel,install,mode):
    import torch
    limit=os.environ.get("ONEDNN_MAX_CPU_ISA")
    wire.need(not os.environ.get("DNNL_MAX_CPU_ISA") and
              ((limit is None and mode==0) or (limit=="AVX2" and mode==1)),
              "reference mode must be selected before PyTorch init")
    lock(torch)
    hint=verbose_kernel("jit:avx2" if mode else "jit:avx512_core")
    reference=custody.verify_torch_custody(wheel,install,torch)
    rng=random.Random(662171)
    with tempfile.TemporaryDirectory(prefix="heptatorch-fma-conv2d-") as td:
        tmp=Path(td)
        for s in ("candidate","parent","baseline"):
            (tmp/s).mkdir()
        elf=wire.stage_executable(tmp/"candidate",wire.pinned_bytes(binary,ELF,wire.ELF_CAP))
        prev=wire.stage_executable(tmp/"parent",wire.pinned_bytes(parent,up.ELF,wire.ELF_CAP))
        canonical="ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e"
        old=wire.stage_executable(tmp/"baseline",wire.pinned_bytes(baseline,canonical,wire.ELF_CAP))
        totals=Counter()
        rows, guards, oldrefusals=[],[],[]
        for n in (2,3):
            x,w,b=(n,2,4,4),(2,2,3,3),(2,)
            target=(n,2,2,2)
            geometry=(x,w,(1,1),(0,0),(1,1))
            for name,ts in samples(torch,n,rng):
                raw=tuple(custody.values(t) for t in ts)
                oracle,kernel=rounding.observe(torch,*ts)
                wire.need(kernel=="aten::mkldnn_convolution","reference did not execute oneDNN")
                expected=custody.values(oracle)
                actual=prior.measure(elf,(*geometry,mode),raw,target,op=128)
                alternate=prior.measure(elf,(*geometry,1-mode),raw,target,op=128)
                diff=up.upstream.bits_differ(expected,actual["output"])
                wire.need(not diff,f"native FMA mismatch N{n} {name} mode{mode}: {diff[:4]}")
                wire.need(tuple(custody.values(t) for t in ts)==raw,"eager inputs mutated")
                totals["bit_exact_words"]+=n*8
                totals["observations"]+=1
                totals["other_mode_mismatch_cases"]+=alternate["output"]!=expected
                if name=="dense-00":
                    # The new explicit route never changes auto refusals or older numeric controls.
                    for op in (125,126):
                        wire.need(prior.measure(elf,(*geometry,1),raw,target,op=op)==
                                  prior.measure(prev,(*geometry,1),raw,target,op=op),
                                  "old explicit command changed")
                    wire.need(prior.measure(elf,(*geometry,1),raw,None,refusal=10)==
                              prior.measure(prev,(*geometry,1),raw,None,refusal=10),
                              "auto enabled must still refuse")
                    for candidate in (prev,old):
                        oldrefusals.append(prior.measure(candidate,(*geometry,mode),raw,None,
                                                        op=128,refusal=4))
                rows.append(dict(name=name,batch=n,oracle_sha256=custody.sha(expected),
                                 input_sha256=list(map(custody.sha,raw)),native_frames=actual["native_frames"],
                                 alternate_sha256=custody.sha(alternate["output"])))
            for index,word in enumerate(("01000000","ffff7f33","01000041","0000807f","0000c07f")):
                first=bytearray(base.previous.pack_values(x,7))
                first[:4]=bytes.fromhex(word)
                raw=(bytes(first),*(base.previous.pack_values(s,9) for s in (w,b)))
                item=prior.measure(elf,(*geometry,mode),raw,None,op=128,refusal=10)
                guards.append(dict(batch=n,control=f"domain-{index}",raw_word=word,**item))
            data=tuple(base.previous.pack_values(s,7) for s in (x,w,b))
            check=prior.measure(elf,(*geometry,2),data,None,op=128,refusal=10)
            guards.append(dict(batch=n,control="mode-2",**check))
            check=prior.measure(elf,(*geometry,mode),data,None,op=128,refusal=7,
                                grad=(False,True,False))
            guards.append(dict(batch=n,control="gradient",**check))
        wire.need(totals["other_mode_mismatch_cases"]>0,"explicit modes no longer discriminate")
    return dict(schema="heptatorch.native-conv2d-explicit-fma.v1",status="PASS_SCOPED",
                source_sha256=SOURCE,native_sha256=ELF,parent_sha256=up.ELF,
                explicit_mode=mode,reference=dict(torch=str(torch.__version__),cpu="AVX2",threads=1,
                                                  **hint,**reference),
                totals=dict(totals),row_stream_sha256=custody.sha(json.dumps(rows,sort_keys=True).encode()),
                representative_rows=[r for r in rows if r["name"] in ("dense-00","signed-domain-edge-0")],
                exact_parent_and_canonical_refusals=oldrefusals,negative_controls=guards,
                full_pytorch_qualification=False)


if __name__=="__main__":
    if sys.argv[1:]==["--describe-kernel"]:
        import torch
        lock(torch)
        with torch.no_grad():
            for n in (2,3):
                torch.nn.functional.conv2d(torch.ones(n,2,4,4),torch.ones(2,2,3,3),torch.ones(2))
    else:
        p=argparse.ArgumentParser(description=__doc__)
        for name in ("binary","parent","baseline","wheel","install"):
            p.add_argument("--"+name,type=Path,required=True)
        p.add_argument("--mode",type=int,choices=(0,1),required=True)
        print(json.dumps(run(**vars(p.parse_args())),sort_keys=True,indent=2))
