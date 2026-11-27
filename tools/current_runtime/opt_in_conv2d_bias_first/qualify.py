"""Replay both bias orders, verifying that the two native opcodes stay distinct."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys

_ROOT = Path(__file__).resolve().parents[1]
_ORDER = _ROOT / "opt_in_conv2d_bias_rounding" / "qualify.py"
_spec = importlib.util.spec_from_file_location("conv_bias_rounding", _ORDER)
rounding = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rounding)
base, wire, custody = rounding.base, rounding.wire, rounding.custody
ELF = "41d17a95bdc89b37bd9050c63372a244497c32114b7ef9c2366f5340fc3e380c"
SOURCE = "318508540c44cdc60c18cb3a876c13b06a6cb60a34153ad4be621e1a964df4f8"


def verify_unchanged(actual, path, key, eid):
    committed = json.loads(path.read_text())
    wire.need(committed[key] == eid, "wrong predecessor qualification")
    actual[key] = eid
    wire.need(json.loads(json.dumps(actual)) == committed, f"{path.parent.name} regression")
    return custody.sha(path.read_bytes())


def measure_first(exe, geometry, raws, wanted, **kwargs):
    previous=base.OP
    try:
        base.OP=126
        return base.measure(exe,geometry,raws,wanted,**kwargs)
    finally:
        base.OP=previous


def run(binary,parent,old_conv,wheel,install):
    import torch
    baseline = base.ELF
    with tempfile.TemporaryDirectory(prefix="heptatorch-bias-first-") as tmp:
        a,b=Path(tmp)/"candidate",Path(tmp)/"parent"
        a.mkdir()
        b.mkdir()
        new=wire.stage_executable(a,wire.pinned_bytes(binary,ELF,wire.ELF_CAP))
        previous=wire.stage_executable(b,wire.pinned_bytes(parent,baseline,wire.ELF_CAP))
        # Run the original test programs on the candidate. Reconcile only the
        # separately verified executable pin; every compared byte is retained.
        def replay(mode):
            args=(str(sys.executable),"-s","-E","-B",str(Path(__file__).resolve()),
                  "--mode",mode,"--binary",str(binary),"--parent",str(parent),"--old-conv",str(old_conv),
                  "--wheel",str(wheel),"--install",str(install))
            result=subprocess.run(args,check=True,stdout=subprocess.PIPE,timeout=180)
            return json.loads(result.stdout)

        complete=replay("exact-post")
        trace=replay("round-order")
        regression=verify_unchanged(complete,_ROOT/"opt_in_conv2d_bias"/"qualification.json",
                                    "native_sha256",baseline)
        boundary=verify_unchanged(trace,_ROOT/"opt_in_conv2d_bias_rounding"/"qualification.json",
                                  "native_sha256",baseline)
        rows=[]
        for record in trace["observations"]:
            if not record["one_dnn_enabled"]:
                # Each geometry/value pair is invariant in the actual native
                # runtime. Its bytes are compared against both Torch settings.
                continue
            geometry=record["geometry"]
            xs,ws=geometry[:2]
            name,first,last,bias,_,pre,post=next(x for x in rounding.CASES if x[0]==record["name"])
            x=torch.ones(xs,dtype=torch.float32)
            w=torch.zeros(ws,dtype=torch.float32)
            w[:,0,0,0]=rounding.value(first)
            w[:,1,2,2]=rounding.value(last)
            bias_tensor=torch.full((2,),rounding.value(bias),dtype=torch.float32)
            raws=tuple(custody.values(t) for t in (x,w,bias_tensor))
            wanted=tuple(record["shape"])
            out=measure_first(new,geometry,raws,wanted)
            out_raw=out.pop("output")
            wire.need(out_raw==bytes.fromhex(pre)*record["words"],"biased first native bits differ")
            refusal=measure_first(previous,geometry,raws,wanted,refusal=4)
            matching=[item for item in trace["observations"] if item["name"]==name
                      and item["geometry"]==geometry]
            wire.need(len(matching)==2,"case did not cover both oneDNN settings")
            for eager in matching:
                slow=eager["eager_profiler_kernel"]=="aten::_slow_conv2d_forward"
                exact=pre==eager["eager_direct_le_hex"]
                wire.need(exact==(slow or pre==post),"wrong explicit route match")
                rows.append(dict(name=name,geometry=geometry,shape=wanted,
                                 native_first_le_hex=pre,native_post_le_hex=post,
                                 eager_kernel=eager["eager_profiler_kernel"],one_dnn_enabled=eager["one_dnn_enabled"],
                                 words=eager["words"],matches_direct_eager=exact,parent_refusal=refusal,**out))
        # The #82 ordinary fixtures give different channel bias values and a
        # wider grouped, dilated and padded geometry (exactly representable).
        dyadic=[]
        for i in range(len(complete["forward_cases"])//3):
            for j in range(3):
                entry=complete["forward_cases"][3*i+j]
                g=entry["geometry"]
                x,w=g[:2]
                shapes=(x,w,(w[0],))
                raw=tuple(base.previous.pack_values(s,7*i+11*j+k+3) for k,s in enumerate(shapes))
                wire.need(list(map(custody.sha,raw))==entry["values_sha256"],"dyadic source changed")
                got=measure_first(new,g,raw,tuple(entry["shape"]))
                output=got.pop("output")
                wire.need(custody.sha(output)==entry["result_sha256"],"dyadic control diverged")
                dyadic.append(dict(geometry=g,forward_sha256=custody.sha(output),**got))
        f=base.previous.FIXTURES[3]
        x,w=f[:2]
        regular=(w[0],)
        controls=[]
        for name,bs,flags,refusal in (
            ("bad-bias-length",(w[0]+1,),{},2),
            ("non-vector-bias",(1,w[0]),{},2),
            ("requires-input-grad",regular,dict(grad=(True,False,False)),7),
            ("requires-weight-grad",regular,dict(grad=(False,True,False)),7),
            ("requires-bias-grad",regular,dict(grad=(False,False,True)),7),
            ("invalid-bias-handle",regular,dict(handle=0),1),
        ):
            raws=tuple(base.previous.pack_values(s,8) for s in (x,w,bs))
            guards=measure_first(new,f,raws,None,bias_shape=bs,refusal=refusal,**flags)
            controls.append(dict(case=name,**guards))
    return dict(schema="heptatorch.native-conv2d-explicit-bias-first.v1",status="PASS_SCOPED",
                source_sha256=SOURCE,native_sha256=ELF,parent_sha256=baseline,
                prior_post_bias_report_sha256=regression,prior_rounding_report_sha256=boundary,
                reference=trace["reference"],backend_auto_selection=False,new_opcode=126,
                traces=rows,dyadic_controls=dyadic,negative_controls=controls,
                full_pytorch_qualification=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","parent","old-conv","wheel","install"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--mode",choices=("primary","exact-post","round-order"),default="primary")
    opts=vars(p.parse_args())
    mode=opts.pop("mode")
    if mode=="primary":
        result=run(**opts)
    else:
        base.ELF=ELF
        if mode=="exact-post":
            result=base.run(opts["binary"],opts["old_conv"],opts["wheel"],opts["install"])
        else:
            result=rounding.run(opts["binary"],opts["wheel"],opts["install"])
    print(json.dumps(result,sort_keys=True,indent=2))
