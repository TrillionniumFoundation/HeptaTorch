"""Replay both exact parents and check real owner slots on a mixed native process."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ELF="e2da21ee898d490b3083d7fe4925fa39a4cda26e26b2bd3a1ed7f199892c5d03"
SOURCE="623ca42c070698c0d9271302e88d56bdd359e45765b02d315ecd145aef23a62d"
SPATIAL_Q_SHA="1279912d9540f20696aa8f91b1d752c4a810882ed4343075b50a26274f726b3e"
SPATIAL_REPORT="446c9b609ff81a0500c1cd39de99a893361b468a579fb06a4a722d2a16260747"
SPATIAL_WIRE="7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fma=load("explicit_conv_fma_qualification",HERE.parent/"opt_in_conv2d_cpu_fma/qualify.py")
wire,custody=fma.wire,fma.custody


def spatial_replay(binary,source,old,wheel,install,spatial_root,tmp):
    directory=spatial_root/"tools/current_runtime/opt_in_spatial_ad"
    program=(directory/"qualify.py").read_bytes()
    pinned=(directory/"qualification.json").read_bytes()
    wire.need(custody.sha(program)==SPATIAL_Q_SHA and custody.sha(pinned)==SPATIAL_REPORT,
              "wrong exact #102 verifier or receipt")
    output=tmp/"spatial-report"
    shim = """import pathlib, sys
p=pathlib.Path(sys.argv.pop(1)); sys.path.insert(0,str(p.parent))
ns={'__name__':'composition_replay','__file__':str(p)}
exec(compile(p.read_bytes(),str(p),'exec'),ns)
ns['ELF'],ns['SOURCE']=sys.argv.pop(1),sys.argv.pop(1)
ns['main']()
"""
    cmd=(str(sys.executable),"-s","-E","-B","-c",shim,str(directory/"qualify.py"),ELF,SOURCE,
         "--binary",str(binary),"--source",str(source),"--parent",str(old),
         "--wheel",str(wheel),"--install",str(install),"--output",str(output))
    clean_env={k:v for k,v in os.environ.items() if k not in ("ONEDNN_MAX_CPU_ISA","DNNL_MAX_CPU_ISA")}
    subprocess.run(cmd,env=clean_env,check=True,stdout=subprocess.PIPE,timeout=180)
    actual=json.loads((output/"qualification.json").read_bytes())
    expected=json.loads(pinned)
    original=expected["candidate_elf_sha256"],expected["source_sha256"]
    wire.need((actual["candidate_elf_sha256"],actual["source_sha256"])==(ELF,SOURCE),
              "spatial replay did not run on candidate")
    actual["candidate_elf_sha256"],actual["source_sha256"]=original
    wire.need(actual==expected,"spatial replay changed beyond exact candidate identity")
    return dict(pinned_report_sha256=SPATIAL_REPORT,
                native_data_frames=actual["native_data_frames"],
                spatial_cases=len(actual["spatial_cases"]),ad_cases=len(actual["ad_cases"]),
                exceptional_runs=len(actual["exceptional_gradient_sha256"]),
                final_non_page_live_counts=actual["final_non_page_live_counts"])


def owner_mix(binary,spatial_root,out,mode):
    import torch
    tests=spatial_root/"tests"
    proto=tests/"integer_tensor_review/wire/protocol.py"
    wire.need(custody.sha(proto.read_bytes())==SPATIAL_WIRE,"wrong exact owner wire")
    w=load("conv_fma_spatial_wire",proto)
    sys.path.insert(0,str(tests))
    from half_logical.native import Memory
    sys.path.pop(0)

    def counts(s):
        m=Memory(s.owner()["bytes"])
        page_storages=set()
        page_holds=set()
        for i in range(256):
            ptr,h=m.q(4096+8*i),m.q(6144+8*i)
            wire.need(bool(ptr)==bool(h),"inconsistent descriptor page")
            if ptr:
                storage=m.row(6,h)[2]
                row=m.row(2,storage)
                wire.need(ptr==(1<<63)+row[2] and row[3]==row[4]==172032,"invalid descriptor owner")
                page_holds.add(h)
                page_storages.add(storage)
        wire.need(len(page_holds)==len(page_storages),"ambiguous descriptor pages")
        live=m.live_counts()
        live[1]-=len(page_storages)
        live[5]-=len(page_holds)
        return live

    def collect(s,cmd):
        return s.run([cmd])[0]

    session=w.Session(binary.resolve(),directory=out)
    taken=[]
    try:
        prime=collect(session,w.tensor((1,),(0.,)))["value"]
        collect(session,(41,prime))
        cold=counts(session)
        stride=4294967297
        for n,which in ((2,128),(3,127),(2,125),(2,126),(3,128),(2,127),(3,125),(3,126)):
            inputs=((n,2,4,4),(2,2,3,3),(2,))
            data=tuple(tuple((((i*(k+11)+k)%19)-9)/8 for i in range(math.prod(shape)))
                       for k,shape in enumerate(inputs))
            eager=tuple(torch.tensor(v,dtype=torch.float32).reshape(shape) for v,shape in zip(data,inputs))
            y=torch.nn.functional.conv2d(*eager)
            expected=(y,torch.square(y),torch.nn.functional.avg_pool2d(torch.square(y),(2,2)))
            nodes=[collect(session,w.tensor(shape,words))["value"]
                   for shape,words in zip(inputs,data,strict=True)]
            pref=mode if which==128 else 0 if which==127 else 1
            outid=collect(session,(which,*nodes,stride,0,stride,pref))["value"]
            squ=collect(session,(3,16,outid))["value"]
            avg=collect(session,(23,23,squ,8589934594,8589934594,0,0))["value"]
            for h in nodes:
                collect(session,(41,h))
            outputs=[]
            owners=[]
            for h,want in zip((outid,squ,avg),expected,strict=True):
                info=w.meta(collect(session,(13,h)))
                dat=collect(session,(39,h,0,want.numel()))["payload"]
                wire.need(dat==custody.values(want) and info["requires_grad"]==0,
                          "mixed FMA/spatial native output differs from actual Torch")
                outputs.append(custody.sha(dat))
                owners.append((info["storage_id"],info["version_id"]))
            wire.need(len({x[0] for x in owners})==len({x[1] for x in owners})==3,
                      "mixed native stages alias physical ownership")
            for h in (avg,squ,outid):
                collect(session,(41,h))
            remain=counts(session)
            wire.need(remain==cold,"live non-page owner change after mixed convolution pipeline")
            taken.append(dict(batch=n,conv_opcode=which,outputs=outputs,non_page_live_counts=remain))
        wire.need(collect(session,(53,))["value"]==1,"owner cleanup invariant")
    finally:
        session.finish()
    return taken


def run(binary,source,spatial_base,slow_base,baseline,wheel,install,spatial_root,mode):
    wire.need(custody.sha(source.read_bytes())==SOURCE,"wrong source")
    before_elf,before_source=fma.ELF,fma.SOURCE
    fma.ELF,fma.SOURCE=ELF,SOURCE
    try:
        actual=fma.run(binary,slow_base,baseline,wheel,install,mode)
    finally:
        fma.ELF,fma.SOURCE=before_elf,before_source
    expected_path=HERE.parent/"opt_in_conv2d_cpu_fma"/(
        "qualification-capped-avx2.json" if mode else "qualification-avx512.json")
    promised=expected_path.read_bytes()
    actual["native_sha256"],actual["source_sha256"]=before_elf,before_source
    wire.need(actual==json.loads(promised),"complete FMA replay changed beyond candidate identity")
    with tempfile.TemporaryDirectory(prefix="heptatorch-compose-spatial-fma-") as tmp:
        tmp=Path(tmp)
        old=spatial_replay(binary,source,spatial_base,wheel,install,spatial_root,tmp)
        mixed=owner_mix(binary,spatial_root,tmp/"mixed",mode)
    return dict(schema="heptatorch.spatial-ad-fma-composition.v1",status="PASS_SCOPED",
                native_sha256=ELF,source_sha256=SOURCE,mode=mode,
                exact_spatial=old,exact_fma_report_sha256=custody.sha(promised),
                exact_fma_native_word_count=actual["totals"]["bit_exact_words"],
                mixed_single_session=mixed,full_pytorch_qualification=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("binary","source","spatial-base","slow-base","baseline","wheel","install","spatial-root"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--mode",type=int,choices=(0,1),required=True)
    print(json.dumps(run(**vars(p.parse_args())),sort_keys=True,indent=2))
