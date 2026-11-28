#!/usr/bin/env python3
"""Compare the native mixed F32/F64 rank-zero copysign against eager PyTorch."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import runpy
import struct
import sys

HERE = Path(__file__).resolve().parent
NARROW = runpy.run_path(str(HERE.parent/"opt_in_copysign_f32/qualify.py"))
WIDE = runpy.run_path(str(HERE.parent/"opt_in_copysign_f64/qualify.py"))
PARENT = "1251c7c7f0b0a79d74e5106da1b72158a2aa5f5ea69d0b04a58f64252cc22900"
CANDIDATE = "372f8764bcb19e89847b4205e6ce821801c90134c1e2983773d9f52f763ea3ed"


def inputs():
    rng = random.Random(97003264)
    wide_edge = list(WIDE["WORDS"])
    for x in (2.0**-150,2.0**-149,2.0**-126,1+2.0**-24,2.0**128-2.0**103):
        w, = struct.unpack("<Q",struct.pack("<d",x))
        wide_edge.extend([w-1,w,w+1])
    for i,z in enumerate(wide_edge):
        for x in (NARROW["WORDS"][i%len(NARROW["WORDS"])],0,0x80000000,0x7F812345):
            for a,b in (((),()),((),(1,)),((1,),())):
                yield a,b,(x,),(z,),NARROW,WIDE
                yield a,b,(z,),(x,),WIDE,NARROW
    for shape in ((0,), (2,0,3),(7,),(8,),(9,),(33,),(2,3),(2,1,3,2)):
        for other,primary in ((NARROW,WIDE),(WIDE,NARROW)):
            for phase in (1,7,15,19):
                w0 = other["WORDS"][phase]
                vv = tuple(primary["WORDS"][(i+phase)%len(primary["WORDS"])]
                           if i%2 == 0 else rng.getrandbits(32 if primary is NARROW else 64)
                           for i in range(math.prod(shape)))
                yield (),shape,(w0,),vv,other,primary
                yield shape,(),vv,(w0,),primary,other


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["repository", "parent", "candidate", "output"]:
        p.add_argument("--"+name, required=True, type=Path)
    a = p.parse_args()
    assert not sys.flags.optimize
    repo, old_image, image, out = (getattr(a,k).resolve() for k in ("repository","parent","candidate","output"))
    sha = lambda b: hashlib.sha256(b).hexdigest()
    assert sha(old_image.read_bytes()) == PARENT and sha(image.read_bytes()) == CANDIDATE
    out.mkdir(parents=True)
    os.chdir(repo)
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    saved = sys.argv
    try:
        sys.argv = [str(repo/"tests/pytorch_oracle.py"), str(out/"NO_NATIVE"), str(out/"oracle-error.json"),
                    str(repo/"compatibility/pytorch-oracle-lock.json")]
        ctx = runpy.run_path(sys.argv[0], run_name="heptatorch_mixed_rankzero_oracle")
    finally:
        sys.argv = saved
    torch, prov = ctx["torch"], ctx["PROVENANCE"]
    assert prov["status"] == "PASS" and torch.get_num_threads() == 1
    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
    spec = importlib.util.spec_from_file_location("wire", repo/"tests/integer_tensor_review/wire/protocol.py")
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    sys.path.insert(0, str(repo/"tests"))
    from half_logical.native import Memory

    def eit(shape, bits, dtype):
        values = torch.frombuffer(bytearray(bits), dtype=dtype) if bits else torch.empty(0, dtype=dtype)
        return values.reshape(shape).clone()

    base, native = [wire.Session(binary, directory=out/name)
                    for binary,name in ((old_image,"parent"),(image,"candidate"))]
    totals = {"cases": 0, "unequal_shape_cases": 0, "zero_output_cases": 0, "words": 0,
              "f32_result_cases": 0, "f64_result_cases": 0, "parent_mixed_dtype_refusals": 0,
              "candidate_incompatible_refusals": 0, "candidate_unrelated_refusals": 0}
    digest = hashlib.sha256()

    def op(session, commands):
        return session.exchange(wire.packet(commands))

    for xs,zs,xw,zw,left,right in inputs():
        xb,zb = left["raw"](xw),right["raw"](zw)
        dtype = lambda kind: torch.float32 if kind is NARROW else torch.float64
        x_ref,z_ref = eit(xs,xb,dtype(left)),eit(zs,zb,dtype(right))
        result = torch.ops.aten.copysign.Tensor(x_ref,z_ref)
        expected = bytes(result.untyped_storage())
        assert bytes(x_ref.untyped_storage()) == xb and bytes(z_ref.untyped_storage()) == zb
        assert result.untyped_storage()._cdata not in (x_ref.untyped_storage()._cdata,z_ref.untyped_storage()._cdata)
        expected_type = NARROW if bool(xs if left is NARROW else zs) and not bool(xs if left is WIDE else zs) else WIDE
        code,width,reader = (0,4,39) if expected_type is NARROW else (4,8,98)
        assert result.dtype == dtype(expected_type) and not result.requires_grad
        shape,stride,n = tuple(result.shape),tuple(result.stride()),result.numel()
        assert n*width == len(expected)
        setup = [left["tensor"](wire,xs,xw), right["tensor"](wire,zs,zw)]
        x,z = [native.run([command])[0]["value"] for command in setup]
        u,v = [base.run([command])[0]["value"] for command in setup]
        inputs_before = list(map(wire.meta,native.run([(13,x),(13,z)])))
        assert all(kind["physical"](native,Memory,h,m)==bits for kind,h,m,bits in zip((left,right),(x,z),inputs_before,(xb,zb),strict=True))
        commands = [(2,16,x,z)]
        frame = op(native,commands)
        status, entries = wire.parse(frame)
        assert status == 0, (xs,zs,status)
        y = entries[0]["value"]
        assert y not in (x,z)
        read = [(13,y),(reader,y,0,n),(96,y,0),(13,x),(13,z)]
        readback = op(native,read)
        status, (metadata, values, dtype, mx, mz) = wire.parse(readback)
        assert status == 0
        meta = wire.meta(metadata)
        assert meta["rank"] == len(shape) and tuple(meta["shape"][:len(shape)]) == shape
        assert tuple(meta["strides"][:len(shape)]) == stride, (xs,zs,meta,stride)
        assert (meta["numel"],meta["storage_bytes"],meta["requires_grad"],meta["storage_offset"],meta["version"]) == (n,n*width,0,0,0)
        assert struct.unpack("<4Q",dtype["payload"])[:2] == (code,width)
        assert values["value"] == n and values["payload"] == expected, (xs,zs,xw,zw,values["payload"].hex(),expected.hex())
        assert [wire.meta(mx),wire.meta(mz)] == inputs_before
        assert all(kind["physical"](native,Memory,h,m)==bits for kind,h,m,bits in zip((left,right),(x,z),inputs_before,(xb,zb),strict=True))
        assert all(meta[k] not in [m[k] for m in inputs_before] for k in ("storage_id","version_id"))
        assert expected_type["physical"](native,Memory,y,meta) == expected
        native.run([(41,x),(41,z)])
        assert native.run([(reader,y,0,n)])[0]["payload"] == expected
        native.run([(41,y)])
        base.error([(2,16,u,v)],4)
        base.run([(41,u),(41,v)])
        totals["unequal_shape_cases"] += xs != zs
        totals["parent_mixed_dtype_refusals"] += 1
        totals["f32_result_cases"] += expected_type is NARROW
        totals["f64_result_cases"] += expected_type is WIDE
        totals["cases"] += 1
        totals["words"] += n
        totals["zero_output_cases"] += n==0
        digest.update(struct.pack("<Q",n))
        digest.update(expected)

    for xs,zs in [((2,3),(2,)), ((2,0,3),(1,4,3)), ((0,),(2,)), ((2,1,3,1),(1,4,2,4))]:
        x,z = [native.run([kind["tensor"](wire,k,(0,)*math.prod(k))])[0]["value"] for kind,k in ((NARROW,xs),(WIDE,zs))]
        try:
            torch.ops.aten.copysign.Tensor(torch.empty(xs,dtype=torch.float32),torch.empty(zs,dtype=torch.float64))
        except RuntimeError:
            pass
        else:
            raise AssertionError("oracle accepts supposedly incompatible shapes")
        native.error([(2,16,x,z)],2)
        totals["candidate_incompatible_refusals"] += 1
        native.run([(41,x),(41,z)])

    ns = native.run([NARROW["tensor"](wire,(),(0,))])[0]["value"]
    ws = native.run([WIDE["tensor"](wire,(),(0,))])[0]["value"]
    nv = native.run([NARROW["tensor"](wire,(1,),(0,))])[0]["value"]
    wv = native.run([WIDE["tensor"](wire,(1,),(0,))])[0]["value"]
    ng = native.run([NARROW["tensor"](wire,(1,),(0,),requires=1)])[0]["value"]
    for x,z,code in ((ng,ws,5),(ws,ng,5),(ng,wv,5),(wv,ng,5)):
        native.error([(2,16,x,z)],code)
        totals["candidate_unrelated_refusals"] += 1
    native.run([(41,k) for k in (ns,ws,nv,wv,ng)])
    for session in (base,native):
        session.finish()
    # A fresh pair runs exact previously admitted dimensions without the
    # divergent accepted rank-zero allocations of the main differential.
    same_base,same_new = [wire.Session(binary,directory=out/name)
                         for binary,name in ((old_image,"old-retained"),(image,"new-retained"))]
    frames = 0
    def same(commands):
        nonlocal frames
        left,right = (op(s,commands) for s in (same_base,same_new))
        assert left == right
        status, records = wire.parse(left)
        assert status == 0
        frames += 1
        return records
    for dtype1,dtype2 in ((NARROW,WIDE),(WIDE,NARROW),(NARROW,NARROW),(WIDE,WIDE)):
        for k in range(0,15,3):
            x, = same([dtype1["tensor"](wire,(1,3),dtype1["WORDS"][k:k+3])])
            z, = same([dtype2["tensor"](wire,(2,1),dtype2["WORDS"][k:k+2])])
            x,z = x["value"],z["value"]
            y, = same([(2,16,x,z)])
            y = y["value"]
            same([(13,y),((39 if dtype1 is dtype2 is NARROW else 98),y,0,6),(41,y),(41,x),(41,z)])
    same_base.finish()
    same_new.finish()
    totals["unchanged_predecessor_frames"] = frames
    evidence = {"status":"SCOPED_PASS","parent_sha256":PARENT,"candidate_sha256":CANDIDATE,
                "oracle_lock_sha256":prov["lock_sha256"],"oracle_checks":prov["checks"],
                "expected_words_sha256":digest.hexdigest(),"physical_storage_exact":True,
                "input_unchanged":True,"fresh_storage_version":True,"survives_input_release":True,
                "canonical_credit":False,**totals}
    (out/"qualification.json").write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n")
    print(json.dumps(evidence,sort_keys=True))


if __name__ == "__main__":
    main()
