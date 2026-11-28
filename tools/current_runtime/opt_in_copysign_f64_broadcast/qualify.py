#!/usr/bin/env python3
"""Compare the pinned native unequal-shape F64 command with eager PyTorch."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import random
import runpy
import struct
import sys

HERE = Path(__file__).resolve().parent
LEGACY = runpy.run_path(str(HERE.parent/"opt_in_copysign_f64/qualify.py"))
PARENT = LEGACY["CANDIDATE"]
CANDIDATE = "5537162e8cfe8edd6e2a30474e7d3c001a76c5731c5ff94c1930bb27adb544a5"
WORDS = LEGACY["WORDS"]
raw, tensor, physical = (LEGACY[name] for name in ("raw", "tensor", "physical"))


def inputs():
    rng = random.Random(970016)
    pairs = [((), ()), ((2,), (2,)), ((0, 3), (0, 3)), ((2, 1, 3), (2, 1, 3))]
    for dimensions in [(7,), (8,), (9,), (17,), (33,), (2, 3), (2, 0, 3), (0,), (2, 1, 3, 2)]:
        pairs.extend([((), dimensions), (dimensions, ())])
    for n in [0, 1, 2, 7, 8, 9, 17, 33]:
        pairs.extend([((1, n), (3, 1)), ((3, 1), (1, n)),
                      ((n, 1, 1), (4, 2)), ((4, 2), (n, 1, 1)),
                      ((2, 1, n, 1), (1, 3, 1, 2)), ((1, 3, 1, 2), (2, 1, n, 1))])
    for a, b in pairs:
        for offset in [0, 7, 15, 21]:
            def words(shape, phase):
                return tuple(WORDS[(i+phase)%len(WORDS)] if i%3 != 2 else rng.getrandbits(64)
                             for i in range(math.prod(shape)))
            yield a, b, words(a, offset), words(b, offset+11)
    for x, z in itertools.product(WORDS, repeat=2):
        yield (), (1,), (x,), (z,)
        yield (1,), (), (x,), (z,)


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
        ctx = runpy.run_path(sys.argv[0], run_name="heptatorch_f64_broadcast_oracle")
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

    def eit(shape, bits):
        values = torch.frombuffer(bytearray(bits), dtype=torch.float64) if bits else torch.empty(0, dtype=torch.float64)
        return values.reshape(shape).clone()

    base, native = [wire.Session(binary, directory=out/name)
                    for binary,name in ((old_image,"parent"),(image,"candidate"))]
    totals = {"cases": 0, "unequal_shape_cases": 0, "zero_output_cases": 0, "words": 0,
              "unchanged_equal_shape_frames": 0, "parent_unequal_shape_refusals": 0,
              "candidate_incompatible_refusals": 0, "candidate_unrelated_refusals": 0}
    digest = hashlib.sha256()

    def op(session, commands):
        return session.exchange(wire.packet(commands))

    for xs,zs,xw,zw in inputs():
        xb,zb = raw(xw),raw(zw)
        x_ref,z_ref = eit(xs,xb),eit(zs,zb)
        result = torch.ops.aten.copysign.Tensor(x_ref,z_ref)
        expected = bytes(result.untyped_storage())
        assert bytes(x_ref.untyped_storage()) == xb and bytes(z_ref.untyped_storage()) == zb
        assert result.untyped_storage()._cdata not in (x_ref.untyped_storage()._cdata,z_ref.untyped_storage()._cdata)
        assert result.dtype == torch.float64 and not result.requires_grad
        shape,stride,n = tuple(result.shape),tuple(result.stride()),result.numel()
        assert n*8 == len(expected)
        setup = [tensor(wire,xs,xw), tensor(wire,zs,zw)]
        x,z = [native.run([command])[0]["value"] for command in setup]
        u,v = [base.run([command])[0]["value"] for command in setup]
        inputs_before = list(map(wire.meta,native.run([(13,x),(13,z)])))
        assert all(physical(native,Memory,h,m)==bits for h,m,bits in zip((x,z),inputs_before,(xb,zb),strict=True))
        commands = [(2,16,x,z)]
        frame = op(native,commands)
        status, entries = wire.parse(frame)
        assert status == 0, (xs,zs,status)
        y = entries[0]["value"]
        assert y not in (x,z)
        read = [(13,y),(98,y,0,n),(96,y,0),(13,x),(13,z)]
        readback = op(native,read)
        status, (metadata, values, dtype, mx, mz) = wire.parse(readback)
        assert status == 0
        meta = wire.meta(metadata)
        assert meta["rank"] == len(shape) and tuple(meta["shape"][:len(shape)]) == shape
        assert tuple(meta["strides"][:len(shape)]) == stride, (xs,zs,meta,stride)
        assert (meta["numel"],meta["storage_bytes"],meta["requires_grad"],meta["storage_offset"],meta["version"]) == (n,n*8,0,0,0)
        assert struct.unpack("<4Q",dtype["payload"])[:2] == (4,8)
        assert values["value"] == n and values["payload"] == expected, (xs,zs)
        assert [wire.meta(mx),wire.meta(mz)] == inputs_before
        assert all(physical(native,Memory,h,m)==bits for h,m,bits in zip((x,z),inputs_before,(xb,zb),strict=True))
        assert all(meta[k] not in [m[k] for m in inputs_before] for k in ("storage_id","version_id"))
        assert physical(native,Memory,y,meta) == expected
        native.run([(41,x),(41,z)])
        assert native.run([(98,y,0,n)])[0]["payload"] == expected
        native.run([(41,y)])
        if xs == zs:
            assert op(base,commands) == frame and op(base,read) == readback
            base.run([(41,u),(41,v),(41,y)])
            totals["unchanged_equal_shape_frames"] += 2
        else:
            base.error([(2,16,u,v)],7)
            base.run([(41,u),(41,v)])
            totals["unequal_shape_cases"] += 1
            totals["parent_unequal_shape_refusals"] += 1
        totals["cases"] += 1
        totals["words"] += n
        totals["zero_output_cases"] += n==0
        digest.update(struct.pack("<Q",n))
        digest.update(expected)

    for xs,zs in [((2,3),(2,)), ((2,0,3),(1,4,3)), ((0,),(2,)), ((2,1,3,1),(1,4,2,4))]:
        x,z = [native.run([tensor(wire,k,(0,)*math.prod(k))])[0]["value"] for k in (xs,zs)]
        try:
            torch.ops.aten.copysign.Tensor(torch.empty(xs,dtype=torch.float64),torch.empty(zs,dtype=torch.float64))
        except RuntimeError:
            pass
        else:
            raise AssertionError("oracle accepts supposedly incompatible shapes")
        native.error([(2,16,x,z)],2)
        totals["candidate_incompatible_refusals"] += 1
        native.run([(41,x),(41,z)])

    x = native.run([tensor(wire,(1,),(0,))])[0]["value"]
    z = native.run([LEGACY["f32_tensor"](wire)])[0]["value"]
    for cmd, code in [([(2,16,x,z)],4), ([(2,16,z,x)],4), ([tensor(wire,(1,),(0,),requires=1)],5)]:
        native.error(cmd,code)
        totals["candidate_unrelated_refusals"] += 1
    native.run([(41,x),(41,z)])
    for session in (base,native):
        session.finish()
    evidence = {"status":"SCOPED_PASS","parent_sha256":PARENT,"candidate_sha256":CANDIDATE,
                "oracle_lock_sha256":prov["lock_sha256"],"oracle_checks":prov["checks"],
                "expected_words_sha256":digest.hexdigest(),"physical_storage_exact":True,
                "input_unchanged":True,"fresh_storage_version":True,"survives_input_release":True,
                "canonical_credit":False,**totals}
    (out/"qualification.json").write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n")
    print(json.dumps(evidence,sort_keys=True))


if __name__ == "__main__":
    main()
