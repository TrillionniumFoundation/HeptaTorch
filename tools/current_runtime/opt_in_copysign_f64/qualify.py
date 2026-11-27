#!/usr/bin/env python3
"""Execute ordinary copysign requests on independent PyTorch and native images."""
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


PARENT = "5e170a6624622fac33ef47fa10e57633a5b4a1587d67cd6726144ae87c28a9c9"
CANDIDATE = "f46436d6f226e7bb257128cec57649c2a24b31f9f4238fcbc7ca57bc7b8774ed"
WORDS = (0, 0x8000000000000000, 1, 0x800FFFFFFFFFFFFF, 0x0010000000000000,
         0x0010000000000001, 0x3FD0000000000000, 0x3FEFFFFFFFFFFFFF, 0x3FF0000000000000,
         0x3FF0000000000001, 0x4000000000000000, 0xC004000000000000,
         0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF, 0x7FF0000000000000, 0xFFF0000000000000,
         0x7FF0123456789ABC, 0xFFF0123456789ABC, 0x7FF8000000000001, 0xFFF8000000000001,
         0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw(words):
    words = tuple(words)
    return struct.pack("<" + "Q" * len(words), *words)


def cases():
    for left, right in itertools.product(WORDS, repeat=2):
        yield (), (left,), (right,), False
    sizes = ((0,), (2, 0, 3), (0, 2, 3, 4), (1,), (7,), (8,), (9,), (15,), (16,), (17,),
             (31,), (32,), (33,), (257,), (2, 3), (2, 1, 17), (2, 3, 2, 7), (1, 1, 1, 4096))
    rng = random.Random(77616)
    for shape in sizes:
        n = math.prod(shape)
        lhs = tuple(WORDS[i % len(WORDS)] if i % 2 else rng.getrandbits(64) for i in range(n))
        rhs = tuple(WORDS[(i+5) % len(WORDS)] if i % 2 else rng.getrandbits(64) for i in range(n))
        yield shape, lhs, rhs, False
        yield shape, lhs, lhs, True


def tensor(wire, shape, words, requires=0):
    bits = raw(words)
    # One record. Configuration begins at 96 and ends at the byte payload offset 160.
    config = struct.pack("<8Q", len(shape), requires, *shape, *([0] * (4 - len(shape))), 160, len(bits))
    return (97, 4, wire.Payload(config + bits))


def f32_tensor(wire):
    return (1, 1, 0, 3, 0, 0, 0, wire.Payload(struct.pack("<3I", 0x40000000, 0x80000000, 0x7F812345),4))


def observe_torch(repo, output, inputs):
    os.chdir(repo)
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    saved_argv = sys.argv
    try:
        sys.argv = [str(repo / "tests/pytorch_oracle.py"), str(output / "NO_NATIVE"),
                    str(output / "oracle-error.json"), str(repo / "compatibility/pytorch-oracle-lock.json")]
        ctx = runpy.run_path(sys.argv[0], run_name="heptatorch_copysign_reference_only")
    finally:
        sys.argv = saved_argv
    assert ctx["PROVENANCE"]["status"] == "PASS"
    torch = ctx["torch"]
    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
    assert torch.get_num_threads() == 1
    out = []
    for shape, words, signs, alias in inputs:
        bits = raw(words)
        seed = torch.frombuffer(bytearray(bits), dtype=torch.float64) if bits else torch.empty(0, dtype=torch.float64)
        x = seed.reshape(shape).clone()
        sign_bits = raw(signs)
        sign_seed = torch.frombuffer(bytearray(sign_bits), dtype=torch.float64) if sign_bits else torch.empty(0, dtype=torch.float64)
        z = x if alias else sign_seed.reshape(shape).clone()
        y = torch.ops.aten.copysign.Tensor(x, z)
        assert bytes(z.untyped_storage()) == sign_bits
        assert y.untyped_storage()._cdata != z.untyped_storage()._cdata
        output_raw = bytes(y.untyped_storage())
        assert tuple(y.shape) == shape and y.dtype == x.dtype and not y.requires_grad
        assert y.storage_offset() == y._version == 0
        assert y.untyped_storage()._cdata != x.untyped_storage()._cdata
        assert bytes(x.untyped_storage()) == bits
        assert output_raw == raw((w & 0x7fffffffffffffff) | (s & 0x8000000000000000) for w, s in zip(words, signs, strict=True))
        out.append((tuple(y.stride()), output_raw))
    return out, ctx["PROVENANCE"]


def physical(session, decoder, token, meta):
    owner = decoder(session.owner()["bytes"])
    node = owner.node(token)
    storage, version = owner.row(2, node[27]), owner.row(3, node[28])
    assert (node[27], node[28], node[29], storage[3], version[2]) == (
        meta["storage_id"], meta["version_id"], meta["storage_offset"], meta["storage_bytes"], meta["version"])
    assert node[72] == 4 and all(node[i] == 0 for i in (4, 5, 6, 7, 45, 48, 49, 50, 51, 83))
    start = owner.start + storage[2]
    value = owner.raw[start:start + storage[3]]
    assert len(value) == storage[3]
    return value


def create_pair(wire, session, shape, words, signs, alias):
    x = session.run([tensor(wire, shape, words)])[0]["value"]
    z = x if alias else session.run([tensor(wire, shape, signs)])[0]["value"]
    return x,z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("repository", "parent", "candidate", "output"):
        parser.add_argument("--" + label, required=True, type=Path)
    args = parser.parse_args()
    assert not sys.flags.optimize
    repo, parent, candidate, out = [getattr(args, name).resolve() for name in ("repository", "parent", "candidate", "output")]
    assert sha(parent.read_bytes()) == PARENT and sha(candidate.read_bytes()) == CANDIDATE
    out.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("copysign_wire", repo/"tests/integer_tensor_review/wire/protocol.py")
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    sys.path.insert(0, str(repo/"tests"))
    from half_logical.native import Memory
    inputs = list(cases())
    oracle, prov = observe_torch(repo, out, inputs)
    base = wire.Session(parent, directory=out/"parent")
    native = wire.Session(candidate, directory=out/"candidate")
    total = 0
    digest = hashlib.sha256()
    for (shape, words, signs, alias), (stride, expected) in zip(inputs, oracle, strict=True):
        n = len(words)
        total += n
        x, z = create_pair(wire, base, shape, words, signs, alias)
        base.error([(2, 16, x, z)], 4)
        base.run([(41, x)] + ([] if alias else [(41, z)]))
        x, z = create_pair(wire, native, shape, words, signs, alias)
        xraw, zraw = raw(words), raw(signs)
        old_x, old_z = map(wire.meta, native.run([(13, x), (13, z)]))
        assert physical(native, Memory, x, old_x) == xraw
        assert physical(native, Memory, z, old_z) == zraw
        y = native.run([(2, 16, x, z)])[0]["value"]
        meta, result, xx, zz = native.run([(13,y), (98,y,0,n), (13,x), (13,z)])
        meta = wire.meta(meta)
        assert y not in (x,z)
        for key in ("storage_id", "version_id"):
            assert meta[key] not in (old_x[key], old_z[key])
        assert (meta["rank"], tuple(meta["shape"][:len(shape)]), tuple(meta["strides"][:len(shape)])) == (len(shape),shape,stride)
        assert meta["numel"] == n and meta["storage_bytes"] == 8*n
        assert meta["requires_grad"] == meta["storage_offset"] == meta["version"] == 0
        assert result["value"] == n and result["payload"] == expected, (shape, result["payload"][:64].hex(), expected[:64].hex())
        dtype = native.run([(96,y,0)])[0]["payload"]
        assert struct.unpack("<4Q", dtype)[:2] == (4, 8)
        assert wire.meta(xx) == old_x and wire.meta(zz) == old_z
        assert physical(native, Memory, x, old_x) == xraw and physical(native, Memory, z, old_z) == zraw
        assert physical(native, Memory, y, meta) == expected
        native.run([(41, x)] + ([] if alias else [(41, z)]))
        assert native.run([(98,y,0,n)])[0]["payload"] == expected
        native.run([(41,y)])
        digest.update(struct.pack("<Q", n))
        digest.update(expected)

    x, z, vec = [native.run([c])[0]["value"] for c in [
        tensor(wire, (2,2), WORDS[:4]), tensor(wire, (2,2), WORDS[4:8]),
        tensor(wire, (2,), WORDS[:2])]]
    f32 = native.run([f32_tensor(wire)])[0]["value"]
    native.error([(2,16,x,f32)], 4)
    native.error([(2,16,f32,x)], 4)
    native.error([(2,16,vec,z)], 7)
    # The unchanged upstream F64 constructor refuses grad/views on this image,
    # independently of the copysign geometry and no-grad guards.
    old_root = base.run([tensor(wire,(2,2),WORDS[:4])])[0]["value"]
    for session, root in ((base, old_root), (native, x)):
        session.error([tensor(wire, (2,2), WORDS[:4], requires=1)], 5)
        for shape, stride, offset in [((2,2),(1,2),0), ((2,),(1,),1)]:
            config = struct.pack("<10Q", len(shape), *shape, *([0]*(4-len(shape))),
                                 *stride, *([0]*(4-len(shape))), offset)
            session.error([(111,root,wire.Payload(config),80)], 7)
    base.run([(41,old_root)])
    cfg = struct.pack("<8Q", 2, 0, 2, 2, 0, 0, 160, 32)
    i64 = native.run([(97,2,wire.Payload(cfg + struct.pack("<4q", 1, -1, 0, 2)))])[0]["value"]
    native.error([(2,16,x,i64)], 4)
    native.error([(2,16,i64,x)], 4)
    native.run([(41,x),(41,z),(41,vec),(41,i64),(41,f32)])
    native.finish()
    base.finish()
    old, new = [wire.Session(exe, directory=out/name) for name, exe in (("prior-parent",parent),("prior-candidate",candidate))]
    frames = 0

    def same(commands):
        nonlocal frames
        request = wire.packet(commands)
        expected, observed = old.exchange(request), new.exchange(request)
        assert expected == observed and wire.parse(expected)[0] == 0
        frames += 1
        return wire.parse(expected)[1]

    lhs, rhs = same([f32_tensor(wire),f32_tensor(wire)])
    x,z = lhs["value"], rhs["value"]
    for op in (1,2,3,4,16):
        result = same([(2,op,x,z)])[0]["value"]
        same([(13,result),(39,result,0,3),(41,result)])
    same([(41,x),(41,z)])
    left = same([tensor(wire, (2,), (0x4000000000000000,0x4008000000000000))])[0]["value"]
    right = same([tensor(wire, (2,), (0x4010000000000000,0x4000000000000000))])[0]["value"]
    for op in (1,2,3,4):
        result = same([(2,op,left,right)])[0]["value"]
        same([(13,result),(98,result,0,2),(41,result)])
    same([(41,left),(41,right)])
    old.finish()
    new.finish()
    evidence = {"status":"SCOPED_PASS","parent_sha256":PARENT,"candidate_sha256":CANDIDATE,
                "oracle_lock_sha256":prov["lock_sha256"],"oracle_checks":prov["checks"],
                "cases":len(inputs),"elements":total,"identical_operand_cases":sum(i[3] for i in inputs),
                "parent_untouched_error_frames":len(inputs),"candidate_operator_untouched_error_frames":5,
                "upstream_constructor_error_frames_across_both_images":6,
                "prior_binary_response_frames":frames,"reference_words_sha256":digest.hexdigest(),
                "physical_storage_exact":True,"fresh_storage_version":True,"input_unchanged":True,
                "output_survived_input_release":True,"canonical_credit":False}
    (out/"qualification.json").write_text(json.dumps(evidence,sort_keys=True,indent=2)+"\n")
    print(json.dumps(evidence,sort_keys=True))


if __name__ == "__main__":
    main()
