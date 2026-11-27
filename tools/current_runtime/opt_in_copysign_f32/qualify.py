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


PARENT = "ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e"
CANDIDATE = "5e170a6624622fac33ef47fa10e57633a5b4a1587d67cd6726144ae87c28a9c9"
WORDS = (0, 0x80000000, 1, 0x807FFFFF, 0x00800000, 0x217FFFFF, 0x21800000,
         0x21800001, 0x26000000, 0x3F7FFFFF, 0x3F800001, 0x3FC00000, 0xC0200000,
         0x5F7FFFFF, 0x5F800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000,
         0x7F812345, 0xFF812345, 0x7FC76543, 0xFFC76543)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw(words):
    words = tuple(words)
    return struct.pack("<" + "I" * len(words), *words)


def cases():
    for left, right in itertools.product(WORDS, repeat=2):
        yield (), (left,), (right,), False
    sizes = ((0,), (2, 0, 3), (0, 2, 3, 4), (1,), (7,), (8,), (9,), (15,), (16,), (17,),
             (31,), (32,), (33,), (257,), (2, 3), (2, 1, 17), (2, 3, 2, 7), (1, 1, 1, 4096))
    rng = random.Random(77616)
    for shape in sizes:
        n = math.prod(shape)
        lhs = tuple(WORDS[i % len(WORDS)] if i % 2 else rng.getrandbits(32) for i in range(n))
        rhs = tuple(WORDS[(i+5) % len(WORDS)] if i % 2 else rng.getrandbits(32) for i in range(n))
        yield shape, lhs, rhs, False
        yield shape, lhs, lhs, True


def tensor(wire, shape, words, requires=0):
    return (1, len(shape), requires, *shape, *([0] * (4 - len(shape))),
            wire.Payload(raw(words), 4))


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
        seed = torch.frombuffer(bytearray(bits), dtype=torch.float32) if bits else torch.empty(0, dtype=torch.float32)
        x = seed.reshape(shape).clone()
        sign_bits = raw(signs)
        sign_seed = torch.frombuffer(bytearray(sign_bits), dtype=torch.float32) if sign_bits else torch.empty(0, dtype=torch.float32)
        z = x if alias else sign_seed.reshape(shape).clone()
        y = torch.ops.aten.copysign.Tensor(x, z)
        assert bytes(z.untyped_storage()) == sign_bits
        assert y.untyped_storage()._cdata != z.untyped_storage()._cdata
        output_raw = bytes(y.untyped_storage())
        assert tuple(y.shape) == shape and y.dtype == x.dtype and not y.requires_grad
        assert y.storage_offset() == y._version == 0
        assert y.untyped_storage()._cdata != x.untyped_storage()._cdata
        assert bytes(x.untyped_storage()) == bits
        assert output_raw == raw((w & 0x7fffffff) | (s & 0x80000000) for w, s in zip(words, signs, strict=True))
        out.append((tuple(y.stride()), output_raw))
    return out, ctx["PROVENANCE"]


def physical(session, decoder, token, meta):
    owner = decoder(session.owner()["bytes"])
    node = owner.node(token)
    storage, version = owner.row(2, node[27]), owner.row(3, node[28])
    assert (node[27], node[28], node[29], storage[3], version[2]) == (
        meta["storage_id"], meta["version_id"], meta["storage_offset"], meta["storage_bytes"], meta["version"])
    assert node[72] == 0 and all(node[i] == 0 for i in (4, 5, 6, 7, 45, 48, 49, 50, 51, 83))
    start = owner.start + storage[2]
    value = owner.raw[start:start + storage[3]]
    assert len(value) == storage[3]
    return value


def create_pair(wire, session, shape, words, signs, alias):
    rows = session.run([tensor(wire, shape, words)] + ([] if alias else [tensor(wire, shape, signs)]))
    return rows[0]["value"], rows[-1]["value"]


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
        meta, result, xx, zz = native.run([(13,y), (39,y,0,n), (13,x), (13,z)])
        meta = wire.meta(meta)
        assert y not in (x,z)
        for key in ("storage_id", "version_id"):
            assert meta[key] not in (old_x[key], old_z[key])
        assert (meta["rank"], tuple(meta["shape"][:len(shape)]), tuple(meta["strides"][:len(shape)])) == (len(shape),shape,stride)
        assert meta["numel"] == n and meta["storage_bytes"] == 4*n
        assert meta["requires_grad"] == meta["storage_offset"] == meta["version"] == 0
        assert result["value"] == n and result["payload"] == expected, (shape, result["payload"][:64].hex(), expected[:64].hex())
        dtype = native.run([(96,y,0)])[0]["payload"]
        assert struct.unpack("<4Q", dtype)[:2] == (0, 4)
        assert wire.meta(xx) == old_x and wire.meta(zz) == old_z
        assert physical(native, Memory, x, old_x) == xraw and physical(native, Memory, z, old_z) == zraw
        assert physical(native, Memory, y, meta) == expected
        native.run([(41, x)] + ([] if alias else [(41, z)]))
        assert native.run([(39,y,0,n)])[0]["payload"] == expected
        native.run([(41,y)])
        digest.update(struct.pack("<Q", n))
        digest.update(expected)

    x, z, g, vec = [r["value"] for r in native.run([
        tensor(wire, (2,2), WORDS[:4]), tensor(wire, (2,2), WORDS[4:8]),
        tensor(wire, (2,2), WORDS[:4], requires=1), tensor(wire, (2,), WORDS[:2])])]
    native.error([(2,16,g,z)], 5)
    native.error([(2,16,x,g)], 5)
    native.error([(2,16,vec,z)], 7)
    for shape, stride, offset in [((2,2),(1,2),0), ((2,),(1,),1)]:
        config = struct.pack("<10Q", len(shape), *shape, *([0]*(4-len(shape))),
                             *stride, *([0]*(4-len(shape))), offset)
        view = native.run([(111,x,wire.Payload(config),80)])[0]["value"]
        peer = z if len(shape) == 2 else vec
        native.error([(2,16,view,peer)], 7)
        native.error([(2,16,peer,view)], 7)
        native.run([(41,view)])
    # Exact typed constructor: i64 2×2 config, payload begins after the 64-byte descriptor.
    cfg = struct.pack("<8Q", 2, 0, 2, 2, 0, 0, 160, 32)
    i64 = native.run([(97,2,wire.Payload(cfg + struct.pack("<4q", 1, -1, 0, 2)))])[0]["value"]
    native.error([(2,16,x,i64)], 4)
    native.error([(2,16,i64,x)], 4)
    native.run([(41,x),(41,z),(41,g),(41,vec),(41,i64)])
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

    for op in (1,2,3,4):
        lhs, rhs = same([tensor(wire,(2,),(0x40000000,0x40400000)),tensor(wire,(2,),(0x40800000,0x40000000))])
        x,z = lhs["value"], rhs["value"]
        result = same([(2,op,x,z)])[0]["value"]
        same([(13,result),(39,result,0,2),(41,result),(41,x),(41,z)])
    old.finish()
    new.finish()
    evidence = {"status":"SCOPED_PASS","parent_sha256":PARENT,"candidate_sha256":CANDIDATE,
                "oracle_lock_sha256":prov["lock_sha256"],"oracle_checks":prov["checks"],
                "cases":len(inputs),"elements":total,"identical_operand_cases":sum(i[3] for i in inputs),
                "parent_untouched_error_frames":len(inputs),"candidate_untouched_error_frames":9,
                "prior_binary_response_frames":frames,"reference_words_sha256":digest.hexdigest(),
                "physical_storage_exact":True,"fresh_storage_version":True,"input_unchanged":True,
                "output_survived_input_release":True,"canonical_credit":False}
    (out/"qualification.json").write_text(json.dumps(evidence,sort_keys=True,indent=2)+"\n")
    print(json.dumps(evidence,sort_keys=True))


if __name__ == "__main__":
    main()
