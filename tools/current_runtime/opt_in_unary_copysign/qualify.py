#!/usr/bin/env python3
"""Replay the exact predecessor suites and both command families in one session."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

ELF = "5cd57293d53ec23d8207603729516fa98d84e0bb9e2d140a2effbb50ab7d547f"
UNARY_PREPARE = "bf89f2430442707d70992c896ac69ed5d5b37b4e17941a95375757dcd9fb2bb4"
SUITES = {"unary": "2e8fcfd6d4d788a4d6d2d5a073e322a249a5f92ffaf7d5aa90f626228b1ec07c",
          "copysign32": "a4284f9d715958279eff30f134c8d8c040b2745f63cc529a156b94308581e2c6",
          "copysign64": "b7c338bc2f837366453e71964e84426bf4c78c913b5c01155916b6bee4eaef3d"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f"unexpected SHA-256: {path.name}")
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    m.__file__ = str(path)
    if name == "prepare":
        sys.modules[name] = m
    exec(compile(raw, str(path), "exec"), vars(m))
    return m


def replay(a, name, m, extra):
    out = a.output / name
    parent = a.f32_parent if name == "copysign64" else a.parent
    if name == "unary":
        m.EXPECTED_ELF = ELF
    else:
        m.CANDIDATE = ELF
    old = sys.argv
    try:
        sys.argv = [m.__file__, "--repository", str(a.repository), "--parent", str(parent),
                    "--candidate", str(a.candidate), "--output", str(out), *extra]
        m.main()
    finally:
        sys.argv = old
    raw = (out / "qualification.json").read_bytes()
    result = json.loads(raw)
    assert result["status"] == "SCOPED_PASS" and result["candidate_sha256"] == ELF
    return {**{k: v for k, v in result.items() if k != "rows"}, "report_sha256": sha(raw)}


def mixed(a, unary, f32, f64):
    spec = importlib.util.spec_from_file_location("mixed_wire", a.repository / "tests/integer_tensor_review/wire/protocol.py")
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    sys.path.insert(0, str(a.repository / "tests"))
    from half_logical.native import Memory

    ops = {name: unary.load_exact(getattr(a, f"{name}_dir") / "qualify.py", pin, "mixed_" + name)
           for name, pin in unary.SUITES.items()}
    data32 = (0, 0x80000000, 1, 0x7F812345, 0xFFC76543, 0x40200000, 0xFF800000, 0x7F7FFFFF)
    sign32 = (0x80000000, 0, 0xFF800000, 0xBF800000, 0, 0x80000000, 0, 0xFFC04000)
    data64 = (0, 0x8000000000000000, 0x7FF0000000001234, 0xFFEFFFFFFFFFFFFF, 1, 0x7FF0000000000000)
    sign64 = (0x8000000000000000, 0, 0xBFF0000000000000, 0, 0xFFF8000000000000, 0)
    shape32, shape64 = (2, 4), (2, 3)
    [(cs32, c32)], _ = f32.observe_torch(a.repository, a.output, [(shape32,data32,sign32,False)])
    [(cs64, c64)], _ = f64.observe_torch(a.repository, a.output, [(shape64,data64,sign64,False)])
    after32 = struct.unpack("<8I",c32)
    refs = {name: m.observe_torch(a.repository,a.output,[(shape32,after32)])[0][0]
            for name,m in ops.items()}
    directory = a.output / "mixed"
    s = wire.Session(a.candidate, directory=directory)
    x32,z32 = f32.create_pair(wire,s,shape32,data32,sign32,False)
    x64,z64 = f64.create_pair(wire,s,shape64,data64,sign64,False)
    storage, version = set(), set()

    def check(token, raw, count, shape, stride, suite):
        meta, payload = s.run([(13,token),(98 if suite is f64 else 39,token,0,count)])
        m = wire.meta(meta)
        assert (m["rank"], tuple(m["shape"][:len(shape)]), m["numel"]) == (len(shape), shape, count)
        assert m["version"] == m["storage_offset"] == m["requires_grad"] == 0
        if stride is not None:
            assert tuple(m["strides"][:len(shape)]) == stride
        assert payload["payload"] == raw and suite.physical(s,Memory,token,m) == raw
        assert m["storage_id"] not in storage and m["version_id"] not in version
        storage.add(m["storage_id"])
        version.add(m["version_id"])
        return m

    inputs = [(x32,data32,shape32,f32),(z32,sign32,shape32,f32),
              (x64,data64,shape64,f64),(z64,sign64,shape64,f64)]
    initial = [check(t,p.raw(w),len(w),sh,None,p) for t,w,sh,p in inputs]
    c = s.run([(2,16,x32,z32)])[0]["value"]
    r, d, q, g = [x["value"] for x in s.run([(3,17,c),(2,16,x64,z64),(3,16,c),(3,18,c)])]
    assert len({x32,z32,x64,z64,c,r,d,q,g}) == 9
    outputs = [(c,c32,8,shape32,cs32,f32),(d,c64,6,shape64,cs64,f64)]
    outputs += [(t,refs[n][1],8,shape32,refs[n][0],f32)
                for n,t in (("reciprocal",r),("square",q),("sign",g))]
    for args in outputs:
        check(*args)
    for (token, words, _, suite), meta in zip(inputs,initial,strict=True):
        (mr,) = s.run([(13,token)])
        assert wire.meta(mr) == meta
        assert suite.physical(s,Memory,token,meta) == suite.raw(words)
    s.run([(41,x32),(41,z32),(41,x64),(41,z64)])
    s.run([(41,c)])
    for token, raw, count, _, _, suite in outputs[1:]:
        (p,) = s.run([(98 if suite is f64 else 39,token,0,count)])
        assert p["payload"] == raw
    s.run([(41,r),(41,d),(41,q),(41,g)])
    s.finish()
    receipt = json.loads((directory/"transport.json").read_bytes())
    assert receipt["executable_sha256"] == ELF and receipt["exit_code"] == 0
    assert receipt["responses_before_stdin_eof"]
    return {"operations": 5,"words": 38,"frames": receipt["frames"], "inputs_unchanged": True,
            "fresh_physical_storage": True, "survives_release": True,
            **{n+"_sha256":sha((directory/(n+".frames.bin")).read_bytes()) for n in ("requests","responses")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("repository","parent","f32-parent","candidate","output","unary-dir","copysign32-dir",
                 "copysign64-dir","square-dir","reciprocal-dir","sign-dir"):
        ap.add_argument(f"--{name}", required=True, type=lambda p: Path(p).resolve())
    a = ap.parse_args()
    assert not sys.flags.optimize and sha(a.candidate.read_bytes()) == ELF
    load("prepare",a.unary_dir/"prepare.py",UNARY_PREPARE)
    modules={name:load("verify_"+name,getattr(a,name+"_dir")/"qualify.py",pin) for name,pin in SUITES.items()}
    a.output.mkdir(parents=True,exist_ok=False)
    args = [item for name in ("square","reciprocal","sign")
            for item in (f"--{name}-dir", str(getattr(a,name+"_dir")))]
    results = {name:replay(a,name,m,args if name=="unary" else []) for name,m in modules.items()}
    cross = mixed(a,modules["unary"],modules["copysign32"],modules["copysign64"])
    output={"status":"SCOPED_PASS","canonical_credit":False,"candidate_sha256":ELF,
            "predecessor_suites":results,"cross_family_session":cross}
    (a.output/"qualification.json").write_text(json.dumps(output,sort_keys=True,indent=2)+"\n")
    print(json.dumps(cross,sort_keys=True))


if __name__ == "__main__":
    main()
