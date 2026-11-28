#!/usr/bin/env python3
"""Replay the published rounding and reduction suites on the same HIA ELF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ELF = "caca1bfb706beb5eafc73dc2832c5731fe204a8801aef340c2b42d0b29691e34"
TOOLS = {
    "rounding_verifier": ("bbb6053445f2a4293347d12f7fe840611b3fa1db6a81dd632ea7a650c30d7970",
                          "c5f542e32675e3e058ca409a5bda57f092f344b8d0a58903797731937f6652c6"),
    "reduction_verifier": ("9e49136bd3f985e4b3ccf6bfad26d6979a005cc22dd4d4b96cadf0e1b339a8bf",
                           "b7ed53dfefda67ac0ecbe6f0799f0c013f426aca55cb645162d14625c11a756a"),
}
PROD = "63cccdb9af6dcd5d848850e0ba25c1160c3440d0c004bb0283edbd86c8e396b5"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_exact(path, pin):
    raw = path.read_bytes()
    if sha(raw) != pin:
        raise ValueError(f"unexpected pin for {path.name}")
    return raw


def mixed(a, prod_path, f32_path):
    p = runpy.run_path(str(prod_path))
    f = runpy.run_path(str(f32_path))
    os.chdir(a.repository)
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    original = sys.argv
    try:
        oracle = a.repository / "tests/pytorch_oracle.py"
        lock = a.repository / "compatibility/pytorch-oracle-lock.json"
        sys.argv = [str(oracle), str(a.output / "NO_NATIVE"), str(a.output / "oracle-error.json"), str(lock)]
        oracle = runpy.run_path(str(oracle), run_name="combined_reductions_oracle")
    finally:
        sys.argv = original
    torch = oracle["torch"]
    assert oracle["PROVENANCE"]["status"] == "PASS"
    assert torch.get_num_threads() == 1 and torch.backends.cpu.get_cpu_capability() == "AVX2"
    wire = p["load"](a.repository / "tests/integer_tensor_review/wire/protocol.py", "mixed_wire", p["WIRE"])
    sys.path.insert(0, str(a.repository / "tests"))
    p["bind"](a.repository / "tests/half_logical/native.py", p["OWNER"])
    from half_logical.native import Memory

    logs = a.output / "mixed"
    s = wire.Session(a.candidate, directory=logs)
    it = torch.tensor([[-9, 4, -2], [3, 7, -11]], dtype=torch.int64)
    bt = torch.tensor([[True, True], [False, True]], dtype=torch.bool)
    words = (0xBFC00000, 0x40080000, 0x3E800000)
    ft = torch.frombuffer(bytearray(f["raw"](words)), dtype=torch.float32).clone()
    ix = p["create"](wire, s, 2, tuple(it.shape), bytes(it.untyped_storage()))
    bx = p["create"](wire, s, 1, tuple(bt.shape), bytes(bt.untyped_storage()))
    fx = s.run([f["tensor"](wire, tuple(ft.shape), words)])[0]["value"]
    at, yt, mt = torch.max(it), torch.ceil(ft), torch.min(bt)
    ax, yx, mx = [r["value"] for r in s.run([(3,24,ix),(3,21,fx),(3,23,bx)])]
    pt, qt, rt = torch.prod(at), torch.square(yt), torch.prod(mt)
    px, qx, rx = [r["value"] for r in s.run([(3,22,ax),(3,16,yx),(3,22,mx)])]
    ins = [(ix,it),(bx,bt),(fx,ft)]
    outs = [(ax,at),(yx,yt),(mx,mt),(px,pt),(qx,qt),(rx,rt)]
    storage, version, metas = set(), set(), {}
    for token, ref in ins + outs:
        meta = p["meta"](wire,s,token)
        assert meta["numel"] == ref.numel() and meta["rank"] == ref.ndim
        assert tuple(meta["shape"][:ref.ndim]) == ref.shape
        assert tuple(meta["strides"][:ref.ndim]) == ref.stride()
        assert meta["storage_offset"] == meta["requires_grad"] == meta["version"] == 0
        metas[token] = meta
        assert meta["storage_id"] not in storage and meta["version_id"] not in version
        storage.add(meta["storage_id"])
        version.add(meta["version_id"])
        raw, node = p["physical"](s,Memory,token,meta)
        assert node[72] == {torch.float32:0, torch.bool:1, torch.int64:2}[ref.dtype]
        assert raw == bytes(ref.untyped_storage())
        (r,) = s.run([(39 if ref.dtype == torch.float32 else 98,token,0,ref.numel())])
        assert r["payload"] == raw
    s.run([(41,h) for h,_ in ins+outs[:3]])
    for token, ref in outs[3:]:
        assert p["meta"](wire,s,token) == metas[token]
        assert p["physical"](s,Memory,token,metas[token])[0] == bytes(ref.untyped_storage())
        (r,) = s.run([(39 if ref.dtype == torch.float32 else 98,token,0,ref.numel())])
        assert r["payload"] == bytes(ref.untyped_storage())
    s.run([(41,h) for h,_ in outs[3:]])
    s.finish()
    t = json.loads((logs/"transport.json").read_bytes())
    assert t["executable_sha256"] == ELF and t["exit_code"] == 0 and t["responses_before_stdin_eof"]
    return {"operations":len(outs),"words":sum(ref.numel() for _,ref in outs),"frames":t["frames"],
            "new_physical_storage_and_version":True,"survives_release":True,
            **{n+"_sha256":sha((logs/(n+".frames.bin")).read_bytes()) for n in ("requests","responses")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("repository","candidate","canonical","rounding-parent","reduction-parent","install-report",
                 "prod-verifier","rounding-verifier","reduction-verifier","output"):
        ap.add_argument("--"+name, type=lambda p: Path(p).resolve(), required=True)
    a = ap.parse_args()
    assert not sys.flags.optimize and sha(a.candidate.read_bytes()) == ELF
    prod_raw = read_exact(a.prod_verifier, PROD)
    verifier = {name: (read_exact(getattr(a,name),digest),old) for name,(digest,old) in TOOLS.items()}
    a.output.mkdir(parents=True, exist_ok=False)
    prod_path = a.output / "prod.py"
    prod_path.write_bytes(prod_raw)
    reports = {}
    for suite, (raw, old) in verifier.items():
        script = a.output / (suite+".py")
        key = ('CANDIDATE = "'+old+'"').encode()
        assert raw.count(key) == 1
        script.write_bytes(raw.replace(key, ('CANDIDATE = "'+ELF+'"').encode()))
        prefix = suite.split("_")[0]
        out = a.output / prefix
        extra = [] if prefix == "rounding" else ["--canonical",a.canonical,"--install-report",a.install_report,
                                                  "--prod-verifier",prod_path]
        subprocess.run([sys.executable,"-I","-B",str(script),"--repository",str(a.repository),"--parent",
                        str(getattr(a,prefix+"_parent")),"--candidate",str(a.candidate),"--output",str(out),
                        *map(str,extra)], cwd=a.repository, check=True, timeout=300)
        result = (out/"qualification.json").read_bytes()
        j = json.loads(result)
        assert j["status"] == "SCOPED_PASS" and j["candidate_sha256"] == ELF
        reports[prefix] = {**{k:v for k,v in j.items() if k!="rows"},"report_sha256":sha(result)}
    cross = mixed(a, prod_path, a.output / "rounding_verifier.py")
    evidence = {"status":"SCOPED_PASS","canonical_credit":False,"candidate_sha256":ELF,
                "predecessor_suites":reports,"same_session":cross}
    (a.output/"qualification.json").write_text(json.dumps(evidence,sort_keys=True,indent=2)+"\n")
    print(json.dumps(cross,sort_keys=True))


if __name__ == "__main__":
    main()
