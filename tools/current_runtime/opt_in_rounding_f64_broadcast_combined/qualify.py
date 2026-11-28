#!/usr/bin/env python3
"""Run two original differential matrices on a pinned combined native image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ELF = "f3081969dbfbf47a71dc958494d9cd094e94d6e14acfab0c235dc3c741985760"
PINS = {
    "opt_in_rounding_f32": "bbb6053445f2a4293347d12f7fe840611b3fa1db6a81dd632ea7a650c30d7970",
    "opt_in_copysign_f64_broadcast": "83687ed8713fd9d07353eea5dfc065fabc2266c04c91d36fd8748990fc25ff10",
    "opt_in_copysign_f64": "b7c338bc2f837366453e71964e84426bf4c78c913b5c01155916b6bee4eaef3d",
}
SUITES = ("opt_in_rounding_f32", "opt_in_copysign_f64_broadcast")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stage(a):
    root = a.output / "pinned-suites"
    for name, expected in PINS.items():
        src = getattr(a, name) / "qualify.py"
        raw = src.read_bytes()
        if sha(raw) != expected:
            raise ValueError(f"unexpected verifier pin: {name}")
        dst = root / name / "qualify.py"
        dst.parent.mkdir(parents=True)
        dst.write_bytes(raw)
    return root


def replay(a, staged, name, parent):
    path = staged / name / "qualify.py"
    module = runpy.run_path(str(path))
    old_pin = module["CANDIDATE"]
    # Only the executable identity changes; F64 shape positives/negatives
    # are the new #96 matrix, not the intentionally superseded #90 guards.
    raw = path.read_bytes()
    old = f'CANDIDATE = "{old_pin}"'.encode()
    new = f'CANDIDATE = "{ELF}"'.encode()
    assert raw.count(old) == 1
    remapped = path.with_name("integrated.py")
    remapped.write_bytes(raw.replace(old, new))
    out = a.output / name
    subprocess.run([sys.executable, "-I", "-B", str(remapped), "--repository", str(a.repository),
                    "--parent", str(parent), "--candidate", str(a.candidate), "--output", str(out)],
                   cwd=a.repository, check=True, timeout=300)
    report = (out / "qualification.json").read_bytes()
    obj = json.loads(report)
    assert obj["status"] == "SCOPED_PASS" and obj["candidate_sha256"] == ELF
    return {**{k: v for k, v in obj.items() if k != "rows"}, "report_sha256": sha(report)}


def mixed(a, stage):
    import importlib.util
    import struct
    f32 = runpy.run_path(str(stage / SUITES[0] / "qualify.py"))
    f64 = runpy.run_path(str(stage / "opt_in_copysign_f64" / "qualify.py"))
    os.chdir(a.repository)
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    original = sys.argv
    try:
        lock = a.repository / "compatibility/pytorch-oracle-lock.json"
        oracle = a.repository / "tests/pytorch_oracle.py"
        sys.argv = [str(oracle), str(a.output / "NO_NATIVE"), str(a.output / "oracle-error.json"), str(lock)]
        ctx = runpy.run_path(str(oracle), run_name="heptatorch_combined_forward_oracle")
    finally:
        sys.argv = original
    torch = ctx["torch"]
    assert ctx["PROVENANCE"]["status"] == "PASS"
    assert torch.backends.cpu.get_cpu_capability() == "AVX2" and torch.get_num_threads() == 1
    spec = importlib.util.spec_from_file_location("wire", a.repository / "tests/integer_tensor_review/wire/protocol.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    sys.path.insert(0, str(a.repository / "tests"))
    from half_logical.native import Memory

    d = a.output / "mixed"
    s = w.Session(a.candidate, directory=d)
    a0 = (0x3ff8000000000000, 0x4002000000000000)
    b0 = (0, 0x8000000000000000, 0xfff0000000000000)
    c0 = (0x8000000000000000,)
    f0 = (0x3fc00000, 0xc0200000, 0x3e800000)
    f1 = (0, 0x80000000, 0)
    a_t0, b_t0, c_t0, f_t0, f_t1 = [
        s.run([command])[0]["value"] for command in (
            f64["tensor"](w, (2,1), a0), f64["tensor"](w, (3,), b0), f64["tensor"](w, (), c0),
            f32["tensor"](w, (3,), f0), f32["tensor"](w, (3,), f1))]

    def ref(words, shape, dtype, raw):
        return torch.frombuffer(bytearray(raw(words)), dtype=dtype).reshape(shape).clone()

    ta, tb, tc = [ref(words, shape, torch.float64, f64["raw"]) for words, shape in [(a0,(2,1)),(b0,(3,)),(c0,())]]
    tf, ts = [ref(x,(3,),torch.float32, f32["raw"]) for x in (f0,f1)]
    bt = torch.copysign(ta, tb)
    ct = torch.copysign(bt, tc)
    ft = torch.floor(tf)
    gt = torch.square(ft)
    st = torch.copysign(tf, ts)
    bid, fid, sid = [x["value"] for x in s.run([(2,16,a_t0,b_t0),(3,20,f_t0),(2,16,f_t0,f_t1)])]
    cid, gid = [x["value"] for x in s.run([(2,16,bid,c_t0),(3,16,fid)])]
    storage, version = set(), set()

    def inspect(token, value):
        n = value.numel()
        wide = value.dtype == torch.float64
        m, raw = s.run([(13,token),(98 if wide else 39,token,0,n)])
        meta = w.meta(m)
        assert (meta["rank"], tuple(meta["shape"][:value.ndim]), tuple(meta["strides"][:value.ndim]), meta["numel"]) == (
            value.ndim, tuple(value.shape), tuple(value.stride()), n)
        assert meta["version"] == meta["storage_offset"] == meta["requires_grad"] == 0
        assert raw["payload"] == bytes(value.untyped_storage())
        assert (f64 if wide else f32)["physical"](s,Memory,token,meta) == raw["payload"]
        assert meta["storage_id"] not in storage and meta["version_id"] not in version
        storage.add(meta["storage_id"])
        version.add(meta["version_id"])
        return meta, raw["payload"]

    owned = [(a_t0,ta),(b_t0,tb),(c_t0,tc),(f_t0,tf),(f_t1,ts)]
    input_meta = [inspect(h,t) for h,t in owned]
    results = [(bid,bt),(fid,ft),(sid,st),(cid,ct),(gid,gt)]
    for h,t in results:
        inspect(h,t)
    for (h,t), (info,want) in zip(owned,input_meta,strict=True):
        m, data = s.run([(13,h),(98 if t.dtype == torch.float64 else 39,h,0,t.numel())])
        assert w.meta(m) == info and data["payload"] == want
    s.run([(41,h) for h,_ in owned] + [(41,bid),(41,fid)])
    for h,t in results[2:]:
        (data,) = s.run([(98 if t.dtype == torch.float64 else 39,h,0,t.numel())])
        assert data["payload"] == bytes(t.untyped_storage())
    s.run([(41,h) for h,_ in results[2:]])
    s.finish()
    receipt = json.loads((d / "transport.json").read_bytes())
    assert receipt["executable_sha256"] == ELF and receipt["exit_code"] == 0
    assert receipt["responses_before_stdin_eof"]
    return {"operations": len(results), "words": sum(t.numel() for _,t in results),
            "inputs_unchanged": True, "fresh_physical_storage": True, "survives_release": True,
            "frames": receipt["frames"],
            **{n + "_sha256": sha((d / (n + ".frames.bin")).read_bytes()) for n in ("requests","responses")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("repository", "candidate", "rounding-parent", "broadcast-parent", "output",
                 *(x.replace("_", "-") for x in PINS)):
        p.add_argument("--" + flag, type=lambda path: Path(path).resolve(), required=True)
    a = p.parse_args()
    assert not sys.flags.optimize and sha(a.candidate.read_bytes()) == ELF
    a.output.mkdir(parents=True, exist_ok=False)
    staged = stage(a)
    replay_results = {name: replay(a, staged, name, parent) for name,parent in zip(
        SUITES,(a.rounding_parent,a.broadcast_parent),strict=True)}
    cross = mixed(a, staged)
    obj = {"status":"SCOPED_PASS","canonical_credit":False,"candidate_sha256":ELF,
           "predecessor_suites":replay_results,"same_session":cross}
    (a.output / "qualification.json").write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    print(json.dumps(cross,sort_keys=True))


if __name__ == "__main__":
    main()
