#!/usr/bin/env python3
"""Compare real PyTorch and opt-in HIA whole-tensor integral products."""
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
CANDIDATE = "571d2317578785b74334087a6a2371287525f64ccf27220a65b3212073c6c325"
ORACLE = "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944"
ORACLE_SCRIPT = "ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae"
WIRE = "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9"
OWNER = "9c2cd19a5bb04eaa037f5905a52bce2dcc31042101f72d07e71205745160a14f"
MASK = (1 << 64) - 1
LOW, HIGH = -(1 << 63), (1 << 63) - 1


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bind(path, digest_expected):
    assert digest(path.read_bytes()) == digest_expected, path.name


def load(path, name, pin):
    bind(path, pin)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def words(dtype, values):
    fmt = {0: "I", 1: "B", 2: "q"}[dtype]
    return struct.pack("<" + fmt * len(values), *values)


def make_cases():
    rng = random.Random(220775)
    integer = [
        ("negative-scalar-min", (), (LOW,)), ("negative-scalar", (), (-5,)),
        ("empty-rank1", (0,), ()), ("empty-rank2", (2, 0), ()),
        ("empty-rank3", (2, 0, 3), ()), ("empty-rank4", (2, 0, 3, 4), ()),
        ("signed-wrap-high", (2,), (HIGH, 3)), ("signed-wrap-low", (2,), (LOW, -1)),
        ("multiply-wrap-zero", (2,), (1 << 32, 1 << 32)),
        ("positive-wrap", (2,), (HIGH, HIGH)), ("mix-zero", (3,), (LOW, 0, HIGH)),
        ("linear-even-odd", (2, 2, 2), (2, 3, 5, -7, 11, -13, 17, 19)),
    ]
    for shape in [(1,), (3,), (7,), (8,), (9,), (15,), (16,), (17,), (31,), (32,),
                  (33,), (65,), (257,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
        n = math.prod(shape)
        values = tuple((rng.getrandbits(64) | 1) + LOW for _ in range(n))
        integer.append(("odd-random", shape, values))
    for n in (4, 16, 64, 256, 1024, 4096):
        integer.append(("long-odd-product", (n,), tuple(itertools.islice(itertools.cycle((3, -5, 7, -11)), n))))
    for label, shape, values in integer:
        yield 2, label, shape, values, None
    bool_shapes = [(), (0,), (2, 0, 3), (0, 2, 2, 3), (1,), (2,), (7,), (16,), (17,),
                   (31,), (32,), (33,), (65,), (257,), (4096,), (2, 3), (2, 3, 5), (2, 3, 4, 5)]
    for shape in bool_shapes:
        count = math.prod(shape)
        yield 1, "all-true", shape, (1,) * count, None
        if count:
            vals = [1] * count
            for idx in {0, count // 2, count - 1}:
                vals[idx] = 0
                yield 1, "exactly-one-false", shape, tuple(vals), None
                vals[idx] = 1
    views = [
        ("transpose", (3, 4), (4, 3), (1, 4), 0),
        ("positive-offset", (3, 4), (3, 2), (4, 1), 1),
        ("diagonal-stride", (5, 5), (5,), (6,), 0),
        ("zero-stride", (2, 1), (2, 3), (1, 0), 0),
        ("overlapping", (8,), (3, 3), (1, 1), 2),
        ("offset-scalar", (5,), (), (), 4),
        ("empty-strided", (10,), (0, 3), (20, 2), 3),
    ]
    for dtype in (1, 2):
        for label, base, shape, stride, offset in views:
            count = math.prod(base)
            values = tuple(range(1, 2 * count, 2)) if dtype == 2 else (1,) * count
            yield dtype, label, base, values, (shape, stride, offset)
            if dtype == 1:
                for i in sorted({offset, count - 1}):
                    masked = list(values)
                    masked[i] = 0
                    yield dtype, label + "-false-location", base, tuple(masked), (shape, stride, offset)


def observe(repo, out, install_report, inputs):
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    lock = repo / "compatibility/pytorch-oracle-lock.json"
    script = repo / "tests/pytorch_oracle.py"
    bind(lock, ORACLE)
    bind(script, ORACLE_SCRIPT)
    upstream = repo / "compatibility/upstream-lock.json"
    bind(upstream, "d204c5a8268a0d0f0a1dd162b5e0ab184beee83310bcad643b084deb691f3f49")
    oracle_home = out / "oracle-inputs"
    for dest, source in (("compatibility/upstream-lock.json", upstream), ("build/pytorch-install.json", install_report)):
        path = oracle_home / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
    home, save = Path.cwd(), sys.argv
    try:
        os.chdir(oracle_home)
        sys.argv = [str(script), str(out / "NO_NATIVE"), str(out / "oracle-error.json"), str(lock)]
        oracle = runpy.run_path(str(script), run_name="heptatorch_integral_prod_reference")
    finally:
        sys.argv = save
        os.chdir(home)
    prov = oracle["PROVENANCE"]
    assert prov["status"] == "PASS" and all(prov["checks"].values())
    torch = oracle["torch"]
    assert torch.backends.cpu.get_cpu_capability() == "AVX2" and torch.get_num_threads() == 1
    reports = []
    for dtype, _, shape, values, view in inputs:
        ty = {1: torch.bool, 2: torch.int64}[dtype]
        root = torch.tensor(values, dtype=ty).reshape(shape).clone()
        x = torch.as_strided(root, *view) if view else root
        observed = torch.ops.aten.prod.default(x)
        assert observed.dtype == torch.int64 and not observed.requires_grad and tuple(observed.shape) == ()
        assert observed.storage_offset() == observed._version == root._version == 0
        assert observed.untyped_storage()._cdata != root.untyped_storage()._cdata
        assert bytes(root.untyped_storage()) == words(dtype, values)
        reports.append((bytes(observed.untyped_storage()), tuple(x.shape), tuple(x.stride()), x.storage_offset()))
    return reports, prov["checks"]


def create(w, s, dtype, shape, raw, requires=0):
    cfg = struct.pack("<8Q", len(shape), requires, *shape, *([0] * (4 - len(shape))), 160, len(raw))
    req = b"HTORCH03" + struct.pack("<3Q", 1, 160 + len(raw), 0) + struct.pack("<8Q", 97, dtype, 96, 0, 0, 0, 0, 0)
    status, records = w.parse(s.exchange(req + cfg + raw))
    assert status == 0 and len(records) == 1, (dtype, shape, status)
    return records[0]["value"]


def view_token(w, s, root, view):
    if view is None:
        return root
    shape, stride, offset = view
    n = len(shape)
    cfg = struct.pack("<10Q", n, *shape, *([0] * (4 - n)), *stride, *([0] * (4 - n)), offset)
    return s.run([(69, root, w.Payload(cfg), 80)])[0]["value"]


def physical(s, decoder, token, meta):
    memory = decoder(s.owner()["bytes"])
    node = memory.node(token)
    storage, version = memory.row(2, node[27]), memory.row(3, node[28])
    assert (node[27], node[28], node[29], storage[3], version[2]) == (
        meta["storage_id"], meta["version_id"], meta["storage_offset"], meta["storage_bytes"], meta["version"])
    assert storage[5] > 0 and storage[7] == 0
    start = memory.start + storage[2]
    end = start + storage[3]
    assert memory.start <= start <= end <= len(memory.raw)
    return memory.raw[start:end], node


def meta(w, s, token):
    return w.meta(s.run([(13, token)])[0])


def check_case(w, decoder, s, parent, case, expected):
    dtype, label, shape, values, view = case
    result, wanted_shape, wanted_stride, wanted_offset = expected
    payload = words(dtype, values)
    x = create(w, s, dtype, shape, payload)
    x1 = view_token(w, s, x, view)
    xp = create(w, parent, dtype, shape, payload)
    x1p = view_token(w, parent, xp, view)
    parent.error([(3, 22, x1p)], 7)
    m = meta(w, s, x1)
    rank = len(wanted_shape)
    assert m["rank"] == rank and tuple(m["shape"][:rank]) == wanted_shape, case
    assert tuple(m["strides"][:rank]) == wanted_stride, (label, wanted_shape, m, wanted_stride)
    assert m["storage_offset"] == wanted_offset * {1: 1, 2: 8}[dtype]
    assert m["numel"] == math.prod(wanted_shape)
    assert physical(s, decoder, x1, m)[0] == payload
    before = meta(w, s, x)
    if view:
        assert (m["storage_id"], m["version_id"]) == (before["storage_id"], before["version_id"])
    y = s.run([(3, 22, x1)])[0]["value"]
    my, info, val = s.run([(13, y), (96, y, 0), (98, y, 0, 1)])
    my = w.meta(my)
    assert y not in (x, x1) and my["storage_id"] != m["storage_id"] and my["version_id"] != m["version_id"]
    assert struct.unpack("<4Q", info["payload"])[:2] == (2, 8)
    assert (my["rank"], my["numel"], my["storage_bytes"]) == (0, 1, 8)
    assert my["requires_grad"] == my["storage_offset"] == my["version"] == 0
    actual, node = physical(s, decoder, y, my)
    assert node[72] == 2 and node[51] == 0 and node[5] == 0
    assert actual == val["payload"] == result, (label, shape, val["payload"].hex(), result.hex())
    assert meta(w, s, x) == before and meta(w, s, x1) == m
    assert physical(s, decoder, x1, m)[0] == payload
    if view:
        s.run([(41, x1)])
        parent.run([(41, x1p)])
    s.run([(41, x)])
    parent.run([(41, xp)])
    assert s.run([(98, y, 0, 1)])[0]["payload"] == result
    assert physical(s, decoder, y, my)[0] == result
    s.run([(41, y)])
    return {"dtype": dtype, "label": label, "shape": wanted_shape, "strides": wanted_stride,
            "offset_elements": wanted_offset, "input_words": math.prod(wanted_shape), "result_hex": result.hex()}


def regression(w, executable, out):
    s = w.Session(executable, directory=out)
    f = s.run([w.tensor((2, 2), (1.0, -2.0, 3.0, -4.0))])[0]["value"]
    i = create(w, s, 2, (2, 2), words(2, (LOW, -1, 2, HIGH)))
    b = create(w, s, 1, (2, 2), b"\x01\x01\x00\x01")
    for op, token, output_dtype in ((5, f, 0), (5, i, 2), (15, i, 2)):
        status, records = w.parse(s.exchange(w.packet([(3, op, token)])))
        if not status:
            y = records[0]["value"]
            s.run([(13, y), (98 if output_dtype == 2 else 39, y, 0, 4), (41, y)])
    for token, dtype in ((i, 2), (b, 2), (f, 0)):
        y = s.run([(115, token, MASK, 0, 0, dtype, 0, 0)])[0]["value"]
        s.run([(13, y), (98 if dtype == 2 else 39, y, 0, 1), (41, y)])
    s.run([(41, f), (41, i), (41, b)])
    s.finish()
    return (out / "responses.frames.bin").read_bytes()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "parent", "candidate", "output", "install-report"):
        p.add_argument("--" + name, required=True, type=lambda value: Path(value).resolve())
    a = p.parse_args()
    assert not sys.flags.optimize
    bind(a.parent, PARENT)
    bind(a.candidate, CANDIDATE)
    a.output.mkdir(parents=True, exist_ok=False)
    w = load(a.repository / "tests/integer_tensor_review/wire/protocol.py", "prod_wire", WIRE)
    sys.path.insert(0, str(a.repository / "tests"))
    bind(a.repository / "tests/half_logical/native.py", OWNER)
    from half_logical.native import Memory
    data = list(make_cases())
    expected, provenance = observe(a.repository, a.output, a.install_report, data)
    cdir, pdir = a.output / "candidate", a.output / "parent"
    candidate, parent = w.Session(a.candidate, directory=cdir), w.Session(a.parent, directory=pdir)
    rows = [check_case(w, Memory, candidate, parent, case, ref) for case, ref in zip(data, expected, strict=True)]
    for s in (candidate, parent):
        f = s.run([w.tensor((2,), (1.0, 2.0)), w.tensor((2,), (1.0, 2.0), requires=1)])
        for r in f:
            s.error([(3, 22, r["value"])], 4)
            s.run([(41, r["value"])])
        z = create(w, s, 2, (), words(2, (3,)))
        s.run([(41, z)])
        s.error([(3, 22, z)], 1)
        s.error([(3, 22, 1)], 1)
        s.finish()
    old = regression(w, a.parent, a.output / "regression-parent")
    new = regression(w, a.candidate, a.output / "regression-candidate")
    assert old == new
    receipt = json.loads((cdir / "transport.json").read_text())
    assert receipt["responses_before_stdin_eof"] is True and receipt["exit_code"] == 0
    summary = {
        "status": "SCOPED_PASS", "canonical_credit": False, "parent_sha256": PARENT, "candidate_sha256": CANDIDATE,
        "oracle_lock_sha256": ORACLE, "oracle_checks": provenance, "cases": len(rows),
        "int64_cases": sum(r["dtype"] == 2 for r in rows), "bool_cases": sum(r["dtype"] == 1 for r in rows),
        "view_cases": sum(c[-1] is not None for c in data), "input_words": sum(r["input_words"] for r in rows),
        "parent_new_op_refusals": len(rows), "unchanged_error_controls_per_image": 4,
        "regression_response_sha256": digest(old), "candidate_frames": receipt["frames"], "rows": rows,
        "candidate_requests_sha256": digest((cdir / "requests.frames.bin").read_bytes()),
        "candidate_responses_sha256": digest((cdir / "responses.frames.bin").read_bytes()),
        "output_distinct_native_storage_and_version": True,
        "output_storage_exact_before_and_after_release": True,
        "input_storage_and_metadata_unchanged": True,
    }
    (a.output / "qualification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
