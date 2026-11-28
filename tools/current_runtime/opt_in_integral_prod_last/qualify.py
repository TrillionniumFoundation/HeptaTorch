#!/usr/bin/env python3
"""Compare native last-axis integral product and replay the prior stack."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

CANONICAL = "ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e"
PROD_ELF = "571d2317578785b74334087a6a2371287525f64ccf27220a65b3212073c6c325"
PARENT = "b7ed53dfefda67ac0ecbe6f0799f0c013f426aca55cb645162d14625c11a756a"
CANDIDATE = "e1a990c1f95b65fd1121380e4d9c1d9c274cd3287f80ceded1881a980605c932"
EXTREMA_VERIFIER = "9e49136bd3f985e4b3ccf6bfad26d6979a005cc22dd4d4b96cadf0e1b339a8bf"
EXTREMA_RECEIPT = "9b1fdc19a9abf29827718d7acf7c16fac349ff2c7c1702e7e669ea10ced4c675"
EXTREMA_ORIGINAL_PROD_REPLAY = "1fc3317736f552cdaff32b7dec6e8d5d7ed01edfd3b7632e20415afb6f4f4c3d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_import(path):
    assert sha(path) == EXTREMA_VERIFIER
    spec = importlib.util.spec_from_file_location("prior_extrema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(w, decoder, prod, session, token, expected, shape, stride, count, independent):
    m = prod.meta(w, session, token)
    assert m["storage_id"] not in {x["storage_id"] for x in independent}
    assert m["version_id"] not in {x["version_id"] for x in independent}
    assert (m["rank"], m["numel"], m["storage_bytes"]) == (len(shape), count, len(expected))
    assert tuple(m["shape"][:len(shape)]) == shape and tuple(m["strides"][:len(shape)]) == stride
    assert m["requires_grad"] == m["storage_offset"] == m["version"] == 0
    raw, node = prod.physical(session, decoder, token, m)
    assert node[72] == 2 and node[51] == 0 and node[5] == 0
    sent = session.run([(98, token, 0, count)])[0]
    assert sent["value"] == count and sent["payload"] == raw == expected
    return m


def exercise(a, prod, w, decoder):
    inputs = list(prod.make_cases())
    geometry = [(2, 1), (17, 257), (2, 3, 4, 0), (0, 512), (3, 0, 0, 1), (1, 2, 3, 1)]
    for dtype in (1, 2):
        for shape in geometry:
            n = math.prod(shape)
            vals = tuple((i % 33) != 2 for i in range(n)) if dtype == 1 else tuple(3 - 2 * (i % 7) for i in range(n))
            inputs.append((dtype, "axis-geometry", shape, vals, None))
    _, provenance = prod.observe(a.repository, a.output, a.install_report, inputs)
    import torch
    s = w.Session(a.candidate, directory=a.output / "candidate")
    p = w.Session(a.parent, directory=a.output / "parent")
    rows = []
    for dtype, label, root_shape, values, view in inputs:
        ty = {1: torch.bool, 2: torch.int64}[dtype]
        root = torch.tensor(values, dtype=ty).reshape(root_shape).clone()
        x = torch.as_strided(root, *view) if view else root
        y = torch.ops.aten.prod.dim_int(x, -1, False)
        expected, shape, stride = bytes(y.untyped_storage()), tuple(y.shape), tuple(y.stride())
        assert y.dtype == torch.int64 and not y.requires_grad and y.storage_offset() == y._version == root._version == 0
        assert y.untyped_storage()._cdata != root.untyped_storage()._cdata
        payload = prod.words(dtype, values)
        base = prod.create(w, s, dtype, root_shape, payload)
        inp = prod.view_token(w, s, base, view)
        parent_base = prod.create(w, p, dtype, root_shape, payload)
        parent_inp = prod.view_token(w, p, parent_base, view)
        p.error([(3, 25, parent_inp)], 7)
        old = prod.meta(w, s, inp)
        assert old["numel"] == x.numel() and prod.physical(s, decoder, inp, old)[0] == payload
        output = s.run([(3, 25, inp)])[0]["value"]
        assert output != inp and output != base
        m = compare(w, decoder, prod, s, output, expected, shape, stride, y.numel(), [old])
        total = s.run([(3, 22, output)])[0]["value"]
        expected_total = bytes(torch.ops.aten.prod.default(y).untyped_storage())
        assert expected_total == bytes(torch.ops.aten.prod.default(x).untyped_storage())
        m_total = compare(w, decoder, prod, s, total, expected_total, (), (), 1, [old, m])
        assert prod.meta(w, s, inp) == old and prod.physical(s, decoder, inp, old)[0] == payload
        assert bytes(root.untyped_storage()) == payload
        if view:
            s.run([(41, inp)])
            p.run([(41, parent_inp)])
        s.run([(41, base)])
        p.run([(41, parent_base)])
        assert s.run([(98, output, 0, y.numel())])[0]["payload"] == expected
        assert prod.physical(s, decoder, output, m)[0] == expected
        s.run([(41, output)])
        assert s.run([(98, total, 0, 1)])[0]["payload"] == expected_total
        assert prod.physical(s, decoder, total, m_total)[0] == expected_total
        s.run([(41, total)])
        rows.append({"dtype": dtype, "label": label, "shape": tuple(x.shape), "stride": tuple(x.stride()),
                     "output_shape": shape, "output_stride": stride, "input_count": x.numel(),
                     "output_count": y.numel(), "output_sha256": hashlib.sha256(expected).hexdigest(),
                     "total_hex": expected_total.hex()})
    for session in (s, p):
        f = session.run([w.tensor((2,), (1.0, -2.0))])[0]["value"]
        stale = prod.create(w, session, 2, (), prod.words(2, (3,)))
        session.run([(41, stale)])
        session.error([(3, 25, f)], 4)
        session.error([(3, 25, stale)], 1)
        session.error([(3, 25, 1)], 1)
        session.run([(41, f)])
        session.finish()
    logs = a.output / "candidate"
    transport = json.loads((logs / "transport.json").read_bytes())
    assert transport["exit_code"] == 0 and transport["responses_before_stdin_eof"] is True
    return {
        "status": "SCOPED_PASS", "canonical_credit": False, "cases": len(rows),
        "view_cases": sum(case[-1] is not None for case in inputs),
        "input_words": sum(row["input_count"] for row in rows),
        "output_words": sum(row["output_count"] for row in rows),
        "mixed_last_then_global_product_words": len(rows),
        "empty_output_cases": sum(row["output_count"] == 0 for row in rows),
        "empty_input_nonempty_output_cases": sum(row["input_count"] == 0 and row["output_count"] != 0 for row in rows),
        "parent_refusals": len(rows), "control_refusals_per_image": 3,
        "fresh_independent_native_storage_versions": True, "outputs_survive_release": True, "input_unchanged": True,
        "oracle_checks": provenance, "candidate_sha256": CANDIDATE, "parent_sha256": PARENT,
        "requests_sha256": sha(logs / "requests.frames.bin"), "responses_sha256": sha(logs / "responses.frames.bin"),
        "frames": transport["frames"], "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "canonical", "prod-elf", "parent", "candidate", "install-report", "output"):
        parser.add_argument("--" + name, required=True, type=lambda p: Path(p).resolve())
    tools = Path(__file__).resolve().parents[1]
    parser.add_argument("--prod-verifier", type=Path, default=tools / "opt_in_integral_prod/qualify.py")
    parser.add_argument("--extrema-verifier", type=Path, default=tools / "opt_in_integral_extrema/qualify.py")
    a = parser.parse_args()
    assert not sys.flags.optimize
    for file, pin in [(a.canonical,CANONICAL),(a.prod_elf,PROD_ELF),(a.parent,PARENT),(a.candidate,CANDIDATE)]:
        assert sha(file) == pin
    extrema = verified_import(a.extrema_verifier.resolve())
    prod = extrema.original(a.prod_verifier.resolve())
    a.output.mkdir(parents=True, exist_ok=False)
    w = prod.load(a.repository / "tests/integer_tensor_review/wire/protocol.py", "last_prod_wire", prod.WIRE)
    sys.path.insert(0, str(a.repository / "tests"))
    prod.bind(a.repository / "tests/half_logical/native.py", prod.OWNER)
    from half_logical.native import Memory
    result = exercise(a, prod, w, Memory)
    old_argv = sys.argv
    replay = a.output / "old-extrema"
    extrema.CANDIDATE = CANDIDATE
    try:
        sys.argv = [str(a.extrema_verifier), "--repository", str(a.repository), "--canonical", str(a.canonical),
                    "--parent", str(a.prod_elf), "--candidate", str(a.candidate), "--output", str(replay),
                    "--prod-verifier", str(a.prod_verifier.resolve()), "--install-report", str(a.install_report)]
        extrema.main()
    finally:
        sys.argv = old_argv
    receipt = json.loads((replay / "qualification.json").read_bytes())
    receipt["candidate_sha256"] = PARENT
    receipt["extended_prod_report_sha256"] = EXTREMA_ORIGINAL_PROD_REPLAY
    normalized = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(normalized).hexdigest() == EXTREMA_RECEIPT
    result.update(original_extrema_report_sha256=EXTREMA_RECEIPT,
                  actual_replayed_extrema_report_sha256=sha(replay / "qualification.json"),
                  prior_product_cases=receipt["prior_prod_cases"], prior_extrema_cases=receipt["cases"])
    (a.output / "qualification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
