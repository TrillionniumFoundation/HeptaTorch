#!/usr/bin/env python3
"""Compare native I64/Bool whole-tensor min/max and retain #100 prod."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

CANONICAL = "ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e"
PARENT = "571d2317578785b74334087a6a2371287525f64ccf27220a65b3212073c6c325"
CANDIDATE = "b7ed53dfefda67ac0ecbe6f0799f0c013f426aca55cb645162d14625c11a756a"
PROD_VERIFIER = "63cccdb9af6dcd5d848850e0ba25c1160c3440d0c004bb0283edbd86c8e396b5"
PROD_RECEIPT = "57578cc21206d7306852176e1cf51ab918bf802fec9bd286ec0d8d4e4b9f92ad"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original(path):
    assert sha(path) == PROD_VERIFIER
    spec = importlib.util.spec_from_file_location("prior_prod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_result(w, decoder, prod, s, y, dtype, wanted, independent_of):
    meta = prod.meta(w, s, y)
    assert meta["storage_id"] not in {m["storage_id"] for m in independent_of}
    assert meta["version_id"] not in {m["version_id"] for m in independent_of}
    width = {1: 1, 2: 8}[dtype]
    assert (meta["rank"], meta["numel"], meta["storage_bytes"]) == (0, 1, width)
    assert meta["requires_grad"] == meta["storage_offset"] == meta["version"] == 0
    raw, node = prod.physical(s, decoder, y, meta)
    assert node[72] == dtype and node[51] == 0 and node[5] == 0
    (wire,) = s.run([(98, y, 0, 1)])
    assert wire["payload"] == raw == wanted, (dtype, wire["payload"].hex(), raw.hex(), wanted.hex())
    return meta


def exercise(a, prod, w, decoder):
    inputs = list(prod.make_cases())
    _, provenance = prod.observe(a.repository, a.output, a.install_report, inputs)
    import torch
    s = w.Session(a.candidate, directory=a.output / "candidate")
    p = w.Session(a.parent, directory=a.output / "parent")
    cases = []
    empty = 0
    for dtype, label, root_shape, values, view in inputs:
        ty = {1: torch.bool, 2: torch.int64}[dtype]
        root = torch.tensor(values, dtype=ty).reshape(root_shape).clone()
        x = torch.as_strided(root, *view) if view else root
        payload = prod.words(dtype, values)
        base = prod.create(w, s, dtype, root_shape, payload)
        token = prod.view_token(w, s, base, view)
        base_p = prod.create(w, p, dtype, root_shape, payload)
        token_p = prod.view_token(w, p, base_p, view)
        old = prod.meta(w, s, token)
        assert old["numel"] == x.numel()
        assert prod.physical(s, decoder, token, old)[0] == payload
        for op in (23, 24):
            p.error([(3, op, token_p)], 7)
        outputs = []
        if not x.numel():
            for cmd, torch_op in [(23, torch.ops.aten.min.default), (24, torch.ops.aten.max.default)]:
                try:
                    torch_op(x)
                except RuntimeError as error:
                    assert "numel() == 0" in str(error), str(error)
                else:
                    raise AssertionError("oracle must reject an empty unqualified reduction")
                s.error([(3, cmd, token)], 7)
            empty += 1
        else:
            native = s.run([(3, 23, token), (3, 24, token)])
            y_min, y_max = [row["value"] for row in native]
            assert len({token, y_min, y_max}) == 3
            ref_rows = [torch.ops.aten.min.default(x), torch.ops.aten.max.default(x)]
            previous = [old]
            for y, ref in zip((y_min, y_max), ref_rows, strict=True):
                assert ref.dtype == x.dtype and not ref.requires_grad and ref.shape == torch.Size([])
                assert ref.untyped_storage()._cdata != root.untyped_storage()._cdata
                expected = bytes(ref.untyped_storage())
                previous.append(check_result(w, decoder, prod, s, y, dtype, expected, previous))
                outputs.append((y, expected, dtype, previous[-1]))
            # A live result from the new max route is used as the input to the
            # original product route before either predecessor token is released.
            check = s.run([(3, 22, y_max)])[0]["value"]
            want_check = bytes(torch.ops.aten.prod.default(ref_rows[1]).untyped_storage())
            previous.append(check_result(w, decoder, prod, s, check, 2, want_check, previous))
            outputs.append((check, want_check, 2, previous[-1]))
            cases.append({"dtype": dtype, "label": label, "shape": tuple(x.shape),
                          "input_count": x.numel(), "min_hex": outputs[0][1].hex(),
                          "max_hex": outputs[1][1].hex(), "max_then_product_hex": want_check.hex()})
        assert prod.meta(w, s, token) == old and prod.physical(s, decoder, token, old)[0] == payload
        assert bytes(root.untyped_storage()) == payload
        if view:
            s.run([(41, token)])
            p.run([(41, token_p)])
        s.run([(41, base)])
        p.run([(41, base_p)])
        for output, want, _, m in outputs:
            assert s.run([(98, output, 0, 1)])[0]["payload"] == want
            assert prod.physical(s, decoder, output, m)[0] == want
            s.run([(41, output)])
    for session in (s, p):
        f = session.run([w.tensor((2,), (1.0, -2.0))])[0]["value"]
        stale = prod.create(w, session, 2, (), prod.words(2, (3,)))
        session.run([(41, stale)])
        for op in (23, 24):
            session.error([(3, op, f)], 4)
            session.error([(3, op, stale)], 1)
            session.error([(3, op, 1)], 1)
        session.run([(41, f)])
        session.finish()
    logs = a.output / "candidate"
    transport = json.loads((logs / "transport.json").read_bytes())
    assert transport["exit_code"] == 0 and transport["responses_before_stdin_eof"] is True
    return {"status": "SCOPED_PASS", "canonical_credit": False, "cases": len(cases),
            "exact_extremum_words": 2 * len(cases), "exact_mixed_max_then_prod_words": len(cases),
            "input_words": sum(row["input_count"] for row in cases),
            "parent_refusals": 2 * (len(cases) + empty), "empty_refusals_in_candidate": 2 * empty,
            "other_error_controls_per_image": 6, "oracle_checks": provenance,
            "input_unchanged": True, "output_fresh_storage_version_survives_release": True,
            "requests_sha256": sha(logs / "requests.frames.bin"),
            "responses_sha256": sha(logs / "responses.frames.bin"),
            "frames": transport["frames"], "rows": cases}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "canonical", "parent", "candidate", "install-report", "output"):
        ap.add_argument("--" + name, required=True, type=lambda p: Path(p).resolve())
    ap.add_argument("--prod-verifier", type=Path, default=Path(__file__).resolve().parents[1] / "opt_in_integral_prod/qualify.py")
    a = ap.parse_args()
    assert not sys.flags.optimize
    assert sha(a.canonical) == CANONICAL and sha(a.parent) == PARENT and sha(a.candidate) == CANDIDATE
    prior = original(a.prod_verifier.resolve())
    a.output.mkdir(parents=True, exist_ok=False)
    w = prior.load(a.repository / "tests/integer_tensor_review/wire/protocol.py", "extrema_wire", prior.WIRE)
    sys.path.insert(0, str(a.repository / "tests"))
    prior.bind(a.repository / "tests/half_logical/native.py", prior.OWNER)
    from half_logical.native import Memory
    result = exercise(a, prior, w, Memory)
    keep = sys.argv
    prior.CANDIDATE = CANDIDATE
    replay = a.output / "old-prod"
    try:
        sys.argv = [str(a.prod_verifier), "--repository", str(a.repository), "--parent", str(a.canonical),
                    "--candidate", str(a.candidate), "--output", str(replay),
                    "--install-report", str(a.install_report)]
        prior.main()
    finally:
        sys.argv = keep
    receipt = json.loads((replay / "qualification.json").read_bytes())
    # Candidate identity is the one allowed to change; every prior frame and
    # inference/physical check must reproduce on the extended candidate.
    receipt["candidate_sha256"] = PARENT
    normalized = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(normalized).hexdigest() == PROD_RECEIPT
    result.update(parent_sha256=PARENT, candidate_sha256=CANDIDATE, prior_prod_cases=receipt["cases"],
                  original_prod_report_sha256=PROD_RECEIPT, extended_prod_report_sha256=sha(replay / "qualification.json"))
    (a.output / "qualification.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k,v in result.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
