#!/usr/bin/env python3
"""Replay all three exact independent tests against one combined ELF."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import SLICES, pinned, slice_inputs

EXPECTED_ELF = "b29dec5be0f1ae3ffedb9984af7579631fbc1b0eab8672db26af96043b979922"
SUITES = {
    "square": "ba91f4e1999325817f540a3d4e625ddb4c6049268f722bb66a345ef82f475fa2",
    "reciprocal": "eedf29bff531f322409860fc628a83571c96afcc2424de9b11ae6e1a5c9c6165",
    "sign": "a472959616fca7f96725ffb2046c33c8517644e93b4722dff7aa26c7f2cd3053",
}


def load_exact(path, digest, name):
    data = pinned(path, digest)
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), vars(module))
    return module


def mixed(a, suites):
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    wire = module("combined_wire", a.repository / "tests/integer_tensor_review/wire/protocol.py")
    sys.path.insert(0, str(a.repository / "tests"))
    from half_logical.native import Memory

    shapes = [(), (0,), (2, 0, 3), (3, 12), (1, 2, 1, 19)]
    words = (0, 0x80000000, 1, 0x807FFFFF, 0x00200000, 0x7F800000, 0xFF800000,
             0x7F812345, 0xFFC76543, 0x3F800000, 0xC0400000, 0x7F7FFFFF)
    import itertools
    import math
    cases = [(shape, tuple(itertools.islice(itertools.cycle(words), math.prod(shape)))) for shape in shapes]
    refs = {name: m.observe_torch(a.repository, a.output, cases)[0] for name, m in suites.items()}
    transport = a.output / "mixed-session"
    session = wire.Session(a.candidate, directory=transport)
    nwords = 0
    for i, (shape, data) in enumerate(cases):
        builder = suites["square"]
        (xrow,) = session.run([builder.tensor(wire, shape, data)])
        x = xrow["value"]
        (xm,) = session.run([(13, x)])
        before = wire.meta(xm)
        assert builder.physical(session, Memory, x, before) == builder.raw(data)
        keys = list(suites)
        keys = keys[i % len(keys):] + keys[:i % len(keys)]
        rows = session.run([(3, SLICES[name][0], x) for name in keys])
        ids = [row["value"] for row in rows]
        assert len(set(ids + [x])) == 1 + len(suites)
        raw_by_id, storage_ids, version_ids = {}, {before["storage_id"]}, {before["version_id"]}
        for name, token in zip(keys, ids, strict=True):
            stride, expected = refs[name][i]
            ym, yb = session.run([(13, token), (39, token, 0, len(data))])
            meta = wire.meta(ym)
            assert meta["storage_id"] not in storage_ids and meta["version_id"] not in version_ids
            storage_ids.add(meta["storage_id"])
            version_ids.add(meta["version_id"])
            assert meta["rank"] == len(shape) and tuple(meta["shape"][:len(shape)]) == shape
            assert tuple(meta["strides"][:len(shape)]) == stride
            assert meta["storage_offset"] == meta["requires_grad"] == meta["version"] == 0
            assert meta["storage_bytes"] == 4 * len(data)
            assert yb["value"] == len(data) and yb["payload"] == expected
            assert builder.physical(session, Memory, token, meta) == expected
            raw_by_id[token] = expected
            nwords += len(data)
        (xm,) = session.run([(13, x)])
        assert wire.meta(xm) == before and builder.physical(session, Memory, x, before) == builder.raw(data)
        session.run([(41, x)])
        for token in reversed(ids):
            (readback,) = session.run([(39, token, 0, len(data))])
            assert readback["payload"] == raw_by_id[token]
            session.run([(41, token)])
    session.finish()
    receipt = json.loads((transport / "transport.json").read_bytes())
    assert receipt["exit_code"] == 0 and receipt["responses_before_stdin_eof"]
    assert receipt["executable_sha256"] == EXPECTED_ELF
    return {"cases": len(cases), "operations": len(cases) * len(suites), "elements": nwords,
            "frames": receipt["frames"], "native_elf_sha256": receipt["executable_sha256"],
            **{kind + "_sha256": sha(transport / (kind + ".frames.bin")) for kind in ("requests", "responses")}}


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "parent", "candidate", "output"):
        ap.add_argument(f"--{name}", required=True, type=lambda p: Path(p).resolve())
    slice_inputs(ap)
    a = ap.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("qualification requires assertions")
    pinned(a.candidate, EXPECTED_ELF)
    suites = {name: load_exact(getattr(a, f"{name}_dir").resolve() / "qualify.py", digest, "combined_" + name)
              for name, digest in SUITES.items()}
    a.output.mkdir(parents=True, exist_ok=False)
    results = {}
    saved_argv = sys.argv
    try:
        for name, m in suites.items():
            target = a.output / name
            m.CANDIDATE = EXPECTED_ELF
            sys.argv = [m.__file__, "--repository", str(a.repository), "--parent", str(a.parent),
                        "--candidate", str(a.candidate), "--output", str(target)]
            m.main()
            report = target / "qualification.json"
            result = json.loads(report.read_bytes())
            assert result["candidate_sha256"] == EXPECTED_ELF and result["status"] == "SCOPED_PASS"
            results[name] = {k: v for k, v in result.items() if k != "rows"}
            results[name]["qualification_sha256"] = sha(report)
    finally:
        sys.argv = saved_argv
    result = {"status": "SCOPED_PASS", "canonical_credit": False, "candidate_sha256": EXPECTED_ELF,
              "exact_suites": results, "mixed": mixed(a, suites)}
    (a.output / "qualification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "exact_suites"}, sort_keys=True))


if __name__ == "__main__":
    main()
