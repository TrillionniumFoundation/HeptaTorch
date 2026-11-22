#!/usr/bin/env python3
"""Focused original-vs-owner800 wire witnesses; not the original35 acceptance runner."""
import argparse
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binding import ROOT, require, sha
from constructor_transition import OWNER800

MAIN_ELF = "ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e"
PROTOCOL_SHA = "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9"
OWNER_CLIENT_SHA = "dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b"
TORCH_LOCK_SHA = "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944"


def read_pinned(path, expected):
    raw = path.read_bytes()
    require(sha(raw) == expected, "fixed bytes changed: " + str(path))
    return raw


def run_case(w, binary, case, base_shape, axes):
    base = torch.arange(math.prod(base_shape), dtype=torch.float32).reshape(base_shape) / 4
    reference_input = base.transpose(*axes).requires_grad_()
    session = w.Session(binary)

    def one(command):
        return session.run([command])[0]

    def new(value, grad=0):
        return one(w.tensor(tuple(value.shape), value.flatten().tolist(), grad))["value"]

    try:
        root = new(base, 1)
        x = one((6, root, *axes))["value"]
        before = w.meta(one((13, x)))
        eps = 1e-5
        if case == "linear":
            wt = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
            bias = torch.tensor([0.5, 1.5])
            command = (15, x, new(wt, 1), new(bias, 1))
            reference = F.linear(reference_input, wt.requires_grad_(), bias.requires_grad_())
        elif case == "layernorm":
            width = reference_input.shape[-1]
            wt, bias = torch.ones(width), torch.zeros(width)
            command = (26, x, new(wt, 1), new(bias, 1), w.bits(eps))
            reference = F.layer_norm(reference_input, (width,), wt.requires_grad_(), bias.requires_grad_(), eps)
        else:
            channels = reference_input.shape[1]
            mean, variance = torch.arange(channels, dtype=torch.float32) / 4, torch.ones(channels)
            wt, bias = torch.ones(channels), torch.zeros(channels)
            command = (28, x, new(mean), new(variance), new(wt, 1), new(bias, 1), w.bits(eps))
            reference = F.batch_norm(reference_input, mean, variance, wt.requires_grad_(),
                                     bias.requires_grad_(), False, 0, eps)
        result = one(command)["value"]
        meta = w.meta(one((13, result)))
        count, rank = meta["numel"], meta["rank"]
        raw = one((39, result, 0, count))["payload"]
        actual = struct.unpack(f"<{count}f", raw)
        expected = reference.flatten().tolist()
        require(len(actual) == len(expected) and all(math.isclose(a, b, abs_tol=3e-5, rel_tol=3e-5)
                                                      for a, b in zip(actual, expected)), "live values: " + case)
        require(tuple(meta["shape"][:rank]) == tuple(reference.shape), "live shape: " + case)
        require(w.meta(one((13, x))) == before, "source metadata preserved: " + case)
        require(meta["storage_id"] != before["storage_id"] and meta["version_id"] != before["version_id"],
                "fresh output Storage and Version identities: " + case)
        return dict(elements=count, native_version=meta["version"], torch_version=reference._version,
                    native_strides=tuple(meta["strides"][:rank]), torch_strides=reference.stride())
    finally:
        session.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    require(not sys.flags.optimize, "wire helper assertions must be enabled")
    for path, expected in ((args.baseline, MAIN_ELF), (args.candidate, OWNER800[1])):
        read_pinned(path, expected)
    lock = json.loads(read_pinned(ROOT / "compatibility/pytorch-oracle-lock.json", TORCH_LOCK_SHA))["oracle"]
    wheel = lock["wheel"]
    distribution = importlib.metadata.distribution("torch")
    metadata = json.loads(distribution.read_text("direct_url.json"))
    require(str(torch.__version__) == distribution.version == lock["version"]
            and torch.version.git_version == lock["git_commit"]
            and metadata["url"] == wheel["url"]
            and metadata["archive_info"]["hashes"]["sha256"] == wheel["sha256"],
            "use the canonical, pinned CPU PyTorch wheel")
    protocol = ROOT / "tests/adam_distributions_gradient_review/protocol.py"
    read_pinned(protocol, PROTOCOL_SHA)
    read_pinned(protocol.with_name("owner_client.py"), OWNER_CLIENT_SHA)
    spec = importlib.util.spec_from_file_location("constructor_witness_wire", protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)

    cases = dict(linear=((2, 4, 3), (1, 2)), layernorm=((2, 4, 3), (1, 2)), batchnorm_eval=((2, 2, 3, 4), (2, 3)))
    known_old_strides = {"layernorm": (12, 1, 3), "batchnorm_eval": (24, 12, 1, 4)}
    rows = {}
    for name, (shape, axes) in cases.items():
        old = run_case(wire, args.baseline.resolve(), name, shape, axes)
        new = run_case(wire, args.candidate.resolve(), name, shape, axes)
        require(new["native_version"] == new["torch_version"]
                and new["native_strides"] == new["torch_strides"], "owner800 Torch metadata: " + name)
        if name == "linear":
            require((old["native_version"], new["native_version"]) == (0, 1), "linear version delta")
        else:
            require(old["native_strides"] == known_old_strides[name] != new["native_strides"],
                    "normalization stride delta: " + name)
        rows[name] = dict(baseline=old, candidate=new)
    print(json.dumps(dict(status="PASS_FOCUSED_CONSTRUCTOR_WITNESSES_ONLY",
                          baseline_sha256=MAIN_ELF, candidate_sha256=OWNER800[1],
                          torch_wheel_sha256=wheel["sha256"],
                          original35_qualification_inferred=False,
                          normalization_allocation_qualified=False, rows=rows), indent=2))


if __name__ == "__main__":
    main()
