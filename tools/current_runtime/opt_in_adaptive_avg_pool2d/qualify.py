"""Compare the opt-in adaptive pool, including actual backward, to eager Torch."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from zipfile import ZipFile

# Treat AVX2 and thread counts as part of this measured comparison, not of the API.
os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import adaptive_pool as pool


WHEEL_SHA = "0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3"

GEOMETRIES = (
    ((1, 1, 1, 1), (1, 1)),
    ((1, 1, 4, 4), (2, 2)),
    ((2, 3, 6, 8), (3, 2)),
    ((1, 2, 7, 9), (1, 1)),
    ((2, 2, 6, 3), (2, 3)),
    ((1, 2, 5, 5), (5, 5)),
    ((1, 1, 2, 12), (2, 2)),
    ((1, 1, 12, 2), (2, 2)),
    ((1, 1, 64, 64), (4, 4)),
)


def make_values(shape, shift):
    count = math.prod(shape)
    return struct.pack(f"<{count}f", *((((i * 13 + shift) % 31) - 15) / 8 for i in range(count)))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def values(tensor):
    return tensor.detach().reshape(-1).numpy().tobytes()


def verify_torch_custody(wheel, install, torch):
    with wheel.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != WHEEL_SHA:
            raise ValueError("reference wheel SHA-256 mismatch")
    raw = install.read_bytes()
    matching = [item for item in json.loads(raw)["install"] if item.get("metadata", {}).get("name") == "torch"]
    if len(matching) != 1 or matching[0]["download_info"]["archive_info"]["hashes"]["sha256"] != WHEEL_SHA:
        raise ValueError("Torch install report not bound to the pinned wheel")
    files = ["torch/__init__.py", "torch/nn/functional.py",
             f"torch/{Path(torch._C.__file__).name}", "torch/lib/libtorch_cpu.so"]
    installed = Path(torch.__file__).resolve().parents[1]
    hashes = {}
    with ZipFile(wheel) as z:
        for name in files:
            with (installed / name).open("rb") as actual, z.open(name) as expected:
                digest = hashlib.file_digest(actual, "sha256").hexdigest()
                if digest != hashlib.file_digest(expected, "sha256").hexdigest():
                    raise ValueError(f"executing Torch component differs from pinned wheel: {name}")
                hashes[name] = digest
    return dict(wheel_sha256=WHEEL_SHA, install_report_sha256=sha(raw), installed_component_sha256=hashes)


def run(binary, parent, wheel, install):
    import torch
    from torch.nn.functional import adaptive_avg_pool2d
    if torch.__version__ != "2.15.0+cpu":
        raise RuntimeError("requires the upstream-locked torch 2.15.0+cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise RuntimeError("expected locked AVX2 eager path")
    custody = verify_torch_custody(wheel, install, torch)
    cases = []
    for index, (shape, out) in enumerate(GEOMETRIES):
        wanted = (*shape[:2], *out)
        source = make_values(shape, index + 3)
        for backward in (False, True):
            x = torch.frombuffer(bytearray(source), dtype=torch.float32).reshape(shape)
            x.requires_grad_(backward)
            y = adaptive_avg_pool2d(x, out)
            if tuple(y.shape) != wanted or y.stride() != tuple(pool.contiguous_strides(wanted)):
                raise RuntimeError("unexpected Torch layout")
            if y.data_ptr() == x.data_ptr():
                raise RuntimeError("unexpected Torch alias")
            seed = make_values(wanted, index + 11) if backward else None
            eager_gradient = None
            if seed is not None:
                g = torch.frombuffer(bytearray(seed), dtype=torch.float32).reshape(wanted)
                y.backward(g)
                eager_gradient = values(x.grad)
                if values(g) != seed:
                    raise RuntimeError("Torch changed seed")
            if values(x) != source:
                raise RuntimeError("Torch changed input")
            forward, gradient, frames = pool.adaptive_avg_pool2d(binary, shape, source, out, seed=seed)
            kh, kw = pool.lowering(shape, out)
            parent_output, parent_gradient, parent_frames = pool._run(
                parent, pool.PARENT_ELF_SHA, shape, source, wanted,
                (kh, kw), (kh, kw), (0, 0), False, seed, refusal=None if backward else 1)
            if backward and (parent_output, parent_gradient) != (forward, gradient):
                raise RuntimeError("existing parent backward route changed")
            eager_forward = values(y)
            if not (forward == eager_forward and gradient == eager_gradient):
                raise RuntimeError(f"raw native/eager mismatch on fixture {index}, backward={backward}")
            cases.append(dict(shape=shape, output_size=out, backward=backward, frames=frames,
                              parent_frames=parent_frames, parent_status=0 if backward else parent_output,
                              eager_native_forward_sha256=sha(forward), forward_words=len(forward)//4,
                              eager_native_backward_sha256=None if gradient is None else sha(gradient),
                              backward_words=0 if gradient is None else len(gradient)//4))
    direct_controls = []
    # Unlike regular adaptive bins, these windows overlap and clip at the border.
    shape, wanted = (2, 2, 5, 6), (2, 2, 3, 5)
    source = make_values(shape, 5)
    for include in (False, True):
        x = torch.frombuffer(bytearray(source), dtype=torch.float32).reshape(shape)
        eager = torch.nn.functional.avg_pool2d(x, (3, 2), (2, 1), (1, 0), count_include_pad=include)
        fwd, _, frames = pool._run(binary, pool.ELF_SHA, shape, source, wanted,
                                  (3, 2), (2, 1), (1, 0), include, None)
        if fwd != values(eager):
            raise RuntimeError("ordinary overlapping/padded average pool mismatch")
        old, _, parent_frames = pool._run(parent, pool.PARENT_ELF_SHA, shape, source, wanted,
                                         (3, 2), (2, 1), (1, 0), include, None, refusal=1)
        direct_controls.append(dict(include=include, parent_status=old, frames=frames, parent_frames=parent_frames,
                                    forward_words=len(fwd)//4, eager_native_forward_sha256=sha(fwd)))
    return dict(schema="heptatorch.opt-in-adaptive-avg-pool2d.v1", status="PASS_SCOPED",
                native_elf_sha256=pool.ELF_SHA, torch_version=torch.__version__, torch_cpu="AVX2",
                torch_threads=torch.get_num_threads(), native_wire_frames=sum(x["frames"] for x in cases),
                parent_elf_sha256=pool.PARENT_ELF_SHA, reference_custody=custody,
                cases=cases, direct_average_pool_controls=direct_controls,
                full_pytorch_rewrite_complete=False, production_qualification=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "parent", "wheel", "install"):
        ap.add_argument("--" + name, required=True, type=Path)
    opt = ap.parse_args()
    print(json.dumps(run(opt.binary, opt.parent, opt.wheel, opt.install), sort_keys=True, indent=2))
