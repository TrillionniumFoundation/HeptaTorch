#!/usr/bin/env python3
"""Project three exact opt-in F32 unary kernels into one immutable main image."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
MAIN = "0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43"
OLD = b"  branch failed failure() route(s,operation,input)\n"
SLICES = {
    "square": (16, "3b113635f209d5f960bb780bf300ee4ae284931862a43c7ed7ce200ba8acef61"),
    "reciprocal": (17, "7e66fdce444f13d929691f9d878193e122a9ec86aec8139bfed9d318ba90a5d8"),
    "sign": (18, "7d54eca7f6048fa8f4baae2b1dcbe24ac1f1ca0f7084c2e1e8b91d70a57b1393"),
}


def pinned(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"unexpected SHA-256 for {path.name}")
    return data


def slice_inputs(parser):
    for name in SLICES:
        parser.add_argument(f"--{name}-dir", required=True, type=Path)


def projection(slices):
    routes = [f"  branch failed failure() {next(iter(SLICES))}_route(s,operation,input)\n"]
    definitions = []
    for (name, (wire, pin)), following in zip(SLICES.items(), [*list(SLICES)[1:], None], strict=True):
        definitions.append(pinned(slices[name] / f"{name}.hepta", pin))
        fallback = f"{following}_route" if following else "route"
        routes.append(f"""block {name}_route(s{wire}:mutbuf,o{wire}:u64,x{wire}:u64)
  d{wire}:u64 = call(s{wire},x{wire}) {{"function":"tensor_dtype"}}
  f{wire}:u64 = const() {{"value":0}}
  op{wire}:u64 = const() {{"value":{wire}}}
  isf{wire}:bool = compare.eq(d{wire},f{wire})
  iso{wire}:bool = compare.eq(o{wire},op{wire})
  ok{wire}:bool = bool.and(isf{wire},iso{wire})
  branch ok{wire} {name}_native(s{wire},x{wire}) {fallback}(s{wire},o{wire},x{wire})
block {name}_native(sn{wire}:mutbuf,xn{wire}:u64)
  y{wire}:u64 = call(sn{wire},xn{wire}) {{"function":"opt_in_{name}_f32"}}
  ret(y{wire})
""")
    return "".join(routes).encode("ascii"), b"".join(definitions)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--expected-build", type=Path)
    slice_inputs(ap)
    args = ap.parse_args()
    tools = args.runtime_tools.resolve()
    source = pinned(tools / "source/runtime.hepta", MAIN)
    begin = source.index(b"\nfn unary(") + 1
    end = source.index(b"\nend\n", begin) + len(b"\nend\n")
    unary = source[begin:end]
    if unary.count(OLD) != 1:
        raise ValueError("canonical unary route changed")
    routes, definitions = projection({name: getattr(args, f"{name}_dir") for name in SLICES})
    candidate = source[:begin] + unary.replace(OLD, routes, 1) + source[end:] + definitions
    with tempfile.TemporaryDirectory(prefix="heptatorch-three-unary-") as temp:
        path = Path(temp) / "runtime.hepta"
        path.write_bytes(candidate)
        command = [sys.executable, "-I", "-B", str(tools / "partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-three-unary-f32",
                   "--prefix", "triple778", "--output", str(args.output)]
        if args.expected_build:
            command += ["--expected-build", str(args.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
