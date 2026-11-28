#!/usr/bin/env python3
"""Add native F32 frac to the exact mixed-precision rounding aggregate."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT_SOURCE = "e7728ab2c5eb35d6c3bd167afc4b1416cac770e5a458600843f5c3203c9d1cd7"
EDITS = {
    b'  upper_rnd:u64 = const() {"value":21}\n': b'  upper_rnd:u64 = const() {"value":22}\n',
    b'  value:u64 = call(w,oe) {"function":"opt_in_round_f32_word"}\n':
        b'  value:u64 = call(w,oe) {"function":"opt_in_round_or_frac_f32_word"}\n',
    b'  branch f64_good rounding_f64_native(sd64,od64,xd64) rounding_refuse(sd64)\n': b'''  frac64:u64 = const() {"value":22}
  no_frac64:bool = compare.ne(od64,frac64)
  allowed64:bool = bool.and(f64_good,no_frac64)
  branch allowed64 rounding_f64_native(sd64,od64,xd64) rounding_refuse(sd64)
''',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-build", type=Path)
    args = parser.parse_args()
    data = args.parent_source.read_bytes()
    if hashlib.sha256(data).hexdigest() != PARENT_SOURCE:
        raise ValueError("unexpected mixed-precision parent source")
    for old, new in EDITS.items():
        if data.count(old) != 1:
            raise ValueError("parent source did not have exact projection site")
        data = data.replace(old,new,1)
    candidate = data + (HERE / "frac.hepta").read_bytes()
    with tempfile.TemporaryDirectory(prefix="heptatorch-frac-") as temp:
        path = Path(temp) / "runtime.hepta"
        path.write_bytes(candidate)
        command = [sys.executable, "-I", "-B", str(args.runtime_tools.resolve() / "partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-f32-frac",
                   "--prefix", "frac785", "--output", str(args.output)]
        if args.expected_build:
            command += ["--expected-build", str(args.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
