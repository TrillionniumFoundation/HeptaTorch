"""Replay the #102 pinned integration checks on the nested-unsqueeze ELF."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from project import SOURCE
from qualify import ELF

OLDER = Path(__file__).resolve().parent.parent / "opt_in_spatial_ad"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for cap in ("binary", "spatial-parent", "source", "wheel", "install", "output"):
        p.add_argument("--" + cap,type=Path,required=True)
    a = p.parse_args()
    assert not sys.flags.optimize
    spec = importlib.util.spec_from_file_location("spatial_qualification", OLDER / "qualify.py")
    earlier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(earlier)
    earlier.ELF, earlier.SOURCE = ELF, SOURCE
    # Only the candidate identity guards change. The exact #94 baseline,
    # fixture data, native command state, and assertions stay unchanged.
    sys.argv = [str(OLDER/"qualify.py"),"--binary",str(a.binary),"--parent",str(a.spatial_parent),
                "--source",str(a.source),"--wheel",str(a.wheel),"--install",str(a.install),
                "--output",str(a.output)]
    earlier.main()
    previous = json.loads((OLDER/"qualification.json").read_text())
    current = json.loads((a.output/"qualification.json").read_text())
    current["candidate_elf_sha256"] = previous["candidate_elf_sha256"]
    current["source_sha256"] = previous["source_sha256"]
    assert current == previous
    print("Exact #102 report replay; executable and source pins are the only changes.")


if __name__ == "__main__":
    main()
