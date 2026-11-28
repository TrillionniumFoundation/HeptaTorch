# Exact-main 775: 29 focused groups passed

The [Current HIA Runtime workflow](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44912529511/job/136871176361)
ran on exact main commit `e9444902df9878060ab389a70ee957c1b7f4a887` and
executed all 29 groups and 13,851 cases on the frozen CPU runtime, with
separately linked current-object Generator probes. The
[record](exact-main-ci.json) binds the original archived build, manifest,
native executable, aggregate source and actual qualification payload,
not the sum of prior component results.

The [original GitHub Actions ZIP](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44912529511/artifacts/14713706649)
has 645,877,944 bytes and is due to expire December 21, 2026 at 15:43:37 UTC.
The original can also be reconstructed byte-for-byte from the four original
transport artifacts on the same run. With the original ZIP available:

```sh
python3 -I -B validation/current-runtime775/review_main_775.py original.zip
```

The reader checks its official digest, all eight ZIP CRCs and the selected
native members, and checks the actual report's exact suite list, case
counts and pass status. It reads the original seal's byte/mode/input/EOF
claims; it does not independently re-admit its 277,502 inner entries or
rerun HIA/PyTorch. Earlier failed runs remain failed. This focused result
is exact-head evidence only. It neither grants full-tree, schema or subsystem
closure, nor replaces qualification of a different head or an opt-in candidate.
The full project remains **HOLD**.
