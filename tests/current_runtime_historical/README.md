# Strict admission of complete historical readers

`run.py` executes either `--suite bool27` or `--suite integral24` through a
whole-source identity adaptation. `profile.py` restores the original 700
reader bytes from the fixed source fixture, proves the complete inverse,
and binds the current source, ELF, and function count. It preserves all
original case loops, expected values, comparisons, and refusal assertions.

The original reports and their expected-refusal `FAIL` results remain separate
artifacts. The outer admission requires all original numerical cases, every
native child exit and both EOFs, full raw frames, unchanged inputs, and only
the explicitly enumerated contract changes. Any other failure makes the gate
fail. The separate current F32 suite is mandatory for the new dtype contract.

Use the common coordinator to prepare the exact build, reference, and owner
arguments in a fresh output directory. The explicit entry requires
`--repository`, `--build-directory`, `--expected-build-sha256`,
`--bool-reference`, `--references-directory`, `--owner-directory`, and
`--output`. Preparation alone never transfers an earlier native PASS.
