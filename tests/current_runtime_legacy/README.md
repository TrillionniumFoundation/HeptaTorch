# Original 76 and F32 Sum32 suites on the current image

`run.py` executes all original view26, scalar32, view14, and input4 cases.
`sum32.py` executes the complete original F32 Sum32 comparator and all 626
reference events. Both bind the current 701-function source, ELF, and actual
cold-build receipt. All original case loops, comparisons, error expectations,
owner checks, and EOF assertions remain unchanged.

`identity-proof.json` records every identity-only edit. The coordinator checks
its complete inverse against the original reader files preserved with SHA256,
size, and mode in `validation/current-runtime-baseline`. The compact original view76
eight-reference fixture and shared Sum32 reference package remain byte-pinned.
The view76 fixture retains the exact original member bytes and archived
modes; its full historical packet can be restored using
`validation/view-reshape-r4-native76/archive-location.json`.

Use `make test` or `tests/qualify_current_runtime.py` for the complete current
gate. The low-level scripts accept `--build-directory`, the actual
`--expected-build-sha256`, and a new `--output` directory. Sum32 additionally
requires its original-source, parent-source, reference-source, and reference
arguments, all supplied by the common coordinator. `--prepare-only` does not
execute native code and cannot satisfy qualification.
