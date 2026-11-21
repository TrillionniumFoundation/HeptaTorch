# F64 packet consumer on the current native objects

The original standalone F64 component compiled fourteen HIA functions and
restored the historical143 runtime solely to check four inherited helper
bodies. This consumer instead admits the complete current cold build and
links its fifteen actual native objects. It compiles three original adapter
bodies: packet entry, byte writer and scalar `sqrt64`. Their only edits are
collision-free symbol renames, checked by an exact inverse. The other eleven
original bodies must be byte-identical in the canonical current source.

Add, subtract, multiply, divide, widen and narrow therefore call the actual
current object kernels. Scalar sqrt remains the original RNE component; this
adapter does not add a public Tensor sqrt operation. No historical143 source
or historical standalone image is built, restored or executed by this path.

## Build and execute

Use the actual source-bound current build and the fixed ABI3 flat runtime
linker produced by the stack toolchain:

```sh
python3 -I -S -B tools/current_f64_consumer/build.py \
  --repository "$PWD" --current-build /absolute/current-build \
  --flat-linker /absolute/stack-build/native/stack-linker-runtime \
  --output /new/f64-consumer-build
python3 -I -S -B tools/current_f64_consumer/run.py \
  --repository "$PWD" --current-build /absolute/current-build \
  --consumer-build /new/f64-consumer-build --reference cpython \
  --output /new/f64-cpython
python3 -I -S -B tools/current_f64_consumer/qualify.py \
  --repository "$PWD" --current-build /absolute/current-build \
  --consumer-build /new/f64-consumer-build --directory /new/f64-cpython \
  --output /new/f64-cpython-readback.json
```

The PyTorch run additionally requires `--reference pytorch`, the successful
`--cpython-directory /new/f64-cpython`, and the actual pinned interpreter via
`--reference-python /absolute/reference-python`. The original eleven
installation predicates and all current raw-wheel custody checks are
mandatory. Prepare the fixed wheel and installed snapshot as documented in
`docs/current-runtime.md`; there is no local-provenance override.

Every run preserves three original comparison bodies:

1. All **55** original strict cases, including sixteen malformed-packet cases.
2. All five original sqrt diagnostic families and the original shape probes.
3. The complete **55-case** RNE/accuracy repeat, including all eighteen integer
   boundary controls, both independent native sqrt checks, the strict
   reference-error bound and comparisons against the original strict reports.

The final reference process records every native request and response,
reference word, process intent, outcome and both EOF streams. The pure reader
reconstructs the original corpora and repeats the unchanged assertions from
physical evidence. It also rechecks every current object, every actual compiler
and linker stream, the selected native entry and the complete stack graph.
PyTorch mode reads the actual before/after installed files, official wheel,
PEP610/pip/RECORD/version data and mapped Torch computation libraries.

The strict result remains `FAIL` if any bitwise comparison differs. Successful
RNE/accuracy checks are reported separately as `PARTIAL`; they do not turn a
strict sqrt mismatch into equality. Non-sqrt mismatches, incomplete evidence
or failed rational/integer checks block the component gate. The complete
PyTorch rewrite remains open.

## Integration

Run this consumer after the current-runtime cold build and pinned-reference
installation. Store its build and both executions below `CURRENT_OUTPUT` so
the complete original archive includes them. Preserve the old F64 workflow
until the new path passes on the exact candidate, including strict reference
custody and independent physical readback. `tools/dtype-f64/f64.hepta` remains
an immutable source witness for the three original adapter bodies and their
inverse; it is not compiled as a separate fourteen-function application here.
