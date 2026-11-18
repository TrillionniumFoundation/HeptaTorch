# Fixed logical-reduction observations

`report.json.xz` contains the exact 223,897-byte report from an actual run of
`tests/logical_reductions/reference.py` on the pinned PyTorch 2.15.0 CPU wheel.
Its decompressed SHA-256 is
`e2874986d52e62725d54767ea5824a76d066e948b5d54a99e3b24f9e32c0e425`.
All 147 requested observations completed: 120 operations returned values and
27 produced their observed expected errors. The original eleven wheel/build/
installation checks, unchanged inputs and unchanged RNG state all passed.

`manifest.json` binds the original bytes, compressed archive, current input
recipes, exact reference source and wheel provenance. `execution.json` records
the original remote job, command, terminal status and complete stdout/stderr
observations. Its source-receipt digest identifies the complete original
HeptaRuntime receipt from which this compact record was selected. No native
runtime is executed by the reference producer.

The native comparison is reproduced without installing Torch:

```sh
python3 -I -S -B tests/logical_reductions/qualify.py \
  --build-directory build/current-runtime --output /new/qualification
```

The wrapper first admits the complete current cold build using
`tools/current_runtime/qualification.py`, restores the exact frozen reference,
and executes all native comparisons and boundary controls. It requires all
147 cases, 18 packet/transaction controls, two actual allocation failures with
successful identical-request recovery, 150 complete native child exits, and
both output EOFs. Missing or incomplete outputs fail the qualification.

`local-validation-notes.json` retains the initial allocation-test setup failure
and its correction, original report/source identities and subsequent executed
results. The correction changed resource accounting in the test setup; native
HIA and actual expected Torch observations remained identical. These notes do
not change the frozen reference or convert it into native execution evidence.

`previous-138-report.json.xz` and `previous-138-execution.json` preserve the
original 138-case observation and its actual execution record. The current
reference repeats those 138 rows exactly and adds nine observations for
gradient-input axis-validation precedence. The wrapper verifies this unchanged
prefix independently; the earlier report is never treated as a new native pass.
