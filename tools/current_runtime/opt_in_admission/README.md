# Offline admission for opt-in current800 builds

Admit named baseline, logical-not, wide abs, and combined candidate evidence
without changing the original runtime readers. This is an **identity check for
retained files**, not a test execution or approval. A successful inspection
does not grant qualifying status, transfer a canonical PASS, or activate an
operator. The checker never invokes the compiler, linker, native binary, or
Torch. Its report hashes the producer's cold-build JSON, but a retained report
is not independent proof of historical execution.

For each named profile, the inspector pins the exact manifest, actual
`runtime.hepta` and the contents of all 15 source/object pairs. It verifies
the independently declared partition changes relative to the baseline, the
compiler/linker snapshots, and the linker packet rebuilt from the pinned
objects in order. It constrains the cold-build report's stated scope and
binds matching scoped receipts to the actual ELF. Receipt pins are fixed code
inputs, not digest values accepted from the supplied JSON. Output is
relocatable: build files are read from the passed build directory, not from
historical absolute paths named by the producer. Stderr and EOF checks
corroborate the retained producer report; the original canonical reader and
its full stack/input custody checks remain separate.

Run from the HeptaTorch repository root; build directories are the complete
cold-builder outputs (not extracted binaries alone) from each candidate's
own reproduction commands:

```sh
BASELINE_MANIFEST=tools/current_runtime/source/manifest.json
CHECK=tools/current_runtime/opt_in_admission/admit.py

python3 -I -B "$CHECK" --profile baseline --baseline-manifest "$BASELINE_MANIFEST" \
  --build "$BASELINE_BUILD"
python3 -I -B "$CHECK" --profile logical_not --baseline-manifest "$BASELINE_MANIFEST" \
  --build "$LOGICAL_BUILD" --strict "$LOGICAL_STRICT_RECEIPT" --layout "$LOGICAL_LAYOUT_RECEIPT"
python3 -I -B "$CHECK" --profile wide_abs --baseline-manifest "$BASELINE_MANIFEST" \
  --build "$ABS_BUILD" --strict "$ABS_SCOPED_RECEIPT"
python3 -I -B "$CHECK" --profile dual --baseline-manifest "$BASELINE_MANIFEST" \
  --build "$DUAL_BUILD" --dual-scoped "$DUAL_SCOPED_RECEIPT"
```

A copied baseline ELF, a single-candidate receipt substituted for dual
evidence, and an edited canonical PASS field cannot pass. The independent
controls run on genuine build files and temporary copies:

```sh
python3 -I -B tools/current_runtime/opt_in_admission/test_admit.py \
  --canonical-reader tools/current_runtime/qualification.py \
  --baseline "$BASELINE_BUILD" --logical-not "$LOGICAL_BUILD" \
  --logical-strict "$LOGICAL_STRICT_RECEIPT" --logical-layout "$LOGICAL_LAYOUT_RECEIPT" \
  --wide-abs "$ABS_BUILD" --abs-strict "$ABS_SCOPED_RECEIPT" \
  --dual "$DUAL_BUILD" --dual-scoped "$DUAL_SCOPED_RECEIPT"
```

## Why this remains separate from the historical reader

In the exact #61 source, the original
[`qualification.admit_build`](https://github.com/TrillionniumFoundation/HeptaTorch/blob/f4f3bd5b44a14868e1af532e689b07c4ff197a89/tools/current_runtime/qualification.py#L39-L55)
admits the exact checked-in manifest, source and image. On the real builds
it admitted the canonical baseline and refused all three alternatives at
line 51 before native execution. These refusals are expected and tested.
Do not swap its source manifest in place. At least four independent gates
must explicitly qualify a successor, each retaining the original refusal,
case and report identities for old runs:

1. The [800→797 source inverse](https://github.com/TrillionniumFoundation/HeptaTorch/blob/f4f3bd5b44a14868e1af532e689b07c4ff197a89/tools/current_runtime/public_f64_view_source_transition.py#L40-L50)
   requires the original complete 800 source hash. A candidate-to-800 inverse
   may establish source lineage, not runtime results.
2. The [F64 view native runner](https://github.com/TrillionniumFoundation/HeptaTorch/blob/f4f3bd5b44a14868e1af532e689b07c4ff197a89/tests/f64_view/run.py#L41-L59)
   uses that canonical admission and requires the exact original image; its
   [independent readback](https://github.com/TrillionniumFoundation/HeptaTorch/blob/f4f3bd5b44a14868e1af532e689b07c4ff197a89/tests/current_runtime_f64_readback.py#L458-L486)
   independently requires the same image and physical build evidence.
3. The [logical192 runner](https://github.com/TrillionniumFoundation/HeptaTorch/blob/f4f3bd5b44a14868e1af532e689b07c4ff197a89/tests/public_f64_logical/run736.py#L187-L218)
   requires its own exact source/image and helper identities before execution.
4. The [hosted final check](https://github.com/TrillionniumFoundation/HeptaTorch/blob/f4f3bd5b44a14868e1af532e689b07c4ff197a89/.github/workflows/current-runtime.yml#L644-L656)
   independently pins the required case/suite roster and the original
   source/image. The offline result is deliberately not that report schema.

Qualification now requires actual candidate execution under a dedicated
profile with independently retained original observations. Old acceptance
tests must not be reclassified to make these new candidates look green.
