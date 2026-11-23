# Retained numeric predicates on the exact joint image

This replay covers the original 485-case numeric-predicate suite on the
opt-in joint abs/logical-not runtime. No canonical source, test predicate,
expected result, or observer data is changed. The original unknown-command
control still rejects opcode 125. It does not assert that the newly assigned
opcode 127 is unknown.

Two call sites (native observation and independent raw readback) switch
reversibly from the historical build reader to a narrow joint profile. At
each site the unchanged canonical reader must first reject the actual joint
image, and the unchanged original historical lineage proof then executes on
the separately authenticated canonical build. The joint profile independently
binds the exact candidate ELF, all 15 source/object pairs, its linking packet
and opt-in receipt. Its full monolithic source must round-trip between the
two exact 800-function sources with only four changed function bodies:
`entry`, `predicate_bits`, `predicate_command` and `wide_neg_tensor`. These
checks establish lineage and identity, not inherited execution results.
Both native observations and physical readback execute against the joint
image or its actual captured wire/owner/mapping frames, respectively.
All 62 checkout input pins match the reviewed #63 tree.

After reproducing fresh canonical and joint builds as in the
[dual build guide](README.md), prepare the pinned official Torch wheel and
install report as in the Current HIA Runtime workflow. Use fresh outputs:

```sh
P="$REPO/tools/current_runtime/opt_in_dual_unary"
python3 -I -B "$P/replay_numeric_predicates.py" \
  --repository "$REPO" --baseline-build "$BASE_BUILD" --dual-build "$DUAL_BUILD" \
  --dual-scoped "$P/qualification.json" --torch-python "$TORCH_PYTHON" \
  --output "$OUT/numeric-predicates"
```

The original reference supervisor freshly observes the exact installed
PyTorch oracle; the original reader requires its pinned wheel evidence.
No PyTorch output comes from a prior baseline run. Actual reports,
transcripts and reference artifacts remain in the output directory. Compact
receipts include hashes of the observed reports; those bytes can vary with
paths, process identities and timings.

This is one mandatory suite, not a pass of the complete 35-suite runtime
gate or production activation. A canonical build, the abs-only build, or
an unrelated opt-in receipt is rejected before starting a native observer.

The confirmed original coverage includes 485 ordered cases and 11
additional sessions: 11,244 frames, 42,645 assertions, 3,615 owner snapshots,
10 mapping snapshots, 56 protocol controls, five successful and five
failed FE transactions, and five actual allocation recoveries. See
[numeric-predicates-evidence.json](numeric-predicates-evidence.json).
