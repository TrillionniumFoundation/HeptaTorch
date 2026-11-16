# NativeDropout, bool Storage and captured AD history

This separate CPU profile implements binary32 and actual one-byte bool Storage
in native HIA. Its source is
`6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3` and ELF is
`830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51`.
There are 523 core functions, 6,881 blocks, 25,112 instructions and eleven
independently compiled native units. The descriptor is 640 bytes; state and
backing tags are `HTDTYP03` and `HTODTY03`. Earlier profiles keep their original
sources, images, formats and evidence.

## Native behavior

Commands 96–100 expose dtype metadata, typed construction, logical byte reads,
native_dropout and typed fill. Bool allocation uses one byte per element.
Qualified aliases, views, clones, Saved values and weak-owner operations retain
the real dtype and storage width. Existing binary32-only operations reject an
unsupported dtype before a four-byte access. Command 102 performs actual bool
logical_not_ with the original Tensor, Storage and Version identities.

NativeDropout returns a real mask Tensor with its own Storage and Version.
For the fixed observed empty cases, output identity and history match the
actual reference: changing a Tensor's current history does not retarget edges
already captured by older graphs. Captured AccumulateGrad targets are real
strong roots. Relative immutable history order guides native backward
execution. A legitimate strong cycle remains live until an operation such as
the actual command101 detach_ breaks it; observation alone must not alter that
ownership.

Retained prehooks may commit their completed gradient before a Saved-version,
freed-Saved or graph-interior error in the explicit partial-backward path.
The independent recipes check these precedence and lifetime distinctions,
while the earlier atomic entry points preserve their complete owner rollback.
An existing optimizer retains a legal member even if a later empty operation
changes that Tensor's current leaf/history status. Its validated private
zero-grad operation uses the owned member without relaxing construction rules
for new optimizer members or ordinary public zero-grad commands.

The SGD correction follows the actual effective gradient. Maximize first
creates the negated pointwise result; nonzero weight decay then creates the
next pointwise result. Initial momentum uses that result's layout. Plain SGD
keeps the previous clone path. Candidate10→11 changes one existing occurrence
body and adds two helpers; the other 520 old function bodies, numerical SGD
element computation and Version increments remain unchanged.

## Evidence and separate counts

The [producer qualification](../validation/native-dropout/qualification.json)
records thirteen named steps, including a genuinely new eleven-unit build,
typed lifetimes, dtype refusals, actual allocation failure and recovery, the
original NativeDropout28, original183-derived8, true-Mul history controls,
existing-optimizer6, effective-SGD-layout6, fixed retained12 and sixteen named
inherited suites. These overlap and must not be summed into unique upstream
test or schema counts. The retained12 success label is accepted only with its
exact fixed reference, twelve executions, twelve qualified variants, zero
failures and all eleven provenance predicates.

Independent gates remain separately executable:

| Gate | Actual result before hosted integration | Evidence |
| --- | --- | --- |
| Nine fixed empty-history recipes | 13 native sessions, 963 frames, 6,888 assertions, 8,388,288 exact rollback bytes and 353,920 full MT-state bytes | [identity9 qualification](../validation/native-dropout-identity9/qualification.json) |
| All 26 actual detach_ cases and all 26 subsequent mutations | 1,676 frames, 14,761 assertions and 960 exact values | [resource replay commands](../validation/dtype-resource-review/cli.json) |
| Both typed-owner directions | 21 actual compiled modules, 12 frames and actual close/unmap | [resource replay commands](../validation/dtype-resource-review/cli.json) |
| Short-output and entry behavior | 12 actual compiled modules, 42 probe processes and 51 interactive frames | [resource replay commands](../validation/dtype-resource-review/cli.json) |
| Finite 640-byte owner-accounting controls | Four native controls, 68 frames, 136 full structural snapshots and 64 explicitly isolated allocation/refresh ledgers; 24 copied-snapshot negative controls refused | [owner-accounting package](../validation/dtype-owner-accounting/package-manifest.json) |

The four finite owner controls distinguish full structural snapshots from
allocation ledgers: four backward/step events were checked structurally and do
not claim a qualified allocation ledger. The integrated cold coordinator now
passes all six independent gates on a newly compiled eleven-unit image; its
[complete execution record](../validation/native-dropout/integration-final/report.json)
and mode-preserving raw archive retain the two earlier failed coordinator runs,
including the entry gate that did not run with an incomplete tool bundle.

The same fixed 523 source and executable also passed actual locked PyTorch
execution on ROG: the unchanged live32 oracle passed 32 cases and 2,132 frames;
the unchanged frozen35 numerical oracle, with a separate complete 640-byte
ownership observer, passed 35 cases, 6,344 frames and 2,590,274 values. All eleven
provenance checks, original EOF and post-close Version gates passed. The 283
legacy explicit-state Version differences remain recorded; parameter Version
differences remain zero. These runs reused the frozen executable and do not
claim another cold compilation or execution of the new GitHub workflow.

The [full original evidence](../validation/dtype-frozen35-accounting/evidence/qualification-summary.json)
preserves the first 32-case close-ledger failure and the successor observer's
four Storage-metadata failures. The independent observer correction follows
the unchanged native release body: expired Storage clears its actual offset
and rounded allocation while retaining its historical requested byte count.
Same-generation transitions must preserve that count and cannot resurrect a
released payload. A real six-frame control, four copied corruptions and the
original eighteen copied-snapshot refusals qualify the correction separately.
All 450 remote and 328 supplemental archive members are retained with exact
bytes, modes and stream EOF checks. The new GitHub source integration remains
pending its own actual CI and original-artifact admission.

## Reproduction

The frozen producer entry is `tools/native_dropout/qualify.py`. Follow its
[complete tool and parent-build instructions](../tools/native_dropout/README.md)
using a real source-bound 446 parent, the pinned HIA checkout, compiler, linker
and flat compiler. Give every run a new output directory. Default dependency
packages resolve in the integrated repository; no retained developer scratch
directory is required.

Use the independently frozen
[identity9 command](../tests/native_dropout_identity9/README.md) against the
newly produced native build. Its protocol is the exact
`tests/retained_nonleaf_review/protocol.py` and same-directory owner client;
the producer's different protocol helper is not interchangeable. The
[three resource commands](../validation/dtype-resource-review/cli.json) and
`tests/dtype_owner_accounting/qualify.py --repo REPO --build NATIVE523 --output NEW`
run the other independent gates. The producer's PASS does not substitute for
any of them.

## Failure history and remaining scope

All original failures are retained. Candidate04's bool Hold path entered a
binary32 reader. Earlier profiles lacked the observed empty-history behavior.
Candidate07 rejected an existing optimizer after a legal same-Tensor history
change. Candidate10 passed identity9 but failed one actual optimizer-layout
recipe and two supplemental maximize-layout recipes. Candidate11 passes
those same immutable references. Host-authoring rejections, observer startup
errors and the retained12 aggregate-label mistake are separately identified;
none is reclassified as native execution or overwritten by a later PASS.

The older183 observation ledger retains its original OPEN classifications.
Additive native replays do not rewrite that frozen historical report.
General promotion, additional floating and integer Tensor dtypes, complex,
sparse and quantized layouts, complete Python objects and diagnostics, forward
and higher-order AD, general hooks, accelerators, distribution and the other
whole-system services remain open. Full upstream schema qualification stays
0 of 2,590, and whole-project acceptance remains HOLD across seventeen gates.
