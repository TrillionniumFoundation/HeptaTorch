# Independent 640-byte owner accounting, candidate 11

This package adds a read-only test observer for the exact native source
`6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3`
and executable
`830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51`.
The state tags are `HTDTYP03` and `HTODTY03`; the descriptor is 640 bytes.
The executable contains eleven independently compiled modules. The existing
355/428/446 observer remains byte-identical at SHA-256 `281f1ea437046a9cd884de2c8da34f38649a3b6482ade23123ac4aa992faed64`.

## What was executed

The four native controls passed in 68 frames. The observer checked 136 complete
committed snapshots and 64 exact isolated-command transitions. Four backward or
optimizer-step setup events received full structural checking only. The report
labels them `SNAPSHOT_ONLY`; no allocation ledger is claimed for those events.

The scenarios use actual native bool masks, Saved records, Weak and Hold
records, captured Accumulators, a genuine unrooted strong-reference cycle,
expired weak retained hooks, duplicate SGD parameters, Adam state Tensors and a
public state Tensor assigned as a gradient. The observer derives the private
reference counts from these records. It does not accept the counters as their
own proof.

The separate pure-data suite rechecks the 63 distinct captured snapshots and
64 original ledgers, then rejects all 24 precisely named mutations. These are
copies of JSON data. They do not start native children, import Torch, access
process memory or mutate an owner. They are not additional PyTorch APIs or
additional native cases.

The portable coordinator was then executed from its public CLI. Its two child
suites passed with the same four native scenarios and 24 copied controls.
This verifies the packaging path; it is not counted as eight unique native
scenarios. No live Torch was run by this package.

## Portable entry point

From a repository containing the unchanged retained-resource protocol and
legacy observer, run:

```sh
python3 -I -B tests/dtype_owner_accounting/qualify.py \
  --repo /absolute/repository \
  --build /absolute/candidate11-native-build \
  --output /absolute/new-review-directory
```

The build directory must contain `build.json`, `runtime.hepta`, the executable
`tensor-dtype-dropout-session`, and all eleven `.hepta`/`.o` unit pairs. The
coordinator verifies their pins and copies these inputs into its output. It
also checks the profile's three repository dependencies before and after use.
There is no hard-coded scratch path and no toolchain build in this command.

The final `report.json` embeds both child JSON reports, their original hashes,
argv, exit codes, stdout/stderr hashes, and all input identities. Success
requires `native_case_count=4`, `native_frames=68`,
`copied_negative_controls=24`, and unchanged inputs. The source-only review of
38 native ownership functions is separate from execution evidence.

## Exact accounting and limits

For each live node, the verifier derives:

- descriptor `+600` from Saved-mask ownership and captured-Accumulator targets;
- descriptor `+312` from actual Optimizer occurrence, initialized-key and
  state-Tensor roles;
- descriptor `+400` from actual public gradient bindings;
- descriptor `+208` from graph parents plus those three derived owner counts.

It also derives Storage strong/weak references, Version references, private
Hold roles and the pool's complete allocated/free extent partition. Public
Saved layouts and bool Weak/Hold dtype information are checked against actual
Storage anchors. Expected Saved versions may legitimately be stale after a
mask mutation; that is not a reason to discard their ownership relationships.

The prior retain hook is weak and may have an expired target. A cached
Accumulator backlink is also weak. The captured Accumulator's forward target
is strong. Strongly connected components are computed from actual owner arcs;
only a component containing a validated captured-Accumulator arc may explain
an unrooted ownership cycle. An anonymous unrooted acyclic node is rejected.

For the supported isolated commands, node and incarnation deltas come from
the command and prior state. Reusing a cached Accumulator during empty
`native_dropout` consumes a history token without a new node; the ledger
requires the exact one-node/one-refresh sequence. It never fills an unexplained
counter remainder. Unsupported transition shapes are rejected explicitly.

This test observer has an explicit 128 MiB owner-read budget. That is not a
production storage limit. It is neither arbitrary-state authentication nor
arbitrary-machine-code certification. It has not yet qualified the entire
frozen35 launcher for this new profile. The original numerical assertions,
Version checks and EOF checks remain independent requirements.

## Preserved failures and custody

The first native-control attempt stopped because the new test expected status
1 for a live Weak record whose Storage had expired. The frozen source and wire
contract specify status 6. The one-literal correction and exact inverse are
recorded; the original executed test, failing report and all raw streams remain
in the archive. No native or observer source was changed for that correction.

The first copied-control launcher used a nonexistent interpreter path. Its
child did not start; its `NOT_RUN` receipt remains. The successful invocation
uses the actual interpreter. The first archive attempt used an unsupported
`tarfile` stream keyword; that failed attempt and helper bytes also remain.
The corrected archiver writes an XZ stream explicitly, verifies every member
size/hash/mode and the XZ footer, then rechecks every original file.

`raw-evidence.tar.xz` contains both original and portable executions, all
snapshots, real transport bytes, original source, build inputs and these
failures. Its manifest is exhaustive. Child JSON absolute paths are original
observations, not rewritten claims that those old paths still exist.

The complete PyTorch rewrite remains **HOLD** with the existing denominator
2590 and full parity false. This component does not alter that gate.
