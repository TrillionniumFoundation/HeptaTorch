# Native dynamic owned tensors

This independent profile executes the existing float32 tensor, autograd and
optimizer code in a genuinely growing HIA-owned allocation. It retains the
canonical 143-function source as the numerical authority and applies a checked
overlay for packet entry, storage/ID helpers, capacity preflights, and constructor
versions. The new public interface and limits are specified in `PROTOCOL.md`.

The frozen candidate is a concrete intermediate result. It is not a completed
PyTorch rewrite, unrestricted memory profile, or completed independent
Storage/Version object integration. Its absolute optimizer counters still need
strict pinned-PyTorch comparison and any necessary native correction.

## Build

Prerequisites are the original canonical tensor source, the pinned upstream HIA
authoring checkout, the qualified owned-module compiler, and the qualified
owned-session linker. The build verifies their source/binary identities and
emits only a new output directory:

```
python -I -S -B tools/owned_tensors/build_runtime.py \
  --repository /absolute/HeptaTorch \
  --source-root /absolute/pinned-hia-source \
  --compiler /absolute/owned-module-compiler \
  --linker /absolute/owned-session-linker \
  --output /absolute/new-dynamic-tensor-build
```

The source organizer validates baseline command/matmul/gather emission against
the exact canonical HIA functions before applying changes. Every unchanged
function is copied from canonical source byte for byte. It then creates two
ordinary numerical modules and a third typed owner/transaction module. The
actual native compiler admits each unit and the actual native linker checks
and connects them to the owned-session startup. Python performs authoring and
wire/file transport, not module admission, native linking, or tensor execution.

The build verifies the exact source/object/executable pins in `profile.json`,
then rechecks producer, source, and compiled unit snapshots after completion.
The output includes generated HIA, native ELF objects, link request, executable,
and a per-function canonical/candidate provenance map. No full second numerical
source tree is vendored here.

## Native qualification

```
python -I -S -B tools/owned_tensors/test_dynamic.py \
  --service /absolute/new-dynamic-tensor-build/owned-tensor-session \
  --build-report /absolute/new-dynamic-tensor-build/build.json \
  --output /absolute/new-dynamic-tensor-qualification

python -I -S -B tools/owned_tensors/test_typed_state.py \
  --compiler /absolute/owned-module-compiler \
  --linker /absolute/owned-session-linker \
  --build-directory /absolute/new-dynamic-tensor-build \
  --output /absolute/new-typed-state-qualification
```

The script uses the neighboring qualified `tools/owned_sessions/client.py` for
pipe transport and own-child observation. It constructs expected results with
exact dyadic arithmetic or explicit independent binary32 rounding. The child
runs the real native HIA code with an empty PATH. No Python/C numerical runtime
substitutes for that code.

The original candidate's 11 groups passed 161 response frames before stdin EOF,
1,078,028 exact binary32 comparisons, and 59,310,080 bytes of whole-owner rollback
comparisons. They cover:

- A 2 MiB parameter through square, sum, backward, SGD and reset; every gradient
  and updated value is compared, with physical state growth from 8 to 16 MiB.
- The exact 65,535-descriptor limit, all 256 descriptor pages, a reset retaining
  the full prefix, and atomic refusal of the next allocation.
- A real 16,777,216-element tensor containing 64 MiB of data in a 128 MiB owner.
- Failed frames after mutation and growth, reset-generation rollback, stale
  handles after slot reuse, and actual native allocation failure under RLIMIT_AS.
- Thirty-two steps each of SGD, Adam and AdamW with 300 retained descriptors,
  a retained 2 MiB tensor, additional growth during training, and optimizer-state
  persistence. These tests establish native execution/state continuity; the
  strict external optimizer-counter comparison is a separate gate.

The independent storage reviewer also passed a nonuniform matmul/transpose/
repeated-index/gather/shared-input VJP across page and allocation growth, using
exact Fraction expectations and the original qualified numerical runtime.
Original receipts retain their own hashes and are not relabeled as a later
candidate or as an unperformed PyTorch run.

Six additional native typed-state cases passed: trusted live-empty and valid
state owners are accepted; a one-byte owner, a wrong-format buffer, an invalid
descriptor page and an invalid next-slot counter return status 7 while preserving
the exact original bytes. Those buffers are allocated by real HIA code in a
separately compiled qualification module. The numerical objects are unchanged.

`function-diffs.patch` records all 17 changed canonical functions individually.
`source-map.json` records every original function's old/new hash, along with the
seven added core helpers. `evidence/` contains the native result, independent
review, expected-value reasoning in the test source, and a compressed archive
of every original input/response stream. Rebuilding reproduces the same native
image; rerunning tests writes a new observation directory.
