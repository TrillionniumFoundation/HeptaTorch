# Native int64 Tensor profile 07

This package authors and cold-builds the isolated 544-function, 640-byte
descriptor successor of the frozen 523-function native-dropout profile.
Tensor allocation, conversion, arithmetic, reference counting and execution
run in HIA. Python performs source generation, byte-custody checks, orchestration
and observation of real native responses. It does not evaluate Tensor kernels.

## Fixed identities

| Component | SHA-256 |
| --- | --- |
| Parent 523 source | `6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3` |
| Parent 523 image | `830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51` |
| Int64 07 source | `adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62` |
| Int64 07 image | `b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08` |
| Native module compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Native session linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |
| Flat compiler used by one inherited helper test | `c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6` |

`profile.json` fixes all 11 complete source/object pairs, 544 functions,
7,263 blocks and 26,361 instructions. Internal formats are `HTINT002` and
`HTOINT02`. The parent and earlier candidates remain separate immutable inputs.

## Cold construction and producer qualification

Install the three package namespaces at the repository root. `PARENT523` must
be a real preceding 523 build containing its runtime, overlay authoring, RNG
and owner modules, and executable. `UPSTREAM` is the already-pinned canonical
HIA authoring bundle; its files are verified against the existing repository
profile. `STACK` contains the fixed native compiler/linker executables. The
output path must not exist.

```sh
python3 -B tools/integer_tensors/qualify.py \
  --parent "$PARENT523" \
  --repository "$REPO" \
  --upstream "$UPSTREAM" \
  --compiler "$STACK/native/stack-module-compiler" \
  --linker "$STACK/native/stack-linker-session" \
  --flat-compiler "$FLAT_COMPILER" \
  --output "$NEW_OUTPUT"
```

The same interface also works from a separately staged package: invoke its
`tools/integer_tensors/qualify.py` and point `--repository` to the unchanged
parent repository. No scratch-path overrides or prebuilt int64 image are
accepted. `rebuild.py` invokes the actual native compiler for every module and
the actual native linker, then requires all 11 object hashes and the final
source/image to match `profile.json` exactly. `native/build.json` is the real
builder report; `native/profile-rebuild.json` records the comparison with the
frozen profile. This distinction is retained for downstream consumers.

The qualifier has ten named top-level gates: one cold build, seven fixed
reference replay gates, the independent unchanged actual40 replay, and one
inherited behavior coordinator. The latter
executes twelve named gates, including the original sixteen inherited gates.
Counts overlap, and no sum is presented as unique upstream API coverage.

| Fixed-reference gate | Scope and exact count |
| --- | --- |
| Original storage | Actual54 rows 0–11; 12 recipes, 235 frames |
| Original arithmetic | Actual54 rows 12–41; 30 recipes, 1,122 frames |
| Original native dropout | Actual54 rows 42–53; 12 recipes, 333 frames |
| Alias mutation | 8 recipes, 16 real Scalar mutations, 516 frames |
| Signed int64 to binary32 RNE | 5 fixed groups, 1,237 integers, 60 frames |
| CPU promotion | 20 recipes, 40 original mixed calls, 2,280 frames |
| Ultimate-base lifetime | 6 recipes, 450 frames; 168 private-role checks reported separately |
| Independent in-place/overlap/error order | 40 recipes, 1,747 frames, 7,178 exact checks |

The three original54 gates must cover the exact ordered indices 0–53 once.
Every reference is bound by complete bytes, probe hash and all eleven fixed
Torch provenance checks. These commands replay stored actual Torch results;
they do not claim to run Torch again. The old F32/bool regression preserves
each historical status and every explicit gap. In particular, retained12
must report exactly `QUALIFIED_EXPOSED_SCOPE`, twelve executed and qualified
variants, zero unexpected failures and the frozen `ca8b5d…` reference. Other
non-FAIL labels are never generalized into a pass.

Source adapters in `tests/integer_tensor_review/inherited-adapters` change
only the documented private tags, dependent observer hash and executable
basename. Their inverse restores every original script byte. The portable
coordinator additionally resolves original files relative to `--repository`;
its independent inverse receipt records that one path-only change. Original
inputs, tolerances, numerical, Version and lifetime assertions are unchanged.

## Raw evidence and failed runs

Every subprocess writes its complete stdout/stderr and original report.
Completed gate output is streamed into XZ tar archives; every member's bytes,
size and mode are compared, and the XZ stream is read to its complete EOF.
Only then may generated duplicate raw files be removed. Reports and
`RESTORE.json` indexes remain beside the original output paths. The cold native
source, image and all modules stay expanded for independent consumers.

An inherited gate may itself contain verified archives. The enclosing archive
keeps those bytes and their original manifests; restore that enclosing gate
before following the nested restoration index. A failure does not overwrite
an earlier output directory or relabel an earlier candidate. A new attempt
requires a new output path. Historical producer, host-observer and native
failures are delivered separately with their original identities and raw
archives, rather than being edited into the current success report.

## Required independent integration evidence

This producer qualifier does not substitute for the separately executed:

- Root's fixed actual40 in-place/overlap/error-order replay.
- Peer resource, 8-byte extent, real allocation-failure and owned-entry gates.
- Legal parent523/int6407 owner consume/return/close in both directions.
- The unchanged live32 Torch oracle on this image.
- The complete strict35 oracle, its int64-profile structural accounting and
  original copied-snapshot rejection controls.

The fixed actual40 reader executes here from the separately installed
`tests/int64_inplace_review` and `validation/int64-inplace-review` namespaces.
Its complete package manifest is pinned in `profile.json`; the original reader
and all assertions are unchanged. The remaining independent reports must bind
this source and image in the integration coordinator.
Earlier 523 or candidate06 results do not transfer to 07. Producer PASS means
only that the named producer gates actually ran and matched their exact
contracts; full integration admission remains a separate decision.

## Explicit boundaries

`PROTOCOL.md` defines the exposed wire contracts. Dtype 2 is genuine signed
int64 stored in eight bytes. Commands 103/104 implement the listed binary
operations; 106 is a real int64 Scalar add with alpha 1. Signed overflow wraps
modulo 2^64. The RNE implementation uses integer operations and is compared
with actual Torch raw binary32 words, without a Python/binary64 intermediate.

Internal CPU promotion constructs real temporary Tensors before layout
selection. This does not expose public `Tensor.to`. Any requires-grad operand
in this mixed arithmetic interface is rejected explicitly; mixed-dtype
autograd is not claimed. True int64 views have an actual ultimate-base Tensor
root. The older F32/bool command69 remains its detached-alias interface.

The 105 structured late-error path commits observed RNG consumption and
releases unpublished temporaries. Its earlier capacity/allocation refusals
keep the existing whole-frame rollback contract; this does not assert every
PyTorch out-of-memory side effect. Error-category normalization is not a
Python exception-text frontend. Rank, capacity and other existing bounded
interfaces are unchanged. Full PyTorch completion remains false, the schema
denominator remains 2,590, and the overall project remains on HOLD.
