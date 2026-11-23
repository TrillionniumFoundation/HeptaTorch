# Exact current800 dual opt-in qualification

This is a scratch-only joint experiment over the exact hosted-green
[#61](https://github.com/TrillionniumFoundation/HeptaTorch/pull/61) head
`f4f3bd5b44a14868e1af532e689b07c4ff197a89`. It composes the bounded
[#59](https://github.com/TrillionniumFoundation/HeptaTorch/pull/59) logical-not
and [#62](https://github.com/TrillionniumFoundation/HeptaTorch/pull/62) wide
Tensor abs source transitions without touching the original sources,
observers, receipts, or default dispatch.

The producer independently re-derives both single-operator sources and verifies
both composition orders. Undoing one transition gives the other preexisting
source byte-for-byte, and undoing both gives canonical current800 byte-for-byte.
Only `wide_neg_tensor`, `predicate_bits`, `predicate_command`, and `entry`
bodies change. Unit 10 also carries the reversible #59 source/export ordering of
the unchanged `logical_truth` callee. All 15 source/object identities are bound;
the three objects for units 06, 10, and 12 exactly match their independently
compiled single-operator objects. Two cold builds produced the same ELF
`5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72` with a
133,840-byte maximum native stack path.

## Reproduce

Use an exact #61 checkout as `REPO`, and pin the compiler and linker as in its
build guide. Set `LOGICAL` to the logical-not directory from #59
`3ba4a0e0bbfea224bcbfdee795d2d0b5611b51ff`
(`tools/current_runtime/typed_logical_not`), and `ABS` to the opt-in wide
abs directory from #62 `5127e9e548736eaf169b590023e8349458a7d1fe`
(`tools/current_runtime/opt_in_wide_abs`). Use the locked installed CPU Torch
interpreter, installation report and fresh scratch paths.

```sh
BASE="$REPO/tools/current_runtime/source"
BUILD="$REPO/tools/current_runtime/build.py"
python3 -I -B prepare_dual.py --base-source-root "$BASE" \
  --logical-transform "$LOGICAL/prepare_runtime_candidate.py" \
  --abs-transform "$ABS/prepare.py" --output "$SCRATCH/dual-source"
python3 -I -B "$BUILD" --manifest "$BASE/manifest.json" --source-root "$BASE" \
  --compiler "$COMPILER" --linker "$LINKER" --output "$SCRATCH/baseline"
python3 -I -B "$BUILD" --manifest "$SCRATCH/dual-source/manifest.json" \
  --source-root "$SCRATCH/dual-source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$SCRATCH/dual-build"
"$TORCH_PYTHON" -s -E -B -W error qualify_dual.py --logical-dir "$LOGICAL" \
  --abs-dir "$ABS" --protocol "$REPO/tests/integer_tensor_review/wire/protocol.py" \
  --oracle-lock "$REPO/compatibility/pytorch-oracle-lock.json" \
  --install-report "$PIP_INSTALL_REPORT" --baseline "$SCRATCH/baseline/heptatorch" \
  --binary "$SCRATCH/dual-build/heptatorch" --output "$SCRATCH/dual-qualification.json"
```

The qualification authenticates the standalone observer source and compiles
explicit in-memory substitutions only for the two candidate ELF identities.
It leaves all frozen standalone files and their original identities unchanged.
On the one joint image it reproduces the 849-frame logical-not native test,
the 28-case strict fixed-wheel test (144,027 comparisons), the 214-case layout
and gradient-boundary test (12,648 comparisons), and the 50-case abs test
(17,192 raw comparisons plus 131,072 after two genuine allocation refusals).
Six new single-frame `logical_not(abs(x))` cases compare both operators' values,
layouts and separate ownership against PyTorch. In six reverse-order frames
where wide abs receives a Bool tensor, refusal rolls back the whole transaction
to identical owner state; the same Bool refusal is checked on the baseline.
Both standalone #62 historical wide unary-7 refusal rows remain baseline-only.
Wrong base manifest, modified transform and standalone-only executable are
rejected without generating a receipt.

See [source-bindings.json](source-bindings.json) and
[qualification.json](qualification.json) for exact identities and observations.
This is not the 35 mandatory suites on the joint image, production registration,
autograd for abs, `out=`, or whole PyTorch parity.
