# Opt-in logical-not: strided layout and gradient-boundary qualification

This is a separate, strict comparison of opcode 127 against the pinned
PyTorch CPU wheel. It does not change the runtime source, its default dispatch,
or any mandatory acceptance suite.

Start from the exact
[#61 hosted reference](https://github.com/TrillionniumFoundation/HeptaTorch/tree/f4f3bd5b44a14868e1af532e689b07c4ff197a89)
as `BASE`. Apply the [existing opt-in build instructions](INTEGRATION.md):
all 15 source partitions, the canonical runtime, and their manifest have the
same Git blob identities as the original #57 source. On Dot, both images were
cold-built from these exact sources with the pinned compiler and linker.
The baseline reproduced `083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5`;
the isolated candidate reproduced
`0cd10504717c5e5b01114455c55030e63a22b62bb1ba43f188355d399d6c158a`.
The original 849-frame native regression and the existing 28-case strict Torch
receipt both replayed successfully on this candidate.

## Run the independent comparison

Use `PYTHON` for the oracle environment containing the locked PyTorch
distribution, `INSTALL_REPORT` for that environment's verified installation
report, and `OUT` for the opt-in build directory, with the same meanings as
in the build instructions. The comparison checks the exact candidate ELF,
exact original protocol and owner transport, and locked wheel provenance
before starting native observation.

```sh
"$PYTHON" -s -E -B -W error qualify_runtime_layout_torch.py \
  --protocol "$BASE/tests/differentiable_view_native_review/protocol.py" \
  --binary "$OUT/candidate-build/heptatorch" \
  --oracle-lock "$BASE/compatibility/pytorch-oracle-lock.json" \
  --install-report "$INSTALL_REPORT"
```

The [receipt](layout-torch-evidence.json) records 214 cases, 3,205 native
frames, and 12,648 numerical comparisons. The F64 cases include ranks 2–4,
dense permutations with and without storage offsets, strides with holes,
broadcast and overlapping strides, singleton axes, and empty dimensions.
The F32 scalar, dense and empty cases test that a gradient-enabled input
remains unchanged and the new Boolean result is detached from autograd.
Every checked result has matching values and layout and its own native
Storage and Version identities. The script returns nonzero on a mismatch.
The baseline image, a wrong oracle lock and a wrong installation report
were each refused before native observation; a repeated admitted run
produced identical receipt bytes.

This covers out-of-place real CPU inputs only in the listed layout and
gradient profiles. It does not implement the `out=` overload, complex inputs,
or activation in the canonical image. The 35 mandatory suites on #61
qualify the baseline ELF only, not the candidate: the existing F64 view
source transition, public-view native/receipt readers, public logical192
source/image loader, and hosted final qualification explicitly pin the
original runtime. Candidate adoption requires a new exact reversible
source profile and fresh full replay, not replacing original evidence or
permitting arbitrary successors.
