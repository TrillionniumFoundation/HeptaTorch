# Integrated CPU forward and integral reductions

This combines the exact [#103 forward aggregate](https://github.com/TrillionniumFoundation/HeptaTorch/pull/103)
with the two native integral reduction helpers from [#105](https://github.com/TrillionniumFoundation/HeptaTorch/pull/105).
The projector checks the canonical unary body and every surrounding byte
against the pinned canonical source, extracts just the new dispatch blocks
and two HIA functions, then redirects two fallback edges to the unchanged
#103 rounding entry. The existing #103 F64 broadcast body is unchanged.

The candidate accepts the original scopes: whole-tensor integral native
product, min and max (wire 3/22–24), F32 rounding (3/19–21), earlier F32 unary
operations and scoped copysign forward. It does not add axes/dimensions,
grad variants, floating-point reductions or device support. This does not
change the canonical image, default or whole-PyTorch coverage.

## Reproduce

Use the original projectors for #103 `3b711a1f69d467ba48837e054cd0b1fdb88764ce`
and #105 `398decf3f5b9d4aa21ac0db3d803312b5b7c6d7f` to produce both
aggregate `runtime.hepta` files, the canonical runtime tools, and the locked
stack compiler/linker and Torch CPU environment. The original pip record
must also be available at `build/pytorch-install.json` in the repository
for the prior rounding verifier.

```sh
R=tools/current_runtime
D=$R/opt_in_forward_integral_reductions
P=/absolute/path/to/fresh-scratch
set -- --runtime-tools "$R" --forward-source "$EXACT_103_SOURCE" \
  --reduction-source "$EXACT_105_SOURCE"
cold() {
  python3 -I -B "$R/build.py" --manifest "$1/manifest.json" --source-root "$1" \
    --compiler "$COMPILER" --linker "$LINKER" --output "$2"
}
python3 -I -B "$D/prepare.py" "$@" --output "$P/src-a"
cold "$P/src-a" "$P/build-a"
python3 -I -B "$D/prepare.py" "$@" --output "$P/src-b" \
  --expected-build "$P/build-a/build.json"
cold "$P/src-b" "$P/build-b"
cmp "$P/build-a/heptatorch" "$P/build-b/heptatorch"
for i in a b; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
    --candidate "$P/build-$i/heptatorch" --output "$P/replay-$i" \
    --canonical "$EXACT_CANONICAL_ELF" --rounding-parent "$EXACT_93_ELF" \
    --reduction-parent "$EXACT_100_ELF" --install-report "$INSTALL_REPORT" \
    --rounding-verifier "$EXACT_97_VERIFIER" --reduction-verifier "$EXACT_105_VERIFIER" \
    --prod-verifier "$EXACT_100_VERIFIER"
done
cmp "$P/replay-a/qualification.json" "$P/replay-b/qualification.json"
```

The original F32 and integral verifier bytes are pinned; only their
expected candidate digest changes. Original parent images, refusals,
input immutability, and physical Storage/Version checks stay in place.
A distinct native process interleaves I64 max → prod, Bool min → I64 prod,
and F32 ceil → square, checking both owners and released-input readback.
The published evidence distinguishes the additional F64 broadcast
regression from the primary replay above.

For the separate F64 regression reported in the evidence: stage the exact
#90 and #96 verifier directories as the siblings
`opt_in_copysign_f64` and `opt_in_copysign_f64_broadcast`. Verify their
SHA-256 digests against those listed in the #103 qualification script.
Replace only the candidate digest in the #96 verifier, from
`5537162e8cfe8edd6e2a30474e7d3c001a76c5731c5ff94c1930bb27adb544a5` to
`caca1bfb706beb5eafc73dc2832c5731fe204a8801aef340c2b42d0b29691e34`.
On each new ELF run this staged verifier with `--repository "$PWD"`,
`--parent "$EXACT_90_ELF"`, `--candidate "$P/build-$i/heptatorch"` and
distinct `--output` paths. Both reports must match byte-for-byte.
