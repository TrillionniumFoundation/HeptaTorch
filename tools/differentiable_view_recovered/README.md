# Differentiable-view recovery source (R1)

This directory installs the seven producer-owned Recovery R1 source files.
Their bytes match the original `source-manifest.json` (SHA-256
`8c75f5e3069e02f9e071064d5bc1a68d96636e7ef6e26ea1aee1b5cb1c703e0d`).
The original manifest is preserved byte-for-byte under
`provenance/original-source-manifest.json`. It describes the original 0600
transport files. This Git installation uses mode 0644; the explicit mapping is
in `provenance/install-custody.json` and the installed modes are recorded by
the new `package-manifest.json`. The original manifest is not relabelled as an
installed-mode manifest.

## Identity and current boundary

`differentiable_view.py` is the recovered original constructor source
(`934c039186eb794b2294cf3cc4713685c14a2b1071b13834779b39e16bba3021`).
The other R1 files have their own recovery orchestration identity; they are not
claimed to be the inaccessible Cloud 01b orchestration files. This installation
does not run a native build or the actual 14-case reference replay.

The parent is the exact frozen 09b runtime (`4f8f73f4...`) and executable
(`53fb43b8...`). It must include all twelve actual module source/object pairs.
The separate `tools/tensor_to_native_recovered/` recipe can produce that parent
from its bundled HIA sources using the pinned native compiler and linker.

## Build interface

From the repository root, after the parent has actually been rebuilt:

```sh
python3 -B tools/differentiable_view_recovered/build.py \
  --previous build/tensor-to-native-recovered \
  --repository . \
  --upstream /absolute/path/to/pinned-hepta \
  --compactor tools/native_dropout/compact_functions.py \
  --compiler /absolute/path/stack-module-compiler \
  --linker /absolute/path/stack-linker-session \
  --output build/differentiable-view-recovered
```

The upstream authoring files are individually pinned by the existing
`tools/owned_tensors/profile.json`; the compactor is pinned to
`36730b55b58c29b210d921a426c62eab50d9f579ad5d7a1e4199b38799218f37`.
The module compiler and session linker retain their exact b1c851a0 and
aae650e8 identities. No original compiler limit or numerical expectation is
changed by this delivery.

The producer emits `runtime.hepta`, `source-map.json`, `emitter-custody.json`,
the new module source/object pairs, its native image and a new `build.json`.
The producer first re-emits the parent constructor exactly, then applies its
declared new source change. Its build report keeps runtime qualification
separate from compilation. A successful native build alone does not establish
the 14 actual-reference cases or inherited semantic gates.

At installation time, no child native source hash, image hash or PASS profile
is asserted here. Any later emitted fragment/source map/profile will be bound
to the actual resulting native build as a separate custody record. Historical
Cloud failures and the original R1 prepared manifest remain unchanged.
Project status remains **HOLD**, 0 closed API schemas / 2,590, full=false.
