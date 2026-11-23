# HeptaTorch masked-select, logic and nonzero coexistence

This is a separate opt-in numerical qualification on exact main
`e9444902df9878060ab389a70ee957c1b7f4a887`. It adopts the byte-exact
operator bodies from #67 `270fde9f8b597c1444fd711026121fef4845b36d`,
#68 `dc4c9fd5efb9526241ea21ac83a4fa5b9196570f` and #69
`2b5b2fcd8b7937d5441317da1becf9cbef360a1b`. The current coexistence
projection is not a default route or a rollout; it does not cover
unreviewed operators, other dtypes/layouts or previous full corpora.

One dispatcher adds candidate-local 130, 131/133/134 and 135 to the
exact original entry. It preserves the original live response cursor
and original fallback for all other opcodes. All 39 operator bodies
and 14 other units remain exact. The projector verifies exact parent
aggregate/unit identity, split coverage, typed/effect checks, native
identifier reservations and byte inverse, without running any compiler.

Two independent cold builds produced matching ELF
`31a267d82b274ae81a956a03f8da1c9eb5a28bbcc903f78e56744333d1a88d4d`.
The 36 preregistered valid ordinary chained cases produced 129 exact
outputs and 925 result values on this combined image. Three existing
individually pinned images separately produced all 129 exact reference
outputs with materialized intermediate masks. The four sessions sent
2,268 ordinary wire frames, including 948 on the combined image.
Independent stdlib readback independently parsed the framed requests,
responses and owner snapshots and checked all bit patterns, shape,
strides, storage extent, offsets, Version 0, disjoint live output
identities, and preservation of earlier live inputs/outputs
(558 combined checks). Twenty-one genuine rank-2 Bool nonzero
cases on each side showed distinct C-order logical coordinates and
column-strided physical storage. Numeric nonzero inputs were rank one
only. Signed zero, subnormal, infinity and NaN payloads are handled at
the raw-bit level. Protected, fault, stress and unsupported cases
were not executed or qualified.

Reproduction: regenerate the frozen v2 fixture with `oracle_v2.py`.
Use `project_coexistence.py MAIN MASKED LOGICAL NONZERO BUILD_PY
COMPILER_AUTHORING OUTPUT` with the four independently reproduced source
directories. Inputs and checker are SHA-bound by the script. Then use
the existing pinned native build procedure and compare the new image
with `pinned-manifest.json`. `locked_v2.py HELPER LOCK INSTALL FIXTURE
OUTPUT` obtains the official frozen-wheel evidence.
`native_ordinary.py HELPER PROTOCOL FIXTURE RECEIPT COMBINED MASKED
LOGICAL NONZERO OUTPUT` records the four ordinary sessions.
`readback_ordinary.py OUTPUT FIXTURE RECEIPT` verifies raw and owner
evidence without Torch or native runtime. The receipts describe the
actual local paths observed, not a hosted qualification or complete
default aggregate.
