# Admitted int64 source and current-main integration

PR29 source `2e66ad094ca62d059a1a7ca5642a8ce27c40a949` is merged at
`7e157f7b1cfa4f77e004fa118a06892f255116b0`. The actual merge tree is
`51b2bf3b1d15e8662253668c0f6ee458885dd3de`, identical to the source-qualified
tree; its ordered parents are previous main `e0ea769c979236b0542af4b258890726cd01a99a`
and the reviewed source head. Original GitHub JSON and the independent root
replay receipt are in [root](root/).

Both dedicated int64 jobs executed the original 33-case, 351,174-value corpus
with the fixed actual PyTorch wheel and all eleven provenance checks. Each
passed thirteen actual steps. Both full Native jobs passed eighty actual
steps, and each new set of original artifacts passed 106,633 independent
offline checks. The integration owner separately reproduced the PR report
with exactly the same SHA-256. All three quick observer jobs passed thirteen
steps each and the independent quick admission passed 52,151 checks. All
eleven GitHub source checks completed successfully.

The [completed checkpoint](checkpoint/README.md) preserves 204 archive members,
including the frozen source readers, the original ordering-only reader failure,
its exact correction, source inverse proofs, original reports, complete API
metadata and the completed source-specific admission. Its archive SHA-256 is
`d392f964441d8bc20c6d9884a2bb4248522e4565d5ba7d547ca9dd6089809042`.
The complete eleven original artifact ZIPs (185,608,714 bytes) are externally
indexed by exact artifact IDs, sizes and SHA-256 values inside that checkpoint;
they are not duplicated in its small archive. The [independent Mac custody receipt](mac-custody/receipt-v2.json) now records
all eleven ZIPs copied and separately read back with identical full hashes
and sizes. The outer backup files deliberately use private mode 0600; the
Cloud originals use 0644. Original ZIP bytes and embedded member modes are
unchanged. The first receipt is preserved; its misleading field name that
included outer-mode equality is corrected in the separate v2 receipt. ZIP CRC
checks also passed. The single original transfer job completed successfully;
this backup does not constitute native execution.

This admission covers the bounded standalone native HIA lane kernel and the
unchanged existing regression profiles. General integer Tensor behavior and
complete upstream schemas remain separate work. Whole-project acceptance is
HOLD: zero complete schemas out of 2,590 and seventeen open or in-progress
whole-system gates.
