# Public gradients and Adam/distribution hosted qualification

Implementation `af33d1065a3b097990e9536df5b22b1a3fb315cf` was merged by PR #28 at
`7e7e37be38457d88f214d45f261924983dc96b56`. The tested and merged tree is
`88ed6bd6a41ca0e58e653b3d76718090b3bbba64`; the merge parents are the previously
verified main `dcb903c…` and the exact tested implementation `af33d10…`.

All eight checks on that implementation succeeded. Both Native jobs executed
76 successful steps. The quick 355 and 428 profiles each executed 13 successful
steps. The main PR run is 43862154942, and the two-profile quick run is
43862154875. The original JSON metadata, complete member bytes and independent
review bind those statements to the actual source and checkout identities.

## Executed scope

Each of the 355-function and 428-function programs ran the unchanged live32 and
frozen35 oracles in both workflows. Each frozen35 result records 6,344 frames,
2,590,274 compared values, all eleven reference provenance checks, every original
EOF and final parameter Version gate, and zero parameter Version mismatches.
The same 283 legacy explicit-state Version differences remain explicit.

The native 355 qualification has sixteen steps; the 428 cold producer has
seventeen and retains the separate ownership, handoff and mixed-layout reviews.
Each quick program ran three actual native controls (thirteen frames), four
positive copied observations and all eighteen original corruption rejections.
The independent offline reader made 59,251 checks with no failures and executed
no new native program or Torch. Its original 5,829,457-byte report is losslessly
compressed in the independent review handoff, with SHA256 recorded in
`qualification.json`.

## Original evidence

`member-custody.json` identifies every byte of all 723 original ZIP members:
58 root reports plus their manifest, 294 main journal members, and 185 members
from each quick profile. Their 206,506,761 expanded bytes are preserved in a
single lossless archive. Restore it from the tracked Git parts with:

```sh
python3 -I -S -B tools/restore_frozen_archive.py \
  --manifest validation/hosted-gradient-adam-distributions-af33/archive-parts.json
```

This is a repackaging of exact original members, not a reconstruction of the
GitHub ZIP containers. The original four downloaded ZIP containers remain
preserved separately; `member-custody.json` records their identities and retained
paths. JSON was never parsed and rewritten for this archive. ZIP export modes
are recorded as export metadata and are not asserted to equal the source
permissions before GitHub artifact upload.

`independent-review-handoff.tar.gz` and its frozen manifest contain the exact
reviewer, fixed published sources, original fifty-report obligations, unchanged
final report, preparation-error explanation and input mapping. Official main
and quick run/job/artifact/merge metadata is under `main/` and `quick/`; the
post-merge Git identities and pre-merge check states are separate original JSON
files. The original report paths and process-specific integer values are intact.

## Boundaries and earlier failures

The initial 30ea frozen35 result remains three PASS and thirty-two FAIL; its
unexecuted EOF/Version assertions are never retrospectively accepted. The first
1d45 copied-snapshot negative-control failures remain failures. Their original
artifacts are in the existing private-node-accounting history directories.
The subsequent reader schema error remains a reader error; the final reader
checks all profile fields plus the additional original byte-count fields.

Public10 remains nine qualified and one partial, incoming25 remains twenty-two
qualified and three unrepresented, and the fixed distribution corpus remains
165 qualified and eighteen unrepresented under the explicit-error interface.
The full target remains PyTorch v2.15.0, 2,590 schemas and seventeen whole-system
gates, with full-schema parity zero and overall HOLD. These finite results do
not claim full PyTorch compatibility or qualify a later native image.
