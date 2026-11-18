# Current HIA runtime: verification evidence, 2026-11-18

The final unified execution reported **`PASS_CURRENT_RUNTIME_FOCUSED`**. All ten groups were attempted on the same HIA source, ELF, and actual cold-build receipt. Its original report SHA is `aa168047fc9a0392707f3c1c08b8ab4cb88c2bc16e1732c4c37c2a3994cc14c0`. The report and every selected original subordinate report are preserved byte-for-byte in `qualification.tar.xz`.

The finite runtime qualification does **not** establish a complete PyTorch rewrite.
The full-project status remains `HOLD`, `schema_closure=false`, and
`full_pytorch_rewrite_complete=false`. CI is a separate observation from the
subsequent real pull-request checks; this package does not predeclare CI success.

## Exact runtime

| Identity | SHA256 or count |
| --- | --- |
| Current aggregate source | `0c1c0dd3acf0285a887a2766c1308dfd992acf3fe51849dcd98749bb64a00f3b` |
| Current native ELF | `6ff13ccab1b30db389307900958937496905e0ed842893081d878788aa7c754c` |
| Main actual cold-build receipt | `aae73ebfbc0d04401a02467f896b48e4cf886ca47f27835f25f24512a2ba57d9` |
| Independent actual cold-build receipt | `4d38adfd93f625d110f3b0585a60829f378e922d4ba9b28c5fb8b3bfaa1163ac` |
| Core HIA functions / complete native units | 700 / 14 |

Both cold builds used the fixed native compiler and linker, compiled all fourteen
objects afresh, and completed fourteen compiler processes plus one linker process.
The original receipts retain `objects_reused=false`. The current shared build
admission read back each complete product set, including actual stack analysis,
source partitions, unit objects, linker input, tool identities, and process EOFs.
All fourteen objects and the final ELF match across the two independent builds.
The actual receipts differ because real paths and execution records differ.

## Unified finite denominator

| Group | Required recipes | Observed result |
| --- | ---: | --- |
| bool27 | 27 | PASS |
| original76 | 76 | PASS |
| integral24 | 24 | PASS |
| sum_f32_17 | 17 | PASS |
| current35 | 35 | PASS |
| base270 | 270 | PASS |
| sum32 | 32 | PASS |
| live32 | 32 | PASS |
| logical147 | 147 | PASS |
| resource7 | 7 | PASS |
| **Total** | **667** | **PASS_CURRENT_RUNTIME_FOCUSED** |

These are named test recipes, not 667 distinct upstream schemas. Separate packet,
dtype, axis, copied-corruption, allocation, and refusal controls remain part of
their original suite contracts and are recorded separately in the reports.
Existing frozen Torch observations remain independent reference inputs. The
live-reference groups use the actually pinned Torch environment. No historical
native PASS is transferred to the current image.

## Included archives

`manifest.json` lists each archive and every selected uncompressed member with
its exact byte count, SHA256, mode, and origin. The archives use standard xz and
tar. All selected original reports are losslessly retained. Generated projections
are explicitly identified as projections rather than original execution files.

| Archive | Contents |
| --- | --- |
| `qualification.tar.xz` | Final unified 667 report, ten primary suite reports, supervised worker receipts/streams, and necessary original subordinate reports. Present only after the complete final result is observed. |
| `failures.tar.xz` | The immutable first660 failure, current35 stage evidence, the pure cross-Python diagnostic, five shared-build admission refusals, and the resource7 wrong-build-receipt refusal. |
| `build-and-review.tar.xz` | Two complete cold-build receipts, their object/ELF reproduction check, final source-retirement review, capability/unused-asset cleanup records, independent resource7 verification, and separate post-compaction observations. |

`execution.json` contains allowlisted public execution facts: actual source/build/
report identities, interpreter identities, exits, and complete stdout/stderr
EOFs. Private MCP service receipts are retained outside this public package and
bound only by their original size and SHA256. Job/account/session metadata is not
published. Command path roles are projected explicitly; functional arguments are
unchanged and the original staging receipt is hash-bound.

The existing fixture archives, executable, native objects, full owner-dump trees,
old source trees, and all intermediate successful runs are not copied here.
The final reports retain their input hashes, and the repository's current build
and qualification entry points provide the reproduction path.

## Failure evidence remains failed

The first unified run attempted all nine then-required groups (660 recipes).
Eight groups passed; current35's subsequent strict/copy-control admission failed.
Its coordinator used Python 3.14 while the live Torch worker used Python 3.12.
A pure configure diagnostic on the unchanged snapshot observed exactly seven
different AST-summary hashes between the interpreters. It executed neither
Torch nor a native test session. The original first-run report remains `FAIL`;
the later run uses Python 3.12 consistently and must pass the original strict
comparisons itself. No assertion or expected tensor result was relaxed.

The Bool/Integral/F32 migrations retain original dtype-control failures when
the current implementation accepts operations that the old controls expected
to refuse. Their original reports remain intact. The current behavior is checked
separately against actual pinned Torch observations, as recorded by the final
qualification. An original failure is not relabeled to manufacture a pass.

Five copied-build admission controls prove rejection of receipts with a timeout,
forced cleanup, execution error, cleanup error, or missing stack-inspector input.
The original valid build receipt and source are unchanged. A separate resource7
control passes the old build-receipt SHA against the independent fresh build and
is rejected before any worker or native session starts.

## Resource and cleanup evidence

The independent resource7 run used the second complete cold build. It actually
passed all seven original recipes with 378 native frames, 1060 comparisons,
391 committed 84Q snapshots, 62,737,664 owner rollback bytes, and 1,835,328 RNG
bytes. Seven native children and three supervised workers completed with exit0
and both stream EOFs. All three real OOM paths restored their original address-
space limit and completed the original identical-command recovery.

The current wrapper preserves every original numeric/resource comparison. Its
source-admission changes have exact whole-file inverses. Two changed R3 wrappers
are justified by exact retained original aliases and actual F32/op0-or44 node
facts; original allocation-budget numbers are unchanged. Execution facts remain
independent of qualification status: observed native execution stays true after
a later failure, missing evidence is unknown, and a pre-start rejection is false.

The final HIA reachability review records seven retired unreachable bodies
(15,647 bytes) and no remaining unreachable core functions. All 700 current core
functions are reachable from the actual owner entry roots. The metadata cleanup
removes obsolete process-era claims without claiming schema completion. The
unused current35 archive/helper and three later zero-consumer assets are retired
with exact prior hashes and source commits. Necessary original comparison
fixtures and their actual consumers remain retained.

Capability cleanup records distinguish the initial metadata retirement, the
immutable ten-group 667 snapshot, and the repository document at evidence
assembly. A later documentary status update is recorded separately from the
pre-execution snapshot. Native source and frozen reference identities are
unchanged by these metadata and unused-asset cleanups.

## Reference compaction after the unified run

The completed 667 snapshot used the earlier archive locations. Its exact report,
frozen input manifest, and source identities remain unchanged. The subsequent
storage migration retains all eight original76 reference/probe members in a
131,848-byte archive, with their original bytes and 0600 modes. The old
12,561,956-byte complete historical archive is recoverable through its committed
`archive-location.json`; it is not reintroduced into the current repository.

The necessary reader/fixture/coordinator transformations have complete source
inverses, retained in the review archive. The strengthened live32 owner/transport
reader also retains the original capture and comparison logic, with explicit
identity inverses. Its prior wrapper is preserved once because it is required
to reconstruct the original 667 snapshot; no prior source tree is duplicated.
Resource7's four consumed members were
also compared byte-for-byte and mode-for-mode to its prior actual restoration.
After applying the compact archive binding, a new execution of all seven
resource recipes passed on the same current source, ELF, and main build receipt:
378 native frames, 1060 comparisons, seven native children, three workers, and
complete zero-exit stdout/stderr EOFs. Its original qualification SHA is
`a45f05345dce0299365c36560bddcba0c4135d7450be4a099146eb1d46c5817d`.

The separate affected-reader regression for original76, Sum32, and live32 is
**`PASS_PARTIAL_POST_667_REGRESSION`**. These three groups total 140 recipes and are independent of
the earlier full 667 observation. Resource7's seven new passes are also reported
separately. No previous PASS is transferred to changed readers, and this storage
migration does not change any HIA function or frozen reference value.

Its newly observed partial qualification report SHA is `0c21f10e1836c9282e9d8dece81d55b14652b1368575c53368691fa9b35396b3`. The original three-group report, original subordinate reports, complete worker streams, frozen inputs, and executed partial orchestration source are retained in the review archive.

## Independent AsStrided16 observation and next CI denominator

The final source02 AsStrided reader independently passed all sixteen original
recipes on the same current HIA source, ELF, and main build receipt. Its original
report SHA is `8300788a5927e0862b0e3ff6561ce878f8509b2f7276933bbb913e077dabf003`. The original four
comparison readers and owner observer have complete source inverses; only the
build admission and two owner-tag literals are adapted. The original native
finish functions and syscalls remain unchanged. A read-only profiler records
their actual code objects, normal return opcodes, empty reads, and closed files.

This new execution covers 1920 native frames, 10,272 assertions, 21,892 exact
binary words, 1,860,608 RNG bytes, and 176 complete owner archives with 416
references. Sixteen native children and four workers exit with all 32 native and
8 worker stream EOFs. The installed pure qualification API independently reread
the entire execution and source custody; the original four oracle observations
retain all eleven provenance checks each. No new Torch execution was needed for
these frozen-reference recipes.

The updated repository now requires eleven groups and 683 recipes. The completed
unified 667 execution, later affected-reader regressions, and independent16 run
are separate observations. **A full unified683 execution is not claimed here.**
The exact publication head must run all eleven groups in its own CI. This package
does not predeclare that CI result or promote these finite cases to full PyTorch
schema parity.

