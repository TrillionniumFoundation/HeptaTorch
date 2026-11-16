# Final public-gradient integration: independent read-only review

No blocker was found in the reviewed Make/CI/source/receipt scope. No root
worktree source was changed and no native suite or real Torch case was rerun.
The original 517-check static review and the later CLI correction remain
separate, unchanged evidence.

## Actual command checks

Four complete oracle commands were extracted from the final parsed native YAML.
Each command retained its original script and arguments, and was executed in
an isolated workspace with the same relative input/output paths plus the
explicit `--prepare-only` flag. The native files were read from the real cold
builds and matched their pinned hashes; no target process was started.

| Runtime | Entry | Actual local result |
| --- | --- | --- |
| 355 functions, be18ab87 image | Unchanged live32, SHA 55467116 | Exit 0, NOT_RUN, 32 planned cases, zero executed |
| 355 functions, be18ab87 image | Unchanged frozen35, SHA 149b41c4 | Exit 0, NOT_RUN, 35 planned cases, frozen AST/custody PASS |
| 428 functions, 67734858 image | Unchanged live32, SHA 55467116 | Exit 0, NOT_RUN, 32 planned cases, zero executed |
| 428 functions, 67734858 image | Unchanged frozen35, SHA 149b41c4 | Exit 0, NOT_RUN, 35 planned cases, frozen AST/custody PASS |

The original bad 355 command was also executed unchanged apart from
`--prepare-only`: its real parser rejected the missing required source hash
with exit 2. Its unsupported support/lock arguments and the original failure
stderr remain preserved. This does not retroactively make the old static
review a CLI execution test. The root's original correction receipt and two
prepared reports are copied byte-for-byte into the reviewed-source evidence.

The final live32 commands provide both source and native hashes; frozen35's
original CLI checks the native image, with its core identity also established
by the paired qualification and live32 entry. Neither entry or its numerical
assertions/tolerances was modified. Exact original shell blocks and argv are
included in `cli-review/report.json`.

## Make and workflow

Make SHA256:
`852dc35c56acb093d5fa7116a5526f90568ae5b9c090b44f33ff4619ae02a34a`.
Native workflow SHA256:
`e4e8c9d55633c0609f5facdeb967d1661d3cb7b75aa014e1947242e7c062857a`.

Removing the six new steps and reversing the intentional timeout change
(20 to 40 minutes) recovers the entire previous workflow byte-for-byte.
The 63 prior YAML-declared steps retain their exact bodies and order; 69 are
now declared. The previously hosted count of 68 also included runner setup/post
steps and is not relabeled as a YAML-declared count. Required envelope paths
increase from 24 to 30, with the original collector unchanged.

The 355 build depends on the existing owned-Optimizer receipt. The combined
build depends on both the 355 output and the 378 Adam/distribution receipt.
The native qualification recipes place final sources/images in the same
`build/gradient-objects` and `build/gradient-adam-distributions` paths consumed
by the four oracle calls. The 428 aggregate's default result path is
`OUTPUT/producer/native`, which matches Make's final copy operation.

## Aggregate source and existing execution receipt

The installed aggregate source is exactly
`42295619cdd761b9bc16837aef612285c8defc6a788df06e8d4f4e3cc125baa9`.
Its actual reviewed report is exactly
`ce41c9ac7492c802cbd551e595c87a6fe72f43e1b3ac3945def7569ed29ef6be`
(10,414,574 original bytes, preserved losslessly as gzip).

The independent reader verified all 425 checked inputs (106,271,505 bytes),
lengths, hashes and modes; the original 17 producer child reports and streams;
the seven outer execution receipts; the three native gates; four unchanged
independent runners; and the strict eight-case stride comparison. Embedded
JSON content matches original raw files, whose byte hashes remain preserved.
The Optimizer24 mapping stays 23 qualifications within exposed scope plus one
retained-nonleaf partial case. The distribution183 atomic and explicit contracts
retain their distinct open categories. Repeated suites are not added as new API
coverage, and all full-project completion flags remain false.

That reviewed aggregate explicitly reused the previous completed 17-step
producer execution. Its default cold branch was reviewed from source and the
actual `--help` parser, but was not rerun by this reviewer. The parent is
separately exercising that default path. The receipt must not be described as
proof that this review executed it.

The first independent aggregate reader omitted the existing `image_sha256`
field from its explicit child-image aliases and stopped on that schema field.
Its unchanged failure and reader snapshot are retained under
`qualifier-review/history`. Adding that precise known field made all 577
read-only checks pass; the target qualifier and reports were never modified.
The separate command/Make/workflow review contains 112 checks. These numbers
are verification assertions, not semantic test cases or PyTorch API coverage.

The checkout binding records that root moved from the initially read reference
merge to a later reference-only base while these reviewed Make/CI/qualifier
bytes stayed identical. No new hosted live32/frozen35 result is claimed here.
