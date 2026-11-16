# Independent admission of the original int64 and int8 hosted evidence

**PASS_SCOPED_ORIGINAL_HOSTED_EVIDENCE: 2,044 checks passed, zero failed.**

This is a read-only review of original GitHub artifacts and source. It does not
run a compiler, a native executable or PyTorch, change a Git reference, or merge
a pull request. Its exact reader is `review_integer_artifacts.py` (SHA-256
`0adc6f30ab5fe8bc980d527519d93e814e8537bbfe6c4fe01a36634183af5a6f`).
The executed result is `review-01/report.json` (553,626 bytes, SHA-256
`d45f1270333287ee4749c89c7421a885a5638647e5a94b793e2cd55496765889`).

## What actually ran

| Hosted execution | Exact source head | Actual job | Each reference |
| --- | --- | --- | --- |
| PR29 int64, run 43851296835 | `0fd81b279c97dd969071c9efc51c8202fc16e45b` | 133244226312 | 33 cases; 351,174 values; zero differences |
| PR30 repaired int8, PR run 43878602383 | `214f6c232ae8b471ac9537bac2aa21fa0df3be1b` | 133338132847 | 28 cases; 1,093,216 values; zero differences |
| Same int8 head, push run 43878592238 | Same exact head | 133338098716 | 28 cases; 1,093,216 values; zero differences |

Each official job has twelve successful actual steps and attempt 1. Both
references are independently computed exact integer comparisons: CPython and
the locked real PyTorch 2.15.0+cpu wheel. The embedded original provenance has
all eleven named checks true, including the exact wheel hash, Python 3.12,
Linux x86-64, CPU backend and wheel-build commit. The release inventory commit
and wheel-build commit remain explicitly distinct under the existing source
equivalence record.

All original reports contain the complete fixed case plan. The reader checks
ordered names, cardinality, uniqueness, modes, lengths, every row's success,
zero mismatch/error fields, exact totals and every input, expected payload and
full response hash. It independently reconstructs these hashes from the fixed
integer corpora without invoking the producer or reference runtimes. The
CPython and PyTorch per-case observations agree exactly. The int8 PR and push
CPython reports are byte-identical, as are their PyTorch reports.

Int64 covers six operations (add, subtract, multiply, AND, OR and XOR), six
empty requests, 529 pairwise edge combinations per operation, seeded 10,000
and 48,000-pair batches per operation, and nine exact protocol refusals. Int8
covers the same six operations, all 65,536 byte pairs per operation, seeded
100,000-pair batches, a separate capacity-boundary batch, empty requests and
nine exact refusals. Overflow comparisons preserve the integer bit pattern.

## Exact source and execution identities

| Item | int64 | int8 |
| --- | --- | --- |
| HIA source SHA-256 | `251705542f4e117e3c9cdaec36cead8ab38252b24ebfef1104d765c546fd0716` | `7ca1765e2cede95ec19b64742f69816581733619b5b1b844f57aa72b743ed8d3` |
| Reported, asserted ELF SHA-256 | `fbb6cf1648385be200c94fc14ed73d1539cba3b0239aeea4f7866058c8395ebc` | `eba8c05bfe514c38ffa1a5e1ccbc9aacd513531f74becd18f4e3dca73db0a0b1` |
| Oracle source SHA-256 | `ed7ad57e5cf8c95198661a9aabcbf848b644dd7a63c14d0bf627bfc70bdd7761` | `0a75c065d9b520c4d3a3949dec09650e972725af09295176101562af92b8c6ec` |

Both jobs use the unchanged native compiler SHA-256
`87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54`,
rebuilt from Hepta commit `f33d8da316de8fdf27b504371f1a5310997ff548`.
The original workflow executes three fixed-point compiler generations, asserts
the final compiler identity, and the oracle compiles each application twice
with byte-identical results. Application execution checks return status,
stderr and exact response bytes. Source and compiler identities remain checked
at completion. Int64 also repeats the final native compiler self-reproduction.

The reader binds these executed assertions to the immutable workflow and
oracle source, original official job log, source/checkout commits and report
identities. The original ZIPs do **not** include ELF/compiler bytes or raw
per-case input/output streams. Therefore the review does not claim a direct
independent hash of unavailable native files. The original complete pip-install
file is also absent; its embedded provenance fields and the eleven executed
checks are verified, without claiming a locally downloaded wheel.

## Checkout and integration boundary

PR29 ran at merge `d993a99ee5cc83e8f186b96958b9df165f838296`, tree
`57b754d41960e16f483046bab36f694a8c0491d9`, with parents
`dcb903ccbb679497f4c7fa3c413dc0f6d19340ef` and its exact int64 head.

The repaired PR30 run used merge `07c3071d21108f659874c9dd2debe4e0a1038e83`,
tree `1c7cb0087ea020b609bdc22c67b29938b11304c7`, with parents
`7e7e37be38457d88f214d45f261924983dc96b56` and the repaired int8 head.
Its push job ran on head `214f6c...`, tree
`7aa4475461787e00d382e5d0c179c1eb964a18eb`.

These are the observed checkouts. Neither is a qualification of the subsequent
retained446 main tree, or of a new combined int64/int8 tree. Such integration
requires its own actual CI execution.

## Original int8 failure and exact repair

The original int8 run 43865611996, job 133293258942, remains failed at the Torch
comparison. Its provenance is `NOT_RUN`, reason `No module named 'torch'`.
The original log records installation into user site-packages followed by an
isolated `python3 -I -S -B` invocation that cannot import that installation.
The CPython result passed and remains byte-identical to the repaired result.

The repaired workflow changes only the Torch command to `python3 -B`.
Replacing that one exact command restores the entire previous workflow byte
for byte. Compiler and CPython isolation, application source, oracle source,
numerical assertions, image pins and all provenance gates are unchanged.
The repair commit's other nine additions preserve the original failure ZIP,
reports, metadata, log, previous workflow and explanation. Their Git blob
identities are independently verified. No old failure is reclassified as a
success.

The review preparation itself preserves two transport/reconstruction issues:
the first aggregate raw-API write exceeded the execution argument limit, so
the unchanged original UTF-8 strings were written in bounded chunks; a patch
reconstruction initially appended a newline despite Git's explicit
no-final-newline marker, and was corrected to obey that marker before its Git
blob was accepted. These did not run any producer or numerical test and did
not alter any source or official artifact. The final offline reader's first
execution passed.

## Original ZIP identities

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| int64 PR 14218775358 | 10,383 | `c7683a60421afc40a4df4d97a80fc1935d96b2676f2fa564e0504648d9f2e0ae` |
| int8 PR 14230882971 | 9,257 | `c61ca3acecf56ab00b23cbc5b32ab27e4974f4ccf59c846036d5d13dc891a15e` |
| int8 push 14231167849 | 9,255 | `55e30b07e3e5e536b1f90cc35ef97e9927f7bf7ee63e61a562aa9f9fba43f39b` |
| int8 original failed 14225745762 | 4,645 | `d56a7c9610ef929a2fe1f674208786c8d1050ba117e925ccdf3574a8d0ffef80` |

All ZIPs, original API UTF-8 responses, decoded official logs, immutable source
files and report bytes are included. Member path safety, CRC, exact member set,
original digest and expanded-copy equality have been checked.

## Reproduce the offline review

```sh
python3 -I -S -B review_integer_artifacts.py \
  --bundle admission-01 \
  --output /absolute/path/to/new-integer-review
```

The output must be a new directory. This is an offline read-only check except
for its own output directory. These bounded standalone packet operations do
not close general Tensor dtype dispatch, broadcasting, storage, autograd, or
full PyTorch schemas. The whole-project status remains HOLD, full-schema
parity 0 of 2,590, and `full_rewrite_complete=false`.
