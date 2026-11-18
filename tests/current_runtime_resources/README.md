# Original resource7 on the current HIA runtime

This entry runs seven original allocation and ownership controls against the
current HIA runtime. The runtime source and all original resource packages remain
unchanged. Historical passes do not qualify the current execution.

## Entry point

Keep `adapter.py`, `qualify.py`, and `sources.json` together in
`tests/current_runtime_resources/`. Invoke the entry with the same Python used
by the parent qualification:

```sh
python3 -I -S -B tests/current_runtime_resources/qualify.py \
  --repository /absolute/path/to/HeptaTorch \
  --build-directory /absolute/path/to/HeptaTorch/build/current-runtime \
  --output /absolute/path/to/fresh/resource7-output
```

`--prepare-only` checks source custody, reference restoration, current cold-build
admission, and derived source inverses without starting a native test session.
The parent coordinator passes `--expected-build-sha256` with the actual receipt
SHA from its own fresh cold build. A standalone invocation may omit it; the
entry then hashes that build's `build.json` before admission. The worker always
receives the explicit `--expected-current-build-sha256`, and each original
R4/backward driver receives the same SHA through its existing receipt argument.
`--suite r4`, `--suite backward`, and `--suite r3` select a diagnostic subgroup;
their successful status is `PASS_SELECTED_RESOURCES`. Only the default complete
seven-case run can report `PASS_CURRENT_RESOURCE7`.

The output directory must be new. Earlier failed attempts are retained.

## Fixed current identity

| Product | Identity |
| --- | --- |
| HIA aggregate | `0c1c0dd3acf0285a887a2766c1308dfd992acf3fe51849dcd98749bb64a00f3b` |
| Native ELF | `6ff13ccab1b30db389307900958937496905e0ed842893081d878788aa7c754c` |
| Products | 700 HIA functions, 14 units, 672-byte descriptors |
| Owner formats | `HTSINT02` / `HTOSIN02` |

Both coordinator and worker call the unchanged
`tools/current_runtime/qualification.py::admit_build`. This checks the actual
current manifest, partitioned source, all objects and native tool identities,
link request, stack budget, build process exits, and complete tool EOFs.
No old 627/658-function build receipt or success stamp is manufactured.
The actual cold-build receipt is not fixed to a developer's local build: paths,
process timings, and other real execution records vary between independent cold
builds. Source, image, native tools, every object, and the 700/14 denominator stay
fixed. The exact selected receipt remains a bound input throughout execution.

## Original seven recipes

| Group | Original recipe |
| --- | --- |
| R4 resource3 | `entry_guards_and_relative_tokens` |
| R4 resource3 | `saved_scalar_and_temporary_lifetime` |
| R4 resource3 | `result_allocation_oom_recovery` |
| Backward private cache1 | `backward_private_preserve_allocation_oom_recovery` |
| R3 preserve3 | `tail_release_reuse` |
| R3 preserve3 | `preserve_capacity_oom_recovery` |
| R3 preserve3 | `tail_then_saved_error_prefix` |

The R4 portable package and backward package run their original pure package
validators, including original-to-portable whole-file inverse checks. The R3
outer package member set includes its original `package-manifest.json`; its
inner source manifest and all 20 original source members retain their exact
bytes, sizes, and modes. `sources.json` separately binds all 177 repository
inputs used by this migration, including the complete original package assets.

The original three R4 recipe bodies, backward `run_case` and geometry helpers,
all Case transports, and the two non-OOM R3 recipe bodies are byte-identical.
The R3 OOM recipe has one explicit source-admission edit: its loop comparing
ten historical HIA body hashes calls a reviewed allocation-path proof. Reversing
that one edit reconstructs the complete original recipe exactly. Every resource
budget value, allocation geometry check, reference comparison, owner equation,
RNG comparison, rollback check, limit restoration, and recovery command remains
unchanged.

## Allocation source proofs

R4 has 37 original HIA obligations. Thirty-six current bodies match their old
hashes exactly. The old `to_command` is retained byte-for-byte as
`sd_previous_to_command`, except for its declaration name. Reversing only that
name yields the original body hash. The current dispatcher prefix is separately
pinned: it reads the opcode, intercepts 115–119, and otherwise delegates all six
arguments unchanged. Each actual recorded request is checked after execution to
confirm that the original recipes did not call any intercepted opcode.

The backward private-cache recipe retains all 24 original HIA allocation-body
hash checks. Every one matches the current source directly.

R3 retains eight of its ten old allocation bodies exactly. For the other two,
the complete original bodies remain in `scalar_previous_typed_descriptor` and
`scalar_previous_incoming_mark_preserve_targets`. Reversing only their
declaration names yields the exact original pins from `oom-budget.json`.

The current `typed_descriptor` wrapper adds a rejection only when dtype is 3 and
rank or requires-grad is nonzero. The actual OOM fixture uses dtype 0. The current
incoming-preserve wrapper first executes the complete old marker; its extra
write requires an op3 multiply with a Saved dtype3 operand. The fixture contains
only F32 op0/op44 live descriptors. The adapter checks and records those actual
descriptor fields before the native OOM command. The added scan reads descriptors
without allocation on this path. The original workspace upper bound 112 bytes,
16-byte small extents, 60–68 KiB free-space interval, and preserve allocation
524300/524304 bytes are unchanged.

No old function bytes are substituted for current functions. Reports include
the actual current bodies, preserved aliases, original hash identities, actual
live descriptor facts, and exact source transformations.

## References and native evidence

The coordinator restores both original observations and probes by exact hash
from `validation/current-runtime76-reference/complete.tar.xz`. It binds the
compact manifest and the historical `archive-location.json` by exact SHA,
preserves each original member's bytes and mode, and records the compact
member name and its original archive member name. The complete historical
native76 archive remains recoverable through that location index; it is not
required for these four unchanged reference/probe inputs. R4 and backward
use the original `positive_one` row from wrapped-f64-reference32. R3 uses the
original input4 finite gradients. Their original eleven oracle provenance
checks, full recorded RNG states, and byte comparisons execute unchanged.
Torch is not run by this migration. The frozen reference's AVX512 environment
label remains a record of the machine that generated that observation.

Each resource recipe starts its own real current ELF. Native behavior is all HIA;
Python transports binary commands, reads its child's committed owner, applies
the original temporary `RLIMIT_AS`, and compares evidence. The original failure
paths still exercise real allocation failures, restore the original limit in a
`finally` block, and retry the identical native command successfully.

The original transport requires the native process to exit with code zero and
both stdout and stderr to reach EOF with no extra output. The coordinator checks
all seven native transport ledgers and exact request/response frame files. Each
response frame is checked against its original live-input ledger row, and all
three OOM cases must explicitly record that their original address-space limit
was restored. Each
of the three workers is additionally supervised by the existing
`tests/current_runtime_process.py`, including exact PID/start-time ownership,
pidfd-based cleanup, full stdout/stderr capture, and no surviving descendants.
The coordinator rechecks every bound input after all selected groups finish.
Execution facts are kept separate from qualification status: an observed native
execution remains `true` even if the worker, a later comparison, or input custody
fails. An attempted worker with no usable original report is `null` (unknown).
A rejection before any worker starts is `false`.

These seven resource contracts do not close any complete PyTorch schema. Every
report retains `full_pytorch_rewrite_complete=false` and `schema_closure=false`.

## Preserved preparation failure

The first R3 preparation attempt, `prepare-r3-01`, failed before any native
session because the new adapter's directory-set check counted the inner source
manifest but omitted the original outer `package-manifest.json`. The correction
uses the original outer file set and checks its relation to the inner manifest.
No fixture, runtime behavior, comparison, or resource budget changed. The failed
report and its exact adapter source snapshot remain in scratch for audit.

The first complete local seven-case run used a fixed local build receipt SHA.
Although all seven native cases passed on that actual image, the entry would
have rejected a newly built CI receipt with different paths/timings. The reviewed
entry now takes the actual receipt SHA explicitly. The previous code and actual
PASS are retained as local-build history; they do not qualify the corrected entry.
The corrected entry is verified with a separate complete cold build, an explicit
wrong-receipt rejection before any native session, and a fresh seven-case run.

## Final isolated execution

`native-all7-04-final` runs the final scripts against a separately cold-built
current image in `fresh-build-03`. The new actual build receipt SHA is
`4d38adfd93f625d110f3b0585a60829f378e922d4ba9b28c5fb8b3bfaa1163ac`.
It reports `PASS_CURRENT_RESOURCE7`: 7 recipes, 378 native frames, 1060 original
checks, 391 committed 84Q snapshots, 62,737,664 owner rollback bytes, and
1,835,328 RNG bytes. All seven native children and three supervised workers
completed successfully. These counts are computed from this actual run.

`wrong-old-receipt-03-final` supplies the earlier local receipt SHA against that
new build. It correctly fails at receipt admission with no worker/native start.
`handoff.json` records the final script, report, and negative-control identities.
