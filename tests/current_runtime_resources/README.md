# Original resource7 on the current HIA runtime

This entry preserves seven original allocation and ownership recipes on the
canonical **773-function, 15-unit** HIA runtime. The complete qualification
requires an actual matching cold build and new native execution. Current combined
execution status is recorded in the [current audit](../../docs/current-audit.md);
historical passes and source preparation do not qualify a later image.

## Entry point

Keep `adapter.py`, `qualify.py`, and `sources.json` together. Use the same pinned
Python as the parent qualification and a new output directory:

```sh
python3 -I -S -B tests/current_runtime_resources/qualify.py \
  --repository /absolute/path/to/HeptaTorch \
  --build-directory /absolute/path/to/HeptaTorch/build/current-runtime \
  --output /absolute/path/to/fresh/resource7-output
```

`--prepare-only` checks source custody, reference restoration, current cold-build
admission and exact source inverses without starting a native test session. Its
status is `PREPARED_NOT_RUN`. The parent passes `--expected-build-sha256` from
its actual cold-build receipt; a standalone invocation can omit it and have the
entry hash `build.json` before admission. Every worker receives that explicit
receipt SHA, and the original R4/backward driver receives the same identity.

`--suite r4`, `--suite backward`, and `--suite r3` are diagnostic subsets with
status `PASS_SELECTED_RESOURCES`. Only all seven recipes can report
`PASS_CURRENT_RESOURCE7`; the report schema is
`heptatorch.current773.resource7.v1`. A group PASS does not close an upstream
schema or the whole project.

## Fixed current identity

| Product | Identity |
| --- | --- |
| HIA aggregate | `4722c53b248780534a846a472784a565c4454ab738db75d655a7801b6623f07f` |
| Main ELF | `5cb190cfc4b920bb03e78803e79296e2e046ec90eb6e07b2d5ba6eebfb534edd` |
| Products | 773 HIA functions, 15 units, 672-byte descriptors |
| Owner formats | `HTSINT02` / `HTOSIN02` |
| Authoritative source | [Current manifest](../../tools/current_runtime/source/manifest.json) |

Both coordinator and worker call
`tools/current_runtime/qualification.py::admit_build`. It checks the actual
manifest, complete partitions, every object and tool identity, exact link
request, stack budget, compiler/linker exits and both EOFs. Independent cold
builds have different paths and timings, so their receipt SHA is an explicit
input rather than a copied developer-local constant. Source, image and the
773/15 denominator remain fixed throughout one qualification.

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

`sources.json` binds 186 repository inputs. Original package manifests, all
original source members, their byte counts and modes remain required. The R4
and backward package validators retain their original whole-file inverse checks;
R3 retains the exact outer manifest and its relation to the inner source manifest.

Recipe bodies, allocation geometry, reference comparisons, owner equations,
RNG comparisons, rollback checks, limit restoration and recovery commands remain
unchanged. The R3 OOM recipe has one explicit source-admission substitution:
its original ten-body hash loop invokes the reviewed allocation-path proof.
Reversing that substitution restores the complete original recipe.

## Actual source and allocation proofs

[`fusion_resource_transition.py`](../../tools/current_runtime/fusion_resource_transition.py)
composes the complete 773 → 771 → 767 → 765 → 761 → 758 → 742 → 740 → Half736 → storage736 → cat700 source inverse.
Every original byte remains constrained by the corresponding full-source hash.
The 773 layer checks the two changed `binary` and `half_mul_cast` bodies,
two appended functions, all 769 unchanged parent bodies and the whole-source
forward round trip. Its actual mixed public F64 route requires dtype4; the
original dtype0/1/2/private3 resource recipes still delegate to the same retained
predecessor body. This source proof supplies no new numerical result.

The 771 layer checks the single changed `binary` body, four appended division
helpers and all 766 unchanged parent bodies. The actual binary body retains the
complete dtype5/6 guard and adds only an operation4 route within that guarded
branch. Exact argument order and the complete preceding add/sub, multiplication
and original-domain delegation remain checked. The original dtype0/1/2/3 resource
domain cannot reach this new division branch; division outcomes require separate
execution evidence.

The 767 layer checks both changed bodies, both appended helpers, all 763
unchanged parent bodies, and an exact whole-source round trip. Its I64/F64
unary wrapper retains the complete previous Half wrapper. Only dtype0/1/3
unary inputs still delegate through both wrappers to the oldest unary body;
new I64/F64 outcomes require their own successor evidence. The independent
resource reader rejects opcode3 in the actual request frames of these seven
unchanged recipes, so their resource evidence does not qualify the new branch.
The 742 BF16 layer checks both modified bodies, both new helpers, all 738
unchanged original bodies and the complete round trip to 740. These source
relations do not execute a historical application or transfer its PASS.

The preserved 740 numeric-predicate extension adds four functions and changes only
two `entry` instructions. Its exact guard admits only opcodes 120–124; all
original 108–119 routes delegate to the unchanged physical `to_command` with
all six arguments in their original order. The R4 obligation binds actual
`entry`, `predicate_command`, and `to_command` hashes, including the complete
guard and delegate blocks. A reconstructed predecessor is never substituted
for an actual executing body.

R4 retains all 37 original obligations. Five bodies have explicit conditional
source proofs for the original dtype0/1/2/private3 owner domain:
`command_argument`, `dtype_audit`, `to_command`, `typed_command`, and
`typed_fill`. The full original owner still rejects public dtype4/5/6 in these
recipes. All remaining obligations retain their original body pins. The
original dispatcher chain, finite dtype guards and six-argument delegation are
checked in full. Actual request frames are inspected after execution as well.

The backward recipe retains all 24 original allocation-body checks. R3 retains
eight of ten bodies directly and the other two through exact declaration-name
inverses in `scalar_previous_typed_descriptor` and
`scalar_previous_incoming_mark_preserve_targets`. Its actual OOM fixture has
only dtype0 and op0/op44 live descriptors, excluding the additional scalar
branches. Those facts are recorded from the actual owner before allocation.
The original 112-byte workspace bound, 16-byte small extents, 60–68 KiB free
interval and 524300/524304-byte preserve allocation remain unchanged.

## References, resource recovery and physical readback

The coordinator restores the exact original reference/probe bytes and modes
from `validation/current-runtime76-reference/complete.tar.xz`, retaining its
manifest and historical [archive location](../../validation/view-reshape-r4-native76/archive-location.json).
R4/backward use the original `positive_one` wrapped-F64 row; R3 uses original
input4 finite gradients. All eleven provenance checks and the original complete
RNG/reference comparisons remain in force. Frozen-reference replay does not
claim a new Torch execution.

Each recipe starts a real current HIA ELF. Python transports the protocol and
observes evidence; the original OOM controls set real child `RLIMIT_AS`, restore
it in `finally`, and retry the same native command. All three limit restorations
must be recorded. The required original seven-case accounting is 378 frames
and 1,060 checks, with complete owner/RNG evidence.

The independent coordinator reopens the native ledgers and complete request/
response frames. All seven children and three supervised workers require exit0,
complete stdout/stderr EOF, no extra output or surviving descendants, and
unchanged bound inputs. Actual execution remains separately recorded as true,
false, or unknown even if qualification fails. Every report retains
`full_pytorch_rewrite_complete=false` and `schema_closure=false`.
