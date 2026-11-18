# Whole-PyTorch HIA rewrite: continuation plan

## Objective and authority

Complete the native HIA implementation of the pinned PyTorch project, preserve
its behavior and close every acceptance requirement before retiring the final
remaining legacy consumers. The target remains PyTorch **2.15.0**, commit
`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`, with **2,590 exact native declarations,
688 derivative declarations and 17 system gates**. These declarations are part
of the target; they do not replace frontend, backend, distributed, compiler or
ecosystem obligations.

Read the current source tree, open work, [capabilities](../compatibility/capabilities.json),
[full-acceptance ledger](../compatibility/full-acceptance.json) and original
source-bound receipts before continuing. New recorded evidence takes precedence
over old task summaries. Preserve active work and use exact-head leases when
publishing concurrently. Whole-project status remains **HOLD**, with zero full
schema or system-gate closures.

## Current default and immediate qualification

The canonical profile is `tensor-generator-native01`: 712 core functions,
9,536 blocks, 34,167 instructions, 15 newly compiled units and the existing
672-byte Tensor descriptor. Its source SHA is
`fac3d413c6c0f6874d224cbea312108ee2c8baa027102d41890d85094a5faa38`;
its actual primary ELF SHA is
`5b7aac203d2c913d43781f0da45d40e9bac2037c2bc074ec30f8dd663710a805`.

The source keeps all prior701 Tensor bodies and twelve Tensor units exact, adds
the original native Generator model and connects both protocols to a genuine
four-argument entropy entry. The fixed compiler and entropy linker build the
application directly; no parent application ELF or old overlay chain is needed.

`make test` and focused CI require nine groups totaling **644 cases**: Bool27,
original76, Integral24, F32-output Sum17, live35, live270, live32, Generator131
and F32 Sum32. The total case and control contracts are fixed. The selected
`HEPTA_TORCH_PYTHON` must be CPython3.12 and must also launch the outer coordinator
with `-I -S -B`. Exact AST proofs are unchanged; an interpreter mismatch fails
before any case. The group timeout budgets total165 minutes; CI has210 minutes.

The current production644 run has **passed** on the exact current source/image.
Independent readback verified all 65,627 bound inputs and all nine original
runner streams. Source-head hosted artifact admission remains **PENDING** and
requires its own actual run and complete artifact review. The independent
Generator131 source03 run and old513 child executions remain separate evidence.
Preserve the old513 aggregate FAIL, its unchanged fixed3.12 pure-read supplement,
original Generator-control FAIL and the zero-case launch failure. Exact identities
and current states are in
[current-runtime712/evidence.json](../validation/current-runtime712/evidence.json).

## Next implementation work

1. Finish the current source-head hosted qualification and retain the completed
   current644 actual evidence. Keep complete raw
   request/response streams, owner boundaries, original reports, all exits/EOFs,
   fixed references, tool receipts and current installation provenance. Confirm
   that every group used the same actual primary source, ELF and build receipt.
2. Extend dtype/layout/promotion and ownership foundations against independent
   upstream observations. Public F64 and later combined candidates remain
   separate from712 until their complete source inverses, true cold build,
   unchanged inherited gates and new feature/resource tests all pass. A finite
   candidate result is not default qualification or a full dtype gate closure.
3. Expand operator, AD, frontend/dispatcher, serialization and device contracts
   against the fixed inventory. Preserve each schema's layout, dtype, error,
   alias, mutation, resource and gradient obligations; do not replace whole
   schema closure with a list of passing examples.
4. Implement distributed, compiler/export, data-pipeline and platform contracts
   in their actual native environments, then run the original complete acceptance
   workload and independent final review. Keep cross-feature composition and
   resource failure tests alongside numeric comparisons.

The original Generator131 gate preserves Torch24/native11/entry59 and adds37
dispatch controls. It does not establish full Tensor/Generator handle
interoperability, global/default bindings, concurrency, every distribution or
accelerator RNG. Those remain explicit work, as do forward/higher-order AD,
complete optimizer modes and broad dtype/device coverage.

## Retirement order and retained dependencies

Retire a path only after its consumer is identified, its required original bytes
are retained once with SHA/mode provenance, its replacement keeps all original
comparisons, and the current-image execution plus source-head CI pass. Preserve
original failed, partial, unknown and prepared-only reports exactly. Source
adaptation or a successful read-only admission never becomes a new execution.

Current consolidation includes the compact original76 fixture, the preserved
old comparison/source baseline, current live32 replacement of duplicate old
workflow calls and the new Generator fixtures. Nineteen old Generator source
files plus the old checker and manifest leave the live implementation path;
all original26 members, the original manifest and seven readable historical
reports remain recoverable. Two duplicate old Generator CI calls are replaced
by the mandatory current group. See the
[retirement index](../validation/current-generator-original26/retirement.json)
and [current audit](current-audit.md) for the exact boundaries.

Do not remove the fixed compiler/bootstrap sources, original three-argument
session and runtime linkers, original Torch entry/support/lock, actual pip report,
RNG source used by other components or unrelated legacy qualifiers. They still
have concrete consumers. Also retain the461 old protocol fixtures, independent
dtype/optimizer/resource contracts and the F64 validator's old143-source witness
until each is migrated and actually qualified. A sparse physical absence does
not make a tracked Git file obsolete.

## Evidence and acceptance rules

- Keep one authoritative current runtime source and ordered unit manifest. New
  source or support/ABI changes require reviewed exact provenance and a fresh
  cold build; prior image PASS is not transferable.
- Keep all original oracle functions, inputs, tolerances, loops and denominators.
  A deliberately changed supported contract retains its original failed
  observation and requires an explicit outer classifier plus new independent
  evidence; unrelated errors must still fail.
- Bind actual native bytes and every process's real output, exit, both EOFs and
  resource cleanup. Source archives, simulated results, prepared-only readers
  and summarized logs cannot replace these observations.
- Preserve mode and complete member identities in durable evidence. Original
  receipts containing absolute input/raw paths require those exact locations
  or an explicit reviewed physical mapping; never rewrite the original receipt.
- Keep every full-system gate's status unchanged until its complete acceptance
  contract is independently satisfied. `make acceptance-status` checks inventory
  consistency; `make full-acceptance` remains unsuccessful while closure is absent.

The [current build guide](current-runtime.md) contains the only default build
and qualification commands. Historical topic documents and retained evidence
keep earlier implementation details; this plan describes the current work order.
