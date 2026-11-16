# Independent review of owned Tensor candidate 03

## Qualification identity and outcome

The independently executed candidate is `owned-tensor-runtime-extension/candidate-03/owned-tensor-session`, SHA-256 `05c5544da2b7b34a8db9bd019a3ad0fa930f9ed5c18fd4b7582e7150f21eaeea`. Its numerical core source is `91e5cb7d80299e735dcac2dc3520bafbf3e272465ccdced6aa2f732800127b07`; its owner-step unit is `a4135efeac62bb72e5e8c4437a9de27b330eab1bee49923708fb6b1848d4fe2c`. The candidate's build receipt records 150 core functions, with 126 of the previous 143 functions byte-identical. This report does not bind the earlier candidate 01 or a future successor.

`candidate-03-review/report.json` records **16 actual interactive frames, 612 exact scalar or binary32-field comparisons, and 41,943,040 bytes compared across ten complete-owner rollback checks**. Responses were obtained before stdin EOF from the same process. The owner actually grew from 131072 to 4194304 bytes. The numerical chain uses an independently differentiated exact-rational reference and a separate execution of the unchanged canonical native runtime. It does not claim execution of real PyTorch; that separate 28-case oracle is prepared under `oracle/` and was locally NOT_RUN because Torch is unavailable.

The report's `created_handle_count=518` counts explicitly checked creation-result handles, including the 260 original leaves and 258 post-reset scalar creations. It is not the count of every intermediate tensor allocated in the whole scenario. All 518 checked handles were unique across those creation events. Raw framed requests/responses, the exact executed test and wire encoder, and the transport observer are retained beside the report.

Sources for this contract are the frozen candidate's `overlay-authoring.py`, `runtime.hepta`, `units/owned_tensor_session.hepta`, and `build.json`; the canonical source `264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233`; and `source-inventory.json`. The authoring text is read as a specification of the HIA it generated. Target execution in the review was the native HIA executable.

## State layout and typed identities

The owner is one real HIA `buffer`, allocated by the pinned native ABI2 allocator. The wrapper transfers it through `step(readbuf,mutbuf,buffer)->(buffer,u32)` and `close(buffer)->()`. Payloads never carry a native pointer. Values stored inside the buffer are integers, tensor handles, or relative offsets; none are reinterpreted as a new ownership capability.

| Byte offset | Meaning | Lifetime |
| --- | --- | --- |
| 0 | Application status | Cleared for each trial frame |
| 8 | Next descriptor slot | Committed state |
| 16 | Arena allocation cursor | Committed state |
| 24 | Backward work epoch | Committed state |
| 32 | Handle generation | Advances only on successful graph reset |
| 40 | Fixed retained leaf prefix | Fixed after the first reset |
| 48 | Required capacity hint | Trial-only; zero in committed state |
| 56 | Retry reason, 1 means allocation capacity | Trial-only; zero in committed state |
| 64 | `HTSTATE2` state-format magic | Checked before cloning a nonempty owner |
| 128 onward | Shape scratch | Existing kernel scratch |
| 2048 onward | Command/configuration scratch | Existing configuration fields extend through the 2080-based optimizer configuration |
| 4096–6144 | 256 directory entries, each u64 | Relative descriptor-page offsets |
| 6144–8192 | Reserved | Not descriptor data |
| 8192–57344 | Descriptor page zero, 256 × 192 bytes | Always present |
| 57344 onward | Tensor payloads, gradient/work buffers, auxiliary records, and later descriptor pages | Arena-relative allocations |

The source inventory propagates the workspace formal through CFG block arguments when collecting direct `getw`/`putw` and `get4`/`put4` accesses. In the old canonical source, direct fixed header accesses are only 0, 8, 16, 24, 32, and 40. It found no old direct fixed access at 48 or 56. This is a literal-access inventory, not a claim that scanning constants proves the absence of every indirect write; initialization and indirect configuration stores were also read separately.

There are 65535 usable descriptor slots, numbered 1 through 65535. The low 16 bits of a Tensor handle select its slot; the high 48 bits encode generation. Slots in the retained prefix use their raw slot number. A handle is valid only when it exactly equals `live_id(slot)` and its slot is nonzero and below `next_slot`. Generation exhaustion is refused, not wrapped.

Directory index is `slot >> 8`; the local index inside a page is `slot & 255`. A descriptor address is `directory[index] + local_index*192`. The page-local mask **must remain 255**, even though handle masks in validation and graph traversal become 65535. Page zero reserves local slot zero because handle zero is invalid. New pages are 49152 bytes and are allocated at the arena tail; their directory entry is written only after successful allocation.

### Descriptor fields

| Descriptor-relative offset | Type and role |
| --- | --- |
| 0, 8 | Rank, element count |
| 16 | Data offset relative to the real arena owner |
| 24 | Leaf gradient offset |
| 32, 40 | Requires-grad flag, AD operation |
| 48, 56 | Parent Tensor handles |
| 64 | Handle of the legacy storage/version-owning Tensor descriptor |
| 72 | Mutation counter, read through the owner descriptor |
| 80, 88 | Saved parent mutation counters |
| 96–120 | Four dimensions |
| 128–152 | Four element strides |
| 160, 168 | Work-gradient offset, backward visit epoch |
| 176, 184 | Operation-discriminated auxiliary fields |

Offsets 176/184 cannot be relocated blindly. Transpose stores two axes; reductions/softmax store axes and flags; matmul stores batch rank and reduction extent; spatial operations store configuration and sometimes argmax offsets; index-select/gather store an axis and index-array offset; cat/stack store an axis and the saved parent-list offset; indexed additions store an axis and an auxiliary-record offset. All underlying arena offsets remain unchanged during owner growth, so no such field needs physical-pointer relocation.

**Identity boundary:** the numerical descriptor still couples its storage-owner and mutation-counter-owner identity in field 64. It does not yet contain the separately implemented byte service's independent Storage, Version, Weak, Saved, and Allocation identities. Real dynamic allocation and valid migration are established here; complete Tensor storage object integration is not.

## Capacity and shape admission

Owner capacities are powers of two, from 65536 through 1073741824 bytes. The element-count cap is 16777216. The per-dimension cap remains 262144; rank remains at most four. The same number appearing elsewhere is not automatically a capacity field: spatial parameters, one-axis extents, convolution groups, and the output-axis length of index-select retain their original limits. `capacity-classification.json` lists each old 262144 occurrence by role.

Descriptor creation checks every dimension and scans for zero before multiplying a nonempty shape. Candidate 03 also makes matmul output count and gather index count two-pass and zero-aware. This admits large leading dimensions followed by a zero extent without ignoring an individually excessive dimension. Nonempty count multiplication stops at 16777216. Keeping the per-dimension bound also keeps rank-four element-stride products representable; removing it would require a new overflow proof.

Work budgets remain separate from allocation capacity. Matmul, convolution, pooling, and indexed-add work refusals must not request a bigger owner. The independent review executed both a per-dimension refusal and a pooling-work-budget refusal and verified complete committed-state equality afterward.

## Growth and atomic frame behavior

The owner step retains the original owner while creating a genuine `buffer.try_new(capacity)` trial. A nonempty state's bytes are copied to the trial using real borrowed-buffer operations; empty state is initialized natively. Only the trial is exposed to the original numerical entry point.

A retry requires all of: response status 3, reason word 56 equal to 1, and required-capacity word 48 greater than the current capacity. The trial is then dropped, a larger power-of-two allocation is attempted, and the entire original frame is replayed from the unchanged committed state. Ordinary status-3 refusals are not sufficient to retry. This distinction is required because shape limits, work limits, descriptor exhaustion, and response capacity also use status 3.

On success, the original owner is dropped and the trial becomes the returned owner. On a normal failure, the trial is dropped, the original owner is returned, and the reply is normalized to a 32-byte header with count zero. This includes failures after earlier commands in the same frame changed data, allocated a descriptor page, or triggered a larger trial. No provisional ID or value is published.

The independent native test mutated a retained parameter, requested growth, then issued an unknown opcode. It also forced a real allocator failure with a 64 MiB process-address-space limit while attempting the next 128 MiB trial. Both returned the original owner pointer and byte-identical committed state. These are actual native executions, not simulated allocator outcomes.

`state_header` checks the format magic; capacity minimum, maximum, and power of two; clean committed status/hint/reason words; slot/generation/prefix bounds; arena bounds; page-zero address; monotone, nonoverlapping, aligned allocated-page extents; and zero entries beyond the last live page. It is a format/header admission check, not a full independent proof of all Tensor/AD record invariants in an arbitrary foreign buffer. The ordinary external protocol only starts from the trusted live-empty owner and passes back owners previously returned by this step.

## Reset and aliases across migration

Reset permits only an owned leaf prefix: each retained descriptor must have AD operation zero and identify itself as storage/version owner. Once generation is nonzero, the retained prefix cannot change. The new cursor is the maximum of:

- 57344, the end of mandatory page zero;
- the end of every descriptor page covering the retained prefix;
- every retained leaf's data, gradient, and work-buffer end.

Including descriptor-page ends is essential for zero-element leaves: their data allocation alone does not keep the rest of their descriptor page live. Candidate 03 includes this rule. It first clears discarded descriptors while their page directory still exists, then clears higher directory entries and the discarded arena tail. Finally it advances generation and updates the prefix, cursor, and next slot. It reclaims arena positions while retaining the current physical owner capacity.

The independent review retained 260 leaves across a page boundary, checked that page one's full end remained inside the cursor, created 258 generation-1 scalars across the next page, reset again, and verified that a generation-2 handle reused the slot while both earlier handle generations were refused. Original tensors, nonuniform gradients, and data survived the growth and resets.

## Deliberately new protocol behavior

The outer owned frame is `HTOWNR01`/`HTOWNS01`. Its body is `HTORCH02`; replies are `HTORSP02`, version 2. Command record width and old numerical operation numbers remain unchanged. Command 38 constructs a constant tensor directly in native storage; commands 39/40 read bounded value/gradient windows. The 1 MiB frame/reply envelope remains, so large tensors require windows.

Constructor payload stores and `constant_tensor` use direct initialization without mutation-counter bumps. Newly constructed Tensor counters therefore start at zero. Explicit mutation and the existing optimizer kernels still increment counters according to their current native implementation. Initial-zero and one explicit write have native evidence; **all optimizer absolute-counter parity is not presumed**. The prepared real-Torch oracle strictly compares every direct parameter counter and retains explicit optimizer-state lifecycle differences. In particular, AdamW or moment-state paths can have more than one real Torch in-place operation per step.

Old protocol handle widths and old initial counters are not silently reinterpreted. This is a new profile and requires its own qualification receipts. The former profile's counter gap remains recorded separately. The original PyTorch inventory denominator remains 2590 and full rewrite completion remains false.
