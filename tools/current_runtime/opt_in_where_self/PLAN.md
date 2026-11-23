# Opt-in CPU where.self: source projection, not physical qualification

Exact stacking base: HeptaTorch draft #65 commit
`809cffaee8e5b148156320245e3337bcf55fb5fd` (confirmed by GitHub),
complete source `11343747e6c86b7776842d26c9eb60fc993cb02cddb21ca21b82ddcb9f88b5f3`.
The same projection is reversible on the immediately earlier constructor-only
source. GitHub branch enumeration (70 heads; next page exhausted), targeted
where branch/PR lookup, repository PR lookup for opcode 129, and exact local
source census found no competing route. This is a scoped collision check,
not a registry guarantee for unpublished or concurrently moving branches.

Contract: reserve **opt-in only** record `[129, condition, x, y, 0, 0, 0, 0]`.
Use rank 0..4, at most 262144 elements, all same shape, ordinary contiguous
zero-offset live owner roots with no autograd/view state; condition Bool
tag 1 with canonical 0/1 reads; both values exact same dtype, among public
F32, Bool, I64, F64, F16, BF16. The source handles rank 0/empty in principle;
report either only after independently verified lifetime/geometry.
Fresh Tensor, Storage and Version (initial 0); no input mutation, even if the
input handles repeat. This is not dtype promotion, broadcasting, arbitrary
layouts, masked nonzero indices, inplace, out=, autograd, or default routing.

One partition only: change `entry` non-128 branch, add
`where_ordinary`, `where_self_tensor`, `where_self_command` to unit12.
The existing `frame_valid`, state/dispatch fallback, `logical_out_apply`,
out-layout helper, and all other original function bodies are byte-identical.
The parser strictly reads the three public handles and unused words and uses
the original entry b45 trial continuation. `shape_command_finish` handles
the normal publish/reply status; no separate result-map or owner backend.
For both input and result bounds, reuse the ordinary-root helper, plus
explicit tagged-span nonwrap guard. Read chosen value as a raw word with
`tget` and store with `typed_store` using dtype **byte width**, after bounds
and checking the consuming read status. No f32/f64 arithmetic or casts.
Use existing record encoding in an opt-in observer; a normal successful
result is a 32-byte record carrying the new opaque handle.

Physical acceptance for later separately assigned compute:
1. One locked, official CPU PyTorch process produces immutable bit fixtures
   for selected asymmetric rank-2/3 masks, alternating and all-true/false,
   same-handle x/y inputs, scalar (both mask values) and typed owned empty.
   Across all six value dtypes observe same exact shape/dtype/raw words.
   Include selected/unselected signed zero, positive and negative
   subnormal/Inf/qNaN/sNaN, signed I64 extrema, Bool combinations. Generate
   with uint word reinterpretation; no Python float normalization, canonical
   words pinned. If official zero/NaN semantics do not match, report exact
   scope and all failures; do not silently relax bit expectations.
2. Run the exact previous image on valid fully materialized 129 input roots,
   and demonstrate actual 129 unknown-op refusal and full owner byte
   equality. Check one malformed frame and same-process recovery separately.
3. Candidate two independent cold native build identities; no Torch in
   process. For each retained new output, inspect result via genuine wire
   ops 98/13 and decoded owner bytes: width/metadata, distinct roots and
   Tensor/Storage/Version IDs, new Version=0, source versions/all bytes
   unchanged; repeat inputs and lifetimes (drop inputs first and result
   remains readable; reverse order and input use remains valid; independent
   weak Storage release). Optional direct same-frame local-result consumer
   exercises unmodified entry result-map.
4. Refusal ledger with before/after *complete* owner bytes: wrong/stale/
   dropped handles (each position), each nonzero padding word, numeric
   condition, different x/y dtypes, equal numel but different shape,
   broadcast only, actual existing view in each position, preexisting
   marked grad/graph, rank/count limit where setup truly succeeds, unknown
   commands, real allocation pressure (no hand-edited state), and successful
   129 followed by known invalid command in one atomic frame. Never count a
   failed setup or unsupported/malformed physical oracle as a positive test.
   Candidate padding/admission rejects set status7; invalid public handles
   preserve common status1; preexisting allocation error is propagated.
   Exact baseline opcode status still must be observed. Outer status must
   yield no public partial results after the failing whole request.
5. Replay inherited #65 output and both constructor matrices and 79/82 as
   immutably as possible. Readback independent of native harness and Torch;
   preserve original bytes, receipts and failure history. Report exact
   source/ELF/oracle hashes and resource floors separately; no hosted/default
   parity claim. Do not make canonical wire registration or change main.

Source-only invocation (in-memory projection; no toolchain/native/Torch):
```
python3 -I -B prepare_where.py --checker /path/to/pinned/build.py \
  --source /path/to/exact/pinned-source
```
Only the future sole compute phase should use the optional
`--materialize-later` flag; it writes the integrated source for the usual
independent cold builder without running that toolchain.
