# Tensor RNG profile, HTSTRNG1

The existing bounded Tensor09 wire framing remains: outer `HTOWNR01`/`HTOWNS01`
with u64 length and bodies at most 1 MiB; request `HTORCH03` plus command count,
exact length and reserved zero; response `HTORSP03`, version 3, count, status.
Each command is eight little-endian u64 words. Request-local references use
`0xFE000000 | earlier_successful_record_number` and retain typed checking.
All old Tensor commands keep the frozen Tensor09 meanings.

The only persistent owner is a real HIA buffer. State prefix is 139,264 bytes;
its magic at byte 64 is `HTSTRNG1`. The backing state magic is `HTORNGS1`,
version 2. The seven 64-byte record tables occupy `[128,128+448*slots)` relative
to the backing state; free extents start at `128+448*slots`, audit scratch at
`144+464*slots`, and the payload pool at `144+488*slots`. Total owner length is
`139264 + 144 + 488*slots + pool_bytes`. Old tags are rejected before mutation.

Kind 7 is Generator, with the same nonzero generation/slot handle scheme as
the existing six kinds. A Generator record owns exactly one private kind-6
Hold of one 5,056-byte Storage. The Hold accounts for the Storage strong
reference; the Generator does not add another count. One Generator's private
Hold cannot be the Hold of another live Generator. Its record tail is zero.
Release drops that Hold, returns the real extent to the pool, and revokes the
typed handle. Reusing the row advances its generation. No owner descriptor is
embedded in the byte pool.

## New commands

Unused fields must be zero. An operation whose `result` is an ID returns the
actual typed handle. Error frames return only the 32-byte response header and
retain the original complete owned state.

| Code | Operation | `a,b,c,d,e` | Result / payload |
| ---: | --- | --- | --- |
| 60 | Create Generator | seed,0,0,0,0 | Generator ID |
| 61 | Manual seed | Generator,seed,0,0,0 | 1 |
| 62 | Current seed | Generator,0,0,0,0 | Full u64 seed |
| 63 | Get state | Generator,0,0,0,0 | 5056 and exactly 5056 bytes |
| 64 | Set state | Generator,payload_offset,5056,0,0 | 1 |
| 65 | Clone Generator | Generator,0,0,0,0 | New independent Generator ID |
| 66 | Release Generator | Generator,0,0,0,0 | 1 |
| 67 | Advance raw32 | Generator,count,0,0,0 | 1 |
| 68 | Uniform fill | Tensor,Generator,from_bits,to_bits,no_grad | Original Tensor ID |
| 69 | Detached strided alias | Tensor,config_offset,80,0,0 | New Tensor ID |

Command 64 uses the frozen RNG03 restore contract: canonical/reachable MT
positions, low-word normalization of the imported MT/next fields, and exact
enabled float/double normal-cache preservation. Ordinary uniform/advance leaves
cached normals untouched; manual seed clears them. This is not every upstream
`set_state` acceptance/error corner and does not implement normal draws.

Command 68 takes finite, exactly representable binary32 endpoints encoded in
u32 words. It validates `from <= to` and the exact endpoint difference against
the binary32 maximum before the empty check. The width is then binary32 RNE;
each element consumes one raw32 word, including equal-endpoint intervals.
The value is low24/2^24, followed by fused multiplication/addition and the CPU
rounded-upper-endpoint clamp. An empty Tensor consumes no words but still
bumps its shared Version once. Any successful fill bumps Version exactly once.

`no_grad=0` accepts only non-gradient Tensors. `no_grad=1` explicitly performs
the supported untracked/no-grad mutation even on required-gradient leaves or
nonleaves; it does not create a uniform backward node. Other values fail 7.
Default tracked nonleaf uniform autograd is outside this component and is
rejected 5. Aliases see the writes and Version; existing Saved checks detect
the mutation. Explicit data aliases with separate Version remain separate.

For nonempty output, size>1 with stride0 is rejected before any draw. Other
nonnegative bounded overlaps are permitted, including nonzero-stride repeated
addresses: the actual serial iterator draw order determines the last value.
The order follows reversed initial dimensions, ascending nonzero strides and
the source's equal-stride shape tie rule. This also produces dense transpose
and channels-last physical iteration.

Command 69 takes ten u64 words: rank, four shapes, four element strides and an
absolute element offset into the source's Storage. Unused shapes/strides are
zero. It produces an untracked detached alias with shared Storage and Version;
it is not a differentiable `as_strided` implementation. Nonempty extent must
fit the actual Storage using overflow-safe subtraction checks. An empty view
may have an offset beyond Storage, but its remaining accessible extent is
zero. Such views support empty read, clone, detach, Saved/unpack and release;
they do not grant nonempty access beyond Storage.

## Bounds and transactions

- Rank 0–4; each dimension at most 262,144; ordinary Tensor numel at most
  16,777,216. Empty shapes validate all dimensions first.
- Uniform and raw32 advance have a separate bound of 1,000,000 words/elements
  per command. Larger requests fail 3 before RNG or Tensor mutation.
- Strides must fit a nonnegative signed u64 value; alias offset is at most
  268,435,456 elements (1 GiB in bytes), including empty views.
- Seven tables share 64–65,536 slots per kind; data pool is 128 KiB–1 GiB.
  Their limits and costs are independent, so not every maximum is jointly
  reachable. Existing quadratic audit remains capped at 4,096 slots.
- Command 54 deliberately retains the old 96-byte statistics payload and its
  six live-kind counts. It does not claim to report a seventh Generator count.
- Each public step allocates a genuine 5,056-byte scratch owner, borrows it for
  the frozen RNG core and drops it on every normal return. Public state still
  uses the unchanged three-parameter owned ABI; no pointer retyping occurs.
- Original and trial owner are separate allocations. A capacity hint discards
  the whole trial and replays the original request; a draw before a later
  growth hint is therefore committed once. All ordinary errors, invalid
  handles, work/output limits and allocation refusal preserve both Tensor and
  Generator bytes, identities and versions. Borrow/trap/provider abnormal
  process termination is not converted into a recoverable frame error.

New operations use existing status values: 1 handle/type; 2 view/shape or
expanded-overlap; 3 capacity/work/output; 5 unsupported gradient mutation;
7 arguments/state; 8 Version/refcount overflow; 9 internal invariant; 100
payload offset/length/alignment. The complete earlier protocol remains the
source for the old commands and finite object limits.
