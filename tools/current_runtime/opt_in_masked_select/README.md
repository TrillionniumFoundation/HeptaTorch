# Opt-in HIA F32 masked_select on ordinary CPU roots

This isolated native candidate is based on exact default main e9444902.
It adds only candidate opcode 130 with wire form [130, value, mask, 0, 0,
0, 0, 0]. It does not change canonical/default activation. Inputs are F32
and Bool, equal shape and positive rank 1–4, contiguous owned no-grad CPU
roots, bounded to 262,144 input elements regardless of output size.
Two stable passes count canonical Bool selections and copy exact F32 raw
words to a fresh rank-one [K] result. K=0 avoids addresses. No general
broadcast/view/rank-zero/autograd/out/device or whole PyTorch claim.

Locked PyTorch and independent no-Torch owner/frame parsing agreed on
all 28 cases / 364,415 output words, including six empties, signed q/sNaN,
negative zero, row-major rank 1–4 compaction, full and sparse selection
at 262,144 input words, and independent result Tensor/Storage/Version.
The actual old executable refused the same 28 valid setups with status
4 and complete owner rollback. All 262 other exact-main setup/read/drop
request and response frames replayed byte-identically; this is not the
full hosted default aggregate. Both cold builds produced identical ELFs.

The physical controls preserve seeded nonzero shape scratch, source
words/owners, zero initial Version, both release orders, weak expiry,
same-frame reference/read and late invalid suffix atomicity. Verified
actual view and grad, invalid shape/dtype/padding/rank and stale handles.
For N=262,656 and intended K=0 or 2, status 3 leaves the entire owner
unchanged. Mirrored clean and refused processes reach identical next
output IDs, owners and raw bytes. Status 3 is an input-cap refusal, not
a physically observed allocator OOM. No allocator OOM claim.

Earlier compiler refusal (discarded scalar call results), stride-unit
failure, and response-cursor failure are retained alongside corrected
evidence. An initial positive-offset view test omitted the op111 payload
size; the supplemental exact candidate run constructed the real view
and verified refusal. A separate default-main cat(34) setup returned
wrong bytes for the large ordinal source; no masked_select was executed
on that setup. The exact immutable input bytes were instead recovered
on both physical images using ordinary broadcast-add then independently
read in full. Neither the test fixture nor the candidate was weakened.

Prepare from exact main source: python3 -I -B prepare_masked.py
EXACT_MAIN_SOURCE EXACT_BUILD_PY NEW_SOURCE. Build cold twice, pin the
second with pinned-manifest.json. Expand selected-u32le.bin via
expand_locked_raw.py and use directory containing receipt.json and raw
readback as MEASURED. The guarded run_locked.py can independently
regenerate these bytes in the locked official wheel. Run native.py
PINNED_HELPERS WIRE ELF prereg.json MEASURED OUT baseline|candidate|guards,
then pure.py prereg.json MEASURED CANDIDATE_ELF_SHA BASELINE CANDIDATE GUARDS
for raw checking. Native capture groups are separate so the two high-cap
regions need not be repeated to qualify view or rollback cases.
Run replay_main.py PINNED_HELPERS WIRE CANDIDATE_ELF BASE_CAPTURE OUT, and
mirror_caps.py PINNED_HELPERS WIRE CANDIDATE_ELF OUT with pure_mirror.py OUT.
