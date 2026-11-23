# Native CPU where.self: explicit dimensioned F32/F64 values

Adds an exact F32/F64 value pair in both orders when both value tensors
have rank at least one and the condition is Bool. Condition rank zero is
allowed. Each operand keeps independent width-4/8 ordinary CPU root and
owner checks before the existing three-way broadcast and typed allocation.
The chosen F64 word passes through directly. The chosen F32 word calls
the unchanged existing bitwise widening helper; output is exact public
F64. Mixed rank-zero values, wrapped Python scalars, other mixed pairs,
views, autograd, out= and default/hosted activation remain unqualified.
Already admitted same-dtype and Bool-pair behavior remains unchanged.

Before the first native build, locked official CPU Torch observed 270
cases and 8,772 output words. Both orders and contrasting phases were
compared at every length 1–65, with deliberate signed sNaN/qNaN, distinct
payloads, and words around vector/tail boundaries. Signed zeros, subnormals,
extrema, infinities and full selected special-value banks were included.
There were zero differences from an independent integer F32 widening model
(sign and payload preserved; F32 sNaN quieted) and selected F64 identity
(including F64 sNaN remaining signaling). This is an exact scoped result
for the locked CPU environment, not a claim about every backend.

Two independent cold builds then produced identical ELFs. A separate
no-Torch capture, read independently by a Cartesian raw oracle, matched
all 270 official cases including two empties, mixed broadcast and rank4.
It observed 4,386 widened F32 words and 4,386 unchanged F64 words. Fresh
distinct result Tensor/Storage/Version with Version zero, unmodified
inputs, both release orders, weak release and request-local result held.

The exact predecessor refused all 270 actual inputs with status 7, no
result and byte-identical complete owner (2,700 frames). The successor's
5,206 new frames included 296 exact rollback checks, six actual
mixed-rank-zero denials, four genuine positive-offset views, incompatible
shapes, compatible overcapacity status 3, unadmitted pair types, stale/
padding/grad and a valid pair followed by invalid suffix.

Two archived F32/F64 dimensioned status-7 inputs became positives with
exactly the same recorded shapes, tags and raw input words. All other
earlier request/response bytes replayed exactly after removing those two
records. Across the successor's new and retained suites: 10,956 frames
and 853 exact owner rollback checks. Original refusals remain separate.

Reproduce the source projection: isolated Python
`prepare_wide.py EXACT_PARENT EXACT_BUILD_PY`. The optional final new
directory materializes the checks. The included manifest pins the second
cold build. Run `oracle_wide.py PINNED_HELPERS ORACLE_LOCK INSTALL_REPORT
FIXTURE` with the locked wheel separately, or expand the exact compressed
reference with `expand_fixture.py wide.fixture.json.zlib.b85 FIXTURE`.
Then use `native_wide.py PINNED_HELPERS PROTOCOL_PY ELF FIXTURE OUT
baseline|candidate` without Torch, and `pure_wide.py FIXTURE
BASELINE_CAPTURE CANDIDATE_CAPTURE` independently. The archive control
reader is `verify_retired.py ORIGINAL_BOOL_FLOAT_CAPTURE
RETAINED_BOOL_FLOAT_CAPTURE NEW_CAPTURE FIXTURE`.
