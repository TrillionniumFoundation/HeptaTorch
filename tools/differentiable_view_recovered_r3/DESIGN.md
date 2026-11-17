# Recovered direct-view R3: actual incoming geometry
Status: AUTHORING_NOT_COMPILED. Parent R2 source197bd / image13cee; original14 PASS is not transferred.
New4 fixed f4abe reference remains unchanged. Current complete R2 native is 1 PASS /3 FAIL (c7c745).
Scope owns this production overlay; autograd owns the independently frozen coverage helper and the unique compile.

## Real values and layout
FM output storage is shared-offset-normalized and spans the maximum effective input/output geometry.
Zero-output/input routes delegate exact R2 factory logic. Nonempty incoming uses the original input
sizes, strides and effective byte offset, backed by a separate real Storage and Version.
Step2 performs whole-output copy or index_add according to the existing pinned geometry predicate
and existing reduction order, then one real Version bump. Step3 executes coverage division on
the entire flat storage when the whole input geometry may overlap, including count1 and count0 holes,
then another actual bump. Positive-count numerator NaNs preserve sign/payload and quiet through the
separately reviewed scoped helper; global divf/addf are unchanged.
The existing gradient layout contract decides whether to steal or clone. No public Version is reset:
non-dense input receiving noncontiguous incoming goes through the existing contiguous clone and yields V0.

## Preflight and bounded private storage
Keep descriptor672 /84Q and all old ownership roles. Extend the existing preserve target/reservation
to nonempty op44 parents. Allocate the maximum required physical span among the visited incoming
edges before backward execution. This is allocation capacity, not the exposed Storage size.
Before the first incoming bind, validate the cache has only its private Hold (strong1/implicitweak1),
its independent Version is V0/ref1, and the exact Hold/Storage/version tokens match.
Trim requested/rounded to the actual incoming span; return any aligned unused tail through the real
so_extent_release allocator before setting the new precise extent. No ordinary Tensor or published
Storage is resized. Later incoming merges reuse existing unique-accumulate/out-of-place behavior.
Old op41/42 first use trims to their existing dense-preserve numel*4 contract.
All failure/unused-cache cleanup remains the exact old incoming_drop_cache ownership path.
Native and independent audit must retain all original equalities; no tolerance or relaxed extent check.

## Source boundaries
Only incoming mark/reserve/describe/construct and to_as_strided_backward change plus explicit new helpers
and four tag literals. R2 constructor, lazy retained-hook transfer, old empty factory, gradient_accumulate,
incoming_obeys_contract/adopt, global arithmetic, API reader, actual expectations, and rollback remain fixed.
New tags will be HTVAD003 / HTOVAD03. Function/unit count will be derived from actual emitted code.
Cold compilation and original14/new4 replay are NOT RUN. Source DSL is builder input only; all Tensor
arithmetic, physical writes, ownership, allocation and Version changes execute as emitted native HIA.
