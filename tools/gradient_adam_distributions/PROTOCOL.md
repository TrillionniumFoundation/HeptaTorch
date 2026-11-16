# Combined gradient, Optimizer and distribution profile

This document describes the combined profile's additions and state boundary. It inherits the exact wire fields of `tools/gradient_objects/PROTOCOL.md` and `tools/adam_distributions/PROTOCOL.md`; it does not reinterpret the older borrowed or owned entry ABIs.

The outer frame is `HTOWNR01` plus a little-endian u64 body length. Responses use `HTOWNS01`. The body remains `HTORCH03`; response bodies remain `HTORSP03`, version3. A command is eight little-endian u64 words. Packet limits and ordinary transactional failure handling are unchanged.

The private numerical state tag is `HTGAD002`; its backing object-state tag is `HTOGAD02`. Descriptors are576 bytes. These tags reject older nonempty gradient-only and Adam/distribution-only states with status7. The checked entry returns the original owner unchanged on this refusal. A live empty owner remains a valid initialization input. Candidate04 and05 share this internal format; candidate05 changes only pointwise layout ordering, not storage representation or admission.

## Public interfaces

| Command | Meaning | Result |
|---|---|---|
| 91 | Read the existing gradient binding of Tensor `a`; all remaining words0 | Existing real gradient Tensor token or0 for None; no payload |
| 92 | Assign real Tensor `b` as Tensor `a`'s gradient; literal0 clears it; remaining words0 | Scalar1; no payload |
| 93,94 | Reserved | Status4 when otherwise well formed |
| 70 | Clear a leaf gradient binding to None | Existing nullable-gradient contract |
| 73 | Optimizer zero_grad, true removes bindings and false zeros existing objects | Existing owned-Optimizer contract |
| 95 | No-grad scalar add of low32-bit binary32 word `b` to every logical element of Tensor `a`; remaining words0 | Existing checked scalar-add contract; one real Version increment, including empty |

The existing 71–79 Optimizer interfaces, command90 leaf requires-grad update, Generator60–69 and distribution80–89 contracts are retained. Commands58,86,87 retain their explicit runtime-error payloads and stop the current packet suffix while committing the specified operation prefix. Ordinary errors still discard the entire trial owner.

Public Tensor handles retain their kind/generation validation. Request-local result references can name only previous real Tensor results, not a setter success scalar or getter None. A published gradient names its real Tensor incarnation, Storage, Version and layout. There is no temporary copy in the getter. Clearing releases the parameter's binding; a public gradient token, independent alias, Saved record or other checked strong owner can keep the object alive.

Metadata13 remains16 u64 words: rank, numel, requires_grad, four shape words, four stride words, Storage ID, Version ID, Version counter, storage offset in bytes, Storage extent in bytes. Strides are element strides. The pointwise layout comparator alone converts them to signed binary32 byte strides as required by the qualified TensorIterator behavior; actual addressing still uses the original element strides.

## Internal ownership fields

The older fields retain their roles: descriptor88 captured AccumulateGrad references;312 Optimizer/private Tensor roots;320 stable Tensor incarnation;384/392 bound gradient node/incarnation;400 private gradient-binding references. The public gradient object and its source parameter do not share these identity fields accidentally.

The incoming layer retains448–536 for the actual incoming Storage/Version, layout, origins and logical cache ownership. The combined profile additionally uses:

| Offset | Temporary role |
|---|---|
| 544 | Physical Preserve-layout reserve address |
| 552 | Private Hold retaining that reserve |
| 560 | Reserve Storage ID |
| 568 | Reserve Version ID |

The derivative work pointer at160 remains a logical-order array. A noncontiguous fresh incoming result can use the separate physical reserve. Writes and merges maintain both representations through checked address calculations; the numerical work array is never silently reinterpreted as a transposed physical Storage.

All incoming fields448–568 are zero at a committed frame boundary. Admission allocates required resources before partial execution. Success and explicit operation errors release temporary Holds, Versions and unused gradient placeholders. Failed admission leaves the original complete owner intact. Constructor-owned work caches may legitimately exist before backward; their Hold/pointer pairing is checked instead of requiring them to be absent.

## Compatibility limits

This profile preserves the existing bounds and errors. The finite scalar-add path is unchanged; the narrow NaN/Inf helper is qualified against the frozen scalar48 report. The signed-byte comparator preserves admission of the eight observed extreme size-one strides. Neither change claims broad dtype, device or ISA equivalence.

Nonleaf retain_grad, general higher-order/autograd identity splitting and the other open boundaries in the README remain unimplemented. No new command is allocated for them here.
