# Proposed Bool destination extension of command 115

State format: HTSINT02. Backing format: HTOSIN02. Descriptor: 672 bytes. No new command number or autograd opcode is allocated.

The existing eight-u64 command layout is retained:
- word 0: 115
- word 1: public Tensor token, decoded by the original public_node path
- word 2: dimension count; UINT64_MAX means None, zero means []
- word 3: dimension payload offset, zero for None/[]
- word 4: keepdim, zero or one
- word 5: destination dtype; the new value is 1 for Bool
- words 6 and 7: reserved zero

Nonempty dimension lists retain the original payload-bound checks and signed-u64 axis representation. None and [] are separate input forms but both reduce all dimensions in this operation. The existing rank0..4, nonnegative-stride and 262144-logical-element finite contract remains. The source must be Bool or I64 and must not require gradients. Other destination/source combinations retain their original dispatch except for these two new explicit Bool cases.

The Bool branch computes the original dimension mask before si_plan clears the shared scratch range. It allocates a new owning output with dtype1 and requires=0. It does not retain the source, record op47, create a Saved variable, alias the input Storage, or create a gradient accumulator. The output raw element is canonical 0/1 and uses one byte. I64 is converted by testing the complete u64 value for zero before Bool addition; there is no low-byte truncation and no integer sum before conversion. Every logical source element is visited, so source bounds checks are retained even after an output becomes true.

Output rank, size, contiguous strides and output element count come from the original si_plan and typed_descriptor bodies. Empty reductions retain the false identity, including a nonempty output arising from a zero reduced extent. An empty output has zero data bytes but still follows the genuine owning Tensor/Storage/Version constructor. Version behavior is a pending new actual-reference obligation, not a guessed final constant.

All original error and rollback contracts remain required for invalid capabilities, dimensions, packet payload and allocation failure. The raw HIA patch preserves every existing packet check; newly admitted dtype1 packets must receive their own actual entry controls. No native execution or error outcome is claimed by this source package.
