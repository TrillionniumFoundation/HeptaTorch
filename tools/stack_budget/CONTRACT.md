# ABI 3 stack contract

## Producer and resource boundary

The producer is the exact native compiler identified in `profile.json`, derived reproducibly from Hepta commit `f33d8da316de8fdf27b504371f1a5310997ff548`. The previous owned profile remains ABI 2 with its original pins and refusal contract. ABI 3 changes the interface version and emits an additional `.hia.stack` section. The private function ABI remains a pointer to a 16-byte-slot argument area in `rdi` and a pointer to a three-slot result area in `rsi`.

Existing exact symbol/signature/effects matching, borrowed-origin and live-loan restrictions, unique owned inputs, move/return rules, module DAG, allocation provider and no user-supplied native owner pointers remain in force. Ordinary code generation and all 125 retained compiler functions are AST-identical to the ABI 2 authoring, apart from the two explicit metadata writers and the one new metadata function. Scalar/owned multi-results retain their existing semantics. ABI 3 does not create a general foreign-pointer ABI, tensor storage table, or new owner opcode.

The native linker is an admission mechanism for byte-controlled products of that producer. It is not a cryptographic attestation service or a general x86 verifier. It rejects structural and call/stack metadata mismatches, but arbitrary machine-code substitutions that preserve these patterns are outside the producer contract. Type/effect/alias semantics and the bounded local temporary stack patterns still rely on the pinned compiler and verified source/object custody.

## Object encoding

The object has nine canonical sections: null, `.text`, `.symtab`, `.strtab`, `.shstrtab`, `.note.GNU-stack`, `.rela.text`, `.hia.interface`, `.hia.stack`. Interface records retain 256 bytes; byte offset 192 now holds the local function ordinal and offsets 200 through 255 must be zero. Signature comparison still covers offsets 16 through 191. Every public/import symbol maps exactly to its function record. Private functions have no public symbol.

The 64-byte stack header contains eight little-endian words:

| Offset | Meaning |
|---:|---|
| 0 | `HIASTK03` |
| 8 | Stack-record format version 1 |
| 16 | Function count including imports, at most 256 |
| 24 | Number of locally defined functions |
| 32 | Function-record size, exactly 64 |
| 40 | Call-matrix row width, `ceil(function_count / 8)` |
| 48 | Maximum closed helper temporary overhead, exactly 40 bytes |
| 56 | Closed helper flags: 0 none, 2 allocator, 3 chain destructor plus allocator; other values reject |

Each 64-byte function record holds ordinal, text-relative offset, code size, frame bytes, symbol ordinal, import flag, highest SSA slot index and a zero reserved word. Imports have zero code offset, size, frame and SSA index. Local functions cover `.text` contiguously in source order; the remainder must be exactly the closed helper sequence indicated by the flags. The trailing matrix has one bit per non-self call and zero alignment padding. Imported rows and diagonal bits are zero.

## Binding metadata to actual code

Frame admission checks the real `push rbp; mov rbp,rsp` prologue and every bounded stack subtraction/read probe. The frame is:

`(highest_SSA_slot_index + 2) * 16 + 576`

The highest legal index is 2047, so the largest frame is 33,360 bytes. This is an index, not the number of named values. Each subtraction is at most 4096 bytes and is followed by the producer's actual stack-page read.

The HIA linker sequentially decodes the fixed backend's supported integer/resource instruction subset. It advances across complete operands and immediate fields; an `e8` byte inside an immediate is not a call. Unknown encodings, indirect calls and direct syscalls inside HIA function bodies reject. A second frame prologue inside a declared function rejects.

For each actual direct `E8` instruction, a sorted unique `PLT32` relocation either resolves the exact imported interface ordinal, or the signed displacement resolves an exact local function start or one of the closed helper entries. Every relocation must occur at a decoded call operand, and all such imports must be accounted for. Calls into a function or helper interior reject. The reconstructed matrix must match every declared byte exactly. Thus both deleting a real private/import edge and inventing an absent edge reject.

Direct jumps and conditional branches must target an already decoded instruction boundary inside the same function and after its frame allocation. Ordinary tail-self calls jump back to the declared block entry after the prologue and add no native frame. The chain destructor and allocator tail are checked byte-for-byte against pinned implementations, including the destructor's actual free-call displacements.

This binding closes the earlier matrix-only admission flaw. It is still not a proof of arbitrary substituted instructions' register, memory or effect behavior.

## Native budget computation

The HIA linker resolves imported function costs through the existing checked module DAG. Within each unit it runs an explicit, bounded DFS with gray/finished colors. Gray edges reject cycles. It computes:

`cost(function) = frame(function) + 16 + max(cost(non-self callee))`

An import copies its provider function's cost. The 16 bytes account for the ordinary call return address and saved `rbp`. Arguments and all three result slots are already within the compiler's fixed frame. All functions, including unreachable definitions, are admitted; the maximum is conservative. Sequential sibling calls take the maximum rather than the sum. Checked subtraction rejects any overflowing or over-budget addition.

The production profile admits at most 8,384,512 path bytes and reserves a further 4096 bytes inside an 8 MiB usable stack. The reserve covers entry/result staging, cleanup/I/O calls and the fixed maximum 40-byte chain-helper path. Allocator helpers have no local push/subtraction or nested call; a direct allocator call adds 8 bytes. Chain push/pop temporarily use 32 bytes plus an 8-byte allocator return address. The shared destructor uses call 8 + saved `rbp` 8 + local 16 + free-call 8 = 40 bytes. These paths do not retain temporary stack across a later HIA call.

Workspace layout preserves all previous tables. Per-object stack pointers begin at 13,112,320; 64 by 256 function costs end at 13,243,904. Ready states, colors and the bounded DFS end at 13,252,608. The reconstructed matrix, boundary bitmap for up to 1 MiB of text, compact local-function map and scratch header end at 13,394,016, below the unchanged 16 MiB linker workspace. Every used temporary region is initialized before use, including dirty and reused workspace execution.

## Startup profiles

| Variant | Request magic | Target buffers | Usable stack |
|---|---|---|---:|
| `session` | `HIASTKS1` | Existing owned step/close framed I/O, 1 MiB each | 8 MiB |
| `compiler` | `HIASTKC1` | Input/output 1 MiB, workspace 2 MiB | 8 MiB |
| `linker` | `HIASTKL1` | Input/output 8 MiB, workspace 16 MiB | 8 MiB |
| `runtime` | `HIASTKR1` | Input/output/workspace 1 MiB each | 8 MiB |
| `qualification` | `HIASTKQ1` | Session, qualification only | 64 KiB |
| `qualification-runtime` | `HIASTKQ2` | Runtime, qualification only | 64 KiB |
| `qualification-cleanup` | `HIASTKT1` | Session plus explicit private test FDs | 8 MiB |
| `qualification-guard-low` | `HIASTKT2` | Deliberate low-guard fault, qualification only | 8 MiB |
| `qualification-guard-high` | `HIASTKT3` | Deliberate high-guard fault, qualification only | 8 MiB |

Every prefix starts code at the same fixed object text origin 4096. It maps `usable_stack + 8192` bytes with `PROT_NONE`, makes only the middle pages readable/writable, and switches to an aligned private top. Linux unsigned syscall error values are checked. It does not enlarge the process stack limit or depend on its default size. Real deeply nested executions are qualified with `RLIMIT_STACK=32768`.

The session entry remains exactly `step(readbuf,mutbuf,buffer)->(buffer,u32)` with effects 15 and `close(buffer)->()` with effects 9. It adopts exactly one returned owner after every normal step, preserves the existing 32..1 MiB response-length guard, and moves the current owner once into close on normal EOF and handled transport failure. Before close, it restores the private call-stack top, so a failed I/O helper's return address does not affect alignment. After close returns it restores the original process stack, unmaps the private stack, then unmaps I/O. Observations separately prove the actual owner, stack and I/O mappings are absent while the qualification process is still alive. Stack setup or release syscall failures exit; process teardown releases remaining mappings.

Fatal HIA traps, fatal signals and external asynchronous signal handlers have no explicit close/recovery/unwind guarantee. SIGPIPE remains ignored so ordinary EPIPE follows normal cleanup. The guard profile is not a signal-recovery or arbitrary asynchronous stack-budget contract.

## Authoritative upstream facts and retained scope

The adaptation receipt binds exact upstream file bytes, including:

- `tools/generate_hia_selfhost.py`: `define`, `function_start`, `frame`, ordinary call and tail-self lowering.
- `tools/hia_selfhost_chain_elf.py`: fixed inline push/pop and iterative destructor fragments.
- `tools/hia_selfhost_multi_results.py`: caller/callee three-slot result areas and resource transfer checks.
- `tools/hia_selfhost_returned_loans.py`: returned-loan provenance; it is retained, not weakened by this profile.
- `docs/HIA_SELFHOST_CORE_SUPPORT.json`: the pinned Core and producer contract.

The local owned profile supplies the exactly pinned mmap/munmap allocator implementation and its explicit drop/pop trap effect. This profile preserves it. Its generated-source hash, complete compiler partition, final compiler/linker images and every startup variant are bound in independent receipts.

Per-unit source, function, block, instruction, SSA, parameter and result limits, the 64-object native linker limit, and 8 MiB I/O limits remain finite. This work closes the native stack-admission path for that declared producer profile and the current Tensor program. It does not close all PyTorch operators, storage semantics, devices, compilation backends or other open compatibility requirements.
