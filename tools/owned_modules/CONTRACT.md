# Native owned module profile

This is the separate `heptatorch.hia.owned-modules.v1` profile. The existing
`heptatorch.hia.borrowed-modules.v1` tools and interface ABI1 are unchanged.
The new profile admits actual linear `buffer` and `chain` resources across
checked HIA module boundaries, including up to three scalar/owner results.
It supplies an HIA-emitted Linux allocator. It does not implement a complete
PyTorch Storage object system or an interactive owned-state service.

## Trust and admission

Inputs are existing HIA assembly syntax. The native HIA compiler checks exact
argument and result types, effect requirements, mutable aliases, loan ancestry,
linear owner use, owner consumption, returns, and control-flow joins. An owner
cannot be consumed, moved, or dropped while a descendant loan remains live.
The native linker checks the complete symbol/signature/effect contract,
canonical ELF/interface layout, module dependency DAG, and symbolic call
relocations before publishing an executable.

These checks have distinct responsibilities. Linker metadata checks do not
authenticate arbitrary x86 machine code. Object files must come from the
pinned compiler under the checked build recipe. A consistently forged object
and signature are outside that producer-custody boundary. The allocator accepts
only exact owner bases produced by this ABI2 backend. It exposes no foreign
pointer input, pointer cast, or user-supplied address protocol.

Python scripts author pinned HIA source, partition complete function bodies,
transport bytes, and observe tests. They do not implement allocation, owner
execution, numerical operations, source admission, or linking for the target.
Production executables have no libc allocator, GNU linker, or Python runtime.
The allocator observer uses `ctypes` and `mprotect` solely to execute and inspect
actual native helper bytes from a compiled object in a constrained child.

## Private call ABI

| Property | Contract |
| --- | --- |
| Target | Linux x86-64 |
| Parameter area | `rdi`, 16 bytes per parameter |
| Result area | `rsi`, 16 bytes per result, at most three results |
| Resource slot | Pointer `u64`, followed by length/count `u64` |
| Scalar slot | Existing pinned HIA exact-width, signed, and boolean rules |
| `buffer` parameter | Linear owner consumed by the call |
| `chain` parameter | Linear owner consumed by the call |
| Owner result | New live owner in the caller |
| Borrowed parameter | Original loan provenance and alias checks retained |
| Borrowed resource result | Rejected in this module profile |
| Symbolic call | `R_X86_64_PLT32`, addend `-4`, real undefined symbol |

Private results retain their declared order. `(buffer,chain,buffer)` differs
from `(buffer,buffer,chain)`. Results may be zero to three admitted scalars,
`unit`, `buffer`, or `chain`; `readbuf` and `mutbuf` results are rejected.
Parameters may be the admitted scalar/borrowed/owner types; `unit` parameters
remain rejected. Module recursion remains rejected by the conservative DAG.

The ordinary runtime entry is still exactly
`(readbuf, mutbuf, mutbuf) -> (u32)`. It cannot be reinterpreted as an owner entry.
Future `step(readbuf,mutbuf,buffer)->(buffer,u32)` and `close(buffer)->()` startup
profiles require their own checked entry gates and owner-preserving wrapper.

## Interface ABI2

The `.hia.interface` header is 64 bytes and each record is 256 bytes. The
header magic remains `HIAOBJ01`; the interface version at byte 8 is **2**.
The version and native link-request magic jointly select the new profile.

| Record offset | Meaning |
| --- | --- |
| 0 | ELF symbol ordinal |
| 8 | Imported flag |
| 16 | Parameter count, at most 16 |
| 24 | Result count, at most 3 |
| 32 | Effects mask: alloc 1, read 2, write 4, trap 8 |
| 40 | First result type, or zero if absent |
| 48–175 | Sixteen parameter type fields; unused fields are zero |
| 176 | Second result type, or zero if absent |
| 184 | Third result type, or zero if absent |
| 192–255 | Reserved, must be zero |

The linker compares all signature/effect fields at offsets 16 through 191.
It validates absent result fields, all reserved fields, and type IDs before
resolving a call. Buffer and chain type IDs are 8 and 9. Old ABI1 objects are
rejected by ABI2 linkers, and the frozen ABI1 linker rejects ABI2 objects.

## Native allocation and release

The compiler emits closed private allocator helpers into an object's text.
Original checked buffer/chain instruction lowering calls them through local
native relocations. No external `calloc` or `free` symbol is admitted.

For a positive allocation request, the helper checks unsigned count×size,
the addition of a 16-byte hidden header, and the signed mapping-length bound.
It invokes Linux `mmap` with private anonymous zero-filled pages. Every result
in the Linux syscall error interval `[-4095,-1]` becomes `NULL`. Successful
allocation stores the exact mapping length in the hidden header and returns
the 16-byte-aligned payload base. Checked HIA buffer access exposes only the
requested payload range; it cannot access this header.

Zero-length `buffer.new` and `buffer.try_new` use a live `(null,0)` descriptor
without invoking `mmap`; `try_new(0)` succeeds. Failed `buffer.try_new` returns
a live empty owner plus `false`. Mandatory `buffer.new` failure terminates by
`UD2`. The original chain representation and move/try-push/pop result semantics
are retained. Chain nodes contain payload pointer, payload length, and next
pointer; node allocation requests 24 payload bytes and uses the same allocator.

Release subtracts the private header size from the exact owner payload base,
reads its mapping length, and invokes `munmap`. Dropping an empty descriptor
does nothing. A nonzero `munmap` result terminates by `UD2`. A normal mapping's
release is not assumed infallible: adjacent VMAs may merge, and a release that
splits a VMA can fail under kernel resource constraints.

This profile therefore explicitly strengthens two effect contracts:

| Operation | Required effects |
| --- | --- |
| `owner.drop` | `alloc,trap` |
| `chain.pop` | `alloc,trap` |
| `buffer.try_new` | `alloc`, retaining its nontrapping allocation result |
| Other owned/borrowed operations | Original checked requirements |

Missing the new `trap` effect is rejected at native compilation. A release
trap terminates the process; this profile makes no resumable-error or cleanup
after-trap promise. The observer's deliberate header corruption is a test-only
way to obtain real `munmap(...,0)` failure. Such corruption is not admitted HIA
source and does not qualify arbitrary foreign pointers.

## Bootstrap and explicit startup profiles

Restoring all owner logic produces 127 functions, 1,582 blocks and 8,229
instructions. The existing native cap is 8,192 instructions per compilation
unit, so direct compilation is rejected. No checker was deleted and no native
cap was raised. The compiler is partitioned into complete, unchanged functions:

| Unit | Definitions | External declarations | Blocks | Instructions |
| --- | ---: | ---: | ---: | ---: |
| Frontend | 63 | 42 | 1,086 | 5,640 |
| Backend | 64 | 0 | 496 | 2,589 |

Every function is present exactly once. The frontend depends on the backend;
each unit passes the same native source checks. The fixed borrowed compiler
builds initial ABI1 objects. A separate ABI1 compiler-startup linker supplies
the compiler's exact 2 MiB workspace. The resulting compiler emits ABI2 objects
for three subsequent generations, which must reproduce byte-identical native
executables. Stage0's ABI1 identity is recorded separately; matching stage0 and
ABI2 executable bytes is an observation, not a required semantic shortcut.

| Linker profile | Object ABI | Request magic | Entry workspace |
| --- | ---: | --- | ---: |
| Bootstrap compiler | 1 | `HIACMP01` | 2 MiB |
| Owned compiler | 2 | `HIAOWNC2` | 2 MiB |
| Owned runtime | 2 | `HIAOWN02` | 1 MiB |

Every startup retains 1 MiB input and output buffers. Native probes check
disjoint buffers, exact workspace lengths, last-byte access and one-past-end
traps. The runtime does not acquire a 2 MiB workspace implicitly. The linkers'
own input/output/workspace capacities remain 8/8/16 MiB.

## Scope of the qualification

The portable qualification records separate groups: one three-module 2 MiB+
lifecycle, 34 source/lifetime/allocation cases, eight release/startup cases,
13 direct allocator/ABI observations, 315 interface/object cases, and 14
independent triple-owner/CFG cases. The 315 group includes 204 existing cases
adapted only for the new request magic, ABI number, admitted alloc bit, and
report schema; it adds 111 checks for the new result fields. Reports disclose
these edits. They do not claim the old ABI1 alloc rejection still applies.

Repeated buffer and chain allocation/drop runs under a 64 MiB address-space
limit verify actual release. The direct helper observer checks overflow,
real allocation failure, zero fill, payload alignment, callee-saved registers,
stack preservation, NULL release and actual release-error trapping. Independent
source audit compares 24 owner/loan/result/effect checker ASTs with the frozen
borrowed compiler and finds them unchanged.

Current HeptaTorch source `264e9c5c…` and unhex source `f1baf503…` also compile
to byte-identical ordinary executables with the baseline expanded compiler.
Recompiling and linking both complete current runtime modules with ABI2 yields
the already qualified `cf78556c…` executable. That equality is recorded instead
of claiming an additional execution of its entire fixture corpus.

General storage IDs, reference counting, version identity, interior free
extents, arbitrary owner lookup, persistent owned state, and complete PyTorch
compatibility remain separate implementation work. Copying raw owner
descriptors to implement rollback is outside this contract.

## Source provenance

The adapters start from Hepta commit
`f33d8da316de8fdf27b504371f1a5310997ff548`, the exact support contract, and the
already qualified expanded-source adaptation receipt. They verify all loaded
upstream source files before and after authoring and first reproduce the
baseline source bytes. Generated full upstream-derived compiler/linker HIA is
kept in the build directory; this package carries source transformations and
new HIA authoring fragments, not an upstream source tree or executable blob.

The underlying semantics are in pinned upstream
`tools/hia_selfhost_owned_elf.py`, `tools/hia_selfhost_chain_elf.py`,
`tools/hia_selfhost_multi_results.py`, `tools/hia_selfhost_returned_loans.py`,
and the owner/loan/CFG functions of `tools/generate_hia_selfhost.py`.
`profile.json` pins generated sources, partition sources, allocator bytes,
compiler/linker executables and startup identities. `source-pins.sha256`
binds this package. Existing upstream attribution and license status are
retained; this package does not assert a new license for upstream code.
