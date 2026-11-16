# Native persistent CPU Generator objects

This component implements a bounded, persistent CPU Generator service in native
HIA. The executable owns a heap buffer containing up to 1024 independent
Generator records, preserves it across framed requests, and releases it through
the checked HIA `close` export. Numerical generation, state management, cloning,
validation, transactions and reclamation all execute in native code. Python is
used only to author source, invoke the native compiler/linker, and qualify the
result. The C++ and CPython generators are independent test references.

The 17 non-entry function bodies from frozen `tools/rng-cpu/rng.hepta` are reused
byte for byte. Their source pin is
`c1de63f5f046bfb20aa4d4f65cbffa6217a38a36c0dc63b05f08c7ddc7612032`.
The native ABI3 compiler continues to enforce its original resource, effect,
alias and whole-program stack checks. No old compiler, linker, session profile,
RNG source or numerical expectation is modified.

## Reproduce and qualify

The existing stack-budget build must already have been reproduced. Its `native`
directory supplies `stack-module-compiler`, `stack-linker-linker`,
`stack-linker-runtime` and the original `stack-linker-session`; its `sources`
directory supplies the pinned original linker/startup authoring. The Hepta
checkout must be the pinned `f33d8da316de8fdf27b504371f1a5310997ff548` source.

```sh
python tools/cpu_generators/rebuild.py \
  --upstream /path/to/pinned/Hepta \
  --stack-build /path/to/reproduced/stack-budget-build \
  --rng-source tools/rng-cpu/rng.hepta \
  --output build/cpu-generators

python tools/cpu_generators/qualify.py \
  --build build/cpu-generators \
  --output build/cpu-generator-qualification

python tools/cpu_generators/test_entry.py \
  --build build/cpu-generators \
  --stack-build /path/to/reproduced/stack-budget-build \
  --output build/cpu-generator-entry-qualification
```

The qualification scripts require Linux x86-64, GNU C++, CPython, permission to
observe their own child through `/proc`, `prlimit`, and the ability to install a
seccomp filter on their own child. They assert failures instead of silently
skipping an unavailable kernel observation. The production executable uses only
Linux syscalls and HIA-generated code, with an empty `PATH` and no libc or Python
dependency.

The reproducible candidate is pinned in `profile.json`. In particular:

| Artifact | SHA-256 |
|---|---|
| Generator executable | `ba344748d42f0def1cd5d88e5e27232a6dfa08e33ef4c42e6f00486f3ca08572` |
| New entropy session linker | `761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e` |
| Linker HIA source | `7bba866ece4f33fb6b1973e33475d868376f2b804d5ce81116cf206cfff31c34` |
| Native ABI3 compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |

## Ownership, transactions and limits

The persistent state is one actual owned `buffer`, initially live-empty `(0,0)`.
Its first successful construction allocates a 64-byte header and one 5088-byte
record per live Generator. Each record stores a handle and the complete 5056-byte
RNG state; it contains no native pointer or embedded owner descriptor. The largest
state is exactly 5,210,176 bytes. Handles start at one, increase monotonically,
and are never reused, even after every Generator has been released.

A mutating request validates its packet, handle, output budget and arithmetic,
allocates a separate trial owner, and commits by dropping the previous owner and
returning the trial owner. `set_state` validates and normalizes into a separate
5056-byte scratch owner before modifying the trial. Any normal application error
returns the original owner and a status-only response. Reads, zero-count draws
and zero-count advance preserve the same state and epoch.

Release copies the surviving records into a smaller owned allocation, fills a
middle hole with the last live record, and drops the old allocation. A live
process `/proc` observation verifies that the old owner address is unmapped after
release. Releasing the last object retains the 64-byte identity header so that
stale handles remain invalid. Because release uses the same transaction scheme,
it can return allocation failure; that error leaves the original object alive.

The Core language has no borrowed slicing opcode. Therefore a selected record
is copied to the real scratch owner to call the frozen RNG functions, then copied
back at commit. Neither its pointer nor its size is fabricated from packet data.
Borrowed views are ended before ownership is transferred or dropped.

The caps are 1024 live Generators, 100000 returned values per request, 1000000
engine words per request, and a 1 MiB request/response body. Uniform float64 and
raw64 consume two words; int64 ranges at least `2**28` wide consume two words.
These limits are checked before generation. The ABI3 linker separately admits
the actual call graph against the guarded 8 MiB private stack. The existing
per-unit compiler and module-count limits are unchanged.

## Nondeterministic `seed()` and its distinct entry ABI

The pinned PyTorch `CPUGeneratorImpl::seed()` obtains nondeterministic bytes,
sets the current seed, and returns it. A default constructed CPU Generator uses
the pinned constant `67280421310721`; the process-global default Generator has
different lifetime and initialization semantics and is outside this service.
See the exact upstream files and SHA-256 values in `primary-sources.json`.

The new entry profile is explicit:

```text
step(readbuf input, mutbuf output, buffer state, readbuf entropy)
    -> (buffer state, u32 response_bytes), effects 15
close(buffer state) -> (), effects 9
```

The native startup obtains eight bytes with Linux x86-64 `getrandom` syscall 318
for operation 3. It supplies a private read-only entropy buffer in parameter slot
3 and clears the bytes and dead slots after the call. It checks errors and zero
progress, retries `EINTR`, and accounts for any partial progress. With flags zero,
Linux uses the same urandom source as `/dev/urandom`; an initialized pool supplies
an eight-byte request without short reads or signal interruption. This behavior
is documented by [getrandom(2)](https://man7.org/linux/man-pages/man2/getrandom.2.html).
The wrapper still checks the return value. An unavailable syscall or another
normal entropy error supplies a zero-length view, which HIA rejects with status
9 and no Generator state change. A real seccomp-induced `-EAGAIN` proves this
failure path. Entropy acquisition can block during initial kernel entropy setup.

The linker request magic is `HIACGS01`. The native admission check requires the
exact four parameters, result order, effect mask and `close` contract. The old
three-parameter owned session profile rejects this service, and the new profile
rejects three parameters, a mutable/scalar/owned entropy parameter, an extra
parameter, wrong results, wrong effects, or missing close. The remaining 41
linker functions have identical normalized ASTs to the frozen ABI3 linker;
`source-audit.json` records that comparison. This is the same producer-bound
native code/metadata qualification boundary as ABI3, not certification of
arbitrary machine code.

The caller can supply only framed bytes. It cannot supply a native pointer,
owner descriptor, private entropy buffer, workspace, or stack address. Normal
EOF and transport failures still converge on the existing single `close` path.
As in the frozen owned-resource profile, an unrecoverable HIA trap, failed
`munmap` trap or external fatal signal is not an application-error rollback or a
claim that cleanup returned successfully.

## Qualification and remaining work

The native numerical/object qualification compares against an independent C++
standard MT19937 initialization plus CPython's MT engine. It covers all modes,
twist boundaries, signed int64 ranges, enabled normal caches, truncation and
padding normalization, cloned-state independence, reseeding, system entropy,
maximum output/work, 1024 objects, full-pool rollback after errors, actual heap
allocation failure, recovery, and real unmapping after interior release.

The separate native entry harness passes real owned buffers through the unchanged
service functions. It proves short-output behavior, internal short-state errors,
the last valid handle, handle and epoch exhaustion, exact entropy length and
entry-profile incompatibilities. It is a qualification program; its ability to
construct synthetic internal states is not part of the public service.

The tests keep two discovered failures explicit. The first source-generation
attempt failed the host Core verifier because its verification document omitted
the frozen RNG callees; adding their exact complete function bodies to that
verification closure fixed it. A first OOM test guessed a fixed 20 MiB process
limit, but the allocation actually succeeded. The replacement test measures the
live address space and applies a real `prlimit` budget too small for the trial.
Neither issue was a numerical failure or a relaxed expectation. A subsequent
direct-ABI review additionally hardened statistics reads for an internally
constructed owner shorter than 64 bytes, then reran the entire numerical suite.

This service is a partial PyTorch component. It has not implemented normal
sampling, dropout, all distributions, global/default Generator bindings,
multithreaded shared Generator mutex semantics, CUDA Generators, or arbitrary
Tensor destination integration. The state importer intentionally accepts the
canonical/reachable MT position subset supported by frozen RNG03, rather than
claiming all of PyTorch's malformed-state acceptance behavior. Actual fixed
PyTorch object-oracle qualification is a separate gate; these local CPython/C++
receipts do not claim to be that run.

The current Tensor runtime owns one buffer whose objects contain relative
offsets. A future Tensor integration must place the 5056-byte RNG state inside
that controlled pool, or use a separately checked multiple-owner transaction.
It cannot insert a foreign Generator owner descriptor into the pool and then
clone it as bytes. An eventual fill operation must validate the destination
extent and version, perform RNG advancement and Tensor writes in the same
transaction, and roll both back on failure. No Tensor command number is reserved
or modified by this component.

## Source attribution

The generated RNG module comes from the already frozen HeptaTorch RNG component.
Its MT19937 source notice and pinned PyTorch source attribution remain applicable.
This package supplies its own HIA service authoring and deterministic adaptations;
it does not vendor the complete upstream Hepta compiler or assert a license for
that upstream project. All upstream authoring modules are verified against the
pinned support contract before import and again after generation. Native tool
and generated-source identities are checked before the build is reported as a
pass.
