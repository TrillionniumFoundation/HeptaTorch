# Fixed-wheel indexed-forward observations: 42 cases

This is an input-only, reference-only producer. It has not run Torch or a
native HIA image during preparation. The existing wheel is selected explicitly
by the caller. The original eleven oracle provenance checks are mandatory:
Torch 2.15.0+cpu, commit d0f2af8b66b20da4baa50644c0cd3f887a2eae2e and locked
wheel SHA0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3.

## Literal scope

There are 42 unique cases: sixteen general no-grad scatter/index additions,
two with transposed inputs, two index alpha cases, the unchanged old
125000-index empty recipe, six gradient-flag combinations, six nonleaf or
changed-history/view cases, eight actual error observations, and one old cat
capacity recipe. Rank0 through rank4, repeated indices, negative axes,
empty indices, an empty nonaxis extent, and identical self/source are explicit.
The exact values, layouts, graph construction and calls are in `cases.json`.

The two original legacy input hex files are retained verbatim. The producer
checks their entire command lists and payloads. The large index array is the
original 1000000 payload bytes. The cat recipe creates all255 original empty
leaves, retains them, then calls cat on the first one. No historical expected
output or status is loaded or rewritten. Raw opcode11 is not mapped to Torch.

The six advanced gradient cases use two independent requiring-grad leaves.
They clone each leaf; two cases use these nonleaves directly. Two retain an
earlier multiplication branch before changing the nonleaf bases in place.
Two create transposed views, retain earlier multiplication branches, then
update each nonleaf base in place. The later indexed operation consumes the
actual current tensors. A supplied literal seed drives backward through that
result. Earlier branches stay alive and their values/metadata are observed
before and after; this does not claim a separate backward through those
earlier branches.

## Evidence and exception boundaries

Only real Torch calls supply output and gradient bytes. All observations keep
logical and complete storage blobs, shape/stride/offset/version, dtype,
leaf/view/grad_fn, actual Tensor/Storage identities, aliases and gradient
bindings. Forward must preserve every input. Backward may change gradient
bindings and bytes, while all primal values and metadata remain unchanged.
Full5056-byte CPU RNG state is bound and must stay unchanged.

Only the operation call is in its operation-error try block. Snapshot,
storage, custody or comparison failures propagate to a failed producer.
Backward has a separate operation-only error record. Observed operator
refusals are reference facts, not a compatibility PASS for HIA.

The launcher retains the requested venv Python spelling in argv. Resolving
the venv symlink for execution would change environments. It separately
binds the resolved executable's actual bytes/SHA/size/mode before launch and
stores an ordinary copy, writes exact argv intent, uses the unchanged36e
process supervisor, and verifies actual PID/start ticks, physical stdout and
stderr, exit0, both EOFs and all producer/report/blob inputs after execution.

The worker rechecks its running `/proc/self/exe`, `sys.orig_argv`, executable
entity and pre-execution intent. It forbids child processes, so it cannot run
HIA, compiler tools, install packages or create an unrecorded producer. It
retains the original eleven checks and binds installed Torch metadata and C
extension bytes. It does not claim to reread an unavailable wheel archive.

## Invocation

Use the exact manifest SHA in the delivery handoff:

```sh
/path/to/supervisor-python -I -S -B /absolute/producer/produce.py \
  --python /absolute/fixed-torch-venv/bin/python \
  --repository /absolute/staged-repository \
  --output /absolute/fresh-indexed-reference42 \
  --expected-producer-sha256 MANIFEST_SHA256
```

The repository must supply the three original pinned oracle support files,
the actual current installation's `build/pytorch-install.json`, and unchanged
`tests/current_runtime_process.py`. Nothing is installed by this producer.
The complete `production.json`, `execution/` and `reference/` are the result.
Do not promote a RUNNING/FAIL or truncated stream to a complete reference.
No current/runtime/whole-project PASS is transferable from these observations.
