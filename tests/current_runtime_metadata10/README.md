# Ten cat metadata regressions

This required group replays the ten fixed CPU F32 cat recipes from the observed
PyTorch report `2b80fe6a…`, including the 62,207 and 62,208 repeated-empty-input
boundaries, rank-one empty inputs skipped by cat, rank-two empty tensors,
negative or otherwise unusual dimensions from the original recipes, and the
one literal nonempty input. It compares the original shape, strides, offset,
number of elements, dtype, element size, storage extent, logical bytes,
requires-grad, leaf status, presence of history, Tensor version, and recursively
observed gradient metadata. Warning text and the names of gradient classes are
outside this fixed comparison. The old source03 report retains its seven
recipes with eighteen version differences; its values are never offset or
relabelled as successful.

The predecessor twelve groups and their 687 cases are unchanged. This group
adds ten required cases, for thirteen groups and 697 cases on the same admitted
cold image. A focused success does not establish full PyTorch schema or project
closure. The project-wide gate remains HOLD.

## Execution and source identity

`frozen/native_v2.py` preserves the original reader, including startup
synchronization, literal recipes, original assertions, comparison function,
input binding, transport, and cleanup. The worker derives its executable source
by replacing exactly the two old source/image hash literals. The complete
inverse must recover the original bytes. The shared current build admission
checks the actual source, manifest, all objects, tool identities, complete cold
process streams and stack evidence. The expected build-receipt SHA is an
explicit command argument, so a new cold build can have its own real receipt.
The metadata source manifest fixes the selected source and executable image.

Example invocation, after constructing that exact current cold build:

```sh
python3 -I -S -B tests/current_runtime_metadata10/worker.py \
  --build-directory /absolute/current-build \
  --expected-build-sha256 ACTUAL_BUILD_JSON_SHA256 \
  --repository /absolute/repository \
  --output /absolute/new-metadata10-result
```

The common coordinator supervises this command with
`tests/current_runtime_process.py`, which records the actual argv, PID, both
complete streams and EOFs. The pure qualifier requires those records in its
separate execution directory. All four worker options, the isolated Python
flags, the actual worker script, the physical interpreter identity and the
worker's own original argv must agree. `--prepare-only` restores and validates
inputs but does not run native cases and cannot satisfy the pure qualifier.

## Passive observations of the original methods

The worker observes only the exact `Native.tensor` and `OwnerSession.finish`
code objects through `sys.setprofile`. It verifies the entire fixed function
source, the compiled code bytes and identity with the original class method.
Every accepted event must be a normal return instruction. Exception returns,
unmatched calls and callback errors reject the group. No method or syscall is
replaced, no native command is added, and reference values are not passed into
the observer. Callback time is recorded separately as evidence overhead.

At every successful Tensor metadata read, the observer saves the complete raw
owner snapshot already read by the original method, all 84 descriptor words,
the contemporaneous public token, returned metadata and the exact three-frame
interval. This includes temporarily published gradients. It never substitutes
a token from a previous getter or assumes a public generation survives release.
At finish entry it records three distinct, actually open transport descriptors
and their `fstat` identities. At normal return it retains the original empty
stdout-tail and stderr read results, the real process exit and closed transport
files, and verifies that each same entry descriptor now returns EBADF.

## Independent pure qualification

`qualify_metadata10(report, repository, build, build_sha, directory, inputs,
execution_directory)` is the coordinator interface. It performs no native,
Torch, compiler or linker execution. It physically rebinds the original
95-input census, the worker's own inputs, the actual interpreter and the whole
cold build. The fixed original owner package and its identity-only adaptation
are independently reconstructed; every original owner predicate remains.

`readback.py` consumes every saved request and response, checks the exact outer
framing and before-stdin-EOF ledger, and reconstructs each recipe with the
original packet codec. It uses the actual 13/96/98 frames and their same-time
raw owner to recover metadata, all 84 descriptor words, Tensor/Storage/Version
relationships, and logical element bytes copied from the observed storage.
It then reproduces each 91/59 gradient-getter sequence and conditional 41
release. A temporary gradient is checked under its own current public token;
its private relationship must survive the release. All original labelled owner
snapshots and every saved owner archive must be used and verified.

The unchanged original comparator then compares this reconstructed actual
metadata with the fixed Torch observations. Numerical or version values are
never taken from the report being checked. The pure reader requires all ten
cases, ten distinct native PIDs, every original frame and assertion count,
and twenty actual native EOF observations. It also binds the outer worker's
complete streams and its normal exit.

A standalone pure readback uses:

```sh
python3 -I -S -B tests/current_runtime_metadata10/qualify.py \
  --repository /absolute/repository \
  --build-directory /absolute/current-build \
  --expected-build-sha256 ACTUAL_BUILD_JSON_SHA256 \
  --result-directory /absolute/completed-metadata10-result \
  --execution-directory /absolute/completed-metadata10-execution \
  --output /absolute/new-pure-readback.json
```

## Reference and interpreter custody

`validation/cat-metadata10-reference` contains fifteen exact original members:
the observed report, original producer, four support inputs, four original
packet fixtures, the original reference interpreter entity, public execution
receipt, staging receipt and both streams. All ten original absolute input
paths map one-to-one to physical members with matching bytes. All eleven
original oracle checks, complete original stream bytes and EOFs are checked.
Original absolute paths and reports remain unchanged. Reading this portable
package is explicitly different from rerunning the original path-strict Torch
producer. The stored interpreter is not executed.

The CI archive separately copies its own actual resolved interpreter as an
ordinary single-link file with a source-path/SHA/size/mode receipt. It checks
that identity against this worker's actual report when available. This current
execution interpreter and the historical reference interpreter retain separate
origins even if their bytes happen to match. The archive also explicitly keeps
`tools/stack_budget/inspect_stack.py` and the compact76 archive-location record,
which are required by existing pure source and resource custody checks.
