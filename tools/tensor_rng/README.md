# Native Tensor / CPU Generator uniform component

This separate profile connects the frozen CPU MT19937 engine to the actual
owned numerical Tensor object runtime. Every generated value, Tensor write,
Generator state transition, Version bump, allocation and rollback runs in the
linked HIA executable. Python files author/check source or run qualification;
they are not a numerical or state runtime. The test-only C++ program supplies
an independent `std::mt19937` seed state.

The executable is **a qualified partial component**, not a completed PyTorch
replacement. It supports binary32 CPU uniform with explicit Generator, rank
0–4 bounded layouts, explicit untracked/no-grad mutation and whole-frame
transactions. Normal nonleaf `uniform_` autograd graph rewriting, normal and
dropout distributions, other Tensor RNG dtypes, CUDA/default-generator/global
locking and general PyTorch dispatch remain open. This standalone profile is
based on frozen Tensor09; it does not claim commands 57–59. A later common
runtime must preserve their separate grad-presence/release/partial-backward
semantics rather than treating two service processes as integration.

## Native qualification

The recorded standalone build has six actual native compilation units and a
1,249,280-byte syscall-only executable. Its SHA-256 is
`190148f50199d54d349cd2fe99eac29482d6956e3fa17a006b61411ff67c7c65`.
All 17 non-entry frozen RNG03 HIA function bodies remain byte-identical; their
internal byte helpers are module-private, so they cannot resolve accidentally
to Tensor object-address helpers. The Tensor baseline body map lists every
layout/entry change and every unchanged body.

The recorded model run passes eight groups / 729 live frames, with 33,193 exact
binary32 values, 160 complete 5,056-byte MT states, and 42 complete owned-state
rollbacks. Thirty groups of 1,024 native values match both an independent
integer FMA/RNE model and the full output SHA-256 values observed from the
fixed real PyTorch wheel. The frozen Torch diagnostic was executed remotely,
with all eleven provenance checks; this native qualification does not pretend
to execute Torch locally. Its actual result is included separately.

The entry/work suite passes 88 cases, including all output capacities 0–31
with empty and live state owners, real allocation refusal, response exhaustion
after trial writes, Version overflow, and an actual one-million-element fill
with every value checked. A 12-module qualification executable moves genuine
old/new HIA-created state owners between both full runtimes: incompatible tags
are rejected without changing the original pointer, length or any byte; the
correct runtime then resumes successfully. Both close exports consume the
owner and its mapping is absent before the process exits.

The first host authoring attempt hit the unchanged assembly parser's whole-file
line budget. The correction parses each exact function for host authoring and
validates the combined Core, then compiles bounded native units. No admission
check or cap was removed. Earlier candidate02 rejected an empty view whose
offset exceeded Storage; candidate03 explicitly represents this bounded empty
case and preserves zero remaining accessible bytes. Both observations remain
in `evidence/`.

## Rebuild and execute

Supply the already reproduced Tensor09 build, frozen RNG03 source, pinned
Hepta source checkout and frozen ABI3 stack-tool build. No upstream compiler
source is vendored or relabeled here.

```sh
python tools/tensor_rng/rebuild.py \
  --baseline build/tensor-objects \
  --rng-source tools/rng-cpu/rng.hepta \
  --upstream /path/to/pinned-hepta \
  --stack-build build/stack-tools \
  --output build/tensor-rng

python tools/tensor_rng/qualify.py \
  --build build/tensor-rng \
  --torch-observation tools/tensor_rng/evidence/pytorch-uniform-probe.json \
  --output build/tensor-rng-qualification

python tools/tensor_rng/test_entry.py \
  --build build/tensor-rng --stack-build build/stack-tools \
  --cpp-reference build/tensor-rng-qualification/std-mt-reference \
  --output build/tensor-rng-entry

python tools/tensor_rng/build_handoff.py \
  --baseline build/tensor-objects --build build/tensor-rng \
  --stack-build build/stack-tools --output build/tensor-rng-handoff
python tools/tensor_rng/test_handoff.py \
  --build build/tensor-rng-handoff --output build/tensor-rng-handoff-tests
```

These qualification tools use only their own child process for `/proc` memory
observations. The production wire never carries an owner descriptor or a host
pointer. `entry_probe.hepta` is a separate, explicitly test-only ABI observer;
it is not exported by the production session. Its fault-injected Version
counter is not a public state-import mechanism.

## Source and scope

`primary-sources.json` binds the official source paths, commit, URLs and file
hashes. The float32 distribution takes the low 24 MT bits, rounds the interval
width to binary32, uses the actual pinned CPU fused multiply-add pipeline, and
maps a rounded upper endpoint back to `from`. TensorIterator determines draw
order by physical/stride iteration, not logical row-major traversal. The fixed
wheel produced zero fused mismatches and 4,914 distinguishing split-operation
results across the recorded 30 ranges. Those facts are observations with exact
pins, not a claim that all possible PyTorch builds use the same contraction.

The seven-kind pool is constructed and consumed by this profile. Its audit
does not certify arbitrary foreign bytes as a valid Tensor graph or prove the
absence of every fabricated owner-state invariant. In particular, a private
Saved row with zero remaining bytes may have no standalone layout; emptiness
is established by its checked producer and graph/layout consumers. The public
wire cannot import arbitrary owner bytes.
