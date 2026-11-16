# Retained nonleaf gradients: fixed native profile

The new native profile is `HTRET001` / `HTORET01`, with 576-byte descriptors,
446 core functions, 5,922 blocks, 21,904 instructions and nine HIA modules.
Its source SHA256 is `666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b`;
its executable SHA256 is `7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5`.
Production operations, allocation, gradients and resource release execute in HIA.
Python coordinates fixed source composition, compilation, transport and independent tests.

## Build and qualify

Use the pinned stack tool directory already produced by the normal build:

```sh
make retained-gradients test-retained-gradients \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools
```

The parent is `build/gradient-adam-distributions`. The module compiler and session
linker are selected from `HEPTA_STACK_BUILD/native/`; their exact hashes are
checked by both the producer and the integration coordinator.

`retained-gradients` builds the fixed overlay into
`build/retained-gradients/tensor-retained-session`. `test-retained-gradients`
actually runs the cold producer and all independent retained-gradient, resource,
typed handoff, source and inherited extension qualifications. It writes
`build/hosted-retained-gradients-native.json` and the complete child-byte archive
`build/retained-gradients-ci-raw.tar.xz`, with its separate member manifest.
A failure preserves available report and raw evidence while returning the original
nonzero status. Successful native copying happens only after qualification.

The coordinator also has an explicit `--producer-review` option for verifying
and copying an already executed seventeen-step producer directory. It checks
all source/object/image pins, every child report and every original stream.
That mode does not run the producer again; all later independent runners still
execute. Original absolute paths and argv in copied reports are never rewritten.
The local first aggregate uses this declared reuse mode; the hosted Make target
uses the real cold mode.

## Native behavior and tests

The command protocol and object roles are defined in
`tools/retained_gradients/PROTOCOL.md`. Retaining a nonleaf gradient is a weak
request flag on the original node. A completed reverse hook materializes a real
public-gradient object. First write makes a contiguous copy; later contributions
make a fresh object with its own Storage and Version while old external aliases
remain alive. Captured forward edges and retained-gradient identity remain separate.

The frozen producer retains seventeen separately counted inherited steps. The
independent review adds twelve actual retained-gradient variants, the two original
Optimizer24 case14 subobservations, eight resource/OOM cases, two real typed owner
directions, a source review and five extension groups. The latter keep their six
commands separate because the extreme-stride observation and comparison are
independent steps. There is no sum claiming additional upstream API rows.

## Evidence identities

The original79/22/32/12 input manifests remain frozen. Root-level documentation
and handoff filenames have explicit published path relocations, with unchanged
bytes. The first integration source `1376074f…` and its actual reuse-mode PASS
`e9819c14…` are preserved. The current source `c71d40ea…` adds failure-evidence
preservation only; an exact success-path AST/byte comparison and a real wrong-pin
rejection are frozen under `validation/retained-integration-review/failure-custody-02/`.
The revised default cold path still requires actual hosted execution.

The original truncated resource archive remains incomplete. New root and
integration typed-handoff runs have their own complete original frame streams;
they do not fill in missing bytes of the historical execution.

## Hosted gates and remaining target

Native CI cold-builds this image and then runs the original live32 and frozen35
oracles. The additional quick profile restores the exact published native archive,
executes three real ownership controls and all eighteen original corruption rules,
then separately runs live32 and frozen35. Restoring the archive is not a new build.
The private-node observer's algorithm is unchanged; its new profile does not relax
Tensor flags or claim to observe arbitrary retained-graph workloads.

At creation of this package the 446 live32/frozen35 jobs have not yet run. The
successful 355/428 hosted record is separately frozen under
`validation/hosted-gradient-adam-distributions-af33/` and cannot qualify this image.
All full upstream schema and subsystem gates remain open, including hooks,
higher-order/forward AD, the eighteen distribution dtype/mask observations,
complete optimizer modes, broader devices and full frontend compatibility.
