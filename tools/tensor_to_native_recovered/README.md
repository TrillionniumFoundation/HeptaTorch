# Rebuild the frozen Tensor-to-copy 09b native image

This directory is a new, self-contained recovery recipe for the already frozen
09b HIA program. It reconstructs the twelve original module source files and,
when explicitly invoked with the pinned tools, compiles and links an image with
the original SHA-256. It does not restore the lost 5179 authoring package or the
a6f91 qualification coordinator.

## Inputs and identity

The four files in `inputs/` are immutable recovered bytes:

| Input | SHA-256 |
| --- | --- |
| Complete 614-function runtime | `4f8f73f463f681b5eb29423d71a35a0b42d807fd92bf976bbf76c1970fbcee33` |
| RNG module | `a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3` |
| Owned-entry module | `9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884` |
| Original build report | `9913ccb35c690f1a8795479ddd468531a966df71c375fcb91ed22d9f3c887557` |

The original report is retained without rewriting its historical absolute paths.
Those paths are evidence, not dependencies used by this recipe. The installed
RNG and owner source files have Git mode 0644; the separate recovery receipt
records that their ROG transport copies had mode 0600. Their bytes are unchanged.

The target has 614 runtime functions, 8,078 blocks and 29,033 instructions,
672-byte descriptors, and state tags HTTOCP08 / HTOTOCP8. It links twelve modules:
one RNG module, ten runtime modules and one owned-entry module.

## Pure source check

From a checkout containing this directory, with Python 3.9 or later:

```sh
python3 -B tools/tensor_to_native_recovered/self_check.py \
  --output build/to-native-recovered-source-check
```

No compiler, linker, existing executable, external Python package, upstream
checkout, old candidate directory or Cloud path is needed for this command.
It writes the twelve reconstructed `.hepta` files, the full runtime, the
unchanged original build report under `reference-original-build.json`, and
`source-check.json`. The output directory must not already exist.

The check verifies package membership, file modes and hashes; exact coverage of
all 614 runtime bodies; unique definitions and resolvable calls; deterministic
sorted-call depth-first topology; the original grouping limits of 3,600
instructions, 1,000 blocks and 65 definitions; at most 256 definitions plus
imports per module; and all twelve source hashes and counts from the original
build report. It records each body-to-module relation.

This is a byte reconstruction and source-integrity check. It does not claim
native admission of the aggregate runtime or successful execution of any
Tensor operation.

## Native rebuild

On Linux x86-64, provide these two actual native ELF files:

| Tool | Required SHA-256 |
| --- | --- |
| Module compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Owned-session linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |

```sh
python3 -B tools/tensor_to_native_recovered/rebuild.py \
  --compiler /absolute/path/stack-module-compiler \
  --linker /absolute/path/stack-linker-session \
  --output build/tensor-to-native-recovered
```

The recipe checks both tool hashes, runs the module compiler separately for each
complete source module, checks each resulting ELF object against the original
build record, then sends the original owned-session `HIASTKS1` request with entry
`step` to the session linker. Imports and exports remain the real original
interfaces. The native compiler and linker perform their ordinary module checks
and whole-graph stack checks.

Successful output includes `runtime.hepta`, all twelve source/object pairs,
compiler stderr files, the exact link request and streams, a **new** `build.json`
with this recipe's invocation identity, and `tensor-to-copy-session` (mode 0500).
The final executable must be 3,129,344 bytes with SHA-256:

`53fb43b869858f020c0b936a65f00f3b14b97d9e37038a74c357e1da471fe083`

The original build report is never relabelled as the new invocation. Every
native tool process has recorded argv, started/completed state, return code and
raw streams. Errors retain partial files and a FAIL receipt; there is no retry
or fallback to a cached object or executable. Inputs and native tools are
rechecked after successful execution.

## Scope and provenance

The reconstruction algorithm follows the separately reviewed
`restore_parent09b.py` (SHA-256
`bbdfebfd63c7fce507f24cb02fb473222de2373ea270451ef9901de6c59e93ed`).
A copy is retained under `provenance/` for comparison only; it is not imported
or executed by this recipe. This entry uses only Python's standard library and
the four bundled inputs.

A successful rebuild proves reproduction of the pinned source modules, object
files and executable. It does not rerun the original live32, full35, ownership,
dtype, autograd or other semantic suites, and does not convert any historical
failure or source-only receipt into a new PASS. Project status remains **HOLD**,
with 0 closed API schemas out of the existing denominator 2,590 and
`full_pytorch_rewrite_complete=false`.
