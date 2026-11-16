# Mixed F32 / I64 autograd, revision 10

This producer extends immutable I64 revision 07 with genuine mixed add, sub,
mul and true-division differentiation and original Saved Tensor access.
Production arithmetic, conversion, Tensor/Storage/Version ownership, graph
execution and allocation run in HIA. Python authors, builds and observes it.

The locked source is `5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49`;
the ELF is `69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6`.
There are 556 core functions, 7,420 blocks, 26,889 instructions, 11 actual native
modules, 640-byte descriptors and new `HTMXAD02` / `HTOMXD02` owner formats.
The revision-10 protocol is explicit about two independently observed fixes.
Revision 09 released actual Saved Tensor roots in the guarded gradient-detach
path. Revision 10 additionally releases an I64 view's ultimate-base reference
when its real Tensor row is destroyed, while preserving any graph metadata
reference and shape/layout descriptor. A live view Tensor row must have its
base role; a rowless metadata descriptor must not. The new revision-02 format
isolates that invariant from prior owners. The 09 numerical cold gates remain
preserved alongside its later independent 13/14 public-Saved lifetime failure.


## Reproduce

With a real I64 07 parent build and the repository's pinned authoring/toolchain
inputs, run the following from a normal checkout. All paths are explicit; no
workspace-specific override or existing mixed executable is used.

```sh
python -B tools/mixed_integer_autograd/qualify.py \
  --parent build/integer-tensors/native \
  --repository . \
  --upstream build/hia-toolchain-cold \
  --compiler build/stack-canonical/native/stack-module-compiler \
  --linker build/stack-canonical/native/stack-linker-session \
  --flat-compiler build/compiler-expansion/compiler-1 \
  --output build/mixed-integer-autograd
```

The caller may supply the actual pinned compiler directories used by its build.
The parent is verified as source `adf6cae6…` / ELF `b7324639…`. `rebuild.py`
executes the author and every native compiler/linker invocation anew, requires
all 11 unit sources and objects plus the full source/ELF to match the locked
profile, and records process-start/completion separately from test outcomes.

The five producer gates are:

1. New cold native construction, all 11 exact modules.
2. All 36 immutable real-Torch mixed-autograd observations, exact values,
   layouts, identities, Versions, None state, Saved behavior and partial errors.
3. The eight existing I64 gates: original 54 split into storage/arithmetic/
   dropout, alias8, integer-to-F32 RNE1237, promotion20, true-view lifetime6 and
   the independent inplace40 reader.
4. The twelve legacy 523 gates, including its sixteen inherited suites.
5. The independent fourteen public Saved lifetime recipes, using the frozen
   original reader and reference. This explicitly covers a held original
   integer view across default backward and final getter release.

The independent fourteen-case package is included under its original
`tests/saved_public_lifetime_review` and `validation/saved-public-lifetime`
namespaces with manifest `144b0cfc7e11787581614403cbc2b3e643d4650846582440a10c676f9e072661`.
All seventeen payload files and the manifest retain their exact bytes and modes.

The gates overlap. Their counts are not summed into unique upstream APIs.
Existing partial classifications and explicit state lifecycle differences in
old reports remain unchanged. Native execution of original-Torch observations
is not a new local execution of Torch.

## Ownership and errors

Command 103's differentiable F32/I64 paths save the actual original arguments,
not temporary promoted F32 copies. A private 32-byte Storage and Hold keep the
two `(node, incarnation)` references; repeated slots retain/release separately.
The existing actual Saved S/V/offset/counter still determines mutation errors.
Command 107 publishes the true saved input when present; absent, spent and
version-mismatched results retain their distinct contracts. Getter observation
can create a public token and must release it if only primitive metadata is held.
Command 58 keeps its established partial-commit semantics. Command 104 continues
to reject a requires-grad input; integer Tensor creation cannot require grad.

The frozen 36-case reader is copied without any numerical or identity changes.
The inherited readers only change necessary private tag/stride/source bindings;
exact inverse receipts accompany them. The ordinary zero-grad cleanup failure
of revision 08 is kept separately from its successful 36 numerical observations.
The fix does not relax audit53 or the original gradient detach guard.

## Remaining integration evidence

The standalone producer result does not replace the independent general-I64
owner checker, mixed Saved-root extension, resources and OOM controls, output
capacity checks, previous-profile typed handoff, live32, or a correctly extended
accounted35 observer. The previous
07 F32-only fixed35 checker assumed the descriptor's new role tail was zero;
that condition cannot be reused for new real Saved Tensor roots.

No higher-order gradients, Python exception-message frontend, arbitrary Python
wrapper reference counting, full 2,590-schema rewrite, or all-system-gates closure
is claimed. These remain separately bounded by the project's global ledger.
