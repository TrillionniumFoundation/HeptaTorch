# Opt-in native F32 unsqueeze and alias lifetime

This canary extends the canonical typed-ReLU 775 source with a separate native
CPU command, `(129, tensor, signed_dim)`, returning a fresh Tensor handle.
Native HIA normalizes the signed axis and computes the stride geometry; the
existing view owner, Storage and Version machinery publishes the alias. No
element is copied. Request words 3 through 7 are reserved and must be zero.
The graph-free F32 input must have rank 0–3; output rank is 1–4. Signed axes
outside `[-rank-1, rank]` are rejected. The inserted stride is one at the end,
otherwise the original size times its stride (including zero-size and
zero-stride geometry), with overflow rejected.

The independently pinned Torch wheel and native process agree in 64 scoped
cases. These cover every positive and negative axis for 12 shapes/layouts:
scalar, empty, ordinary contiguous, transposed, nonzero-offset and zero-stride
views, including a singleton with a large legal stride. The independent owner
decoder reads the physical Tensor descriptor and the referenced Storage and
Version rows. It confirms shared identities, exact backing bytes, visibility
of an actual base mutation and shared version increment, view survival after
both public base/input handles are released, stale-generation refusal, and
restoration of the original physical live object counts on final release.
Two clean native builds and independent qualification runs must agree.

This does not implement differentiable unsqueeze, in-place unsqueeze,
arbitrary rank/dtype, or non-CPU layouts. It does not modify the existing
checked-in source, active image, default admission, or full-contract counts.
The public wire change exists only in the opt-in projection.

## Reproduce

From a checkout of this draft; provide the fixed toolchain and the pinned
Torch 2.15.0+cpu CPython 3.12 environment with the repository's untouched
wheel installation receipt (see the repository build guide):

```sh
p=tools/current_runtime/opt_in_unsqueeze_f32
w=$(mktemp -d)
python3 -B "$p/prepare.py" --output "$w/source"
for build in a b; do
  python3 -B tools/current_runtime/build.py \
    --manifest "$w/source/manifest.json" --source-root "$w/source" \
    --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" \
    --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/$build"
  "$HEPTA_TORCH_PYTHON" -s -E -B "$p/qualify.py" --repository . \
    --parent "$CANONICAL_TYPED_RELU775_ELF" --candidate "$w/$build/heptatorch" \
    --candidate-source "$w/$build/runtime.hepta" --output "$w/q-$build"
done
cmp "$w/a/heptatorch" "$w/b/heptatorch"
cmp "$w/q-a/qualification.json" "$w/q-b/qualification.json"
```

The canonical baseline refuses command 129 on all 64 inputs. Candidate
invalid-dimension, reserved-field, out-of-scope dtype/rank/grad and
transaction-rollback cases leave the published owner byte-exact. Six unrelated
parent/candidate response frames are also compared byte-for-byte on the old
view and mutation path. See [the evidence receipt](evidence.json).
