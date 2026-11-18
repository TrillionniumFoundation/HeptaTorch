# Explicit F32 Sum on the current HIA image

The twelve fixed PR38 reader files execute 17 comparison cases, four axis
errors, seven historical dtype controls, ten entry checks in one session, and
two allocation-control sessions: 31 native children. Their original cases,
numerical comparisons, and control expectations remain byte-exact.

`fusion.py` runs this complete unchanged reader on the current HIA image. Its
original report remains FAIL because the two old F32/wrapped-F64-to-Bool
refusals now succeed. The outer reader accepts only those exact two observed
contract changes; every other original comparison and control must pass. It
then runs two independent pinned-Torch Bool comparisons from
`validation/bool-dtype-extension-reference`, including output bytes and
metadata, fresh ownership, result survival after input release, storage expiry,
and unchanged input/RNG state. The combined qualification has 33 native
children and retains the original failed report without rewriting it.

The source proof first reverses the exact current logical-reduction additions,
five changed function bodies, seven unreachable-body removals, and the module
header to reconstruct the full PR38 aggregate. It then executes all three
original inverse layers: the exact cat empty-gradient assignment, six
eight-lane reduction constants, and the original F32 Sum dispatch/function.
The original PR38 cat, reduction-order and F32-cast bodies are preserved.

Use `make test` or `tests/qualify_current_runtime.py`. The coordinator copies
the fixed reader into a fresh review directory and restores the exact
historical aggregate from `validation/current-runtime-baseline`. The fixed
17-case Torch reference and its 29 original source/observation members are in
`validation/sum-f32-reference17`.

The allocation cases separately demonstrate failure to allocate cast Storage
and failure to allocate the result after a successful cast. Both retain the
actual command108 witness under the same resource limit, complete rollback,
restored limits, and successful retry of the same command115. Every raw frame
and complete owner boundary is retained. The native reader generates no
replacement expected tensor values. These finite cases do not establish
entire PyTorch API or schema coverage.
