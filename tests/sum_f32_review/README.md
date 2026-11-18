# Explicit F32 Sum on the current HIA image

The fixed reader executes 17 comparison cases, four axis errors, seven current
dtype controls, ten entry checks in one session, and two allocation-control
sessions: 31 native children in total. The original cases, numerical
comparisons, and control expectations remain byte-exact.

The source proof executes three complete inverse layers: the exact cat
empty-gradient assignment, six AVX2-width constants in two reduction
functions, and the original added F32 Sum dispatch/function. Earlier failed
reports remain unchanged. The fixed 17-case Torch reference and all 29
original source/observation members are in `validation/sum-f32-reference17`.

Use `make test` or `tests/qualify_current_runtime.py`. The coordinator copies
this fixed reader into the fresh review directory and restores the exact
historical aggregate source needed by the inverse proof from
`validation/current-runtime-baseline`.

The allocation cases separately demonstrate failure to allocate cast Storage
and failure to allocate the result after a successful cast. Both retain the
actual command108 witness under the same resource limit, complete rollback,
restored limits, and successful retry of the same command115. Every raw frame
and complete owner boundary is retained. The native reader generates no
replacement expected tensor values.
