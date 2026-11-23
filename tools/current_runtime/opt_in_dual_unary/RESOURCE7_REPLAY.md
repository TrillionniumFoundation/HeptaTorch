# Retained resource7 on the exact joint image

This package re-executes the original seven resource recipes, including three
actual allocation failures/recoveries, on the exact joint candidate. The
original source recipes, comparison bytes, allocation thresholds, and complete
owner predicates stay unchanged. The original canonical admission still
rejects the joint image. A separate profile verifies all of the original
cold-compiler, linker, stack and custody predicates for the exact joint bytes.

The scoped source proof first reverses every changed joint function byte to
the exact 800-function canonical source and checks the forward reconstruction.
It then runs the complete original historical resource proof on that ancestor.
This is only a source lineage check; the ancestor ELF is not executed or
credited to the candidate. The original caller/owner relation is retained:
both changed entry and predicate dispatch guards add only equality to 127,
with the exact original branch arguments and remaining body. Every captured
original resource request excludes 127, in addition to the original exclusions
of unary opcode 3 and opcodes 115–119. The candidate resource route and its
actual changed entry/predicate body identities are bound separately from the
canonical source comparison.

The three actual worker children, all seven native child streams, and their
independent readback are retained. The independent reader uses the original
`qualify_resource7` function with only the source proof call redirected to the
joint-specific proof, with candidate identity globals. No capacity or numeric
test is removed and no prior pass is transferred.

## Reproduce

In a fresh checkout containing this package, build the exact joint source and
ELF as in [the dual build guide](README.md). Run:

```sh
P="$REPO/tools/current_runtime/opt_in_dual_unary"
python3 -I -B "$P/replay_resource7.py" --repository "$REPO" \
  --dual-build "$SCRATCH/dual-build" --dual-source "$SCRATCH/dual-source" \
  --dual-scoped "$P/qualification.json" --output "$SCRATCH/resource7"
```

Both cold builds of the same joint image passed all seven original resource
recipes, with 378 actual frames and 1,060 assertions per replay. The three
original allocation-limit restoration controls passed, and the independent
physical reader verified all seven native children and three worker
sessions. A canonical build, the wrong staged source profile and a different
scoped receipt are rejected before worker launch. Report hashes can vary
by path and process identity. See [compact evidence](resource7-evidence.json).

This replay does not activate the candidate, close any broader schema,
or claim the other 34 retained suites.
