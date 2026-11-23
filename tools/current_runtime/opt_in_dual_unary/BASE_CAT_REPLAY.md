# Retained base270 and Cat4 on the exact joint image

The canonical source and gates are unchanged. This scoped wrapper checks that
the original canonical admission still refuses the opt-in ELF because its
source manifest is different. It then binds the candidate using the exact
dual-image checker. A separate in-memory copy of the complete original
cold-build verifier changes just the manifest path to the candidate profile.
It reruns the original source-partition, compiler/linker, object and stack
checks for the candidate. Both the canonical refusal and this result are
checked at each original observer admission point.

The wrapper uses the exact observers, historical archives, per-case controls
and independent physical readers. Only their admission identity changes.
The base270 independent reader uses the separately admitted candidate
profile in memory, with all numeric, transport, and provenance predicates
unchanged. The Cat4 independent readback invokes the same full candidate
admission used by its producer before independently recomputing the original
source, wire-response, and owner evidence. No canonical or original source
is edited. The source catalog binds 28 exact checkout blobs verified against
the #63 tree; the original harness also checks its archived data and
frozen legacy reference bindings.

## Reproduce

Use a fresh checkout containing #63 and this wrapper. Create the full
candidate source and cold native build as in [the joint build guide](README.md).
Use the fixed Torch CPU wheel interpreter and `build/pytorch-install.json`
from the official locked install used by Current HIA Runtime. Run:

```sh
P="$REPO/tools/current_runtime/opt_in_dual_unary"
for suite in base270 cat4; do
  "$TORCH_PYTHON" -I -B "$P/replay_base_cat.py" --suite "$suite" \
    --repository "$REPO" --dual-build "$SCRATCH/dual-build" \
    --dual-source "$SCRATCH/dual-source" --dual-scoped "$P/qualification.json" \
    --output "$SCRATCH/retained-$suite" || exit 1
done
```

| Retained suite | Cases | Physical frames | Independent result |
| --- | ---: | ---: | --- |
| base270 | 270 | 284 | 13,076 actual Torch comparisons, 14 same-process success/refusal pairs, 11 live oracle provenance checks |
| Cat4 | 4 | 4 | Three byte-exact original success responses, one original expanded-capacity control, eight physical owner snapshots (8,038,976 raw owner bytes) |

Both passes were independently observed on each of two cold builds of the
same joint ELF. The original capacity failure is still retained by name:
the capacity case is not claimed to reproduce that failure. A canonical
ELF, the wrong candidate profile, or a different scoped receipt cannot start
the observer. See [retained evidence](base-cat-evidence.json) for the two
replays and the fixed input identities.

The raw reports and wire/owner evidence remain in the output directory;
per-run report hashes can vary with paths, timing, and process metadata.
These are two suites, not the full 35-suite requirement or activation of
either opt-in operator.
