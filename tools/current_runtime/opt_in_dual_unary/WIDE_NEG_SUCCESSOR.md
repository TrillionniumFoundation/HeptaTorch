# Exact joint wide-neg replay and two unary-7 successors

This independently runs **wide_neg** on opt-in joint ELF `5af79385…`, separately
from the canonical current800 ELF `083a92ad…`. It changes no canonical
sources, suite roster, defaults, or historical physical observations. The
baseline's identical I64 and F64 unary-7 requests still return status 7 with
no records. The opt-in joint image deliberately implements abs for those
requests; the original predicate therefore fails in exactly cases 403 and
404 when run unchanged after separate joint-image admission.

The added launcher pins 31 exact source files. The original admission must
deny the joint executable at its original manifest check. The separate
admission checker then binds the exact joint build and its existing scoped
receipt. The candidate-specific comparison substitutes one reversible
branch in memory for operation 7 only, retaining the original 405 cases,
the numerical negation observations and shared actual typed-ReLU reference,
and all other predicates. The same two original packets are then accepted
with status 0 and opcode 3; their raw output words and complete metadata
match `torch.abs` on the same original fixtures through the locked Torch
2.15.0+cpu AVX2 oracle. Each result has fresh Tensor/Storage/Version and no
input, view or autograd retention edges. All input descriptor words and
values are unchanged. The original `qualify` readback repeats the full
physical comparison including the two explicit successors. All 403 other
full per-case records match baseline.

| Exact native run | Result | Frames | Checks |
| --- | --- | ---: | ---: |
| Baseline, original reader | PASS | 10,522 | 37,490 |
| Joint, original predicates | FAIL at the two historical unary-7 refusal checks only | 10,452 | 37,250 |
| Joint, explicit successors | PASS plus physical readback | 10,536 | 37,556 |

The candidate's readback covers 405 native sessions, 46 protocol controls
and two real allocation refusals with identical-request recovery.
[Compact evidence](wide-neg-successor-evidence.json) binds both probe
request/response frame digests, the original reports and supplemental
PyTorch observation. Full reports and wire transcripts are produced under
the requested local output directories. Some report hashes change across
physical runs because reports include paths, process identities or timing.

## Reproduce

Start from the #63 checkout with the [dual candidate build](README.md) and
its exact canonical comparison build. Follow the original Current HIA
Runtime workflow for installing the locked Torch reference. Use absolute
`REPO`, `BASE_BUILD`, `DUAL_BUILD`, `TORCH_PYTHON`, and existing output
parent `OUT`. Create the shared typed-ReLU reference with the original
process supervisor:

```sh
python3 -I -B - "$REPO" "$TORCH_PYTHON" "$OUT" <<'PY'
from pathlib import Path
import importlib.util
import json
import sys
repo, torch, out = map(Path, sys.argv[1:])
p = repo/'tests/current_runtime_process.py'
spec = importlib.util.spec_from_file_location('wide_neg_process', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
execution = out/'relu-execution'
execution.mkdir()
r = m.execute([str(torch), '-B', str(repo/'tests/typed_relu/reference.py'),
               '--output', str(out/'relu-reference')], execution, 300, cwd=repo)
(execution/'process.json').write_text(json.dumps(r, sort_keys=True, indent=2)+'\n')
assert r['exit_code'] == 0
PY
S="$REPO/tools/current_runtime/opt_in_dual_unary"
BASE_SHA="$(sha256sum "$BASE_BUILD/build.json" | cut -d' ' -f1)"
python3 -I -B "$REPO/tests/wide_neg/run.py" --build-directory "$BASE_BUILD" \
  --expected-build-sha256 "$BASE_SHA" --torch-python "$TORCH_PYTHON" \
  --output "$OUT/wide-neg-baseline" --typed-relu-reference-directory "$OUT/relu-reference" \
  --typed-relu-reference-execution "$OUT/relu-execution"
COMMON=(--repository "$REPO" --baseline-build "$BASE_BUILD" \
  --baseline-run "$OUT/wide-neg-baseline" --dual-build "$DUAL_BUILD" \
  --dual-scoped "$S/qualification.json" --torch-python "$TORCH_PYTHON")
code=0
python3 -I -B "$S/replay_wide_neg.py" "${COMMON[@]}" \
  --mode retained --output "$OUT/wide-neg-retained" || code=$?
test "$code" -eq 2 || exit 1
python3 -I -B "$S/replay_wide_neg.py" "${COMMON[@]}" \
  --mode successor --retained "$OUT/wide-neg-retained/observer" \
  --output "$OUT/wide-neg-successor"
```

Exit 2 intentionally retains the old predicate's nonzero result; unexpected
failure exits 1. An abs-only executable and a relabeled canonical evidence
run are both rejected before native start. This is the joint-image
`wide_neg` suite only: independent `F64_48`, `typed_relu` and the complete
35-suite gate are not established by this result.
