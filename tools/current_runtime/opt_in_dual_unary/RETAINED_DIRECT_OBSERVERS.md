# Retained BF16-copy and typed ReLU on the joint image

This package physically replays the original two mandatory observers from
HeptaTorch [#63](https://github.com/TrillionniumFoundation/HeptaTorch/pull/63).
Both run against the same exact joint ELF `5af79385…`. It does not change an
original result predicate, qualify an unexecuted suite, or move the default
runtime.

The original qualification reader still accepts the canonical ELF and refuses
the joint build at its manifest identity check. Only the two admission calls
in each exact, hash-bound observer are replaced reversibly in memory. The
first admits the candidate for native execution; the second, reached only
after a successful observer run, checks the same exact build for independent
physical readback. The separate joint-image admission binds all 15 source and
object files, exact link packet, ELF and scoped evidence. The original
reference observer captures fresh values from the pinned CPU wheel; all
comparisons and result predicates remain unchanged.

| Mandatory suite | Native result on joint ELF | Physical frames | Checks |
| --- | --- | ---: | ---: |
| BF16 copy | 280/280, independent physical readback passed | 6,088 | 23,096 |
| Typed ReLU | 774/776; two original legacy-refusal assertions failed | 21,793 | 74,820 |

Typed ReLU's cases 403 and 404 issue the same original unary operation 7
request on, respectively, an I64 `[1, INT64_MIN]` tensor and an F64
`[+1, −0]` tensor. The old test requires status 7, reflecting an unimplemented
typed unary operation. The joint image actually returns status 0 and one
opcode-3 result record, because operation 7 now implements abs. The failure
and the original request/response hashes are retained. No successor
expectation, omitted row, or pass is synthesized here, and the original
post-pass physical typed ReLU readback does not run. A future separate
successor must verify the new output and ownership while retaining the
original refusal on the canonical image. The fixtures' value words match
the two wide-negation successor rows, but this suite has separate evidence.

## Reproduce

From an exact #63 checkout (with this package), set `REPO` to that checkout,
`BASE_BUILD` and `DUAL_BUILD` to the corresponding complete cold builds, and
`TORCH_PYTHON` to the fixed CPU wheel interpreter from the ordinary workflow.
The checkout must have no preexisting bytecode caches. Use fresh outputs.

```sh
CHECK="$REPO/tools/current_runtime/opt_in_dual_unary"
for suite in bf16_copy typed_relu; do
  code=0
  python3 -I -B "$CHECK/replay_direct_observers.py" --suite "$suite" \
    --repository "$REPO" --baseline-build "$BASE_BUILD" --dual-build "$DUAL_BUILD" \
    --dual-scoped "$CHECK/qualification.json" --torch-python "$TORCH_PYTHON" \
    --output "$SCRATCH/$suite" || code=$?
  # The typed ReLU run must retain exit 1 and the two observed failures.
  case "$suite" in bf16_copy) test "$code" -eq 0;; typed_relu) test "$code" -eq 1;; esac || exit 1
done
```

Successful BF16 readback and original ReLU failure evidence are kept under
their respective `observer` directories. Receipts bind physical report hashes,
which may differ by run owing to paths, PIDs and timings. Wrong canonical
and abs-only inputs and an unrelated scope receipt all fail before starting
the original observer. See [compact evidence](direct-observer-evidence.json).

Other original lanes are not covered here. Historical resources, Cat and
view groups bind separate original reference packages and workers; the
indexed and higher-level forward suites also assert older source lineage
proofs. Those cannot be counted from BF16 identity, and the legacy 35-suite
coordinator remains unchanged.
