# Retained typed ReLU and the two unary-7 successors

The joint HeptaTorch [#63](https://github.com/TrillionniumFoundation/HeptaTorch/pull/63)
ELF `5af79385…` deliberately assigns unary operation 7 to I64/F64 abs.
The original typed ReLU observer itself uses unary operation 5. Of its 776
cases, only 403 and 404 assert that unary operation 7 must be refused on
their exact I64 `[1, INT64_MIN]` and F64 `[+1, -0]` inputs.
These probes resemble the separate wide-negation probes, but this replay
binds the actual ReLU rows, reference, and raw transport independent of them.

1. The unchanged canonical observer passes 776/776 on #61 ELF `083a92ad…`,
   with the original pinned AVX2 Torch reference and independent readback.
   Its actual wire evidence retains status 7 and no records for both
   historical requests.
2. The joint ELF, with only explicit candidate build admission, fails
   exactly those two original predicates. The actual same requests
   returned status 0 and one opcode-3 result. The failed native report
   and original nonzero exit are retained, not rewritten.
3. The separate successor reuses the same original captured ReLU reference
   and runs actual, independent pinned AVX2 Torch `abs` on those exact two
   inputs. Every other original comparison is retained. Both added
   success checks compare complete result bytes, dtype, shape, strides,
   offset, and metadata; verify fresh Tensor/Storage/Version handles;
   check no retained input/autograd edges; and require all 84 input
   descriptor words and all input bytes unchanged. The original post-pass
   physical readback then replays all cases, including the exact successor
   checks from the captured transport.

The admissions bind all 15 native objects, link packet, source bytes, ELF,
and previously accepted scoped receipt. The two candidate runs still call
and confirm the original canonical admission refusal before using the
separate candidate gate. Original files and all canonical checks remain
untouched. This does not qualify the full coordinator or enable an operator.

## Reproduce

Use a clean #63 checkout, its pinned PyTorch CPU wheel installed as in the
ordinary workflow, and fresh directories. Set `BASE_BUILD` to the complete
canonical cold build and `DUAL_BUILD` to the exact joint cold build.
From the repository root:

```sh
R="$REPO/tools/current_runtime/opt_in_dual_unary"
BASE_SHA="$(sha256sum "$BASE_BUILD/build.json" | cut -d' ' -f1)"
python3 -I -B "$REPO/tests/typed_relu/run.py" --build-directory "$BASE_BUILD" \
  --expected-build-sha256 "$BASE_SHA" --torch-python "$TORCH_PYTHON" \
  --output "$SCRATCH/typed-relu-base"
code=0
python3 -I -B "$R/replay_typed_relu_successor.py" --repository "$REPO" \
  --baseline-build "$BASE_BUILD" --baseline-run "$SCRATCH/typed-relu-base" \
  --dual-build "$DUAL_BUILD" --dual-scoped "$R/qualification.json" \
  --torch-python "$TORCH_PYTHON" --mode retained --output "$SCRATCH/typed-relu-retained" || code=$?
test "$code" -eq 2 || exit 1
python3 -I -B "$R/replay_typed_relu_successor.py" --repository "$REPO" \
  --baseline-build "$BASE_BUILD" --baseline-run "$SCRATCH/typed-relu-base" \
  --dual-build "$DUAL_BUILD" --dual-scoped "$R/qualification.json" \
  --torch-python "$TORCH_PYTHON" --mode successor \
  --retained "$SCRATCH/typed-relu-retained/observer" --output "$SCRATCH/typed-relu-successor"
```

Wrapper exit 2 preserves the retained observer's actual exit 1 and exact
two failed observations. Wrapper exit 1 means some other check or physical
requirement did not pass. The [compact evidence](typed-relu-successor-evidence.json)
keeps exact actual report and wire hashes separately; process identities and
raw report hashes may differ legitimately on rerun.

| Physical run | Native outcome | Frames | Checks |
| --- | --- | ---: | ---: |
| Canonical #61 | 776 passed | 21,863 | 75,060 |
| Joint, original predicates | 774 passed; 403 and 404 fail | 21,793 | 74,820 |
| Joint, explicit two successors | 776 passed | 21,877 | 75,126 |

The successor retains all 90 protocol controls and four genuine allocation
refusals, each with an identical successful retry after restoring the limit.
The 774 unaffected case reports equal their canonical counterparts. Both
abs-only substitution and relabeling the baseline as a candidate failure
are refused before entering the candidate observer.
