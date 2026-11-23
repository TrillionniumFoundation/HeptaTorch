# Bounded F32 gradient fan-in word profile

Deploy-disabled addition on exact [HeptaTorch #64 517c011](https://github.com/TrillionniumFoundation/HeptaTorch/commit/517c011f5eaebc45ff1e0fdb8433c8b6a20114c5). It changes only `incoming_merge_edge` in `runtime800_02`, and only that function's **already-present contribution** branch. It does not change generic `addf`, `mulf`, first-contribution storage/version handling, the default runtime or canonical files.

The opt-in guard requires the producer and leaf receiver to be rank-1 real F32 with matching extent, stride one, an actual matching slot-zero gradient edge, and stride one in both the existing incoming metadata and the newly described layout. It admits only the single-parent Clone and opt-in abs graph tags 17 and 201, and only a leaf receiver. Ordinary sums and both different raw NaNs use the original native result. The special cases quiet and retain the sole or identical-raw NaN, or return pinned x86 negative indefinite for opposite infinities. Raw-distinct sNaN and qNaN with otherwise equal payload bits remain a fallback, not an asserted equivalence.

## Independent physical result

The predecessor and candidate consumed the byte-identical 4,296-frame Cat/Clone diagnostic corpus: 204 cases, 3,670 final leaf words, and 7,342 retained edge words. PyTorch 2.15.0+cpu AVX2 ran each hooked test again without hooks and required an identical output. Cat slots and seed stayed fixed as the Clone creation order changed. A three-arm finite nonassociative control independently demonstrated the scheduling difference: arrival C,B,A gave +0; B,A,C gave +1 in both executables and real PyTorch. Both orders and all retained branch bytes are preserved.

* All 1,580 sole q/s NaNs, 132 identical-raw dual q/s NaNs and 22 opposite infinities now match, closing 1,734 observed mismatches. The 1,558 ordinary two-arm words are unchanged and exact.
* All 376 different-raw dual-NaN comparisons retain the original canonicalization mismatch. PyTorch selected the second **actual arrival** in these Cat/Clone cases; the result does not generalize across layouts/engine routes and the candidate deliberately does not implement that rule.
* The previous full focused 2,443-frame corpus was repeated. All 16 request streams and 14 non-fan-in response streams replay byte-for-byte. Each prior abs-plus-identity fan-in case now matches two more words; two raw-distinct sNaN/qNaN words per case still differ. The 314 multiply dual-NaN fallback mismatches and seven additional whole-owner observations on nested partial wire58 remain unchanged in classification. These extra owner differences are not evidence of leaks.
* Explicit rank-2 scope controls ran physically on both images in both creation orders. All 26 request/response frames per image are byte-identical and retain the legacy output.

Two exact cold builds produced identical 15 objects and executable `1f0c6efc64e96fe25106acee7bf77a2ef86dd3d5b129b55bb773f5e5ff46e768`. See [evidence](evidence.json), [predecessor diagnostic](readback-predecessor.json), [candidate diagnostic](readback.json), and [retained corpus](readback-retained.json). The pure readers do not execute a native binary or Torch. This is a bounded operator result, not full AD, profiler parity or an admission. The candidate did not run the 35 mandatory suites.

## Reproduction

Use the exact #64 517c011 `opt_in_f32_mul_backward_words` scratch source/image (`PREDECESSOR_SOURCE`, `PREDECESSOR_ELF`), observer/readback (`MUL_PROFILE`), #61 builder and pinned tools, #62 preexisting oracle loader/install, and a pinned CPython 3.12 Torch environment.

```sh
python3 -I -B prepare.py "$PREDECESSOR_SOURCE" "$SCRATCH/fanin-source"
python3 -I -B "$REPO/tools/current_runtime/build.py" \
  --manifest "$SCRATCH/fanin-source/manifest.json" --source-root "$SCRATCH/fanin-source" \
  --compiler "$COMPILER" --linker "$LINKER" --output "$SCRATCH/fanin-build"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$TORCH_PYTHON" -I -B -W error observe.py \
  --stack-observer "$MUL_PROFILE/observe.py" --loader "$ABS62/qualify.py" \
  --lock "$REPO/compatibility/pytorch-oracle-lock.json" --install "$TORCH_INSTALL_REPORT" \
  --protocol "$REPO/tests/integer_tensor_review/wire/protocol.py" \
  --native "$SCRATCH/fanin-build/heptatorch" --output "$SCRATCH/fanin-observation"
python3 -I -S -B readback.py "$SCRATCH/fanin-observation" "$MUL_PROFILE/readback.py"
```

The original diagnostic is independently reproducible with `observe-predecessor.py` and `readback-predecessor.py`, passing `PREDECESSOR_ELF` to the observer. The retained-corpus scripts use the same CLI and pinned #63/#64 historical inputs as the predecessor multiply package. The compressed raw archive contains both diagnostics, the retained replay and rank-2 control, including untouched per-child request/response frames. The retained observer exits 1 because the broader physical result is still a failure.
