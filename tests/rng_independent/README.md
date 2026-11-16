# Independent CPU RNG review

These programs are qualification harnesses. The production target is the
unchanged native HIA RNG executable. The C++ standard-library engine and Python
arithmetic are independent references; neither is linked into the target.

The final reviewed source is
`c1de63f5f046bfb20aa4d4f65cbffa6217a38a36c0dc63b05f08c7ddc7612032` and
the executable is
`cffc7c2f6574bf08a184e7d4f84e41a38ce0fb9be2f645bf9c332c93c2621eac`.

Run from the repository root on Linux x86-64:

```sh
python tests/rng_independent/flat_probe.py \
  --native BUILD/rng-native --expect-short return-zero \
  --output BUILD/rng-flat-review
python tests/rng_independent/review_rng.py \
  --native BUILD/rng-native --source tools/rng-cpu/rng.hepta \
  --output BUILD/rng-state-review
```

GNU C++ is required for the second command. Output directories must not already
exist. The scripts bind the native and source hashes; they do not accept an
unidentified implementation merely because a sample happened to match.

## Actual tests

The flat test calls the real six-word bridge present in the pinned ELF. Its
offset is derived from the observed startup `call` and checked against the
bridge bytes. The machine code is mapped unchanged. Each invocation runs in a
forked process, allowing the parent to observe both a signal and shared-memory
canaries without continuing execution after a trap.

The initial executable `1a1daa83...` trapped for output capacities below 24.
For capacities 1, 7, 8, 15, 16 and 23, it had already written that many bytes
within the declared output region. No capacity-exceeding write was observed,
and workspace remained unchanged. This was a short-buffer error-path defect,
not a packet exploit of the executable's fixed 1 MiB output allocation.

The final implementation returns zero without writing either buffer when
output capacity is below 24. Capacities 24 and 5,079 produce the ordinary
24-byte capacity-error reply; 5,080 suffices for a zero-value state reply.
All ten observations pass. The original failure is retained separately; it is
not overwritten by the fixed result. To reproduce that historical observation,
use the pinned old executable and `--expect-short trap`.

The independent state review has 25 cases. It imports manually constructed
states and uses inverse tempering to force exact engine outputs. These are
test inputs, not special cases in the implementation. They exercise:

- The precise half and bfloat16 rounding thresholds that become 1 and must be
  mapped back to zero by the pinned PyTorch uniform kernel.
- Double's low 53 random bits and high-word-first composition, plus a two-word
  draw straddling an MT refresh.
- Integer spans below, at and above `2**28`, followed by identifiable raw draws
  to prove the exact one-word/two-word consumption change.
- All 624 serialized state-word truncations, upper next-index truncation,
  boolean seeded canonicalization, dead legacy fields and padding.
- Bit-preserving normal-cache import, including signaling-NaN bit patterns,
  zero-count commands and cache clearing on reseed. Normal generation and
  consuming these caches are outside the implemented command set.
- Exactly 1,000,000 words of compound work, and refusal before workspace
  mutation when a later double or threshold-range draw exceeds that limit.
- Malformed headers, state fields, reserved words, counts, late invalid
  commands and buffer boundaries. Fifteen failure cases prove full workspace
  preservation and absence of writes beyond the 24-byte error reply.

Every successful state case compares all 5,056 serialized bytes, not only the
sampled values. The MT reference loads the same 624 words into an independent
`std::mt19937`; it does not copy the HIA twist implementation. It exports the
standard engine's post-draw words and position for comparison.

## Source findings and scope

The pinned PyTorch `CPUGeneratorImpl` uses a legacy structure with an initial
64-bit seed, 32-bit left/seeded fields, a serialized 64-bit next index and 624
64-bit state words. On import, engine words and next are truncated to 32 bits.
The double cache is at byte 5,024 with its flag at 5,040; the float cache is at
5,048 with its one-byte flag at 5,052. The reviewed HIA offsets and
canonicalization match these source operations.

The HIA state contract is intentionally narrower than every byte sequence
accepted by upstream `set_state`: it accepts canonical positions `(1,0)` or
`left+next=625` with `1 <= next <= 624`, and requires cache flags to be 0 or 1.
Upstream's engine validator does not enforce the position relation. Some
inconsistent positions are unsafe; others, such as `left=1,next=1`, safely
force a refresh but are still rejected by this profile. This is not full
malformed-state acceptance parity.

The `2**28` consumption threshold belongs to the pinned non-FBCODE source
branch. Integer range mapping uses modulo, as that implementation does; it is
not replaced with a rejection sampler. Uniform half/bfloat16 generation uses
float opmath and subsequent conversion, rather than reducing the source
random mask to the destination mantissa width.

Only `entry` changed between the two HIA candidates. The other 17 function
bodies are byte-identical. Raw sequence/state behavior passed the independent
review on both candidates. The final report is bound to the corrected image.

This review does not run a local PyTorch installation. The primary oracle's
locked-wheel job must separately establish actual PyTorch value/state parity.
Tensor RNG integration, default/global generator behavior, concurrent access,
normal distributions, dropout and complete PyTorch RNG support remain open.
The six upstream source hashes and exact source locations are recorded in
`validation/native-rng-independent-source-audit.json`.
