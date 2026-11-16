# Native HIA CPU MT19937 and uniform transforms

`rng.hepta` executes MT19937 initialization, refresh, tempering, state import and
export, raw words, unit-interval floating values and signed integer ranges. The
production ELF uses HIA unsigned integer instructions and the already qualified
half/bfloat16 rounding helpers. It has no Python, C++ standard-library, PyTorch
or host RNG dependency. Those implementations are independent test references.

This component targets the CPU generator and non-FBCODE distribution paths of
PyTorch commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. Six original files and
their exact byte hashes are retained in `upstream-sources.json` and
`third_party/pytorch/rng/`. The original MT19937 redistribution notice is retained
in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt), in addition to the existing
PyTorch notices; include it when distributing the executable.

## Numerical and state contract

The 64-bit seed is preserved as metadata; engine initialization uses its low
32 bits. A raw64 result consumes a high uint32 word followed by a low uint32
word. A binary32 unit value uses the low 24 bits divided by 2^24; a binary64 unit
value uses the low 53 bits of a raw64 value divided by 2^53. Half and bfloat16
use binary32 opmath followed by the existing nearest-even conversion. When that
conversion produces 1, the result maps to 0 as in the pinned CPU uniform kernel.

Signed int64 intervals are half open, `[low, high)`. The generator consumes one
word for a width below 2^28 and two words otherwise, applies the pinned modulo
mapping, then adds the signed lower bound. This deliberately preserves the
upstream sequence and consumption rule, including its modulo bias.

State output is the complete 5,056-byte current CPU representation, with canonical
padding and fields. Each of the 624 engine words occupies a 64-bit serialized
slot containing a uint32 engine value. Import truncates high engine and `next`
bits as the upstream implementation does. The serialized normal caches are
preserved when valid, including their exact floating bit patterns; uniform and
integer operations do not consume these caches. Reseeding clears both caches.
Normal random values themselves are not implemented in this component.

| State field | Offset | Size |
| --- | ---: | ---: |
| Initial seed | 0 | 8 |
| `left` | 8 | 4 |
| `seeded` | 12 | 4 |
| `next` | 16 | 8 |
| 624 engine words | 24 | 4,992 |
| Legacy normal x | 5,016 | 8 |
| Double normal cache / legacy y | 5,024 | 8 |
| Legacy normal rho | 5,032 | 8 |
| Double cache valid flag | 5,040 | 4 |
| Padding | 5,044 | 4 |
| Float normal cache | 5,048 | 4 |
| Float cache valid flag | 5,052 | 1 |
| Padding | 5,053 | 3 |

Import accepts seeded canonical/reachable MT positions: the initial
`left=1,next=0` pair, or `next>0` and `left+next=625`, with both counters in range.
Cache flags must be 0 or 1. This is narrower than upstream's permissive state
validator: some inconsistent upstream positions safely refresh while others can
access invalid memory. This service rejects all noncanonical positions and does
not claim complete `set_state` input-admission parity. Legacy 5,048-byte states
are also outside this first profile.

## Packet format

All integer fields are little endian. The 32-byte header is `HTRNG001`, a uint32
initialization mode, a uint32 command count, a uint64 seed and a uint64 state
length. Mode 0 requires state length 0 and initializes from the seed. Mode 1
requires seed 0 and exactly 5,056 subsequent state bytes. Commands follow the
state. Each occupies 32 bytes: uint32 opcode, uint32 count, uint64 `a`, uint64
`b` and a reserved uint64 zero word. The complete request length must match.

| Opcode | Operation | Parameters |
| --- | --- | --- |
| 0 | Raw uint32 | `a=b=0` |
| 1 | Raw uint64 | `a=b=0` |
| 2 | Unit binary32 | `a=b=0` |
| 3 | Unit binary64 | `a=b=0` |
| 4 | Signed int64 interval | `a=low`, `b=high`, encoded as uint64 |
| 5 | Unit IEEE half | `a=b=0` |
| 6 | Unit bfloat16 | `a=b=0` |
| 7 | Reseed | `a=seed`, `b=0`, count 0 |
| 8 | Discard raw uint32 words | `a=b=0` |

A successful response contains three uint64 header words: status 0, returned
value count, state length 5,056. It then contains the final complete state and
one uint64 slot per returned value. Floating results are raw encodings, with
unused high bits zero. Reseed and discard emit no values.

The component admits at most 128 commands, 100,000 returned values and 1,000,000
consumed MT words. All commands, total consumption and output capacity are
validated before state mutation. Failure returns a 24-byte header with status 1
for malformed framing/commands, 2 for invalid state or 3 for resource limits;
count and state length are zero. Workspace remains unchanged. The flat ABI
requires 5,056 workspace bytes; when output capacity is less than 24, it returns
zero without writing output or workspace. Input/output/workspace aliasing is
outside this borrowed ABI contract.

## Executed local qualification and CI

The native compiler pin is
`87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54`.
Source SHA256 is
`c1de63f5f046bfb20aa4d4f65cbffa6217a38a36c0dc63b05f08c7ddc7612032`;
the executable pin is
`cffc7c2f6574bf08a184e7d4f84e41a38ce0fb9be2f645bf9c332c93c2621eac`.
There are 18 HIA functions. Eight reused helper bodies remain byte-identical to
their existing qualified source. The native source is compiled twice with empty
PATH; both images must match exactly before executing the test corpus.

```sh
python3 -I -S -B tests/check_rng_custody.py
python3 -I -S -B tests/rng_cpu_oracle.py \
  --compiler /absolute/path/to/native-hia-compiler \
  --reference cpython --output /absolute/path/to/new-rng-review
python3 tests/rng_cpu_oracle.py \
  --compiler /absolute/path/to/native-hia-compiler \
  --reference pytorch --output /absolute/path/to/new-torch-review
```

The executed CPython/standard-library campaign passed 74 cases, 110 native
invocations, 718,619 exact values and 434,816 compared serialized state bytes.
For 15 seeds the C++ standard MT19937 reference and CPython engine agree on
4,097 raw words and the resulting state before serving as references. The
separate frozen review passed 35 actual native checks: ten direct flat ABI
capacity cases and 25 state, transform, consumption and atomicity cases. It
constructs precise half/bfloat16 thresholds using inverse tempering, checks
cross-refresh raw64, the 2^28 consumption boundary and all 624 high-bit truncations,
and proves 15 failure scenarios leave the complete workspace unchanged.

The first native candidate trapped after partially writing inside a short output
buffer. That observed failure is retained in `validation/native-rng-independent.json`.
The corrected entry returns zero without writing. The other 17 function bodies
and every request/response row of the original 74-case corpus are unchanged.

The PyTorch mode compares actual `rand`, `randint`, raw64 and complete generator
state with the exact locked wheel, including all eleven provenance checks and
two normal-cache import observations. Raw32 full outputs use the independent
standard/CPython reference; the Torch comparison additionally checks their low31
projection and exact full-state advancement. Run [43806361208](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43806361208) passed all twelve workflow steps: 76 actual PyTorch cases, 114 native invocations, 719,337 exact values and 455,040 state bytes, with all eleven provenance checks. The full source-bound receipt is `validation/hosted-native-rng.json`.

This component has explicit state input/output. Persistent Generator objects,
default/global RNG state, concurrent access, numerical Tensor filling, other
distributions, accelerators and complete PyTorch RNG semantics remain open. All
2,590 upstream schemas and the full subsystem gates remain in the denominator.
