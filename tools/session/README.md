# Native persistent sessions

This package builds an interactive native HeptaTorch process from the existing
authoritative HIA numerical source. It substitutes only the checked session
entry in build storage, independently compiles real module objects, links a
position-independent image through a native HIA linker, and packs the actual
read-frame/respond loop through a native HIA packer. Read [CONTRACT.md](CONTRACT.md)
for framing, transaction semantics, transport errors and the storage boundary.

The ordinary `src/heptatorch.hepta` entry remains the single-request program.
No second copy of the 143-function numerical implementation is stored here.
`entry.hepta` is the one-function overlay; `entry-authoring.txt` records its
authoring form and `entry-functions.json` its unchanged public signature.
`profile.json` pins the source, compiler/linker/packer profiles and both runtime
identities. Source or profile updates must be reviewed and repinned explicitly.

## Reproduce the native tools

Start with the already verified expanded compiler and borrowed module compiler
from `tools/compiler-profile` and `tools/modules`. Use the pinned external Hepta
checkout at commit `f33d8da316de8fdf27b504371f1a5310997ff548`. The following
variables are absolute caller-selected paths. Each output directory must be
new; generated upstream-derived HIA stays in build storage.

```bash
python3 -I -S -B tools/session/build_flat_linker.py \
  --source-root "$upstream" \
  --capacity-adaptation "$capacity_generated/adaptation.json" \
  --output "$session_generated/flat"
python3 -I -S -B tools/session/build_packer.py \
  --source-root "$upstream" \
  --capacity-adaptation "$capacity_generated/adaptation.json" \
  --output "$session_generated/packer"
python3 -I -S -B tools/session/rebuild_flat_linker.py \
  --compiler "$expanded_compiler" --module-compiler "$module_compiler" \
  --generated "$session_generated/flat" --output "$session_native/flat"
python3 -I -S -B tools/session/rebuild_packer.py \
  --compiler "$expanded_compiler" --module-compiler "$module_compiler" \
  --generated "$session_generated/packer" --output "$session_native/packer"
```

The two authoring adapters verify the upstream closure before and after
imports. The flat-linker adapter first reproduces the frozen borrowed linker,
then verifies that only its magic and output-header functions changed. The
packer adapter first reproduces the qualified small session prefix, then
emits the wider HIA packer and the HIA tool that constructs its real I/O wrapper.
Native compilation occurs only in the rebuild scripts. Both unchanged native
compilers must produce identical images; the scripts verify the final pins.
Python performs source organization, file transport and checks, with no host
ELF linking or production numerical execution.

| Artifact | SHA-256 |
| --- | --- |
| Flat-linker HIA | `f2096b101480369d8e51b507fe1d6266b29818ebcd741c83ef6b7f8de4a20a19` |
| Native flat linker | `4c36e3409b397788664f1501896aff2719f667306795cb5fa95061d710fd6139` |
| Wide session-packer HIA | `280353a1e789be497e604ba7568d2fef90c99d3447a151e389670c6984e87a7a` |
| Native wide session packer | `22600cee8853b6a49dfbf66154ddcdc99bdbbc6fcfb2face2605ad7117b20a44` |
| HIA tool-packer source | `285d15f403d77080c82e904493e8e69d81df47f0c790638778f1abe0d29230f1` |
| Native HIA tool packer | `ebcf4be893e647bf1e832088f1f1d4da588f41c1513af4a093e1462ba4930798` |

The flat linker has 27 functions, 570 blocks and 2,318 instructions. The wide
session packer has 6 functions, 30 blocks and 376 instructions. Their compiler
and linker capacity profiles remain separate from the one-MiB tensor arena.

## Build the runtime

The modular path uses the existing complete source partitioner and frozen
borrowed module compiler. All numerical function bytes are accounted for.

```bash
python3 -I -S -B tools/session/build_runtime.py \
  --source src/heptatorch.hepta --mode modular \
  --module-compiler "$module_compiler" \
  --flat-linker "$session_native/flat/flat-linker" \
  --packer "$session_native/packer/session-packer" \
  --output "$runtime_native/session"
```

The executable is `$runtime_native/session/application`. The build receipt
records the source/overlay, every unchanged function hash, each real object,
the raw linked image, packer and final executable. Its source overlay has
143 functions, 1,786 blocks and 7,078 instructions. `init` and the other 141
non-entry functions remain byte-identical to the original program.

The monolithic flat build is retained as a separately identified reference:

```bash
python3 -I -S -B tools/session/build_runtime.py \
  --source src/heptatorch.hepta --mode monolithic \
  --compiler "$expanded_compiler" \
  --packer "$session_native/packer/session-packer" \
  --output "$runtime_native/session-monolithic"
```

For canonical source `264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233`:

| Runtime | SHA-256 | Bytes |
| --- | --- | ---: |
| Modular session | `193c1b59323ced38fc5e2cf9ec939ade70ccb62e239bb65d01e22c188409b6db` | 733,958 |
| Monolithic session reference | `ce6ffe8f8530b0ad68ac93e01f47d3473b2749697021590c41c18e6c847d4077` | 730,806 |

The wider packer reproduces those exact binaries. The complete original
single-request fixture suite must still run against the ordinary runtimes.
When replayed through fresh sessions, successful responses are unchanged;
failed responses follow the session's canonical 32-byte transactional refusal
rule. Do not compare a failed session response to an old partial-output packet
and call the intentional protocol difference a numerical regression.

## Native growth and packing gates

```bash
python3 -I -S -B tools/session/tests/packer.py \
  --executable "$session_native/packer/session-packer" \
  --image "$session_native/packer/session-packer.image" \
  --generated "$session_generated/packer" --output "$reviews/session-packer"
python3 -I -S -B tools/session/tests/modular_capacity.py \
  --compiler "$module_compiler" --legacy-compiler "$expanded_compiler" \
  --flat-linker "$session_native/flat/flat-linker" \
  --packer "$session_native/packer/session-packer" \
  --output "$reviews/session-growth"
```

These tests run native executables with an empty `PATH`. Guard observation is
performed in a separate stdlib-only child and binds input/image hashes. The
packer suite exercises both the actual wide I/O wrapper and its flat kernel,
including an eight-MiB image, exact output capacity, one-byte-short output,
dirty workspace, malformed lengths and failure-output atomicity.

The growth suite first observes refusal of a valid 8,381-instruction single
unit, then independently compiles and executes the same arithmetic as two
units. A twelve-unit case contains 49,591 instructions and actually executes
49,248 additions per request. Its real 1,204,224-byte linked image is packed and
executed, with two responses observed before stdin EOF. Reversed object order
forces entry selection away from the first text section. These synthetic
capacity probes establish build/execution paths; they add no PyTorch schema
coverage.

Keep the independent live-session suite and the real-PyTorch cross-frame
oracle as separate gates. The finite native receipts do not stand in for the
upstream oracle or close the project's full compatibility gates.
