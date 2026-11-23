# logical_not(out=): direct public F64 dense permutation

Adds only disjoint, zero-offset, positive-stride, nonempty, whole-root
dense public F64 permutation destinations (dtype 4, eight-byte width).
The ordinary direct root uses the existing opcode111. F32, Bool and I64
remain supported by the same guarded branch; private dtype 3 is not
admitted. No other dtype, layout, default activation, or full-runtime
qualification is claimed.

Two independent cold builds reproduce one ELF. A separate locked CPU
Torch process produced exact bitwise fixtures before native testing.
The predecessor constructed and refused all three F64 destinations and
restored the full owner byte snapshot unchanged. The candidate matched
three each of F64, I64, Bool and F32, with 46 exact rollback controls
in 394 frames. F64 uses both signed zeros, positive and negative
subnormals and normals, infinities, and signed quiet and signaling NaN
payloads as exact raw inputs. The pure reader checks those unchanged
input words and independently computes the result from the magnitude
bits, where only signed zeros yield one. It reparses direct opcode111
construction records, verifies physical/logical byte placement,
root/view/sibling ownership, increments one shared Version once and
checks disjoint source and rollback bytes. No constructor control was
skipped. The prior 79 positives/82 rollbacks were replayed on this
image and all 876 raw frames match byte for byte.

prepare_f64.py takes the prior projected I64 source root, a new output,
the existing build.py and optional discovery receipt. Use the included
pinned manifest for an exact repeat. The raw observers accept the same
arguments as the preceding I64 bundle. The unchanged original
numeric-self readback verifies the retained 79 replay. See evidence.json
for exact hashes. Unique raw output remains in temporary storage.
