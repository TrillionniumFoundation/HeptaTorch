# R3 multiaxis72 native evidence

This is one real R3 replay of the immutable 72-case PyTorch observations. The
candidate is 627 functions / 13 actual compiled units, source 3c2c0a12,
image 7841b190. All 72 cases passed the unchanged gradient-word comparisons:
1,224 wire frames, 21,312 exact F32 gradient words, 72 EOF exits with code zero.
All 58 bound inputs remained byte-identical. No Torch was executed in this run.

The R1 script used ones for the base tensor while the original probe uses
torch.arange(256). R3 restores the original 0..255 F32 input bytes. Every emitted
base-create packet is checked independently in the outer execution report. The
old script, report and complete historical archive remain separate original
bytes. reader-source/exact-inverse.json inside this archive records all eight
profile/input edits and proves that the comparison/error/finish code is unchanged.

This gate compares base.grad and retained view.grad logical F32 bits, presence
and lengths, successful wire operations, input packet bytes and EOF. Gradient
Version is recorded, not compared to an absent reference field. It does not
establish full Tensor/Storage identities, Saved lifetime, full RNG state,
arbitrary CPU dispatch or complete PyTorch schema parity.

The archive contains the original native requests/responses and reports,
durable job metadata/stdout/stderr, the exact reference/probe, reader and inverse,
all R3 source and 13 source/object units, ELF and build metadata. custody.json
binds a complete member-by-member size/SHA/mode check and one complete XZ stream.
Outer Git copies are mode 0644; original inner modes are preserved.
