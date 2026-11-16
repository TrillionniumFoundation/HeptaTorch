# Native HIA float8 E4M3FNUZ conversion

Standalone native `float32 ↔ float8_e4m3fnuz` conversion, not a routed PyTorch operation. Both directions and packet bounds execute as HIA-compiled x86-64 machine code without an upstream numerical runtime.

E4M3FNUZ uses exponent bias 8, 127 finite positive encodings, a single unsigned NaN code 0x80, and no negative zero. Finite saturation and infinity mapping must be confirmed on the pinned PyTorch 2.15.0+cpu oracle, rather than inferred from installed local PyTorch builds. An independent CPython exact finite-lattice reference compares every 256 encoding, every finite midpoint and adjacent binary32 word, 100,000 deterministic random input words, both maximum packet sizes, and malformed packet refusals. Original and newly observed failures must remain visible.

Packet input: little-endian u32 mode (0 = round f32→E4M3FNUZ, 1 = widen), u32 count, and a precisely sized payload of f32 words or float8 octets. Output: u32 status (0 success, 1 refusal) followed by packed converted values. At most 200,000 values; no trailing bytes admitted.

The test compiles the same source twice with the fixed standalone HIA compiler, executes both under an empty PATH, and checks the full provenance of the pinned torch CPU wheel when requested. A passing component proves only these conversions; it does not qualify Tensor storage, dtype dispatch, promotion, autograd, devices or any complete upstream native schema. All 2,590 declarations and 17 full-subsystem gates remain open.
