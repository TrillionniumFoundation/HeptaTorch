# Native HIA float8 E4M3FN conversion

Independent Core-v8 HIA machine-code implementation of f32 -> float8_e4m3fn (RNE) and float8_e4m3fn -> f32. The 4-bit exponent has bias 7, a 3-bit fraction and no infinity encoding; signed 0x7F is the single NaN code. Finite values range from ±2^-9 to ±448. The locked PyTorch 2.15.0+cpu contract saturates finite overflow and signed infinity to ±448 (E4M3FN codes 0x7E/0xFE); 464 and above also saturate. Input NaNs alone produce signed 0x7F/0xFF. The earlier local 2.10.0 behavior disagrees on overflow; the original pinned-reference CI failure is retained rather than hiding this version difference. Normal and subnormal rounding uses integer shifts, exact tie-to-even and no host numerical operations.

Input: exact little-endian uint32 mode and count, followed by f32 words (mode 0) or E4M3FN octets (mode 1). Output: uint32 status then converted octets or f32 words. Mode/count/length refusals are strict; maximum count 200000.

The production implementation is compiled HIA machine code; Python and PyTorch serve only as independent reference and CI provenance checks. The test compares all 256 bit patterns, all 127 finite midpoint/overflow boundaries with both signs and adjacent f32 encodings, 100000 seeded random f32 inputs, maximum packet limits, and malformed requests. NaN checks preserve classification and sign, not exact payload parity. Exact-PyTorch parity must be demonstrated by the locked 2.15.0+cpu wheel and 11 provenance checks in the workflow; local tests with another version do not establish this qualification.

This standalone component does not yet implement Tensor dtype dispatch, promotion, device transfer, autograd or any complete upstream PyTorch schema. All 2590 schemas and 17 full acceptance gates remain HOLD.
