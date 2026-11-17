# Wrapped binary64 scalar reference32

This packet preserves the single actual Torch execution 1a1e70727ca-392-3cb7. The original probe and report remain observation-only; no native or general public F64 Tensor API qualification is inferred.

- Probe c0ddafbae6ea81bd98f6066bc867f71e0b6e8d4f73e6c5d7f15f7fe3d29569a3.
- Actual f3ecc9dd7db2ce8fbd0e424557ad5b95ac2e899a1c09cf38dad47e4fecc2b0c6: 32 observed recipes, 192 successful operations, no probe errors.
- Execution custody ffe9bb611f53958a842a44471710ab405874ea1693360fff86247c8935284401 binds all 25 inputs before/after, ten declared inputs, two complete fixed upstream sources, the exact installed wheel and actual pip report. All eleven provenance checks are also independently recomputed from their full records.
- Independent pure admission job 1a1e71b5aed-392-3cd7 passed 70,859 checks. The complete deterministic report is 767923408156d36d1f2247ce85b16f70f228bcf222a6f2068bf1b18c118f1e6d. This job did not execute Torch, a native Tensor session, or a conversion helper.

The 32 inputs cover signed zero, ordinary signed values, even/odd rounding ties and neighbors, subnormal boundaries, overflow thresholds, infinities and signed payload qNaN/sNaN. The original binary64 input is built as a Python float and independently packed before/after multiplication and backward. Every actual roundtrip and Saved.other eight-byte value remains the original word, including signaling NaNs. All F32 outputs and gradients are observed bytes, not synthesized expected values.

The admission records complete Tensor and Storage identities, layouts, physical bytes, logical bytes, Versions, real MulBackward edges, original Saved.other float64 storage and default-backward Saved-release exceptions. It retains the full RNG state and all operation/getter warnings and errors. The observed F32 input gradient and retained output gradient are Version zero in these cases. Inf multiplied by zero produces observed word 0xffc00000. These statements do not establish every dual-NaN operand priority or general F64 arithmetic.

## Pure verification and restore

Run with a system Python and no site imports:

    python -I -S -B verify_packet.py --packet PACKET

To restore all original modes and replay the independent admission:

    python -I -S -B verify_packet.py --packet PACKET --restore NEW_ROOT --admission-output NEW_REPORT

The verifier restores the nested 91-member original packet under NEW_ROOT/original-packet/payload, then executes only the pinned admit_actual32.py and its unchanged parent_admit_actual26_r03.py dependency. The entire replayed report must match the original admission bytes. No installed Torch or native executable is needed. Outer packet files may be installed into Git as 0644; their original 0600 modes remain recorded and all archive-member modes are restored exactly.

Source04's proposed compilation was not run. Source05 compilation and any native comparison are separate evidence, with independent identities and statuses; they cannot change this reference report.
