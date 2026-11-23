# Standalone CPU logical_or and logical_xor

This extends the exact independently qualified logical_and draft. Candidate-only op 133 implements functional logical_or and 134 implements logical_xor. Both use the existing bounded descriptor and element loop for ordinary, canonical contiguous nonempty CPU roots, same dtype and equal shape, covering six public dtypes and ranks 0–4. Results are fresh Bool tensors, even when both inputs have the same owner. The wire 131 AND route and its output bits are unchanged; operation 132 remains unsupported.

Two cold builds matched byte for byte. Locked CPU PyTorch matched native HIA and an independent no-Torch raw reader on 156 cases and 1,101 bytes per operator. Each includes 24 true scalars and six shared input handles, including signed signaling/quiet NaNs; shared XOR is all zero. Physical input bytes and owner identity, returned Bool width and metadata, release paths, transaction rollback and same-frame cursor were checked. The predecessor refused all 312 new calls with status 4. All 2,494 qualified AND frames and all 290 prior-main frames replayed exactly with no exclusions, including the original unsupported op132 refusal and owner bytes.

No empty, mixed dtype, broadcast, view, autograd, out/in-place, wider device or default activation claim.
