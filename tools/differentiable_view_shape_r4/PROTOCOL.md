# R4 concrete Tensor view / reshape protocol

Status: authoring and source preparation. No native qualification is asserted.

Parent is immutable R3 source 3c2c0a125bcac6d47075bf58fc3f78c0e7b9a0b1370757981b99dcd3793b21c6, image 7841b19051960b6fa13cb98841dbb8e4d6ead97f03e2ad65af415e1a35b03221. R3 commands and packages stay frozen. New owner tags HTVSHP01 / HTOVSHP1 distinguish this profile; descriptor size remains 672 bytes.

Commands 112 and 113 are introduced only in this runtime protocol after checking the entire parent entry/command dispatch. A command is eight little-endian u64 words:
[operation, source_handle, config_absolute_packet_offset, 40, 0, 0, 0, 0].
The payload is exactly five little-endian 64-bit words [rank, size0, size1, size2, size3]. Active sizes are signed int64 encoded as their raw two's-complement words. Rank is 0..4. Unused size words must be zero. A single -1 is inferred by genuine runtime arithmetic; all other negative sizes, invalid inference, incompatible element count and dimensions above 262144 are rejected. Packet offsets must be 8-byte aligned, inside the payload area and bounded by the actual packet length.

112 performs concrete view: it uses infer_size and computeStride, returns a new true view with shared Storage/Version and ultimate Tensor base, or raises normalized status 2 if shape/stride compatibility fails. 113 performs concrete reshape: a contiguous source selects the same view path; a compatible noncontiguous source produces a true _reshape_alias view; an incompatible source creates a real contiguous clone and then a distinct _unsafe_view Tensor. Existing legacy command 17 is unchanged.

F32, bool and signed int64 preserve real element widths 4, 1 and 8. Only F32 admits first-order AD. The new view/reshape paths capture the immediate input gradient edge independently from the ultimate-base Tensor root. Operation 16 remains ViewBackward, 45 is ReshapeAliasBackward, 46 is UnsafeViewBackward, and the real intermediate clone uses operation 17 CloneBackward. The three reshape/view backward paths use grad.reshape on the original input shape; they do not use AsStridedBackward's storage scatter/reduction.

Ordinary errors keep the existing full-frame atomic rollback and output-capacity precedence. Backward partial failure behavior remains the existing explicit command 58 contract. A legacy alias marked view without a genuine ViewInfo base is rejected at the new boundary until its base contract is implemented; it is not silently relabeled a proper view.

Python is used only for source emission, command construction, provenance and strict observation. No Python computes production output shapes, strides, values or gradients.
