# Native CPU where.self: dimensioned F16/F64 and BF16/F64

Adds only unordered public F16/F64 and BF16/F64 value pairs when both
value ranks are positive and the original ordinary contiguous no-grad
CPU-root checks pass. The Bool condition may be scalar or broadcast.
Each input retains its own width; output is fresh public F64.
Selected F64 words pass through verbatim. For chosen half inputs, the
existing exact bit helpers widen to the F32 encoding and then to F64;
this is not a floating intermediate operation. It quiets a selected half
signaling NaN while retaining sign and shifted payload. It does not
apply the prior F16→F32 FBGEMM-tail profile. Default routing, global
casts, allocation/broadcast mechanics, wire, rank-zero mixed restrictions
and other mixed pairs are unchanged.

Locked CPU Torch checked all 65,536 raw patterns per half type through
direct .to(torch.float64) and through actual where selecting every
pattern in each order. Boundary and mismatched-shape controls repeated at
source lengths 7/8/9/15/16/17. There were zero differences in 662,560 raw
words over 392 rows. The compact profile records row checksums. Two cold
native builds reproduced byte-identical ELFs. Separate no-Torch native
and independent offline owner/raw readbacks agreed with the locked CPU
fixture across 70 cases and 400,256 result words, including rank-four,
empty and exact old-control fixtures. At source lengths 32767/32768/32769,
the F16/F64 route retained signed half NaN payloads in both orders, while
the old F16/F32 and F16/BF16 F32-result profile retained eight genuine
32768-source refusals.

The actual predecessor refused all 70 exact setups (700 frames, status 7,
full owner rollback). The successor's 1,412 frames include 101 owner
rollback checks, twelve rank-zero mixed tests, six genuine views (two
positive-offset F64 and four zero-offset half views), cap, incompatibility,
stale/padding and valid-then-invalid suffix. Fresh Tensor, Storage and
Version IDs, initial Version zero, input payload/owner preservation,
both release orders, weak expiry and same-frame handle reuse were checked.
Twelve exact archived negative frames across four previous campaigns
were proved against matching accepted witnesses and are retained offline.
All 22,180 other prior request/response frames replayed byte-identically,
with 1,504 comparable full owner rollback controls. No full PyTorch
compatibility or production/default admission is claimed.

Reproduce the split projection:
```
python3 -I -B prepare_half_f64.py EXACT_PARENT_SOURCE EXACT_BUILD_PY OPTIONAL_NEW_SOURCE
```
The checked-in source manifest pins the second cold build. Generate
the independent official profile or physical fixture in the locked CPU wheel
with `profile.py` or `oracle_half_f64.py PINNED_HELPERS ORACLE_LOCK INSTALL_RECEIPT OUT`.
The precomputed fixture can be expanded without Torch. In isolated no-Torch
Python run `native_half_f64.py PINNED_HELPERS PROTOCOL_PY ELF FIXTURE OUT
baseline|candidate` against both ELFs, followed by
`pure_half_f64.py FIXTURE BASELINE_CAPTURE CANDIDATE_CAPTURE`.
For predecessor captures with their original directory layout,
`retain_previous.py PINNED_HELPERS PROTOCOL_PY CANDIDATE_ELF PREVIOUS_CAPTURE_ROOT
FIXTURE OUT` verifies the frozen raw request/response replay and rollback.

The original F32 allocation-pressure frame is reproduced under its
recorded process-local limit, restored before recovery. An initial
transport-only attempt omitted that fixture setup and predictably did
not reproduce the pressure result. The final replay restores the exact
original cap despite the successor ELF adding a page.
