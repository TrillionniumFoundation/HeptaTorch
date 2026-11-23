# Native HIA two-layer F32 CPU inference, narrow scope

Measured on exact existing #71 current800 source f8f924a82763a28d8ad12adbd371f25458ed86d1fbdee2406bb93912847afb4b,
ELF 5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d.
No HIA code or build changes. The app uses one owned ELF session per
case. It sends five external F32 literal roots and forwards each live
output handle directly into the next operation: matrix multiplication,
bias broadcast Add, ReLU, matrix multiplication and bias broadcast Add.
The host sends/reads wire bytes; no intermediate tensor is synthesized
or computed in the host, and Torch is not the backend.

Three immutable v2 admitted cases matched their original exact
single-thread locked eager Torch references and stdlib expectations:
15 output stages, 80 words, 82 input words, 384 valid response frames.
Each numerical result was bit-exact in both logical output and physical
storage, with exact shape/strides/Version0/zero offset and fresh
Tensor/Storage/Version identities. An independent pure reader checked
18 full owners (135 live tensor records) and exact wire ancestry.
Both inputs-first and results-first release orders were verified on
live survivors. No fault, invalid, pressure or unrelated route used.

One original fourth case was measured only in Torch. Its hidden-width-one
second matrix product produced two -0 words where the frozen oracle
predicted +0. That original fixture and partial receipt are preserved
unchanged; this case is excluded from v2 and was never issued to HIA.
No parity or performance claim beyond the three exact finite fixtures,
no claim about signed-zero behavior in HIA, no full frontend or default
qualification.
