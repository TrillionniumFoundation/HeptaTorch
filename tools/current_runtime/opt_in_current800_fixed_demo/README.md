# Fixed CPU application demos

On Linux x86-64, with Python 3.10 or later and the exact already-built
HeptaTorch #71 current800 opt-in ELF, run:

    python3 run_demo.py --binary /path/to/heptatorch --both

Use either `--inference` or `--sgd` to choose a single profile.
The binary SHA-256 must be
`5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d`
(current800 combined source SHA-256
`f8f924a82763a28d8ad12adbd371f25458ed86d1fbdee2406bb93912847afb4b`).
This runner does not build or download the binary. Its temporary
directory must be writable and allow executing the verified private
copy. Do not substitute an unpinned/default image.

The inference demo verifies three fixed F32 fixtures and all 15
intermediate/final stages. The SGD demo verifies one step: initial
loss 8, gradient [-4,8], updated weight [1.5,0], final loss 2.25, and
one parameter Version increment. Outputs are parsed from the native
wire. Both profiles use only known fixtures and live opaque handles;
there is no arbitrary graph, opcode, packet, model, or network input.
No Torch, third-party libraries, private process inspection or shell
is required. Native storage access in this caller is through the
ordinary wire, not direct owner-memory inspection.

The published inference v2 scope is unchanged. The fourth original
signed-zero mismatch remains explicitly excluded and is not sent to
the native runtime. This is not a general-purpose frontend, training
or optimizer suite, default qualification, or complete PyTorch parity.

Run the standard-library-only positive call-plan and packet checks with:

    python3 -B -m unittest -v test_demo
