# Offline ordinary evidence gate

In this directory, run:

```sh
python3 -I -B ordinary_gate.py --archive ./ordinary-raw-evidence.tar.gz
python3 -I -B test_ordinary_gate.py
```

The first command requires an explicitly supplied, previously retained #71
ordinary raw archive (1,654,860 bytes, SHA-256
`01d79608824cc6ae5962ee3cfffb8739d1b30428b84cc4b62f4e31697b32707a`).
The archive is not included here. The gate does not locate, fetch or publish
it. With published files alone, no raw-evidence pass is possible.

The gate binds the sealed fixture, locked Torch receipt, pinned image
manifest, offline reader and expected summary, and independently rereads
the archived request/response and owner bytes for the three lane identities.
It prints a machine-readable proof only when every check passes. It opens no
compiler, Torch, native executable, external process, credentials, GitHub
workflow or network connection. Temporary extraction is bounded and uses
only the expected ordinary evidence files.

A pass concerns the sealed ordinary coexistence observations (171 combined
and 171 separate reference outputs). It is not a new execution receipt,
hosted qualification, mutation or unsupported-path coverage, or default
routing approval. This entrypoint provides no fresh-run or rebuild option.
