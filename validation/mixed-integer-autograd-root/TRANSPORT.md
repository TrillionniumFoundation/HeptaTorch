# Exact transport of the original cold7 archive

The original 15,368,228-byte XZ exceeds the connector's 16 MiB request limit
after base64 encoding. Its two opaque transport parts concatenate, in the order
in `transport/index.json`, to exactly the original archive. The archive SHA-256
is `0ee1ad222782b03e3f065153c08ba4e14cfec8e02b41fe1cefd5351ead26e5bf`;
its original Git blob SHA-1 is `0a63e420a48235ad8139afb2f10219e4c7ecf2a2`.
No report, tar member, compression byte, or original manifest was changed.

The original whole archive remains in the original execution checkpoint. The
Git checkout distributes the two parts. `handoff-manifest-original.json` and
`handoff-path-map.json` retain their original import history; the transport
index records the later distribution representation separately.

Verify the entire archive, every one of its 2,225 member sizes, modes and hashes,
and the complete single XZ stream directly from the parts, without extraction:

```sh
python3 -B validation/mixed-integer-autograd-root/verify_evidence.py --parts-only --output NEW_REPORT.json
```

The verifier also accepts the original whole archive when present. The
`--parts-only` flag deliberately prevents that fallback. Its original verifier
is retained under `transport/history/`; every original evidence predicate is
preserved. The actual byte review is separate from the seven native commands
already recorded in the immutable report. There were 474 inputs captured before
execution and 648 total bound files at completion. Neither count denotes APIs.

To reconstruct the original archive, concatenate `raw-evidence.tar.xz.part01`
and `raw-evidence.tar.xz.part02` in `transport/`, verify the whole SHA-256 above,
and set its mode to 0644. Its size must be exactly 15,368,228 bytes. Original
native/Torch results and the whole-project HOLD decision remain unchanged.
