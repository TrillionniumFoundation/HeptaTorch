# Fixed Torch observations for public Half logical reductions

This package contains literal inputs and a reference-only observer. It has no expected result values. The 264 calls comprise six exhaustive encoding observations and 258 small shape, axis, special-value and error observations. The source has not yet been executed in a fixed Torch environment.

For each of `torch.float16` and `torch.bfloat16`, the exhaustive calls use all 65,536 raw u16 encodings without conversion through Python float or Float32:

| Operation | Input shape | Axes | Purpose |
|---|---|---|---|
| `aten.any.dims` | `[65536]` | `[]` | Observe every encoding's elementwise truth |
| `aten.all.dims` | `[65536]` | `[]` | Observe every encoding's elementwise truth |
| `aten.sum.dim_IntList`, `dtype=torch.bool` | `[65536, 1]` | `[1]` | Observe every encoding through a singleton Bool reduction |

The smaller calls cover rank 0 through rank 4; positive and negative zero; subnormals, infinities, quiet and signaling NaNs; opposite nonzero values; positive, negative and unsorted axes; `None` versus empty axes; `keepdim`; normalized duplicates; out-of-range axes; and empty input or output dimensions. Errors are observed from actual Torch and retained with their original type and message. No expected Bool output or accepted/error census is synthesized by the host.

Successful inputs are ordinary dense owned tensors with `requires_grad=False`. General views and Half autograd are outside this candidate's native scope. Source input bytes, physical storage, metadata, identity relations and the complete 5,056-byte CPU RNG are captured before and after each operation. Every actual result's logical and physical storage bytes are saved as complete content-addressed blobs. Standalone rows must equal the aggregate report.

## Required environment

Use an existing environment with the repository's exact official Torch wheel:

- Version: `2.15.0+cpu`.
- Wheel build commit: `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`.
- Wheel SHA-256: `0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3`.
- CPython 3.12, Linux x86_64.
- All 11 original provenance checks in the unchanged `tests/pytorch_oracle.py` must pass. Its whole source, the oracle lock and the upstream lock are pinned. The actual environment's pip installation report is bound and checked by those original predicates; a historical installation report must not be substituted for a new environment's producer input.

The launcher does not create an environment, install packages, probe a remote service, or execute a native HIA image. An archived Python executable alone is not a usable Torch environment.

## Execution

Invoke the launcher from a coordinator or supervised CI step, using the manifest SHA from the candidate handoff:

```sh
python3 -I -B /path/to/reference-producer/produce.py \
  --python /path/to/fixed-wheel-venv/bin/python \
  --repository /path/to/HeptaTorch \
  --expected-producer-sha256 PRODUCER_MANIFEST_SHA256 \
  --output /path/to/fresh-half-logical-reference
```

The requested venv Python spelling is retained in `argv`; only its executable identity is resolved. Before execution, the launcher snapshots that executable and binds the complete command and the resolved entity in `execution/intent.json`. The unchanged repository process supervisor retains both physical streams, PID/start-time records, exit code and EOF. The worker checks that its own actual argv, executable and process incarnation agree with that intent. The producer checks all source inputs again after the observation and retains any failure without deleting intermediate evidence.

The result is `reference/report.json` with status `OBSERVED_HALF_LOGICAL_REFERENCE`, accompanied by complete blobs and 264 standalone observations. `production.json` binds that report to its actual process and streams. These are reference observations, not a native compatibility qualification or a project completion gate.
