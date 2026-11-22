# Opt-in logical-not runtime integration

This candidate patches exactly three bodies (`entry`, `predicate_command`,
`predicate_bits`) from the frozen [800-function #57 source](https://github.com/TrillionniumFoundation/HeptaTorch/tree/118560581713b2c2ed3e6a12e5ab5770d3ebb385).
It wires **experimental opcode 127** through the existing pointwise
predicate allocation/error path, returning a fresh Boolean Tensor. The
numeric truth and original Bool-only in-place opcode 102 paths are unchanged.
No file in the active runtime or mandatory qualification suite is modified.

The transformer reads and checks the exact #57 manifest, aggregate source,
and all 15 source partitions before writing one new aggregate source outside
that source tree. Do not point its output into the production source tree.

## Build and observe the experimental image

Use the exact above #57 checkout (`BASE`), an empty private directory
(`OUT`), and the pinned ABI3 compiler (`COMPILER`) and entropy session
linker (`LINKER`) described in [the build guide](../../../docs/current-runtime.md).
Run from the directory containing these opt-in tools (`TOOLS`); use
absolute paths for the variables.

```sh
src="$BASE/tools/current_runtime/source"
python3 -I -B "$TOOLS/prepare_runtime_candidate.py" \
  --base-source-root "$src" --output "$OUT/candidate.hepta"
python3 -I -S -B "$BASE/tools/current_runtime/partition.py" \
  --runtime "$OUT/candidate.hepta" --rng "$src/units/rng03.hepta" \
  --owners "$src/units/owners.hepta" --name current800_typed_logical_not \
  --prefix runtime800 --output "$OUT/candidate-source"
python3 -I -S -B "$BASE/tools/current_runtime/build.py" \
  --manifest "$src/manifest.json" --source-root "$src" \
  --compiler "$COMPILER" --linker "$LINKER" --output "$OUT/base-build"
python3 -I -S -B "$BASE/tools/current_runtime/build.py" \
  --manifest "$OUT/candidate-source/manifest.json" --source-root "$OUT/candidate-source" \
  --compiler "$COMPILER" --linker "$LINKER" --output "$OUT/candidate-build"
python3 -B "$TOOLS/test_runtime_candidate.py" \
  --protocol "$BASE/tests/differentiable_view_native_review/protocol.py" \
  --baseline "$OUT/base-build/heptatorch" --binary "$OUT/candidate-build/heptatorch" \
  --evidence-dir "$OUT/evidence"
```

Both locally cold-built images generated 15 fresh ABI3 objects. The
baseline reproduced every #57 original object/image pin; the candidate
source SHA-256 is `e3fe2e56a812dec618716ae14f7b95b821d4dc3195caf4dd19ee54c96fe0b4ba`
and the candidate ELF SHA-256 is
`0cd10504717c5e5b01114455c55030e63a22b62bb1ba43f188355d399d6c158a`.
The runner authenticates both executable identities and the original
production protocol/transport before executing them.

This same-image native campaign ran 849 frames: 18 dense, scalar or empty
factory cases over six CPU dtypes, one transposed F64 view, one successful
same-frame factory/predicate transaction, 72 comparisons of the six original
predicate opcodes against the baseline image, and 12 explicit refusal or
rollback controls. It also lowered the actual child address ceiling, reached
allocation status 3 on the fourth attempt, checked complete Owner bytes,
restored the original limit, and successfully retried the identical request
for 65,536 F16 values. Checks included Boolean contents, shape, transpose
strides, distinct Storage/Version and unchanged input. All three native
sessions exited with zero status and complete streams.

The initial 849-frame campaign used an independent numerical oracle. The
separate strict Torch replay below verifies numerical and output metadata
parity on the **same** candidate ELF; it does not replace the transaction,
OOM or pre-existing opcode evidence.

## Strict fixed-wheel Torch replay on this candidate

With the exact candidate ELF already built, use a clean CPython 3.12 Linux
x86-64 environment and the repository's existing pinned wheel install
procedure. Run from the checkout of these tools (not from the historical
owner checkout). Supply absolute paths for `BASE`, `TOOLS`, and `OUT`.

```sh
python3.12 -m venv "$OUT/torch-env"
PY="$OUT/torch-env/bin/python"
mkdir -p "$OUT/torch-provenance"
wheel=$("$PY" -c 'import json; w=json.load(open("compatibility/pytorch-oracle-lock.json"))["oracle"]["wheel"]; print(w["url"]+"#sha256="+w["sha256"])')
"$PY" -m pip install --disable-pip-version-check --force-reinstall \
  --index-url https://download.pytorch.org/whl/cpu \
  --report "$OUT/torch-provenance/install.json" \
  -r compatibility/pytorch-reference-requirements.txt "$wheel"
"$PY" -s -E -B -W error "$TOOLS/qualify_runtime_torch.py" \
  --protocol "$BASE/tests/differentiable_view_native_review/protocol.py" \
  --binary "$OUT/candidate-build/heptatorch" \
  --oracle-lock compatibility/pytorch-oracle-lock.json \
  --install-report "$OUT/torch-provenance/install.json"
```

The replay refuses any other native image or oracle lock and checks PEP 610,
the pip install report, the distribution metadata digest, and the loaded
Torch version/build/backend before numerical comparisons. It uses an actual
native wire session and actual `torch.logical_not`, not a Python model of
the HIA operation.

The [recorded local replay](strict-torch-runtime-evidence.json) has 28
cases and 144,027 elementwise comparisons in 373 native frames. It covers
F32, Bool, I64, F64, F16 and BF16 in dense, scalar and empty forms, every F16
and BF16 raw word, and transposed, offset, stride-with-holes and zero-stride
F64 views via the existing F64 view opcode. Output Boolean bytes and
shape/strides/offset/grad flag match Torch; native output has distinct
Storage/Version and input is unchanged. The baseline image, altered lock,
and install-record hash substitutions were separately refused.

This only qualifies the scoped experimental opcode. Green hosted main
checks used the main ELF, not this candidate. Neither this replay nor the
849-frame campaign enables a production opcode, qualifies every layout
or the full runtime, or changes the original mandatory tests or any
complete-contract status.
