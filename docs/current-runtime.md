# Build and qualify the current HIA runtime

## Source and tools

The canonical application is [`tools/current_runtime/source/runtime.hepta`](../tools/current_runtime/source/runtime.hepta). Its [manifest](../tools/current_runtime/source/manifest.json) fixes complete functions, ordered units and native object/image identities. The current candidate has **765 core functions in 15 units (13 core, RNG and owner support)**. HIA machine code performs all tensor numerical work, allocation, ownership, AD, RNG and command execution in a static Linux x86-64 ELF.

Use Hepta source commit `f33d8da316de8fdf27b504371f1a5310997ff548`. The exact bootstrap sequence in [Current HIA Runtime CI](../.github/workflows/current-runtime.yml) reproduces the initial compiler, expanded profile, borrowed modules, owned ABI2, interactive sessions, ABI3 stack tools and entropy linker. Its initial bootstrap uses Python and GNU ld.

| Native tool | SHA-256 |
| --- | --- |
| ABI3 module compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Four-argument entropy session linker | `761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e` |

Build the entropy tool with a fresh actual receipt:

```sh
python3 -I -S -B tools/current_runtime/entropy_linker/build_tool.py \
  --upstream /absolute/path/to/fixed-hepta-source \
  --stack-build /absolute/path/to/stack-budget-tools \
  --output /new/current-entropy-tools
```

Generator admission requires its actual `build.json`, all 40 inputs and both native process streams. A copied executable alone is insufficient. See the [entropy tool contract](../tools/current_runtime/entropy_linker/README.md).

## Independent reference environment

Use CPython 3.12 on Linux x86-64 and the exact official CPU wheel in [`pytorch-oracle-lock.json`](../compatibility/pytorch-oracle-lock.json). From the repository root:

```sh
/path/to/pinned-env/bin/python - <<'PY'
from pathlib import Path
import json
import subprocess
import sys

lock = json.loads(Path('compatibility/pytorch-oracle-lock.json').read_text())['oracle']
report = Path(lock['install_report'])
report.parent.mkdir(parents=True, exist_ok=True)
wheel = lock['wheel']['url'] + '#sha256=' + lock['wheel']['sha256']
subprocess.run([sys.executable, '-m', 'pip', 'install',
                '--disable-pip-version-check', '--force-reinstall',
                '--index-url', 'https://download.pytorch.org/whl/cpu',
                '--report', str(report),
                '-r', 'compatibility/pytorch-reference-requirements.txt', wheel], check=True)
result = subprocess.run([sys.executable, '-B', '-W', 'error', '-c',
                         'import numpy; import torch'], capture_output=True, timeout=60)
assert result.returncode == 0 and not result.stdout and not result.stderr
PY
```

Retain the real pip report at `build/pytorch-install.json`. All eleven original provenance checks remain required. The SHA-pinned NumPy wheel resolves the missing-dependency warning while preserving strict reference-stderr admission. Both libraries are test dependencies.

Use the **same CPython 3.12 interpreter** for the coordinator and live readers; original source inverses contain Python-version-sensitive AST identities. The BF16 exact-bit reference records and requires the actual AVX2/AVX512 path. The Half multiplication, negation and addition/subtraction references explicitly select AVX2 in their own processes and retain the original single-thread setting. Their scalar and vector NaN results are compared bit for bit. DEFAULT dispatch and other architectures remain separate full dtype/platform requirements.

## Validate and cold-build

Source-only validation:

```sh
python3 -I -S -B tools/current_runtime/build.py \
  --manifest tools/current_runtime/source/manifest.json \
  --source-root tools/current_runtime/source \
  --check-only --output /new/current-source-check
```

Cold-build every application object:

```sh
make HEPTA_SOURCE_ROOT=/absolute/path/to/fixed-hepta-source \
     HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools \
     HEPTA_ENTROPY_BUILD=/absolute/path/to/current-entropy-tools
```

Equivalent standalone build:

```sh
python3 -I -S -B tools/current_runtime/build.py \
  --manifest tools/current_runtime/source/manifest.json \
  --source-root tools/current_runtime/source \
  --compiler /absolute/path/to/stack-budget-tools/native/stack-module-compiler \
  --linker /absolute/path/to/current-entropy-tools/generator-session-linker \
  --output /new/current-runtime-build
```

The compiler retains its original limits: 1 MiB source/object, 256 functions including imports, 2,560 blocks, 256 blocks per function, 8,192 instructions and 16 parameters per unit. The linker admits 64 objects and 8 MiB input/output. Native stack admission remains 8,384,512 bytes inside the guarded 8 MiB stack. The independent inspector checks object metadata, decoded frames and the actual call graph.

The build receipt binds all source/tool/object/image bytes, modes, real exits and both stream EOFs. `PASS_NATIVE_BUILD_ONLY` describes compilation and linking. Runtime qualification is separate.

## Twenty-four mandatory groups

```sh
make test HEPTA_SOURCE_ROOT=/absolute/path/to/fixed-hepta-source \
          HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools \
          HEPTA_ENTROPY_BUILD=/absolute/path/to/current-entropy-tools \
          HEPTA_TORCH_PYTHON=/absolute/path/to/pinned-env/bin/python
```

Or consume an explicit new build:

```sh
/path/to/pinned-env/bin/python -I -S -B tests/qualify_current_runtime.py \
  --build-directory /absolute/path/to/current-runtime-build \
  --tool-build /absolute/path/to/current-entropy-tools \
  --stack-build /absolute/path/to/stack-budget-tools \
  --torch-python /absolute/path/to/pinned-env/bin/python \
  --output /new/current-runtime-review
```

| Group | Cases | Preserved contract |
| --- | ---: | --- |
| `bool27` | 27 | Bool sum, full owner/RNG and allocation recovery |
| `original76` | 76 | Original view26/scalar32/view14/input4; 82,503 frames and 481,580 assertions |
| `integral24` | 24 | Full-width integer sums, overflow, layouts and lifetime |
| `sum_f32_17` | 17 | Per-element cast before sum and both allocation-failure stages |
| `current35` | 35 | Original lifecycle, complete owner equations and observer controls |
| `base270` | 270 | Original operators, NN, optimizers, training and prefix/refusal pairs |
| `sum32` | 32 | Original F32 sum rounding and accumulation order |
| `live32` | 32 | Actual Tensor/RNG/backward reference and full replay |
| `logical147` | 147 | Logical reductions across original dtypes, views and gradients |
| `resource7` | 7 | Original budgets, actual allocation refusals and recovery |
| `asstrided16` | 16 | Original copy/reduction/lanes/branches and complete owner/stream checks |
| `cat4` | 4 | Original raw packets and explicit capacity control |
| `metadata10` | 10 | Actual cat metadata, Version and gradient presence |
| `generator131` | 131 | Original24/native11/entry59/dispatch37 plus separate current-object probes |
| `public_f64_logical192` | 192 | Original dense F64 logical results and separate scope controls |
| `f64_48` | 48 | Original numeric/boundary/entry/OOM cases and 87 refusals |
| `half47` | 47 | All original cases; 239 entry requests comprise 173 unchanged refusals, 22 multiplication, two negation and 42 exact add/sub successors, including two original same-frame FE packets; four original allocation recoveries |
| `half_logical264` | 264 | Actual Half results/errors, exhaustive encodings and allocation recovery |
| `half_transactions12` | 12 | Direct successes and invalid-suffix rollbacks |
| `numeric_predicates` | 485 | Five classification operations and fresh actual Torch reference |
| `bf16_copy` | 280 | Exact BF16 words and metadata across scalar, strided, NaN and finite inputs |
| `half_mul` | 312 | Fresh actual Torch multiplication bits, metadata, promotion, errors, owner/input/RNG/lifetime, and all 22 former refusal recipes |
| `half_neg` | 371 | Exact negation bits and layouts, every Half word, scalar NaN payloads, all 24 rank-four dense permutations and two real OOM/recovery controls |
| `half_addsub` | 1006 | All public Half-containing add/sub pairs, fixed alpha=1, exact bytes, 40 historical single-command successes, two historical FE successes, Bool errors and four real OOM/recovery controls |
| **Total** | **3,845** | Every group and its original or explicitly mapped successor controls are mandatory |

Production cases use the same main image. Generator ABI/reordered probes are separate current-object images, each bound to its own actual execution. New source identities require a complete new run. The 765→761→758→742→740 whole-source inverse composes with unchanged predecessor proofs; source identity never substitutes for execution.

Original numerical assertions, tolerances, resource budgets and failure records remain. The complete 239-request Half entry now requires 173 original refusals and 66 explicitly mapped successes: 22 multiplication, two negation, 40 single-command additions/subtractions and two original 228-byte factory/FE-add frames. The FE owner extension derives three allocations and one promotion retirement from the fixed HIA source. It checks every word of the three descriptors, all eight owner tables, allocator offsets and free extents, original Storage payloads and RNG; it constructs no observed intermediate state. The complete linked `rng03` unit and its two scalar rounding leaves are included in the source call graph.

Each actual reference (312 multiplication, 371 negation and 1006 addition/subtraction cases) is generated once. Half47 shares all three. The multiplication group also binds the same 1006 reference for four former add/sub refusal controls; those original requests and numerical checks are retained. Actual process argv, interpreters, input files, outputs and EOFs remain bound. The original BF16 group keeps its exact-bit comparisons. Missing inputs, skips, timeout, incomplete EOF, altered inputs or failed physical readback prevent combined PASS. The CI archive preserves raw frames, full owner/storage snapshots, reference inputs, actual process streams and native tool receipts. Historical reference-interpreter gaps remain explicitly recorded.

The live32 cold-build adapter preserves its original predicates while mapping all seven unit/process inventory literals to the actual fifteen-unit, sixteen-process build. The preceding complete22 and old-head complete23 failures are retained. A focused corrected32 PASS does not substitute for the required complete execution on each new head.

Current results are in [the audit](current-audit.md). The 3,845 finite cases remain distinct from the complete 2,590 native declarations, 688 derivative declarations and 17 system contracts.

## Changes and retirement

Edit the canonical source and regenerate ordered units with [`partition.py`](../tools/current_runtime/partition.py). Cold-build every object and run affected qualification. Before deleting a predecessor path, migrate its live consumers, preserve original outcomes at immutable Git identities and run their current replacements. See [the history index](history.md); licenses and active reference/bootstrap inputs remain in the current tree.
