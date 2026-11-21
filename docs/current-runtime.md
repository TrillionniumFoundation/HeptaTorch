# Build and qualify the current HIA runtime

## Source and tools

The canonical application is [`tools/current_runtime/source/runtime.hepta`](../tools/current_runtime/source/runtime.hepta). Its [manifest](../tools/current_runtime/source/manifest.json) fixes complete functions, ordered units and native object/image identities. The current candidate has **781 core functions in 15 units (13 core, RNG and owner support)**. HIA machine code performs all tensor numerical work, allocation, ownership, AD, RNG and command execution in a static Linux x86-64 ELF.

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

Use the **same CPython 3.12 interpreter** for the coordinator and live readers; original source inverses contain Python-version-sensitive AST identities. The BF16 exact-bit reference records and requires the actual AVX2/AVX512 path. The Half multiplication, negation, addition/subtraction, true-division, mixed public F64 and same public F64 arithmetic and typed ReLU references explicitly select AVX2 in their own processes and retain the original single-thread setting. Their scalar and vector NaN results are compared bit for bit. DEFAULT dispatch and other architectures remain separate full dtype/platform requirements.

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

## Thirty-one mandatory groups

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
| `f64_48` | 48 | All original numeric/boundary/entry/OOM recipes; all87 entry requests comprise63 retained refusals,one negation,22 mixed arithmetic commands andone FE successor;3,259 frames/17,923 assertions |
| `half47` | 47 | All original cases; 239 entry requests comprise 151 unchanged refusals, 22 multiplication, two negation, 42 exact add/sub and22 true-division successors, including two original same-frame FE packets; four original allocation recoveries;3,887 frames/20,133 assertions |
| `half_logical264` | 264 | Actual Half results/errors, exhaustive encodings and allocation recovery |
| `half_transactions12` | 12 | Direct successes and invalid-suffix rollbacks |
| `numeric_predicates` | 485 | Five classification operations and fresh actual Torch reference |
| `bf16_copy` | 280 | Exact BF16 words and metadata across scalar, strided, NaN and finite inputs |
| `half_mul` | 312 | Fresh actual Torch multiplication bits, metadata, promotion, errors, owner/input/RNG/lifetime, and all 22 former refusal recipes |
| `half_neg` | 371 | Exact negation bits and layouts, every Half word, scalar NaN payloads, all 24 rank-four dense permutations and two real OOM/recovery controls |
| `half_addsub` | 1006 | All public Half-containing add/sub pairs, fixed alpha=1, exact bytes, 40 historical single-command successes, two historical FE successes, Bool errors and four real OOM/recovery controls |
| `wide_neg` | 405 | Exact I64/F64 negation bytes, INT64_MIN, all F64 exponent classes, NaN payloads, layouts, independent owners,46 protocol controls and two actual OOM/recovery pairs |
| `half_div` | 1196 | True division with at least one Half operand; original RHS scalar precision, exact bytes and NaN payloads, all words, rank-aware promotion, layouts, 22 original division requests,32 packet controls and two real OOM/recovery pairs |
| `public_f64_arithmetic` | 2541 | Mixed public F64/F32/Bool/I64 arithmetic; direct I64 nearest-even conversion, exact bits, rank-aware promotion/layouts,23 original arithmetic successors,82 controls andfour real OOM/recovery pairs |
| `same_f64_arithmetic` | 5088 | Same public F64 pairs; exact bits including NaNs and signed zero, original3288 recipes plus1800 singleton rank/stride controls, independent ownership,88 protocol controls andfour real OOM/recovery pairs |
| `typed_relu` | 776 | Exact I64/F64/F16/BF16 ReLU bits, negative zero, all Half words, scalar/vector NaNs, rank0..4 layouts, 90 protocol controls and four actual OOM/recovery pairs |
| `signbit` | 97 | Seven dtype codes, exhaustive F16/BF16 sign bits, signed zero/NaN, thirteen protocol controls, six successful and six failed FE transactions, and two actual OOM/recovery pairs |
| `typed_index_select` | 866 | Six tensor dtypes, scalar/rank0–4, noncontiguous storage/offsets, exact copied bits, five actual AD cases, 78 protocol controls, two actual OOM/recoveries and the original Bool packet successor |
| `indexed_f32` | 331 | F32 gather/scatter_add/index_add with and without AD; explicit no-grad geometry, 193 actual gradients and full raw replay |
| **Total** | **15,145** | Every group and its original or explicitly mapped successor controls are mandatory |

Production cases use the same main image. Generator ABI/reordered probes are separate current-object images, each bound to its own actual execution. New source identities require a complete new run. The indexed781→index/signbit778→typed-ReLU775→same-F64773→mixed-F64773→771→767→765→761→758→742→740 whole-source inverse composes with unchanged predecessor proofs; source identity never substitutes for execution. The typed-ReLU successor changes only the old `unary` body, retains the other772 old bodies and adds `typed_relu_value` and `typed_relu_tensor`. Its [source review](../validation/current-runtime775/source-review.json) and [qualification plan](../validation/current-runtime781/qualification-plan.json) keep static identity, scoped numerical results and the full32-group execution distinct.

Original numerical assertions, tolerances, resource budgets and failure records remain. The complete 239-request Half entry now requires 151 original refusals and 88 explicitly mapped successes: 22 multiplication, two negation, 40 single-command additions/subtractions, two original 228-byte factory/FE-add frames and22 true divisions. The FE owner extension derives three allocations and one promotion retirement from the fixed HIA source. It checks every word of the three descriptors, all eight owner tables, allocator offsets and free extents, original Storage payloads and RNG; it constructs no observed intermediate state. The complete linked `rng03` unit and its two scalar rounding leaves are included in the source call graph.

Each actual reference (312 multiplication, 371 Half negation, 1006 addition/subtraction,405 I64/F64 negation 1196 Half true-division and2541 mixed public F64 arithmetic cases) is generated once. F6448 and wide_neg405 share the identical actual405 reference, process receipt and input bindings; the unchanged original87-request F64 recipe keeps63 refusals and requires the actual405 result for its original negation successor. It additionally shares one actual2541 reference with public_f64_arithmetic for22 single-command results andone original240-byte FE transaction, while both original Bool subtraction refusals remain. The F64 FE owner rule checks all84 words of three descriptors, all eight owner tables, first-fit extents and temporary-promotion reclamation. Half47 shares the multiplication, Half negation, add/sub and division references. The multiplication group also binds the same 1006 reference for four former add/sub refusal controls; those original requests and numerical checks are retained. The same actual1196 division reference also supplies exact numerical successors for the two original division controls in multiplication312 and four in addition/subtraction1006. Their original observer files are recovered byte for byte by a complete inverse before applying these explicit successors. Actual process argv, interpreters, input files, outputs and EOFs remain bound. The original BF16 group keeps its exact-bit comparisons. Missing inputs, skips, timeout, incomplete EOF, altered inputs or failed physical readback prevent combined PASS. The CI archive preserves raw frames, full owner/storage snapshots, reference inputs, actual process streams and native tool receipts. Historical reference-interpreter gaps remain explicitly recorded.

The typed_relu776, half_neg371 and wide_neg405 groups share one new actual776 ReLU reference and its exact process receipt. Four original command3/operation5 requests receive explicit numerical successors: Half cases369/370 map to ReLU774/775, and wide cases403/404 map to ReLU403/404. All original input words and request bytes remain; each old negation group retains46 control slots (typed_relu retains90), with two old slots now requiring exact result bytes, geometry, fresh Tensor/Storage/Version ownership, unchanged source/negation/RNG and complete release. The original239 Half and87 F64 entry partitions contain no ReLU requests and remain unchanged. See [the ReLU contract](native-typed-relu.md).

The live32 cold-build adapter preserves its original predicates while mapping all seven unit/process inventory literals to the actual fifteen-unit, sixteen-process build. The preceding complete22 and old-head complete23 failures are retained. A focused corrected32 PASS does not substitute for the required complete execution on each new head.

Current results are in [the audit](current-audit.md). The 15,145 finite cases remain distinct from the complete 2,590 native declarations, 688 derivative declarations and 17 system contracts.

## Changes and retirement

Edit the canonical source and regenerate ordered units with [`partition.py`](../tools/current_runtime/partition.py). Cold-build every object and run affected qualification. Before deleting a predecessor path, migrate its live consumers, preserve original outcomes at immutable Git identities and run their current replacements. See [the history index](history.md); licenses and active reference/bootstrap inputs remain in the current tree.

The two new groups have explicit exploratory local flags for development. The common coordinator never supplies them; strict qualification and sealing reject local reports without all original Torch installation provenance checks. See [the native indexing/signbit scope](native-index-signbit.md).

The early source-admission step checks the actual cold build, full inverse chains, current reader identities, generated owners and the exact group/case inventory. It also prepares all four AsStrided and three resource readers. Twelve independent regression controls exercise stale bytecode, incorrect case counts, inactive workflow guards, full file modes and actual reader context entry. The sealer requires all five preparation reports. These checks grant no runtime qualification and add no numerical cases; all32 numerical groups still execute. The independent [RNG packet consumer](../tools/current_rng_consumer/README.md) now links the current cold build’s RNG object and preserves all185 original tests, with strict new-tree CI still required.
