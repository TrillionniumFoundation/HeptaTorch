# Owned Tensor v2: prepared real PyTorch oracle

This is a **28-case test adapter**, not a numerical runtime. It is prepared for the repository's existing pinned CPU PyTorch installation and its unchanged 11 provenance checks. Local execution found no installed `torch`; the local result is **NOT_RUN**, with zero native cases executed. The separate exact-rational/native migration review is not presented as a PyTorch result.

## Inputs and reproducibility

The adapter verifies both original files before executing any of their code:

| Input | SHA-256 |
| --- | --- |
| `tests/pytorch_session_oracle.py` | `35fa5fa3e1b740f901c11d46dca484079b77805004586c03ae42b35be2265577` |
| `tests/pytorch_oracle.py` | `ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae` |
| Initial candidate executable | `05c5544da2b7b34a8db9bd019a3ad0fa930f9ed5c18fd4b7582e7150f21eaeea` |
| Initial oracle lock | `ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944` |

The original support file runs under `runpy` with its usual non-main name, exactly as in the original session suite. It must pass all 11 existing provenance checks. The adapter does not replace Torch, disable provenance, run the support file's separate 270-case `main`, install a package, or regenerate expected numerical values from the native result. A different native successor requires an explicit `--expected-native-sha256`; it does not require changing the case inputs or tolerances.

Run from a checkout with the fixed PyTorch environment and its existing pip installation receipt:

```sh
python tests/pytorch_owned_tensor_oracle.py \
  build/heptatorch-owned-tensor build/pytorch-owned-tensor.json \
  --session-reference tests/pytorch_session_oracle.py \
  --reference tests/pytorch_oracle.py \
  --lock compatibility/pytorch-oracle-lock.json \
  --repo .
```

The first two positional paths and all named file paths can be absolute. `--repo` supplies the working directory needed by the unchanged support code's `compatibility/upstream-lock.json` and locked pip-report path. `--prepare-only` validates source custody, the reversible overlay, native identity, and the complete case plan without importing Torch or invoking the native runtime.

## Exactly what changes in the inherited suite

The adapter parses the original session source and selects its 12 class/function definitions plus the original 21 case-registration statements. It changes three byte-string literals in `Session.exchange`: outer request `HTOSES01` to `HTOWNR01`, outer response `HTOSRP01` to `HTOWNS01`, and body response `HTORSP01` to `HTORSP02`. It changes the expected body-response version from 1 to 2. A subclass of the unchanged reference `Program` changes handle generation from `generation << 8` to `generation << 16`, and changes only the first eight encoded body bytes to `HTORCH02`. New methods serialize command 38 (`full`) and commands 39/40 (value/gradient windows).

Six additional call sites perform version observation or assertion: one final alias assertion, two optimizer observation calls (initialization and each completed update), two MLP observation calls, and one assertion after each case's normal session close. The adapter reverses these exact edits in memory and requires the resulting AST of every original definition to equal the original AST. The old numerical statements, deterministic seeds, case registrations, and tolerances remain unchanged. The report records original definition hashes, overlay counts, AST hashes, and the complete fixed case plan.

The version observation calls add metadata frames. They do not change the numerical frames or parameter/state data. Every real response must arrive while stdin remains open. Failed application requests retain the inherited 32-byte/count-zero requirement and leave serializer ID bookkeeping uncommitted.

## Numerical and version acceptance

The original 21 cases keep their existing tolerances. The alias case now also requires absolute initial counters and counters after one explicit shared write to equal real `Tensor._version`, in addition to the original shared increment and independent clone checks. The former bounded-session profile's absolute-counter gap remains separately OPEN; a successful v2 test does not relabel its older evidence.

For every SGD, Adam, AdamW, and MLP update, the adapter records native parameter versions and the corresponding real Torch parameter versions. **Every direct parameter absolute-counter mismatch makes the case fail.** These mismatches accumulate until the end of that case so all original 64/128-step numerical inputs still run; they are not converted to delta-only acceptance. A numerical, metadata, or transport failure still stops its case immediately under the original behavior.

Native optimizer states are explicit zero tensors supplied before the first step. `torch.optim` constructs its states when needed. The report retains the initial absence of a Torch state, each later state version, and every absolute difference. State construction/lifecycle parity remains OPEN. This distinction does not weaken the direct parameter assertion, and the script never claims full optimizer state-version compatibility merely because numerical state values pass.

## Seven additional cases

| New case | Fixed input and acceptance | `rtol` / `atol` |
| --- | --- | --- |
| Paged descriptors | 300 retained leaves; nonuniform `matmul → transpose → index_select → gather → square` VJP twice; reset and stale-handle rejection; a real generation-1 handle | 0 / 0 |
| Large tensor | Native `full((512,1024), .125)`; real Torch square/sum/backward; every value and gradient read through bounded 131072-element windows; one actual SGD update; oversized all-at-once response refused | 0 / 0 |
| SGD across growth | 300 retained leaves, 32 actual Torch optimizer steps, deterministic gradients, state/parameter numerical comparisons, 1 MiB and 2 MiB temporary tensors at steps 1 and 17 | `5e-5` / `5e-6` |
| Adam across growth | Same prefix/growth/step schedule, actual Torch `exp_avg` and `exp_avg_sq` references | `1e-4` / `4e-6` |
| AdamW across growth | Same prefix/growth/step schedule; strict direct parameter version assertions retained | `1e-4` / `4e-6` |
| Empty broadcast matmul | `(262144,262144,0,1) @ (1,1,1,0)`, exact output shape `(262144,262144,0,0)`, true empty seeded VJP and both leaf metadata | 0 / 0 |
| Empty gather | Input `(262144,262144,0,1)`, axis 3, signed-int64 index shape `(262144,262144,0,0)`, true empty seeded VJP and metadata | 0 / 0 |

The fixed total is 28: 21 inherited cases plus these seven. Case count, unique names, order, and tolerances are checked again after execution. Counts from the separate native qualification suites are not added to this upstream-oracle denominator.

## Scope

The candidate uses real owned native state and a paged relative-offset arena. This profile remains CPU binary32, rank 0 through 4, bounded per-axis/numel/work budgets, and a fixed typed wire interface. It does not yet connect the separate general byte-storage service's independent Storage and Version objects to every numerical Tensor. No GPU, general dtype, Python frontend, foreign pointer, arbitrary allocator, or whole-PyTorch closure is claimed. The original inventory denominator remains 2590 and `full_rewrite_complete` remains false.
