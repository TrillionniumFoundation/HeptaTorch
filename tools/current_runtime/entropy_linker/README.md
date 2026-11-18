# Fixed entropy linker: tool-only cold rebuild

This package executes the unchanged **tool-only prefix** of the original
Generator builder, SHA
`56cddf74330705c3058752f65858c0cb021ec2c3da905040dd0113ae986b2552`.
The selected 51 whole AST statements are original lines 83–185. They start at
the fixed Hepta contract check and end after linking the newly compiled entropy
linker object. `main()` is never called, and the subsequent RNG, model, owner
and Generator application build section is never selected.

The prefix keeps the exact original linker/startup authoring transformations.
It may execute exactly two programs, in this order:

1. `STACK/native/stack-module-compiler` on the newly authored `linker.hepta`.
2. `STACK/native/stack-linker-linker` on a real `HIASTKL1` request containing
   only the actual newly compiled object, with entry `f0`.

It never executes the produced `generator-session-linker`, a Tensor image, an
old Generator application or Torch. The native compiler/linker calls retain the
original PATH-empty environment and 90-second timeout. Actual stdin/stdout/
stderr, exit status, both EOF observations and input hashes are retained for
each invocation. All bound inputs include physical mode and are rechecked.
Any failure retains its output directory and a failing `build.json`.

## Exact inputs

| Input | SHA-256 |
|---|---|
| Module compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Tool linker | `fa3065b5dc42bfe09fd29cf773c5adc5a9296c125a6a161daecf2fa885ea160b` |
| Base linker authoring | `b0adabf3db5bdda3ee46c0c74630938c0f1532e97bc532d65529c3ea0af83ac7` |
| Base startup authoring | `1271a7805412930cf682d1f08f5aeed2955862f40747d2a4253784f4055fe8b0` |
| Hepta authoring contract | `8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c` |

Every original authoring dependency in `AUTHORING_FILES` is bound to that
contract. Every actually imported upstream module must be in the fixed set.
The report distinguishes the bound set from the actually loaded subset; it does
not pretend that the unused application authoring tail was executed. The
expected upstream revision is `f33d8da316de8fdf27b504371f1a5310997ff548`;
the byte contract is verified, with no separate Git command or revision claim.

## Exact required outputs

| Output | SHA-256 |
|---|---|
| `linker-authoring.txt` | `771160ccfdde84bec0dc8fafd16de78bf4327add86b0913644dd13d05414b831` |
| `linker.hepta` | `7bba866ece4f33fb6b1973e33475d868376f2b804d5ce81116cf206cfff31c34` |
| `startup-prefix.bin` | `75b74f8441acd38b84d5a4784ab3a37f48652e06caa1ca938595aa1324424a00` |
| `generator-session-linker` | `761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e` |

Before compiler execution, the new HIA source, prefix and authoring identities
must already match. After linking, the actual tool bytes must match as well.
The 44-function linker authoring check requires exactly three changed functions
(`link`, `headers`, `module_entry`) and 41 unchanged function ASTs.

## Root execution

```sh
python3 -S -B "$PACKAGE/build_tool.py" \
  --upstream /home/qian-qi/heptatorch-continuation-20261116/toolchain \
  --stack-build /home/qian-qi/heptatorch-continuation-20261116/runs/dview-r1-fixed-toolchain-cold-02/hepta-stack-budget \
  --output "$FRESH_TOOL_BUILD"
```

Success is `PASS_NATIVE_ENTROPY_LINKER_BUILD_ONLY`, with exactly two successful
native tool processes, complete stdout/stderr, unchanged inputs and all four
required output identities. The tool still has not been executed as a session
linker at that point; the separate combined-runtime cold build does that.

For a pure source-boundary check only:

```sh
python3 -S -B "$PACKAGE/build_tool.py" --review-only --output "$FRESH_REVIEW"
```

That reports `SOURCE_AUDITED_NATIVE_TOOLS_NOT_RUN` and zero processes. The local
author performed this source check only; no native bootstrap result is claimed.
