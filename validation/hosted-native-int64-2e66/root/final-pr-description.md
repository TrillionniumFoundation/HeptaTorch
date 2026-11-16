## Problem and scope

The native runtime needs an integer arithmetic foundation for the remaining dtype work. This PR adds a standalone native HIA Linux x86-64 int64 executable for add, sub, mul, bitwise-and, bitwise-or and bitwise-xor, with exact two's-complement wrap semantics. Native HIA performs production arithmetic and packet I/O. It admits at most 48,000 operand pairs and rejects nine malformed packet classes.

## Source and current-main integration

The reviewed source head is `2e66ad094ca62d059a1a7ca5642a8ce27c40a949`, tree `51b2bf3b1d15e8662253668c0f6ee458885dd3de`, with ordered parents `0fd81b279c97dd969071c9efc51c8202fc16e45b` and admitted main `e0ea769c979236b0542af4b258890726cd01a99a`. All 3,833 existing main files retain their exact bytes and modes; 28 paths are added.

- HIA source: `251705542f4e117e3c9cdaec36cead8ab38252b24ebfef1104d765c546fd0716`.
- Fixed native compiler: `87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54`.
- Twice-reproduced native ELF: `fbb6cf1648385be200c94fc14ed73d1539cba3b0239aeea4f7866058c8395ebc`.

The original kernel, corpus, numeric expectations, refusal checks and assertions are unchanged. Additive observers preserve both compiler executions, the actual ELF, each original request/response/stderr/return code, and independently computed CPython or actual PyTorch expected bytes. CI exports a mode-preserving archive with strict offline readers. Empty requests are explicitly recorded as performing no Torch arithmetic operation.

## Actual validation of this head

All eleven GitHub checks completed successfully. Both dedicated int64 jobs passed all thirteen actual steps: PR run [43886148312](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43886148312) and push [43886136338](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43886136338). Each retains the actual 33-case, 351,174-value result with all eleven fixed PyTorch provenance checks. Independent byte readers verified the full compiler and per-case material: 2,057 CPython checks and 2,262 Torch checks per run.

Both full Native jobs passed all eighty actual steps: PR [43886148135](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43886148135) and push [43886136381](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43886136381). Independent reads of each new set of original artifacts passed 106,633 checks with zero failures. The PR report was separately reproduced by the integration owner with identical bytes:

- PR report SHA-256: `7ef5d7be530d70ad99013085de5030577f04e5350cf4d002cd4b565e5e6aa750`.
- Push report SHA-256: `f05cdd7dae1dc32f8427eff25f5d18578724b67da0b37dcf45456a2402c0d193`.

The three quick profiles (355, 428 and 446) each passed thirteen actual steps in [43886148242](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43886148242), followed by 52,151 independent checks. The checks retain original live32/frozen35 numerical results, all provenance requirements, complete journals, EOF/post-close Version behavior, the 283 explicit-state Version differences, and the fresh retained-gradient compilation gates. The actual PR merge checkout `1a28a87524e7dd6ffe94ab0ba032d42c4cfb2031` has the reviewed tree and the expected main/source parents.

## Evidence and remaining work

Original v1/v2 evidence, actual provenance controls and historical integer admission remain under `validation/integer-byte-evidence/` and `validation/integer-original-hosted-admission/`. The first hosted wrapper's ordering-only reader failure is preserved; its successor compares unordered checks by exact content and multiplicity while retaining exact order for deterministic archive parts and equality for every other field.

This standalone lane kernel does not qualify general Tensor construction, broadcasting, strides, aliases, mutation, promotion, dispatch, autograd or complete upstream schemas. Whole PyTorch acceptance remains HOLD: 0 of 2,590 complete native schemas, 688 derivatives in scope, and all seventeen whole-system gates open or in progress.
