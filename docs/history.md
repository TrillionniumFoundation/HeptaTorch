# 历史验证记录

默认运行时和验收入口见 [当前构建](current-runtime.md)。已结束的阶段记录通过下列固定 Git 快照查阅；其原 PASS、FAIL、输入、输出、文件模式和归档字节均留在原提交。当前工作树保留执行当前构建和测试所需的输入。

| 原目录 | 文件数 | 字节数 | 固定历史位置 |
| --- | ---: | ---: | --- |
| `validation/current-runtime-local` | 18 | 5,207,389 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/current-runtime-local) |
| `validation/current-runtime-20261118` | 6 | 5,298,195 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/current-runtime-20261118) |
| `validation/hosted-gradient-adam-distributions-af33` | 27 | 26,022,502 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/hosted-gradient-adam-distributions-af33) |
| `validation/mixed-integer-autograd-root` | 34 | 16,262,010 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/mixed-integer-autograd-root) |
| `validation/mixed-frozen35-accounting` | 55 | 16,169,939 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/mixed-frozen35-accounting) |
| `validation/hosted-integer-tensors-343` | 6 | 6,237,470 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/hosted-integer-tensors-343) |
| `validation/differentiable-view-r1-native14-history` | 6 | 5,210,659 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/differentiable-view-r1-native14-history) |
| `validation/differentiable-view-r2-native14` | 6 | 4,032,975 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/differentiable-view-r2-native14) |
| `validation/hosted-adam-distributions-3e75b3a` | 23 | 3,938,511 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/hosted-adam-distributions-3e75b3a) |
| `validation/differentiable-view-r3-build-history` | 5 | 3,442,402 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/differentiable-view-r3-build-history) |
| `validation/hosted-reference-9a20ead` | 18 | 3,428,121 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/hosted-reference-9a20ead) |
| `validation/hosted-adam-distributions-predecessors-abc2046` | 7 | 2,643,595 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/hosted-adam-distributions-predecessors-abc2046) |
| `validation/differentiable-view-r2-build-history` | 5 | 2,142,143 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/differentiable-view-r2-build-history) |
| `validation/differentiable-view-r3-multiaxis72` | 8 | 2,120,953 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/differentiable-view-r3-multiaxis72) |
| `validation/hosted-native-int64-2e66` | 32 | 1,523,436 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/hosted-native-int64-2e66) |
| `validation/mixed-integer-autograd-integration-review` | 13 | 1,289,824 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/9ea2ec5522685a8b1a0080edda88b89dec094474/validation/mixed-integer-autograd-integration-review) |

以上 16 个目录共 269 个文件、104,970,124 字节。删除前逐一核对 Git blob、长度和文件模式，并核对所有已跟踪文本中的路径引用；当前构建、测试、CI、能力清单及活跃冻结包不读取这些目录。原 736 资格计划中的历史路径按其创建时的提交解释，该计划保持原字节。

需要重现这些历史运行时，可单独检出固定提交：

```sh
git worktree add --detach /path/to/historical-heptatorch 9ea2ec5522685a8b1a0080edda88b89dec094474
```

原始参考包、失败控制、工具链输入和许可证继续随仓库提供。原生源码、当前案例清单和完整 2,590 条算子／688 条导数／17 项系统验收目标不受此次整理影响。当前候选的数值结果由对应源码与映像的完整运行判定。

## 后续整理：R2 与 R4 已结束的失败阶段

下列三个阶段的运行记录已从当前工作树退休，全部原始文件仍由固定提交 `0e006aebddfc9de564d5d14879d7aca190817ce7` 保存。原来的失败、成功子集、输入输出及归档均保持原身份；它们不计入当前映像的验收结果。

| 原目录 | 文件数 | 字节数 | 固定历史位置 |
| --- | ---: | ---: | --- |
| `validation/view-reshape-r4-scalar05-history` | 10 | 4,394,815 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/0e006aebddfc9de564d5d14879d7aca190817ce7/validation/view-reshape-r4-scalar05-history) |
| `validation/view-reshape-r4-scalar06-history` | 8 | 9,426,991 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/0e006aebddfc9de564d5d14879d7aca190817ce7/validation/view-reshape-r4-scalar06-history) |
| `validation/differentiable-view-r2-input4-history` | 7 | 4,257,121 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/0e006aebddfc9de564d5d14879d7aca190817ce7/validation/differentiable-view-r2-input4-history) |

本次移除 25 个文件、18,078,927 字节。已核对构建、测试、CI、源码恢复代码及其固定清单，这些目录没有活跃读取依赖；清理不改动原生实现、案例、参考结果、许可证或工具链。

`tests/differentiable_view_input4_review/README.md` 属于已固定 SHA 的原始测试包，因此保持原字节。它提到的 R2 历史路径按上表固定提交解释；测试包的原 manifest 和比较器继续完整保留。

## 后续整理：整数、优化器与恢复过程归档

以下六个完整过程目录及其五份未被执行代码读取的历史摘要，统一退休到同一固定提交 `66582c67f76cee822e942c73435b5cebe126a132`。原记录、失败、源文件清单和归档字节仍可按该提交复核。

| 原路径 | 文件数 | 字节数 | 固定历史位置 |
| --- | ---: | ---: | --- |
| `validation/integer-byte-evidence` | 17 | 20,497,229 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/66582c67f76cee822e942c73435b5cebe126a132/validation/integer-byte-evidence) |
| `validation/tensor-to-frozen35-remote09b` | 43 | 13,029,517 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/66582c67f76cee822e942c73435b5cebe126a132/validation/tensor-to-frozen35-remote09b) |
| `validation/adam-distributions-root` | 8 | 4,794,855 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/66582c67f76cee822e942c73435b5cebe126a132/validation/adam-distributions-root) |
| `validation/cpu-distribution-native-history` | 4 | 2,027,537 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/66582c67f76cee822e942c73435b5cebe126a132/validation/cpu-distribution-native-history) |
| `validation/recovery-integration-review` | 41 | 658,029 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/66582c67f76cee822e942c73435b5cebe126a132/validation/recovery-integration-review) |
| `validation/recovery-integration-independent-review` | 8 | 239,569 | [原目录](https://github.com/TrillionniumFoundation/HeptaTorch/tree/66582c67f76cee822e942c73435b5cebe126a132/validation/recovery-integration-independent-review) |
| `validation/native-dropout/base-integration.json` | 1 | 49,994 | [原文件](https://github.com/TrillionniumFoundation/HeptaTorch/blob/66582c67f76cee822e942c73435b5cebe126a132/validation/native-dropout/base-integration.json) |
| `validation/adam-distributions-root-native.json` | 1 | 1,595 | [原文件](https://github.com/TrillionniumFoundation/HeptaTorch/blob/66582c67f76cee822e942c73435b5cebe126a132/validation/adam-distributions-root-native.json) |
| `validation/adam-distributions-archive-transport-review.json` | 1 | 1,392 | [原文件](https://github.com/TrillionniumFoundation/HeptaTorch/blob/66582c67f76cee822e942c73435b5cebe126a132/validation/adam-distributions-archive-transport-review.json) |
| `validation/cpu-distributions-base-integration-review.json` | 1 | 10,310 | [原文件](https://github.com/TrillionniumFoundation/HeptaTorch/blob/66582c67f76cee822e942c73435b5cebe126a132/validation/cpu-distributions-base-integration-review.json) |
| `validation/cpu-distributions-root-native.json` | 1 | 25,629 | [原文件](https://github.com/TrillionniumFoundation/HeptaTorch/blob/66582c67f76cee822e942c73435b5cebe126a132/validation/cpu-distributions-root-native.json) |

本次移除 126 个文件、41,335,656 字节。两个恢复审查目录作为完整历史包共同退休，其固定 manifest 不被改写。构建、比较器、源码恢复程序及其清单没有指向这些路径的活跃读取；它们也不属于 PR48 完整原始 CI 归档中的仓库输入。767 候选新增执行代码已另行检查，未增加这些依赖。

仍保留的旧路径按[当前构建与验收依赖](runtime-dependencies.md)逐项迁移。每次删除都需要核对真实消费者；目录名称、文件后缀或较早日期不构成删除依据。

### 只封存当前候选的资格计划

原 736、740、742、758、761、765 六份 `qualification-plan.json`（共 20,743 字节）也退休到[上述固定提交](https://github.com/TrillionniumFoundation/HeptaTorch/tree/66582c67f76cee822e942c73435b5cebe126a132/validation)。当前封存器只额外收录 767 计划。旧计划不提供数值参考、源码逆变换或实际输入清单；PR48 原始归档确认四份计划只是额外归档成员，其声明输入引用均为零。

修改仅移除 `tools/seal_current_ci_evidence.py` 的四个旧计划路径常量；全部函数体、25 组要求、必要输出、工具链和原始参考来源检查保持完整。实际执行结果、失败归档、完整输入以及仍被读取的源文件和 manifest 保留原字节。以上两部分合计退休 132 个文件、41,356,399 字节。
