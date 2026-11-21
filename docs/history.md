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
