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

原 736、740、742、758、761、765 六份 `qualification-plan.json`（共 20,743 字节）也退休到[上述固定提交](https://github.com/TrillionniumFoundation/HeptaTorch/tree/66582c67f76cee822e942c73435b5cebe126a132/validation)。PR50的封存器当时只额外收录767计划。旧计划不提供数值参考、源码逆变换或实际输入清单；PR48 原始归档确认四份计划只是额外归档成员，其声明输入引用均为零。

修改仅移除 `tools/seal_current_ci_evidence.py` 的四个旧计划路径常量；全部函数体、25 组要求、必要输出、工具链和原始参考来源检查保持完整。实际执行结果、失败归档、完整输入以及仍被读取的源文件和 manifest 保留原字节。以上两部分合计退休 132 个文件、41,356,399 字节。

### 当前771计划替换767计划

771继续退休已被替代的767资格计划（1个文件、1,444字节），原字节保存在[固定提交f8ad6df](https://github.com/TrillionniumFoundation/HeptaTorch/blob/f8ad6df1d0d27776b7efb2c62db1356189523f39/validation/current-runtime767/qualification-plan.json)。当前能力清单、完整验收账本和文档改用771计划；封存器只额外收录这一份当前计划。它不提供数值参考或源码逆变换，原始实际执行结果保持独立身份。精确blob、模式和SHA见[当前计划退休记录](../validation/current-runtime771/retirement.json)。累计移出当前工作树427个过程文件、164,406,894字节；53个已替代源码文件另行记账。


## Additional retired process directories after current771

The following completed process packages are retained in immutable [Git history at 87702d6](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation). This batch removes 114 files / 12,877,907 bytes from the active tree. It changes no native implementation, numerical fixture, comparison, resource limit or ownership/EOF predicate. All 2,345 selected code, documentation and manifest blobs were checked against the exact Git tree, including every one of the 1,111 tracked Python/shell/workflow/Makefile entries. None of these paths appears in the independently read original767 archive's 150,929 members, 153,351 declared input bindings or 69 input lists; that prior failed gate is not a qualification of the new tree.

| Retired directory | Files | Bytes | Fixed original |
| --- | ---: | ---: | --- |
| validation/current-runtime740 | 3 | 1,946,203 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/current-runtime740) |
| validation/current-half-logical1391 | 1 | 2,363 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/current-half-logical1391) |
| validation/current-runtime-ci-failures | 1 | 2,181 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/current-runtime-ci-failures) |
| validation/differentiable-view-r3-ci-independent-review | 7 | 352,372 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/differentiable-view-r3-ci-independent-review) |
| validation/differentiable-view-r3-input4 | 7 | 3,244,014 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/differentiable-view-r3-input4) |
| validation/differentiable-view-r3-native14 | 7 | 4,097,893 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/differentiable-view-r3-native14) |
| validation/differentiable-view-r3-resource-ci-review | 12 | 224,798 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/differentiable-view-r3-resource-ci-review) |
| validation/hosted-retained-gradients-cef47 | 25 | 2,394,336 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/hosted-retained-gradients-cef47) |
| validation/integer-frozen35-accounting | 5 | 55,100 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/integer-frozen35-accounting) |
| validation/original16-input4-integration-review | 7 | 59,581 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/original16-input4-integration-review) |
| validation/tensor-to-frozen35-copied672 | 18 | 195,862 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/tensor-to-frozen35-copied672) |
| validation/view-reshape-r4-identity-epochs | 10 | 188,444 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/view-reshape-r4-identity-epochs) |
| validation/wrapped-scalar-resource-portable-prepare | 11 | 114,760 | [Original](https://github.com/TrillionniumFoundation/HeptaTorch/tree/87702d6b129198b28098c6c26d905a2e91117512/validation/wrapped-scalar-resource-portable-prepare) |

The immutable `tests/differentiable_view_input4_review/README.md` remains byte-identical because active original-source manifests bind it. Its historical R3 path mentions refer to the fixed snapshot above; the current upstream audit now links directly to the two exact historical reports. Previously recorded successful and failed executions are retained as their original Git blobs. The current-runtime740 plan mentioned in the earlier retirement record was already removed by a different batch; that historical record is unchanged.


## 当前773计划与分区替换

773同时退休上述114个过程文件、771旧资格计划（2,158字节）及旧分区生成摘要（3,788字节）。累计过程退休543个文件、177,290,747字节、38个完整阶段目录；完整原件和固定Git身份见[本批退休清单](../validation/current-runtime773/retirement.json)。原13个 `logical_half_*.hepta` 单元由规范773分区替换，全部原函数由完整源码逆变换约束；这13份替换单独记账，不计为13项被移除功能或净源码能力退休。

此处记录773第一阶段时旧143源码仍被跟踪；后续same-F64候选移除跟踪来源，按固定历史字节供旧消费者恢复，具体执行状态见下节。原始参考、失败记录、许可证及仍有真实消费者的包继续保留。

## 当前775计划、分区与原失败结果

775前身树当时只保留775的活动资格计划。same-F64的原计划（3,964字节）保存在[固定PR53提交](https://github.com/TrillionniumFoundation/HeptaTorch/blob/83ed6b41ea67aa37182d23159ec0cb436e632dac/validation/current-runtime-same-f64/qualification-plan.json)；13份被替换的773核心分区也由该提交保存。完整SHA、Git blob和文件模式见[775退休清单](../validation/current-runtime775/retirement.json)。新15单元保留全部旧函数，仅增加ReLU实现所需的两函数和一条分流；分区替换不计为删除功能。

PR52与PR53的完整Current门分别失败，原归档和结果不升级为通过。PR53失败原件14700680070已完整读回；25组／13,007例通过，另外3组／68例在原生案例前被来源校验阻止，资格后的源码缺失检查未执行。775需要重新完成全部29组／13,851例及原件复核。固定旧143源码已从跟踪树移除，但旧Native/F64/RNG及Make消费者仍实际恢复使用它；其完整语义迁移尚未完成。

## 历史778候选计划与分区

778候选当时只保留其活动计划及15个活动单元。13份被替换的775分区和775计划保存在固定PR55提交 `6378ae1b469e9e98a75547fa2670f7cc9e9badf6`，完整路径、Git blob、字节数与模式见[固定778退休记录](https://github.com/TrillionniumFoundation/HeptaTorch/blob/8993eb44a034c2ae388936c6a9ce777c6dad0cf6/validation/current-runtime778/retirement.json)。新增两原生契约需要完整31组／14,814例重新执行；历史成功与失败均保留各自身份。


## 当前781索引修复与消费者迁移

当前树保留781完整HIA源码、15个活动单元与唯一32组／15,145例资格计划。13份778分区和4份被替代候选元数据共2,626,149字节按原Git身份保存在 `8993eb44a034c2ae388936c6a9ce777c6dad0cf6`，见[781退休清单](../validation/current-runtime781/retirement.json)。这17项替换不计为删除功能，也不减少全量PyTorch的目标。

781补入无梯度gather/scatter_add/index_add的前向修复，并保留778的signbit和typed index_select。完整源逆链精确保留769个775原函数和全部旧AD助手。新的源码准备门先检查实际加载的读取器、Owner guards、完整资源来源及32组分母；原全部数值门仍要在本次冷构建上执行。

独立RNG packet消费者改为复用当前实际对象，完整185例及其原参考、错误、资源条件继续保留。它不再恢复143；Native/F64及显式历史Make消费者仍有真实依赖。只有本次严格新树运行与原件审读通过，才能把对应迁移标为验收完成。
