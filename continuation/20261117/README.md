# HIA 重写续接：Integral699 与 Bool700 源码候选

**状态：运行验证被 HeptaRuntime 隧道断连阻塞；整个 PyTorch 重写仍为 HOLD。**

本分支保存已经完成的实现、真实执行收据、完整源码和精确断点。目标仍为固定 PyTorch v2.15.0，commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`：2,590 条 native schema、688 条导数声明、17 项系统验收门禁。完整 schema 闭合数为 **0 / 2,590**。局部功能和测试进展不能改变这个全量分母。

本目录的机器可读入口是 [checkpoint.json](checkpoint.json)，完整文件索引是 [file-manifest.json](file-manifest.json)。代码中的 Tensor、Storage、数值规约与自动微分实现使用 HIA；Python 文件承担源码恢复、观察、封存和验收工作。

## 已实际完成与尚待验证

| 工作 | 已有证据 | 当前边界 |
| --- | --- | --- |
| PR36 | 已合并到 `deaddeebc1cbd0509048f085017b3f63fb8b9fc0`；完整托管产物准入 1,919,212 项比较，29,975 个输入不变 | 有限范围验收；没有全量 schema 闭合 |
| Sum03 / 696 函数映像 | 原 32 个 Sum 用例、同映像原 76 个继承用例已有实际通过；原32录制证据独立准入完成 | 后续移植、补充封存及对应新源码提交的托管验收仍待完成 |
| Integral699 冷构建 | 699 函数、9,300 个块、33,297 条指令、14 个全新编译单元、15 个进程、30 个完整 EOF；283 个输入不变 | 编译通过不代替全部运行语义验证 |
| Integral24 固定参考 | 24 个用例、260 个事件已在固定 PyTorch 实际观察；完整 4,586,123 B 报告已核 SHA | 5 个额外 dtype 控制不计入 Torch 参考分母 |
| Integral24 原生任务 | 原任务 `1a1e8fcdd64-392-410d` 已 exit 0，stdout/stderr 均完整 EOF；报告前缀声明 PASS | 完整报告 5,722,831 B；root 只完成 36 / 88 个分块读取。**独立完整准入仍 PENDING** |
| Integral699 原 76 项回归 | 纯准备门已通过，451 个输入不变，四份未来执行请求已冻结 | 四个原生套件尚未启动；旧696结果不转移到699 |
| Integral699 原 Sum32 | 源包及两个候选请求已冻结 | root 尚须完整源码审查；prepare / actual 均未启动 |
| 显式 Bool 求和候选 | HIA 源码、全函数变更关系、静态绑定检查及独立源审已完成 | 700 是候选函数数；尚未应用到完整 runtime、发射、编译或运行 |

已合并工作见 [PR36](https://github.com/TrillionniumFoundation/HeptaTorch/pull/36)。全量验收标准继续采用仓库的 [full-acceptance.json](../../compatibility/full-acceptance.json) 和 [upstream-audit.md](../../docs/upstream-audit.md)。

## 新的 HIA Bool 求和实现

源码入口：[sb_sum_bool.hepta](source-packages/sum-bool-source-01/hia/sb_sum_bool.hepta)。

候选支持原 Bool / I64 输入显式指定 Bool 输出。它逐个读取完整输入值，先判断是否为零，将值规范化为 0 / 1，再以 OR 累积。输出使用真实的独立 Bool Tensor、Storage 与 Version，按 **1 字节**初始化和存储，requires_grad 为 false。实际 strides、offset、holes、overlap 和 zero-stride 仍通过原寻址路径处理。

这项顺序对整数相消输入有意义，例如 `[1, -1]`：必须先转 Bool 再规约。代码未使用整数总和最后转 Bool，也未截取低字节。固定上游依据保存在 [upstream-source-review.json](source-packages/sum-bool-source-01/upstream-source-review.json)，包括 [ReduceOpsUtils.h](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/ReduceOpsUtils.h)、[SumKernel.cpp](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/SumKernel.cpp) 和 [TypeCast.h](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/headeronly/util/TypeCast.h)。

父源为 `afd54444849ed10f7b4af53231c9db6542854b70f33e3927f8b05dfbb23ed1e3`，699 函数。候选新增 1 个函数、修改 6 个函数、保留 693 个原函数的完整字节。状态标签为 HTSINT02 / HTOSIN02，描述符仍为 672 字节。

源审已经核对：

- 新函数的 7 条文字编辑可完整逆回原 integral Sum 函数。
- 六个修改函数的完整正向和反向字节关系成立。
- 11 个真实 ABI 依赖函数与实际699清单一致，祖先全文也已经完整读取。
- 七个函数、163 个基本块通过变量定义、分支参数和返回类型的静态检查。
- 独立审阅未在受审范围内确认具体阻断。

静态检查记录保留了检查器自身两次语法识别修正的原结果。HIA 源码没有为了这些识别错误被修改。以上检查均不等于真实 HIA 编译器验证。

后续需要新的 Bool 参考观测、明确的 Bool / 1字节 owner 分支、七项新 dtype 控制合同、冷构建和同映像回归。原699版本、原参考24以及原五项拒绝控制的历史记录保持冻结。完整要求见 [future-qualification-contract.json](source-packages/sum-bool-source-01/future-qualification-contract.json)。

## 两套录制验收源码

### Integral24

[源码包](source-packages/sum-integral-pure-admission-source-01/README.md)包含 26 个文件、391,759 B。它保留原24 Case、原 Native 构造器和原五控制函数，并为控制记录真实的“两次手工 capture”结构提供独立适配。

最终准入要求：

1. 原始请求与响应逐帧完全一致。
2. 24 个原 Case 和五个原控制分别执行全部原断言。
3. 全 owner、RNG、回滚、错误分类和双 EOF 符合原要求。
4. 所有子报告及最终 5,722,831 B 报告都能逐字重建。

原报告头部的 PASS 或进程 exit 0 均不足以通过这些门。封存器只证明原始字节保全，不授予语义资格。本源码包尚未运行。

### 旧 Sum03 的原76项补充验收

[Source02](source-packages/sum-source03-original76-recorded-source-02/README.md)有 18 个 payload、485,479 B。它通过原 owner 的真实 `bind_profile` API 绑定696映像，保留原 guard 和比较断言。Source01原件及其完整逆变换也一并保全。

原 Source01 的两个远端写入结果仍为 UNKNOWN。恢复脚本先核对两个命名空间的所有已有成员；已有文件必须在完整字节、哈希和模式上相同才能复用，缺件仅 create-only 新建，遇到不一致保留现场并失败。脚本见 [materialize_saved_source.py](recovery/materialize_saved_source.py)，尚未运行。

Git 不能表达0600与0660等完整权限差异。本目录中的源码文本是恢复载体，真实主机恢复后须按照原清单重新检查权限、输入和路径，才能执行。

## 当前连接阻塞与精确续接

2026-11-17 11:05:44 UTC 的连接复查返回：

`404 / tunnel_client_not_seen`

服务消息为 “Tunnel-client has not been seen for 300 seconds. Ensure tunnel-client is running and connected.” Mac 与 ROG 在此前检查中均受同一连接故障影响。独立云执行入口也返回 exec-server 缺失，当前没有可调用的环境修复工具。GitHub 连接可用，因此源码与断点已保存在此分支。

恢复动作是使现有 **Mac HeptaRuntime / tunnel-client** 重新上线，随后验证原 Mac → ROG 执行链路。没有假定其他主机、账号或凭证。

运行续接顺序：

1. 确认连接恢复及没有并发父执行器。先完成原 native24 剩余52个分块的固定版本读取，核对完整 SHA；不重新启动已结束的任务410d。
2. 对账原 Source01 两次 UNKNOWN 写入，以及旧失败归档的 UNKNOWN04 上传块，复用原请求和 upload_id。
3. 在原699映像上串行运行已审查的原76四套件，再完成原32的完整源审及首次执行。
4. 恢复、核对新录制源码，冻结真实执行收据与 seal plan，执行完整封存及独立录制准入。
5. 对 Bool 候选进行完整父源应用、实际编译、新参考和同映像运行验证。
6. 完成 Sum03 的可移植构建和源码提交对应的托管验收，再继续其他全量门禁。

原失败归档的已确认 ACK 是 **27,262,976 B**。下一块长度为1,048,576 B，结果 UNKNOWN，整个归档未 commit。[完整原参数](evidence/failed-sum02-unknown04-exact-request.json)含原 base64 载荷；[状态账](evidence/failed-sum02-transfer-checkpoint.json)仅含不带载荷的参数元数据，两者明确分开。它们的保存不意味着原上传已成功。

本目录没有接入新的默认运行目标。分支推送可能触发仓库现有工作流；这些工作流的既有目标不能自动授予本目录源码候选资格。

## 证据入口

- [真实进程终态与双 EOF](evidence/native24-terminal-and-EOF.json)
- [原报告分块索引](evidence/native24-partial-read-index.json)
- [已读取的原始报告前缀](evidence/native24-report-prefix-2359296.txt)——这是前缀，不是完整 JSON 报告
- [原 native24 完整源码与执行意图](evidence/root-sum-integral-native24-reviewed-intent-v2.json)
- [原76四套件完整源码与执行意图](evidence/root-sum-integral-original76-reviewed-intent-v1.json)
- [上一个双主机完整检查点](evidence/continuation-pr36-merged-integral699-cold-pass-v1.json)
- [源码审查记录](reviews/)
- [原始完整传输对象](transfers/)

**整个 PyTorch 重写没有完成，所有 blocker 没有闭合。** 当前实现、测试结果、待执行项和连接故障在检查点中分别记录；任何后续完成声明都必须由对应的真实完整验收支持。
