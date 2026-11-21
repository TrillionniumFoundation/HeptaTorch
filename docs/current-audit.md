# 当前 HIA 重写审计

审计日期：2026-11-21。当前源码由 [manifest](../tools/current_runtime/source/manifest.json) 固定，完整目标由 [full-acceptance.json](../compatibility/full-acceptance.json) 固定。

## 当前状态

当前候选包含 **781 个 HIA 核心函数、15 个编译单元**。新增原生 `signbit`、多类型／标量 `index_select`，修复 F32 无梯度 `index_select` 取错行，以及无梯度 `gather`、`scatter_add`、`index_add` 的前向几何问题。完整当前门为 **32 组、15,145 个案例**；原29组13,851例、全部数值、错误、资源、所有权与进程检查继续必需。新源码的完整 CI 和原件独立验收仍待执行，不继承任何旧候选的 PASS。

全量目标固定为 PyTorch v2.15.0 提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457` 的 **2,590 条原生声明、688 条导数声明、17 项完整系统合同**。10项系统进行中、7项开放；三类完整闭合数均为 **0**，整体 **HOLD**。当前是 Linux x86-64 CPU 上有界 rank0–4、选定类型／布局的原生实现，尚未完整替代 PyTorch。

| 当前781身份 | SHA-256 |
| --- | --- |
| 完整 HIA 源码 | `ee214f05b56401ea17a5f55e792aaccf73447974ba0b7e0416990439bff6a8ff` |
| 主 ELF | `71309d9e8ef794b9052603fd28f4d4da4886aff6924d8413489af856744ec272` |
| Manifest | `a53b2bf1865a56adfc9f56763feae1b7c5ac9c23243399efe1c201c75db3a8c9` |

## 原生修复与实际验证

相对 PR55 的完整775源码，修改 `entry`、`index_select`、`gather`、`indexed_add`、`predicate_bits`、`predicate_command` 六个函数，追加六个助手函数；其余 **769 个原函数正文和定义顺序逐字保留**。完整原 F32 `index_select` 保留为 `typed_index_select_previous`。新的 gather/indexed-add 助手在输出需要梯度时仍进入完整原前向助手，所有 backward 正文不变。完整781→778→775逆变换保留两个中间身份，并要求整体正向重构逐字相等，见[源码审阅](../validation/current-runtime781/source-review.json)。两次独立冷构建生成相同 ELF；冷构建本身不授予运行资格。

原无梯度 `index_select` 对 `[[1,2,3],[4,5,6]]`、axis0、indices `[1,0,1]` 错误地返回三份第一行。新的原始位复制路径返回第二行、第一行、第二行。无梯度 `gather` 对 axis1、indices `[[2,0],[1,2]]` 错误返回 `[[3,1],[2,3]]`，现在返回 `[[3,1],[5,6]]`。两个 indexed-add 操作在两个输入都无梯度时错误拒绝有效输入，现在使用显式输入几何计算；旧有 AD 路径保持原字节。具体实现和范围见 [signbit/index_select](native-index-signbit.md) 与 [F32 索引前向](native-indexed-forward.md)。

| 相同781源码／ELF上的本机观察，各自保留完整构建收据 | signbit | index_select | indexed F32 |
| --- | ---: | ---: | ---: |
| 独立实际 Torch 案例 | 97 | 866 | 331 |
| 实际原生进程 | 106 | 869 | 334 |
| 完整请求／响应帧 | 2,347 | 17,153 | 9,938 |
| 逐项断言 | 8,843 | 64,271 | 37,094 |
| 完整 Owner 快照 | 774 | 5,759 | 2,918 |
| 映射快照 | 12 | 26 | 30 |
| 协议控制 | 13 | 78 | 56 |
| 真实 OOM／相同请求恢复 | 2 | 2 | 3 |
| 实际 F32 AD 对照 | — | 5 | 193 |

三组均已实际运行，并在禁止新进程和 Torch 导入的纯读取器中重放原始物理证据。本机 Torch2.15.0+cpu 缺少原始安装报告和 direct_url 元数据，结果明确为 **探索验证**，不满足生产来源准入。默认严格入口拒绝这些报告；统一32组门不传本地降级选项。完整11项 wheel／安装来源检查、真实参考进程、精确 argv／解释器、全部输入、Owner 与双流 EOF 仍是 CI 必要条件。[实际观察记录](../validation/current-runtime781/native-review.json)固定每份原报告身份。

原公开 F64 logical192 也已在781上实际重跑：192例、196个原生进程、3,711帧、8,489断言、18个协议控制、2次 OOM恢复，并由原读取器完整复核。它使用原完整参考和原参考进程收据；2个原生边界观察与6个范围外 Torch 观察不计入兼容性，历史来源限制不改写。

## 完整门、准入预检与历史证据

[唯一活动资格计划](../validation/current-runtime781/qualification-plan.json)、协调器、工作流与封存器固定32组／15,145例及顺序。全部组使用本次完整冷构建；Generator ABI／重排探针由本次实际对象分别链接并执行。缺组、跳过、超时、来源或输入变化、缺 EOF、物理复核失败均阻止通过。

新增早期源码准入检查完整源逆链、观察器逆变换、实际加载的公开 F64 读取器、16个模块的33项身份、224项文件绑定、56个完整 HIA 函数及实际协调器／工作流的分母。另独立执行两类 Owner 的原完整 guards／负例，以及4套 AsStrided、3套资源读取器的准备流程。12项独立反例检查覆盖有效但过期的字节码、实际用例统计、失效的工作流拒绝条件、文件权限和实际读取器上下文。封存器要求这5份原准备报告及其输入；源码准入、Owner控制、准备流程和反例检查分别保留自己的状态，均不授予原生运行资格，也不替代随后任何数值案例。

PR55 的原29组／13,851例 Current门已通过，其单一原始包有277,502个成员；逐字节、模式、349,065项输入绑定、完整冷编译、33段原stdout、源码链与实际Git树已独立核对。该包的587,749对物理帧中 opcode126 未出现，opcode125 的50次原拒绝均保留。PR55的6个工作流、16个检查均成功。它是有自身固定身份的前驱证据，不授予781资格；Native整日志的工具检索缺口也不能用状态摘要替代。

PR52、PR53、PR54 的完整 Current 失败，以及778候选后来发现的旧身份／函数数／字节绑定遗漏均保留原记录。旧 F64 独立组件严格 Torch sqrt 比较仍有4／55组失败；AVX512的1,984处差异和AVX2的4,008处差异保留，诊断不改写严格 FAIL，未证明 CPU能力差异就是原因。该旧 sqrt 未接入当前公开 F64 unary，完整类型、设备、架构与 AD 仍开放。

## 清理与剩余旧消费者

默认应用只保留完整781源码及15份当前单元；13份778核心分片和4份被替代候选元数据按固定Git身份退休，见[本次清单](../validation/current-runtime781/retirement.json)。775及更早分区的原字节也保存在固定Git历史。当前所需的源码逆变换、数值参考、原错误记录和许可证继续保留。

RNG独立消费者已实现迁移：只编译原数据包入口的5个函数，并链接完整当前冷构建的实际 `rng03.o`。原74／76／10／25共185项测试、本机247个请求包、25次完整借用缓冲区观察已通过；原17个非入口数值函数和当前单元逐字相同。此流程不恢复旧143源码、不执行旧应用 ELF，见[当前RNG消费者](../tools/current_rng_consumer/README.md)。修复后的纯读取器从原进程意图、父子双收据、原包、完整借用缓冲区和实际安装文件重建判定；14项同步变造反例均被拒绝，原185项证据搬移后仍通过完整复核。本机原始安装实体重算11项来源条件时只有4项为真、7项为假，严格入口据此拒绝生产资格；最终精确树的严格 CI 与其原件验收仍必需。[观察记录](../validation/current-runtime781/native-review.json)保留独立消费者的范围和原报告身份。

旧143源码已从跟踪树移除，但 Native、独立 F64 与显式历史 Make 目标仍从固定提交恢复它。当前默认构建不恢复旧源码，CI继续在运行前后检查其缺失。剩余消费者完成等价的数值、错误、所有权、资源和原件验收以前，不能删除它们需要的原输入。完整旧整套语义迁移尚未完成；活消费者关系见[依赖清单](runtime-dependencies.md)。

完整合同验收器的生产注册表仍为空。源码绑定、有限案例和清理工作不能把任一完整声明或系统闭合数改成非零。

## 17 项完整系统要求

| 系统 | 状态 | 主要剩余实现 |
| --- | --- | --- |
| 模块 ABI | 进行中 | 完整公共 Tensor ABI、签名／效果、外部资源来源及资源图 |
| Storage 生命周期 | 进行中 | 外部／设备分配、custom deleter、并发强弱引用、持久恢复 |
| dtype/layout | 进行中 | 全类型提升／转换、任意 rank、complex/sparse/nested/quantized、设备语义 |
| 原生声明 | 进行中 | 2,590 条完整合同、生成变体及 backend/dtype/layout 组合，关联 688 条导数 |
| 视图与 mutation | 进行中 | 全索引／overlap／stride／out/inplace 与别名、Version、错误行为 |
| AD 与 transforms | 进行中 | 前向／高阶 AD、create_graph、hooks、custom function、checkpoint、vmap |
| 前端与 dispatcher | 开放 | Python/C++ 公共对象／异常、schema/meta、override、context 和扩展注册 |
| NN | 进行中 | Module/Parameter/buffer 树、train/eval、attention/recurrent 及完整 functional |
| 优化器 | 进行中 | 全优化器／scheduler、参数组、state_dict、可微／capturable／设备模式 |
| RNG 与确定性 | 进行中 | 全分布、设备 Generator、并发状态及确定性控制 |
| 序列化 | 开放 | checkpoint 格式、alias、map_location、兼容、中断及分布式恢复 |
| 加速器 | 开放 | CUDA/ROCm/MPS/XPU 分配、传输、stream/event、kernel、同步和错误 |
| 分布式 | 开放 | rendezvous、process group、collective、异步失败、DDP/FSDP/DTensor |
| compiler/export | 开放 | 图捕获、符号形状、分解、AOT/compile/export、fusion 与产物兼容 |
| 数据管线 | 开放 | Dataset/DataLoader、worker、多进程、pinning、采样、重放与取消 |
| 平台与打包 | 进行中 | 其他平台／架构、安装、扩展及 profiling 接口 |
| 最终集成 | 开放 | 全上游／应用／性能／资源／错误验收及最终替代路径清理 |

下一阶段的依赖顺序见[执行计划](continuation-plan.md)。`make acceptance-status` 验证账本及源码注册的完整契约证据；`make full-acceptance` 当前返回 2／HOLD，因为各项完整原生合同及其完整验收运行器尚待实现。
