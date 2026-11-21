# 当前 HIA 重写审计

审计日期：2026-11-20。源码身份以 [source manifest](../tools/current_runtime/source/manifest.json) 为准；完整目标以 [full-acceptance.json](../compatibility/full-acceptance.json) 为准。

## 当前状态

已合并的 [PR45](https://github.com/TrillionniumFoundation/HeptaTorch/pull/45) 和 [PR46](https://github.com/TrillionniumFoundation/HeptaTorch/pull/46) 将主分支推进到 **21 组／2,156 项实际验收**的 742 版本。两次原始归档和合并树均已独立核验。后继 [PR47](https://github.com/TrillionniumFoundation/HeptaTorch/pull/47) 的 761 版本仍在完成修正后的统一 23 组门；不能把其分项结果记成合并后的完成状态。

本次候选为 **765 个 HIA 核心函数、15 个编译单元**，在乘法和取负基础上增加 Half 无梯度加法／减法。当前完整入口要求 **24 组／3,845 个有限案例**。已完成的 1,006 项新组、原 Half47 和受影响乘法312均有本映像的实际执行及独立物理回放；统一24组仍待新提交执行。

完整目标为 PyTorch v2.15.0，提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`：**2,590 条原生声明、688 条导数声明、17 项完整系统要求**。**10 项进行中、7 项开放；完整声明和整项系统要求闭合数均为 0，整体 HOLD。** 有限 CPU 案例、HIA 函数数量和提交合并不等于完整 PyTorch 替代。

| 当前候选身份 | SHA-256 |
| --- | --- |
| 完整765源码 | `b4b2996978dce56c9375ce97b2d2638472cbb154a4c0a7c5de822234e31a9c68` |
| 主 ELF | `62f74e9232ae30027a64ca61881bb581320b1787529c6ee9e96818a0f8447afe` |
| Manifest | `100f60ce946ac809949d79faf4eca8e8e16d5f4aeacfd907d228edc1ea0c1fd2` |

## 新增能力及实际验证

命令2的操作1／2支持至少一个 F16/BF16 操作数的普通加法／减法，协议固定 alpha=1。覆盖公开 dtype 有序组合、rank 相关提升、广播、空张量和当前可达布局，并逐位处理固定 AVX2／单线程内核的 NaN 行为。Bool 减法保留实际 Torch 错误；私有 dtype3、Half AD 和其他未实现接口继续按原边界拒绝。运算、类型转换、分配和释放由 HIA 原生程序执行，没有宿主数值回退。

只改变前驱 `binary` 并增加四个 HIA helper；其余760个函数、全部乘法／取负 helper 和原命令参数检查逐字不变。完整765→761→758→742→740→736→700源码逆证明与实际新执行同时保留。细节见[加减法契约](native-half-addsub.md)、[乘法契约](native-half-multiplication.md)和[取负契约](native-half-negation.md)。

| 本765映像实际执行及物理回放 | 新加减1006 | 原乘法312及四个加减后继 | 原 Half47 |
| --- | ---: | ---: | ---: |
| 原生子会话 | 1,006 | 312 | 47 |
| 完整协议帧 | 31,468 | 9,664 | 3,513 |
| 断言 | 119,842 | 37,168 | 17,713 |
| Owner 快照 | 9,166 | 2,868 | 561 个原件、7,921 次完整快照谓词 |
| 精确结果元素 | 2,899,312 | 1,313,606 | 保留原908,216个输入及全部后继比较 |
| 真实分配拒绝／同请求恢复 | 4 / 4 | 保留该组原有控制 | 4 / 4 |
| 结果 | PASS | PASS | PASS_PHYSICAL_HALF47 |

新1006组包含原1004案例的完整不变前缀及两个遗漏的原始 FE 配方，共70项协议控制、12次映射观察。另一个独立实现的加减审查通过 **7,712 例、342,369 个逐位结果**。这些有限执行均保留各自来源与原件身份，摘要见[765范围记录](../validation/current-runtime765/native-review.json)。前驱761的371例取负结果仍保留761身份；新统一24组必须在765上映像重新执行该组。

## 原始47组与239请求完整继承

冻结的 `entry_controls` 完整7,497字节继续执行，原239请求现在划分为 **173个拒绝 +22个乘法成功 +2个取负成功 +40个单命令加减成功 +2个原始同帧 FE 加法成功**。其中4个 Bool 减法错误和8个私有dtype3加减拒绝仍返回原错误。原47子会话、94个双流EOF、输入／RNG、数值／元数据和四个真实OOM恢复全部保留。

两个 FE 请求保持原228字节报文。程序先创建 Half 工厂结果，再创建 F32 转换临时对象和 F32 加法结果，提交时释放转换临时对象。验证器从固定 HIA 源码和真实前态推导这一过程：检查三个对象全部84字、全部八张所有权表、first-free槽位和代数、first-fit偏移、Hold连续复用、最终空闲区、原Storage字节及Generator状态。它不伪造中间快照，也不拆分原报文。

12个分配路径函数、44个构造／清理函数和rng03中的两个普通HIA舍入叶函数均按源码逐字绑定；未找到的被调用函数仍会被拒绝。原乘法与取负适配文件保持原字节，实际拒绝闭包继续复用。独立审查对4对真实前后快照及496项定向损坏／异常控制验证了该限定准入。最终47回放之外，又独立核对了全部4,789个物理输入的模式、长度和SHA，以及561个完整owner压缩原件的展开EOF。

## 发现并修复的验收缺口

原始22组组合运行只有21组通过，live32在原生执行前被旧的14单元／15进程假设拒绝。PR47旧头的[23组运行](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44733625790)同样只有live32失败，22个其他组通过，实际通过2,807例。原ZIP完整保留并已核读。修正提交 `f762a26b6a912fc22f4b7a30a509e861a1c2967e` 将七个精确库存字面值映射到真实15单元／16进程；原文件、模式、子进程、EOF、所有权和来源谓词不变。修正后的32例实际运行及独立回读已通过；[新完整23组运行](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44738335381)仍需自己的完整证据。

加减法原47最初暴露出两个单命令循环外的 FE 成功后继；随后独立破坏试验发现临时对象和已退休Hold未被完整检查，最终改为完整源推导。另一次真实运行在原生执行前暴露出跨rng03单元的调用图遗漏，现已严格加入两个已链接HIA叶函数和完整单位绑定。旧失败报告、原始输入及其字节身份保留；最终PASS来自修正后的真实47运行。原始1004配方的完整逆向证明也保留。

## 当前统一验收要求

当前协调器、CI摘要和原件封存都要求 **24组／3,845例**，组顺序不变并追加half_addsub1006。共享实际312／371／1006参考的原始进程、解释器、输入、输出及EOF均被绑定；原乘法的四个加减后继使用与Half47相同的实际1006参考。原Generator探针仍由当前对象单独链接并执行，不能替换主映像生产案例。

**尚未记录765候选完整24组统一PASS。** 分项成功不相加为整体验收，旧映像结果不转移。每个最终头需要全部实际检查、完整原始归档独立读取及精确树核验后才能合并。[资格计划](../validation/current-runtime765/qualification-plan.json)保留这一要求；全项目完整合同仍为HOLD。

## 已合并运行与原始证据

| 验收 | 固定结果 |
| --- | --- |
| [740 CI 44721607552](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44721607552) | 20 组／1,876 项 PASS；126,110 个归档成员及 108,730 个声明输入全部核对 |
| [742 CI 44723515662](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44723515662) | 21 组／2,156 项 PASS；129,674 个归档成员及 113,586 个声明输入全部核对 |
| 原始 EOF／格式 | 两次完整归档均通过 SHA、模式、完整 XZ 流及 TAR 尾部检查 |
| Native HIA | 两个 PR 各 112 个步骤全部成功，无跳过或失败 |
| 最终合并 | PR45 `fe58075c20e0b533768c23b65d724c7523ee8c01`；PR46 `9d8e90823fa7ca486aa2e843844b3d68d471034f` |

742 原始 ZIP 为 549,050,337 字节，超过连接器下载限制。[独立包装 CI](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44731179672)保留整个原 ZIP 字节不变，用外层压缩得到 460,003,574 字节；解包后首先核对原 SHA `ef3e391ff1a64696f039cfc2340fb36028764cb55a56d2b8b9944b4b2606681c`，随后验证全部原始内容。没有重新生成或改写旧执行结果。后续工作流直接压缩外层 ZIP 中的 JSON 清单和报告，保留全部内部内容。

[最初的 740 失败 CI](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44455767362)仍保留：19 组通过，485 项分类参考因缺少 NumPy 发出警告，在原生执行前被严格 stderr 条件拒绝。SHA 固定的 NumPy 2.3.5 修复了环境，严格比较、11 项来源验证及 stderr 条件均保留；成功运行的该参考 stderr 为 0。

## 清理与最终完成条件

[清理提交 36740c7](https://github.com/TrillionniumFoundation/HeptaTorch/commit/36740c730440ab8ae6a47ea05e4be825845aaddd)已删除 16 个过程目录、269 个文件、104,970,124 字节，原 Git blob、模式和实际消费者均已检查；更早的 53 个旧源码文件也已按固定身份退休。默认生产路径只保留一份当前完整 HIA 源码及相应分区，历史由[固定 Git 索引](history.md)定位。

仍有比较器、冻结参考与 bootstrap 消费旧路径，包括 `src/heptatorch.hepta`。彻底移除旧整套需要先迁移这些真实消费者并完成替代功能。许可证、上游通知及当前真实依赖保留。最后的“全量 HIA 原生整套”条件仍由下面 17 项完整要求决定。

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

下一阶段的依赖顺序见[执行计划](continuation-plan.md)。`make acceptance-status` 当前验证账本一致性；`make full-acceptance` 仍失败，因为各项完整原生合同及其完整验收运行器尚待实现。
