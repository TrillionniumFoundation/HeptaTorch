# 当前 HIA 重写审计

审计日期：2026-11-21。当前源码与映像由 [manifest](../tools/current_runtime/source/manifest.json) 固定，完整目标由 [full-acceptance.json](../compatibility/full-acceptance.json) 固定。

## 当前状态

当前候选有 **797 个 HIA 核心函数、15 个编译单元**。统一入口要求 **35 组、16,341 个有限案例**；此前32组15,145例及全部原比较继续必需。新的源绑定参考、统一完整 CI 和原件独立验收必须完成，才能把本候选的有限组合状态标为通过。

全量目标仍是 PyTorch v2.15.0 提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457` 的 **2,590 条原生声明、688 条导数声明、17 项完整系统合同**。10项系统进行中、7项开放，三类完整闭合数均为 **0**，整体 **HOLD**。当前实现面向 Linux x86-64 CPU、rank0–4及选定类型／布局；完整前端、设备与高阶 AD 等要求继续开放。

| 当前797身份 | SHA-256 |
| --- | --- |
| 完整 HIA 源码 | `0061acfaa8e886885053e8f539cf08187be1c1e10265c19e52df07435a71594b` |
| 主 ELF | `955ffc996b8870ef67752f45647f5c1ca9a093f41ff402487091f03223372293` |
| Manifest | `2e442fea4961a5111004a1a721fb46e817f5568f50c7068047fc767721df8e21` |

## 本轮实际发现与原生修复

| 原生路径 | 原行为 | 当前 HIA 修复 |
| --- | --- | --- |
| 无梯度 `matmul`／`linear` | 有效矩阵输入导致 SIGILL | 直接传入活输入的能力和几何；保留原 AD 数值助手 |
| 无梯度 `softmax`／`log_softmax` | 有效输入被拒绝 | 直接传入操作种类、axis及输入，无须临时写入 AD 边 |
| 无梯度卷积／池化／按轴归约 | 错误拒绝或错误结果；继而影响 layernorm、cross_entropy | 新前向助手读取实际输入和输出几何，输出需要梯度时仍走原助手 |
| 无梯度 `cat`／`stack` | cat 覆盖前半输出，stack 导致 SIGILL | 原生布局与元素复制使用完整输入列表、axis和实际输出 |
| `layer_norm`／`batch_norm` | 24个转置输入案例的输出步长错误 | 用原生连续复制及原 AD17 保证结果布局；batchnorm保留实际要求的channels-last |
| 非连续 `linear` | 3个有效案例的输出 Version为0，实际Torch为1 | 按固定参考的融合／非融合条件设置新输出版本；保留之后的实际变更检查 |

线性层版本规则另外由固定上游 [Linear.cpp](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/native/Linear.cpp) 和324次实际Torch观察核对。矩阵乘法的vector×matrix版本差异也保留独立观察。它们是实现依据，不能代替当前完整门。

相对固定781源码，本轮修改 `linear`、`reduce_axis`、`softmax`、`layernorm`、`batchnorm_eval`、`conv2d`、`pool2d`、`matmul`、`tensor_join` 九个原函数，追加16个函数；其余 **772 个原函数逐字保留**。相对775的完整合成逆链则为15个修改、22个新增、760个原函数不变。全部原反向函数、record和原AD数值助手保持原字节。两次新冷构建重新编译15个单元、分别执行16个编译／链接进程，得到相同ELF；源码逆链和编译结果各保留自己的资格范围，见[源码审阅](../validation/current-runtime797/source-review.json)。

## 当前映像上的有限观察

| 必需组 | 实际参考案例 | 原生会话 | 帧 | 断言 | 完整Owner | 映射 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `forward_nograd` | 462 | 465 | 12,108 | 48,630 | 3,609 | 30 |
| `join_forward` | 414 | 416 | 14,256 | 53,638 | 4,050 | 24 |
| `spatial_forward` | 320 | 323 | 10,513 | 57,082 | 3,201 | 24 |

这三组保留691个实际反向对照、join中的194个未定义梯度检查、123项协议控制、8次成功及31次失败同帧事务、8次实际分配失败与相同请求恢复。新797映像也重新执行signbit97、typed_index_select866和indexed_f32 331；各自原数值和控制不变。额外4项channels-last观察确认layernorm连续输出和batchnorm保留channels-last，包括梯度；它们不计入统一分母。

本机参考缺少原始pip安装报告和direct_url元数据，因此这些结果都是**探索验证**。新的默认严格入口必须拒绝它们，不得通过改报告标记或移用旧源码的PASS取得生产资格。[原生观察记录](../validation/current-runtime797/native-review.json)绑定真实报告、当前源码／ELF和来源限制。完整35组仍需在同一冷构建和精确Git树上执行。

## 参考来源与独立验收

独立复核实际复现了旧来源检查的缺口：同时改 `_C` 或 `torch/__init__.py`、安装RECORD及收据摘要，原11项缓存检查可以继续为真。新来源核验保留原11个表达式，并增加固定官方wheel的完整原字节和独立成员锚。官方wheel固定SHA为 `0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3`，209,530,490字节、13,337成员；完整CRC、原RECORD和文件身份已独立读取。

每个新参考在实际数值执行前后核对12,567份不可变安装文件，共751,277,549字节，包含12个共享库；这些原始安装字节和wheel只保存一份共享快照。纯读取器重读完整快照、安装／PEP610／pip实体、实际进程意图、代码映射与双流EOF。原安装RECORD的自报摘要不能替代官方成员锚。映射中的外来或已删除库、无执行权限的代码段和非法地址范围必须拒绝。所有原数值容差、错误、资源和Owner比较保持不变。

[唯一活动计划](../validation/current-runtime797/qualification-plan.json)、协调器、工作流和封存器的组表与总数必须一致。20项已执行源码准入包括完整源逆链、原观察器逆变换、空间组及实际公开F64读取器；224项源合同文件绑定、56个完整HIA函数、16模块33身份全部核对。原12项拒绝条件检查、Owner guards、4套AsStrided及3套资源准备也已通过；它们不计入数值案例，也不授予运行资格。官方wheel准备先于需要绑定它的源码准入，完整原安装和准入步骤的检查条件均保留。

最终来源读取器的14个完整输入篡改反例、16个映射解析反例及1个完整读取器下合法排序的外来已删除库反例全部被拒绝。实际完整wheel／安装快照及搬移读取通过，原683严格参考的15份实体／11个原条件也保持可读。来源反例只验证读取器拒绝具体篡改，不是新的数值案例或完整CI资格。

## 前驱证据及失败保留

PR55原29组／13,851例的完整Current包、277,502个成员、349,065项输入绑定和587,749对物理帧已独立核对；它不能授予797资格。683前驱的Native门已成功，14份原ZIP共27,049成员、7,137个压缩流及23份历史143恢复收据已独立读取。Native整作业日志仍有工具检索缺口，不能用状态摘要补充声称读过的内容。

PR52、PR53、PR54及778／781准备失败保持原始状态。683 RNG的185个实际案例和原11项来源条件通过，但最终读取器发现Torch导入后的cache环境变化而失败；新实现事先明确该输出目录下的cache路径，继续要求完整环境相等。旧独立F64组件仍保留严格sqrt的4／55失败；早先的1,984处差异记录及683原诊断中标为AVX512的4,008处差异不改写为PASS。该旧sqrt没有接入当前公开F64 unary。

## 清理和仍需迁移的消费者

当前生产目录保留797完整源码与15个活动单元。13份781分区、4份被替代元数据及1份无活动读取的重复RNG交接摘要，共 **18份、2,654,116字节**，按固定提交 `683adf048bf083d4f1116a5ef711d18a36f39e24` 退休；[逐项清单](../validation/current-runtime797/retirement.json)保留路径、Git blob、SHA和模式。源码逆变换、独立参考、原失败及许可证继续保留。

RNG packet消费者使用本次完整冷构建的 `rng03.o`，只另编译5个原入口／字节／错误处理函数，保留原74／76／10／25共185项，不恢复旧143应用。其最终严格新树CI与原件验收仍是迁移完成条件。Native、独立F64和显式历史Make目标仍从固定历史恢复143源码；旧实现的完整语义迁移尚未完成，见[实际依赖](runtime-dependencies.md)。

生产完整合同验证器注册表仍为空。删除过程文件、局部案例通过和来源补强都不增加完整声明或系统闭合数。

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
