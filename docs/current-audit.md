# 当前 HIA 重写审计

审计日期：2026-11-20。源码身份以 [source manifest](../tools/current_runtime/source/manifest.json) 为准；完整目标以 [full-acceptance.json](../compatibility/full-acceptance.json) 为准。

## 当前状态

已合并的 [PR45](https://github.com/TrillionniumFoundation/HeptaTorch/pull/45) 和 [PR46](https://github.com/TrillionniumFoundation/HeptaTorch/pull/46) 将主分支推进到完成 **21 组／2,156 项实际验收**的 742 版本。两个 PR 的全部 10 个已观察 CI 检查通过，完整原始归档读回通过，实际合并树与被测试源码树一致。

当前新增候选为 **761 个 HIA 核心函数、15 个编译单元**，包括 13 个核心分区及 RNG、owner 支持。它在上述主分支基础上增加 F16/BF16 无梯度 Tensor 乘法和取负，直接生成 Linux x86-64 静态 ELF。

完整目标仍为 PyTorch v2.15.0，提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`：**2,590 条原生声明、688 条导数声明、17 项完整系统要求**。**10 项进行中、7 项开放；完整声明和整项系统要求的闭合数仍均为 0**。有限 CPU 案例、源码函数数目和 PR 合并不代表整套 PyTorch 已完成。

| 候选身份 | SHA-256 |
| --- | --- |
| 完整源码 | `ab3ed0e0898d2006fbdf145357c79cafd599c2be344fcb3cb65fd6ad55efb231` |
| 主 ELF | `756de9418c05f4ecfc12319c7262cf1ff22e335e4ec32b140721b6c77ee226d7` |
| Manifest | `ef76a93ddee90e2fc546b2a6f95a5133b7663bd535e1f7693bae3b519345060c` |

## 新增原生能力

### Half 乘法

命令 `2`、操作 `3` 支持至少一个 F16/BF16 操作数的全部 **20 种公开 dtype 有序组合**，覆盖 Bool/I64/F32/F64/F16/BF16、rank 0–4、广播、空张量及当前协议可构造的非连续布局。实现处理维数相关类型提升、转换后的布局、右侧 reduced-dtype 标量的原始精度，以及 FBGEMM 转换尾段和实际乘法内核的 NaN 位模式。

758 前驱的 312 项主验证通过：1,313,602 个结果元素、9,600 帧、36,876 次断言、2,848 个 owner 快照及全部物理回放。该映像另有 216 项成对 NaN 和 540 项短向量尾段的独立比较通过。**这些执行仍保留 758 身份；761 的完整验收将重新执行全部 312 项。** 761 的 16 个乘法 helper 与对应测试源码字节不变。详见[乘法契约](native-half-multiplication.md)及[原有限审查摘要](../validation/current-runtime758/half-multiplication-review.json)。

### Half 取负

命令 `3`、操作 `6` 支持 F16/BF16 取负，保留 dtype、形状和适当布局。新结果具有独立 Tensor/Storage/Version；输入与 RNG 保持不变，结果不保留输入或 AD 边。

固定 Torch 内核具有可观察的位模式区别：完整 32 元素向量组只翻转符号位，保留 NaN payload 和 signaling 位；标量 F16 尾段同时将 signaling NaN 安静化，标量 BF16 NaN 输出 `0x7fc0`。参考分别在自己的进程中明确固定 AVX2 与原单线程设置。其他构建、线程配置、架构和设备需要各自验证。

最终 761 映像的实际取负验证：

| 验证 | 实际结果 |
| --- | ---: |
| 主案例／原生进程 | 371 / 371 通过 |
| 精确结果元素比较 | 272,444 |
| 原始协议帧 | 11,377 |
| 断言 | 37,744 |
| owner 快照 | 2,717 |
| 协议附加控制 | 46 |
| 真实 OOM／相同请求恢复 | 2 / 2 |
| 映射观察 | 4 |
| 全部原始材料的物理回放 | PASS |

覆盖每种 dtype 的全部 65,536 个字、全部带符号 NaN payload 的标量批次、长度 0–70、全部 24 种 rank-four 稠密轴置换，以及可达带间隔／广播／空／singleton 布局。原构造诊断发现 raw-set12 的版本增量与最初参考输入写法不一致；改用实际 Torch 整数 dtype 视图逐项赋值后，输入位模式与共享版本均一致。原失败参考和原始帧保留，原生取负实现没有因此改动。

仅改变 758 的 `unary`、`command_argument` 两个函数，追加三个 HIA helper；其余 756 个函数、全部乘法 helper 和对应测试保持字节一致。完整 761→758→742→740→736→700 源逆证明与真实新映像执行共同保留。详见[取负契约](native-half-negation.md)及[本次有限审查](../validation/current-runtime761/native-review.json)。

## 原 Half47 的完整继承验证

最终 761 映像已重新通过全部 47 个原生子进程和独立物理回放：**2,797 帧、13,105 次断言**，包括原 26 个数值组、16 个边界、一个入口组及四个真实 OOM／恢复案例。

冻结 `entry_controls` 的完整 7,497 字节及 **239 个原始请求**继续执行：**215 个原拒绝 + 22 个乘法成功 + 2 个取负成功**。新增成功路径核对原标签、字面输入、完整报文、真实 Torch 输出、全部输入描述字、独立所有权及释放。私有 dtype3、Half AD、其余算术和格式错误等控制继续保留。

既有 22 条乘法适配的完整文件字节不变。两条取负适配重用其实际拒绝闭包和 owner hook，显式更新终端分区；取负自身使用原 transport 与 owner 方法。所有原快照字段、完整帧回滚和原数值比较继续检查。

## 当前完整验收要求

当前协调器要求 **23 组／2,839 项**：原 21 组／2,156 项输入及全部附加控制或上述明确成功后继，加乘法 312 项和取负 371 项。每个新参考运行一次，并由 Half47 与对应主组共同绑定其真实进程、解释器、输入、输出流及数据块。

**761 候选尚未记录完整 23 组统一 PASS。** 371 项和原 Half47 的独立结果分别保留范围；前驱运行不转记为新映像完成。[资格计划](../validation/current-runtime761/qualification-plan.json)记录要求，最终结果必须来自同一本次冷构建主映像的完整执行、独立 Generator 探针及全部原始证据。

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
