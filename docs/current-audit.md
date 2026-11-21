# 当前 HIA 重写审计

审计日期：2026-11-21。当前源码由 [manifest](../tools/current_runtime/source/manifest.json) 固定，完整目标由 [full-acceptance.json](../compatibility/full-acceptance.json) 固定。

## 当前状态

当前候选为 **778 个 HIA 核心函数、15 个编译单元**，增加原生 `signbit` 与多类型／标量 `index_select`，并修复原 F32 无梯度 `index_select` 取错行。完整当前门扩展为 **31 组、14,814 个案例**；此前 29 组及其全部比较、资源与所有权控制继续必需。当前完整运行状态尚待精确提交上的新 CI 和全部原件读回，不继承其他源码的 PASS。

全量目标仍为固定 PyTorch v2.15.0 提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457` 的 **2,590 条原生声明、688 条导数声明、17 项完整系统要求**。独立盘点与账本一致；10 项系统进行中、7 项开放，三类完整闭合数均为 **0**，整体 **HOLD**。当前实现仍限定于 Linux x86-64 CPU、有界 rank0–4 和选定类型／布局。

| 当前778身份 | SHA-256 |
| --- | --- |
| 完整源码 | `db88eaad9fb69010ae4f6925f4e8c54edce815b9f46507bc0c7c2536a4cb2c0b` |
| 主 ELF | `929529e7c86ea3ce12e522f9b01b1a7c4180feb00bde812bc26268091d5032e4` |
| Manifest | `361609f574cdde8854d176f22c810aea9a72ef80adcfd0d8beeb1c67ae4b6f52` |

## 原生修改与实际定向验证

相对 PR55 的完整775源码，仅修改 `entry`、`index_select`、`predicate_bits`、`predicate_command`，追加三个原生索引函数；其余 **771 个原函数正文和定义顺序逐字保留**。原 F32 `index_select` 整个函数保留为 `typed_index_select_previous`，非标量 requires-grad 输入继续使用它；无梯度输入走新的逐元素原始位复制路径。[源码审阅](../validation/current-runtime778/source-review.json)和完整逆变换约束所有前后字节，来源证明不授予数值资格。

原缺陷可由输入 `[[1,2,3],[4,5,6]]`、axis0、indices `[1,0,1]` 复现：旧无梯度路径错误输出三份第一行。新路径按预期产生第二行、第一行、第二行，并覆盖标量、空输出、重复索引、各轴、非连续布局、源偏移、特殊位模式、失败事务和真实分配失败恢复。

| 同一778映像上的实际本地观察 | signbit | index_select |
| --- | ---: | ---: |
| 独立 Torch 数值案例 | 97 | 866 |
| 实际原生进程 | 106 | 869 |
| 完整请求／响应帧 | 2,347 | 17,153 |
| 逐项断言 | 8,843 | 64,271 |
| Owner 快照 | 774 | 5,759 |
| 映射快照 | 12 | 26 |
| 协议控制 | 13 | 78 |
| 真实 OOM／原请求恢复 | 2 | 2 |
| 实际 F32 AD 对照 | — | 5 |

两组均已实际执行并完整物理回放；本机 Torch 2.15.0+cpu 缺少原始安装报告和 direct_url 元数据，因此这些结果明确标为 **探索验证**，不满足完整生产来源准入。严格默认入口已确认拒绝这类报告；统一31组门不传本地降级选项。精确报告摘要与限制见[实际观察记录](../validation/current-runtime778/native-review.json)。全部11项原始 wheel／安装来源检查、真实参考进程及双流 EOF 仍是 CI 必要条件。

`signbit` 使用新 opcode126，保留125的拒绝。完整 F16/BF16 编码枚举和其余类型观察覆盖负零、NaN 符号、空／标量／布局；六个成功 FE 与六个无效后缀回滚检查完整 Owner/RNG/映射。旧五种分类运算485例也在独立 signbit 候选上完整通过；该结果不代替当前778的完整485重跑。

## 完整门与历史失败

统一协调器、工作流与封存器固定31组／14,814例及顺序。所有新增和原有测试读取本次实际冷构建；Generator ABI／重排探针继续由当前对象分别链接执行。缺组、跳过、超时、输入变化、缺 EOF、物理重读失败或未验证的参考来源均阻止 PASS。[当前资格计划](../validation/current-runtime778/qualification-plan.json)是唯一活动计划。

PR52 的完整27组运行因公开 F64 192 来源准备失败，PR53 的完整28组运行因 Bool27／Integral24／F32 Sum17 的旧绑定失败，PR54 的完整29组运行仍有 F32 后置绑定、live32 和 numeric485 准入失败。原 FAIL、原报告和原归档不改写。PR55 修复三处准入，当前778在其后增加完整源码逆变换，全部原数值谓词保留。历史执行细节仍在[固定775审计](https://github.com/TrillionniumFoundation/HeptaTorch/blob/6378ae1b469e9e98a75547fa2670f7cc9e9badf6/docs/current-audit.md)与原证据中。

旧 F64 独立组件的严格 Torch sqrt 比较仍有4／55组失败：历史记录含 AVX512 的1,984差异及 AVX2 的4,008差异。诊断不改写严格 FAIL，亦未证明 CPU 能力差异是原因。该旧组件 sqrt 未接入当前公开 F64 unary 路径；完整 dtype、设备、架构及 AD 仍开放。

## 清理与尚未完成的迁移

默认应用目录只保留一份完整778 HIA源码及15份当前编译单元。被替换的13份775核心分片和旧活动资格计划移至固定 Git 历史，详见[本次退休清单](../validation/current-runtime778/retirement.json)。未使用的独立 signbit 构造辅助文件不进入当前树。源码分片替换不代表删除功能，771个未变函数和完整旧AD函数均受逆变换约束。

旧143源码已从跟踪树移除，但 Native／F64／RNG 历史流程仍从固定提交恢复它。它们的完整语义迁移尚未完成；当前代码仍需保留实际读取的比较器、参考包、bootstrap 与许可文件。默认当前构建不恢复旧源码，CI继续在资格前后检查其缺失；旧 CI 仍必须从缺缓存状态完成原恢复控制。只有消费者迁移并通过同等数值、所有权、错误与资源验证后才能删除最后的旧整套依赖。

完整契约验收器的生产注册表仍为空。已完成的目标盘点、源码绑定和有限运行机制无法代替完整声明／系统实现，不能把闭合数改成非零。

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
