# 当前 HIA 重写审计

审计日期：2026-11-20。当前源码和完整目标分别以 [source manifest](../tools/current_runtime/source/manifest.json) 与 [full-acceptance.json](../compatibility/full-acceptance.json) 为准。

## 当前结论

当前候选是 **742 个 HIA 核心函数、14 个编译单元**，运行时直接生成 Linux x86-64 静态 ELF。它在 740 版本的 Half 逻辑归约和五种数值分类操作之上，修复了 BF16 NaN 转换的布局语义。

固定重写目标仍为 PyTorch v2.15.0，提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`：**2,590 条原生声明、688 条导数声明、17 项完整系统要求**。目前 **10 项进行中、7 项开放，完整声明及完整系统要求的闭合数均为 0**。应用数值工作已存在真实 HIA 实现；全量 PyTorch 兼容仍需大量原生功能和整项验收。

| 当前身份 | SHA-256 |
| --- | --- |
| 完整源码 | `abbc84b8eca16670e72b8872d1d7ab479f8a1d44f7b2b65deea1649f7682b2c7` |
| 主 ELF | `606f69aaecedbedc7ed6ad8f033275e0b64697f70632036d180106d315243c32` |
| Manifest | `a750f6e388c4339c29ff299aca0dabda9fa8b0ff7cc69d5bfa45f69cdd86e009` |

## 本轮已验证的实现

### BF16 的逐位转换

原 F32→BF16 路径保留 NaN 符号和 payload，与固定 Torch 的 copy kernel 不一致。实际参考使用 AVX2/AVX512 向量路径时输出 `0xffff`，标量路径输出 `0x7fc0`。HIA 修复根据**实际分配的目标布局及同轴源步长**决定路径，并分别处理 rank-zero、singleton、广播、带间隔视图和稠密置换。

仅修改原 `half_convert_value`、`half_copy_tensor` 两个函数，增加 `half_bf16_nan_word`、`half_narrow_bf16`；其余 **738 个原核心函数及 RNG/owner 支持保持完整字节一致**。有限值仍调用原 `round_bf16`。新增源码层可以精确还原完整 740 前驱，再组合原 740→736→700 等逆证明。

独立固定 Torch 对照及完整物理读回已通过：

| 验证 | 实际结果 |
| --- | ---: |
| 新 BF16 案例 | 280 / 280 |
| 逐位元素比较 | 1,312,796 |
| 原始协议帧 | 6,088 |
| 断言 | 23,096 |
| 完整 owner 快照 | 1,960 |
| 原生进程 | 280 |
| 原始帧独立复核 | PASS |
| 另一条独立审查的实际原生转换 | 6,848 次逐位及元数据比较通过 |
| 独立布局／路径参考观察 | 112,744 次实际 Torch 转换，未发现反例 |

独立审查覆盖 1,200 个 AVX512 布局和 512 个强制 AVX2 布局，使用两个命令与两种内存格式。强制 AVX2 仍在同一 AVX512 主机执行；DEFAULT、其他架构与设备需要各自实现和验证。详见 [BF16 契约](../tests/bf16_copy/README.md)和[本次数值审查摘要](../validation/current-runtime742/bf16-review.json)。

### 参考环境修复

原 [PR45 CI 44455767362](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44455767362) 的 19 组／1,391 项通过，485 项数值分类组在原生执行前被严格 stderr 检查拒绝。完整原始材料显示，实际 Torch 已产生 485 项参考，但因未安装 NumPy 输出了 307 字节警告；该组原生帧数为 0。

[修复 c41bb43](https://github.com/TrillionniumFoundation/HeptaTorch/commit/c41bb43aaa259f3d20ae4c0fafabdf20581d9af8)随原 Torch wheel 安装 SHA 固定的 NumPy 2.3.5 wheel，保留真实 pip 回执，并增加严格导入检查。原数值比较、11 项来源验证及 stderr 要求保持不变。修复环境下的新 485 项参考已完整产生，stderr 为 0 字节。

## 当前完整组合验收

当前协调器要求 **21 组／2,156 项**：原 20 组／1,876 项及全部原附加控制继续保留，新增 BF16 280 项。生产案例绑定同一主映像和实际冷构建；Generator ABI／重排探针从当前对象另行链接并执行。完整组表见[当前构建说明](current-runtime.md)。

**742 候选的完整 21 组运行正在执行，尚未记录统一 PASS。** 上述 BF16 独立结果与 740 的结果分别保留原身份。当前[资格计划](../validation/current-runtime742/qualification-plan.json)记录要求，完整实际结果须单独读回。

较早的 736／18 组／1,127 项通过记录仍位于[原 main CI 观察](../validation/current-runtime736/main-ci-44363462524.json)。历史运行的完整详细审计见[固定旧审计](https://github.com/TrillionniumFoundation/HeptaTorch/blob/9ea2ec5522685a8b1a0080edda88b89dec094474/docs/current-audit.md)，不与当前候选混写。

## 仓库清理与分支核对

[清理提交 36740c7](https://github.com/TrillionniumFoundation/HeptaTorch/commit/36740c730440ab8ae6a47ea05e4be825845aaddd)删除 **16 个过程目录、269 个文件、104,970,124 字节**。所有删除文件核对了原 Git blob、字节和模式；独立复核确认没有活跃代码、冻结包、CI 声明输入或实际归档消费者读取这些目录。原始结果保留在[固定历史索引](history.md)中。

仍被旧比较器、F64 数值来源或工具链 bootstrap 读取的源文件和 fixture 保留。当前默认生产源码已统一；要最终移除所有旧路径，仍需迁移其实际消费者。许可证和上游通知保留。

[PR41](https://github.com/TrillionniumFoundation/HeptaTorch/pull/41)已按“被取代”关闭：Generator 功能已通过 PR43 进入当前运行时，完整源逆证明保留其原 712 函数与 15 个单元，原九组／644 项仍在必需回归内。它未被伪称为祖先合并，独有历史结果保留于原提交。

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
