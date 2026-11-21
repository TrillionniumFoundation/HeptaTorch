# HeptaTorch

HeptaTorch 使用 **HIA 原生源码**实现张量数值、对象分配、梯度传播、随机数和参数更新，直接生成 Linux x86-64 静态可执行文件。唯一默认应用源码位于 [`tools/current_runtime/source/`](tools/current_runtime/source/)；[manifest](tools/current_runtime/source/manifest.json)给出完整源码、编译分区和映像身份。

**整个 PyTorch 的全量重写尚未完成。** 固定目标为 PyTorch **2.15.0**：**2,590 条原生声明、688 条导数声明、17 项完整系统验收要求**。完整声明和整项系统验收的闭合数均为 0。有限案例的实际结果及剩余实现见[当前审计](docs/current-audit.md)。

## 当前运行时

当前候选包含 **781 个核心函数、15 个独立编译单元**。每次构建使用固定 HIA ABI3 编译器重新生成全部应用对象，再由原生 HIA 链接器生成 ELF。运行时接口是有界原生二进制命令协议。

| 能力 | 当前实现 |
| --- | --- |
| CPU 张量 | F32 算术与一阶反向 AD；选定 Bool/I64 路径；受限无梯度 F64，以及 F64 与 F32/Bool/I64 的混合加减乘除；F16/BF16 存储、复制、填充及 F32 转换；含 Half 操作数的 20 种公开 dtype 有序组合的无梯度乘法，F16/BF16 原生取负，固定 alpha=1 的无梯度加法／减法，以及原生真除法 |
| 对象与生命周期 | 独立 Storage/Version、Saved 张量、可微视图、公开梯度、所有权转移和失败事务回滚 |
| 逻辑与分类 | `any`、`all`、显式 Bool/F32 `sum`；`isnan`、`isinf`、`isfinite`、`isposinf`、`isneginf`、`signbit` |
| 原生索引 | F32／Bool／I64／F64／F16／BF16 `index_select`，标量、非连续步长和独立输出；修复 F32 无梯度取错行，保留原 F32 一阶 AD |
| 训练 | 选定 NN 算子、反向传播、SGD/momentum、Adam/AdamW、CPU Generator 和部分随机分布 |
| Half 乘法 | rank 相关类型提升、广播、可达布局、原始右侧标量精度，以及固定 AVX2／单线程 NaN 位模式；结果具有独立所有权 |
| Half 真除法 | 保留原始右侧标量的 F32 运算精度，按固定 Torch 提升、广播和布局规则执行；1,196 项真实对照逐位检查 NaN、尾段和独立所有权 |
| Half 加减法 | rank 相关类型提升、广播、可达布局及固定 AVX2／单线程位模式；保留 Bool 减法的实际错误和两条原始同帧 FE 加法 |
| 混合 F64 算术 | F64 与 F32/Bool/I64 双向组合，rank 相关提升、广播与可达布局；I64 直接按最近偶数转换到 F64，2,541 项实际对照及原 23 个请求后继已通过专项验证 |
| 同类型 F64 算术 | F64/F64 加减乘除使用原生内核；单元素广播按实际维度排序选择 NaN 载荷。5,088 例保留原 3,288 例并追加 1,800 例维数、步长和特殊值组合，必须在当前映像上完整验证 |
| I64／F64／F16／BF16 ReLU | 命令 3／操作 5 原生执行无梯度 ReLU；保留负零，按固定 AVX2 路径匹配 NaN 位模式。776 例与原两套取负共 1,552 例已完成同构建、共用参考的专项验证 |
| I64／F64 取负 | I64 环绕取负及 F64 符号位翻转；保留 NaN payload、输入字节、布局及独立所有权，真实 OOM／恢复已验证 |
| Half 取负 | 保留 dtype、形状及输出布局；精确处理完整 32 元素向量组与标量尾段的不同 NaN 位模式 |
| BF16 转换 | 根据源和目标布局选择原生 NaN 转换路径，逐位匹配固定 Torch AVX2/AVX512 的标量／向量结果；有限值沿用原舍入实现 |
| 平台与规模 | Linux x86-64 CPU，rank 0–4；各命令独立检查形状、存储、工作量及资源限制 |

完整 Python/C++ 前端、通用 dispatcher、完整 dtype/layout/promotion、前向及高阶 AD、序列化、加速器、分布式、compile/export、数据管线及跨平台支持仍需完成。精确边界见[能力清单](compatibility/capabilities.json)及对应测试说明。

## 构建与测试

按[当前构建说明](docs/current-runtime.md)准备固定工具链和独立参考环境，然后运行：

```sh
make HEPTA_SOURCE_ROOT=/absolute/path/to/fixed-hepta-source \
     HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools \
     HEPTA_ENTROPY_BUILD=/absolute/path/to/current-entropy-tools

make test HEPTA_SOURCE_ROOT=/absolute/path/to/fixed-hepta-source \
          HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools \
          HEPTA_ENTROPY_BUILD=/absolute/path/to/current-entropy-tools \
          HEPTA_TORCH_PYTHON=/absolute/path/to/pinned-env/bin/python
```

产物为 `build/current-runtime/heptatorch`。`build.json` 绑定完整源码、全部新对象、工具、ELF、栈检查以及真实编译／链接退出和双流 EOF。

参考环境固定为 CPython 3.12、Linux x86-64、官方 CPU Torch 2.15.0 wheel，以及 [SHA 固定的 NumPy 依赖](compatibility/pytorch-reference-requirements.txt)。Python 组织构建和独立验收，PyTorch/NumPy 用于参考；原生应用无需这些宿主库，也无需 libm/libc。编译器首次 bootstrap 的宿主依赖见[工具链构建](docs/build.md)。

## 一个当前验收入口

`make test` 和 [Current HIA Runtime CI](.github/workflows/current-runtime.yml)共用 [`tests/qualify_current_runtime.py`](tests/qualify_current_runtime.py)。当前必需范围为 **32 组、15,145 个有限案例**：保留全部前 29 组／13,851 项及原控制或明确验证的成功后继，新增 `signbit` 97 项、`index_select` 866 项及 F32 gather/scatter_add/index_add 331 项实际对照。原 F64 入口 87 个请求完整执行：63 个保留拒绝，1 个取负、22 个混合算术单命令和 1 个原始同帧 FE 事务要求实际 Torch 数值及完整所有权验证。原 Half 入口的 239 个请求完整执行：22 个乘法、2 个取负、40 个单命令加减法、2 个原始同帧 FE 加法及 22 个真除法要求实际数值和所有权验证，其余 151 个保留原拒绝行为。Half47、乘法312、加减1006和除法1196共享同一份实际1196参考及其原始进程、输入和输出。F6448与混合算术2541共享同一份实际2541参考；两条Bool减法错误仍保留。新同类型 F64 组保留两次真实数值失败记录，并要求全部 88 项协议控制和四次真实 OOM／同请求恢复。ReLU776、Half取负371与宽类型取负405必须共享同一份新 ReLU 参考及实际进程收据；原取负中的四条 ReLU 请求保持原字节，改用完整数值、布局与所有权验收。组表、额外控制和独立执行命令见[当前构建说明](docs/current-runtime.md)。

生产案例绑定本次冷构建的同一主映像；Generator 的 ABI／重排探针由当前对象另行链接并执行。每次源码或映像变化都需要新的完整运行。781 候选的完整组合状态以[当前审计](docs/current-audit.md)及源绑定执行记录为准。

`make acceptance-status` 检查完整目标账本及已注册的完整契约证据。`make full-acceptance` 只有在全部声明和 17 项系统契约通过独立验收后才能返回成功。目前生产注册表为空，完整闭合数仍为 0；[完整契约验收机制](docs/full-contract-qualification.md)要求当前 HIA 源码、映像、真实构建与执行、独立参考和物理读回全部绑定，账本文字或有限案例的 PASS 不能增加闭合数。

## 仓库组织

默认生产目录保留一份完整当前源码及一组有序分区。已退休的过程源码和约 177 MB 的额外历史验证文件已移出当前目录；[历史索引](docs/history.md)指向固定 Git 提交。仍被原比较器、参考包或 bootstrap 读取的内容保留，后续按实际依赖迁移。

| 文档 | 内容 |
| --- | --- |
| [当前构建](docs/current-runtime.md) | 唯一源码、固定工具、参考环境及完整当前验收 |
| [当前审计](docs/current-audit.md) | 实际结果、修复及 17 项完整系统要求 |
| [继续执行](docs/continuation-plan.md) | 后续实现依赖及完成条件 |
| [上游审计](docs/upstream-audit.md) | 固定上游目标及完整声明分母 |
| [Half 乘法](docs/native-half-multiplication.md) | 公开 dtype 提升、固定内核语义、312 项验证及边界 |
| [Half 真除法](docs/native-half-division.md) | 原生运算精度、1,196 项实际对照、原请求继承及物理回放 |
| [Half 加减法](docs/native-half-addsub.md) | 固定内核语义、1,006 项验证、同帧所有权及原 239 请求继承 |
| [混合 F64 算术](docs/native-public-f64-arithmetic.md) | 直接 I64 转换、2,541 项实际对照、原 87 请求继承和同帧所有权 |
| [同类型 F64 算术](docs/native-same-f64-arithmetic.md) | 维度排序与 NaN 载荷修复、原失败保留、5,088 项实际对照及独立物理回放 |
| [原生 signbit 与 index_select](docs/native-index-signbit.md) | 准确范围、F32 缺陷修复、963 项对照及严格 CI 要求 |
| [原生 ReLU](docs/native-typed-relu.md) | 四种 dtype 的位模式、776 例、四条原请求的数值后继及共用参考验证 |
| [I64／F64 取负](docs/native-wide-negation.md) | 405 项实际对照、原 F64 请求继承及完整物理回放 |
| [Half 取负](docs/native-half-negation.md) | NaN、布局、371 项验证及真实 OOM／恢复控制 |
| [BF16 转换](tests/bf16_copy/README.md) | 布局语义、逐位对照及适用 CPU 路径 |
| [保留依赖](docs/runtime-dependencies.md) | 当前构建、32组及旧CI仍读取的文件与迁移要求 |
| [完整契约验收](docs/full-contract-qualification.md) | 固定完整目标、独立验证器、证据约束和真实闭合条件 |
| [历史记录](docs/history.md) | 已退休过程文件的固定历史位置 |

许可证与上游通知见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [`third_party/pytorch/`](third_party/pytorch/)。

RNG 原 74／76／10／25 项消费者已迁到[当前实际 RNG 对象](tools/current_rng_consumer/README.md)，不再恢复旧143应用源码；这条迁移仍需本次严格CI。完整PyTorch的2,590条声明、688条导数和17项系统合同仍未闭合，详见当前审计。
