# HeptaTorch

HeptaTorch 的当前应用运行时由 **HIA 原生源码**实现，直接生成 Linux x86-64 静态可执行文件。张量数值、对象分配、梯度传播、随机数和参数更新均在该 HIA 程序中执行。当前完整源码为 700 个核心函数、14 个独立编译单元；精确源码、ELF 身份及分区以[当前 manifest](tools/current_runtime/source/manifest.json)为准。

**整个 PyTorch 的全量重写尚未完成，项目状态为 HOLD。** 固定目标是 PyTorch **2.15.0**，分母保留 **2,590 条原生算子声明、688 条导数声明、17 项系统验收门禁**。目前 10 项进行中、7 项开放，完整上游 schema 和整项门禁的闭合数均为 0。[当前审计](docs/current-audit.md)说明已有实现、实际证据和剩余范围。

**已发布前版已在 GitHub 实际通过统一 11 组、683 个案例及全部附加控制。** [PR40](https://github.com/TrillionniumFoundation/HeptaTorch/pull/40) 已合并，实际 CI 文件树与合并树一致。继续审计又发现大列表 cat 的分配失败处理、空轴反向性能和空梯度 Version 差异；本次新映像已通过 4 项原始 cat/stack 回归及 10 项独立 Torch 元数据对照。默认门禁扩为 **13 组、697 例**，其在新映像上的完整组合仍待执行，旧 683 的成功不转移给新源码。原 660、旧 cat 崩溃／超时与版本差异报告保留原结果。见[当前审计](docs/current-audit.md)。

## 当前实现

当前接口是受限的原生二进制命令协议，包含 CPU F32 算术和一阶反向 AD、Bool/I64 的选定张量路径、Tensor/Storage/Version 所有权、Saved 和可微视图、原生 CPU Generator、选定 NN 算子以及 SGD/Adam/AdamW。完整 Python 前端、dispatcher 和所有上游重载仍需实现。

本次融合加入以下功能与修正：

- **`any` / `all`**：命令 118 / 119 接受 Bool、I64、F32 和已有的 rank-zero wrapped F64 Scalar，输出拥有独立 Storage/Version 的 Bool 张量。实际真值转换、归约和分配均由 HIA 执行。
- **`sum(dtype=bool)`**：命令 115 扩展到上述源类型，保留它与 `any` / `all` 不同的空轴语义。浮点输入需要梯度时，先检查重复／越界轴，再按契约拒绝 Bool 求和。
- **`sum(dtype=float32)`**：Bool/I64 逐元素转成 F32 后进入 F32 求和内核，保留舍入次序；私有转换对象在成功和失败路径都回收。
- **F32 累加次序**：保留固定参考 SumKernel 的 8-lane 累加次序修正。
- **`cat` 分配与空梯度**：检查 saved-list 分配失败后再写入；零输出轴跳过无效前缀扫描；普通拼接元数据直接读取。rank-one 空输入的反向使用已有零值构造流程，保持真实 Storage/Version 语义。实际 62,207 次重复输入的空梯度 Version 与固定 Torch 同为 62,208。
- **统一原始回归**：当前映像接入完整原 35 生命周期案例、原 270 基础／NN／训练案例、原 32 Tensor/RNG/backward 轨迹、原 7 个资源配方和原 16 个 AsStrided 案例。原数值比较、分配预算、来源条件和失败记录保留。

### 明确的边界

| 项目 | 当前范围 |
| --- | --- |
| 平台与设备 | Linux x86-64 CPU |
| dtype | F32 为主要数值／AD 路径；Bool、I64 仅实现明确支持的操作；F64 仅有已有的 wrapped Scalar 表示，尚非通用 F64 Tensor 服务 |
| rank 与规模 | rank 0–4；每维最多 262,144；命令 115 / 118 / 119 最多 262,144 个逻辑元素；其他操作保留各自的工作、存储及输出限制 |
| `any` / `all` | 空轴元组不归约；空归约分别返回 False / True；带梯度浮点输入的结果为 detached Bool |
| Bool `sum` | 空轴元组归约全部轴；带梯度浮点输入被拒绝；无 `out=` 或未实现 dtype/backend 的完整契约 |
| AD 与系统功能 | 一阶反向 AD 的受限路径；高阶／前向 AD、完整序列化、加速器、分布式、compile/export 和数据管线尚未闭合 |

新命令的字段、轴和错误优先级见[逻辑归约说明](tests/logical_reductions/README.md)。当前能力边界集中在[能力清单](compatibility/capabilities.json)，专题文档中的历史有限资格不自动授予当前映像。

## 一个默认源码与构建入口

完整应用源码及有序分区位于 [`tools/current_runtime/source/`](tools/current_runtime/source/)。默认构建直接编译该目录的当前源码；每个对象均由固定 HIA ABI3 编译器新生成，再由原生 HIA 链接器生成当前 ELF。

```sh
make HEPTA_CURRENT_COMPILER=/absolute/path/to/stack-module-compiler \
     HEPTA_CURRENT_LINKER=/absolute/path/to/stack-linker-session

make test HEPTA_CURRENT_COMPILER=/absolute/path/to/stack-module-compiler \
          HEPTA_CURRENT_LINKER=/absolute/path/to/stack-linker-session \
          HEPTA_TORCH_PYTHON=/absolute/path/to/pinned-env/bin/python
```

默认产物是 `build/current-runtime/heptatorch`。`build.json` 绑定完整源码、全部新对象、最终 ELF、独立栈检查和所有编译／链接进程的退出及双流 EOF。`PASS_NATIVE_BUILD_ONLY` 只表示构建成功。工具身份、准备步骤及单独运行命令见[当前构建说明](docs/current-runtime.md)。

HIA 应用运行时依赖 Linux 内核与 x86-64 硬件。Python 用于构建组织、独立观察和验收；固定 PyTorch 只用于独立参考。编译器最初的 bootstrap 仍有明确宿主依赖，见[工具链构建](docs/build.md)。

参考环境固定为 CPython 3.12、Linux x86-64 和官方 CPU PyTorch 2.15.0 wheel，必须通过 11 项来源检查及真实安装回执。按[锁文件](compatibility/pytorch-oracle-lock.json)和 [CI 安装步骤](.github/workflows/current-runtime.yml)准备环境，再使用 `HEPTA_TORCH_PYTHON` 选择它。`make test` 使用这一解释器运行协调器及各检查阶段，避免不同 Python 版本的 AST 结构摘要差异；它不会隐式联网安装。

## 同一映像上的十三组必需门禁

`make test` 与 [Current runtime CI](.github/workflows/current-runtime.yml)共用 [`tests/qualify_current_runtime.py`](tests/qualify_current_runtime.py)。每组必须绑定同一份当前源码、实际 ELF 和本次冷构建报告。固定分母为 **697 个案例**，各组额外控制另行保留；案例数不等于完整上游 schema 数。

| 组名 | 案例数 | 保留的主要检查 |
| --- | ---: | --- |
| `bool27` | 27 | 原 Bool 求和参考、完整 owner/RNG、原 dtype／入口观察和分配失败恢复 |
| `original76` | 76 | 原 26 + 32 + 14 + 4 视图／reshape／标量案例；82,503 帧、481,580 项断言 |
| `integral24` | 24 | 原 260 个事件、全宽整数、溢出、空／非连续布局及生命周期 |
| `sum_f32_17` | 17 | 逐元素 cast 后求和、精确参考字节、临时对象回收与两阶段 OOM |
| `current35` | 35 | 6,344 帧、2,590,274 个数值；11 项来源条件及原 4 正／18 负控制 |
| `base270` | 270 | 基础算子、NN、优化器、训练、拼接与索引；270 个原生进程、284 帧和 14 个真实前缀／拒绝对 |
| `sum32` | 32 | 原 F32 求和的 626 个事件、64,415 帧、460,075 项断言及完整 owner |
| `live32` | 32 | Tensor/RNG/backward；2,132 帧、17,455 个精确 binary32 字、1,137,600 字节 Generator 状态 |
| `logical147` | 147 | 49 份固定输入分别比较 any、all、Bool sum；另有 18 个包／回滚控制和 2 个真实分配失败／恢复控制 |
| `resource7` | 7 | 原 R4 resource3、backward private cache1、R3 preserve3；378 帧、1,060 项检查，包含完整 owner/RNG 和 3 个真实 RLIMIT 分配失败／恢复控制 |
| `asstrided16` | 16 | 原 copy4、reduction4、lanes2、branches6；1,920 帧、10,272 项检查、176 份完整 owner 归档，以及 16 个原生退出／32 次 EOF |
| `cat4` | 4 | 三份完整原始成功响应与一份明确的扩容控制；完整请求逆转换、四个实际退出、双 EOF 和完整 owner |
| `metadata10` | 10 | 固定实际 Torch 对照：空／非空拼接、62,207 次重复梯度、62,208 次前向、形状／步长／offset／Version／grad presence；逐帧与完整 owner 读回 |

`current35`、`base270`、`live32` 运行固定 live Torch 参考；八个原参考组继续使用固定实际观察，`cat4` 使用四份原始协议夹具，`metadata10` 使用独立产生并封存的十项实际 Torch 观察。缺失、跳过、仅准备、失败、超时、不完整 EOF 或输入变化都不能得到统一 PASS。

功能扩展已取代的旧“应拒绝”控制仍保留原 FAIL。当前独立测试单独验证新增契约，不能改写旧期望或把旧失败重分类。原 35 中的 283 项显式 optimizer-state 绝对 Version 差异继续单独记录；参数 Version 比较仍要求原断言通过。

## 仓库整理

默认目录只提供一个当前完整 HIA 源码和一组当前分区。已删除从真实 owner 入口调用图不可达的 7 个核心函数，并将完整原始回归通过当前身份层接入同一构建；full35 只保留一个当前执行入口。原 76 参考包由 12,561,956 字节缩减为 131,848 字节，所需 8 个文件与旧归档逐字、逐权限核对一致；完整旧证据由固定 Git 历史和封存位置索引保护。已撤下 18 次由当前必需组替代的重复 CI 调用。

README 与能力清单只描述当前版本。旧默认 profile 的长篇过程元数据可从 Git 历史追溯。仍被比较器、来源证明或 bootstrap 实际使用的 fixture 保留原 SHA／模式；它们需要先完成对应迁移与真实回归，才能删除旧路径。清理不会缩减 2,590 / 688 / 17 的目标，也不会取消门禁。

| 文档 | 内容 |
| --- | --- |
| [当前构建](docs/current-runtime.md) | 唯一默认源码、固定工具与同一映像验收 |
| [当前审计](docs/current-audit.md) | 基线、PR38／PR39 融合、真实执行状态、删除证明与未闭合范围 |
| [继续执行](docs/continuation-plan.md) | 后续工作及独立证据要求 |
| [上游审计](docs/upstream-audit.md) | 固定目标、原始声明分母和覆盖要求 |
| [逻辑归约](tests/logical_reductions/README.md) | any/all、Bool sum 的协议、独立参考及资源控制 |
| [原 35 适配](tests/current_runtime_original35/README.md) | 原比较器、完整 owner 方程和显式 ViewInfo 根引用 |
| [原资源 7 适配](tests/current_runtime_resources/README.md) | 原配方／预算、真实分配失败、限额恢复与完整源码逆证明 |
| [原 AsStrided 16 适配](tests/current_runtime_asstrided/README.md) | 四套原比较器、完整帧／owner 读回、源码逆证明与真实进程退出 |
| [视图与标量](docs/view-reshape-scalar-r4.md)／[梯度对象](docs/gradient-objects-and-composition.md) | 视图、mutation、Saved、Version 与梯度生命周期的受限实现 |

`make acceptance-status` 检查目标账本一致性；`make full-acceptance` 在全量目标未闭合时保持失败。许可证与上游通知保留在 [LICENSE](LICENSE) 和 [`third_party/pytorch/`](third_party/pytorch/)。
