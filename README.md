# HeptaTorch

HIA 原生张量、自动微分、训练与 CPU Generator 运行时。当前默认源码包含 **712 个 HIA 核心函数、15 个独立编译单元**，直接生成 Linux x86-64 静态可执行文件。张量数值、分配、梯度传播、随机数与参数更新都在原生 HIA 程序中执行。

**整个 PyTorch 的全量重写仍未完成，项目状态为 HOLD。** 固定目标是 PyTorch **2.15.0**，原始分母为 **2,590 条原生算子声明、688 条导数声明、17 项系统验收门禁**。10 项进行中、7 项开放，没有完整上游 schema 或整项系统门禁被标为闭合。[当前审计](docs/current-audit.md)列出实现范围、原始证据、剩余工作与退役条件。

**当前 712 已在同一实际映像上完整通过九组 644 案例；对应新提交的 hosted 资格仍为 PENDING。** 独立读回再次核验了 65,627 个绑定输入的字节、模式、SHA，以及九个 runner 的退出状态、完整双流 EOF 和空 stderr。原始控制失败保持原文，[执行索引](validation/current-runtime712/evidence.json)记录本次新执行、独立准入与历史证据。

## 默认构建与验收

唯一完整应用源码和有序分区清单位于 [`tools/current_runtime/source/`](tools/current_runtime/source/)。固定 HIA 工具从每个完整源单元重新编译 15 个对象，再链接为当前映像。默认构建不读取旧应用 ELF，也不重放历次应用补丁链。

```sh
make HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/Hepta \
     HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools

make test HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/Hepta \
          HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools \
          HEPTA_TORCH_PYTHON=/absolute/path/to/pinned-env/bin/python
```

默认产物是 `build/current-runtime/heptatorch`。构建首先复现支持四参数入口的 entropy linker，并保存真实工具构建及原始流；当前应用的 `build.json` 绑定完整源码、15 个新对象、最终 ELF、独立栈检查和全部 16 个编译／链接进程的退出及双流 EOF。工具来源、直接构建命令和复用已完成冷构建的入口见[当前构建说明](docs/current-runtime.md)。

验收使用固定 CPython 3.12、Linux x86-64 和官方 CPU PyTorch 2.15.0 wheel，核对 11 项来源条件及真实安装回执 `build/pytorch-install.json`。用[锁文件](compatibility/pytorch-oracle-lock.json)和 [CI 安装步骤](.github/workflows/current-runtime.yml)准备参考环境，再通过 `HEPTA_TORCH_PYTHON` 明确选择它。外层协调器与 live-reference reader 必须使用同一个解释器；外层在运行任何案例前检查版本、真实路径和隔离参数。`make test` 不会隐式联网安装依赖。

应用运行时不依赖 Python、PyTorch、NumPy、libm 或 libc。Python 用于组织构建、独立观察和验收。需要真实 Torch 的四个组为 live35、live270、live32 及 Generator131 内的原 24 案例；其余组使用固定原始参考或独立原生／C++ 参考。初始编译器 bootstrap 有明确宿主依赖，见[工具链构建](docs/build.md)。

## 同一映像上的九组必需检查

`make test` 与 [Current runtime CI](.github/workflows/current-runtime.yml)调用同一个 coordinator。各组绑定当前完整源码、实际 ELF 和同一份真实冷构建报告。总门禁要求以下 **644 个案例**及各自额外控制；644 不是完整上游 schema 的覆盖数。

| 组 | 案例数 | 主要检查 |
| --- | ---: | --- |
| Bool 求和 | 27 | 固定参考、完整 owner/RNG、dtype、入口、真实分配失败和恢复 |
| 原视图／reshape／标量 | 76 | 26 + 32 + 14 + 4 原案例、完整原断言和 EOF |
| 原整数求和 | 24 | 全宽整数、溢出、空／非连续布局与生命周期 |
| Bool/I64 `sum(dtype=float32)` | 17 | 逐元素转换后求和、精确字节、私有临时对象回收、两阶段 OOM |
| 原跨帧对象生命周期 | 35 | 原数值与生命周期比较、11 项来源条件、4 个正控制与 18 个负控制 |
| 原基础算子与训练 | 270 | 广播、NN、优化器、训练、拼接、索引及 14 个真实部分成功／拒绝路径 |
| 原 Tensor／RNG／反向传播 | 32 | 原完整结构、Generator 状态、4 个真实部分失败和 1 个拒绝 |
| 原生 CPU Generator 与入口 | 131 | 原 Torch24 + 原 native11 + 原 entry59 + 新跨协议／入口控制37 |
| 原 F32 求和 | 32 | 全部 626 个原参考事件、数值、布局、身份和 owner 状态 |

缺失、跳过、仅准备、失败、超时、不完整 EOF 或输入变化都会阻止总 PASS。扩展功能取代旧“应拒绝”期望时，原 FAIL 仍保留，由外层精确核对已改变的契约和新独立控制；不会把旧原始报告改成 PASS。

## 当前范围与身份

当前有限 CPU 配置包含 F32 数值和一阶反向 AD、Bool/I64 张量及部分混合路径、Tensor/Storage/Version 所有权、强弱引用与 Saved 生命周期、视图和原地历史、选定 NN 算子、SGD/Adam/AdamW、CPU RNG 与持久 Generator 对象。新入口把原 Tensor 与 Generator 协议交给各自原生处理器，并接入真实 entropy 输入。二者句柄可互换、完整公共 dispatcher、并发 Generator、全部 dtype／设备／分布等能力仍未被声明为完成。

源码 SHA-256：`fac3d413c6c0f6874d224cbea312108ee2c8baa027102d41890d85094a5faa38`

实际冷构建 ELF SHA-256：`5b7aac203d2c913d43781f0da45d40e9bac2037c2bc074ec30f8dd663710a805`

源码有 9,536 个核心块、34,167 条核心指令，Tensor 描述符仍为 672 字节。完整来源证明保持原 701 个 Tensor 核心函数和 12 个 Tensor 单元的字节不变。独立 F64 与后续合成工作不属于本次 712 默认源码、门禁或资格。具体接口边界以[能力清单](compatibility/capabilities.json)为准。

## 仓库整理与后续工作

当前实现只保留一套 canonical 源与当前分区。仍作为固定参考的旧比较器、来源证明与原始记录集中在按 SHA／mode 校验的 fixture 中。原 76 组只展开所需的 8 份参考文件，完整历史归档保留恢复索引。独立旧 Generator 的 19 个源文件、旧 checker 和原 manifest 退出活代码路径，其精确原件与 7 份历史报告仍可恢复；两项重复旧 CI 调用改由当前 Generator131 必需组覆盖。清理与资格状态见[退役索引](validation/current-generator-original26/retirement.json)和[当前执行索引](validation/current-runtime712/evidence.json)。

编译器 bootstrap、461 对旧协议夹具、其他独立 dtype／优化器／资源门禁，以及 F64 原验证器对旧源的绑定仍有实际消费者。原 live32 的 13 次旧 Native 调用和 3 个旧观察器 profile 调用已由完整当前组接替，其余独立检查保留。继续迁移必须保持完整原比较、原始失败与新映像的真实执行，不能靠删测试完成退役。

| 文档 | 内容 |
| --- | --- |
| [当前构建](docs/current-runtime.md) | 唯一默认源码、entropy 工具、冷构建和九组验收 |
| [当前审计](docs/current-audit.md) | 当前证据、17 项未闭合门禁、清理边界 |
| [继续执行](docs/continuation-plan.md) | 资格与实现工作的依赖顺序 |
| [上游审计](docs/upstream-audit.md) | 固定目标、schema 分母和覆盖约束 |
| [整数张量](docs/integer-tensors.md)／[混合 AD](docs/mixed-integer-autograd.md) | dtype 与自动微分的有限范围 |
| [可微视图](docs/tensor-to-differentiable-views.md)／[R4 视图与标量](docs/view-reshape-scalar-r4.md) | view、mutation、Saved、Version |
| [公开梯度](docs/gradient-objects-and-composition.md)／[保留梯度](docs/retained-gradients.md) | 对象生命周期与组合验证 |
| [CPU 分布](docs/cpu-distributions.md)／[owned Adam](docs/owned-adam.md) | RNG、优化器状态与历史有限资格 |

`make acceptance-status`核对清单一致性。`make full-acceptance`在全量目标尚未闭合时保持失败。完整前端／调度、全部 dtype 和布局、全算子／AD 契约、序列化、加速器、分布式、编译导出、数据管线及全部平台仍需实现与验收。

许可证与上游通知保留在 [LICENSE](LICENSE) 和 [`third_party/pytorch/`](third_party/pytorch/)。
