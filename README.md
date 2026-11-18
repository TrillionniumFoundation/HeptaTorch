# HeptaTorch

HIA 原生张量、自动微分与训练运行时。当前默认源码包含 **701 个 HIA 核心函数、14 个独立编译单元**，直接生成 Linux x86-64 静态可执行文件。张量数值、分配、梯度传播、随机数和参数更新均由该 HIA 程序执行。

**整个 PyTorch 的全量重写仍未完成，项目状态为 HOLD。** 目标固定为 PyTorch **2.15.0**，原始分母为 **2,590 条原生算子声明、688 条导数声明、17 项系统验收门禁**。目前 10 项进行中、7 项开放，没有完整上游 schema 或整项系统门禁被标为闭合。[当前审计](docs/current-audit.md)列出实现范围、实际执行结果、剩余工作与退役条件。

## 一个默认源码和验收入口

完整应用源码与有序分区清单位于 [`tools/current_runtime/source/`](tools/current_runtime/source/)。默认构建不读取旧应用 ELF，也不重放历次应用补丁链；14 个对象均由固定的 HIA ABI3 编译器新编译，再由原生 HIA 链接器生成当前映像。

```sh
make HEPTA_CURRENT_COMPILER=/absolute/path/to/stack-module-compiler \
     HEPTA_CURRENT_LINKER=/absolute/path/to/stack-linker-session

make test HEPTA_CURRENT_COMPILER=/absolute/path/to/stack-module-compiler \
          HEPTA_CURRENT_LINKER=/absolute/path/to/stack-linker-session \
          HEPTA_TORCH_PYTHON=/absolute/path/to/pinned-env/bin/python
```

默认产物为 `build/current-runtime/heptatorch`。`build.json` 绑定完整源码、14 个新对象、最终 ELF、独立栈检查以及全部 15 个编译／链接进程的退出和双流 EOF。工具身份、冷构建及单独执行命令见 [当前构建说明](docs/current-runtime.md)。

验收使用固定 CPython 3.12、Linux x86-64 和官方 CPU PyTorch 2.15.0 wheel，核对 11 项来源条件及真实安装回执 `build/pytorch-install.json`。使用 [锁文件](compatibility/pytorch-oracle-lock.json)和 [CI 安装步骤](.github/workflows/current-runtime.yml)准备参考环境，再通过 `HEPTA_TORCH_PYTHON` 明确选择它；`make test` 不会隐式联网安装依赖。

HIA 应用运行时不依赖 Python、PyTorch、NumPy、libm 或 libc。Python 用于构建组织、独立观察与验收；两组完整 live-reference 测试使用固定 PyTorch 计算独立期望。编译器的初始 bootstrap 另有明确的宿主依赖，见 [工具链构建](docs/build.md)。

## 同一映像上的七组必需检查

`make test` 与 [Current runtime CI](.github/workflows/current-runtime.yml)调用同一个 coordinator。每组均绑定当前源码、实际 ELF 和本次冷构建报告。总门禁必须执行以下 **481 个案例**，并完成各自额外的错误、生命周期和资源控制；它们不是 481 个完整上游 schema。

| 组 | 案例数 | 保留的主要检查 |
| --- | ---: | --- |
| Bool 求和 | 27 | 原参考、完整 owner/RNG、dtype 与入口控制、真实分配失败和恢复 |
| 原视图／reshape／标量 | 76 | 26 + 32 + 14 + 4 原案例、82,503 帧、481,580 项断言 |
| 原 F32 求和 | 32 | 全部原参考事件、数值、布局、身份与完整 owner 检查 |
| 原整数求和 | 24 | 全宽整数、溢出、空／非连续布局与原生命周期检查 |
| Bool/I64 `sum(dtype=float32)` | 17 | 先逐元素转 F32 再求和，精确参考字节、私有转换对象回收、两阶段 OOM |
| 原跨帧与对象生命周期 | 35 | 6,344 帧、2,590,274 个数值；11 项 provenance 与原独立正／负控制 |
| 原基础算子与训练 | 270 | 广播、NN、优化器、长训练、拼接、索引及 14 个原拒绝案例；270 个原生进程、284 帧 |

缺失、跳过、仅准备、失败、超时、不完整 EOF 或输入变化都不能生成总 PASS。旧控制中已经被明确扩展功能取代的“应拒绝”期望仍保留原 FAIL，由当前独立控制验证新增契约。实际运行状态与完整报告身份见 [执行清单](validation/current-runtime701/evidence.json)。

## 当前实现与本次修正

当前 CPU 受限配置包含 F32 数值与一阶反向 AD、Bool/I64 张量及部分混合路径、Tensor/Storage/Version 所有权、强弱引用和 Saved 生命周期、视图与原地历史、原生 CPU RNG、选定 NN 算子及 SGD/Adam/AdamW。具体 dtype、布局、设备和接口边界以 [能力清单](compatibility/capabilities.json)和各专题文档为准。

本次实现新增 Bool/I64 到 F32 的求和路径，并修复两项真实执行发现的问题：

- 固定参考 wheel 的求和内核使用 AVX2 的 8 个 F32 lane。HIA 现在按该实际累加顺序执行；首次错误结果及独立 cast／sum 诊断保留。
- 重复出现、只有自身需要梯度的空 `cat` 输入，使用可累加的自有空梯度缓存。修正保留原成功返回值和共享存储保护；原 270 项中的首次失败与修复后的完整新运行分别记录。

源码 SHA-256：`82360ce87dae6f9a004ef50be4f991b6f83905a83f4b17ca8d6634827b7bd313`

冷构建 ELF SHA-256：`80445a8c433f20314d3bc7b00ebb68f6a70fa200ee2e6aba193cff21aabfbc17`

当前源码有 9,352 个核心块、33,473 条核心指令，Tensor 描述符仍为 672 字节。完整源码的变化与三层逆向证明按实际身份绑定；改变源码或清单不会转移旧映像的资格。

## 仓库整理与继续工作

默认目录保留当前完整源码及其当前分区。旧的 12 个 `sum_bool_*` 分区由新的 `cat_f32_*` 分区替代；仍需作为独立参考的旧比较器与原聚合源码集中为一次 hash／mode 绑定的 fixture。README 不再逐批重复旧默认实现和过程记录。

尚有实际依赖的旧文件必须先迁移：编译器 bootstrap 与固定来源证明、旧协议的 461 对夹具、原 live32／Generator24 的仓库入口、独立 dtype／优化器／资源门禁，以及 F64 验证器对旧 143 函数源的绑定。原 live32 已在同一新映像上独立通过 32 条轨迹、2,132 帧和 17,455 个精确 binary32 字比较；这批新增执行尚未加入上述七组默认门禁，其旧 CI 调用仍待迁移。不能通过删掉测试或改写失败记录完成退役。[继续执行计划](docs/continuation-plan.md)与 [当前审计](docs/current-audit.md)记录这些依赖。

| 文档 | 内容 |
| --- | --- |
| [当前构建](docs/current-runtime.md) | 唯一默认源码、工具身份、冷编译与七组验收 |
| [当前审计](docs/current-audit.md) | 实际证据、17 项未闭合门禁、清理边界 |
| [继续执行](docs/continuation-plan.md) | 当前身份、工作顺序与原始证据规则 |
| [上游审计](docs/upstream-audit.md) | 固定目标、schema 分母与覆盖约束 |
| [整数张量](docs/integer-tensors.md)／[混合 AD](docs/mixed-integer-autograd.md) | 独立来源的 dtype 和自动微分范围 |
| [可微视图](docs/tensor-to-differentiable-views.md)／[R4 视图与标量](docs/view-reshape-scalar-r4.md) | view、mutation、Saved、Version 及其边界 |
| [公开梯度](docs/gradient-objects-and-composition.md)／[保留梯度](docs/retained-gradients.md) | 对象生命周期与组合验证 |
| [CPU 分布](docs/cpu-distributions.md)／[owned Adam](docs/owned-adam.md) | RNG、优化器状态与历史有限资格 |

`make acceptance-status`核对清单一致性。`make full-acceptance`在全量目标尚未闭合时保持失败。完整前端／调度、全部 dtype 与布局、全算子／AD 契约、序列化、加速器、分布式、编译导出、数据管线和全部平台仍需实现与验收。

许可证与上游通知保留在 [LICENSE](LICENSE) 和 [`third_party/pytorch/`](third_party/pytorch/)。
