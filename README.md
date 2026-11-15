# HeptaTorch

HIA 原生张量、自动微分和神经网络训练运行时。核心实现由 HIA 汇编器直接生成 Linux x86-64 静态可执行文件；数值运算、梯度传播和参数更新均在该程序内执行。

**当前状态：CPU binary32 训练内核已经运行并验证；整个 PyTorch 的全量重写尚未完成。** 完整目标固定为 PyTorch **v2.15.0**。仓库保留全部 **2,590 条原生算子声明**和子系统验收清单，记录已实现的具体范围及仍需完成的工作。

## 已实现

| 范围 | 当前实现 |
| --- | --- |
| 数值 | IEEE binary32 四则运算、平方根、整数转换、软件融合乘加；原生 exp、log、sigmoid、tanh |
| 张量 | rank 0–4、形状、步长、共享存储、广播、空维度、transpose、连续 view、clone、detach |
| 自动微分 | 动态反向图、显式梯度种子／向量雅可比积、分支和广播梯度累加、重复 backward、zero_grad、保存值版本检查 |
| 基本算子 | 加减乘除、ReLU、neg、sqrt、exp、log、sigmoid、tanh、向量和广播批量矩阵乘法、全维和单轴 sum/mean |
| 索引 | index_select、gather；精确 i64 索引载荷、重复索引梯度累加、非连续输入和空索引 |
| 神经网络 | rank 1–4 输入的 Linear、MSE、softmax/log-softmax、最后一维 LayerNorm、冻结统计量的 BatchNorm、概率目标交叉熵 |
| 空间算子 | NCHW/OIHW conv2d、分组与 depthwise 卷积、逐轴 stride/dilation/padding、平均池化和最大池化，以及对应反向传播 |
| 优化器 | SGD、momentum、Nesterov、Adam、AdamW；显式状态张量与版本失效检查 |
| 训练生命周期 | 保留参数／状态的图回收、工作区复用、带代际信息的张量 ID、旧句柄拒绝；单次请求内反复训练 |
| 执行接口 | 有边界检查的二进制命令协议、明确错误状态、原生 HIA 十六进制解码工具 |

本次实现使用软件浮点算术。运行时无 Python、PyTorch、NumPy、libm 或 libc 依赖。HIA 编译器的初始 bootstrap 使用其上游 Python 实现和 GNU ld，随后由 HIA 编译器自行重建；构建与运行依赖见 [构建说明](docs/build.md)。

## 构建与验收

```sh
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler test
```

编译器、上游源版本与校验和均固定。GitHub Actions 从固定的 Hepta 源码重新 bootstrap 编译器，再编译本仓库并执行原生回归夹具。`make test` 使用本仓库的 HIA 解码器，不需要 Python 或 `xxd`。

当前采用单独命名的扩容编译器配置，允许 256 个函数和 2,560 个总基本块。构建会从原始固定编译器出发，通过 HIA 编译与 HIA ELF 打包完成四代重建。生成配置及原生重建配方见 [构建说明](docs/build.md#reproducing-the-capacity-profile-from-the-original-compiler)。

运行已经验证的 1,000 步 SGD 训练命令流：

```sh
./build/unhex < tests/fixtures/session-review-reset-SGD-1000-real-iterations.input.hex > build/train.request
./build/heptatorch < build/train.request > build/train.response
./build/unhex < tests/fixtures/session-review-reset-SGD-1000-real-iterations.output.hex > build/train.expected
cmp build/train.response build/train.expected
```

命令流最小化平方损失，每一步执行前向、反向、参数更新和图回收。参数保持原 ID，临时图复用描述符与工作区。原有的两层 ReLU／18 步 Adam 训练夹具也保留在回归集中。

## 已观察到的验证结果

- 基础数值层：**2,314,020** 次原生运算与独立 NumPy binary32 参考逐位一致，算术 NaN 统一为规范 quiet NaN。
- exp/log/sigmoid/tanh：**3,424,359** 次测量；另用 90 位精度参考检查 **20,025** 个输入。测量误差及适用解释见 [数值说明](docs/numerics.md)。
- 集成：广播、视图、梯度、空间算子、模块、优化器、异常输入和容量边界，均有独立参考或明确拒绝结果。
- 当前集成版本：**275 对原生回归夹具**通过，原有 221 份请求输入未变，其中 5 份 Adam 预期响应已随舍入修正重新通过独立参考验证。图生命周期的 **26 项／40,123 个值**独立检查通过，包括单次请求中的 **1,000 步 SGD、1,000 步 Adam**，以及在 1 MiB 工作区内反复回收共 **9,600,000 字节**的临时张量数据和梯度缓冲。另有 **10 项**失败原子性检查，每项比较除错误状态外的全部 **1,048,568 字节**，均未发生修改。[当前原生验证](validation/session-native.json)保存具体案例、误差和最终源码／二进制身份。
- 上一批算子扩展的矩阵与 Linear **63 项／4,593 个值**检查、**2,514 项**独立矩阵形状／梯度／拒绝检查，以及 **11 项／529 个值**跨算子集成检查均通过；对应历史源码身份保留在[算子扩展验证](validation/native-extension.json)。
- 托管 CI：当前版本的 **159 项／8,471 个真实 PyTorch 数值对照**全部通过，包含无掩码／无 dropout 注意力、**1,024 步 SGD、512 步 Adam 和 512 步 AdamW**。长 SGD 对照逐位一致；长 Adam／AdamW 最大绝对误差分别为 **1.49×10⁻⁸／5.96×10⁻⁸**。[成功运行](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43743468220)同时验证了编译器自重建、275 对原生夹具和 11 项 oracle 来源检查。[当前完整回执](validation/hosted-compiler-profile.json)保留逐项结果；[首次失败回执](validation/hosted-session-failure.json)也保留，用于说明 Adam 舍入问题及修正过程。
- 编译容量：扩大配置的四代原生镜像和可执行文件逐字节一致，**16 项容量门禁**、256 个显式标量导出的实际调用、高编号调用图及脏工作区检查通过。原始和扩容编译器生成的当前 HeptaTorch 与 unhex 二进制完全相同。[工具链验证](validation/compiler-profile.json)保存具体配置、来源、内存布局及独立验证记录；[扩容构建托管回执](validation/hosted-compiler-profile.json)确认了整个新构建流程、容量门禁及原有数值门禁通过。
- 线性回归：**30 次 SGD 更新**，更新前损失从 **5.1570835** 降至 **0.0342863**。
- 两层 ReLU 分类器：**18 次 Adam 更新**，更新前交叉熵从 **0.6822657** 降至 **0.0017283**；该固定 8 样本任务更新后分类正确率 **100%**。

训练结果是对已实现执行链的功能验证。完整 PyTorch 兼容性、规模扩展和性能仍有独立验收任务。[验证记录](validation/README.md)将组件验证、最终集成验证和 CI 运行证据分别绑定到源码及可执行文件哈希。

## 当前边界与完整目标

当前设备为 CPU，dtype 为 binary32，张量 rank 不超过 4；同时存活的张量描述符最多 255 个，可以通过图回收在同一请求内复用。输入、输出和工作区仍各为 1 MiB；跨请求持久化尚未实现。卷积、矩阵乘法和池化各有每次 16,777,216 项的计算预算。整数索引载荷不等于完整整数张量 dtype 支持。空间、优化器和自动微分的具体限制见 [接口协议](docs/protocol.md)及 [能力清单](compatibility/capabilities.json)。

本次审计修复了池化缺少计算预算和 Adam 长训练舍入偏差，统一了二维与批量矩阵乘法，实现了保留参数／状态的图回收，并扩大了编译器容量。当前编译单元使用 121／256 个函数名额、1,439／2,560 个基本块及 5,751／8,192 条指令。后续的架构重点是带资源约束的跨模块导入与张量 ABI、更多 dtype 和设备、规模扩展与跨请求状态管理。具体证据及完整目标见 [审计更新](docs/upstream-audit.md#compiler-capacity-profile)。

完整目标尚包括：完整 dtype/layout/promotion 语义、其余算子及重载、高阶和前向自动微分、CUDA/ROCm/MPS/XPU、分布式、图捕获与编译优化、数据加载、RNG、完整 checkpoint/serialization、Python/C++ 前端兼容和跨平台构建。[上游审计](docs/upstream-audit.md)列出这些目标的验收条件。

## 仓库结构

```text
src/heptatorch.hepta       唯一运行时实现
tools/unhex.hepta          原生测试数据解码器
docs/                     构建、协议、数值与上游审计
compatibility/            固定上游清单与已验证能力
tests/fixtures/           独立校验后的命令流和预期结果
validation/               验证记录与来源绑定
third_party/pytorch/      上游许可证与通知
```

`build/` 为本地构建产物，不进入版本库。仓库中的许可证、测试、兼容性清单和失败边界均是最终交付的一部分。
