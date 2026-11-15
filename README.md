# HeptaTorch

HIA 原生张量、自动微分和神经网络训练运行时。核心实现由 HIA 汇编器直接生成 Linux x86-64 静态可执行文件；数值运算、梯度传播和参数更新均在该程序内执行。

**当前状态：CPU binary32 训练内核已经运行并验证；整个 PyTorch 的全量重写尚未完成。** 完整目标固定为 PyTorch **v2.15.0**。仓库保留全部 **2,590 条原生算子声明**和子系统验收清单，记录已实现的具体范围及仍需完成的工作。

## 已实现

| 范围 | 当前实现 |
| --- | --- |
| 数值 | IEEE binary32 四则运算、平方根、整数转换；原生 exp、log、sigmoid、tanh |
| 张量 | rank 0–4、形状、步长、共享存储、广播、空维度、transpose、连续 view、clone、detach |
| 自动微分 | 动态反向图、显式梯度种子／向量雅可比积、分支和广播梯度累加、重复 backward、zero_grad、保存值版本检查 |
| 基本算子 | 加减乘除、ReLU、neg、sqrt、exp、log、sigmoid、tanh、向量和广播批量矩阵乘法、全维和单轴 sum/mean |
| 索引 | index_select、gather；精确 i64 索引载荷、重复索引梯度累加、非连续输入和空索引 |
| 神经网络 | rank 1–4 输入的 Linear、MSE、softmax/log-softmax、最后一维 LayerNorm、冻结统计量的 BatchNorm、概率目标交叉熵 |
| 空间算子 | NCHW/OIHW conv2d、分组与 depthwise 卷积、逐轴 stride/dilation/padding、平均池化和最大池化，以及对应反向传播 |
| 优化器 | SGD、momentum、Nesterov、Adam、AdamW；显式状态张量与版本失效检查 |
| 执行接口 | 有边界检查的二进制命令协议、明确错误状态、原生 HIA 十六进制解码工具 |

本次实现使用软件浮点算术。运行时无 Python、PyTorch、NumPy、libm 或 libc 依赖。HIA 编译器的初始 bootstrap 使用其上游 Python 实现和 GNU ld，随后由 HIA 编译器自行重建；构建与运行依赖见 [构建说明](docs/build.md)。

## 构建与验收

```sh
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler test
```

编译器、上游源版本与校验和均固定。GitHub Actions 从固定的 Hepta 源码重新 bootstrap 编译器，再编译本仓库并执行原生回归夹具。`make test` 使用本仓库的 HIA 解码器，不需要 Python 或 `xxd`。

运行已经验证的两层 ReLU 网络训练命令流：

```sh
./build/unhex < tests/fixtures/mlp-adam-training-18-steps.input.hex > build/train.request
./build/heptatorch < build/train.request > build/train.response
./build/unhex < tests/fixtures/mlp-adam-training-18-steps.output.hex > build/train.expected
cmp build/train.response build/train.expected
```

命令流包含实际的前向、反向和 18 次 Adam 更新。输入、标签、初始参数及优化器状态均由请求提供。

## 已观察到的验证结果

- 基础数值层：**2,314,020** 次原生运算与独立 NumPy binary32 参考逐位一致，算术 NaN 统一为规范 quiet NaN。
- exp/log/sigmoid/tanh：**3,424,359** 次测量；另用 90 位精度参考检查 **20,025** 个输入。测量误差及适用解释见 [数值说明](docs/numerics.md)。
- 集成：广播、视图、梯度、空间算子、模块、优化器、异常输入和容量边界，均有独立参考或明确拒绝结果。
- 当前集成版本：**221 对原生回归夹具**通过；新增矩阵与 Linear 的 **63 项／4,593 个值**检查、**2,514 项**独立矩阵形状／梯度／拒绝检查，以及 **11 项／529 个值**跨算子集成检查通过。各项的单位、独立参考和源码身份见 [本次原生验证](validation/native-extension.json)。
- 托管 CI：原基线的 **76 项／2,754 个真实 PyTorch 数值对照**已通过，保留其[成功运行](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43738165041)和[历史回执](validation/hosted.json)。当前源码的独立 PyTorch 门扩展为 **153 项**，包括三种无掩码、无 dropout 的注意力组合；实际运行状态单独记录在 [本次托管回执](validation/hosted-extension.json)。
- 线性回归：**30 次 SGD 更新**，更新前损失从 **5.1570835** 降至 **0.0342863**。
- 两层 ReLU 分类器：**18 次 Adam 更新**，更新前交叉熵从 **0.6822657** 降至 **0.0017283**；该固定 8 样本任务更新后分类正确率 **100%**。

训练结果是对已实现执行链的功能验证。完整 PyTorch 兼容性、规模扩展和性能仍有独立验收任务。[验证记录](validation/README.md)将组件验证、最终集成验证和 CI 运行证据分别绑定到源码及可执行文件哈希。

## 当前边界与完整目标

当前设备为 CPU，dtype 为 binary32，张量 rank 不超过 4；一次请求最多 255 个张量描述符，输入、输出和工作区各为 1 MiB。卷积、矩阵乘法和池化各有每次 16,777,216 项的计算预算。整数索引载荷不等于完整整数张量 dtype 支持。空间、优化器和自动微分的具体限制见 [接口协议](docs/protocol.md)及 [能力清单](compatibility/capabilities.json)。

本次审计修复了池化缺少计算预算的问题，并统一了二维与批量矩阵乘法的原生实现。后续的架构重点是 HIA 原生模块链接与张量 ABI、可回收的训练图及持久参数状态：当前编译单元已使用固定编译器的 117／128 个函数名额和 1,364／1,536 个基本块。具体证据及完整目标见 [审计更新](docs/upstream-audit.md#native-extension-audit)。

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
