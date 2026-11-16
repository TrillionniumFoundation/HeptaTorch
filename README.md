# HeptaTorch

HIA 原生张量、自动微分和神经网络训练运行时。核心实现由 HIA 汇编器直接生成 Linux x86-64 静态可执行文件；数值运算、梯度传播和参数更新均在该程序内执行。

**当前状态：CPU binary32 训练内核、拼接与索引累加已经原生运行并验证；整个 PyTorch 的全量重写尚未完成。** 完整目标固定为 PyTorch **v2.15.0**。仓库保留全部 **2,590 条原生算子声明**和 [17 项完整验收门禁](compatibility/full-acceptance.json)，目前没有完整上游 schema 或整项门禁被标为已闭合。

## 已实现

| 范围 | 当前实现 |
| --- | --- |
| 数值 | IEEE binary32 四则运算、平方根、整数转换、软件融合乘加；原生 exp、log、sigmoid、tanh |
| 张量 | rank 0–4、形状、步长、共享存储、广播、空维度、transpose、连续 view、clone、detach；空张量完整维度检查与规范连续步长 |
| 自动微分 | 动态反向图、显式梯度种子／向量雅可比积、分支和广播梯度累加、重复 backward、zero_grad、保存值版本检查 |
| 基本算子 | 加减乘除、ReLU、neg、sqrt、exp、log、sigmoid、tanh、向量和广播批量矩阵乘法、全维和单轴 sum/mean |
| 拼接 | cat、stack；非连续输入、负轴、空张量、重复父节点梯度累加，以及实际 channels-last 输出布局 |
| 索引 | index_select、gather、scatter_add、带 binary32 alpha 的 index_add；精确 i64 载荷、重复索引累加、非连续输入和空索引 |
| 神经网络 | rank 1–4 输入的 Linear、MSE、softmax/log-softmax、最后一维 LayerNorm、冻结统计量的 BatchNorm、概率目标交叉熵 |
| 空间算子 | NCHW/OIHW conv2d、分组与 depthwise 卷积、逐轴 stride/dilation/padding、平均池化和最大池化，以及对应反向传播 |
| 优化器 | SGD、momentum、Nesterov、Adam、AdamW；显式状态张量与版本失效检查 |
| 训练生命周期 | 保留参数／状态的图回收、工作区复用、带代际信息的张量 ID、旧句柄拒绝；单次请求内反复训练 |
| 执行接口 | 有边界检查的二进制命令协议、明确错误状态、原生 HIA 十六进制解码工具 |
| 原生模块 | 同一编译器配置下的借用缓冲区模块编译、签名／效果检查和 HIA 静态链接；通用资源所有权 ABI 仍未完成 |

本次实现使用软件浮点算术。运行时无 Python、PyTorch、NumPy、libm 或 libc 依赖。HIA 编译器的初始 bootstrap 使用其上游 Python 实现和 GNU ld，随后由 HIA 编译器自行重建；构建与运行依赖见 [构建说明](docs/build.md)。

## 构建与验收

```sh
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler \
     HEPTA_MODULE_COMPILER=/absolute/path/to/hia-module-compiler \
     HEPTA_MODULE_LINKER=/absolute/path/to/hia-module-linker \
     build/heptatorch test
```

三个路径分别指向普通编译器、模块编译器和原生模块链接器，均有固定版本与校验和。默认 `build/heptatorch` 经独立模块编译和 HIA 静态链接生成。模块构建使用 Python 组织源码和传递封包；源程序检查、机器码生成、模块接纳、链接和数值执行均由 HIA 原生程序完成。`make test` 的回归循环使用本仓库的 HIA 解码器和原生程序，不需要 Python 或 `xxd` 处理测试数据。

保留单编译单元参考构建与同一套 461 对夹具：

```sh
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler \
     monolithic test-monolithic
```

参考产物为 `build/heptatorch-monolithic`。GitHub Actions 从固定的 Hepta 源码重新 bootstrap 工具链，执行两条原生构建路径，并分别运行同一份真实 PyTorch 对照门禁。

当前采用单独命名的扩容编译器配置，允许 256 个函数和 2,560 个总基本块。构建会从原始固定编译器出发，通过 HIA 编译与 HIA ELF 打包完成四代重建。生成配置及原生重建配方见 [构建说明](docs/build.md#reproducing-the-capacity-profile-from-the-original-compiler)。

默认模块路径将完整的 143 个函数分为 63 个内核函数和 80 个运行时函数，实际执行带 52 个外部依赖的原生模块编译、接口接纳与静态链接。函数体仍来自唯一的 `src/heptatorch.hepta`，生成的模块不是第二份手工维护实现。模块运行与边界证据单独记录在 [原生模块验证](validation/native-modules.json)，不据此宣称通用所有权、任意外部指针或所有平台 ABI 已完成。

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
- 本轮合并原生检查：cat/stack **81 项／1,673 个精确值**及 **16 项失败原子性**通过；scatter_add/index_add **100 项／1,553 个值**、补充 **9 项／1,652 个值**通过；独立组合检查 **64 项／3,738 个值**通过，包含共享梯度、矩阵乘法和多轮图回收。另有 **3 项／102 个值**检查把 cat、stack、scatter_add、index_add 和视图接在同一张图中。[本轮原生证据](validation/tensor-composition-native.json)绑定实际合并源码与二进制。
- 默认模块链接映像已通过 **461 对原生回归夹具**：原有 275 对，加 81 对拼接、100 对索引累加、3 对跨算子组合和 2 对含矩阵乘法的组合。原有请求输入保持不变；空形状 `[2,0]` 的旧步长 golden 仅在响应 byte 280 从 0 改为 1，对应 `[0,1]` 修正为 `[1,1]`。该修正已有固定源码和真实 PyTorch 观察支持。单编译单元参考路径使用同一份 461 对集合；两份二进制的独立身份和实际回放结果各自记录。
- 真实 PyTorch 边界观察：**22 项空张量、12 项索引边界、6 项 alpha 舍入和 8 项布局选择**已执行并保留各自 CI 来源。合并程序重放布局观察的 **924 个值及全部步长**、索引边界的 **64 个值及拒绝阶段**、alpha 的 **102 个值**，均一致。[原始观察](validation/hosted-boundary-observations.json)与最终二进制回放分别保存，观察所用的早期源码身份未改写。
- 本轮主 PyTorch 对照门禁已在两份实际映像上分别 **PASS：每份 268 项／12,932 个数值／11 项来源检查，零失败**。保持原 159 项的输入和容差，新增 35 项拼接及 74 项索引累加；[实际运行](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43788929713)验证提交 `724cbf2eb19d2ef8d400615d7196f5397cb402f6`，同时通过两份 461 对原生夹具、模块边界和工具链重建。[完整新回执](validation/hosted-tensor-composition.json)分别绑定两份可执行文件，保留逐项结果。
- 历史图生命周期批次：**275 对原生回归夹具**通过，原有 221 份请求输入未变，其中 5 份 Adam 预期响应已随舍入修正重新通过独立参考验证。图生命周期的 **26 项／40,123 个值**独立检查通过，包括单次请求中的 **1,000 步 SGD、1,000 步 Adam**，以及在 1 MiB 工作区内反复回收共 **9,600,000 字节**的临时张量数据和梯度缓冲。另有 **10 项**失败原子性检查，每项比较除错误状态外的全部 **1,048,568 字节**，均未发生修改。[历史原生验证](validation/session-native.json)继续保留该批次的具体案例、误差和源码／二进制身份。
- 上一批算子扩展的矩阵与 Linear **63 项／4,593 个值**检查、**2,514 项**独立矩阵形状／梯度／拒绝检查，以及 **11 项／529 个值**跨算子集成检查均通过；对应历史源码身份保留在[算子扩展验证](validation/native-extension.json)。
- 历史托管 CI：上一批次的 **159 项／8,471 个真实 PyTorch 数值对照**全部通过，包含无掩码／无 dropout 注意力、**1,024 步 SGD、512 步 Adam 和 512 步 AdamW**。长 SGD 对照逐位一致；长 Adam／AdamW 最大绝对误差分别为 **1.49×10⁻⁸／5.96×10⁻⁸**。[成功运行](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43743468220)同时验证了编译器自重建、275 对原生夹具和 11 项 oracle 来源检查。[该批次完整回执](validation/hosted-compiler-profile.json)保留逐项结果；[首次失败回执](validation/hosted-session-failure.json)也保留，用于说明 Adam 舍入问题及修正过程。
- 编译容量：扩大配置的四代原生镜像和可执行文件逐字节一致，**16 项容量门禁**、256 个显式标量导出的实际调用、高编号调用图及脏工作区检查通过。原始和扩容编译器对当时 121 个函数的 HeptaTorch 及 unhex 生成完全相同的二进制；当前 143 个函数的版本使用扩容配置。[工具链验证](validation/compiler-profile.json)保存具体配置、来源、内存布局及独立验证记录；[扩容构建托管回执](validation/hosted-compiler-profile.json)继续绑定其当时实际执行的源码。
- 线性回归：**30 次 SGD 更新**，更新前损失从 **5.1570835** 降至 **0.0342863**。
- 两层 ReLU 分类器：**18 次 Adam 更新**，更新前交叉熵从 **0.6822657** 降至 **0.0017283**；该固定 8 样本任务更新后分类正确率 **100%**。

训练结果是对已实现执行链的功能验证。完整 PyTorch 兼容性、规模扩展和性能仍有独立验收任务。[验证记录](validation/README.md)将组件验证、最终集成验证和 CI 运行证据分别绑定到源码及可执行文件哈希。

## 当前边界与完整目标

当前设备为 CPU，dtype 为 binary32，张量 rank 不超过 4；同时存活的张量描述符最多 255 个，可以通过图回收在同一请求内复用。输入、输出和工作区仍各为 1 MiB；跨请求持久化尚未实现。卷积、矩阵乘法和池化各有每次 16,777,216 项的计算预算。整数索引载荷不等于完整整数张量 dtype 支持。空间、优化器和自动微分的具体限制见 [接口协议](docs/protocol.md)及 [能力清单](compatibility/capabilities.json)。

本轮在此前矩阵、VJP、图回收和 Adam 舍入修正基础上，实现了 cat/stack 与 scatter_add/index_add，并修复通用空张量的连续步长和晚出现零维度的形状接纳。cat 的 channels-last 是实际存储布局；index_add 的 alpha 舍入按已测固定 CPU 路径区分 rank 0/1 和更高 rank，不代表所有 ISA 都采用同一路径。具体行为见 [协议](docs/protocol.md)与 [数值说明](docs/numerics.md)。

当前权威源码使用 143／256 个函数名额、1,783／2,560 个基本块及 7,064／8,192 条指令。已运行的借用缓冲区模块桥接推进了独立原生编译与链接；通用 owned resource、资源返回值、分配所有权、跨请求状态、任意外部 ABI 和规模化生命周期仍需实现和验收。具体证据及完整目标见 [审计更新](docs/upstream-audit.md#tensor-composition-and-indexed-addition-audit)。

完整目标尚包括：完整 dtype/layout/promotion 语义、其余算子及重载、高阶和前向自动微分、CUDA/ROCm/MPS/XPU、分布式、图捕获与编译优化、数据加载、RNG、完整 checkpoint/serialization、Python/C++ 前端兼容和跨平台构建。[上游审计](docs/upstream-audit.md)列出这些目标的验收条件。

## 仓库结构

```text
src/heptatorch.hepta       唯一权威运行时源码；模块由其派生
tools/unhex.hepta          原生测试数据解码器
docs/                     构建、协议、数值与上游审计
compatibility/            固定上游清单与已验证能力
tests/fixtures/           独立校验后的命令流和预期结果
validation/               验证记录与来源绑定
third_party/pytorch/      上游许可证与通知
```

`build/` 为本地构建产物，不进入版本库。仓库中的许可证、测试、兼容性清单和失败边界均是最终交付的一部分。
