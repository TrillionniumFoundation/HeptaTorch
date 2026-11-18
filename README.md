# HeptaTorch

HIA 原生张量、自动微分和神经网络训练运行时。核心实现由 HIA 汇编器直接生成 Linux x86-64 静态可执行文件；数值运算、梯度传播和参数更新均在该程序内执行。

**当前默认实现为 700 个 HIA 核心函数、14 个独立编译单元的原生运行时。** 当前完整源码位于 [`tools/current_runtime/source/`](tools/current_runtime/source/)，直接由最终 ABI3 HIA 编译器和链接器冷构建。它包含持续演进的 Tensor／Storage／Version、自动微分、RNG、视图及求和实现；应用构建已不再依赖历次运行时的可执行文件或补丁生成链。

**整个 PyTorch 的全量重写尚未完成。** 完整目标固定为 PyTorch **v2.15.0**。仓库保留全部 **2,590 条原生算子声明**、**688 条导数声明**和 [17 项完整验收门禁](compatibility/full-acceptance.json)：10 项进行中、7 项开放，目前没有完整上游 schema 或整项门禁被标为已闭合。[当前审计](docs/current-audit.md)区分已执行的有限资格、尚未实现的系统功能和旧实现退出条件。

## 当前构建与测试

```sh
make HEPTA_CURRENT_COMPILER=/absolute/path/to/stack-module-compiler \
     HEPTA_CURRENT_LINKER=/absolute/path/to/stack-linker-session
make test HEPTA_CURRENT_COMPILER=/absolute/path/to/stack-module-compiler \
          HEPTA_CURRENT_LINKER=/absolute/path/to/stack-linker-session
```

默认产物为 `build/current-runtime/heptatorch`；`build.json` 绑定完整 HIA 源码、每个新对象、最终 ELF、实际栈预算及全部编译／链接进程的退出和 EOF。两个工具默认也可从 `HEPTA_STACK_BUILD/native/` 读取，其固定校验和与复现方式见 [当前构建说明](docs/current-runtime.md)。

`make test` 与 [Current runtime CI](.github/workflows/current-runtime.yml) 使用同一个执行入口，在同一冷构建镜像上实际运行 Bool27、原视图／reshape／标量76、原浮点 Sum32 和原整数 Sum24 四组回归。它们分别保留各自的用例、事件、数值、身份、生命周期、错误、完整 owner 和 RNG 分母；四组用例合计 159，不代表 159 个完整上游 schema。缺少任何必需报告、实际执行或完整 EOF 都不能通过总门禁。原先要求 Bool 目标 dtype 失败的两条历史控制保留原文，当前明确支持该路径并由新的七项 dtype 控制验收。

700 函数源码 SHA-256 为 `b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4`，冷构建 ELF 为 `c6afeedc14ba8cc408a62d6fbef0abb5b78dff0039b8099f3449d813e290a90e`。运行时不依赖 Python、PyTorch、NumPy、libm 或 libc；构建组织、独立观察器和验收工具仍使用 Python。它们不替代 HIA 执行张量计算。

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
| 训练生命周期 | 保留参数／状态的图回收、工作区复用、带代际信息的张量 ID、旧句柄拒绝；可选持久会话支持跨帧训练及完整帧失败回滚 |
| 执行接口 | 有边界检查的二进制命令协议、明确错误状态、原生 HIA 十六进制解码工具，以及逐帧读写的原生持久会话 |
| 原生模块 | 独立借用 ABI1 与 owned ABI2 配置；原生签名／效果／线性所有权检查、buffer/chain 资源传递和 HIA 静态链接；通用张量对象 ABI 仍未完成 |

本次实现使用软件浮点算术。运行时无 Python、PyTorch、NumPy、libm 或 libc 依赖。HIA 编译器的初始 bootstrap 使用其上游 Python 实现和 GNU ld，随后由 HIA 编译器自行重建；构建与运行依赖见 [构建说明](docs/build.md)。

## 历史 143 函数基线与工具链回归

以下构建保留早期协议和固定夹具，用于历史工具链及行为回归。它们通过显式 `legacy`／`test-legacy` 入口调用；当前默认构建和测试见上一节。

```sh
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler \
     HEPTA_MODULE_COMPILER=/absolute/path/to/hia-module-compiler \
     HEPTA_MODULE_LINKER=/absolute/path/to/hia-module-linker \
     legacy test-legacy
```

三个路径分别指向普通编译器、模块编译器和原生模块链接器，均有固定版本与校验和。历史 `build/heptatorch` 经独立模块编译和 HIA 静态链接生成。模块构建使用 Python 组织源码和传递封包；源程序检查、机器码生成、模块接纳、链接和数值执行均由 HIA 原生程序完成。`make test-legacy` 的回归循环使用本仓库的 HIA 解码器和原生程序，不需要 Python 或 `xxd` 处理测试数据。

保留单编译单元参考构建与同一套 461 对夹具：

```sh
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler \
     monolithic test-monolithic
```

参考产物为 `build/heptatorch-monolithic`。GitHub Actions 从固定的 Hepta 源码重新 bootstrap 工具链，执行两条原生构建路径，并分别运行同一份真实 PyTorch 对照门禁。

当前采用单独命名的扩容编译器配置，允许 256 个函数和 2,560 个总基本块。构建会从原始固定编译器出发，通过 HIA 编译与 HIA ELF 打包完成四代重建。生成配置及原生重建配方见 [构建说明](docs/build.md#reproducing-the-capacity-profile-from-the-original-compiler)。

历史模块路径将该基线的 143 个函数分为 63 个内核函数和 80 个运行时函数，实际执行带 52 个外部依赖的原生模块编译、接口接纳与静态链接。其函数体来自 `src/heptatorch.hepta`，生成的模块不是第二份手工维护实现。模块运行与边界证据单独记录在 [原生模块验证](validation/native-modules.json)，不据此宣称通用所有权、任意外部指针或所有平台 ABI 已完成。

运行已经验证的 1,000 步 SGD 训练命令流：

```sh
./build/unhex < tests/fixtures/session-review-reset-SGD-1000-real-iterations.input.hex > build/train.request
./build/heptatorch < build/train.request > build/train.response
./build/unhex < tests/fixtures/session-review-reset-SGD-1000-real-iterations.output.hex > build/train.expected
cmp build/train.response build/train.expected
```

命令流最小化平方损失，每一步执行前向、反向、参数更新和图回收。参数保持原 ID，临时图复用描述符与工作区。原有的两层 ReLU／18 步 Adam 训练夹具也保留在回归集中。

## 原生 binary64 标量基础

[HIA binary64 实现](tools/dtype-f64/README.md)提供加、减、乘、除、平方根及 binary32 双向转换，目标执行只使用 HIA 整数指令。固定的 CPython 55 项语料通过 1,968,156 个非 NaN 逐位比较及 66,910 个 NaN 分类检查。原版真实 PyTorch 逐位门禁仍为 **51／55 通过、4 项 sqrt 分组失败**；4,008 个差异全部保留。

[实际托管独立资格](validation/hosted-native-f64.json)重新执行完整 55 项语料，保留非 sqrt 的全部精确断言和原报告的逐流哈希，并对 236,892 个有限正 sqrt 输入使用两种精确整数方法检查 HIA 最近偶数舍入，同时验证固定 MKL 参考的严格小于 1 ULP 误差界。本轮两类错误计数均为零，11 项参考来源检查全部通过。结果明确记录 **PARTIAL／组件资格 PASS**，原版严格门禁的实际退出码仍为 1，完整 dtype 门禁和固定后端逐位兼容仍开放；Tensor dtype、提升规则和设备路径尚未接入。

## 原生 CPU 随机数基础

[HIA MT19937 与均匀变换](tools/rng-cpu/README.md)实现完整 5,056 字节 CPU RNG 状态、seed、raw32/raw64、binary32/binary64/half/bfloat16 的单位区间值、int64 半开区间及 discard。生成、状态规范化、消费计数和数值变换全部由原生 HIA 执行。原始 MT19937 与固定 PyTorch 源码及许可声明均保留。

本地独立参考资格已通过 **74 项／110 次原生调用／718,619 个精确值／434,816 字节完整状态比较**，另有 **35 项独立原生边界检查**通过。短输出缓冲区的实际失败及修复复测原样保留；原有 74 项请求和响应结果未变。[实际托管 PyTorch 对照](validation/hosted-native-rng.json)已通过 **76 项／114 次原生调用／719,337 个精确值／455,040 字节状态比较／11 项来源检查**。独立持久 Generator 服务见下一节；默认全局状态、Tensor 填充、normal/dropout 等其他分布、并发和加速设备仍待实现。

## 持久 CPU Generator 对象

[独立 HIA Generator 服务](tools/cpu_generators/README.md)把已固定的 17 个 RNG 函数体原样接入真实 heap owner，实现创建、默认构造、重设种子、系统随机种子、完整状态导入导出、克隆、交错消费和释放。最多同时持有 1,024 个独立 Generator；对象身份在释放后不复用，失败时保留完整原始 owner。`seed()` 的八字节熵来自原生 Linux `getrandom`，使用单独的四参数 ABI3 入口，原有三参数会话合同不变。

冻结原生资格通过 **11 组／1,715 帧／121,611 个精确值／471 次完整状态检查／72 次完整池回滚**，另有 **59 项实际入口边界检查**。测试包含实际分配失败、系统熵失败、释放后活进程内地址解映射、短内部 owner 和句柄／epoch 耗尽；[包清单](validation/cpu-generator-package.sha256)与[失败历史](validation/native-cpu-generator-failure-history.json)保留完整来源。[实际托管对象对照](validation/hosted-native-cpu-generators.json)已通过 **24 项／2,297 帧／26,466 个精确值／4,181,312 字节完整状态比较／11 项来源检查**。全部 46 步原生与 PyTorch CI 通过，完整原始报告按字节保存。

这是有界 CPU 对象服务。全局 Generator、Tensor 写入、normal/dropout、其他分布、共享对象并发、非规范历史状态和加速设备仍开放。内部状态诊断的边界是本服务创建并持有的 owner，不接纳调用方构造的任意本机指针或 owner 描述符。

## 原生资源所有权

独立的 ABI2 工具链支持 `buffer`／`chain` 在线性调用中消费、返回和释放，允许最多三个标量／owner 返回值，并由 HIA 原生程序执行接口接纳与链接。分配和回收直接调用 Linux `mmap`／`munmap`；零长度、溢出、真实分配失败、借用未结束时的移动／释放，以及错误效果声明都有明确检查。源码、接口和分配合同见 [owned 模块说明](tools/owned_modules/CONTRACT.md)。

该编译器自身有 8,229 条指令，通过两个合法编译单元完成原生重建，保留单元容量限制和全部既有 ownership checker。三代 ABI2 重建字节一致；正式源码重新构建与 **385 项**原生资格检查通过，并为当前完整 runtime 和 unhex 生成与已有构建逐字相同的映像。[资格证据](validation/native-owned-modules.json)包含 315 项接口攻击、资源生命周期和独立跨模块执行。[实际托管 CI](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43795584310)的 **32 个步骤全部通过**，包含全部 385 项 owned 资格、原有双 270 项普通 PyTorch 门禁和双 21 项交互门禁；[完整托管回执](validation/hosted-owned-modules.json)绑定实现提交 `030694be9373aefc9a6740c3ccc8d742098e5c51`。

这是原生所有权基础。原有张量内核和 borrowed 持久会话保留有界 arena；新的数值 Tensor 候选已接入可增长 owner，独立 Storage／Version 对象的完整连接继续进行。原生字节存储服务已分别验证强弱引用、独立版本与可增长内存池。此处不把复制 owner 的两个机器字解释为合法所有权复制，也不接纳任意外部指针。

## 可增长原生存储与 owned 交互入口

[字节存储服务](tools/storage_objects/README.md)使用真正的 HIA `buffer` owner，维护独立的逻辑 Tensor、Storage、Version、Weak、Saved 和 StrongHold 句柄。每次访问检查类型、存活状态和 generation；强引用释放后回收数据区，弱引用可保留已过期的对象身份。空闲区可拆分、合并并复用，已有句柄和相对偏移在扩容后保持有效。新的 Version 对象从 0 开始；用户写入递增共享的 Version。

服务已实际分配超过 1 MiB 的载荷，并把池迁移到 3,145,744 字节。地址空间限制下真实的 `mmap` 失败保持已提交 owner 和全部状态不变。每帧在独立 trial owner 上执行，成功才提交；失败释放 trial 并返回原 owner。这个服务目前处理字节值和字节区间，数值张量、shape/stride、自动求导和优化器仍需继续接入。

[owned 交互入口](tools/owned_sessions/README.md)由单独固定的原生 HIA linker 接纳 `step(readbuf,mutbuf,buffer)->(buffer,u32)` 和 `close(buffer)->()`。原生 startup 在 stdin 未关闭时返回响应，先接管返回的唯一 owner，再处理输出或退出。正常 EOF、断管及其他可恢复传输失败会调用真实 close。单独标识的资格程序在进程退出前观测到了实际解除映射；真实 EINTR 和短写也已执行验证。

正式源码的独立资格包含 **131 项 owner 入口／传输检查**、有限 trace 与交互模式各 **12 组／63 帧**、**5 项当前极限状态守卫**，以及独立参考模型的 **2,952 帧／11,177 次精确标量比较／93 次完整状态回滚比较**。[源码与原始证据](validation/storage-object-handoff.json)保留 456 个文件的逐字节清单；[两种模块排列的独立模型回执](validation/storage-independent-models.json)分别绑定两个不同映像。重复执行同一语料不增加上游 PyTorch 测试或 schema 覆盖数。实际构建与报告门禁见 [构建配方](docs/build.md#building-owned-interactive-storage)。 [实际托管 CI](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43797863695)已在提交 `8186da151687808085479392f4e4720b8ac71769` 上通过全部 **34 个步骤**，包括新的存储／所有权资格、原有双 270 项普通 PyTorch 门禁和双 21 项交互门禁。[完整托管证据](validation/hosted-owned-storage.json)保留实际逐项报告、源码和二进制身份。

## 可增长数值 Tensor 候选

[owned Tensor 候选](tools/owned_tensors/README.md)将现有 binary32 数值内核接入真正的 HIA owner：描述符按页分配，最多 65,535 个存活句柄；状态实际按需扩容，旧句柄和数组偏移在扩容后继续有效。完整请求在独立 owner 中执行，成功才提交；可恢复错误和真实内存分配失败保留原状态。构建对 143 个原函数逐个记录来源，其中 126 个函数体逐字保留，17 个函数只在明确列出的入口、分配、句柄和版本路径中修改，另新增 7 个辅助函数。

正式源码重新构建已通过 **11 组／161 帧／1,078,028 个 binary32 精确比较**，包括 64 MiB Tensor、128 MiB 实际 owner、全部 65,535 个描述符、跨扩容训练和 59,310,080 字节完整失败回滚比较。另有 6 项真实 typed-owner 状态检查；独立审查运行了 16 帧、612 个精确字段比较和 41,943,040 字节完整状态回滚比较，连接非均匀 matmul、转置、重复 gather、反向传播及 SGD。[本地构建回执](validation/owned-tensor-build-integration.json)记录实际执行结果与源码身份。[构建配方](docs/build.md#building-growable-numerical-tensors)可重现这些结果。

真实 PyTorch 门禁另设 **28 项**，保留原 21 项数值期望和容差，并新增分页、扩容、大张量及空张量反向传播检查。它加强了参数绝对版本计数比较；原生资格通过不代表该独立门禁已经通过。构造时版本从 0 开始已实现，优化器逐操作版本语义仍需由这一门禁检查。此候选仍为 CPU binary32、rank 0–4、受明确资源限额约束；独立 Storage／Version 对象与数值 Tensor 的连接、完整 stride／dtype／device 语义和整个 PyTorch 验收继续开放。

[首次严格门禁](validation/hosted-owned-tensor-initial-failure.json)已实际完成全部 28 项、6,233 帧和 2,590,207 个数值比较，其中两项因非零权重衰减 AdamW 的绝对版本计数而失败，原始结果完整保留。[原生修正版](tools/owned_tensor_versions/README.md)仅替换 `adam` 的版本收尾调用并新增一个 HIA 辅助函数，修正参数和二阶矩计数；原 28 项 PyTorch oracle 源码与断言保持逐字不变。修正版原生资格还检查了零／负零衰减、零学习率和空张量的版本行为。[实际修复 CI](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43801012881)已通过全部 37 个步骤，严格门禁 **28／28 PASS、6,233 帧、2,590,207 个值、11 项来源检查**，参数绝对计数差异为 0；另有 22 项真实上游生命周期观察。[完整回执](validation/hosted-owned-tensors.json)绑定实现、二进制及全部实际报告。预先提供的 momentum 状态与 torch.optim 新建状态之间的生命周期差异仍保留为开放问题。

## 原生栈预算与保护页

[ABI3 栈配置](tools/stack_budget/README.md)由 HIA 原生编译器生成栈帧和调用图元数据，原生链接器逐条解析受支持的机器指令，核对真实调用与栈帧后计算最长调用路径。生产程序分配带两侧保护页的 8 MiB 私有栈，正常退出前释放 owner、私有栈和 I/O 映射。源文件、编译器与对象都有明确身份；这项检查不声称验证任意外来机器码的全部寄存器、内存或效果行为。

当前 **151 函数**的数值 Tensor 程序经该配置重新编译和链接，所有模块源文件、函数体及链接后的数值机器码保持逐字一致。最长调用路径实际为 **61,488 字节**，另预留 4,096 字节。原生资格包含 **86 项**栈边界／真实深调用／保护页／退出清理／工作区复用／元数据攻击检查，以及另外编写的 **9 项**直接调用与栈帧伪造复验；随后执行同一份数值 Tensor 回归。[完整本地证据](validation/native-stack-budget.json)保留早期错误接纳及修复后的原始结果，[构建说明](docs/build.md#guarded-stack-and-native-call-path-admission)提供复现入口。[实际 CI](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43803875841) 在实现提交 `e1bc9ba5407a4c5b811b7376e9d1c44a8450f358` 上通过全部 **40 步**，新栈映像通过未改动的 **28 项 PyTorch 严格回归、6,233 帧、2,590,207 个数值比较和 11 项来源检查**，参数绝对版本差异为 0。[完整托管回执](validation/hosted-stack-budget.json)保留全部新资格报告与原始差分结果；momentum 状态的 283 项生命周期观察仍开放，全量 17 项子系统验收仍开放。

## 数值 Tensor 独立对象候选

[独立对象实现](tools/tensor_objects/README.md)把已有 binary32 数值内核接到独立的 Tensor、Storage、Version、Weak、Saved 和 Hold 记录。`.data` 共享存储而使用独立版本；clone 保留已验证的稠密转置及空张量步长；求导按实际保存的数据选择版本检查，矩阵乘法的折叠和广播复制使用真实的私有存储。公开 Tensor 句柄可撤销，梯度叶节点的内部持有独立管理。载荷可回收复用，池迁移和失败事务仍由原生 owner 执行。

237 个 HIA 核心函数与 3 个 owner 入口分成 5 个编译单元。ABI2 和带保护栈的 ABI3 已在新目录复现冻结映像，两者 ELF 的代码载荷逐字一致；143 个规范函数中 99 个函数体逐字未变，44 个变更均有完整源码映射和差分。每种映像实际执行 **731 帧原生检查、38,056,912 字节完整 owner 回滚比较**，并通过另外 **6 项／19 帧**协议桥接检查。冻结包及历史失败保存在 [来源清单](validation/tensor-object-package.json)、[独立审查](validation/tensor-object-independent-history.json)和 [PyTorch 测试来源](validation/tensor-object-oracle-package.json)。

新增 CI 为两种映像分别执行 **35 项**真实 PyTorch 对照，其中原有 28 项数值测试的输入、数值断言、容差及绝对参数版本断言保持不变，另外 7 项检查对象生命周期。[实际托管完整回执](validation/hosted-native-tensor-objects.json)已确认两种映像各 **35／35 通过、6,344 帧、2,590,274 个值、11 项来源检查**，完整运行的 **44 个 CI 步骤全部通过**。两份原始报告逐字无损保存在 `validation/hosted-tensor-object-oracles/`；参数绝对版本差异为零，283 项显式优化器状态生命周期观察继续保留。默认 backward 后自动释放 Saved 数据仍未实现，当前候选保留图直到显式释放或 reset；该差异被明确观察为 OPEN。CPU binary32、rank 0–4 和池／表容量等限制仍存在。完整目标仍是 2,590 个上游 schema，17 项全量子系统门禁仍开放。

## 默认反向传播与保存值释放

[独立默认反向组件](tools/backward_release/README.md)增加命令 57，支持成功反向后的 Saved 数据释放与显式 `retain_graph`。它会释放已消费的保存值和临时工作区，保留叶梯度及 AccumulateGrad 的存储义务；无需保存数据的图仍能重复反向。两套完整 HIA 实现之间的合法 owner 交接测试验证内部状态格式互拒、原字节不变、原实现继续执行及实际释放。

该组件的原生合同通过 10 组／144 帧、14 项独立控制／207 帧和双向交接 28 帧验证。对实际固定 PyTorch 的 36 项观测，35 项匹配；命令 57 保留整帧失败回滚，因此还有 **1 项明确的部分失败副作用差异**。这一结果始终记录为 `QUALIFIED_WITH_OPEN_DIFFERENCE`，不是完整反向兼容。[实际托管回执](validation/hosted-backward-release.json)保留来源提交、完整 47 步 CI 和逐字压缩的原始报告。

## 部分反向失败、首次梯度与 Tensor RNG 组合

[部分执行组件](tools/backward_partial/README.md)的命令 58 按依赖和就绪顺序执行节点，在反向运行时错误后保留已经完成的节点和帧内前缀，并跳过后缀；命令 59 读取真实梯度存在状态。[首次梯度组件](tools/gradient_materialization/README.md)区分首次赋值与已有梯度相加，并保留 gather 等归约的零初始化行为。原有 13 组逐位测试在旧版中为 4 通过／9 失败，新版用同一脚本全部通过；实际 16 组部分失败观测也全部匹配。正式资格检查重新执行旧失败并保留退出码 1。8 组梯度存在／优化器配方仍明确保留 8 项状态版本和 65 项惰性状态创建差异。

[联合原生实现](tools/tensor_rng_composition/README.md)把完整首次梯度组件与 Tensor 均匀填充、独立 Generator 表放进同一个 HIA 可执行文件和真实 owner。命令 58 失败时只保留前缀的随机消费；命令 57 和普通命令错误仍整帧回滚。7 个原生单元重建得到固定映像 `cdcaebb2785639585de7fd2a4e1f98d41a56e885f1657c3485ce5f187f89d726`，原生交叉事务、增长重放、两种反向入口的训练轨迹及四种前驱状态交接均通过。接近 binary32 最大值的区间接纳按真实 PyTorch 的 binary64 减法判断，88 项实际观测逐项匹配；旧 RNG03 的严格拒绝失败与独立测试输入适配记录均保留。

[实际托管联合资格](validation/hosted-native-tensor-rng-backward.json)已在实现提交 `44bd2a3bed4dab600bb31298c2b7f698df75d89d` 上通过全部 **51 个 CI 步骤**。主审独立的 **32 组 live PyTorch 联合测试全部通过**：2,132 帧、17,455 个精确 binary32 字、1,137,600 字节完整 Generator 状态、238 次绝对版本和 166 次梯度存在比较；同时检查实际 Storage 字节数和偏移。原样 **35 组 Tensor 回归全部通过**：6,344 帧、2,590,274 个数值比较、零参数绝对版本差异。两套实际参考的 11 项来源检查全部通过，原始报告逐字压缩保存在 `validation/hosted-tensor-rng-oracles/`；原有 283 项 momentum 生命周期观察和另一组 8／65 项状态观察分别保留，不相加计数。`validation/tensor-rng-backward-oracle-preparation.json` 保持 `NOT_RUN` 的准备身份，不能当作执行通过记录。完整 Python API、其他 dtype/device、其他分布、非叶原地随机操作的完整自动微分和优化器对象生命周期仍开放。

## 梯度清除与同一 owner 内的重新物化

[独立 nullable-gradient 组件](tools/gradient_none/README.md)增加命令 70：实际释放叶梯度的私有 Storage／Hold，把梯度存在状态改为 None；后续反向传播重新取得真实存储，并继续保留首次写入的精确位模式。命令 8 只清零已存在的梯度，None 时不分配、不写空地址。命令 58 失败时，已经完成的叶梯度保留，尚未物化的预留存储被回收。

[联合实现](tools/tensor_rng_nullable/README.md)把同样的行为接入现有 Tensor／Generator owner。全部 7 个原生单元的源码、对象文件和最终映像已经重新构建并逐字匹配。其新私有状态使用明确的格式屏障；真实跨模块 owner 交接检查了拒绝后的完整内容、原运行时恢复，以及进程仍存活时解除映射。

[本地正式资格](validation/gradient-none-root-native.json)对独立和联合两个映像分别通过 6 组／179 帧、131,101 个精确梯度字及 261 次提交节点不变量检查；还分别重放了全部 10 组实际 PyTorch 梯度清除观测和原有 13／16／8 组回归。新增联合检查覆盖 24 次训练更新、34 份完整随机状态和 2,500,896 字节失败回滚。10 组梯度清除重放中的 6 个 Adam 位差异仍按原有容差记录；11 项状态版本、9 项状态创建和 30 项缺失 step Tensor 的观察独立保留。旧的 8／65 项观察没有被替换或相加。

[实际托管资格](validation/hosted-native-gradient-none.json)已通过实现提交 `cb7c676a8455be70a653093eee1ac7e9f4ad2a68` 的全部 **54 个 CI 步骤**。新映像上的原样 **32 组 live PyTorch 联合测试全部通过**：2,132 帧、17,455 个精确字和 1,137,600 字节完整 Generator 状态；原样 **35 组 Tensor 回归全部通过**：6,344 帧、2,590,274 个值、零参数绝对版本差异。两套独立参考的 11 项来源检查全部通过，12 份完整原始报告逐字压缩保留。公开 `.grad` Tensor 身份／版本和外部别名存活期、真正的 Optimizer 对象及完整 Python 接口仍开放。[后续实现所需的真实参考](validation/hosted-object-distribution-observations.json)另外冻结了 24 组优化器对象、6 组叶求导标志及 183 组 Bernoulli／dropout 观测；这些参考观测不作为 HIA 已实现的数量。

## 跨请求持久会话

会话进程读取 `HTOSES01` 外层帧，执行其中的 `HTORCH01` 请求，在 stdin 仍打开时立即输出 `HTOSRP01` 响应帧，然后读取下一帧。每个完整请求在 shadow 工作区执行；成功才提交，失败返回保留原错误码的 32 字节、零记录响应，并保持已提交状态不变。外层格式、EOF／EINTR／断管处理及边界见 [会话合同](tools/session/CONTRACT.md)。

```sh
make session session-monolithic test-session test-session-monolithic \
  HEPTA_COMPILER=/absolute/path/to/hia-native-compiler \
  HEPTA_MODULE_COMPILER=/absolute/path/to/hia-module-compiler \
  HEPTA_FLAT_LINKER=/absolute/path/to/hia-flat-linker \
  HEPTA_SESSION_PACKER=/absolute/path/to/hia-session-packer
```

四个工具均有精确 SHA-256 门禁；后两个工具的源码生成与原生重建见 [完整配方](tools/session/README.md)。默认会话映像为 `build/heptatorch-session`，单体参考为 `build/heptatorch-session-monolithic`。构建只替换一个经过校验的 HIA entry，另外 142 个函数逐字保留；数值实现仍来自唯一权威源码。多模块增长测试已实际执行超过单元 8,192 指令限制、生成超过 1 MiB 的程序；这些构建能力见证不增加上游算子覆盖计数。

## 已观察到的验证结果

- 基础数值层：**2,314,020** 次原生运算与独立 NumPy binary32 参考逐位一致，算术 NaN 统一为规范 quiet NaN。
- exp/log/sigmoid/tanh：**3,424,359** 次测量；另用 90 位精度参考检查 **20,025** 个输入。测量误差及适用解释见 [数值说明](docs/numerics.md)。
- 集成：广播、视图、梯度、空间算子、模块、优化器、异常输入和容量边界，均有独立参考或明确拒绝结果。
- 本轮合并原生检查：cat/stack **81 项／1,673 个精确值**及 **16 项失败原子性**通过；scatter_add/index_add **100 项／1,553 个值**、补充 **9 项／1,652 个值**通过；独立组合检查 **64 项／3,738 个值**通过，包含共享梯度、矩阵乘法和多轮图回收。另有 **3 项／102 个值**检查把 cat、stack、scatter_add、index_add 和视图接在同一张图中。[本轮原生证据](validation/tensor-composition-native.json)绑定实际合并源码与二进制。
- 默认模块链接映像已通过 **461 对原生回归夹具**：原有 275 对，加 81 对拼接、100 对索引累加、3 对跨算子组合和 2 对含矩阵乘法的组合。原有请求输入保持不变；空形状 `[2,0]` 的旧步长 golden 仅在响应 byte 280 从 0 改为 1，对应 `[0,1]` 修正为 `[1,1]`。该修正已有固定源码和真实 PyTorch 观察支持。单编译单元参考路径使用同一份 461 对集合；两份二进制的独立身份和实际回放结果各自记录。
- 真实 PyTorch 边界观察：**22 项空张量、12 项索引边界、6 项 alpha 舍入和 8 项布局选择**已执行并保留各自 CI 来源。合并程序重放布局观察的 **924 个值及全部步长**、索引边界的 **64 个值及拒绝阶段**、alpha 的 **102 个值**，均一致。[原始观察](validation/hosted-boundary-observations.json)与最终二进制回放分别保存，观察所用的早期源码身份未改写。
- 拼接与索引扩展批次的 PyTorch 门禁在两份实际映像上分别 **PASS：每份 268 项／12,932 个数值／11 项来源检查，零失败**。保持原 159 项的输入和容差，新增 35 项拼接及 74 项索引累加；[实际运行](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43788929713)验证提交 `724cbf2eb19d2ef8d400615d7196f5397cb402f6`，同时通过两份 461 对原生夹具、模块边界和工具链重建。[完整新回执](validation/hosted-tensor-composition.json)分别绑定两份可执行文件，保留逐项结果。
- 当前普通请求 PyTorch 门禁进一步扩展到每份 **270 项／13,076 个数值／11 项来源检查**。新增两个 12 步 SGD／AdamW 轨迹将拼接、转置、共享梯度、重复索引累加和图回收接在同一训练流程中，新增 144 个值逐位一致。[实际托管回执](validation/hosted-cross-feature-training.json)绑定提交 `91665f48c95f3bd8bb9bf4ec725c4f30ca66b511`，原 268 项函数和容差保持不变。
- 原生持久会话：模块化与单体参考各 **477 项独立检查**通过，含全部 461 对既有夹具互操作、五类完整 1 MiB 状态回滚、64 次分帧 SGD 更新及真实断管退出；另有 **22 项打包边界、18 项 flat-linker 检查和 3 项模块增长检查**。两种映像均已在 stdin 保持打开时实际返回多帧响应。[完整原生证据](validation/persistent-session-native.json)保留首次 SIGPIPE 失败、后续修正与单独标识的真实 EINTR 测试 adapter。两份映像的新增真实 PyTorch 跨帧门禁均已 **PASS：各 21 项／4,681 帧／491,847 个值／11 项来源检查**；[完整托管回执](validation/hosted-persistent-session.json)绑定实际提交 `c627e1f7a5ab44a882a047a6c63ee4c8ca8a3c3a`。其别名测试严格比较共享递增及 clone 独立性；packet 与 `Tensor._version` 的绝对计数差异仍开放，首次 19／21 结果和两项合同断言修正保留在[原始失败记录](validation/hosted-session-oracle-contract-failure.json)。
- 历史图生命周期批次：**275 对原生回归夹具**通过，原有 221 份请求输入未变，其中 5 份 Adam 预期响应已随舍入修正重新通过独立参考验证。图生命周期的 **26 项／40,123 个值**独立检查通过，包括单次请求中的 **1,000 步 SGD、1,000 步 Adam**，以及在 1 MiB 工作区内反复回收共 **9,600,000 字节**的临时张量数据和梯度缓冲。另有 **10 项**失败原子性检查，每项比较除错误状态外的全部 **1,048,568 字节**，均未发生修改。[历史原生验证](validation/session-native.json)继续保留该批次的具体案例、误差和源码／二进制身份。
- 上一批算子扩展的矩阵与 Linear **63 项／4,593 个值**检查、**2,514 项**独立矩阵形状／梯度／拒绝检查，以及 **11 项／529 个值**跨算子集成检查均通过；对应历史源码身份保留在[算子扩展验证](validation/native-extension.json)。
- 历史托管 CI：上一批次的 **159 项／8,471 个真实 PyTorch 数值对照**全部通过，包含无掩码／无 dropout 注意力、**1,024 步 SGD、512 步 Adam 和 512 步 AdamW**。长 SGD 对照逐位一致；长 Adam／AdamW 最大绝对误差分别为 **1.49×10⁻⁸／5.96×10⁻⁸**。[成功运行](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43743468220)同时验证了编译器自重建、275 对原生夹具和 11 项 oracle 来源检查。[该批次完整回执](validation/hosted-compiler-profile.json)保留逐项结果；[首次失败回执](validation/hosted-session-failure.json)也保留，用于说明 Adam 舍入问题及修正过程。
- 编译容量：扩大配置的四代原生镜像和可执行文件逐字节一致，**16 项容量门禁**、256 个显式标量导出的实际调用、高编号调用图及脏工作区检查通过。原始和扩容编译器对当时 121 个函数的 HeptaTorch 及 unhex 生成完全相同的二进制；当前 143 个函数的版本使用扩容配置。[工具链验证](validation/compiler-profile.json)保存具体配置、来源、内存布局及独立验证记录；[扩容构建托管回执](validation/hosted-compiler-profile.json)继续绑定其当时实际执行的源码。
- 线性回归：**30 次 SGD 更新**，更新前损失从 **5.1570835** 降至 **0.0342863**。
- 两层 ReLU 分类器：**18 次 Adam 更新**，更新前交叉熵从 **0.6822657** 降至 **0.0017283**；该固定 8 样本任务更新后分类正确率 **100%**。

训练结果是对已实现执行链的功能验证。完整 PyTorch 兼容性、规模扩展和性能仍有独立验收任务。[验证记录](validation/README.md)将组件验证、最终集成验证和 CI 运行证据分别绑定到源码及可执行文件哈希。

## 当前边界与完整目标

默认静态运行时的设备为 CPU，dtype 为 binary32，张量 rank 不超过 4；该配置同时存活的张量描述符最多 255 个，可以通过图回收在同一请求内复用。后续 owned 配置使用真实可增长 Storage 和分页面描述符，具有各自固定的源码、协议与验收记录；默认静态配置的容量不能代替它们的边界。普通请求的输入、输出和工作区各为 1 MiB。可选持久会话在同一进程内保存状态，每帧输入／输出仍各为 1 MiB，使用 durable／shadow 两份 1 MiB 工作区提供完整帧事务；进程重启后的持久化尚未实现。卷积、矩阵乘法和池化各有每次 16,777,216 项的计算预算。整数索引载荷不等于完整整数张量 dtype 支持。空间、优化器和自动微分的具体限制见 [接口协议](docs/protocol.md)及 [能力清单](compatibility/capabilities.json)。

本轮在此前矩阵、VJP、图回收和 Adam 舍入修正基础上，实现了 cat/stack 与 scatter_add/index_add，并修复通用空张量的连续步长和晚出现零维度的形状接纳。cat 的 channels-last 是实际存储布局；index_add 的 alpha 舍入按已测固定 CPU 路径区分 rank 0/1 和更高 rank，不代表所有 ISA 都采用同一路径。具体行为见 [协议](docs/protocol.md)与 [数值说明](docs/numerics.md)。

默认静态源码使用 143／256 个函数名额、1,783／2,560 个基本块及 7,064／8,192 条指令。各个 owned 后继的完整 HIA 源码由其独立、哈希固定的输入生成，下面各节分别说明其原生模块和函数计数。已运行的借用缓冲区模块桥接推进了独立原生编译与链接；独立 owned ABI2 已实现资源传递、返回及原生 syscall 分配；新的 owned Tensor 候选已运行可增长的数值存储；独立对象生命周期和任意外部 ABI 仍需完整实现和验收。原 borrowed 会话保留有界 arena；新的 owned 会话显式持有和移交 heap owner。具体证据及完整目标见 [审计更新](docs/upstream-audit.md#tensor-composition-and-indexed-addition-audit)。

完整目标尚包括：完整 dtype/layout/promotion 语义、其余算子及重载、高阶和前向自动微分、CUDA/ROCm/MPS/XPU、分布式、图捕获与编译优化、数据加载、RNG、完整 checkpoint/serialization、Python/C++ 前端兼容和跨平台构建。[上游审计](docs/upstream-audit.md)列出这些目标的验收条件。

## 仓库结构

```text
src/heptatorch.hepta       默认静态运行时源码；默认模块由其派生
tools/*/                  各个 owned 后继的固定 HIA 源码输入与构建入口
tools/unhex.hepta          原生测试数据解码器
docs/                     构建、协议、数值与上游审计
compatibility/            固定上游清单与已验证能力
tests/fixtures/           独立校验后的命令流和预期结果
validation/               验证记录与来源绑定
third_party/pytorch/      上游许可证与通知
```

`build/` 为本地构建产物，不进入版本库。仓库中的许可证、测试、兼容性清单和失败边界均是最终交付的一部分。

## 真实 owned SGD Optimizer 与前向捕获的梯度边

[HIA 对象实现](tools/owned_optimizers/README.md)增加真实 Optimizer、参数组、独立参数所有权与惰性 momentum Tensor。关闭公开参数句柄后，Optimizer 和计算图按各自的真实引用义务继续持有参数；重新发布句柄使用新 generation，Tensor 身份、Storage、Version 与梯度保持不变。前向时捕获的梯度边与叶节点当前 `requires_grad` 分开管理，后改标志不会给旧图增加边。

[根集成资格](validation/owned-optimizer-root-native.json)独立冷重建 7 个完整 HIA 模块，复现相同源码、对象和 ELF，并执行 19 份资格报告。固定 24 组 Optimizer 观测中，18 组在当前公开 SGD 范围内通过、1 组保留非叶参数场景仅部分覆盖、5 组仍开放；6 组叶标志对照和 3 组累积器生命周期对照全部通过。另验证公开句柄关闭、descriptor 跨页、表扩容、owner 迁移、真实内存失败与最终 Weak 过期。两种真实旧／新运行时的 owner 交接共 4 向、91 帧通过。

历史 candidate06 的非求导 transpose 叶身份错误和前向后开启梯度错误在根集成中重新复现，保留预期退出码 1；新候选在相同输入与断言下通过。原来的 13／16／8／10 组回归和两套 RNG 组合测试保持数值、容差及错误断言不变。旧显式状态 API 的版本、惰性创建和缺失 step Tensor 差异继续单独保留。[实际托管资格](validation/hosted-native-owned-optimizers.json)已通过实现提交 `0cdd8641e9ccefe94b86da687f053f7579f3d806` 的两条 Native CI，各 **57 个步骤全部通过**。相同 ELF 上的原样 **32 组 live 联合测试**通过：2,132 帧、17,455 个精确字、1,137,600 字节完整 Generator 状态；原样 **35 组 Tensor 回归**通过：6,344 帧、2,590,274 个值、零参数绝对版本差异。旧显式状态的 283 个版本差异仍明确保留。每套实际参考的 11 项来源检查全部通过，15 份完整原始报告逐字压缩保存。

公开 `.grad` 对象另有 [10 组实际参考观测](validation/hosted-public-gradient-object-observations.json)，覆盖身份、布局、赋值、别名、Version 和生命周期；参考观测本身不代表原生实现通过。真正的公开梯度对象、完整 state_dict、任意优化器失败副作用和其他后端继续实现；owned Adam／AdamW 的新独立配置见下一节。整个 PyTorch 的 2,590 条 schema 与 17 项门禁保持未闭合。

## 真实 owned Adam／AdamW 状态 Tensor

[新的原生对象配置](docs/owned-adam.md)在冻结 SGD08 数值核心之上加入按参数惰性创建的真实 `step`、`exp_avg` 和 `exp_avg_sq` Tensor。三种状态分别拥有 Tensor／Storage／Version 身份，None 梯度跳过创建和更新，重复参数按原顺序共享同一套状态。公开状态别名与 Saved 引用观察真实修改；释放 Optimizer 后仍有独立引用的状态继续存活。命令 95 在原生程序中按逻辑索引执行 no-grad 标量加法，并统一递增一次共享 Version。

[根集成资格](validation/owned-adam-root-native.json)已冷重建全部 **8 个 HIA 模块**，逐字复现源码 `25343864…` 与 ELF `fa923569…`，并实际通过 **19 份原生资格报告**。独立重放固定真实参考中的 Adam／AdamW 四组案例，共 **908 帧／3,469 次精确比较**；210 个参数和 moment 数值位置逐位一致，原容差未变。与原 SGD 报告按不重叠案例合并后，[24 组参考的当前覆盖](validation/adam-object-review/report.json)为 **22 组在已暴露范围内通过、1 组部分覆盖、1 组公开梯度对象仍开放**；旧 18／1／5 报告原样保留，没有重分类。

原生资格还验证 900 个实际状态 Tensor 的扩容与私有引用、实际分配失败的整帧回滚、两种状态格式的合法 owner 交接、旧 6 组叶标志和 3 组累积器生命周期，以及原样 13／16／8／10 组反向回归和两套 RNG 组合测试。`make owned-adam test-owned-adam` 可重建并执行这些门禁。[构建说明](docs/build.md#owned-adam-and-adamw-state-tensors)列出依赖、产物和命令。最终 Adam ELF 上新增的原样 **32 组 live PyTorch** 与 **35 组冻结 Tensor** CI 已配置，结果尚待实际执行；此前 SGD ELF 的通过记录不作为新映像的通过证据。

此配置明确保留限制：当前 bias correction 只接纳实际 float32 自增后为正整数且小于 2³² 的 step；±0 可恢复为 1，2²⁴ 处的 float32 停滞仍递增 Version，任意实数指数仍未完成。AMSGrad、其他 dtype／后端、公开 `.grad` 对象组合、完整 state_dict 与优化器任意异常的部分副作用仍开放。完整 PyTorch 的 2,590 条 schema 和 17 项完成门禁不因本批通过而关闭。
## 原生 CPU Bernoulli 与 dropout

[分布组件](docs/cpu-distributions.md)在同一真实 Tensor／Generator owner 内执行 float32 Bernoulli、普通 CPU dropout、Saved noise 和相应反向传播。双精度标量概率保留原始 IEEE 位模式；Tensor 概率、实际 stride／alias 写入、完整 5,056 字节 Generator 状态和共享 Version 均由原生 HIA 处理。原子 profile03 与显式部分提交后继分别冻结，原生编译器、模块接口和全调用图栈门禁未放宽。

固定实际 PyTorch 的 **183 组观测**保持原样：旧命令 80–85 在两个映像上均保留 **161 PASS、4 个事务差异、18 OPEN**。新命令 86／87 明确表达晚错误的前缀提交和后缀停止，使同一参考的 **165／183** 组严格通过；闭合的零起始索引仅为 **101、103、105、179**。错误前的 Tensor 写入和 RNG 消费可以提交，失败的分布操作本身不额外增加 Version；普通准入错误仍完整回滚。旧接口没有被悄悄改义。

`make test-cpu-distributions` 先实际执行冻结的 5 套原子资格和 6 套后继资格，再核对两份 37／28 文件清单、全部 11 个子报告和完整原始归档。CI 另外要求最终后继映像运行原样 32 组 live PyTorch 联合对照和 35 组 Tensor 回归，不修改其数值期望、容差或 11 项来源检查。构建方式与精确源码／映像身份见[构建说明](docs/build.md#native-cpu-bernoulli-and-dropout)。

剩余 **18 OPEN** 是 8 组 float64 概率 Tensor、2 组 int64 概率 Tensor 和 8 组独立 `native_dropout` 布尔 mask 接口观测；双精度标量不等于双精度 Tensor。普通梯度模式下非叶原地操作的完整自动微分、与 kind-8 Optimizer／公开梯度对象的共同运行时及其他分布仍开放。组件状态保持 **PARTIAL**，整个 PyTorch 验收保持 **HOLD**；全部 RNG、dtype 和全量重写完成标志仍为 false，2,590 项分母不变。

## 同一 owner 内的 Adam 与 CPU 分布

[新的共同原生配置](tools/adam_distributions/README.md)将前两节冻结的 Adam02 和分布 partial01 接入同一个实际 owner。参数、真实 `step`／moment Tensor、Generator、梯度、Saved 和计算图共享同一套受检查的对象存储；原有两个独立配置及其历史结果保持不变。新源码包含 **378 个函数、4,971 个块、18,539 条指令**，以 **8 个完整 HIA 单元**通过原生编译和全调用图栈门禁，绑定源码 `5d3a3c04…` 与 ELF `3fc59ab7…`。八 kind 的迁移、Optimizer 私有引用和前向梯度边捕获继续使用 Adam 实现。

[独立交叉资格](validation/adam-distributions-review/README.md)已实际通过 **6 组／462 帧**：Bernoulli 图与 Optimizer／AccumulateGrad 分别持有参数、dropout noise 和真实 state 的 Saved／别名寿命、Adam 更新后命令 86／87 的晚错误提交，以及 Adam／AdamW 各四轮 dropout、反向、更新、清除与图回收。检查包含 249 个逐位 binary32 值、22 份完整 Generator 状态和 4,892,928 字节整 owner 回滚。首轮测试期望丢失负零符号的 FAIL、对应真实 PyTorch 观察、修正和全部原始帧均保留；原生源码和映像没有为该测试修正而变化。

`make adam-distributions test-adam-distributions` 重建该共同映像并执行根资格入口；[构建说明](docs/build.md#combined-owned-adam-and-cpu-distributions)列出两个前驱、严格复现及报告路径。该配置已在实现提交 `3e75b3aa…` 的两条 Native CI 中各通过全部 **68 步**，并合入 `e4cf7b6d…`。同一 ELF 的原样 **32 组 live PyTorch** 和 **35 组 Tensor** 均通过，所有 11 项来源检查通过，零参数绝对 Version 差异；原有 283 项显式状态生命周期差异保留。[完整托管证据](validation/hosted-adam-distributions-3e75b3a/)保存原始 ZIP、50 份报告、24 份 envelope、主审及独立逐字审查。后续新映像继续单独运行这些门禁。

同一份固定 183 组参考在旧原子接口上仍保留 161 PASS／4 个事务差异／18 OPEN，在显式错误接口上为 165 PASS／18 OPEN；24 组 Optimizer 参考仍是 22 组在已暴露范围内通过、1 组部分覆盖、1 组公开梯度对象开放。旧原子负控曾将 86／87 视为未知命令，其原始失败单独保留，新增准入检查不冒称旧负控原样通过。共同配置没有增加新的上游 API 覆盖，也未接入公开 `.grad` 对象后继；完整 RNG、dtype、优化器模式及整个 PyTorch 的 2,590 条 schema／17 项门禁继续保持未闭合。


## 公开梯度 Tensor 与共同运行时

[公开梯度对象实现](tools/gradient_objects/README.md)为 `.grad` 提供真实 Tensor、Storage、Version、引用和别名寿命。主集成从冻结 SGD 前驱重新编译 355 函数配置，实际完成 **16 项冷资格**；仅用 HIA overlay 重建和重走完整作者生成链得到逐字相同的源码与 ELF。固定 public10 为 9 组已覆盖／1 组部分覆盖，incoming25 为 22 组已覆盖／3 组未表示，alias12 全部通过。历史失败、实际帧和明确的未表示边界保存在 [根集成证据](validation/gradient-objects-root/)。

[共同配置 candidate05](tools/gradient_adam_distributions/README.md)把上述真实梯度对象与 Adam／AdamW、CPU Bernoulli／dropout 接入同一 owner。源码 `1fed8ea7…`、ELF `67734858…` 包含 **428 个 core 函数、5,738 个块、21,348 条指令和 9 个原生模块**。主集成再次冷构建后完成全部 **17 项生产资格**。新物理 incoming Storage 与逻辑梯度工作区分别管理，解决转置零贡献先到达时的梯度混排；pointwise 输出使用实际布局地址；scalar-add 保留固定 CPU 参考的 NaN 符号和 payload。

独立资格分别在同一最终映像上重新执行原生重编、owner 入口、四向 typed handoff、48 组 scalar NaN、2 组实际混合布局、6 组跨功能生命周期、2 组数学反例以及 8 组极大 size-one stride 对照。candidate04 在这 8 组中真实失败 4 组；candidate05 仅修正 signed byte-stride 比较，同一参考与比较器得到 8／8 通过。原失败没有改写为通过。这些重叠套件分别计数，不汇总为上游 API 覆盖率。

原 Optimizer24 的 case20 已通过实际 `.grad` 外部引用、`zero_grad(None/False)`、原地标量更新和弱引用到期验证；与原 SGD 及 Adam 的不重复索引合并后为 **23 组已覆盖／1 组部分覆盖**。该 428 配置仍未实现非叶 `retain_grad`，后继 446 配置的实际实现见下一节。hooks／高阶导数、18 组分布 dtype／bool-mask 情形及其余全量子系统继续开放。355 和 428 两份 ELF 已在提交 `af33d106…` 的两条 Native CI 中各通过全部 76 个实际步骤，两条快速 CI 各通过 13 步；两个映像分别真实运行并通过原样 live32 与全部 frozen35。[完整托管原件及独立审查](validation/hosted-gradient-adam-distributions-af33/)保存 58 份主报告、原快照和四组对照；59,251 项独立离线检查均通过。原 `30ea38d…` 的 32 组 close 失败、首次复制负控失败和未执行的旧后置门均保留原状态。PR #28 已合入 `7e7e37be…`，源码树与验证时相同。

[集成构建说明](docs/gradient-objects-and-composition.md)给出复现命令、输入绑定、失效历史和剩余边界。整个项目仍为 **HOLD**；2,590 条原始 schema 和 17 项完整验收门禁不变。

## 保留非叶梯度的原生后继

[446 函数配置](tools/retained_gradients/README.md)在同一 HIA owner 中实现
`retain_grad` 的实际 Tensor 身份、首次连续梯度副本、后续 out-of-place 累加和
外部旧梯度别名。每次后续累加创建新的 Tensor／Storage／Version，旧外部引用继续
指向旧对象；叶节点与不要求梯度的节点采用独立准入规则。显式部分反向命令 58
可以先提交已经执行的 retain hook，再报告自己的 Saved 版本／释放错误；原子命令
7／29／57 继续保留整帧回滚。

源码 `666230f0…` 与 ELF `7568c0a4…` 含 446 个核心函数、5,922 个块、
21,904 条指令、9 个原生模块，描述符仍为 576 字节。根集成已实际重新冷编译并通过
17 项继承资格。独立的 12 个保留梯度变体执行 3,849 帧、38,529 个断言，
1,352 个 binary32 位置逐位一致；188 个 Adam 参数／moment 位置无位差，
9,087,312 字节失败回滚逐字一致。原 Optimizer24 的第 14 组另以增量重放验证
其两个保留非叶子观察；原报告的 23／1 汇总仍不改写。

另有 8 个真实资源／内存失败用例、2 向真正的 19 模块 owner 交接和 5 组继承扩展
回归通过。扩展回归执行 947 帧，包含原 NaN48、mixed2、cross6、math2 和 extreme8
的期望；交叉测试只对私有格式标签作可逆适配。原资源归档曾截断，原始缺口保留，
本次完整的交接帧来自新实际执行。[根证据](validation/retained-gradients-root/)和
[集成证据](validation/retained-integration-review/)分别保留冷执行、明确复用、失败现场、
全部子报告和原始封包。

`make retained-gradients test-retained-gradients` 可以构建和执行固定资格。
[详细复现说明](docs/retained-gradients.md)给出工具路径、原始档案与尚待执行的门。
第三个 private-node profile 仅增加精确的 446 源码／ELF／标签；所有分配、引用、
Version、整 owner 回滚与原 18 个负控保持原规则。446 映像上的 live32 与 frozen35
现已在实现提交 `cef47b8f…` 上实际通过，两条 Native job 各完成全部 80 步，
355／428／446 三条快速资格各完成 13 步。独立离线审阅通过 106,570 项检查；
主控再次执行得到逐字相同的完整报告。PR #31 已合入 `e0ea769c…`，
[完整原件与审阅记录](validation/hosted-retained-gradients-cef47/)保留两个原始
reader 失败、局部修复及最终成功，6 份原始 ZIP 共 94,716,715 字节另有完整来源回执。

hooks、前向／高阶导数、任意 dtype／设备、完整优化器状态与整个 PyTorch 兼容性
继续按原 2,590 条 schema 和 17 项门禁验收。

## 已合并的原生 int64 算术基础

[独立 HIA int64 内核](tools/native-i64/README.md)实现六种有界整数算术与位运算，
两份实际 PyTorch 专用作业均通过 **33 组／351,174 个精确值／11 项来源检查**。
接入当时主线后，两份 Native 作业各通过 80 步；两份原始产物分别通过
106,633 项独立检查，主审另外重放得到逐字一致的报告。
[完整准入记录](validation/hosted-native-int64-2e66/)保留所有原始失败与来源。
PR #29 已合入 `7e157f7b…`，合并树与实测树一致。整数 Tensor、推广规则、
布局及其完整自动微分接口仍须各自实现和验收。

## 实际 bool Storage、NativeDropout 与捕获的反向历史

[523 函数原生后继](docs/native-dropout-history.md)在真实 HIA owner 中加入
一字节 bool Storage、公开 mask Tensor、保留的不可变 AD 边与 AccumulateGrad
目标、真实 `detach_` 和 `logical_not_`。11 个原生模块重新编译后逐字复现固定
源码与 ELF；13 个生产资格步骤通过。独立的 9 组实际历史配方展开为 13 个
原生进程，全部通过；26 组 detach 原地操作及全部后续操作、双向 typed owner
交接和短输出入口检查另有各自完整证据。

空 NativeDropout 改变当前 Tensor 历史后，合法已有优化器的审计与清零，以及
SGD 首次 momentum 在 maximize／weight-decay 路径上的布局，均由实际固定
PyTorch 观察约束。此前真实失败和修复后的独立结果分别保留。有限的 640 字节
owner 审计通过 4 个有效控制和 24 个复制快照负控；这组有限控制与完整
frozen35 分配账本分别验收。新映像已在固定 ROG PyTorch 环境实际通过原样
live32，以及完整 640 字节独立 owner 观察器下的全部 frozen35：
**35 组／6,344 帧／2,590,274 个值／11 项来源检查**。原 EOF、关闭后的
Version 检查和 283 项旧显式状态差异完整保留；参数 Version 差异为 0。
[完整原始记录](validation/dtype-frozen35-accounting/evidence/qualification-summary.json)
保留两次旧检查器失败及真实释放控制，原生源码和 ELF 未因检查器修正而改变。
新集成入口已实际冷编译并通过全部 6 个独立资格门；新的 GitHub 源码组合仍待
其自身 CI 与原始产物准入。

计数按各自参考与运行来源保留，原 183 组参考中的 18 OPEN 不改写。其他 dtype、
promotion、完整 Python 接口和所有 17 项系统门禁继续开放。

## 原生 int64 Tensor 后继配置

[独立的 int64 Tensor 配置](docs/integer-tensors.md)新增实际八字节 Storage、类型化算术与原地操作、最近偶数舍入、类型提升临时对象，以及真正持有 ultimate-base Tensor 的 int64 视图。新的 544 函数／十一模块映像已通过实际冷构建、固定参考重放和独立资源、入口、跨版本 owner 交接检查。原地操作首轮三个失败和后续修复均保留。

完整报告、原始字节和四门统一冷构建回执见[整合证据](validation/integer-tensors/integration-final/custody.json)。各组检查存在重叠，不能累加为唯一算子或 schema 覆盖数。该映像的 live32、完整35和 PR #33 两份托管 Native 作业已实际通过，两份作业各完成 96 步；PR／push 原始产物分别通过 217,481／217,482 项完整审查。合并提交 `26f3b2d0…` 的代码树与验收树完全一致，[原件与准入记录](validation/hosted-integer-tensors-343/)保留历史失败及证据范围。公开 Tensor.to、其他 dtype／设备和全部 17 项全项目门槛继续开放。全量完成标记仍为 false。

## 原生 F32／int64 自动求导与 Saved Tensor 生命周期

[556 函数／十一模块的原生后继](docs/mixed-integer-autograd.md)加入混合
F32／int64 加、减、乘和真除法的有限自动求导，以及原始 Saved Tensor 访问。
计算图保存原输入的真实 Tensor、Storage 和 Version；普通 F32 乘除法同样使用
实际 Saved 引用，重复输入按各自槽位计数和释放。

新的五门冷构建已重新编译全部模块，并通过固定 36 组混合 AD、旧 int64／523
回归和独立 14 组公开 Saved 生命周期检查。原 `zero_grad(False)` 遗漏释放
Saved Hold、以及整数 view 最后一个公开 Saved 引用释放后 Storage 仍存活的
两次真实失败均已修复，原失败记录完整保留。独立资源、入口容量、Saved getter
错误与双向 owner 交接另有原生执行证据，见[完整手包](validation/mixed-integer-autograd-evidence/)。

[七个外层命令的统一冷构建](validation/mixed-integer-autograd-root/report.json)已实际全部通过，
完整 2,225 个原始证据成员另经独立字节复核。新映像的 live32 和完整 35 组参考也已实际通过，
全部 6,344 次状态转移再经独立重算；原来源、EOF、关闭后 Version 检查和历史差异均保留。
[完整参考证据](validation/mixed-frozen35-accounting/remote10/)与该源提交自己的托管产物
继续分别验收。计数按原参考范围保留，重叠测试不累加为 schema 覆盖率。
整个 PyTorch 重写状态继续为 **HOLD**，完整 schema 闭合数仍为 **0／2,590**。

## 可恢复的 Tensor.to 构建与可微视图历史

[当前 R3 HIA 视图后继与完整证据](docs/tensor-to-differentiable-views.md)已在精确
R2 父版本之上实际编译 **627 个函数／13 个模块**。它保留此前视图历史刷新的
梯度 hook 修复，并真实恢复重叠、带空洞输入的梯度 Storage、偏移和 strides，
使原有 clone 合同产生正确的公开梯度 Version；局部覆盖归一化保留实际 NaN
符号和 payload，不修改全局浮点除法。

同一 R3 映像已分别通过[原 14 项](validation/differentiable-view-r3-native14/report.json)
（15,135 帧／96,461 项严格断言）和[新增 4 项](validation/differentiable-view-r3-input4/report.json)
（12,353 帧／77,432 项严格断言）。完整 Tensor／Storage／Version 关系、
Saved 与重试行为、完整 RNG、失败事务和最终 Weak 生命周期按各自实际参考比较，
所有会话均正常 EOF。R2 在新增四项上的 **1 通过／3 失败**、更早的
13 通过／1 失败以及宿主诊断失败均保留原始记录。

[另一次 72 项梯度重放](validation/differentiable-view-r3-multiaxis72/report.json)
通过 1,224 帧、21,312 个梯度原始字，逐会话核实输入为参考的 0 至 255 序列并
正常 EOF。这一门不比较完整 owner／RNG，也不把记录但未比较的梯度 Version
算作资格；旧 R1 使用全一输入的范围差异明确保留。三组检查不累加为唯一 API 数。

[独立资源证据](validation/differentiable-view-r3-preserve-resources/qualification.json)
进一步验证了三类有限行为：梯度缓存尾部释放与复用、经过两个视图节点后的
Saved 错误与原子回滚，以及临时地址空间限制下的 OOM 回滚和恢复。首轮仍记为
**2 通过／1 准备失败**；修正测试填充维度后仅重跑 OOM，一项通过。
成功控制合计 113 帧／243 项检查；包含准备失败的完整历史为
126 帧／260 项检查／4 个正常 EOF。原 R3 配置的历史 NOT_RUN 字段保持原样，
完整资源资格与当前源码提交的托管 CI 仍需独立验收。

父版本完整 35 与原始回归继续保留各自来源，新 R3 的更广继承、完整资源资格和
该源码组合的托管 CI 必须分别取得实际证据。任意视图几何与图分支、全部
dtype／设备／接口和高阶自动微分仍未闭合；17 项系统门禁保持开放，
完整 schema 闭合数继续为 **0／2,590**，整体状态为 **HOLD**。
