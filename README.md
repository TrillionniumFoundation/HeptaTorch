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
| 训练生命周期 | 保留参数／状态的图回收、工作区复用、带代际信息的张量 ID、旧句柄拒绝；可选持久会话支持跨帧训练及完整帧失败回滚 |
| 执行接口 | 有边界检查的二进制命令协议、明确错误状态、原生 HIA 十六进制解码工具，以及逐帧读写的原生持久会话 |
| 原生模块 | 独立借用 ABI1 与 owned ABI2 配置；原生签名／效果／线性所有权检查、buffer/chain 资源传递和 HIA 静态链接；通用张量对象 ABI 仍未完成 |

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

## 原生 binary64 标量基础

[HIA binary64 实现](tools/dtype-f64/README.md)提供加、减、乘、除、平方根及 binary32 双向转换，目标执行只使用 HIA 整数指令。固定的 CPython 55 项语料通过 1,968,156 个非 NaN 逐位比较及 66,910 个 NaN 分类检查。原版真实 PyTorch 逐位门禁仍为 **51／55 通过、4 项 sqrt 分组失败**；4,008 个差异全部保留。

[实际托管独立资格](validation/hosted-native-f64.json)重新执行完整 55 项语料，保留非 sqrt 的全部精确断言和原报告的逐流哈希，并对 236,892 个有限正 sqrt 输入使用两种精确整数方法检查 HIA 最近偶数舍入，同时验证固定 MKL 参考的严格小于 1 ULP 误差界。本轮两类错误计数均为零，11 项参考来源检查全部通过。结果明确记录 **PARTIAL／组件资格 PASS**，原版严格门禁的实际退出码仍为 1，完整 dtype 门禁和固定后端逐位兼容仍开放；Tensor dtype、提升规则和设备路径尚未接入。

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

当前设备为 CPU，dtype 为 binary32，张量 rank 不超过 4；同时存活的张量描述符最多 255 个，可以通过图回收在同一请求内复用。普通请求的输入、输出和工作区各为 1 MiB。可选持久会话在同一进程内保存状态，每帧输入／输出仍各为 1 MiB，使用 durable／shadow 两份 1 MiB 工作区提供完整帧事务；进程重启后的持久化尚未实现。卷积、矩阵乘法和池化各有每次 16,777,216 项的计算预算。整数索引载荷不等于完整整数张量 dtype 支持。空间、优化器和自动微分的具体限制见 [接口协议](docs/protocol.md)及 [能力清单](compatibility/capabilities.json)。

本轮在此前矩阵、VJP、图回收和 Adam 舍入修正基础上，实现了 cat/stack 与 scatter_add/index_add，并修复通用空张量的连续步长和晚出现零维度的形状接纳。cat 的 channels-last 是实际存储布局；index_add 的 alpha 舍入按已测固定 CPU 路径区分 rank 0/1 和更高 rank，不代表所有 ISA 都采用同一路径。具体行为见 [协议](docs/protocol.md)与 [数值说明](docs/numerics.md)。

当前权威源码使用 143／256 个函数名额、1,783／2,560 个基本块及 7,064／8,192 条指令。已运行的借用缓冲区模块桥接推进了独立原生编译与链接；独立 owned ABI2 已实现资源传递、返回及原生 syscall 分配；新的 owned Tensor 候选已运行可增长的数值存储；独立对象生命周期和任意外部 ABI 仍需完整实现和验收。原 borrowed 会话保留有界 arena；新的 owned 会话显式持有和移交 heap owner。具体证据及完整目标见 [审计更新](docs/upstream-audit.md#tensor-composition-and-indexed-addition-audit)。

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
