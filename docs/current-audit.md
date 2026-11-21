# 当前 HIA 重写审计

审计日期：2026-11-21。当前生产源码以 [manifest](../tools/current_runtime/source/manifest.json) 为准；完整目标以 [full-acceptance.json](../compatibility/full-acceptance.json) 为准。

## 当前状态

当前候选包含 **773个HIA核心函数、15个编译单元**。新增混合F64算术的2,541例及受影响的原F6448，已在同一规范映像上实际执行并完成独立物理回读，合计2,589例。完整最终树仍需新的 **27组／7,987例**共同门、其他实际头检查和完整原件复核；专项结果不能相加或移植成整树PASS。

完整目标保持PyTorch v2.15.0、固定提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`：**2,590条原生声明、688条导数声明、17项完整系统要求**。当前10项进行中、7项开放，完整声明及整项系统合同闭合数均为0，整体 **HOLD**。当前实现是Linux x86-64 CPU、有界rank0–4及选定dtype/layout的原生运行时，完整前端、设备与系统能力仍需继续实现。

| 当前773身份 | SHA-256 |
| --- | --- |
| 完整源码 | `4722c53b248780534a846a472784a565c4454ab738db75d655a7801b6623f07f` |
| 主ELF | `5cb190cfc4b920bb03e78803e79296e2e046ec90eb6e07b2d5ba6eebfb534edd` |
| Manifest | `be3eaf71fe817f7d365f70eee21574fcbefa8d6027babb2a57796764b100e362` |

## 新能力与实际结果

命令2的操作1–4已接受公开F64与F32、Bool、I64双向组合的无梯度加减乘除，保留rank相关类型提升、广播、可达布局和独立Tensor/Storage/Version所有权。I64直接按最近偶数转换到F64，避免经过F32造成预先舍入。原两条Bool减法错误继续拒绝。同类型F64仍走原实现，其完整NaN payload一致性尚未闭合；AD、out/inplace、其他alpha／rounding_mode和完整设备／架构语义也仍开放。

相对771只改变 `binary` 和 `half_mul_cast`，其余769个函数正文逐字保留，追加两个实际HIA函数。完整773→771逆变换及正向往返继续接入此前的完整源码证明链。原Half分流正文保持不变；Half同帧所有权合同另逐函数核对，并明确证明新的I64→F64 guard不属于原Half/F32同帧请求域。

| 实际专项执行及物理回读 | 混合F64算术2541 | 原F6448 |
| --- | ---: | ---: |
| 案例／原生子会话 | 2,541 | 48 |
| 完整帧 | 80,200 | 3,259 |
| 断言／检查 | 304,449 | 17,923 |
| Owner快照／原件 | 22,960 | 626 |
| 逐位结果元素 | 460,792 | 原完整比较及24个明确数值后继 |
| 真实OOM及相同请求恢复 | 4 | 2 |
| 实际执行、完整物理回读 | PASS | PASS |

原F64入口87个请求实测为63拒绝、1个取负、22个混合算术单命令和1个原始同帧FE事务。23个新增后继与2541组共享同一份真实Torch参考；取负继续共享真实405参考。原请求、Bool错误、两次OOM、所有权及退出判定均保留。最初两次新测试失败保留原件：一次是Bool错误类型假设错误，一次是未使用命令字段的拒绝假设错误；最终测试按固定Torch及既有协议修正，HIA源码未随这两次修正变化。详情见[混合F64契约](native-public-f64-arithmetic.md)和[实际源码绑定记录](../validation/current-runtime773/native-review.json)。

已实现的Half加减乘除、取负、BF16复制、I64／F64取负、对象生命周期、选定反向传播、NN、优化器和RNG仍纳入完整当前门。其过去源码上的专项成功仅为对应历史事实；新773需要全部重跑。Half239请求仍要求151拒绝、22乘法、2取负、40单命令加减、2原始同帧FE加法及22除法。Half47、乘法312、加减1006及除法1196共享同一实际1196参考。

## 完整门与已核验基线

[统一协调器](../tests/qualify_current_runtime.py)、CI与封存器固定27组／7,987例及其精确顺序。新参考、原始请求／响应、全Owner与Storage、真实工具执行、全部输入模式、退出和双流EOF均必须保留。缺组、跳过、超时、输入变化、缺EOF或物理回读失败都会阻止PASS。Generator ABI／重排探针继续使用当前对象分别链接并实际执行。当前完整状态以[资格计划](../validation/current-runtime773/qualification-plan.json)和精确Git树的实际运行记录为准。

[PR48](https://github.com/TrillionniumFoundation/HeptaTorch/pull/48) 已合并于 `6a031f2d1a25e5c4ccbc1efc3dc13bd71175544e`。其765映像完整24组／3,845例、全部十项头检查及完整原件回读通过，测试与合并树同为 `8e0c611401e4796aa697e14962f0db9b8bb490ae`。[固定核验](../validation/current-runtime765/ci-review.json)覆盖151,794个原始归档成员、152,506项输入绑定、全部源码／工具和文件模式。官方ZIP为504,716,668字节，SHA-256 `10c00546bfb489bdb4c2ba56cfa43d5cb2f812644f76172075af691bd6189dd4`。

PR49旧头 `66582c67f76cee822e942c73435b5cebe126a132` 的完整运行通过23组／3,755例，numeric485与metadata10在原生执行前因源码准入失败。其完整失败原件保留于[767核验](../validation/current-runtime767/ci-review.json)。修复仅补接遗漏的完整源码逆链与正确清单SHA；767和771各自实际重跑495例并通过，[771修复记录](../validation/current-runtime771/admission-review.json)保持原始输入身份。旧PR50同样的两个准入失败也保留，不改写成通过。

修复后的[PR49](https://github.com/TrillionniumFoundation/HeptaTorch/pull/49)和[PR50](https://github.com/TrillionniumFoundation/HeptaTorch/pull/50)均已通过各自完整25组／4,250例、13项头检查及完整原件独立回读，并分别合并为 `e4d43b8fa4b4749ff2b5638cb70608959cc9633c` 和 `094abcd2b69c64041c879d489ef602a0ab6187c7`。合并树与各自实际测试树完全一致，分别为 `059aba70325e34965ba38dee558cd296f7524e41` 和 `dc0e095e2b9056b9ef005194f0068a88f52197f8`。对应[运行44863163319](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44863163319)的原ZIP为523,591,829字节、SHA-256 `4ba1b9a5ca3af447d4c5d9f7a13bb742060f4f2b987340be074131034e064e67`；[运行44863162989](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44863162989)的原ZIP为523,388,554字节、SHA-256 `eef99e1eb23ccb92fb77e18924716078b77e1899c637fa922d2ea58a44b04604`。两份原件分别完整复核157,320和157,316个成员、各161,651项输入绑定，包含原生工具、文件模式、单XZ流和精确EOF。

PR51原头 `87702d6b129198b28098c6c26d905a2e91117512` 的[运行44863684626](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44863684626)已执行完整26组／5,446例并通过封存和上传，但最后的归档元数据GET返回HTTP404，整条工作流失败。官方原件14692339174为552,213,259字节、SHA-256 `cd83292e71a8a4e1991c31514c76443d6b37c59790c4513c7cd42c3c3980666c`，完整独立回读仍须完成。修复提交 `8a93be6867d6e12877b8645068decfc3b78ede71` 仅为同一归档／产生运行的GET增加最多7次、90秒内的有限重试，并保留全部身份、摘要、大小和分片验证；23项传输测试通过。当前773继承同一修复，仍要求自己的完整新树运行。旧成功、同一个ELF、来源静态证明或不同快照的专项数字均不替代新树的实际完整执行。

完整官方artifact ZIP始终原样上传。额外传输作业完整核对官方ZIP身份、长度和SHA，再保留精确200 MiB区间；重组须等于独立查询的官方原件。完整原件、单XZ流、全部TAR成员与尾部均需验证。

## 当前仓库整理与旧实现迁移

默认生产目录只保留一份完整当前HIA源码、一份规范manifest和15个当前单元。本批移出114个已核对无活跃消费者的过程文件、771旧资格计划和过期分区生成摘要，共116个过程文件、12,883,853字节；累计543个过程文件、177,290,747字节、38个完整阶段目录。旧13个核心单元由773分区替换，单独计为分区替换，不计为移除13项功能。精确原件字节、Git身份、依赖边与固定历史位置见[退休清单](../validation/current-runtime773/retirement.json)和[历史索引](history.md)。

依赖审计核对全树1,111个Python／shell／workflow／Makefile入口及相关代码、清单、文档，共2,345个精确Git blob。114个待退休文件的12,877,907字节已从固定87702d6提交全部读取并核对SHA及Git blob；此前实际归档的153,351项输入绑定对这13个目录没有引用，新增独立复核也确认它们均不在成功PR50完整4,250例的原始归档中。该证据限定为静态依赖和实际声明输入图，最终27组仍需在清理后的精确树执行。

旧143函数源码仍由部分旧Native、F64、RNG及独立作者路径读取。来源迁移第一阶段已实际验证固定历史下载、离线缓存和现有源码准入，16项拒绝／恢复检查通过；原Native107步骤及F64/RNG各8步骤完整保留。[来源记录](../validation/current-runtime773/legacy-source143-review.json)明确旧流水线仍须全部执行后才能继续移除跟踪的旧源码。当前参考包、原失败证据、许可证与真实依赖保留，完整迁移路线见[依赖清单](runtime-dependencies.md)。

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
