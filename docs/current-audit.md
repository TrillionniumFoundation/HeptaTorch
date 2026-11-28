# 当前 HIA 重写审计

审计日期：2026-11-21。当前源码以 [manifest](../tools/current_runtime/source/manifest.json) 为准；完整目标以 [full-acceptance.json](../compatibility/full-acceptance.json) 为准。

## 当前状态

精确主分支提交 `e9444902df9878060ab389a70ee957c1b7f4a887` 的 Current HIA Runtime 已实际通过 **29 组／13,851 例**。原始 ZIP、构建、源码、ELF 及完整资格报告绑定见[精确主分支读回记录](../validation/current-runtime775/EXACT-MAIN-REVIEW.md)。这不改变全量重写 HOLD，也不替代其他提交的资格验证。

当前候选包含 **775 个 HIA 核心函数、15 个编译单元**。新增 I64、F64、F16、BF16 的原生无梯度 ReLU；两套原取负与ReLU共 **1,552例**已在同一实际冷构建上使用同一份ReLU参考运行，通过原始物理读回。每个新候选仍须在固定提交上共同执行全部 **29 组／13,851 例**，其他精确树检查也须各自完成。

完整目标保持PyTorch v2.15.0、固定提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`：**2,590条原生声明、688条导数声明、17项完整系统要求**。当前10项进行中、7项开放，完整声明和整项系统契约闭合数均为0，整体 **HOLD**。当前实现限定于Linux x86-64 CPU、有界rank0–4及选定dtype/layout；完整前端、设备与系统能力仍需继续实现。

| 当前775身份 | SHA-256 |
| --- | --- |
| 完整源码 | `0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43` |
| 主ELF | `ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e` |
| Manifest | `348ed8e949256f985cdca0d9c612dc646ae738beee6013c27ab405d2df3e67c8` |

## 原生 ReLU 与实际专项结果

命令3／操作5现接受上述四种dtype的无梯度张量。新增值内核只处理原始整数位模式，分配、迭代、所有权及失败回滚均由HIA执行。负数变为正零，负零保留；F64的NaN载荷保留。Half完整32元素向量组按固定AVX2参考执行NaN quieting／BF16规范化，标量尾段保留原载荷。原F32分支保持正文，Bool ReLU仍拒绝。[ReLU契约](native-typed-relu.md)说明准确范围。

相对PR53的773函数源码，仅修改原 `unary` 函数，增加 `typed_relu_value` 与 `typed_relu_tensor`；另772个原函数正文和定义顺序完整保留。[源码审阅](../validation/current-runtime775/source-review.json)验证完整775→773→原始源码链以及旧比较器逆变换。原两套取负中的四条ReLU请求保留完整输入和96字节原请求，其拒绝判定改成实际数值、布局、新对象及释放验收；其余原控制均保留。Half239和F6487入口分区没有ReLU请求，分区计数不变。

| 同一构建和同一ReLU参考的实际结果 | Half取负 | I64／F64取负 | ReLU |
| --- | ---: | ---: | ---: |
| 案例／原生会话 | 371 | 405 | 776 |
| 完整帧 | 11,415 | 10,522 | 21,863 |
| 断言／检查 | 37,920 | 37,490 | 75,060 |
| Owner快照 | 2,729 | 2,971 | 5,672 |
| 逐位结果元素 | 272,448 | 287,279 | 559,719 |
| 协议控制槽位 | 46 | 46 | 90 |
| 真实OOM及相同请求恢复 | 2 | 2 | 4 |
| 原始运行及物理读回 | PASS | PASS | PASS |

这三组共1,552例、43,800帧、150,470项检查。原生运行与固定wheel参考均实际执行，所有11项来源检查通过；完整41份原报告、进程收据和输出共6,834,992字节已逐字读取、核对模式并复核全部案例。远端原始物理读回读取了全部请求／响应、Owner及参考blob；本地独立复核已覆盖全部报告，尚未覆盖全部远端原始帧／blob。当前仓库保留[完整报告归档和实际记录](../validation/current-runtime775/native-review.json)，完整CI仍须另行读回全部原件。

先前独立ReLU776成功、准备阶段的摘要不匹配、上传超时缺包，以及历史68补验中漏传依赖文件的失败均保留各自身份。准备失败没有执行原生案例。报告复核器对build.units数据形状和参考状态名的两次错误假设已单独保留，修复仅依据原报告格式，不改写原报告、不重跑原生程序。

## 当前775的三处准入修复

固定提交 `e856c3580e0a0e09d92500c39763cfa2b9a2f963` 的源码中另有三处旧绑定：数值谓词入口直接向773逆变换传入775源码，live32准备器仍要求773函数，F32后置资格器仍固定旧helper摘要及773标签。本次仅在这三个函数内补接完整775逆变换或修正精确身份；HIA源码、15单元及主ELF均不变，其余22个Python函数／类正文逐字保留。

数值谓词旧入口的实际源码准入失败和live32旧准备失败均保留；修复后的数值谓词通过101份实际源码／冷构建输入检查。三处修复合并后，live32完整原包的准备流程重新实际通过93份输入、正常退出与双流EOF，原生及Torch执行标志均为false。这些结果只证明准入和准备。

另一次775历史补验实际完成Bool27、Integral24和F32的全部68例、98个原生进程、28,797帧；原任务因F32后置资格器的旧helper摘要而以exit1／FAIL结束。完整原包29,087,484字节、9,666个成员及304,960,331字节原负载已逐份回读。修复后的资格器在同一批原始请求、响应、Owner和原参考上通过全部后置物理重读，12个篡改对照全部拒绝。原FAIL任务和原报告不改写，没有重跑原生程序或重算参考值。[窄修复及实际读回记录](../validation/current-runtime775/admission-repair-review.json)明确区分原任务结果、源码准入、后置重读与完整CI资格；此处的窄修复本身不代表 29 组完整资格；随后的精确主分支结果见上方记录。

## 继承能力与未通过的前身

同类型F64加减乘除及单元素广播NaN选择已在前身773实际执行5088例；混合F64另有2541例，原F6448和公开F64逻辑192也有独立原始读回。PR53完整运行中这些组均通过，但其余三组失败使整套门保持FAIL。它们的原始来源、两次真实NaN数值失败及修复见[同类型F64契约](native-same-f64-arithmetic.md)和[前身执行记录](../validation/current-runtime-same-f64/native-review.json)。所有组都必须在775重新运行，不能继承前身资格。

原F6487请求保持63拒绝、1取负、22混合算术单命令及1原同帧FE事务。Half239保持151拒绝、22乘法、2取负、40单命令加减、2原同帧FE加法及22除法。Half47、乘法312、加减1006及除法1196共用同一份本次1196参考；F6448与混合2541共用同一份本次2541参考。ReLU776和两套取负共用本次776参考。协调器对这些共同执行关系逐项绑定。

独立旧F64组件的严格Torch比较仍保留4／55组失败：PR52在AVX512记录1,984个逐位差异，PR53在AVX2记录4,008个逐位差异。精确有理数RNE复核和小于1 ULP的诊断不改写严格FAIL，也未证明CPU能力差异是成因。旧组件的 `sqrt64` 未接入canonical775，当前公开F64 sqrt仍拒绝。完整dtype、AD、设备、架构和全部算子契约继续开放。

## 完整门与已核验基线

[统一协调器](../tests/qualify_current_runtime.py)、CI与封存器固定29组／13,851例及其精确顺序。新参考、原始请求／响应、全Owner与Storage、真实工具执行、全部输入模式、退出和双流EOF均必须保留。缺组、跳过、超时、输入变化、缺EOF或物理回读失败都会阻止PASS。Generator ABI／重排探针继续使用当前对象分别链接并实际执行。当前完整状态以[资格计划](../validation/current-runtime775/qualification-plan.json)和精确Git树的实际运行记录为准。

[PR48](https://github.com/TrillionniumFoundation/HeptaTorch/pull/48) 已合并于 `6a031f2d1a25e5c4ccbc1efc3dc13bd71175544e`。其765映像完整24组／3,845例、全部十项头检查及完整原件回读通过，测试与合并树同为 `8e0c611401e4796aa697e14962f0db9b8bb490ae`。[固定核验](../validation/current-runtime765/ci-review.json)覆盖151,794个原始归档成员、152,506项输入绑定、全部源码／工具和文件模式。官方ZIP为504,716,668字节，SHA-256 `10c00546bfb489bdb4c2ba56cfa43d5cb2f812644f76172075af691bd6189dd4`。

PR49旧头 `66582c67f76cee822e942c73435b5cebe126a132` 的完整运行通过23组／3,755例，numeric485与metadata10在原生执行前因源码准入失败。其完整失败原件保留于[767核验](../validation/current-runtime767/ci-review.json)。修复仅补接遗漏的完整源码逆链与正确清单SHA；767和771各自实际重跑495例并通过，[771修复记录](../validation/current-runtime771/admission-review.json)保持原始输入身份。旧PR50同样的两个准入失败也保留，不改写成通过。

修复后的[PR49](https://github.com/TrillionniumFoundation/HeptaTorch/pull/49)和[PR50](https://github.com/TrillionniumFoundation/HeptaTorch/pull/50)均已通过各自完整25组／4,250例、13项头检查及完整原件独立回读，并分别合并为 `e4d43b8fa4b4749ff2b5638cb70608959cc9633c` 和 `094abcd2b69c64041c879d489ef602a0ab6187c7`。合并树与各自实际测试树完全一致，分别为 `059aba70325e34965ba38dee558cd296f7524e41` 和 `dc0e095e2b9056b9ef005194f0068a88f52197f8`。对应[运行44863163319](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44863163319)的原ZIP为523,591,829字节、SHA-256 `4ba1b9a5ca3af447d4c5d9f7a13bb742060f4f2b987340be074131034e064e67`；[运行44863162989](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44863162989)的原ZIP为523,388,554字节、SHA-256 `eef99e1eb23ccb92fb77e18924716078b77e1899c637fa922d2ea58a44b04604`。两份原件分别完整复核157,320和157,316个成员、各161,651项输入绑定，包含原生工具、文件模式、单XZ流和精确EOF。

PR51原头 `87702d6b129198b28098c6c26d905a2e91117512` 的[运行44863684626](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44863684626)已执行完整26组／5,446例并通过封存和上传，但最后的归档元数据GET返回HTTP404，整条工作流失败。官方原件14692339174为552,213,259字节、SHA-256 `cd83292e71a8a4e1991c31514c76443d6b37c59790c4513c7cd42c3c3980666c`，完整独立回读仍须完成。修复提交 `8a93be6867d6e12877b8645068decfc3b78ede71` 仅为同一归档／产生运行的GET增加最多7次、90秒内的有限重试，并保留全部身份、摘要、大小和分片验证；23项传输测试通过。当前775继承同一修复，仍要求自己的完整新树运行。旧成功、同一个ELF、来源静态证明或不同快照的专项数字均不替代新树的实际完整执行。

修复后的 PR51 [运行44877738258](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44877738258)已通过完整 26 组／5,446 例及无损分片传输。官方原件 14698175373 为 552,307,293 字节，SHA-256 `89456a06a3a8fa5c5289d7bfd621b10702f2a61b32fa7377de0ed652e2b96cee`；已完整读取 173,079 个成员、199,039 项声明输入和 80 份输入清单。该成功运行未出现元数据重试，不能据此反推原 HTTP404 的根因。原 Half239 读回保留 237 份静态请求字节，另两条 stale-capability 请求按完整创建／释放／拒绝身份链检查动态 generation，不能要求版本间所有动态 token 相同。 其全部 13 项精确头检查、13 份 Native 原 ZIP（270,720,214 字节、90 份原报告／53 份必需报告）、三份 observer 原 ZIP、完整源码逆链、冷构建和分别链接的 Generator 探针均已通过独立读回；PR51 已合并为 `e1a2f67e894b3d9b823e9fbcc5dfc79e12275757`，合并树与实际测试树同为 `512127e458e9778513bb1eb830304f82cbac25fc`。

[PR52](https://github.com/TrillionniumFoundation/HeptaTorch/pull/52) 的 [运行44878107007](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44878107007)原样保留 **FAIL**：26 组／7,795 例通过，公开 F64 逻辑 192 例因外层包装器仍 pin 旧 `derive736.py`／`run736.py` 而在准备阶段未执行。官方原件 14698316360 为 591,467,482 字节，SHA-256 `fe597fab9872f7f3b368f72f3f180d13f88ca44ffdbacf0eea3591f31b408dab`；完整 203,536 个成员、242,254 项声明输入及 85 份输入清单已独立读取。773后继重新从原六个冻结模块派生包装器，补齐内外全部 pin，并在该773映像及PR53实际补验192例；775仍须重新执行这192例。失败 PR52 不作通过基线，775 后继仍须执行全部 29 组。

[PR53](https://github.com/TrillionniumFoundation/HeptaTorch/pull/53) 固定头 `83ed6b41ea67aa37182d23159ec0cb436e632dac` 的 [Current运行44883329985](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44883329985)同样保留 **FAIL**：28 组中25组／13,007例通过；Bool27、整数24和F32 Sum17共68例在原生案例前因同一个旧摘要 pin 被拒绝。三组原报告的 `native_executed=False` 和空案例表保持不变；源码和映像身份本身正确。修复更新共享来源校验链，保留原数值判定和严格真实执行要求。新775仍须重新实际执行完整29组。

PR53官方失败原件14700680070为602,609,133字节、SHA-256 `a68e321c0bf540aed4ecc09f456821084ffcc72e75eb82e2d3ff81e7d00ea37d`。已完整读回257,749个成员、2,736,999,083原始载荷字节、324,293项输入绑定及94份清单；成功25组的原始协议、Owner、Generator、源码链和实际冷构建均完成独立复核。资格步骤以退出1终止，其后的两条旧源码缺失断言没有执行，不能记为通过。开始和资格前的缺失断言确实执行。完整归档可读不改变整套失败结论。

完整官方artifact ZIP始终原样上传。额外传输作业完整核对官方ZIP身份、长度和SHA，再保留精确200 MiB区间；重组须等于独立查询的官方原件。完整原件、单XZ流、全部TAR成员与尾部均需验证。

## 当前仓库整理与旧实现迁移

默认生产目录只保留一份完整当前HIA源码、一份规范manifest和15个当前单元。混合 F64 前身移出114个已核对无活跃消费者的过程文件、771旧资格计划和过期分区生成摘要，共116个过程文件、12,883,853字节；累计543个过程文件、177,290,747字节、38个完整阶段目录。旧13个核心单元由773分区替换，单独计为分区替换，不计为移除13项功能。精确原件字节、Git身份、依赖边与固定历史位置见[退休清单](../validation/current-runtime773/retirement.json)和[历史索引](history.md)。

依赖审计核对全树1,111个Python／shell／workflow／Makefile入口及相关代码、清单、文档，共2,345个精确Git blob。114个待退休文件的12,877,907字节已从固定87702d6提交全部读取并核对SHA及Git blob；此前实际归档的153,351项输入绑定对这13个目录没有引用，新增独立复核也确认它们均不在成功PR50完整4,250例的原始归档中。该证据限定为静态依赖和实际声明输入图，当前后继仍须在最终清理后的精确树执行全部29组。

当前候选进一步移除跟踪的旧 `src/heptatorch.hepta`（481,516 字节），并只保留775后继的活动资格计划。固定历史源码恢复器、manifest、原 16 项控制及冻结作者／比较器均不修改；缺失源码的真实 HTTPS 冷恢复、离线缓存恢复、旧来源／RNG／F64／143 分区检查和八条 Make 依赖路线已验证，漏掉恢复前置条件的 `test-storage-objects` 已补齐。相关原件嵌入[移除记录](../validation/current-runtime-same-f64/legacy-source143-retirement.json)。

775 同时以新分区替换前身的13个核心单元，退休前身3,964字节资格计划；原函数保留情况由完整源码逆变换约束。固定Git身份和模式见[775退休记录](../validation/current-runtime775/retirement.json)。这13个文件是分区替换，不是删除13项功能。

默认当前构建没有新增旧源码恢复依赖；CI配置在开始及当前资格执行前后检查旧文件缺失。PR53因资格失败未执行最后两条检查，775必须重新实际通过全部检查。旧 Native、F64 和 RNG 流水线则必须从缺源码／缺缓存状态实际恢复固定历史字节；原 107／8／8 个测试步骤及已增加的来源检查全部保留。这仅移除了跟踪的历史源码，旧协议消费者的完整语义迁移尚未完成。当前参考包、原失败证据、许可证及真实依赖保留，路线见[依赖清单](runtime-dependencies.md)。

完整契约检查已改为源码内注册的独立验证器准入：证据需绑定当前 manifest、全部单元、原生映像、构建／执行／参考／读回及验证器版本；重复 inode、硬链接别名、符号链接、读取中替换、伪造计数和只写 PASS 均不能通过。30 项目标保护负例与 25 项框架测试通过。生产注册表仍为空，三类完整闭合数保持 0。详情见[完整契约验收](full-contract-qualification.md)。

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

下一阶段的依赖顺序见[执行计划](continuation-plan.md)。`make acceptance-status` 验证账本及源码注册的完整契约证据；`make full-acceptance` 当前返回 2／HOLD，因为各项完整原生合同及其完整验收运行器尚待实现。
