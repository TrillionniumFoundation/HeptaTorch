# 当前 HIA 重写审计

审计日期：2026-11-21。当前生产源码以 [manifest](../tools/current_runtime/source/manifest.json) 为准；完整目标以 [full-acceptance.json](../compatibility/full-acceptance.json) 为准。

## 当前状态

当前候选包含 **773 个 HIA 核心函数、15 个编译单元**。本轮已修复同类型 F64 加减乘除及单元素广播的 NaN 载荷选择，实际通过同类型 F64 5,088 例、原混合 F64 2,541 例、原 F64 48 例及公开 F64 逻辑 192 例，并分别完成原始物理读回。这些专项绑定各自真实冷构建及同一源码／ELF 身份；它们不构成一次完整组合运行。最终候选仍要求 **28 组／13,075 例**共同门、其他精确树检查和全部原件复核。
完整目标保持PyTorch v2.15.0、固定提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`：**2,590条原生声明、688条导数声明、17项完整系统要求**。当前10项进行中、7项开放，完整声明及整项系统合同闭合数均为0，整体 **HOLD**。当前实现是Linux x86-64 CPU、有界rank0–4及选定dtype/layout的原生运行时，完整前端、设备与系统能力仍需继续实现。

| 当前773身份 | SHA-256 |
| --- | --- |
| 完整源码 | `846e592beaa7847383cee668bc33d25cd81e79052bd4612252e2f72a83d96d05` |
| 主ELF | `5cba6f11269f71efb319aa7368009055bb2e9b9571329daeae9bbc453287c3ad` |
| Manifest | `62795753a0e782fe3f75b9086653d41aba743fda356673398c12432d576fac15` |

## 新能力与实际结果

命令 2 的操作 1–4 在既有 F64 与 F32／Bool／I64 混合算术基础上，新增同类型 F64/F64 的原生分流。全部运算、类型提升、对象分配和回滚仍由 HIA 执行。I64 直接按最近偶数转换到 F64；Bool 减法错误保留。适用范围仍是固定 AVX2／单线程 CPU、rank 0–4 和可达布局，AD、out/inplace、其他 alpha／rounding_mode 及完整设备／架构能力尚未实现。

相对 mixed-F64773 前身，只改变 `public_f64_binary` 和 `half_mul_priority` 两个函数，其他 771 个函数正文及所有定义顺序逐字保留。乘法在结果为 F64、元素数为 1、左右 rank 不等时，复用既有 HIA 维度排序，读取最后一个排序轴的实际广播步长；单纯选固定最外轴会出错。该 guard 检查的是提升后的结果 dtype，也可能覆盖 Half/F64 提升到 F64 的请求，因此原 Half 组仍须完整重跑。原 F16/BF16 结果分支、算术值内核、所有权及协议函数保持原正文。[源码审阅](../validation/current-runtime-same-f64/source-review.json)给出精确边界。

| 实际专项执行及物理回读 | 同类型 F64 5088 | 混合 F64 2541 | 原 F64 48 | 公开 F64 逻辑 192 |
| --- | ---: | ---: | ---: | ---: |
| 案例 | 5,088 | 2,541 | 48 | 192 |
| 原生子会话 | 5,088 | 2,541 | 48 | 196 |
| 完整帧 | 155,576 | 80,200 | 3,259 | 3,711 |
| 断言／检查 | 601,692 | 304,449 | 17,923 | 8,489 |
| Owner 快照／原件 | 46,048 | 22,960 | 626 | 原完整 Owner 归档读回 |
| 逐位结果元素 | 769,644 | 460,792 | 原完整比较及 24 个数值后继 | 原 192 例完整比较 |
| 真实 OOM 及相同请求恢复 | 4 | 4 | 2 | 2 |
| 实际执行、完整物理回读 | PASS | PASS | PASS | PASS |

新 5,088 例保留原 3,288 例全部配方与顺序，追加 1,800 例单元素 rank／NaN／正负零／有限值／步长组合，覆盖 rank 0–4 的 25 种有序组合。第一版在 case2224 选错 NaN 载荷；第二版通过全部原 3,288 例，但新增矩阵揭示 9 个不规则步长失败。第三版修复维度排序，原参考字节及全部测试配方均未修改。两次原生失败及相关准备／外层收据错误均分别保留；原生已成功执行的 F64 48 例在外层重复写收据后报错，随后只恢复其原物理记录的独立读取，没有重跑或改写原生结果。详情见[同类型 F64 契约](native-same-f64-arithmetic.md)和[实际执行记录](../validation/current-runtime-same-f64/native-review.json)。

原 F64 入口 87 个配方保持为 63 个拒绝、1 个取负、22 个混合算术单命令和 1 个原始同帧 FE 事务。F64 48 例自己的本次实际 2541／405 参考与后继绑定完整通过；独立 mixed2541 专项使用其自己的实际参考。完整 28 组协调器另外强制 F6448 与 mixed2541 共享同一份本次实际参考，不能把两个独立专项拼成该共同运行。原 Bool 错误、两次 OOM、全部 Owner 与退出判定保留；前身混合算术的两次测试假设错误及其原始记录继续保留于[原执行记录](../validation/current-runtime773/native-review.json)。

独立旧 F64 组件在 PR52 的 AVX512 严格 Torch 对照中仍有 4／55 组失败、1,984 个正有限结果相差 1 ULP。其 14 函数组件的 `sqrt64` 未接入当前 canonical773；当前公开 F64 一元 sqrt 仍拒绝。该旧组件的严格失败、旧 F6448 参考中的历史差异和当前固定 AVX2 算术资格各自保留，不能互相替代。

已实现的Half加减乘除、取负、BF16复制、I64／F64取负、对象生命周期、选定反向传播、NN、优化器和RNG仍纳入完整当前门。其过去源码上的专项成功仅为对应历史事实；新773需要全部重跑。Half239请求仍要求151拒绝、22乘法、2取负、40单命令加减、2原始同帧FE加法及22除法。Half47、乘法312、加减1006及除法1196共享同一实际1196参考。

## 完整门与已核验基线

[统一协调器](../tests/qualify_current_runtime.py)、CI与封存器固定28组／13,075例及其精确顺序。新参考、原始请求／响应、全Owner与Storage、真实工具执行、全部输入模式、退出和双流EOF均必须保留。缺组、跳过、超时、输入变化、缺EOF或物理回读失败都会阻止PASS。Generator ABI／重排探针继续使用当前对象分别链接并实际执行。当前完整状态以[资格计划](../validation/current-runtime-same-f64/qualification-plan.json)和精确Git树的实际运行记录为准。

[PR48](https://github.com/TrillionniumFoundation/HeptaTorch/pull/48) 已合并于 `6a031f2d1a25e5c4ccbc1efc3dc13bd71175544e`。其765映像完整24组／3,845例、全部十项头检查及完整原件回读通过，测试与合并树同为 `8e0c611401e4796aa697e14962f0db9b8bb490ae`。[固定核验](../validation/current-runtime765/ci-review.json)覆盖151,794个原始归档成员、152,506项输入绑定、全部源码／工具和文件模式。官方ZIP为504,716,668字节，SHA-256 `10c00546bfb489bdb4c2ba56cfa43d5cb2f812644f76172075af691bd6189dd4`。

PR49旧头 `66582c67f76cee822e942c73435b5cebe126a132` 的完整运行通过23组／3,755例，numeric485与metadata10在原生执行前因源码准入失败。其完整失败原件保留于[767核验](../validation/current-runtime767/ci-review.json)。修复仅补接遗漏的完整源码逆链与正确清单SHA；767和771各自实际重跑495例并通过，[771修复记录](../validation/current-runtime771/admission-review.json)保持原始输入身份。旧PR50同样的两个准入失败也保留，不改写成通过。

修复后的[PR49](https://github.com/TrillionniumFoundation/HeptaTorch/pull/49)和[PR50](https://github.com/TrillionniumFoundation/HeptaTorch/pull/50)均已通过各自完整25组／4,250例、13项头检查及完整原件独立回读，并分别合并为 `e4d43b8fa4b4749ff2b5638cb70608959cc9633c` 和 `094abcd2b69c64041c879d489ef602a0ab6187c7`。合并树与各自实际测试树完全一致，分别为 `059aba70325e34965ba38dee558cd296f7524e41` 和 `dc0e095e2b9056b9ef005194f0068a88f52197f8`。对应[运行44863163319](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44863163319)的原ZIP为523,591,829字节、SHA-256 `4ba1b9a5ca3af447d4c5d9f7a13bb742060f4f2b987340be074131034e064e67`；[运行44863162989](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44863162989)的原ZIP为523,388,554字节、SHA-256 `eef99e1eb23ccb92fb77e18924716078b77e1899c637fa922d2ea58a44b04604`。两份原件分别完整复核157,320和157,316个成员、各161,651项输入绑定，包含原生工具、文件模式、单XZ流和精确EOF。

PR51原头 `87702d6b129198b28098c6c26d905a2e91117512` 的[运行44863684626](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44863684626)已执行完整26组／5,446例并通过封存和上传，但最后的归档元数据GET返回HTTP404，整条工作流失败。官方原件14692339174为552,213,259字节、SHA-256 `cd83292e71a8a4e1991c31514c76443d6b37c59790c4513c7cd42c3c3980666c`，完整独立回读仍须完成。修复提交 `8a93be6867d6e12877b8645068decfc3b78ede71` 仅为同一归档／产生运行的GET增加最多7次、90秒内的有限重试，并保留全部身份、摘要、大小和分片验证；23项传输测试通过。当前773继承同一修复，仍要求自己的完整新树运行。旧成功、同一个ELF、来源静态证明或不同快照的专项数字均不替代新树的实际完整执行。

修复后的 PR51 [运行44877738258](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44877738258)已通过完整 26 组／5,446 例及无损分片传输。官方原件 14698175373 为 552,307,293 字节，SHA-256 `89456a06a3a8fa5c5289d7bfd621b10702f2a61b32fa7377de0ed652e2b96cee`；已完整读取 173,079 个成员、199,039 项声明输入和 80 份输入清单。该成功运行未出现元数据重试，不能据此反推原 HTTP404 的根因。原 Half239 读回保留 237 份静态请求字节，另两条 stale-capability 请求按完整创建／释放／拒绝身份链检查动态 generation，不能要求版本间所有动态 token 相同。 其全部 13 项精确头检查、13 份 Native 原 ZIP（270,720,214 字节、90 份原报告／53 份必需报告）、三份 observer 原 ZIP、完整源码逆链、冷构建和分别链接的 Generator 探针均已通过独立读回；PR51 已合并为 `e1a2f67e894b3d9b823e9fbcc5dfc79e12275757`，合并树与实际测试树同为 `512127e458e9778513bb1eb830304f82cbac25fc`。

[PR52](https://github.com/TrillionniumFoundation/HeptaTorch/pull/52) 的 [运行44878107007](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44878107007)原样保留 **FAIL**：26 组／7,795 例通过，公开 F64 逻辑 192 例因外层包装器仍 pin 旧 `derive736.py`／`run736.py` 而在准备阶段未执行。官方原件 14698316360 为 591,467,482 字节，SHA-256 `fe597fab9872f7f3b368f72f3f180d13f88ca44ffdbacf0eea3591f31b408dab`；完整 203,536 个成员、242,254 项声明输入及 85 份输入清单已独立读取。当前后继重新从原六个冻结模块派生包装器，补齐内外全部 pin，并已实际补验 192 例。失败 PR52 不作通过基线，后继仍须执行全部 28 组。

完整官方artifact ZIP始终原样上传。额外传输作业完整核对官方ZIP身份、长度和SHA，再保留精确200 MiB区间；重组须等于独立查询的官方原件。完整原件、单XZ流、全部TAR成员与尾部均需验证。

## 当前仓库整理与旧实现迁移

默认生产目录只保留一份完整当前HIA源码、一份规范manifest和15个当前单元。混合 F64 前身移出114个已核对无活跃消费者的过程文件、771旧资格计划和过期分区生成摘要，共116个过程文件、12,883,853字节；累计543个过程文件、177,290,747字节、38个完整阶段目录。旧13个核心单元由773分区替换，单独计为分区替换，不计为移除13项功能。精确原件字节、Git身份、依赖边与固定历史位置见[退休清单](../validation/current-runtime773/retirement.json)和[历史索引](history.md)。

依赖审计核对全树1,111个Python／shell／workflow／Makefile入口及相关代码、清单、文档，共2,345个精确Git blob。114个待退休文件的12,877,907字节已从固定87702d6提交全部读取并核对SHA及Git blob；此前实际归档的153,351项输入绑定对这13个目录没有引用，新增独立复核也确认它们均不在成功PR50完整4,250例的原始归档中。该证据限定为静态依赖和实际声明输入图，当前后继仍须在最终清理后的精确树执行全部28组。

当前候选进一步移除跟踪的旧 `src/heptatorch.hepta`（481,516 字节），并只保留同类型 F64 后继的活动资格计划。固定历史源码恢复器、manifest、原 16 项控制及冻结作者／比较器均不修改；缺失源码的真实 HTTPS 冷恢复、离线缓存恢复、旧来源／RNG／F64／143 分区检查和八条 Make 依赖路线已验证，漏掉恢复前置条件的 `test-storage-objects` 已补齐。相关原件嵌入[移除记录](../validation/current-runtime-same-f64/legacy-source143-retirement.json)。

默认当前构建没有新增旧源码恢复依赖，CI 在开始及当前资格执行前后强制验证旧文件缺失。旧 Native、F64 和 RNG 流水线则必须从缺源码／缺缓存状态实际恢复固定历史字节；原 107／8／8 个测试步骤及已增加的来源检查全部保留。这仅移除了跟踪的历史源码，旧协议消费者的完整语义迁移尚未完成。当前参考包、原失败证据、许可证及真实依赖保留，路线见[依赖清单](runtime-dependencies.md)。

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
