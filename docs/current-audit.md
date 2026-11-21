# 当前 HIA 重写审计

审计日期：2026-11-20。源码身份以 [source manifest](../tools/current_runtime/source/manifest.json) 为准；完整目标以 [full-acceptance.json](../compatibility/full-acceptance.json) 为准。

## 当前状态

当前候选包含 **771 个 HIA 核心函数、15 个编译单元**，在767的I64／F64取负基础上继续实现Half真除法。新的1196例除法、原312例乘法、1006例加减法及Half47已在同一771映像上实际执行并通过独立物理回放，四组共2,561例，使用同一份实际1196参考。这些分组成功不等于完整组合PASS；当前仍要求新的 **26组／5,446例**统一验收。

完整目标仍是 PyTorch v2.15.0、固定提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`：**2,590 条原生声明、688 条导数声明、17 项完整系统要求**。10项进行中、7项开放；完整声明与整项系统验收闭合数均为0，整体HOLD。当前运行时是Linux x86-64、有界rank0–4及选定dtype/layout的原生实现，不能据此推断完整PyTorch兼容。

| 当前771身份 | SHA-256 |
| --- | --- |
| 完整源码 | `f534c77069abc15ba15879d44492aacbc5f3709f6a861a0f0b9e0e365232fca1` |
| 主ELF | `db935c74bf0a2b359038c7be56f5a5721af8f0c2d757a2330527bd751861ae86` |
| Manifest | `15337cd10313c9cd485e0c1db70b6d6da46d2cd4afa21053db9eb7f86d57301e` |

## 新能力与实际执行

命令2／操作4接受至少一个公开F16/BF16操作数的无梯度真除法，使用现有rank相关类型提升、广播、布局及独立Tensor/Storage/Version所有权。约化浮点结果在右操作数被TensorIterator识别为标量时保留原始F32运算精度；完整向量组、尾段、NaN payload及错误行为以固定Torch实际输出为准。所有数值、分配和回滚都由HIA执行。`rounding_mode`、out/inplace、Half AD和其他平台仍未闭合。

相对于767，只改变原 `binary` 分流并追加四个helper，其余766个原函数逐字保留。完整771→767逆变换检查全部源码和正向往返，继续接入767→765→761→758→742→740→736→700证明链。312／1006观察器保留旧完整逆变换入口，在原逆链之前只撤销明确的除法成功后继；原数值、错误、资源、所有权与退出判定保持完整。

| 同一771映像的实际验证 | 除法1196 | 原乘法312 | 原加减1006 | 原Half47 |
| --- | ---: | ---: | ---: | ---: |
| 原生子会话 | 1,196 | 312 | 1,006 | 47 |
| 完整协议帧 | 38,348 | 9,696 | 31,532 | 3,887 |
| 断言 | 144,654 | 37,314 | 120,134 | 20,133 |
| Owner快照／原件 | 10,816 | 2,878 | 9,186 | 605 |
| 逐位比较的结果元素 | 2,250,592 | 1,313,608 | 2,899,316 | 原908,216个输入及显式成功后继 |
| 实际执行及独立物理回放 | PASS | PASS | PASS | PASS |

实际参考使用固定官方CPU Torch2.15.0、CPython3.12、NumPy2.3.5，实际AVX2／单线程。1196参考只执行一次，四个消费者绑定相同原件、进程、解释器、全部输入／输出和EOF。除法1196包含32个协议控制及两次真实OOM／相同请求恢复；原1006保留70个控制和四次真实OOM／恢复，Half47保留四个原始OOM控制。

Half47的完整239请求实测分区为：151拒绝、22乘法、2取负、40单命令加减、2原始同帧FE加法及22除法。四条private-dtype3除法边界继续拒绝。乘法312中的两条原除法控制、加减1006中的四条原除法控制均使用相同1196参考逐位比较，并继续验证完整输入和所有权。

此前提到的native769原始源码、构建与证据在已检查范围内未恢复，不能视为本次结果。771的新1196矩阵独立实现并实际执行，没有沿用旧1128项声称。详情见[真除法契约](native-half-division.md)及[771实际记录](../validation/current-runtime771/native-review.json)。767取负的405例及原F6448的先前实际结果另见[取负契约](native-wide-negation.md)和[767记录](../validation/current-runtime767/native-review.json)，不转移为771的执行结果。

## 完整组合门

协调器、CI与封存器要求26组／5,446例，保留原25组及全部原请求或明确验证的成功后继。Half47、乘法312、加减1006与除法1196必须共享实际1196参考；F6448与wide_neg405仍必须共享实际405参考。Generator ABI／重排探针继续使用当前对象单独链接并实际执行。

每次源码或映像变化均冷编译全部应用对象，检查源码、工具、ELF、模式、引用清单、进程退出和双流EOF。缺组、跳过、超时、输入变化、缺少EOF或物理回放失败都会阻止PASS。新规范manifest的严格入场已验证；旧未固定manifest的771初探构建会被明确拒绝，不能修写收据后冒充正式构建。计划见[26组资格计划](../validation/current-runtime771/qualification-plan.json)。

## 767完整门暴露的入场缺陷与修复

PR49原头 `66582c67f76cee822e942c73435b5cebe126a132` 的完整运行通过23组／3,755例；其余495例在启动原生执行前失败。`numeric_predicates`漏接767→765源码逆变换，`metadata10`仍固定旧source.json哈希。原始失败ZIP、150,929个归档成员、153,351项输入绑定及其精确Git树已完整复核；当时Native CI的108步全部通过。失败记录原样保留于[767 CI核验](../validation/current-runtime767/ci-review.json)，不能记作完整PASS。

767修复只补充数值判定入口的精确源码逆链和元数据清单固定值；771入口依次撤销771→767→765。两个版本各自重新实际执行495例并通过物理回读，771记录见[入场修复实测](../validation/current-runtime771/admission-review.json)。485例数值判定保留11,244帧、42,645项断言、3,615份Owner快照、10份mapping快照及五次真实分配失败／恢复；10例metadata保留791帧、814项原断言及20个原生双流EOF观察。所有原数值、资源、错误、所有权和退出判定不变。

修复后PR49头为 `0d7eb2d6c2207ffed9e84b19f01d302491159b4e`，PR50头为 `0fdc0ab86b32725e92ca6ebc7997d8465f5c177c`；两棵树各自需要新的25组完整门和原件复核。前述四组Half专项原件仍绑定其执行时的完整输入快照（包括修复前numeric_predicates入口），只能在那份快照上复读；不能与后来的495例相加后声称最终771树已经完整通过。

完整官方artifact ZIP继续原样上传。额外传输作业逐个完整读取并核对官方ZIP的身份、长度和SHA-256，再保留精确200 MiB区间；重新拼接必须等于独立查询的官方原件。该机制已对PR48的504,716,668字节原ZIP完成完整三块重组验证，不删选成员、不改变原件内容。

## 已完成的765完整原件核验

[PR48](https://github.com/TrillionniumFoundation/HeptaTorch/pull/48) 合并于 `6a031f2d1a25e5c4ccbc1efc3dc13bd71175544e`。765映像的完整24组／3,845例及全部十项头检查通过，完整记录见[765 CI核验](../validation/current-runtime765/ci-review.json)。[Current HIA Runtime CI44743018163](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44743018163)和[Native HIA CI44743018190](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44743018190)均成功。

151,794个原始归档成员、152,506项输入绑定（106,105个唯一输入）、66份输入清单全部核对；1,931个仓库文件与精确Git blob／模式一致，34个工具链源码与固定Hepta提交一致。15个真实编译及1个链接全部退出成功并有双流EOF。

原ZIP为504,716,668字节，SHA-256 `10c00546bfb489bdb4c2ba56cfa43d5cb2f812644f76172075af691bd6189dd4`。内层XZ单流、全部文件字节和TAR尾部均验证。测试／合并树同为 `8e0c611401e4796aa697e14962f0db9b8bb490ae`。这一完整PASS只属于765，不转移到767或771。

## 仓库整理

默认生产路径保留一份完整当前HIA源码和有序分区。两批清理共退休157个文件、59,435,326字节，包括九个完整旧阶段目录、无消费者的摘要及六份旧资格计划；此前已退休269个过程文件、104,970,124字节，以及53个被替代源码文件。清理时检查了全树1,106个执行脚本及相关配置、清单共2,070个精确Git blob。PR50封存器仅移除四条旧计划路径，全部函数体和当时25组要求保持原样。原记录在[固定历史索引](history.md)，精确清单见[退休记录](../validation/current-runtime767/retirement.json)。

771的封存器另行接入完整26组及其真实共享参考和物理原件，并以771计划替换唯一的旧767计划（1,444字节）。累计退休427个过程文件、164,406,894字节，详细当前计划记录见[退休证明](../validation/current-runtime771/retirement.json)。仍被当前比较器、固定参考包或bootstrap读取的旧源码，需要迁移消费者并实际验证后才能删除。当前构建、26组和旧CI的具体依赖见[保留依赖清单](runtime-dependencies.md)。固定SHA的input4 README、许可证、上游通知及真实依赖保留。所有完整系统合同闭合并完成剩余消费者迁移后，才能完成最终整套替代和清理。

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
