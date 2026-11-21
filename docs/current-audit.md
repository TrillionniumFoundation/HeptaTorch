# 当前 HIA 重写审计

审计日期：2026-11-20。源码身份以 [source manifest](../tools/current_runtime/source/manifest.json) 为准；完整目标以 [full-acceptance.json](../compatibility/full-acceptance.json) 为准。

## 当前状态

[PR48](https://github.com/TrillionniumFoundation/HeptaTorch/pull/48) 已合并到 `6a031f2d1a25e5c4ccbc1efc3dc13bd71175544e`。765 映像的完整 **24 组／3,845 例**及全部十项头检查成功；原始归档完整读取、模式和哈希校验通过，合并树与实际测试树一致。完整记录见 [765 CI 核验](../validation/current-runtime765/ci-review.json)。

当前候选增加 I64／公开 F64 原生取负，包含 **767 个 HIA 核心函数、15 个编译单元**。它要求新的 **25 组／4,250 例**组合验收。新405例与原F64 48例已在同一767映像上实际执行并通过独立物理回放；此文生成时完整25组尚待CI执行，不能把453项分组成功当成统一PASS。

完整目标仍是 PyTorch v2.15.0、固定提交 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`：**2,590 条原生声明、688 条导数声明、17 项完整系统要求**。10项进行中、7项开放；完整声明与整项系统验收闭合数均为0，整体HOLD。

| 当前767身份 | SHA-256 |
| --- | --- |
| 完整源码 | `e0c8e0a3c3ee16d11cad8af6be1b52bd56295581e74bdcc65131f71a57540025` |
| 主ELF | `3ee1fae0e9d20d10dfd2645a3575d23357bcaca0dae03e5f0ff6bf1d4b4197a5` |
| Manifest | `32d826cfaaaaca910c7e2b209d977dd705a314f64267cfd0808f01f08d3865d1` |

## 新能力与实际执行

命令3／操作6接受公开I64和F64无梯度张量。I64执行64位环绕取负，INT64_MIN保持同一位模式；F64只翻转符号位，保留正负零、无穷及全部NaN payload。布局、分配、所有权与失败回滚全部由HIA实现，没有宿主数值回退。直接F64 as_strided、F64 AD及其他未实现的typed unary行为不在此次扩展中。

相对于765，只改变 `unary` 与 `command_argument` 两个原函数并追加两个helper；其余763个原函数逐字保留。完整767→765逆变换校验所有函数、原始字节及正向往返，继续接入765→761→758→742→740→736→700证明链。源码证明和实际执行各自保留。

| 同一767映像的实际验证 | 新wide_neg405 | 原F64 48例及成功后继 |
| --- | ---: | ---: |
| 原生子会话 | 405 | 48 |
| 完整协议帧 | 10,484 | 2,867 |
| 断言 | 37,314 | 15,421 |
| 完整Owner快照原件 | 2,959 | 579 |
| 比较元素 | 287,275个结果逐位一致 | 原1,798,174个输入及新增成功后继 |
| 真实OOM／相同请求恢复 | 2／2 | 2／2 |
| 实际执行及独立物理回放 | PASS | PASS |

新405参考使用固定官方CPU Torch 2.15.0、CPython3.12、NumPy2.3.5，实际AVX2／单线程。原F64入口的完整6,998字节配方及87个原始请求不变：86个拒绝保留，`unsupported-F64-neg`改为共享同一实际405参考的数值成功验证。完整输入字节、84字描述符、Owner表、RNG、输出独立性及两个原分配失败恢复仍被检查。其余84个拒绝期望逐字保留；之前两项malformed-Half适配维持原约束。

独立补充审查还覆盖2,400个元素、非连续／零stride／重叠布局、输入突变隔离、反复取负和真实OOM恢复；每个dtype600个输出的增长试验验证Owner表64→1024及数据池524,288→1,048,576。这些辅助NumPy检查明确独立于固定Torch正式验收。详见[取负契约](native-wide-negation.md)及[767范围记录](../validation/current-runtime767/native-review.json)。

## 完整组合门

协调器、CI与封存器均要求25组／4,250例。全部旧组保留，包括Half47的239请求：173拒绝、22乘法、2取负、40单命令加减、2原始同帧FE加法；原FE分配、临时对象退休及所有权谓词不变。Generator ABI／重排探针仍使用当前对象单独链接并实际执行。

F6448与wide_neg405必须绑定同一实际405参考及原始进程、解释器、全部输入／输出和双流EOF。全部新的应用对象必须冷编译，源码、工具、ELF、模式、引用清单和物理原件必须一致。缺组、超时、跳过、缺少EOF、输入变化或物理回放失败均阻止组合PASS。计划见[25组资格计划](../validation/current-runtime767/qualification-plan.json)。

## 已合并765的完整原件核验

[Current HIA Runtime CI 44743018163](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44743018163)及[Native HIA CI 44743018190](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44743018190)通过。151,794个归档成员、152,506项输入绑定（106,105个唯一输入）、66份输入清单全部核对；1,931个仓库文件与精确Git blob／模式一致，34个工具链源码与固定Hepta提交一致。15个真实编译及1个链接全部退出成功并有双流EOF。

原ZIP为504,716,668字节，SHA-256 `10c00546bfb489bdb4c2ba56cfa43d5cb2f812644f76172075af691bd6189dd4`。内层XZ单流、全部原文件字节和TAR尾部均验证。精确测试／合并树为 `8e0c611401e4796aa697e14962f0db9b8bb490ae`。这一PASS只属于765；不转移到767。

## 仓库整理

默认生产路径保留一份完整当前HIA源码和有序分区。此次清理再退休三个无活跃消费者的旧阶段目录，25个文件、18,078,927字节；此前已退休269个过程文件、104,970,124字节，以及53个被替代源码文件。全树1,100个执行脚本及相关配置、清单共2,062个精确Git blob已检查。原记录在[固定历史索引](history.md)，清单见[退休记录](../validation/current-runtime767/retirement.json)。

仍被当前比较器、固定参考包及bootstrap读取的旧源码不能在替代前删除。固定SHA的input4 README保留原字节，其历史路径由索引指向原提交。许可证、上游通知及真实依赖保留。迁移剩余消费者并完成所有系统合同后，才能完成用户要求的最终整套清理。

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
