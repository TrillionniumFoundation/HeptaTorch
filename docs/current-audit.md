# 当前 HIA 重写审计

## 结论与当前状态

当前默认应用由一份完整 HIA 源码直接冷编译为 Linux x86-64 原生 ELF。当前清单包含 **700 个核心函数、14 个编译单元、672 字节 Tensor 描述符**；源码、ELF、完整函数统计和有序分区以 [`tools/current_runtime/source/manifest.json`](../tools/current_runtime/source/manifest.json)为唯一身份依据。

**全量 PyTorch 重写仍为 HOLD。** 固定目标及分母来自[上游锁文件](../compatibility/upstream-lock.json)：PyTorch 2.15.0、2,590 条原生声明、688 条导数声明。完整 schema 资格为 0；[17 项系统门禁](../compatibility/full-acceptance.json)保持 10 项 IN_PROGRESS、7 项 OPEN、0 项 CLOSED。受限算子、案例或函数数量均不能换算为全量完成率。

当前 cat 修复源码的完整冷构建已实际通过，共享 admission 检查原始输入、完整分区、全部新对象、link request、ELF、栈预算及编译／链接进程的退出和双流 EOF。原始四项 cat/stack 回归与十项独立 Torch 元数据对照均已在此新映像实际通过。新十三组、697 例的完整统一运行仍待完成；前一映像的成功不转移到这里。

已发布前版的 [Current runtime CI 44294832902](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44294832902) 已实际通过全部十一组、683 例；[Native HIA CI 44294832918](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44294832918) 的 116 个步骤成功完成。PR40 原头 `82ac3b65f5d91fcc28122cf1760c3995bec77c44` 的十项适用检查全部成功，2026-11-18 12:41:37 UTC 已合并为 main `584b95d7f6615f27e95618bc61215df5633c4ca4`。实际 CI checkout `dda27a2ee4554085a326a90dae3248e8fff700e2` 与合并提交的 tree 同为 `cdb4879b58e1d88456ef0ad173ac628f8124b6c9`。原 683 资格报告 SHA 为 `343dce42de965d04c61af6429ec7d7844da4a449aae59666403071635f4bc7ab`；其原始构建与测试均绑定前版源码 `0c1c0dd3acf0285a887a2766c1308dfd992acf3fe51849dcd98749bb64a00f3b`。

该原始 CI archive 已逐字检查 60,077 项文件清单，再分别重放 600 项原纯读回 API 和 83 项明确路径映射后的 API。原绝对 CI 路径未完整重建；补充的 Python 实体与两个 Git 文件单列，未冒充原 archive 成员。这个证据边界不改变原 CI 的实际结果，也不授予新 cat 映像资格。

更早的首轮 9 组、660 例实际返回 FAIL：8 组通过，`current35` 的后续检查因跨 Python 版本的 AST 身份差异被拒绝。原失败与只读诊断保留，未被重新标成成功。

更早的第二次完整执行通过全部 10 组、667 个案例和附加控制。所有组均在固定 CPython 3.12 下重新执行，使用相同 HIA 源、ELF 和实际冷构建回执；全部 1,699 个仓库输入在运行前后逐字节、逐模式复核不变。外层实际 exit 0、stdout 6,464 字节、stderr 0 字节，双流 EOF 完整，无超时或强制清理。完整资格报告 SHA 为 `aa168047fc9a0392707f3c1c08b8ab4cb88c2bc16e1732c4c37c2a3994cc14c0`，其[原验证包](../validation/current-runtime-20261118/manifest.json)保持原字节与范围。

## GitHub 起点与并行变更

本轮起点是 main [`b22622b371d79478864cb80154b1782e708a5a81`](https://github.com/TrillionniumFoundation/HeptaTorch/commit/b22622b371d79478864cb80154b1782e708a5a81)，对应 tree `dbbbc4b0ad0a95fd01b7ae369532bf7a10c3f0f7`。当时默认完整源码已有 700 个核心函数，但当前映像的统一门禁只有四组、159 个案例。树清单包含 5,747 个 blob，其中 236 个 `.hepta`、993 个 `.py`；文件数量说明仓库同时承载运行时、工具、比较器和历史证明，不能用语言后缀比例判断原生实现完成度。

已读取的基线 CI 中，[Current HIA Runtime 44255480944](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44255480944) 和 [Native HIA 44255480937](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44255480937)均在该 main 身份上成功。它们只证明当时所执行的源与范围。

2026-11-18 09:13 UTC 的分支／PR 复核发现并行 [PR38](https://github.com/TrillionniumFoundation/HeptaTorch/pull/38)：`codex/hia-current701-native-20261118` 的头为 `329facee729bf8c07348db8a4f128510e078df15`，相对上述 main ahead 1 / behind 0，共 68 个变更文件。此前 38 个非 main 分支仍是 main 的祖先；新分支包含未合并工作，不能作为冗余分支删除。

PR38 已合并为 main `33b4b542e99551846ddeaf57f1d7c17a3453935e`，其全部 11 个检查已成功。当前工作树保留 PR38 的 Bool/I64→F32 求和、共享 F32 累加次序、空 `cat` 梯度和固定参考迁移，并融合本轮 any/all、扩展 Bool sum、错误优先级及原资源 7。随后读取并融合 [PR39](https://github.com/TrillionniumFoundation/HeptaTorch/pull/39) 的固定头 `5e21e85a58a3b5d25d8d105fbb8ef000d31e6411`：保留其 live32 原始帧／owner 读回、压缩参考包及 16 次旧 CI 调用退休，不覆盖并行工作。

## 一个当前运行时与构建链

默认构建读取 [`tools/current_runtime/source/`](../tools/current_runtime/source/)，由固定 HIA ABI3 编译器生成全部 14 个新对象，再由固定 HIA 原生链接器生成 ELF。它不需要旧应用 ELF 或逐批应用补丁链。工具链最初的 bootstrap 与编译器证据仍是独立依赖，详见[构建说明](current-runtime.md)和[工具链文档](build.md)。

[`qualification.py`](../tools/current_runtime/qualification.py)把测试与当前 manifest、本次实际 build report 绑定。构建的 `PASS_NATIVE_BUILD_ONLY` 不授予运行资格；案例、资源错误、对象生命周期和固定参考比较必须另行执行。Python 负责宿主组织与独立观察，张量计算、对象分配、随机数、梯度和更新由 HIA 执行。

对外接口仍是受限二进制协议。CPU F32 是主要数值／AD 路径；Bool、I64 仅有明确支持的类型和混合路径，F64 只保留已有 rank-zero wrapped Scalar。rank 上限为 4；一般构造与各操作的工作上限分别检查，不能把一个命令的上限推广到整个库。命令 115 / 118 / 119 限定最多 262,144 个逻辑元素，输入／输出 body 各最多 1 MiB。完整边界集中在[当前能力清单](../compatibility/capabilities.json)。

## 本轮原生功能与真实错误修正

### 逻辑归约和 Bool sum

命令 118 `any`、119 `all` 与命令 115 的显式 Bool sum 共享 HIA 归约几何、真值转换和结果分配。输入覆盖 Bool、I64、F32 及已有 wrapped F64 Scalar；±0 为 False，非零次正规数、NaN 和无穷为 True。结果有独立 Tensor/Storage/Version，不持有输入的梯度或 Saved 边。

`any` / `all` 的空轴元组表示不归约；`sum` 的空轴元组表示归约全部维度。标量允许轴 0 或 -1；负轴先规范化再检查重复；空归约使用各操作的恒等值。浮点 `any` / `all` 可以接受 requires_grad 输入并返回 detached Bool，浮点 Bool sum 保留明确的梯度拒绝。

审查发现首次 Bool sum 将梯度拒绝放在轴验证之前，与固定 Torch 的重复／越界轴错误优先级不符。本轮已将轴验证移到梯度拒绝之前，并新增三份 requires_grad + 非法轴配方，分别用于三种操作。独立参考总数由 138 扩展为 147，前 138 份实际观察保持原字节。完整协议、参考生成与 18 个包／回滚和 2 个分配／恢复控制见[逻辑归约说明](../tests/logical_reductions/README.md)。

### F32 cast 后求和与空 cat

PR38 的 `sf_sum_f32` 逐元素把 Bool/I64 转为 F32，再执行 F32 归约；它不会先按整数求和再最终 cast。私有转换对象在成功和失败时都释放。其首次实际 17 案例报告中的 2 个数值失败及 cast／sum 分离诊断保持原身份和结果。

共享求和内核的修正来自固定参考 wheel 的 SumKernel 注册／累加次序证据；按该路径使用 8 个 F32 lane。仓库说明中的 AVX512 硬件能力与 AVX2 内核次序是不同事实，不能把来源说明当作本次硬件观测。最终数值资格仍取决于当前映像的独立实际比较。

PR38 的空 `cat` 修改把 rank 与输出不一致的空父输入改为 owned incoming mode，保留成功返回值 1。本次继续审计发现，这还没有完整执行固定 Torch 对 rank-one 空输入使用的 zeros 构造语义：实际十项对照中，七项出现梯度 Version 差异，其余数值、形状、步长、offset 与梯度存在性相同。原版本差异报告 SHA `44ece48a56f75ea99f353594ef684bede750d2cb5c835fbd68a002040fb86e68` 保持 FAIL。

当前修复在 `incoming_describe_cat` 的 cat 分支内，针对 rank 1 且 numel 0 的输入选择已有零工厂，执行其原有两次 checked Version increment；没有直接补写版本计数。其它 rank 保留原 narrow/alias 路径，stack 分支保持原字节。实际 62,207 次重复输入的梯度 Version 为 62,208；独立的 rank-two 空布局继续保留第二个梯度的非零 storage offset。新十项报告 `025237173c936d4ed13c76cd27a7c4752b701d7fb9fb936aecf4157d807e5bab` 全部匹配、差异为零。

另两项运行时修正补上 `tensor_join` 的分配失败检查，并移除空输出轴的无效前缀扫描与 `concat_copy` 普通元数据读取中的 parent-mask 查找。源证明限定三个完整函数变更，其余 697 个核心函数完整不变；先精确逆回前版 700 源，再执行原逻辑归约、PR38、AVX2 和基线的全部来源证明。旧崩溃、两次超时、旧 overflow 拒绝与原比较期望均保留；扩大本实现 arena 的成功控制不冒充 PyTorch 的容量契约。

## 当前十三组 697 个案例的必需门禁

[`tests/qualify_current_runtime.py`](../tests/qualify_current_runtime.py)是本地和 [CI](../.github/workflows/current-runtime.yml)共用的当前入口。现要求全部十三组、697 例在同一源、ELF 和 build receipt 上实际执行。旧十一组 683 已在前版 CI 统一通过，本次另增加 cat4 与 metadata10。下面保留各自实际执行的归属；新映像的完整 697 尚待运行，额外控制不合并为新的 schema 数量。

| 组 | 案例 | 额外分母或关键约束 | 已有实际记录归属 |
| --- | ---: | --- | --- |
| `bool27` | 27 | 原 owner/RNG、dtype／入口观察、真实分配失败与恢复 | PASS |
| `original76` | 76 | 原 26 + 32 + 14 + 4；82,503 帧、481,580 断言 | PASS |
| `integral24` | 24 | 260 原始事件、全宽整数、溢出与生命周期 | PASS |
| `sum_f32_17` | 17 | cast 后求和、dtype／axis／入口、两阶段 OOM | PASS |
| `current35` | 35 | 6,344 帧、2,590,274 数值、11 来源条件、4 正／18 负控制 | PASS |
| `base270` | 270 | 十个原组、270 原生进程、284 帧、14 真实前缀／拒绝对 | PASS |
| `sum32` | 32 | 626 事件、64,415 帧、460,075 断言 | PASS |
| `live32` | 32 | 2,132 帧、17,455 精确 F32 字、1,137,600 RNG 状态字节 | PASS |
| `logical147` | 147 | 49 配方 × 3 操作；18 包／回滚控制和 2 分配／恢复控制 | PASS |
| `resource7` | 7 | 原 R4 3 + backward 1 + R3 3；378 帧、1,060 检查、3 次实际限额恢复 | PASS |
| `asstrided16` | 16 | 原 copy4／reduction4／lanes2／branches6；1,920 帧、10,272 检查、完整 owner／RNG、16 个退出／32 次 EOF | 前版统一 683 PASS |
| `cat4` | 4 | 三份原始成功字节与一份扩容控制；原始夹具、完整 owner、真实退出和双 EOF | 新映像独立 PASS |
| `metadata10` | 10 | 固定 Torch 形状、stride、offset、Version、dtype、原始数据和 grad presence；逐帧及完整 owner | 新映像独立 PASS |

表中前十一组的 PASS 均已有前版同一次完整 683 CI 记录；后两组来自本次新映像的独立执行，不能拼接成完整 697 PASS。`current35`、`base270`、`live32` 使用固定 live Torch；八个原参考组、cat4 与 metadata10 分别消费各自固定观察或原协议夹具。固定环境必须满足 11 项来源条件；原源码、输入、实际回执和全部进程 EOF 都要重新绑定。仅准备、跳过、超时、输入变化或任何未经解释的失败均不能得到总 PASS。

旧拒绝控制的合法契约扩展采用双层记录：原比较器继续产生原 FAIL，当前资格器只允许明确列出的扩展差异，并由独立新控制验证真实结果。不会修改原始期望、改写原报告或把历史失败直接重分类。

### PR39 融合后的独立回归

统一 667 使用迁移前的原始输入快照。此后迁移参考包存放及 source／build 身份适配，并加强 live32 原始读回；HIA 源、ELF 和实际冷构建不变。所有涉及的原比较器和数值期望保持。原资源 7 已在新包装上重新实跑通过；原 76、Sum32 与加强后的 live32 合计 140 例也已全部实际通过，报告 SHA 为 `0c21f10e1836c9282e9d8dece81d55b14652b1368575c53368691fa9b35396b3`。外层 exit 0、stdout 4,076 字节、stderr 0 字节，双流 EOF 完整，无超时、取消或输入变化。

加强后的 live32 逐个读取 2,132 对实际帧、4,264 份原始／解码 owner、17,455 个精确 F32 字和 1,137,600 字节 RNG 状态，并核实 32 个原生退出／64 次 EOF、11 项来源条件及原部分写入／拒绝控制。该独立回归明确记录 `full_667_executed=false`。原 667 的报告、输入清单和成功范围保持不变，未把后续测试源变更追记到旧运行。

### 原 AsStrided 16 的当前映像迁移

[`tests/current_runtime_asstrided/qualify.py`](../tests/current_runtime_asstrided/qualify.py)保留原 copy4、reduction4、lanes2、branches6 的完整比较器与 owner 观察器。八份完整源码逆证明只允许四处 build admission 替换和四个观察器的当前身份标签替换；原数值比较、完整帧、finish 函数和系统调用不变。

最终 source02 已在新目录实际运行全部 16 例：1,920 帧、10,272 项检查、21,892 个精确数值字、1,860,608 字节 RNG 状态，以及 176 份完整 owner 压缩归档全部通过。实际报告 SHA 为 `8300788a5927e0862b0e3ff6561ce878f8509b2f7276933bbb913e077dabf003`。四个 worker 和 16 个互异的原生进程均真实退出成功，完整双流 EOF、源码绑定及输入前后身份齐全；安装后的独立纯读回入口再次核验全部原件。

退出观察以只读方式绑定原 finish 的精确代码对象，并要求正常返回及真实关闭证据。EOF 证明限定于已审核的四套原调用图，其中没有提前关闭输出的路径。错误 build receipt、缺组、错误案例分母和未执行 native 均不能获准。原有恢复构建仍有其他消费者，故只退休已被新必需组替代的旧 16 调用及两个专用上传路径。

### 首轮 660 执行与解释器修正

首轮完整协调器实际尝试全部九组，输入复查未变；`bool27`、`original76`、`integral24`、`sum_f32_17`、`base270`、`sum32`、`live32`、`logical147` 均通过。`current35` 的 native/live 阶段已经通过原 35 个案例，但 strict checker 和 copied controls 在执行比较前被身份入场拒绝，因此该组和总报告保持 FAIL。

原因是 native/live 阶段使用固定 CPython 3.12，而后续阶段继承系统 CPython 3.14。同一源码的 `ast.dump` 在这两个版本产生不同的 7 个 AST 摘要字段。只读诊断未执行新的 native 或 Torch 用例，确认 3.12 的完整适配身份与原实际报告一致，3.14 仅上述 7 个字段不同。原 660 总报告 SHA 为 `b822874a60ff322577e49ee098f8ce73e2c0d90cc86246a71ed88fecace46c47`。

修正是使用固定 CPython 3.12 运行协调器及全部阶段；所有原数值、owner、来源与身份比较继续保持。`make test` 现通过 `HEPTA_TORCH_PYTHON` 同时选择协调器和 Torch 解释器。修正后的 667 门禁已完整重跑并通过；首轮 FAIL 与两版本的只读诊断仍以原字节保留。

### 原 35 与完整 owner

唯一当前入口为 [`tests/current_runtime35.py`](../tests/current_runtime35.py)，复用固定 SHA 的原 launcher、checker 和 controls。所有原数字、训练轨迹、EOF 和关闭后的参数 Version 比较保留；原有 283 项 optimizer-state 绝对 Version 差异继续单独记录，参数 Version 断言不放宽。

首次迁移时 7 个案例因旧 fixed35 方程未计入已有 ViewInfo 根引用而失败。当前适配器通过已发布的 6 个 HIA 根引用函数和 20 个构造／布局函数 pin，明确把每个实际 ViewInfo base 纳入 `600/208` 精确方程及 reset 闭包。原完整 84 字段结构检查继续执行；历史字段 608/616/624/632 仍必须为零。三处变换均保留完整逆向证明，没有快照裁剪或残差计数。

该适配器在第二次统一 667 中已通过全部 35、6,344 帧、2,590,274 数值及原 4 正／18 负控制。首次迁移的 7 FAIL、跨解释器失败和最终成功均分别保留。当前 observer 报告增加 cold-build receipt 前的完整字节另存，原数字报告不改写。实现详见[原 35 适配说明](../tests/current_runtime_original35/README.md)。

### 原 270 与 live32

原 270 的固定 archive 保留 runner、四个比较器、wire 和 transport。当前入口只明确重绑实际 owner 身份／标签和共享 cold-build admission；原 Torch 输入、seed、容差、十组案例函数不变。14 个拒绝案例在同一实际进程中执行成功前缀和末尾拒绝，保留两个真实响应及完整 owner 回滚，不能合成旧错误记录。

原 live32 是独立必需组，与 F32 `sum32` 分别保留 32 个案例分母。PR39 强化后的入口保留原 Audit、transport taps、`check_actual`、帧解码及 owner 读回函数的完整字节；显式、可逆地绑定当前 source／ELF、700 个函数及完整冷构建。它重新读取 2,132 对实际请求／响应和 4,264 份原始／解码的完整 84Q owner 快照，检查 32 个真实子进程退出与 64 次 EOF，并保留原 partial-write／refusal 控制。

### 原资源 7 配方与实际恢复

[`tests/current_runtime_resources/qualify.py`](../tests/current_runtime_resources/qualify.py)执行原 R4 resource3、backward private cache1 和 R3 preserve3。它保留原入口／相对 token、Saved／临时对象、完整 84Q owner 方程、完整 RNG、分配预算、错误状态和恢复比较。固定原包包含全部外层 manifest，两个原参考分别保留 11 项来源检查。三个 worker 启动七个真实原生子进程，逐帧绑定 378 个请求／响应与 1,060 项原检查；三个 OOM 配方真实设置并恢复子进程 RLIMIT_AS，再验证原命令恢复成功。

R4 的旧 dispatcher 完整保留于命名后的函数，新前导分支对原配方使用的命令直接委托。R3 的十个分配证明函数中八个逐字未变，另外两个保留完整原函数体及只改变函数名的逆证明；实际全部 live node 均为 F32、op0 或 op44，新分支在该路径不可达。R3 配方只替换这一源码入场证明，其余原数值预算和断言逐字保留。

独立重新冷构建后的最终资源运行通过 7/7，原请求／响应、双 EOF、完整源码逆转换和三次限额恢复均已复核。实际 build receipt 由 CLI 显式传入；旧 receipt 控制在启动前拒绝，避免把本地构建路径写死为 CI 身份。失败记录中的 native 执行事实单独保留为 true／false／unknown，不能把运行后失败误记成未运行。资源 7 随后在统一 667 和压缩参考包后的独立运行中分别全量通过。后一次原报告 SHA 为 `a45f05345dce0299365c36560bddcba0c4135d7450be4a099146eb1d46c5817d`；仍不关闭完整系统门禁。详情见[资源适配说明](../tests/current_runtime_resources/README.md)。

## 已验证的清理范围

从真实 [`owners.hepta`](../tools/current_runtime/source/units/owners.hepta) 的 9 个 external 根出发，遍历 HIA 完整调用闭包：`clone_state`、`entry`、`frame_valid`、`getw`、`init_state`、`object_size`、`reply_error`、`so_uget`、`state_header`。PR38 的 701 个核心函数中只有以下 7 个不可达，合计 15,647 字节；本轮已删除。

| 删除的函数 | 原函数字节 |
| --- | ---: |
| `growth_capacity` | 546 |
| `init` | 1,087 |
| `integer_view_end` | 1,427 |
| `integer_view_root_valid` | 4,565 |
| `mark_concat` | 1,019 |
| `to_as_strided_direct_copy` | 3,133 |
| `to_view_root_valid` | 3,870 |

PR40 阶段相对 PR38 保留 689 个完整函数体、修改 5 个、新增 6 个共享／逻辑归约函数并删除上述 7 个。那一阶段完整保留的 `incoming_describe_cat` 在本次继续修正；其余 SumKernel 与 F32 cast 修正保持。当前 cat 变更通过三个完整函数的逆转换精确重建 PR40 源，保留原删除证明与真实入口调用闭包。静态来源证明不代替新映像的运行回归。

README 和能力清单已移除旧 143 函数默认配置及多代 profile 的冗长过程元数据，只描述当前源、边界与十三组要求。旧元数据可从 Git 历史追溯，删除清单和字节差异单独记录。已移除未使用的旧 current35 源包、冗余 profile／fixture-relocation 元数据和未使用恢复入口。

原 76 参考 archive 从 12,561,956 字节压缩为 131,848 字节，恰好保留 8 个实际必需文件、33,559,322 字节原内容及原 0600 模式。删除当前树中的旧完整 archive 前，逐成员与旧包直接比较，并保留[完整旧证据的固定 Git／封存恢复索引](../validation/view-reshape-r4-native76/archive-location.json)。PR39 的 16 次重复 CI 调用、本轮 R3 重复调用及原 AsStrided16 调用合计 18 次已退休。R4 的另外两次资源调用仍由 `tools/seal_r4_ci_evidence.py` 消费；其封包依赖迁移前保留，不能只删调用导致原证据链缺项。

仍活跃的 fixture 保留：固定编译器 bootstrap／来源证明、461 对旧协议夹具、原始比较器与参考 archive、Generator24、尚未迁移的 dtype／优化器／资源门禁，以及 F64 验证器仍消费的旧源证明。原 live32 的当前入口已经迁移，其他历史 CI 消费者仍按依赖逐项处理。相同字节的两条路径也可能分别被固定 manifest 引用，不能仅按重复 hash 删除。

退役顺序是识别真实消费者，将必要原输入集中为经 SHA／mode 绑定的 fixture，迁移当前读者并真实执行全部原契约，再移除旧路径。删除原报告、减少上游声明或关闭门禁不构成退役。

## 全量目标仍需闭合的 17 项

下表以[系统账本](../compatibility/full-acceptance.json)为准；已有受限基础不等于完成所有 dtype、布局、设备、重载和错误契约。

| 门禁 | 状态 | 仍需完成 |
| --- | --- | --- |
| `module-abi` | IN_PROGRESS | 通用公开 Tensor ABI、外部资源来源和不限于当前图的模块／资源组合 |
| `storage-lifetime` | IN_PROGRESS | 全部别名／生命周期、外部与设备分配、并发和重启持久性 |
| `dtype-layout` | IN_PROGRESS | 完整类型提升／转换、complex／quantized／sparse／nested 布局及设备变体 |
| `native-schemas` | IN_PROGRESS | 2,590 条固定原生声明的完整契约和可执行资格 |
| `views-mutation` | IN_PROGRESS | 全布局、rank、dtype 和设备上的 alias／mutation 规则 |
| `autodiff-transforms` | IN_PROGRESS | 高阶、create_graph、前向 AD、hooks、transforms 及全部可微重载 |
| `frontend-dispatch` | OPEN | 完整前端、dispatcher、重载、上下文和公开 API |
| `neural-networks` | IN_PROGRESS | 全部模块、状态、训练／评估模式及算子／后端变体 |
| `optimizers` | IN_PROGRESS | 全优化器／scheduler、state_dict、模式及 dtype／device 行为 |
| `rng-determinism` | IN_PROGRESS | 全分布／设备、共享／并发 Generator 和确定性控制 |
| `serialization` | OPEN | checkpoint、格式兼容、存储别名、故障与分布式恢复 |
| `accelerators` | OPEN | CUDA／ROCm／MPS／XPU 的分配、传输、stream／event、内核和错误行为 |
| `distributed` | OPEN | rendezvous、process group、collectives、DDP／FSDP／DTensor 和失败语义 |
| `compiler-export` | OPEN | graph capture、符号 shape、compile／export、转换和产物兼容 |
| `data-pipelines` | OPEN | dataset／dataloader、worker、multiprocessing、采样和取消 |
| `platform-packaging` | IN_PROGRESS | 其余平台／架构、安装、扩展、profiling 接口和可复现工具链 |
| `final-integration` | OPEN | 全应用／上游验收、资源／性能／错误审查和旧路径最终退役 |

`make acceptance-status` 只验证账本一致性。`make full-acceptance` 在上述目标未闭合时保持失败。十三组 697 即使全部通过，也只能授予当前受限配置的统一运行资格；全量完成标志仍保持 false。
