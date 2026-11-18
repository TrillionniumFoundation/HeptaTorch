# R4 可微 view/reshape、wrapped scalar 与资源恢复

本轮在固定 HIA 运行时上推进具体 view/reshape 和 wrapped F64 scalar 行为，并补齐独立的资源失败与恢复验证。生产 Tensor、Storage、算术和自动微分继续由 HIA 原生程序执行；Python 负责源码生成、协议传输和证据读取。

**全项目仍为 HOLD。** 本轮没有把任何完整上游 schema 标记为闭合：固定 PyTorch 2.15.0 / commit 31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457 的 2,590 个 schema、688 个导数声明和 17 项系统门槛继续作为总分母。完整状态见 [全量验收账本](../compatibility/full-acceptance.json)，本轮机器可读记录见 [R4 profile](../compatibility/view-reshape-r4-profile.json)。

## 实现与固定构建

本轮依次从 R3 的 627 个函数构建 shape 阶段的 638 个函数，再构建 Scalar07b 的 658 个函数。最终程序具有 8,661 个块、30,980 条指令、13 个原生单元及 672 字节 descriptor。状态和 backing 格式分别是 HTVSHP02 和 HTOVSHP2。

| 原件 | SHA-256 |
| --- | --- |
| 最终 HIA 源码 | c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a |
| 最终 ELF | 0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74 |
| 原始构建报告 | b31a45a850f06196432b2786ad0a39b56088ccf6c7b100d0da3517a44de5ba48 |

便携构建入口会重新生成并编译两个阶段，核对各单元源码、对象文件、链接输入、成功标记和最终 ELF。旧的本地 PASS 不会自动赋予新 CI 构建资格。

## 已实际完成的有限验证

| 验证集合 | 实际范围 | 结果 |
| --- | --- | --- |
| view26、scalar32、原 view14、input4 | 同一个 Scalar07b 程序上的 76 个用例、82,503 帧、481,580 项原始检查 | PASS |
| 原始 76 用例的独立字节准入 | 645,052 项比较、2,050 个规范化快照、11,204 次物理 Storage 读取 | PASS |
| 三项标量资源测试 | 3 个用例、243 帧、771 项检查 | PASS |
| 反向传播私有 preserve 缓存分配失败与恢复 | 1 个用例、22 帧、23 个快照、46 项原生检查 | PASS |
| 上述反向传播原件的独立准入 | 4,546 项检查、1 对完整回滚、96,064 字节 RNG | PASS |
| CI 文件保全控制 | 7 项独立 fixture，覆盖成功、输入损坏、竞态、遍历和读取失败 | PASS |

76 用例总共比较了 66,165 个精确值字和 11,295,104 字节 RNG。原生报告的回滚字节分母是 2,143,906,608；独立读取器从已导出的相邻前后原件重新计算 2,204 对回滚，共 1,540,648,896 字节。另有 863 次拒绝没有导出紧邻的 before 状态，仍按原始源码和报告绑定保存；不把它们算入独立重新计算的分母。

反向传播资源用例先暖启动同一个 backward 入口，再利用实际空闲区间、公共槽位和进程地址空间限制，令私有 preserve 缓存的扩容失败。它检查完整 owner 的 4,369,040 字节、映射与 RNG 保持不变；恢复限制后用相同 backward 请求成功恢复，并确认已有梯度对象的身份不变。内部失败位置的归因来自冻结的 HIA 源码和实际分配几何；没有声称进行了指令级跟踪。数值一致性由上面的独立 76 用例验证。

## 原始证据

以下目录保留原始报告和归档身份；其中 FAIL、UNKNOWN、NOT_RUN 及早期版本不会被新的 PASS 覆盖。原生 76 用例的完整包已从默认工作树移除，可通过 [恢复索引](../validation/view-reshape-r4-native76/archive-location.json)取得经完整读回核对的原件。当前门禁使用保留全部原字节与成员模式的 [8 份参考文件](../validation/current-runtime76-reference/manifest.json)。

- [原生 76 用例](../validation/view-reshape-r4-native76/)
- [76 用例的独立字节准入](../validation/view-reshape-r4-raw-admission/)
- [三项标量资源证据](../validation/view-reshape-r4-resource3/)
- [反向传播私有缓存证据](../validation/view-reshape-r4-backward-resource/)
- [集成、源码检查与文件保全控制](../validation/view-reshape-r4-integration/)

原件归档核对单一 XZ 流的 EOF、TAR 成员集合、大小、SHA-256、POSIX 模式与完整尾部零块。外层安装为 Git 可携带的文件模式时，原来的成员模式继续由归档和清单保存。

## 可复现入口

设置固定工具链来源和 stack 构建目录后执行：

~~~sh
make differentiable-view-r4-scalar07b \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/stack-build

make test-differentiable-view-r4-native76 \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/stack-build

make test-differentiable-view-r4-resource3 \
  test-differentiable-view-r4-backward-resource \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/stack-build
~~~

两个资源入口都依赖新构建的 Scalar07b 成功标记和完整 scalar32 参考包。便携 wrapper 检查全部源码、13 个构建单元、实际编译/链接结果及当前构建报告，不接受仅复制旧报告作为本次执行。

CI 保留此前 116 个步骤，加入 10 个 R4 步骤，合计 126 个。新增的 11 份顶层报告在总索引之前生成。文件保全步骤使用 always()，尽量保存失败时仍可读取的原件，并把缺失或变化记录为 FAIL；它自己的两条实时日志由外层 artifact 单独保存，避免把尚在写入的日志当成稳定输入。

**此文冻结时，新集成提交的 hosted CI 尚未运行。** 便携资源入口已通过源码一致性检查；其实际 CI 执行状态需要由新提交对应的原始报告确认。CI 文件保全的 PASS 只说明保存和核验了文件，13 个单元构建和各用例的语义结果仍分别接受独立准入。

## 未闭合范围

本轮证据覆盖记录下来的有限 dtype、设备、布局、形状和自动微分路径。通用 dtype/device/layout、范围外形状、高阶自动微分与 transforms、完整 Python frontend/dispatch、加速器、分布式、compiler/export、序列化、数据管线及完整发布兼容性仍需继续实现和验收。不能用有限用例数替代上游 schema 数，不能把源码审查、文件保全或构建成功计成整个 PyTorch 的兼容闭合。
