# 原 AsStrided16 在当前 HIA 运行时上的验收

本入口保留四套原始 reader、完整 owner 观察和全部 16 个配方，绑定当前
**767 函数、15 编译单元**的 HIA 主映像。统一执行状态见[当前审计](../../docs/current-audit.md)。
源码准备、历史结果和其他映像的 PASS 均不授予当前运行资格。

## 固定配方与分母

原数值、shape、stride、offset、Version、getter 顺序、梯度边、保留梯度、
所有 RNG 字节、owner 审计和原异常条件均保持不变。

| 原组 | 配方 | 帧 | 比较 | 精确浮点字 | RNG 字节 |
| --- | ---: | ---: | ---: | ---: | ---: |
| copy4 | 4 | 480 | 2,568 | 384 | 465,152 |
| reduction4 | 4 | 480 | 2,568 | 384 | 465,152 |
| lanes2 | 2 | 240 | 1,284 | 552 | 232,576 |
| branches6 | 6 | 720 | 3,852 | 20,572 | 697,728 |
| 合计 | 16 | 1,920 | 10,272 | 21,892 | 1,860,608 |

这些是有限配方，不能计作 16 条完整上游 schema。
`full_pytorch_rewrite_complete=false`、`schema_closure=false` 与全项目
`HOLD` 保持。原测试没有弱引用回收保证，本入口也不增加该保证。

## 原始输入与当前源码入场

原比较器来自 `tests/tensor_to_original16_review/`，协议和 transport 来自
`tests/retained_nonleaf_review/`。固定参考仍为
`validation/tensor-to-original16-recovery/` 中四个 reference、四个 probe
及三个原始说明/manifest/custody 文件，共 11 项；原字节、权限和所有
11 项 Torch 来源检查保留。`sources.json` 同时绑定 13 项当前所需仓库文件。

每个 reader 只把原 528 字节构建入场段替换为共享 admission；observer
只更新 state/backing tag。每份完整派生文件都有精确逆变换。四个 owner
布局函数继续与原 614 函数映像中的定义逐字相同，当前 owner tag 函数
也单独核验。原数值断言、清理流程、owner 方程和 finish 实现均不替换。

当前 [manifest](../../tools/current_runtime/source/manifest.json) 与实际
`build.json` 必须匹配。共享 admission 核对全部源码/对象/工具、精确链接
输入、栈预算及编译/链接进程的退出和双 EOF；不会制造旧 profile 的回执。

当前 [wide-neg source inverse](../../tools/current_runtime/wide_neg_source_transition.py)
把完整 767 源码精确还原为 765，核对两个改变函数、两个新增 helper、
763 个未改原函数体和完整正向回建。随后依次还原 Half add/sub 的 765、
Half neg 的 761 和 Half mul 的 758 源码，接入原 742 证明。
[BF16 source inverse](../../tools/current_runtime/bf16_source_transition.py)
将完整 742 源码逐字还原为 740：两个原函数变化、两个新增 helper 与其余
738 个未改原函数体均分别检查，完整正向回建必须一致。随后接续原
[predicate source inverse](../../tools/current_runtime/predicate_source_transition.py)
把完整 740 函数源码精确逆回 Half736：除 `entry` 两条指令外，原 735 个
函数体全部不变。原 `to_command` 仍是实际执行体；完整 120–124 guard
和六参数原序 delegate block 单独验证，原命令不会进入新增谓词分支。
该源码关系不取代实际 AsStrided 运行，也不把旧源码当成当前物理源码。

## 实际执行与独立读回

每组由独立受监督 Python worker 执行，每个配方创建真实原生子进程。
原 `OwnerSession.finish` 和 syscall 不替换；worker 仅用
`sys.setprofile` 观察原方法正常返回，并绑定实际 `.__code__` 对象。
必须同时观察正常返回、完整回调、原报告 PASS、native exit0、空的
`error`/`extra`、描述符和归档流关闭，才能承认双 EOF。

该证明限定于这四个原 reader/observer 的完整调用图，其中不存在提前
`close_output` 路径；不能推广为任意 reader 的 EOF 保证。回调错误使
外层资格失败，不覆盖原异常。

纯读回重新检查全部 1,920 个请求/响应帧、record 边界和顺序、原报告引用
的完整压缩 owner 档案、解压字节/SHA/大小、所有原分母、四个 worker
和 16 个 native 子进程的退出/EOF，以及运行前后全部输入不变性。
已观察到执行仍记为 true，即使之后失败；未启动为 false；证据不足为 null。
固定参考重放不记作新执行 Torch。

## 入口与结果 schema

```sh
python3 -I -S -B tests/current_runtime_asstrided/qualify.py \
  --repository "$PWD" \
  --build-directory build/current-runtime \
  --output build/current-asstrided16
```

使用与总协调器相同的固定 Python。可用 `--expected-build-sha256` 指定
实际 cold-build receipt SHA。输出目录必须为新目录；`--prepare-only`
只运行入场、固定输入和源码逆证明，返回 `PREPARED_NOT_RUN`，不启动
native/Torch。

完整执行的 schema 是 `heptatorch.current767.asstrided16.v1`，成功状态
为 `PASS_CURRENT_ASSTRIDED16`。协调器通过
`qualify_asstrided16(report, build_sha, directory, inputs)` 独立重复完整
物理读回；该函数不创建进程、不改结果、不继承旧 PASS。实际执行与安装
目录可以不同，但四个执行源文件、repository/build/reference 路径和
worker 参数必须与原收据逐字匹配。
