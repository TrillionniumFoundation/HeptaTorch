# 当前 HIA 映像的原始 AsStrided16 资格检查

本入口在一个完整当前原生构建上执行原来的 `copy4`、`reduction4`、
`lanes2`、`branches6` 四组测试。固定 Torch 参考值、原数值比较器、
Tensor/Storage 身份、`_base`、Version、getter 顺序、梯度边、保留梯度、
所有 RNG 字节、owner 审计与原异常条件均保持不变。

| 原组 | 配方 | 帧 | 比较 | 精确浮点字 | RNG 字节 |
| --- | ---: | ---: | ---: | ---: | ---: |
| copy4 | 4 | 480 | 2,568 | 384 | 465,152 |
| reduction4 | 4 | 480 | 2,568 | 384 | 465,152 |
| lanes2 | 2 | 240 | 1,284 | 552 | 232,576 |
| branches6 | 6 | 720 | 3,852 | 20,572 | 697,728 |
| 合计 | 16 | 1,920 | 10,272 | 21,892 | 1,860,608 |

这些是固定的有限配方，不是 16 个上游 schema 已完整闭合。
`full_pytorch_rewrite_complete=false`、`schema_closure=false` 与全项目
`HOLD` 均保持。原测试不作弱引用回收保证；这里也不增加这项保证。

## 原始输入与允许变化

原始比较器继续来自 `tests/tensor_to_original16_review/`，协议和进程
transport 来自 `tests/retained_nonleaf_review/`。参考包复用仓库已有的
`validation/tensor-to-original16-recovery/`：四个 `reference/` JSON、
四个 `original-probes/` 源文件及 `README.md`、`package-manifest.json`、
`installation-custody.json`，共 11 个固定文件。
`sources.json` 保留它们的 SHA256、字节数、原模式与四组全部分母。
每组仍须满足原来的全部 11 项 Torch 来源检查。

运行时只生成两种派生源：每个 reader 原来的 528 字节构建入场段改为
调用共享 `tools/current_runtime/qualification.py`；observer 只把旧
state/backing tag 改为当前格式。每份完整派生文件都有精确逆变换，
任何额外源码差异都会被拒绝。四个 owner 布局函数与原 614 函数映像
中的定义逐字相同；实际运行仍须绑定当前 700 函数、14 单元的完整构建。
该入口不执行旧应用 ELF，也不读取完整旧业务 HIA 源码作为运行依赖。

当前 source/ELF 必须满足固定审阅 profile；具体 `build.json` 的 SHA
由实际路径中的新构建决定，可以显式传入，不写死某次机器上的收据。
编译对象、链接输入、工具、栈检查、真实进程退出及 EOF 均通过共享
完整构建入场重新验证。

## 真实执行与读回

每组使用独立受监督 Python worker；每个原始配方创建自己的原生子进程。
原 `OwnerSession.finish` 和所有 syscall 均不替换。worker 只用
`sys.setprofile` 观察原方法返回帧，并绑定真正的 `.__code__` 对象。
只有 `RETURN_VALUE` / `RETURN_CONST` 正常返回、回调完整执行、原报告
PASS、真实 ledger 退出 0、实际 `error` 与 `extra` 都是空 bytes、文件
描述符与归档流均关闭时，才承认该子进程的双 EOF。

该 EOF 证明仅用于这里固定的四个 reader 和 observer 闭包：这些路径
不会提前调用 `close_output`。它不能直接推广为任意 reader 的 EOF 证明。
回调异常不会覆盖原异常；它会使独立外层资格检查失败。

外层完整读回全部 1,920 个请求/响应帧、record 顺序和边界，以及每个
原报告引用的压缩 owner 档案，逐个验证解压后的完整字节、SHA256 和大小。
它同时重新检查源码逆变换、所有原分母、四个 worker 的退出与双 EOF、
十六个原生子进程的真实关闭观察及运行前后输入不变性。

执行事实与资格结果分开记录：已观察到原生响应仍记为执行过，即使
之后失败；未启动为 `false`；中断而没有充分证据时为 `null`。
固定参考重放不会被记为重新执行了 Torch。历史 native PASS 不转授。

## 入口

在仓库根目录执行：

```sh
python3 -I -S -B tests/current_runtime_asstrided/qualify.py \
  --repository "$PWD" \
  --build-directory build/current-runtime \
  --output build/current-asstrided16
```

可加 `--expected-build-sha256` 指定本次实际收据 SHA。
`--prepare-only` 只检查入场、源码逆变换和固定输入，不启动 native/Torch。
所有输出目录必须为新目录，失败结果不覆盖。

`qualify_asstrided16(report, build_sha, directory, inputs)` 是协调器的只读
入口。它独立重复完整物理读回，不创建进程、不改结果、不沿用旧 PASS。
实际执行目录和当前安装目录可以不同，但四个执行源文件必须逐字相同，
实际 repository/build/reference 路径及 worker 参数必须与原收据一致。
