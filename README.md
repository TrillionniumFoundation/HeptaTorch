# HeptaTorch

HeptaTorch 使用 **HIA 原生源码**实现张量数值、对象分配、梯度传播、随机数和参数更新，直接生成 Linux x86-64 静态可执行文件。唯一默认应用源码位于 [`tools/current_runtime/source/`](tools/current_runtime/source/)；[manifest](tools/current_runtime/source/manifest.json)给出完整源码、编译分区和映像身份。

**整个 PyTorch 的全量重写尚未完成。** 固定目标为 PyTorch **2.15.0**：**2,590 条原生声明、688 条导数声明、17 项完整系统验收要求**。完整声明和整项系统验收的闭合数均为 0。有限案例的实际结果及剩余实现见[当前审计](docs/current-audit.md)。

## 当前运行时

当前候选包含 **765 个核心函数、15 个独立编译单元**。每次构建使用固定 HIA ABI3 编译器重新生成全部应用对象，再由原生 HIA 链接器生成 ELF。运行时接口是有界原生二进制命令协议。

| 能力 | 当前实现 |
| --- | --- |
| CPU 张量 | F32 算术与一阶反向 AD；选定 Bool/I64 路径；受限 dense 无梯度 F64；F16/BF16 存储、复制、填充及 F32 转换；含 Half 操作数的 20 种公开 dtype 有序组合的无梯度乘法，F16/BF16 原生取负，以及固定 alpha=1 的无梯度加法／减法 |
| 对象与生命周期 | 独立 Storage/Version、Saved 张量、可微视图、公开梯度、所有权转移和失败事务回滚 |
| 逻辑与分类 | `any`、`all`、显式 Bool/F32 `sum`；`isnan`、`isinf`、`isfinite`、`isposinf`、`isneginf` |
| 训练 | 选定 NN 算子、反向传播、SGD/momentum、Adam/AdamW、CPU Generator 和部分随机分布 |
| Half 乘法 | rank 相关类型提升、广播、可达布局、原始右侧标量精度，以及固定 AVX2／单线程 NaN 位模式；结果具有独立所有权 |
| Half 加减法 | rank 相关类型提升、广播、可达布局及固定 AVX2／单线程位模式；保留 Bool 减法的实际错误和两条原始同帧 FE 加法 |
| Half 取负 | 保留 dtype、形状及输出布局；精确处理完整 32 元素向量组与标量尾段的不同 NaN 位模式 |
| BF16 转换 | 根据源和目标布局选择原生 NaN 转换路径，逐位匹配固定 Torch AVX2/AVX512 的标量／向量结果；有限值沿用原舍入实现 |
| 平台与规模 | Linux x86-64 CPU，rank 0–4；各命令独立检查形状、存储、工作量及资源限制 |

完整 Python/C++ 前端、通用 dispatcher、完整 dtype/layout/promotion、前向及高阶 AD、序列化、加速器、分布式、compile/export、数据管线及跨平台支持仍需完成。精确边界见[能力清单](compatibility/capabilities.json)及对应测试说明。

## 构建与测试

按[当前构建说明](docs/current-runtime.md)准备固定工具链和独立参考环境，然后运行：

```sh
make HEPTA_SOURCE_ROOT=/absolute/path/to/fixed-hepta-source \
     HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools \
     HEPTA_ENTROPY_BUILD=/absolute/path/to/current-entropy-tools

make test HEPTA_SOURCE_ROOT=/absolute/path/to/fixed-hepta-source \
          HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools \
          HEPTA_ENTROPY_BUILD=/absolute/path/to/current-entropy-tools \
          HEPTA_TORCH_PYTHON=/absolute/path/to/pinned-env/bin/python
```

产物为 `build/current-runtime/heptatorch`。`build.json` 绑定完整源码、全部新对象、工具、ELF、栈检查以及真实编译／链接退出和双流 EOF。

参考环境固定为 CPython 3.12、Linux x86-64、官方 CPU Torch 2.15.0 wheel，以及 [SHA 固定的 NumPy 依赖](compatibility/pytorch-reference-requirements.txt)。Python 组织构建和独立验收，PyTorch/NumPy 用于参考；原生应用无需这些宿主库，也无需 libm/libc。编译器首次 bootstrap 的宿主依赖见[工具链构建](docs/build.md)。

## 一个当前验收入口

`make test` 和 [Current HIA Runtime CI](.github/workflows/current-runtime.yml)共用 [`tests/qualify_current_runtime.py`](tests/qualify_current_runtime.py)。当前必需范围为 **24 组、3,845 个有限案例**：保留前 23 组／2,839 项输入及附加控制或明确验证的成功后继，增加 Half 加减法的 1,006 项对照。原 Half 入口的 239 个请求完整执行：22 个乘法、2 个取负、40 个单命令加减法和 2 个原始同帧 FE 加法要求实际数值及所有权验证，其余 173 个保留原拒绝行为。组表、额外控制和独立执行命令见[当前构建说明](docs/current-runtime.md)。

生产案例绑定本次冷构建的同一主映像；Generator 的 ABI／重排探针由当前对象另行链接并执行。每次源码或映像变化都需要新的完整运行。765 候选的完整组合状态以[当前审计](docs/current-audit.md)及源绑定执行记录为准。

`make acceptance-status` 检查完整目标账本的一致性。`make full-acceptance` 在全部合同尚未实现和验收前返回失败；两者均不替代原生测试。

## 仓库组织

默认生产目录保留一份完整当前源码及一组有序分区。已退休的过程源码和约 105 MB 的额外历史验证文件已移出当前目录；[历史索引](docs/history.md)指向固定 Git 提交。仍被原比较器、参考包或 bootstrap 读取的内容保留，后续按实际依赖迁移。

| 文档 | 内容 |
| --- | --- |
| [当前构建](docs/current-runtime.md) | 唯一源码、固定工具、参考环境及完整当前验收 |
| [当前审计](docs/current-audit.md) | 实际结果、修复及 17 项完整系统要求 |
| [继续执行](docs/continuation-plan.md) | 后续实现依赖及完成条件 |
| [上游审计](docs/upstream-audit.md) | 固定上游目标及完整声明分母 |
| [Half 乘法](docs/native-half-multiplication.md) | 公开 dtype 提升、固定内核语义、312 项验证及边界 |
| [Half 加减法](docs/native-half-addsub.md) | 固定内核语义、1,006 项验证、同帧所有权及原 239 请求继承 |
| [Half 取负](docs/native-half-negation.md) | NaN、布局、371 项验证及真实 OOM／恢复控制 |
| [BF16 转换](tests/bf16_copy/README.md) | 布局语义、逐位对照及适用 CPU 路径 |
| [历史记录](docs/history.md) | 已退休过程文件的固定历史位置 |

许可证与上游通知见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [`third_party/pytorch/`](third_party/pytorch/)。
