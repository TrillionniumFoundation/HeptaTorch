# 当前构建与验收依赖

本页记录 native771 候选的实际入口；其来源和清理基线为PR50提交 `f8ad6df1d0d27776b7efb2c62db1356189523f39`和仍需保留的历史输入。当前源码、15 个编译单元和预期映像由 [source/manifest.json](../tools/current_runtime/source/manifest.json) 固定；26 组、5,446 个案例的入口由 [qualify_current_runtime.py](../tests/qualify_current_runtime.py) 定义。新增或迁移消费者时，应同时更新本页。

依赖判断结合两类证据：该提交全部已跟踪执行代码与相关清单的路径检查，以及 PR48 原始完整 24 组 CI 归档中的声明输入和仓库成员。后者证明当次765运行的输入关系；767新增的 `wide_neg`、F64 neg successor，以及771新增的 `half_div` 和旧Half请求的除法成功后继另按当前执行代码检查。旧运行结果不能转作771的完整26组资格结果，也不能替代系统调用级文件访问追踪。

## 默认原生构建

[Makefile](../Makefile) 的默认目标 `all` 进入 `current-runtime`。构建读取当前 `tools/current_runtime/source/` 的完整源码、单元文件及 manifest，交给固定 HIA 编译器与 entropy session linker；[build.py](../tools/current_runtime/build.py) 检查源码身份、分区、工具身份、栈预算和最终映像。[entropy linker](../tools/current_runtime/entropy_linker/README.md) 的工具构建使用固定 Hepta 上游作者输入，并核对产物字节。

此路径不重建历史 HeptaTorch 应用实现。默认 CI 中的 compiler-capacity 历史见证来自单独检出的固定提交 `31d5b1483714b32254730ba56787884c5c76e2fb`，其路径是 `.toolchain/heptatorch-legacy/src/heptatorch.hepta`；该依赖与当前仓库的 `src/heptatorch.hepta` 分开核对。

## 26组验收的保留输入

表中列出跨目录依赖边。各套件自身的 `.py`、`.json`、固定包和 source-transition 输入仍由其原始 manifest 与绑定代码完整约束；迁移时不能只搬动表中某一个压缩包。

| 当前消费者 | 必须保留的参考与旧输入 | 读取入口 |
| --- | --- | --- |
| `bool27`、`integral24`、`original76`、`sum32`、`sum_f32_17` | `validation/current-runtime-baseline`、`current-runtime-references`、`current-runtime76-reference`、`sum-bool27-reference`、`sum-f32-reference17`；`continuation/20261117/evidence` 下两份 `root-sum-integral-*-reviewed-intent-*.json` | [fixtures](../tests/current_runtime_fixtures.py)、[historical](../tests/current_runtime_historical/run.py)、[legacy76](../tests/current_runtime_legacy/run.py)、[sum32](../tests/current_runtime_legacy/sum32.py)、[F32 Sum](../tests/sum_f32_review/fusion.py) |
| `current35`、`base270`、`live32`、`cat4` | 当前固定 Torch 安装与实际执行参考；270和32原始方法分别位于 `validation/current-runtime270-source` 与 `current-runtime32-source`。Cat4另读保留的四份 packet fixtures | [lifecycle35](../tests/current_runtime35.py)、[base270](../tests/current_runtime270.py)、[live32](../tests/current_runtime_live32.py)、[cat4](../tests/current_runtime_cat4/run.py) |
| `logical147` | `validation/logical-reductions-reference` 的当前及原138例参考、执行记录和 manifest | [logical qualifier](../tests/logical_reductions/qualify.py) |
| `resource7` | `validation/current-runtime76-reference`、`validation/view-reshape-r4-native76/archive-location.json`、`tools/differentiable_view_recovered_r3/profile.json`，以及资源包装器固定的完整原始输入 | [resource sources](../tests/current_runtime_resources/sources.json)、[resource qualifier](../tests/current_runtime_resources/qualify.py) |
| `asstrided16` | `validation/tensor-to-original16-recovery` 的原探针、16例参考、安装来源及完整 package manifest | [AsStrided sources](../tests/current_runtime_asstrided/sources.json)、[AsStrided qualifier](../tests/current_runtime_asstrided/qualify.py) |
| `metadata10` | `validation/cat-metadata10-reference` 的完整参考包，以及固定的原方法、失败控制和源身份 | [metadata sources](../tests/current_runtime_metadata10/source.json)、[metadata worker](../tests/current_runtime_metadata10/worker.py) |
| `generator131` | `validation/current-generator-candidate`、`current-generator-original26`、`current-generator131-source`；固定 Torch 的实际参考和当前 entropy 工具构建记录 | [Generator fixtures](../tests/current_runtime_generator_fixtures.py)、[coordinator](../tests/qualify_current_runtime.py) |
| `public_f64_logical192`、`f64_48` | `validation/public-f64-logical192-reference`、`current-f64-48-source`、`current-f64-48-reference`；原F64 neg请求使用本次 `wide_neg` 的实际Torch参考验证显式successor | [public F64](../tests/current_runtime_public_f64_logical192.py)、[F64 fixtures](../tests/current_runtime_f64_fixtures.py) |
| `half47`、`half_logical264`、`half_transactions12` | `validation/current-half47-source`、`current-half47-reference`、`half-logical264-reference`；原拒绝请求保留身份，显式算术successor读取本次乘法、取负和加减参考 | [Half fixtures](../tests/current_runtime_half_fixtures.py)、[logical reference](../tests/half_logical/reference.py)、[logical runner](../tests/half_logical/run.py) |
| `numeric_predicates`、`bf16_copy`、`half_mul`、`half_neg`、`half_addsub`、`wide_neg`、`half_div` | 各套件新运行的固定Torch参考、完整原生请求/响应和当前源码逆变换输入；乘法组还读本次加减参考；Half47、乘法312、加减1006和除法1196共享同一1196参考；F64与Half原始读回共享对应新参考 | [coordinator](../tests/qualify_current_runtime.py) 与各 `tests/<suite>/run.py`、`reference.py` |

PR48原始归档中，部分共享包由总协调器声明为输入，随后解包给子套件使用。因此，归档里只出现协调器的绑定不表示该包没有子套件消费者。目录扫描还必须检查动态 `glob`/`rglob`、包内源文件清单、源码逆变换和完整包校验。

## 尚不能移除的旧实现与其他CI

`src/heptatorch.hepta` 仍由 [native workflow](../.github/workflows/native.yml) 的分区、编译及源身份检查读取，[Makefile](../Makefile) 的 `legacy`、monolithic、session和owned-tensor目标也读取该文件。[F64 oracle](../tests/f64_oracle.py) 还直接检查其规范源码字节。退休整个旧实现前，必须把这些真实消费者迁移到当前运行时，保留它们原有案例、失败控制及来源验证，再移除旧入口。

独立的 private-node observer CI 仍通过 [restore_native_inputs.py](../tests/private_node_accounting/restore_native_inputs.py) 恢复三个固定旧映像：

| Profile | 必须保留的归档 |
| --- | --- |
| 355 | `validation/gradient-objects-root/root-cold-qualification-01.tar.xz` |
| 428 | `validation/gradient-adam-distributions-root/root-integrated-cold-05.tar.xz` |
| 446 | `validation/retained-gradients-root/root-cold-and-strict-01.tar.xz` |

`validation/adam-distributions-root` 已无活跃读取，可按[历史记录](history.md)退休；`validation/gradient-adam-distributions-root` 是另一个仍在读取的目录。两个名称的子串重合不能作为删除判断。

## 已完成的最小迁移

[CI封存器](../tools/seal_current_ci_evidence.py) 只额外封存当前771的 `qualification-plan.json`。736、740、742、758、761、765旧计划保存在固定Git历史；被771替换的767计划也按[精确退休记录](../validation/current-runtime771/retirement.json)保存在固定提交f8ad6df。742/758/761/765原来仅作为额外归档成员收录，PR48声明输入和输入清单对这六份计划的引用均为零。迁移仅移除四个 `REPOSITORY_PATHS` 常量；函数体、当时全部25组要求、必要输出、工具链和参考来源检查保持一致。

旧计划退休不影响仍被能力账本引用的真实结果，例如 `validation/current-runtime736/main-ci-44363462524.json`、`validation/current-runtime742/bf16-review.json` 和 `validation/current-runtime765/ci-review.json`。原始CI归档、完整输入包和源码逆变换输入也继续保持原字节。

清理状态以当前消费者为准。[完整验收目标](../compatibility/full-acceptance.json) 的2,590条原生算子、688条导数及17项系统契约没有因删除历史文件而缩小；当前公共运行时仍需继续补齐该目标。
