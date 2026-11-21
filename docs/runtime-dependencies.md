# 当前构建与验收依赖

本页记录当前775函数typed-ReLU后继的实际入口；历史来源使用固定提交 `f8ad6df1d0d27776b7efb2c62db1356189523f39`，本次来源移除的审计基线为PR52提交 `be44d3c77c1e3da1e2c859eaf6581ac98ffbfa4d`。当前源码、15个编译单元和预期映像由 [source/manifest.json](../tools/current_runtime/source/manifest.json) 固定；29组、13,851个案例的入口由 [qualify_current_runtime.py](../tests/qualify_current_runtime.py) 定义。新增或迁移消费者时，应同时更新本页。

依赖判断结合两类证据：固定树全部已跟踪执行代码与相关清单的路径检查，以及PR48/49原始完整CI归档中的声明输入和仓库成员。历史归档只证明各自运行的输入关系；其后增加的 `half_div`、`public_f64_arithmetic`、`same_f64_arithmetic`、`typed_relu` 及原请求的显式成功后继另按当前执行代码检查。旧运行结果不能转作当前完整29组的资格结果，也不能替代系统调用级文件访问追踪。

## 默认原生构建

[Makefile](../Makefile) 的默认目标 `all` 进入 `current-runtime`。构建读取当前 `tools/current_runtime/source/` 的完整源码、单元文件及 manifest，交给固定 HIA 编译器与 entropy session linker；[build.py](../tools/current_runtime/build.py) 检查源码身份、分区、工具身份、栈预算和最终映像。[entropy linker](../tools/current_runtime/entropy_linker/README.md) 的工具构建使用固定 Hepta 上游作者输入，并核对产物字节。

此路径不重建历史 HeptaTorch 应用实现。默认 CI 中的 compiler-capacity 历史见证来自单独检出的固定提交 `31d5b1483714b32254730ba56787884c5c76e2fb`，其路径是 `.toolchain/heptatorch-legacy/src/heptatorch.hepta`；该依赖与当前仓库的 `src/heptatorch.hepta` 分开核对。

## 29组验收的保留输入

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
| `public_f64_logical192`、`f64_48` | `validation/public-f64-logical192-reference`、`current-f64-48-source`、`current-f64-48-reference`；原F64 neg请求使用本次 `wide_neg` 实际405参考；22个混合算术单命令与1个原始FE事务使用同一份新实际2541参考验证显式successor | [public F64](../tests/current_runtime_public_f64_logical192.py)、[F64 fixtures](../tests/current_runtime_f64_fixtures.py) |
| `half47`、`half_logical264`、`half_transactions12` | `validation/current-half47-source`、`current-half47-reference`、`half-logical264-reference`；原拒绝请求保留身份，显式算术successor读取本次乘法、取负和加减参考 | [Half fixtures](../tests/current_runtime_half_fixtures.py)、[logical reference](../tests/half_logical/reference.py)、[logical runner](../tests/half_logical/run.py) |
| `numeric_predicates`、`bf16_copy`、`half_mul`、`half_neg`、`half_addsub`、`wide_neg`、`half_div`、`public_f64_arithmetic`、`same_f64_arithmetic`、`typed_relu` | 各套件新运行的固定Torch参考、完整原生请求/响应和当前源码逆变换输入；乘法组还读本次加减参考；Half47、乘法312、加减1006和除法1196共享同一1196参考；F64与Half原始读回共享对应新参考；same-F64的5088例使用本次实际参考并保留原NaN payload广播失败记录；ReLU776与两套取负共用本次实际776参考，四条原控制请求执行显式数值后继 | [coordinator](../tests/qualify_current_runtime.py) 与各 `tests/<suite>/run.py`、`reference.py` |

PR48原始归档中，部分共享包由总协调器声明为输入，随后解包给子套件使用。因此，归档里只出现协调器的绑定不表示该包没有子套件消费者。目录扫描还必须检查动态 `glob`/`rglob`、包内源文件清单、源码逆变换和完整包校验。

## 保留的历史实现验收与其他CI

历史143函数源码由 [native workflow](../.github/workflows/native.yml) 的分区、编译及源身份检查、[Makefile](../Makefile) 的显式旧目标和独立F64/RNG流水线继续使用。本次来源迁移从当前Git树移除481,516字节的 `src/heptatorch.hepta`；这些消费者按原路径恢复固定历史字节，继续验证原实现。**整个旧实现的语义迁移尚未完成**；原始案例、失败报告、源身份、映像、所有权和资源判定均继续保留。

### 历史143来源迁移第二阶段

[来源清单](../tools/legacy_source143/manifest.json) 固定提交 `f8ad6df1d0d27776b7efb2c62db1356189523f39`、Git blob `2e994b4c15f5c6760d65aa555c0bca6c0bd88a56`、SHA-256、143个完整函数和原文件模式。[恢复器](../tools/legacy_source143/restore.py) 保持第一阶段原始字节；缺源码时从该固定HTTPS地址或已核对缓存恢复，拒绝覆盖不同字节。恢复后的文件仅作为忽略跟踪的历史输入使用，不成为当前canonical源码。

| 活动消费者 | 缺源码时的准备入口 |
| --- | --- |
| Native旧分区、单体/模块比较、session、owned-module及其报告核验 | 工作流在第一个消费者之前恢复历史源码；原数值和资源步骤保留 |
| F64原oracle及其复用函数逐体核验 | F64工作流先恢复；原oracle、严格失败结果及RNE/sqrt核验保持原字节 |
| RNG的profile驱动读取与原许可证检查 | RNG工作流先恢复；8个复用函数中6个仍从恢复后的143源码读取 |
| `legacy`、`monolithic`、`session`、`session-monolithic`、`owned-tensors` | Make源码prerequisite调用 `legacy-source`；versions与tensor-object构建继承这条依赖 |
| `test-storage-objects` 最后的源摘要核验 | 显式增加 `legacy-source` prerequisite，避免仅运行storage目标时遗漏恢复 |
| 独立旧作者工具与四个 `check_*_reports.py` | 直接调用前执行 `make legacy-source`，随后运行原始命令；冻结helper和profile不改路径或哈希 |

例如，独立核对旧session来源和RNG复用函数时使用：

```sh
make legacy-source
python3 -I -S -B tests/check_session_reports.py sources
python3 -I -S -B tests/check_rng_custody.py
```

`owned_modules/tests/review_runtime.py` 使用 `('heptatorch', 'src')` 动态拼接路径，也由上述Native准备入口覆盖。`owned_tensors/authoring.py` 的调用者有两类：早期owned-Tensor和tensor-object构建读取恢复后的143源码；后续backward/gradient作者在独立输出目录生成自己的 `src/heptatorch.hepta` 上下文，这些生成内容不能替换成143字节。

Native、F64和RNG工作流现在要求Git索引不再跟踪旧源码，而且源码和缓存目录在准备步骤开始时都不存在。因此本次CI必须实际完成固定HTTPS冷恢复，不能以existing-source或预热缓存分支代替。随后仍运行原Native107步骤、F64/RNG各8步骤和16项来源恢复/拒绝控制，保存完整来源收据；它们通过之前，不应宣称本次精确树的来源迁移已验证完成。[第一阶段记录](../validation/current-runtime773/legacy-source143-review.json)继续作为原始历史证据保留。

当前runtime工作流不增加143恢复依赖，并在开始构建以及完整资格运行前后断言旧源码缺失。`base270` 的原runner显式使用实际当前构建的 `runtime.hepta`；源码逆变换使用其固定canonical/归档输入。编译器扩容的31d5b历史见证仍位于单独检出的 `.toolchain/heptatorch-legacy`，两份历史来源不能混用。新精确树仍必须执行全部当前组及所有历史门，静态路径检查和来源恢复控制不能替代这些原生运行。

独立的 private-node observer CI 仍通过 [restore_native_inputs.py](../tests/private_node_accounting/restore_native_inputs.py) 恢复三个固定旧映像：

| Profile | 必须保留的归档 |
| --- | --- |
| 355 | `validation/gradient-objects-root/root-cold-qualification-01.tar.xz` |
| 428 | `validation/gradient-adam-distributions-root/root-integrated-cold-05.tar.xz` |
| 446 | `validation/retained-gradients-root/root-cold-and-strict-01.tar.xz` |

`validation/adam-distributions-root` 已无活跃读取，可按[历史记录](history.md)退休；`validation/gradient-adam-distributions-root` 是另一个仍在读取的目录。两个名称的子串重合不能作为删除判断。

## 已完成的最小迁移

[CI封存器](../tools/seal_current_ci_evidence.py) 只额外封存当前775后继的[唯一活动计划](../validation/current-runtime775/qualification-plan.json)。被替换的 same-F64 计划（3,964 字节）按原字节保存在固定 PR53 提交 `83ed6b41ea67aa37182d23159ec0cb436e632dac`，见[本次退休记录](../validation/current-runtime775/retirement.json)。更早被替换的 mixed773 计划按原字节保存在固定PR52提交 `be44d3c77c1e3da1e2c859eaf6581ac98ffbfa4d` 的 `validation/current-runtime773/qualification-plan.json`；该计划移除不删除任何实际参考包。736、740、742、758、761、765旧计划继续保存在固定Git历史；被771替换的767计划按[原退休记录](../validation/current-runtime771/retirement.json)保存在固定提交f8ad6df。773曾替换771计划并退休旧分区生成摘要，[对应记录](../validation/current-runtime773/retirement.json)保留固定87702d6历史身份。早期计划的封存选择清理未改函数体、必要输出、工具链或参考来源检查；当前29组的完整要求继续由实际协调器与封存器强制执行。

旧计划退休不影响仍被能力账本引用的真实结果，例如 `validation/current-runtime736/main-ci-44363462524.json`、`validation/current-runtime742/bf16-review.json` 和 `validation/current-runtime765/ci-review.json`。原始CI归档、完整输入包和源码逆变换输入也继续保持原字节。

清理状态以当前消费者为准。[完整验收目标](../compatibility/full-acceptance.json) 的2,590条原生算子、688条导数及17项系统契约没有因删除历史文件而缩小；当前公共运行时仍需继续补齐该目标。
