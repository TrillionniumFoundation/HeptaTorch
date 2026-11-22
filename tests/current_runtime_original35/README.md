# 当前映像上的原始 35 个生命周期用例

唯一当前入口为 `tests/current_runtime35.py`。该入口使用共享进程监督器，顺序执行当前目录 `worker.py` 的三个阶段：原 native + live Torch 比较、原严格报告检查、原 4 个正例和 18 个负例。

`observer.py` 直接引用仓库中固定 SHA256 的原始 launcher、observer、checker 和 controls；不再复制另一套过程版本。所有数字输入、容差、训练轨迹、EOF 和关闭后的参数 Version 断言保留原实现。当前源和映像必须通过 `tools/current_runtime/qualification.py` 的完整冷构建检查。

适配仅包含当前源/映像/函数数量、实际 owner 格式标签和显式 ViewInfo 根引用：每个实际 live ViewInfo 的 base 计入 `node.600` 和 `node.208` 的精确方程以及 reset 可达闭包。相关 6 个 HIA 根引用函数和 20 个构造/布局函数必须匹配已经发布的独立源代码 pin，或满足下述 owner800 精确逆变换。原完整 84 字段结构检查、分配/梯度/回滚方程及控制失败顺序保留；历史字段 608/616/624/632 仍必须为零。每项代码变换均保留完整逆变换证明。

实际合格分母始终为 35 cases、6,344 frames、2,590,274 compared values、11 项原 Torch 来源检查、4 个正例和 18 个负例。原有 283 项 optimizer state 的绝对 Version 差异单独保留，不计为参数 Version 通过。`--prepare-only` 只用于 worker 的静态准备，必须为 0 个实际用例，不能产生运行合格结论。

原始 observer report 的字节保存在 `observer-report.before-build-binding.json`。当前执行只添加经独立检查的 cold-build / adapter 输入凭据；原始数字报告 `original-report.json` 不改写。所有执行和准备结果均保留全项目 `HOLD`。

精确 owner800（源 aab2e736…、映像 083a92ad…）仅接受三项已审核的构造变更。linear、layernorm、batchnorm_eval 的整段逆变换必须回到原 pin，两个新增辅助函数也须匹配。该审核不向其他源码或映像转移，也不改变历史构造计数。新线性辅助函数调整输出 Version，并同步相应 ViewInfo 版本；归一化辅助函数可以规范化或更换输出布局。原始核验的所有分配、Saved、EOF/参数 Version 和 4+18 控制均须重新执行。实际 35 份 owner 日志另须包含 48 条线性指令、零条两类归一化指令；该范围不证明非连续归一化的分配语义。
