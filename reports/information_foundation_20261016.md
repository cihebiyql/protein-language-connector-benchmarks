# C1/C2 基础诊断：2026-10-16

四种新连接器在 A1 已知靶和双新两种划分上的八项真实 C1/C2 均完成，全部标准归档获准。每项均按协议 v3 提取完整 TRAIN/VALIDATION/TEST，运行冻结参数推理，并验证 exit0、进程组消失。

已知靶划分 2062/147/152，双新 2063/431/442。连接器来自本轮各三轮、seed17 的验证集精选检查点；语言模型保持冻结。

## 实测结果

C1 的 G 是 correct/off 对同一原始 gold 的答案 token NLL 差，单位 nats/answer-token；ΔMSE 是相对 B 的测试 MSE 减少，单位 pKD²，正值表示误差降低。

| 场景 | 方法 | correct NLL | off NLL | G | B MSE | raw ΔMSE | message ΔMSE | receiver ΔMSE | worker秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 已知靶 | internal_cross_attention | 0.653916 | 1.211985 | 0.558069 | 3.047589 | 0.062795 | -0.166997 | -0.320128 | 134.4 |
| 已知靶 | qformer_style | 0.647621 | 1.211985 | 0.564364 | 3.047589 | 0.062795 | -0.050341 | -0.270914 | 139.0 |
| 已知靶 | input_mlp | 0.667195 | 1.211985 | 0.544790 | 3.047589 | 0.062795 | 0.137702 | -0.210961 | 142.0 |
| 已知靶 | constant | 0.656413 | 1.211985 | 0.555572 | 3.047589 | 0.062795 | 0.000000 | -0.057913 | 145.5 |
| 双新 | internal_cross_attention | 0.658278 | 1.203665 | 0.545387 | 3.169058 | -0.148403 | 0.023418 | -0.043077 | 271.0 |
| 双新 | qformer_style | 0.661190 | 1.203665 | 0.542475 | 3.169058 | -0.148403 | 0.087787 | -0.116038 | 300.1 |
| 双新 | input_mlp | 0.660284 | 1.203665 | 0.543382 | 3.169058 | -0.148403 | -0.052462 | -0.041686 | 268.0 |
| 双新 | constant | 0.653141 | 1.203665 | 0.550524 | 3.169058 | -0.148403 | 0.000000 | -0.123997 | 301.9 |

逐模型 MAE、RMSE、所选 alpha、完整五组验证得分、按物理 group_id 的 2000 次 bootstrap 95% 区间、测试预测和表示矩阵见 `all8_actual_summary.json` 与各 job 的 `artifacts/`。

三种数据相关连接器的六个 message ΔMSE 的组 bootstrap 95% 区间均跨零，当前实验的信息增益证据较弱。receiver 的八个 ΔMSE 点估计均为负。constant 与数据相关连接器获得接近的 C1 NLL 改善，因此完整 correct/off 改善应结合常量对照解释为答案格式和分布适配等因素的综合作用。

## 评价定义与解释

C2 仅输入提示词，不输入 gold 答案。raw 为肽和靶蛋白各自 ESM2 layer30 均值拼接的1280维；message为连接器实际输出（内部跨注意力取最后有效prompt token，prefix取memory token均值）；receiver为最后decoder block的最后有效prompt token状态。

条件 B=现有组成/化学特征52维+现有ContextEncoder148维，共200维。每个表示用固定seed42高斯投影到128维；标准化统计及探针参数仅从TRAIN拟合，alpha从相同五个候选按VALIDATION MSE选择。基线使用B+zero128，四个探针输入均328维，TEST保持固定。

这些ΔMSE表示在所设B条件下的线性标签可读出增量。B的文本哈希有损，receiver的变化也可能来自语言模型重新编码已有prompt/context；结合raw、message和constant比较，不能把receiver单独解释为蛋白来源信息或互信息。C1的NLL差也包含答案格式和数值序列分布的适配。

两场景严格独立donor均为0：匹配要求同精确靶序列、肽长度、允许化学/实验上下文，并要求不同肽序列和不同物理group_id。未知实验条件维持未知。完整correct/off C1与G已执行；P/H和独立错配项保持未定义，未放宽准入规则。

## 执行与可追溯证据

原监督器d524的首个作业在冻结合同阶段因缺dataset manifest字段被正确拒绝，worker启动0次；原路径/字节与旧wrapper完整保留。独立审查通过的58c49f metadata-only新wrapper明确将manifest_path/SHA绑定benchmark_lock.json，样本文件path/SHA保留；首个真实成功作业使用attempt2新路径，其余七项使用原计划路径。科学v3脚本始终为同一SHA。

27项CPU测试通过，包括source归档准入正负控；另有真实非模板frozen合同完整文件校验及缺/错manifest SHA负控。每项实际运行先freshprobe、冻结合同并验证可执行，再监督新PID/PGID；每项上限840秒、2线程、RSS12GiB、Torch11.5GiB，原始07:18起点及17:18截止保留。

主要文件：`scripts/foundation_20261016/information_diagnostics.py`、原监督器 `execute_information.py`、合同修正监督器 `execute_information_contractfix.py`、`tests/test_foundation_information.py`、`results/foundation_20261016/information_diagnostics/` 与本报告。

本轮写入与简化：复用现有连接器、数据校验、缓存、ContextEncoder和组成特征，不新增依赖、不更新backbone、不修改冻结训练算法。新增信息诊断、监督入口、CPU测试和实际结果。单种子基础测试的跨种子稳定性、未知实验条件和严格独立donor覆盖仍是后续验证边界。

CPU 事后完整性审计通过：八项所有归档 artifact SHA、精选检查点指纹、样本数与独立 group_id 数均一致，NPZ 中的 MAE/MSE/RMSE/ΔMSE 全部复算相符；同场景四种方法的 raw 与 B 测试预测逐位一致，constant message 恒定且与 B 探针预测相同。证据：`results/foundation_20261016/information_diagnostics/all8_numeric_integrity_audit.json`。

八项 worker 合计 1701.999 秒（28.37 分钟），逐项 134.38–301.91 秒。11:12（中国标准时间）复查全部 worker PGID、controller 及launcher均已退出，GPU compute进程为空；原d524监督器与首失败目录字节保持。交接证据：`gpu_release_after_information_receipt.json`。
