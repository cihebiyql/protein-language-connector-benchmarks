# Qwen底座与连接器对照：核验结果

已核验并重新计算 **0/24组**；队列状态：{"failed":0,"not_started":21,"running":3,"succeeded":0}。
队列终态收据：未观察到；这是进行中的快照。
所有分数来自规范终态已接受、当前文件SHA核验和模型无关重算一致的任务。失败／未启动／运行中／重验拒绝均保留，未作为模型负分。
此版本使用用户明确授权的node1共享策略，最多3卡；CPU/利用率/其他计算仅作观测，实际内存余量保留强制门槛。其他进程不会被终止或抢占。
固定划分、严格JSON解析、完整测试分母和验证NLL选点不变。数值头不回填语言输出；格式改善本身不能证明预测误差下降。
历史v3结果未填入本轮。初期负分或无提升不构成技术失败；本工具不运行或恢复训练。

## 主要比较

| 底座 | 场景 | 指标 | 同种子M0−M3均值±样本SD | 完整种子 | 判断 |
|---|---|---|---|---|---|
| qwen25_1p5b | new_peptide_known_target | LM_full_sample_mae | 未定义 ± 未定义 | 0/3 | 种子不足，不能判断稳定提升 |
| qwen25_1p5b | new_peptide_known_target | numeric_mae | 未定义 ± 未定义 | 0/3 | 种子不足，不能判断稳定提升 |
| qwen25_1p5b | new_peptide_new_target | LM_full_sample_mae | 未定义 ± 未定义 | 0/3 | 种子不足，不能判断稳定提升 |
| qwen25_1p5b | new_peptide_new_target | numeric_mae | 未定义 ± 未定义 | 0/3 | 种子不足，不能判断稳定提升 |
| qwen3_4b_2507 | new_peptide_known_target | LM_full_sample_mae | 未定义 ± 未定义 | 0/3 | 种子不足，不能判断稳定提升 |
| qwen3_4b_2507 | new_peptide_known_target | numeric_mae | 未定义 ± 未定义 | 0/3 | 种子不足，不能判断稳定提升 |
| qwen3_4b_2507 | new_peptide_new_target | LM_full_sample_mae | 未定义 ± 未定义 | 0/3 | 种子不足，不能判断稳定提升 |
| qwen3_4b_2507 | new_peptide_new_target | numeric_mae | 未定义 ± 未定义 | 0/3 | 种子不足，不能判断稳定提升 |

正的M0−M3表示完整连接器误差更低。仅3/3完整种子报告均值和样本SD；三个种子同向仍不能代替新数据确认或统计功效分析。

## 底座与连接器交互

| 场景 | 指标 | (M0−M3)4B−(M0−M3)1.5B | 完整四格种子 |
|---|---|---|---|
| new_peptide_known_target | LM_full_sample_mae | 未定义 ± 未定义 | 0/3 |
| new_peptide_known_target | numeric_mae | 未定义 ± 未定义 | 0/3 |
| new_peptide_new_target | LM_full_sample_mae | 未定义 ± 未定义 | 0/3 |
| new_peptide_new_target | numeric_mae | 未定义 ± 未定义 | 0/3 |

该值为正表示4B上的完整连接器误差收益更大。只有同一个种子的四格全部通过核验才计算该种子。底座主效应另列于JSON/CSV；跨底座LM参数、接口总参数和FLOPs不同。

## 全部24位置

| 底座 | 场景 | 接口 | 种子 | 队列／分析状态 | LM MAE | JSON有效率 | 数值MAE | 原因 |
|---|---|---|---|---|---|---|---|---|
| qwen25_1p5b | new_peptide_known_target | M0 | 17 | running/not_accepted | 未定义 | 未定义 | 未定义 |  |
| qwen25_1p5b | new_peptide_known_target | M3 | 17 | running/not_accepted | 未定义 | 未定义 | 未定义 |  |
| qwen3_4b_2507 | new_peptide_known_target | M0 | 17 | running/not_accepted | 未定义 | 未定义 | 未定义 |  |
| qwen3_4b_2507 | new_peptide_known_target | M3 | 17 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen25_1p5b | new_peptide_new_target | M0 | 17 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen25_1p5b | new_peptide_new_target | M3 | 17 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen3_4b_2507 | new_peptide_new_target | M0 | 17 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen3_4b_2507 | new_peptide_new_target | M3 | 17 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen25_1p5b | new_peptide_known_target | M0 | 42 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen25_1p5b | new_peptide_known_target | M3 | 42 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen3_4b_2507 | new_peptide_known_target | M0 | 42 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen3_4b_2507 | new_peptide_known_target | M3 | 42 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen25_1p5b | new_peptide_new_target | M0 | 42 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen25_1p5b | new_peptide_new_target | M3 | 42 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen3_4b_2507 | new_peptide_new_target | M0 | 42 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen3_4b_2507 | new_peptide_new_target | M3 | 42 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen25_1p5b | new_peptide_known_target | M0 | 2026 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen25_1p5b | new_peptide_known_target | M3 | 2026 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen3_4b_2507 | new_peptide_known_target | M0 | 2026 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen3_4b_2507 | new_peptide_known_target | M3 | 2026 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen25_1p5b | new_peptide_new_target | M0 | 2026 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen25_1p5b | new_peptide_new_target | M3 | 2026 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen3_4b_2507 | new_peptide_new_target | M0 | 2026 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |
| qwen3_4b_2507 | new_peptide_new_target | M3 | 2026 | not_started/not_accepted | 未定义 | 未定义 | 未定义 | fixed_preregistered_matrix |

## 多指标与计算预算

metrics.csv和formal_summary.json同时保留RMSE、含并列秩处理的Spearman、按靶点宏平均误差、JSON有效率、数值覆盖率及同权重C1 on/off。LM仅当整个人口全部有效时才报告完整人口RMSE／Spearman／宏平均；有效子集指标不会冒充完整测试指标。
每个通过任务保存真实训练／评估／进程组耗时、实际接口参数、显存allocated/reserved峰值、RSS和GPU分配。相同通信预算为58×128×float32＝29,696逻辑字节；CUDA上下文／库开销不包含在PyTorch allocator上限内。
C1 G_off为关闭连接器后每答案token NLL的平均增量；numeric_off_minus_on_MAE为关闭后的数值误差增量。它们衡量同权重分支依赖，不构成生物学因果／实验可比性证明。当前无独立生物donor或同条件配对金标准，P／H保持未定义。
temperature／pH／buffer／实际construct缺失时，只报告缺失数量，不虚构条件分层或可比配对。
绝对截止：2026-10-18T03:26:57+08:00。读取与分析不会延长或重开训练窗口。

| 底座 | 已接受组数 | 接口参数 | 已接受任务累计秒 | 最大reserved GiB | 最大RSS GiB |
|---|---|---|---|---|---|
| qwen25_1p5b | 0 | 未记录 | 未定义 | 未定义 | 未定义 |
| qwen3_4b_2507 | 0 | 未记录 | 未定义 | 未定义 | 未定义 |

计算资源表仅描述已接受任务，不把未完成任务估算为已运行，也不把并行任务累计秒当作自然耗时。
