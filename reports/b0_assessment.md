# B0 真实数据工程试跑

本次成功跑通 Benz2022 T1 数据清洗、固定分组划分和两项训练集基线。结果为 engineering pilot，不能直接作为最终发表基准。

EV4 的 ELM Benchmarking Dataset 实读 3049 行，合并为 2724 条序列，其中 322 条多标签；候选为实际出现的 34 个 bait。论文的 35 个研究结构域不是此表的候选数。

连接分组 1573 个，实际序列数：训练 1914，验证 393，测试 387，EV8关联整组排除 30。70/15/15 为组数比例，非精确样本比例；seed=42。

同序列合并所有已记录阳性并保留原始行号；整母体 accession/gene 分组比仅合并重叠区段更保守。另用可含gap的序列一致性≥80%、双方残基覆盖≥90%阈值连通合组。

EV8 全35行（含示踪肽）保守用作排除锚点，展开打印的wt/mut序列。EV8并非已确认的24对核验清单；修饰、额外Y、坐标异常和同源家族映射仍需人工对齐。D2未联合清洗，因此当前不声称完整跨源holdout清洁。

| 划分 | 方法 | Recall@3 | MRR | 样本数 |
|---|---|---:|---:|---:|
| validation | train_frequency | 0.353690 | 0.335217 | 393 |
| validation | sequence_1nn | 0.549618 | 0.566598 | 393 |
| test | train_frequency | 0.334195 | 0.315419 | 387 |
| test | sequence_1nn | 0.507321 | 0.526817 | 387 |

频率基线每条训练序列总权重为1，多标签均分；1NN仅检索训练序列，邻居标签优先后用训练频率补全排名。全部34个候选保持固定，训练未见标签不会从测试候选中删掉。指标对独立序列等权，未记录配对不作为负例。

测试覆盖 31 个靶点，训练未见测试靶点为 []。逐样本完整排名及邻居ID已保存 results/b0_predictions.jsonl。

本次验证了真实数据的清洗—划分—预测—评估路径。ESM2/Qwen接口的可学习性及连接层位比较，需进一步用B1、六组连接、T2干预、T3配对、3种子和组级置信区间评估。

复现：`python scripts/prepare_benz_baseline.py`；回归测试：`python -m unittest discover -s tests -p test_benz_baseline.py`。输入Excel的SHA256保存在manifest；执行时间见 b0_metrics.json。
