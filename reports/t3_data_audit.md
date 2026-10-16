# T3 真实肽对数据审计

日期：2026-10-15（Asia/Shanghai）。生成器：`scripts/prepare_t3_real.py`。所有统计来自本地原始数据实读；本报告只报告数据准备结果，模型成绩由训练脚本另存。

元数据 v3 修正：G1 的突变系列仅解析完整基因名之后的完整突变字段，修复 BCL11A/KMT2A/KMT2D/SCNN1B 内部字符误匹配以及无空格 `BUB1S492F`。原始 v2 G1 文件和 manifest 保存在 `*_preseriesfix_v2.*`；标签、序列、靶点、母体 group 和 split 逐字段验证保持一致。SCNN1B 两突变正确拆分，BUB1 同一 S492F 跨靶点测量正确合并，主 17 行仍对应 13 个变体系列。manifest 新增生成器 SHA-256 及生成文件 SHA-256，供所有 runner 闭合追溯。

## 可直接训练和测试的数据

Kliche2024 的 201 个候选组中，77 组具有可定位关系及筛选计数方向；排除 G1 的 12 个整母体组，得到原始 65 对。独立复核发现其中 6 对至少一端输入库覆盖为零，追加双端 `peptide_coverage_counts > 0` 质量门后，最终保留 **59 对真实 16 肽**。这些标签属于真实筛选计数方向的弱监督；EV3 端点 p 值的比较对象尚无公开索引，具体肽对的统计显著性未获确认。原 65 对及原始分组清单保存在 `*_precoverage65.*` 快照中，6 条质量门排除记录保留完整证据。原有 group/split 保持，未重划或替换同组肽对。

| 数据 | 样本数 | 分组数 | A 更强 | B 更强 |
|---|---:|---:|---:|---:|
| D2 train | 34 | 26 | 14 | 20 |
| D2 val | 16 | 6 | 11 | 5 |
| D2 test | 9 | 6 | 2 | 7 |
| G1 可评分测量 | 17 | 12 母体、13 突变系列 | 13 | 4 |
| G1 来源序列一致敏感性子集 | 14 | 10 母体 | 11 | 3 |

`sequence_a` 固定为实验库 WT 肽，`sequence_b` 为对应变体；`label` 为相对筛选丰度更高的 A/B。训练可对 A/B 顺序作同步交换增强，独立样本数保持不变。D2 `target_id` 保留 EV6 的完整 bait 标识，`target_info` 为 EV2 的基因、结构域及构建体残基范围；所有方法使用相同靶点信息。EV2 `SMURF2_WW__Q9HAU4_298_330` 与 EV3/6 `SMURF2_WW_3_Q9HAU4_298_330` 按 accession/构建体坐标对齐。G1 的 KEAP1 标识统一为同一 EV2 构建体 `Keap1_KELCH__Q14145_321_609`；KPNA4 major/minor 分别在 ID 附加 `|pocket=major/minor`，测量口袋身份明确区分，D2 未指定口袋的构建体保持独立身份。

主文件：

- `data/processed/t3_d2_pairs.jsonl`：最终 59 对，含固定 split、原始来源行及证据。
- `data/processed/t3_d2_input_coverage_exclusions.jsonl`：6 条覆盖质量门排除记录；原 65 对位于 `t3_d2_pairs_precoverage65.jsonl`。
- `data/processed/t3_d2_candidate_audit.jsonl`：全部 201 组、每个关系的接受或排除原因。
- `data/processed/t3_split_manifest.json`：来源 SHA-256、统计、固定分组清单及泄漏检查。
- `data/processed/t3_g1_records.jsonl`：Table 1 全部 29 个原始测量行，含小变化、界限与身份冲突。
- `data/processed/t3_g1_direction_pairs.jsonl`：17 个可评分测量行。
- `data/processed/t3_g1_clean_direction_pairs.jsonl`：同一 17 行，均通过既有 T1 训练暴露检查。
- `data/processed/t3_g1_source_consistent_direction_pairs.jsonl`：14 行 Table1/EV8/附录序列一致敏感性子集。
- `data/processed/t3_d2_clean_test_pairs.jsonl`：D2 test 中未在 T1 训练暴露的 5 对/4 组。

## 具体肽对的标签依据

EV6 `Legend` 将 `number_significant_peptide_pairs` 关联到 EV3 的 **p < 0.01**，但 EV3 未给出端点 p 值对应的另一比较对象。全库 433 条双端可定位关系中，276 对 p 值相同、157 对不同；WT 对应多变体时尤其无法确定单端 p 指向哪一变体。因此最终证据字段为 `endpoint_reported_p_values` 和 `single_pair_significance_verified=false`，原 `pair_p_value` 字段已移除。既有端点 p 阈值及相等条件仅作为固定数据选择历史，标签方向由实测丰度给出，EV6 `mutation_effect_on_PPI` 只保存在候选审计中。

每一条纳入记录同时满足：

1. EV1 mutant 的 `relates_to` 指向 WT，WT 的分号分隔 `relates_to` 列表包含该 mutant；母体 accession、起始坐标一致；物理 `library_peptide` 均为不同的 16 个标准残基序列，且两端输入库覆盖均大于零。使用实际库序列，保留设计时 C→A 等变化。
2. 证据满足以下两类之一：
   - 双端丰度证据：WT 和 mutant 均出现在 EV3，属于正确母体和突变身份；方向由两条 `mean_counts_%` 大小决定。
   - 单端加完整零观测证据：一个端点在 EV3 有正丰度；另一个类型在 EV6 中 `*_observed == 0`，且全部 `*_proportion` 数值为零。双端输入库覆盖质量门排除“起始库缺失导致筛选零”的已知混杂。

单纯“未在 EV3 中出现”不视为零，也不据此认定不结合。存在重复、不可唯一定位的 EV3 端点不用于配对；缺少关系或计数证据的条目逐行登记。最终 201 组的去向为：59 纳入、6 输入库覆盖质量门排除、12 G1 整母体排除、124 缺少足以定位计数方向的证据。统计显著性保留在候选组层级；G1 以实测 KD 承担独立方向检验。

## G1 定量保留与分母

Table 1 的 **29 行测量全部保留**。每行按 EV8 的实际合成肽序列匹配，乙酰化/酰胺化状态和额外 Y 保存在记录中。G1 肽长为 14–18，未截断成 16 肽。

| 状态 | 测量行数 | 主方向评分规则 |
|---|---:|---|
| 可确定方向 | 17 | 具有数值或足以确定 ≥2 倍方向的界限；较低 KD 一端更强 |
| 小变化 | 5 | 两个点估计的差异 <2 倍，单列保留 |
| `n.b.` 无数值界限 | 6 | Table 1 未提供数值下限，单列保留 |
| 肽身份冲突 | 1 | KRT1 的 Table 1 与 EV8 突变位置不同，保留原行并排除评分 |

`n.b.` 未被代入论文图示用的 20 倍，`n.d.` 未作为 KD。方向阈值使用发表的点估计或上下限，报告误差保留；没有把测量误差当作置信区间。17 行来自 12 个母体、13 个突变系列，置信区间应按母体/系列重采样，重复口袋测量仍属于同一分子系列。

发现并处理的身份问题：

- **BCL11A**：EV1、EV8 和 Table 1 均支持 C48F；正文 C47F 记录为来源文字冲突；保留实验 C50A 背景及末端 Y。
- **ABRAXAS1/BCL11A 附录冲突**：Appendix Figure S5（印刷第 5 页）将 ABRAXAS1 写为 `LDLDDRWQF(K/Q)RSRLLD`，而 Table1/EV8 为 `LDLDDRWQFKRS(R/Q)LLD`；BCL11A 图注为 `DHDLLT(C/F)GQCQMNY`，Table1/EV8 为 `DLLT(C/F)GQAQMNFPY`。主 17 行优先采用 Table1+EV8，逐行标注 `appendix_s5_sequence_conflict`；敏感性子集排除 ABRAXAS1 两口袋及 BCL11A–CLTC 共 3 行，保留 14 行。FANCA 图注也有序列冲突，原已因 `n.b.` 不评分。主 17 行的来源序列并非全部一致。
- **ZNF526**：Table 1 文本写 L160T，而序列 K/T、EV8 与 EV1 均支持 K160T。输入使用一致的实测序列，保留文字冲突记录。
- **KMT2D**：发表区段 `2993–3308` 与 P2301S 不一致；EV8 序列及 EV1 `GASSPSYGSPNLGFVD`、0-based start 2292 支持肽身份。保留错误区段文字和解析依据。
- **KRT1**：Table 1 `QTKWE(L/P)LQQVDTSTRT` 与 EV1 L214P 一致，EV8 `QTKWEL(L/P)QQVDTSTRT` 却将变体移到邻位 L。合成身份尚无法唯一确定，该测量不进入 17 行方向分母。
- **CDC45–KPNA7**：正确 WT KD 为 **6.6 ± 0.7 μM**，mutant 为 **380 ± 15 μM**；EV9 的 7.5 μM 属于 FITC-MYC 探针。

## 划分与跨任务暴露

G1 的 20 个母体基因及对应 accession 从 D2 全部排除，包括不同区段和同一母体的其他变体系列。原 65 对按共享母体 accession、或者任一端点 ≥80% identity 且双方覆盖率 ≥90% 连为组件；按 A/B 标签集合分层、seed42 洗牌，分配 28/6/6 组。覆盖质量门从原训练/验证分别移除 3 对，测试零移除；保留所有剩余原 group/split，最终为 34/16/9 对、26/6/6 组。没有查看模型成绩后重新筛分。

对任意 D2 跨 split 样本，母体重复和达到去重阈值的序列相似关系均为零。G1 与 D2 的同母体基因交集为零。

既有 Benz `benz_split_manifest.json` 保持原文件；SHA-256 为 `530cd6db379132b353cb1eff6776552232efc1c0e77013eae072cbe716ce492a`。按照用户要求，T1 不重新划分。对 stage A 训练暴露另作审计：

| 测试/来源 | 与既有 Benz train 同母体或近序列的记录 |
|---|---:|
| D2 train | 9/34 |
| D2 val | 13/16 |
| D2 test | 4/9 |
| G1 全部测量 | 1/29（MAPT，属于 `n.b.` 非评分行） |
| G1 可评分测量 | **0/17** |

T3 从 T1 初始化并回放时，D2 主结果需要同时附上未在 T1 train 暴露的 5 对/4 组子集结果。G1 的 17 行均符合本次训练数据层面的洁净条件；底座 ESM/Qwen 的预训练暴露不在此审计范围。

## 来源与验证

原文：[Kliche et al., Molecular Systems Biology 2024](https://doi.org/10.1038/s44320-024-00055-4)。本地原文 HTML 的方法部分说明 mutation enrichment score 汇总全部重叠 WT/mutant 肽与重复的归一化 NGS counts；附录第 3 页 Appendix Figure S3 描述单肽显著支持。官方 [ProP-PD portal](https://slim.icr.ac.uk/proppd/proppd_data_viewer?table_type=bait_index&library=GenDi) 于本轮直接请求返回 Cloudflare 403；本次恢复严格限于已公开、已下载的 EV1/2/3/6/8 和 Table 1。

追加取得官方 MOESM14/15/16 的 Figure 2/4/5 原始 ZIP，分别保存为 `data/raw/kliche2024/source_data_fig2.zip`、`source_data_fig4.zip`、`source_data_fig5.zip`，均验证 ZIP 及 SHA-256，链接、压缩包成员和一次超时重试记录在 `source_data_fig_manifest.json`。压缩包主要含 FP 原始读数、ITC/co-IP 等 Excel，留供后续核查；本轮已冻结的训练标签及测试筛选未按这些追加文件或模型成绩调整。

验证命令：`python scripts/prepare_t3_real.py`；`python -m unittest discover -s tests -p test_t3_real.py -v`。12 项测试覆盖界限、多突变、实测方向、零观测、完整记账、身份冲突、泄漏、原 Benz 哈希、覆盖质量门保留 split、14 行敏感性子集和靶点/口袋身份，以及突变后缀解析、元数据修复前后输入/标签一致性和生成文件哈希。训练结果结合 9 对 D2 test、17 行 G1 主面板、14 行来源一致子集及分组不确定性解释。
