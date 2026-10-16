# 文献与原始数据获取审计

审计日期：2026-10-15（Asia/Shanghai）。依据 `sources/conversation_read_thread_page1.json` 的全部五轮对话，以最新一轮的 T1/T2/T3 为当前基准范围。

## 已获取内容

| 来源 | 文献身份 | 本地文件 | 状态 |
|---|---|---|---|
| Benz 等，MSB 2022 | [Proteome-scale mapping of binding sites in the unstructured regions of the human proteome](https://doi.org/10.15252/msb.202110584) | `papers/benz2022/article.pdf`、`appendix.pdf`；`data/raw/benz2022/EV1.xlsx` 至 `EV11.xlsx` | 原文、附录及 11 个数据集齐全 |
| Kliche 等，MSB 2024 | [Proteome-scale characterisation of motif-based interactome rewiring by disease mutations](https://doi.org/10.1038/s44320-024-00055-4) | `papers/kliche2024/article.pdf`、`appendix.pdf`；`data/raw/kliche2024/EV1.xlsx` 至 `EV11.xlsx`；`table1.html`、`table1_expanded.csv` | 原文、附录、11 个数据集及正文亲和力表齐全 |
| Torres 等，Scientific Data 2025 | [A Paired Database of Predicted and Experimental Protein Peptide Binding Information](https://doi.org/10.1038/s41597-025-05754-7) | `papers/pepbi2025/article.pdf`、`Table_2_-_RIA_Properties.pdf`；`data/raw/pepbi/` 元数据 | 原文和官方关联补充 PDF 已获取；主数据下载未成功 |

全部已保存 PDF 均验证 `%PDF-` 文件头，Excel 均验证 ZIP 文件头并可用 openpyxl 读取。来源链接、字节数、SHA-256 和失败请求记录在 `sources/literature_data_manifest.json`。本次未新增依赖，未执行任何下载的代码。

## T1：Benz 数据已具备启动条件

关键入口是 `EV4.xlsx` 的 `ELM Benchmarking Dataset`，不是整本 Excel 的所有 sheet。

| 实测项目 | 数值 |
|---|---:|
| 数据行（不含表头） | 3,049 |
| 实际列数 | 35 |
| 独立 `Hit` 序列 | 2,724 |
| 独立 bait 构建体 | 34 |
| 肽长度 | 全部为 16 |
| confidence 2 / 3 / 4 | 1,715 / 938 / 396 |

原对话中的“35 个研究结构域”不能直接作为该主表的类别数；主表实际只有 34 个 bait。`EV2.xlsx` 给出构建体和结果概览。`EV4.xlsx` 的 KPNB1、Control Baits、Non primary paralogue 等 sheet 单列保存，未混入这 3,049 条。

能够建立的是已知阳性检索任务。合并同序列多靶点记录、按母体/重叠区段分组、隔离后续定量核验家族后，才能确定最终划分规模。没有记录的靶点仍属于未知，不应生成“实验证实不结合”标签。基序、PSSM、GO 注释及名称列可用于审计，直接送入序列任务会泄露标签。

## T3：候选组已核实，严格的肽对标签仍需对齐

| 文件与 sheet | 实测内容 |
|---|---|
| EV1 / B_GenVar_Peptides | 56,913 个带 accession 的原始行；唯一 `library_peptide` 和 `designed_peptide` 均为 56,911；关系记录不能直接视为独立序列 |
| EV3 / A-binding_peptides | 2,225 个数据行；26 个实际有值列（Excel 标记宽度 29） |
| EV3 / B-peptide_region | 1,317 个数据行 |
| EV3 / D-bait-prey | 1,230 个数据行 |
| EV6 / A Domain-mutation_pairs | 1,773 个数据行；51 个实际有值列（标记宽度 56）；80 个 bait |
| EV6 的原始效应标签 | 1,407 no_effect、173 diminished、193 enhanced |
| 显著组且 `number_significant_peptide_pairs > 0` | 201 |
| EV8 / Peptides | 35 个数据行、5 列；含合成肽/探针等身份，不能等同于 35 个独立测试对 |
| EV9 / KD values | 18 个数据行、4 列；是 FITC 探针饱和结合常数 |
| 正文 Table 1 | 29 个测量行；已正确展开 HTML 合并单元格；包括 major/minor pocket 等重复测量层级 |

Excel 多个 sheet 带有延伸到第 1,000 行等处的空格式，不能用 `max_row` 作为样本数。完整逐 sheet 非空行/有效列、表头样例和统计结果见 `data/raw/source_audit.json`，可运行 `python data/raw/inspect_sources.py` 重算。

**原对话中必须修正的定量来源：**EV9 不是 24 对 wild-type/mutant 的定量结果表。配对结果应从正文 Table 1 取得。CDC45–KPNA7 在 Table 1 中为 wild-type **6.6 ± 0.7 μM**、mutant **380 ± 15 μM**；对话所用 **7.5 μM** 实际出自 EV9 的 **FITC-MYC 探针**，不可替代 CDC45 的测量。

Table 1 的 29 行不是 29 个独立变体系列，更不是已经确定的 24 个可评分样本。当前保留作者原始文字；没有静默修正以下冲突：BCL11A 的正文 C47F 与表中 C48F、ZNF526 表中 L160T 与括号序列 K/T 及正文 K160T、KMT2D 的突变/区段坐标文字异常。必须联查 EV8、EV1、实际肽序列和位点后才生成金标准。`n.b.`、`n.d.`、`<` 界限及小于两倍差异应分别保留；不能将 `no_effect` 当作已证实等效。

要导出 D2，需要从 201 个候选组回到 EV1 的 `relationship_id` / `relates_to` 与原始计数，识别有直接证据的具体肽对；不能把 EV6 的 39 残基 mutation-centred 序列当作实验 16 肽，也不能把组级效应复制给全部重叠肽。定量测试须先固定并整组排除训练/调参暴露。当前审计未声称已完成该分组、标签及泄漏排除。

## PEPBI：主表仍是获取缺口

[Dryad 官方数据版本](https://datadryad.org/dataset/doi:10.5061/dryad.wstqjq2wk)（version 370880，2025-06-17）列出五个文件，总量 52,624,715 字节。已保存公开 API 的 dataset/files JSON，包含官方 SHA-256：

| 文件 | 官方字节数 | 当前结果 |
|---|---:|---|
| Binding_Group_Structures.zip | 52,066,711 | 网页下载 403；API 下载 401 |
| PEPBI.py | 1,481 | 同上；没有执行 |
| PEPBI.xlsx | 217,930 | 同上；未下载，不报告实际行列 |
| README.md | 15,388 | 同上；公开数据页说明可读 |
| Technical_Validation.csv | 323,205 | 同上 |

API download 明确返回 `Unauthorized, must have current bearer token`。网页 `file_stream` 返回 AWS 403。官方 [Zenodo 关联记录](https://zenodo.org/records/15678269)仅包含 RIA 属性说明 PDF，已下载；没有 PEPBI.xlsx 副本。DataONE 对 DOI 的公开索引查询返回 0 条。失败未保存为伪 Excel。

PEPBI 的论文口径是 181 个实验复合物、329 个结构条目、32 个 binding groups；这些未冒充本次 Excel 实测数。即使拿到表，也须去掉结构副本，匹配同靶点序列、构建体/口袋、温度及化学状态后才能组成外部肽对。Rosetta RIA 值属于计算性质，不能充当实验亲和力标签。

## 当前可行性结论

T1 的原始数据和构建体信息已本地齐全，足以启动数据准备与基线运行。T2 可复用 T1 的固定测试划分。T3 原始材料已齐全，真正的障碍是具体肽对、探针与被测对象、口袋以及外部保留家族的严格匹配；没有清洗依据时，不应把“201”直接填成最终训练样本数。PEPBI 属于外部扩展，不阻塞先验证 T1。
