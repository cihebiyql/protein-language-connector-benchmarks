# A1 数据准备与来源核验（2026-10-15）

本轮已继续完成 A1 v2 外部隔离与真实训练；当前入口见 [PEPBI恢复与v2划分](bench_v02_pepbi_recovery.md)及[实测报告](bench_v02_panel_results.md)。下文保留初次表级清洗与来源审计。

PPIKB 官方四份 Excel 已下载并逐行检查，得到 **5,122 条广覆盖 KD 回归候选记录**。这些记录提供真实的蛋白质参考序列、可表示的线性标准氨基酸肽及明确末端状态；实际实验构建体和缓冲液等条件仍待原始论文补充。**目前条件受控配对数为 0，正式训练/验证/测试划分数为 0。**

PepInter 的数据可复现性状态需要更新：当前官方仓库已发布亲和力 train/test CSV，Zenodo 新版也已发布四份亲和力来源表。README 的“PepAffDB coming soon”已滞后。模型推理、环境和权重兼容性尚未运行验证。

## 官方来源与已下载材料

PPIKB 论文：[Scientific Data，10.1038/s41597-026-07962-1](https://www.nature.com/articles/s41597-026-07962-1)。论文 PDF、网页和全文抽取已保存在 `papers/bench_v02/ppikb/`。论文报告的 21,845 条测量和 10,607 条 KD 与 Main.xlsx 一致。[官方数据下载页](https://ppikb.duanlab.ac/downloads/)仍将当前版本标为 PPIKB v2025。

| 本地文件 | 官方链接 | 实际数据行数 |
|---|---|---:|
| Main.xlsx | [Affinity Dataset(main).xlsx](https://ppikb.duanlab.ac/downloads/Affinity%20Dataset%28main%29.xlsx) | 21,845 |
| research.xlsx | [research.xlsx](https://ppikb.duanlab.ac/downloads/research.xlsx) | 19,509 |
| patent.xlsx | [patent.xlsx](https://ppikb.duanlab.ac/downloads/patent.xlsx) | 2,815 |
| Branch.xlsx | [Affinity Dataset(branch).xlsx](https://ppikb.duanlab.ac/downloads/Affinity%20Dataset%28branch%29.xlsx) | 3,180 |

四表均保存在 `data/raw/bench_v02/ppikb/`，URL、字节数、SHA256 和来源快照在 `sources/bench_v02/affinity_sources.json`。官方数据库页声明 CC BY 4.0；文章页面声明 CC BY-NC-ND 4.0，两者分别记录。

## 字段与清洗规则

`a1_normalized.jsonl` 每行对应一个 Main.xlsx 原记录，保留完整 `raw_metadata`，专利行和不合格行仍在逐行审计中。research 独有记录仅保存在逐行审计中。

| 标准字段 | 含义/来源 |
|---|---|
| `sample_id`, `source_record_id` | `ppikb:ID` 与原始 ID |
| `task`, `panel_id` | `affinity`, `A1` |
| `peptide_sequence` | Main 的 Peptide_Sequence，保留大小写，不将 D-AA 大写化 |
| `target_sequence`, `target_sequence_role` | Main 的真实 Protein_Sequence；身份明确为数据库参考序列，实际 assay 构建体未核验 |
| `measurement_type`, `relation`, `kd_molar`, `pkd`, `pkd_relation` | 仅解析 Main 的明确单 KD 字符串；单位转 M、计算 −log10；上下限保留并反转 pKD 不等号 |
| `label_exact`, `reported_error_molar` | 界限不能进入 exact 回归；误差另存 |
| `chemical_state` | 拓扑、残基修饰、N/C 末端状态；当前支持 free/acetyl N 端及 free/amide C 端 |
| `experimental_context` | assay 保留；pH、温度、buffer、实际构建体明确为 null |
| `source_doi`, `groups` | DOI、肽、靶点参考序列、UniProt、疑似重复测量哈希；家族与原始实验组尚未分配 |
| `eligibility.broad_reference_target` | 研究论文来源、明确 KD、5–50 aa 标准线性肽、化学状态明确、真实靶点序列、DOI、跨表身份及数值一致 |
| `eligibility.strict_comparable_pair` | 行具备实际构建体和可比较实验条件才可进入配对；当前全部 false |
| `split_status`, `readiness_reasons` | 合格候选仍为 pending_global_group_split，其余 quarantined；原因逐行可追踪 |
| `provenance.per_field_source` | 每类字段来源；research 导出重写的 Affinity 单独保留 |

缺失实验条件不会形成“相同条件组”。同一 DOI、同一参考序列和空条件也不会自动配成强弱肽对。重复记录全部保留，并标注候选重复组；没有将相同字符串当作已确认独立实验。

## 核验发现与实际保留量

| 审计项 | 实际数值 |
|---|---:|
| Main 标准化字段标为 KD | 10,607 |
| 广覆盖参考靶点候选（均为数据库点 KD） | **5,122** |
| 广覆盖上下限候选 | 0 |
| 严格条件可比行 / 已构建配对 | **0 / 0** |
| 正式监督划分记录 | **0** |
| research 有但 Main 缺失的记录 | 387 |
| Main / research 肽序列冲突 | 3 |
| Main 原始 Affinity 与标准化浓度冲突 | 56 |
| 候选中疑似重复测量组 / 记录数 | 100 / 236 |

5,122 是合格数据库记录数，独立分子家族或独立实验数尚未确定。0 条上下限候选表示本次下载表和解析保留量；数据库标签曾否简化原论文上下限，需要源文献复核。

两个具体问题解释了为何不能直接使用 research 的 KD 列：

1. PPIRE00001 的 research 导出为 `Kd=46.6 uM`；Main 标准字段也标 Kd/46.6 uM，但 Main `Affinity` 为 `Ki(app)=2.2×10^-6 M`。该行进入终点不一致隔离。
2. PPIRE00002 在 Main 的序列包含非标准 `X` 和小写 `f`，research 导出将其写成标准字母且丢失末端信息。该行进入分子身份冲突隔离。

research.xlsx 的 N/C-terminal_Modification 两列全部为空；Main 的对应字段更完整。Main 研究论文来源只有 19,122 行，而 research 导出为 19,509 行。387 个 orphan ID 的 DOI 和完整化学状态未在 Main 核实，保持隔离。Branch 仅作为对应结构/来源辅助表，未与 Main 叠加扩大样本数。

## 外部金标准锁定与划分阻塞

`a1_external_locks.json` 已保存 Kliche 原论文 DOI、PEPBI 论文和数据 DOI、旧 T3 external/test/val 及 Benz test/val 的已暴露肽序列和记录 ID，同时对旧文件记录 SHA256。当前 Main 未发现这些序列的完全相同命中。

PEPBI 的实际数据工作簿尚未在本地取得，因此原始实验论文 DOI 清单不完整。仅排除 PEPBI 数据库论文 DOI 不足以排除跨库同源实验。PPIKB 的 target Gene 是靶点基因，不能当作肽的母体基因去判断系列重复。肽近似关系、母体区段、变体系列、靶点家族和跨库实验去重留给全局划分审计。完成这些步骤前，候选记录继续保持 pending 状态。

## PepInter 官方可复现性更新

论文：[Communications Chemistry，10.1038/s42004-026-02168-3](https://www.nature.com/articles/s42004-026-02168-3)；[官方仓库](https://github.com/MolAstra/PepInter)；当前记录的 Git tree SHA 为 `b28bf758653cbd58973f45649cb41c849b461515`。

| 本地已下载官方数据 | 实际行数 | 状态 |
|---|---:|---|
| dataset/pepinter_aff/train.csv | 3,494 | 序列、cluster、value、affinity、split、label 可获取 |
| dataset/pepinter_aff/test.csv | 138 | 文件内 split 列为 val，需按官方流程核对用途 |
| 新 Zenodo PDBbind.csv | 937 | 包含 PDB、chain、序列、终点、单位、部分 PubMed |
| 新 Zenodo SabDab.csv | 246 | 序列和 KD；缺逐测量 DOI/化学状态 |
| 新 Zenodo Umami.csv | 2,069 | 序列和 KD；缺逐测量 DOI/化学状态 |
| 新 Zenodo skempi_v2.csv | 7,085 | 分号分隔的原始 SKEMPI 记录，含 Reference/Temperature 等，尚非已核验肽子集 |

[Zenodo v1 模型记录](https://zenodo.org/records/18604811)包含分类 6.9 GB、回归 7.0 GB 权重，元数据已下载，权重未下载。[latest 解析的新记录 21715291](https://zenodo.org/records/21715291)公开来源数据，四份亲和力 CSV 已下载；23.4 MB 的 ppi3d 结构预训练文件只记录元数据。本步骤没有执行其外部代码。

官方 dataset README 的来源计数与当前文件对应数存在错位：README 将 SKEMPI 写为 246、SabDab 为 2,069、Umami 为 1,869，实际下载分别为原始 SKEMPI 7,085、SabDab 246、Umami 2,069；经处理 train+test 为 3,632，也不同于论文报告 3,670。正式复现需要固定版本并核对过滤/划分过程。官方代码 MIT；模型记录声明 CC BY 4.0，原始来源数据的上游许可仍随来源保存。

## 复现与验证

```powershell
python scripts/bench_v02/prepare_affinity.py
python -m unittest discover -s tests -p test_v02_affinity.py -v
```

本轮 **9 项测试通过**，覆盖单位换算、上下限、科学记数/误差、非 KD 隔离、标准化冲突、分子身份冲突、D-AA/环化隔离、真实靶点缺失和条件缺失 gate。准备脚本不下载依赖、不执行训练、不修改旧数据或 results。产物为标准 JSONL、逐行 audit、候选集、外部锁定清单和带输入哈希的汇总。
