# PepBridge-Bench v0.2：ELM 与 B1/B2/C2 证据审计

2026-10-15。B1 已形成两个可复现、独立保存的候选范围：人类敏感性版本为 **34 类、567 条记录/519 条不同序列**；补充常用无病原模型真核物种后为 **63 类、1157 条记录/1039 条不同序列**。后者测试部分为 **5 类、125 条记录**。两者均待全局同源审计和描述复核，尚未冻结为正式训练版本。

## 官方版本与实际获取量

[ELM 2024 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10767929/)（DOI 10.1093/nar/gkad1058）报告 356 类、4283 个实例。[官方下载页](http://elm.eu.org/downloads.html)提供实时导出，未列出可验证的 2024 固定归档。本次导出自身的日期是 2026-10-15，版本字段是 TSV schema 1.4，内容为 **353 类、4277 条实例**。因此资源登记使用 `official_current_snapshot_not_2024_archive`。数量差异的原因尚未取得官方版本说明。

| 文件 | 内容/实际量 |
|---|---|
| `data/raw/bench_v02/elm/classes.tsv` | 353 类；类别说明、regex、频率和数据库计数 |
| `data/raw/bench_v02/elm/instances.tsv` | 4277 条；4047 条标记为 curated true positive |
| `data/raw/bench_v02/elm/instances.fasta` | 2728 个母体 accession 的官方 FASTA |
| `data/raw/bench_v02/elm/interactions.tsv` | 官方互作、实验来源与部分亲和力字段原样留存 |
| `data/raw/bench_v02/elm/methods.tsv` | 官方实验方法词表 |
| `data/raw/bench_v02/elm/pdbs.tsv` | 官方结构交叉引用 |

完整下载 URL、SHA256、字节数及论文获取状态见 `sources/bench_v02/elm_sources.json`。论文全文 HTML 已保存于 `papers/bench_v02/elm/ELM_2024_landing.html`；PDF 获取遇到仓储 403、429 和 PMC HTML 检查页，未保存冒充 PDF 的文件。

**下载参数有实际截断风险：**无参数和 `q=None` 的实例 TSV 本次均只返回 100 条；构建只读取显式 `q=*` 获取且文件头报告 4277 条的 `instances.tsv`。对应 100 条文件保留为下载诊断材料。

## B1 human 历史敏感性版本的构建与划分

输入范围是人类母体蛋白中的局部功能区段。实例坐标按 1-based inclusive 在同一批官方下载的母体 FASTA 上截取；缺失母体、越界和非标准残基会进入隔离审计。ELM 的局部功能注释保留母体语境，游离肽药效字段固定为 false。

类别首先按固定 SHA256 规则划分：余数 0/1 为测试、2 为验证、3–9 为训练。随后，跨类别边界的共同母体、精确序列及原始 PubMed 来源进入 quarantine。对现有 Benz、Kliche D2/G1 的所有已登记母体和重叠片段实施保守排除，并锁定 Benz PMID 35044719、Kliche PMID 39009827。当前共检查 1654 个跨库母体 accession。

在这些过滤之后，每类至少需要 10 个独立母体区段。相同母体上有重叠的区间按连通分量只计一次；远隔区间可各计一个来源区段，母体仍参与跨 split 隔离。

| 划分 | 类别 | 实例记录 |
|---|---:|---:|
| 训练 | 30 | 522 |
| 验证 | 1 | 11 |
| 测试 | 3 | 34 |
| 合计 | 34 | 567 |

类别可用性与计数在 `b1_class_eligibility.jsonl`，所有实例的保留/剔除原因在 `b1_instance_audit.jsonl`。多个过滤原因可同时命中同一记录，原因计数不应相加当作删除总数。

`b1_model_inputs.jsonl`仅提供匿名样本 ID、局部序列和共同候选目录引用，隐藏母体名称与 gold 类别。主候选文件 `b1_candidates_main.jsonl` 包含同一个 34 类候选集合，覆盖已见和未见类。类别映射留在独立标签/审计文件。

主面板对明确的正则语法和序列模式作保守遮罩，所有候选描述仍标记 `description_review_required=true`。完整说明和规则单列于 `b1_classes_regex_full.jsonl`，用于 regex-full 面板及规则匹配基线。正式使用前需人工核对规则残留与文本含义；遮罩也可能移除生物学缩写。

主指标应按未见类别计算 Macro-MRR、Recall@1/3，并另报已见类表现；共同候选范围需随结果保留。48 条记录与其他记录共享精确序列，统计推断应按母体/序列组处理，不能把 567 条记录全部当作独立序列。ELM 未标注关系保持 unknown。

human 当前 readiness 为 `audit_ready_training_requires_global_source_and_description_review`。父级协议还需完成跨新引入数据库的来源/同源聚类审计，检查其他训练来源的测试类监督，并评估该敏感性版本仅 3 个测试类的覆盖范围。

## 补充：预先指定模型真核物种的范围审计

原协议未要求限定人类，因此增加 `model_eukaryotes` 范围并保持已有 human 文件原样。物种集合为 taxids 9606、10090、10116、7227、6239、4932、4896、3702、7955，分别对应人、小鼠、大鼠、果蝇、线虫、芽殖酵母、裂殖酵母、拟南芥和斑马鱼。范围依据研究对象和数据覆盖设定，整个步骤未使用模型预测分数。

物种名和 ID 经官方 UniProt taxonomy JSON 核实并本地固定。ELM Organism 字段只接受官方 scientificName 或 otherNames 的大小写无关精确匹配。芽殖酵母 S288c（559292）和裂殖酵母 972h-（284812）的官方别名及 lineage 分别指向白名单物种 4932 和 4896，因此纳入；其他未验证株系、名称前缀及模糊名称均不匹配。每个保留实例携带原始物种名、taxid、白名单物种 taxid、官方 URL 和本地 taxonomy 文件来源。

两范围使用同一个生成器、相同筛选与隔离规则，模型物种版本采用单独版本化的 70/15/15 class-holdout hash，保持 curated true positive、官方 FASTA 序列核验、每类至少 10 独立区段、母体/序列/PMID 跨划分隔离及 Benz/Kliche 排除策略。

| 范围 | 可用类别 | 可用记录 | 独特序列 | 训练类别/记录 | 验证类别/记录 | 测试类别/记录 |
|---|---:|---:|---:|---:|---:|---:|
| human（已有文件） | 34 | 567 | 519 | 30/522 | 1/11 | 3/34 |
| model_eukaryotes（独立目录；70/15/15 规则） | 63 | 1157 | 1039 | 44/749 | 14/283 | 5/125 |

模型物种版本按最新协议采用名义 70/15/15 类别分配：SHA256 前缀固定为 `pepbridge-v02-class-model_eukaryotes-70_15_15-v1-20261015:`，hash 对 20 取余，0–2 为测试、3–5 为验证、6–19 为训练。隔离和每类实例门槛会使最终计数偏离名义比例；本次不为平衡覆盖重新选择 seed。原先 70/10/20 的模型物种准备版本全部九个文件已保留于同目录 `prepared_rule_70_10_20/`，并记录逐文件 SHA256。human 历史文件和原先 hash 规则保持不变。

模型真核物种范围含 3382 条原始 curated true positive 记录；严格隔离和类别门槛后保留 1157 条。扩大物种范围会改变类别是否满足实例门槛，因此新范围中人类记录数也会变化。两个范围的记录数差不应直接解释为“新增物种记录数”。

新范围全部写入 `data/processed/bench_v02/b1_model_eukaryotes/`，包括 `b1_samples.jsonl`、匿名 `b1_model_inputs.jsonl`、共同 `b1_candidates_main.jsonl`、`b1_manifest.json` 和完整隔离审计。跨物种的同源蛋白和近似序列仍需全局聚类隔离；现有精确母体/序列/PMID 隔离不替代同源审计。主范围与人类敏感性面板的最终用途待协议冻结，当前仅报告真实覆盖与待完成条件。

```powershell
python scripts/bench_v02/prepare_elm.py --scope model_eukaryotes --out data/processed/bench_v02/b1_model_eukaryotes
```

## B2 与 C2 的证据界面

`b2_evidence_registry.jsonl` 保存 **4277 张真实 ELM 证据卡**：实例 ID、母体和坐标、原始 InstanceLogic、方法、原始文献链接、类级背景引用和结构交叉引用。每张卡均为 `review_pending`、`qa_eligible=false`；已人工核查的组合 QA 数量为 **0**。

形成 B2 样本还需核对原始实验、选择独立的类级关系、去掉样本答案泄漏、定义联合答案与推断边界，并锁定“关系—问题组合”留出。证据卡数量只表示待核查证据记录数。具有相同分子的跨性质测量必须有可核实的连接键，才能形成共同金标准。

C2 保留 ELM PDB 交叉引用作为结构来源。原始结构中的链、残基映射及接触需要另行核验；功能热点需要突变、活性或结合变化等直接证据。因此卡片中的 `structural_link_is_functional_hotspot` 固定为 false。

## 复现与验证

```powershell
python scripts/bench_v02/prepare_elm.py
python -m unittest discover -s tests -p test_v02_elm.py -v
```

脚本仅使用标准库、只在 v0.2 路径写入派生数据，不启动网络和训练。十一项测试核对区段独立计数、规则遮罩、冲突 FASTA 的 fail-closed、两个范围真实划分的类别/母体/序列/PMID 隔离、模型输入的 gold 字段隐藏、B2/C2 证据状态、taxonomy 精确核实、模型物种范围覆盖 human 输出的保护、70/15/15 比例规则及历史快照哈希。旧版 `data/processed`、`results` 和现有 human 根目录产物保持原样。
