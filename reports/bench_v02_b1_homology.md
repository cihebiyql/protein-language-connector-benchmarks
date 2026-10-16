# B1 同源隔离与受限范围冻结

外部补充已闭环：1654个母体及2813个有效区段关联均核验，新增同源命中0；原锁和样本不变。见[补充证据](../docs/PUBLICATION_SCOPE.md)及[真实训练结果](bench_v02_panel_results.md)。下文记录冻结时的初始覆盖。

2026-10-15。B1 已完成实际全母体序列及局部区段上下文比对，并冻结为可执行的 **panel-specific predictor** 基准：**31 类、588 条记录、558 条不同局部序列**。训练、验证和测试均有保留；现有面板 gate 已实际通过。外部序列尚有缺口，因此锁为 `frozen_restricted_scope`。

| 划分 | 类别 | 记录 |
|---|---:|---:|
| 训练 | 22 | 378 |
| 验证 | 7 | 134 |
| 测试 | 2 | 76 |

输入沿用 `b1_model_eukaryotes` 的固定 70/15/15 class-holdout 规则和原匿名候选 ID，未根据模型表现改 seed。两条未见测试类别为 `MOD_N-GLC_1`、`MOD_NMyristoyl`；所得结果应按这两个类别的覆盖范围解释。

## 实际比对及预先固定阈值

探测发现 Windows 默认环境和 WSL 系统 Python 均未提供 Biopython；WSL 已有 MMseqs2，未安装依赖。实际版本与执行记录保存在 `tool_manifest.json`、`mmseqs_execution.log`。

| 层面 | 输入 | 命中阈值 | 分组含义 |
|---|---|---|---|
| 全母体蛋白 | 840 个 B1 母体查询 × 2487 个母体目标 | identity ≥30%，查询和目标 coverage 均 ≥80%，E ≤10⁻⁵ | 全母体序列操作性同源组 |
| 局部上下文 | 1157 个 B1 区段查询 × 3937 个区段目标 | identity ≥50%，双方 coverage 均 ≥80%，alignment ≥30 aa，E ≤10⁻⁵ | 局部序列上下文相似组 |

局部上下文由实例真实坐标处区段及两侧各最多 32 个母体残基构成，序列来自 ELM 官方母体 FASTA。Benz/Kliche 的外部上下文只在公开局部序列能精确定位到已取得母体时建立；变体和合成附加残基不靠猜测删除或补齐。短 motif 的共享不作为全蛋白同源证据。

比对采用 [MMseqs2 官方全对全 fake-prefilter 方法](https://github.com/soedinglab/MMseqs2/wiki#how-to-create-a-fake-prefiltering-for-all-vs-all-alignments)：为每个查询显式列出全部目标，交给实际 Smith–Waterman 对齐模块，绕过 k-mer 候选预筛。`parent_alignments.tsv` 和 `context_alignments.tsv` 保存 identity、双方 coverage、对齐长度、E-value、起止位置及全序列长度。独立测试核对每个 query 的候选索引均指向完整 target 列表。

实际得到 580 条非自身母体命中、336 条非自身局部上下文命中。命中组成连通分量，跨既定 split 或连接到已解析外部参照的分量整体进入 quarantine。局部精确序列相同的实例也合并到同一局部家族，防止不同上下文绕过直接序列隔离。

## 独立计数与来源隔离

每类门槛采用保守独立单位：**同一个全母体同源连通分量，每类最多计一个独立单位，至少需要 10 个不同分量**。`class_independence_audit.jsonl` 同时保留原母体重叠区段计数、记录数和新的独立同源家族数。这样，同族 ortholog 的多条实例不会扩大独立单位数；同一母体远隔区段也不会在新门槛中重复计数。

| 测试类 | 记录 | 原母体重叠区段计数 | 独立母体同源家族 |
|---|---:|---:|---:|
| MOD_N-GLC_1 | 52 | 51 | 15 |
| MOD_NMyristoyl | 24 | 24 | 23 |

原始 PubMed 列表按**每个 PMID**检查，未把整条列表当一个不可拆分 ID。没有可核对原始 PMID 的实例进入 quarantine，未用虚构来源编号代替文献证据。

最终五个维度均为零跨划分重叠：局部精确序列、全母体同源家族、局部上下文家族、每个原始 PMID 和功能类别。已解析的外部母体/局部家族与保留实例之间，跨组件命中数也为零。这些零计数对应明确阈值和已取得的参照范围。

## 外部范围与解释边界

外部 Benz/Kliche 共 1654 个母体 accession：292 个在 ELM 官方 FASTA 中已有，其余通过 [UniProt 官方 REST](https://rest.uniprot.org/) 获取。实际取得 **1647/1654**；未取得的七个 accession 为 A4D174、A8MYX2、I3L1I5、O15320、Q32Q52、Q6ZW33、Q9UPP5。另有 **41 条外部记录的 canonical 局部上下文未能精确定位**，逐条见 `external_context_audit.jsonl`。

因此 `external_isolation_verified=false`。此版本支持已取得参照范围内的内部划分比较；完整外部隔离以及统一多任务连接器的正式主张继续受原协议约束。序列阈值也不能排除所有远缘演化关系。

## 功能描述与共同候选

保留的 31 类均逐条对照官方原始 Description 字段重写功能语义。去除了 regex、明确序列例子、残基组成/位置偏好以及当前样本答案；类别作用关系和识别对象保留。`description_review.jsonl` 并列保存原说明、隐藏规则、重写文本和审查依据。

例如，CaM 结合类保留“calcium-independent calmodulin binding”，删除原说明的位置 1、2、6、7、11 等残基答案；PP1 docking 类保留 phosphatase docking/dephosphorylation 语义，删除正则残片。此为代理逐条来源复核，未声明人工专家审核。

所有系统使用同一个 **31 类共同候选目录**，包含已见与未见类。匿名 candidate ID 沿用原版本；模型输入仅提供局部序列及共同候选目录引用。每条记录附 `known_positive_class_ids` 和 `known_positive_candidate_ids`，归并当前候选范围内相同局部序列的全部已注释正类。缺失标注关系仍是 unknown。

## 交付、锁和复现

全部新增数据位于 `data/processed/bench_v02/b1_homology_locked/`：

- `b1_samples.jsonl`：真实训练/验证/测试记录与同源分组。
- `test_gold.jsonl`：固定测试金标准。
- `b1_candidates_main.jsonl`、`b1_classes_main.jsonl`：共同候选与隐藏类别映射。
- `group_audit.json`、`accepted_homology_edges.jsonl`：实际命中、组件成员和零重叠检查。
- `exclusion_audit.jsonl`、`class_independence_audit.jsonl`、`description_review.jsonl`：逐记录、逐类、逐描述审计。
- `benchmark_lock.json`：`scope=panel_specific_predictor`、`panel_id=B1`、`status=frozen_restricted_scope`，pin 样本、gold、候选、类别映射、真实比对、分组审计、实际输入序列、协议和构建脚本 SHA256。

```powershell
python scripts/bench_v02/isolate_elm.py --stage prepare
wsl.exe -d Ubuntu-22.04 -- python3 PROJECT_ROOT/scripts/bench_v02/isolate_elm.py --stage align
python scripts/bench_v02/isolate_elm.py --stage build
python -m unittest discover -s tests -p test_v02_elm_homology.py -v
```

下载得到的外部 FASTA 及来源请求登记已保存于 `raw_external/`；重建阶段使用这些固定本地输入。**八项同源锁测试通过**，涵盖全目标对齐索引、实际命中到组件的映射、五维零重叠、同源家族独立门槛、已知正类归并、描述规则、物理哈希及外部受限状态。既有准备快照、配置、多任务合同和基线脚本保持原样。
