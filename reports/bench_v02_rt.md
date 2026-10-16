# A2 保留时间：真实数据审计与 DeepLC 基线更新

2026-10-15。**已证明源候选的113,215个数值全部是Guan2019原始iRT，逐项完全相等；修复221条遗漏的N端乙酰化，恢复氧化位点和固定Cys状态。仍有98,192条因上游D/E有损编码而化学身份不确定，且2026精确源版本未证实，原定训练继续blocked。没有运行训练、DeepLC、GPU或上游代码。**

## 来源与实际规模

[2026 年 DeepLC 迁移论文](https://www.nature.com/articles/s41467-026-68981-5)指定 PXD005573 为预训练源域，基础色谱条件改变实验来自 PXD064311 的 Experiment 2, March。论文报告的规模为 113,213 与 6,363；本地文件统计如下，两个口径差异尚未解释，不能据此认定论文数字错误。

| 文件 | 实际数据行 | 独立序列 | 序列＋原修饰键 | 状态 |
|---|---:|---:|---:|---|
| PRIDE `peptides_ph9_march.csv` | 7,070 | 6,929 | 7,070 | 真实目标条件测量可用 |
| 2021 官方 Zenodo `PXD005573_mcp.csv` | 113,215 | 107,096 | 113,197 | 数值已确定为iRT；旧表原样隔离 |

[PRIDE 官方目标文件](https://ftp.pride.ebi.ac.uk/pride/data/archive/2026/02/PXD064311/peptides_ph9_march.csv)仅 419,200 字节；条件为 pH 9、HeLa S3、March 实验、作者跨检测取平均 apex RT，单位为分钟。保留 `RT_std` 和 `RT_count`。所有 7,070 条均满足原论文 CV≤10% 的描述；未额外筛成 6,363 条。最后一次有界核查在2026官方归档中取到 `basic_cv_train_fold_1.csv`：实际为6,363个独立peptidoforms，全部包含在完整7,070条目标表中；另取到一个测试fold为707条。这为6,363属于90%交叉验证训练fold提供了直接文件证据，不能把它当成清洗后的完整数据总数。

源域文件来自 [2021 原论文的官方 Zenodo 数据](https://zenodo.org/records/4542884)，是 `train_eval.zip` 中 3,651,331 字节的小成员。使用 HTTP Range 读取 ZIP 中央目录和约 1.9 MB 压缩成员，由 Python zipfile 校验成员 CRC；未下载 118 MB 整包或 GB 级原始质谱文件。源域 CSV 有21,722个负 `tr`，全部保留。第一轮隔离表的 `rt_unit=null` 被保留作为审计快照；新增恢复表已依据原始数组的逐项核对明确标成 **iRT**。没有平移、改数值或转换成分钟。

源域7,874个Oxidation注释采用0-based M位点；原始one-hot的J=M(Oxidation)逐行验证了这个约定。新增恢复表以原始编码重建1-based位置，N端乙酰化位于0。Cys按原实验固定Carbamidomethyl及原作者编码去除该修饰文字的证据显式恢复；不能把旧CSV空修饰解释成无CAM。

另外取到的 ProteomicsML `PRIDE_MQ.zip` 中没有 PXD005573（逐行验证为 0 条），因此未用其他项目替代指定源域。2026 Zenodo 的完整 tar 为 40,396,094,976 字节，仅获取部分目录头及指定小成员，完整包未下载。

进一步从 2026 官方归档只取出 419,169 字节的 `deeplc_one/ph9_march_A.csv`：同样有 7,070 条，与 PRIDE 的序列＋修饰键完全相符，RT 只有约 5×10⁻⁹ 分钟的四舍五入差异。这支持目标数据的来源一致性。归档中的 `predicted_tr` 列是模型预测，未进入真值或输入。

## 数据输出与协议

- `data/processed/bench_v02/a2_target_march.jsonl`：7,070 条，包含序列、保留 PTM 的 peptidoform、结构化修饰、条件 ID、分钟 RT、来源行号、分组和测量资格。
- `data/processed/bench_v02/a2_source_candidates.jsonl`：113,215 条保留原始信息的候选，`measurement_valid=false`，记录阻塞原因，不输入训练就绪记录。
- `data/processed/bench_v02/a2_target_budgets.json`：条件迁移允许源域序列重叠，以及条件＋新序列迁移两种难度；两者分别列明 32、128、512、2,048 条的真实 label IDs、fit IDs、validation IDs、校准可用 IDs 和固定测试 IDs。
- `data/processed/bench_v02/a2_audit.json`：实际计数、SHA256、重复、范围、重叠与限制。
- `data/processed/bench_v02/a2_source_irt_reconciled.jsonl`：新增iRT恢复表，113,215条，包含原始one-hot、准确PTM位置、尺度证据及逐行identity资格。
- `data/processed/bench_v02/a2_source_irt_row_mapping.jsonl`：113,215条旧CSV行号到原数组行号的对应、221条乙酰化修复账本。
- `data/processed/bench_v02/a2_source_irt_reconciliation.json`：当前源尺度和化学状态的权威恢复摘要；旧a2_audit隔离快照未覆盖改写。

目标按基础序列分组，所有修饰变体一起进入拟定 train / validation / test，当前各为 4,952 / 1,061 / 1,057 行。它是全局冻结前的可检查提案。**目标内诊断划分不替代协议指定的 PXD005573 源条件内新肽测试。**

四档目标标注子集为嵌套、无重复的真实记录；fit、校准、验证、早停和选择使用的全部目标标签必须包含在 `label_ids` 的并集中，不能另加免费验证集。某预算若不足，记录 unsupported，禁止有放回抽样凑数。目前所有跨条件预算均因源域语义 gate 标为 unsupported；有足够记录也不解除该 gate。

源域与目标有 4,263 条不同的基础序列重叠；目标有 2,666 条基础序列在该源候选中未出现。新序列难度已用这个可核实交集准备候选 ID。跨源修饰分子重叠未核清，母体蛋白映射、近同源与重叠片段也尚未完成全局审计，**不能宣称新肽家族泛化**。

## DeepLC 面板

[官方 README](https://github.com/CompOmics/DeepLC)明确将 **v3.1.13** 用于 2026 论文复现。静态仓库已取到：

| 面板 | 本地仓库 | 固定 commit | 使用边界 |
|---|---|---|---|
| 论文复现 / 公平重训起点 | `repos/bench_v02/DeepLC-v3.1.13` | `0409a7809259eaeac8705327d79772f3ad75b380` | 需独立环境；没有执行 |
| 当前公开系统实际使用 | `repos/bench_v02/DeepLC-current` | `e24a5822bdb5304fcb1beb9ee5b307e1ab643221` | 6,543 setups 多条件预训练；暴露未知，不混入公平源域重训排名 |

当前模型训练覆盖范围未完全审计。即使之后可以直接给出结果，也只能先进入实际使用面板，不能当作未接触目标条件的公平重训结果。公平重训要求同源数据、同目标标注预算和同固定 holdout。

## 复验与下一道门

离线运行 `python scripts/bench_v02/prepare_rt.py` 重建第一轮隔离表；运行 `python scripts/bench_v02/reconcile_rt_source.py` 重建新增恢复表。`python -m unittest discover -s tests -p test_v02_rt.py -v` 的11项测试通过，包括原有预算/状态规则及静态opcode不执行GLOBAL/REDUCE、CRC不一致拒绝和测量标签不一致拒绝。没有调用 `pickle.load`。

恢复原方案训练前仍需确定2026使用的确切源处理版本、解决或隔离上游D/E身份丢失、规定重复处理，再按修复后的化学分子重做跨源隔离和全局划分。源iRT尺度已经核实，不能继续以“尺度未知”描述最新恢复表；也不能将iRT误称为分钟。

完整来源与文件校验见 `sources/bench_v02/rt_sources.json`。

## 最后一次有界来源核查

核查在三分钟上限内结束，没有修改已审计CSV、标准化标签或划分文件。保留审计记录 `data/raw/bench_v02/rt/bounded_followup_20261015.json`。

已检查2026归档 `deeplc_one` 的排序目录范围至PXD008289，其中覆盖PXD005573应出现的项目编号范围，但未见该源CSV；这不是对40GB归档全局不存在的证明。另用HTTP Range取末尾4MB，定位并取出上述真实CV小表。归档 `14ptm.py` 只引用 `full_hc_PXD005573_pub_*.keras` 基础模型，没有给出源CSV或其转换过程。

同时静态检查官方 DeepLCRetrainer commit `dc0ea5c6589304949918d2f0e3f0dbb34b778e00` 的两个主要源码文件，没有113213或PXD005573的特定筛选规则。`cnn_functions.read_infile` 的通用辅助函数会在最小值为负时给所有 `tr` 加上 `abs(min_tr)`；这不足以说明旧源表单位，也不能证明该函数用于2026指定源数据。因而没有把该平移应用到源候选标签。**本轮未取得2026指定源表及可核验的源专属RT转换，源域gate继续blocked。**

## 续查得到的确定性修复

[Guan2019原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6773555/)及[其官方数据](https://zenodo.org/records/2652602)说明目标为indexed retention time。训练数据来自PXD005573的HeLa/HEK293三次DIA技术运行、1m柱和4h条件，并使用汇总的median iRT。只下载约7MB压缩processed成员；908,729,044字节的数组数据在内存中分块流式静态解析，未落地巨大解压数组，未执行pickle中的任何global/reducer。

原始 `training_y` 的113,215个float64数值，与2021 CSV `tr` **逐项完全相等，最大绝对差为0**，原始pickle CRC32与ZIP中央目录的 `2c920f6e` 一致。原始字节SHA256为 `d12e53ea1b1eaaf5dd66e13ae221369ecc1c0f3b29d1cd4593bd9c192cb5af1a`。数值既不是分钟，也没有被本项目做 `abs(min)`、标准化或其他变换。

原始one-hot词表的B表示N端Acetyl，J表示氧化M。112,994条与旧CSV的序列及变量修饰完全吻合；其余221条恰好缺失B对应的N端Acetyl，恢复表据原始数组在位置0补回，行级mapping完整保存。固定CAM依据[Bruderer2017原实验](https://pmc.ncbi.nlm.nih.gov/articles/PMC5724188/)的碘乙酰胺处理、固定Cys搜索及Guan编码去除CAM文字的共同证据，显式标注于每个Cys的1-based位置。

Guan原论文和编码器还明确把deamidated N/Q分别编码为D/E，原始数组无法区分它们与native D/E。恢复表将含D/E的98,192条标为 `chemical_identity_resolved=false`，不能把该字段当作准确的天然AA分子身份。其余15,023条不受这一具体信息丢失影响，但全部记录的 `exact_2026_source_version_verified=false`、`training_ready=false`，没有因局部修复提前解除源版本gate。

2026归档v1为32GB ZIP，已仅Range读取完整中央目录（8,621项）和小脚本；该版本未列出PXD005573源CSV，只含基础模型及空输出目录。v2仍仅有部分tar索引，不能据v1全局缺失推断v2同样缺失。两版均未整包下载。原2021训练脚本的两次划分各跳过一个边界索引，算术上可使113,215变为113,213，但这不确定2026使用的具体两条记录，因此未为凑论文数删掉任何行。
