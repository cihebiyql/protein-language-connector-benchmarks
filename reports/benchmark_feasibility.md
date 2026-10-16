# PepBridge 基准可运行性评估

评估日期：2026-10-15，Asia/Shanghai。工作目录：`PROJECT_ROOT`。

**T1 已用真实实验数据跑通清洗、固定分组、序列近邻和 ESM2＋MLP；ESM2→Qwen 六组连接的前向、反向及参数更新也已通过。** 本机 RTX 5080 16GB 支持当前小模型的工程试跑。下一项核心实验是同一数据划分上的 L30-J0 与 L15-J14 正式训练比较。T3 的主要工作在具体肽对和定量标签对齐。

## 已获取材料

已完整读取原对话的五轮用户消息和回复，分别保存为 [可读对话](../docs/PUBLICATION_SCOPE.md) 和 [原始读取结果](../docs/PUBLICATION_SCOPE.md)。采用最后一轮确定的三测试方案；对话中用于展示的假数字没有进入结果文件。

| 材料 | 实际取得 |
|---|---|
| 原始实验资料 | Benz2022、Kliche2024、PEPBI 三篇原文；两份论文附录；两篇各11个 EV Excel；Kliche 正文 Table 1；PEPBI 官方 RIA 属性说明 PDF |
| 方法文献 | 10份方法 PDF（含 OPUS 补充表），另有 ProtNote 完整 XML、Pep-Instructions 出版社全文 HTML |
| 代码 | 12个浅克隆仓库：OPUS-PLLM、ProtT3、MiniMind、MiniMind-V、C2C、Draft-KV、LatentMAS、StateBridge、ProtNote、Prot2Text-V2、BioReason-Pro、Pep-Instructions |
| 本轮模型 | ESM2-150M、Qwen2.5-1.5B-Instruct 的固定 revision、配置、tokenizer、模型卡和 safetensors 权重；两份权重 SHA-256 与官方 LFS 元数据一致 |
| 总范围清单 | 五轮对话中的64项方法、模型及主要证据资源逐项记录，区分已取得、延后、歧义和下载受阻 |

本地共有16份有效 PDF。仓库 commit、许可、入口及原始资源 URL 见 [仓库清单](../docs/PUBLICATION_SCOPE.md)，全部提及资源的处理状态见 [范围清单](../docs/PUBLICATION_SCOPE.md)。这12个外部仓库保留原貌，只做静态审阅。运行试验使用本项目脚本和既有环境，没有安装其旧依赖栈。

下载缺口有两项：PEPBI 的主表及结构包下载网页返回403、API返回401；BioReason-Pro 论文PDF多个官方入口返回429/403。对应公开元数据已保存；PEPBI 的关联 Zenodo 仅提供属性说明 PDF。后续补取按清单中的官方 URL 和摘要校验。

## T1 真实数据和基线成绩

Benz EV4 的 `ELM Benchmarking Dataset` 实读3,049条记录，合并为2,724条唯一16残基序列，其中322条为多标签。该表实际有34个 bait；论文研究的35个结构域不能直接当成本次候选类别数。模型输入仅为序列，来源名称、GO、基序和正确靶点等审计字段不送入序列预测器。

同一序列合并阳性靶点；同母体 accession/gene 整组连接，再按序列一致性≥80%且双方覆盖≥90%的可含gap阈值合组。分组 seed=42，70/15/15 是组数比例。

| 划分 | 独立序列 | 独立组 |
|---|---:|---:|
| 训练 | 1,914 | 1,093 |
| 验证 | 393 | 234 |
| 测试 | 387 | 235 |
| EV8关联保守排除 | 30 | 11 |

EV8的35行肽身份（含探针）都作为保守排除锚点。3行原始 gene 字段遭 Excel 日期化，保留异常值并使用 accession 分组。直接跨集合相似性检查未发现达到上述阈值的配对。划分已验证重跑字节一致，见 [固定划分](../docs/PUBLICATION_SCOPE.md)。

以下均为相同387条测试序列上的实际计算结果，分母和逐样本排名已保存：

| 方法 | Known-positive Recall@3 | MRR |
|---|---:|---:|
| 训练集靶点频率 | 0.334195 | 0.315419 |
| 训练集序列1NN | 0.507321 | 0.526817 |
| 冻结 ESM2-150M 第30层均值＋MLP | **0.754953** | **0.681412** |

B1 使用640→256→34的预测头，训练 seed=17、AdamW、学习率3e-4、batch64，上限30轮；仅按验证集MRR选 checkpoint，连续5轮未改善停止。实际运行27轮，选择第22轮。均值和标准差只在训练特征上拟合，ESM的BOS/EOS/padding已剔除。当前缓存两层的池化表示，供后续读取层比较使用。

目标函数将每条序列的权重均分到其已记录阳性，采用全候选softmax交叉熵。它学习已知阳性排序，并会压低其他候选的相对分数；因此不是未知标签无惩罚或无偏的PU学习。未记录关系在数据中仍保留为未知，不作为实验证实阴性。

本次为单训练种子的工程试跑。EV8到定量24对的身份、完整同源家族和D2跨源映射尚待完成；最终基准需在完成该映射后重新冻结划分，再做三种子和组级置信区间。当前成绩支持T1流程可运行。

证据：[B0结果](../results/legacy/b0_metrics.json)、[B1结果与训练轨迹](../docs/PUBLICATION_SCOPE.md)、[B1逐样本测试排名](../docs/PUBLICATION_SCOPE.md)、[预测头checkpoint](../docs/PUBLICATION_SCOPE.md)。

## 两模型连接和 B2 流水线

连接模块为LayerNorm(640)→Linear(640,256)→GELU→Linear(256,1536)→LayerNorm(1536)，加一个门控标量，共563,201参数。基础模型BF16、连接模块FP32。ESM提取第15/30层；Qwen分别在输入端、第7层后、第14层后加入消息，只改预留分子位置。

在64残基合成字母序列、batch2、总张量长度512下，各接口执行2个AdamW更新；有效非padding长度为109。每组均检查损失有限、连接参数梯度非零且有限、连接权重更新、冻结语言模型无参数梯度、关闭消息会改变答案位置logits。

| ESM读取层 | Qwen写入位置 | 前向与反向 | CUDA峰值allocated MiB |
|---|---|---|---:|
| 15 | 输入端 | 通过 | 7514.36 |
| 15 | 第7层后 | 通过 | 6952.48 |
| 15 | 第14层后 | 通过 | 6399.84 |
| 30 | 输入端 | 通过 | 7505.61 |
| 30 | 第7层后 | 通过 | 6952.48 |
| 30 | 第14层后 | 通过 | 6399.84 |

最大CUDA reserved为8354 MiB。这里是本进程PyTorch分配/保留显存，另有桌面显存开销；ESM特征提取后已卸载编码器。完整有效512-token上下文、更长答案和梯度累积的正式训练需单独计时计显存。

这项检查建立工程可运行性。64残基输入端两组的两步损失有上升，已原样保存；当前没有六组真实数据训练成绩，不能据此比较层位优劣。16残基的早期记录保留为历史版本，当前64残基记录对应现有脚本，二者各自带源文件指纹。

[64残基接口检查结果](../results/legacy/bridge_smoke_64aa.json)；[连接模块实现](../historical/scripts/bridge.py)。

B2也完成了实际调用：将B1固定测试顺序的前3条预测送入同一个冻结Qwen，3/3生成可解析JSON并保留原Top3顺序。原始输出包含JSON代码围栏，解析前按明确规则去除。排序继承B1；解释遵循约束模板，这3例只检查流水线，不评价机制理解。结果见 [B2调用记录](../results/legacy/b2_smoke.json)。

## T3 标签需要先修正

Kliche EV6实际含1,773个结构域—变体组，其中173 diminished、193 enhanced，另1,407 no_effect；201个显著组具有单肽配对支持。应回到EV1的relationship_id/relates_to、具体16肽和计数选择直接支持的肽对，不能把组标签赋给所有重叠肽。

EV9实为18条FITC探针饱和结合KD。真正的配对测量来自正文Table1，其29个测量行含口袋/追加靶点等重复，已展开合并格保存为 [Table 1](../docs/PUBLICATION_SCOPE.md)。独立变体系列与有效方向分母需结合EV8逐项确认，尤其是BCL11A、ZNF526和KMT2D的命名/坐标冲突。

原对话CDC45–KPNA7例子的野生型值需改为**6.6±0.7 μM**，变体为**380±15 μM**；原用7.5 μM来自另一FITC-MYC探针。`n.b.`、`n.d.`、界限测量和小于两倍的差异各自保留。完整核查见 [数据审计](data_inventory.md)。

## 公开方法复现成本和近期研究

OPUS原脚本使用ESM2-650M和旧量化/注意力依赖；其已训练接口不能直接替换为本次150M编码器。ProtT3使用本轮相同的ESM150M，适合作为后续Q-Former对照，但README评测命令需要按源码修正。其他领域大模型的显存、许可和输入条件分别记录在 [方法评估](repository_assessment.md)。

新取得的[When Do Biological Reasoning Models Use Their Biological Inputs?](https://arxiv.org/abs/2610.00898)直接研究生物模型的输入扰动与冲突，发表时间为2026-10-01预印本。它使T2成为已有直接先例的必要有效性检查。项目的方法增量需由公平条件下的内部读写层比较、真实实验差异和超出强预测流水线的收益共同建立。

## 复现入口和验证

保留两个既有运行环境，无新增依赖。Windows默认Python3.11.9负责openpyxl/NumPy数据准备；WSL Ubuntu-22.04的 `LOCAL_HOME/.venvs/peptide-omnipanel-plm5080/bin/python` 为Python3.10.12、PyTorch2.12.1+cu130、Transformers4.57.6，负责模型。

在工作目录PowerShell执行：

```powershell
python scripts/prepare_benz_baseline.py
python -m unittest discover -s tests -p test_benz_baseline.py -v
wsl.exe -d Ubuntu-22.04 -- bash -lc 'cd PROJECT_ROOT/ && LOCAL_HOME/.venvs/peptide-omnipanel-plm5080/bin/python scripts/run_b1_pilot.py'
wsl.exe -d Ubuntu-22.04 -- bash -lc 'cd PROJECT_ROOT/ && LOCAL_HOME/.venvs/peptide-omnipanel-plm5080/bin/python scripts/smoke_bridge.py --residues 64 --output results/bridge_smoke_64aa.json'
wsl.exe -d Ubuntu-22.04 -- bash -lc 'cd PROJECT_ROOT/ && LOCAL_HOME/.venvs/peptide-omnipanel-plm5080/bin/python scripts/smoke_b2.py'
```

Windows环境13项数据/划分/指标测试通过；WSL环境4项桥/指标测试通过；本项目脚本语法检查通过。两个环境各执行其适用测试：GPU venv未安装openpyxl，不在其中运行数据准备测试。独立审查确认桥索引、slot范围、梯度通路、训练统计隔离、验证选点、逐样本指标和指纹一致。

下一轮顺序：先完成G1/D2身份映射并冻结最终划分，再在相同预算下训练L30-J0和L15-J14，随后补全六组、三训练种子、正确/关闭/错配/恒定消息对照及组级区间；最后接入T3配对和PEPBI外部面板。
