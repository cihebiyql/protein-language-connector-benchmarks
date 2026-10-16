# A1 限定范围的双场景锁定（2026-10-15）

当前训练使用 A1 v2；完整PEPBI表恢复后新增隔离14条候选，两测试集哈希保持原样。见[补充隔离](bench_v02_pepbi_recovery.md)和[最终实测](bench_v02_panel_results.md)。下文为保留的v1划分证据。

**两个 A1 场景已完成实际划分，可立即运行同输入专项预测基线。** 从 5,122 条候选测量中隔离 2,172 条外部相关记录，保留 2,950 条测量。PEPBI 实际工作簿仍未下载成功，锁定状态明确为 `frozen_restricted_scope`，外部零暴露声明暂不成立。

| 场景 | 训练 | 验证 | 测试 | 另外隔离 |
|---|---:|---:|---:|---:|
| new_peptide_known_target | 2,065 | 147 | 152 | 586 条验证/测试候选的**同一靶点序列**未出现在训练中 |
| new_peptide_new_target | 2,065 | 443 | 442 | 0 |

两个场景各自保存 `samples.jsonl`、`test_gold.jsonl`、`manifest.json`、`benchmark_lock.json` 和 `split_exclusion_audit.jsonl`。训练可使用的主入口分别是：

- `data/processed/bench_v02/a1_locked/new_peptide_known_target/samples.jsonl`
- `data/processed/bench_v02/a1_locked/new_peptide_new_target/samples.jsonl`

每条完整保留 `sample_id, peptide_sequence, target_sequence, pkd, split, group_id, chemical_state, experimental_context, groups, record_reasons` 及原始来源信息。`split` 仅使用 `train/validation/test`；隔离记录另存。模型输入应只取协议允许的分子序列、化学状态和明确缺失的实验上下文。

## 先固定的同源与分组规则

`a1_locked/protocol.json` 在划分前写入，完全不读取 pKD 值决定分区：

1. **肽家族：**标准全局 Levenshtein 距离，`1 − distance/max_length ≥ 0.8`，符合边的连通分量归为同家族。对全部 2,950 个候选不同肽穷举长度允许的比较，使用精确 Myers 位运算距离；组成下界仅作数学安全剪枝。实际保存 2,909 条家族边和 1,852 个家族。短肽也执行同一精确距离计算。
2. **靶点家族：**MMseqs2 13-45111+ds-2，30% identity、双方至少 80% coverage。采用官方 fake-prefilter 全 query×target 对齐，**实际执行 933×933＝870,489 对**，构建 697 个连通家族。未依赖 k-mer 预筛命中来证明不存在近同源。
3. **同源外部对象：**对 2,074 个外部靶点/参考构建体序列与 933 个候选靶点执行 **1,935,042 对**实际对齐。外部查询覆盖至少 80%、identity 至少 30%，候选长靶点覆盖可低于 80%，用于保守隔离参考全长中的已知外部结构域。
4. **划分组：**同 DOI 和同肽家族先合并；双新场景再合并同靶点家族。分组按大小及固定 SHA256 次序分配，目标比例 70/15/15，不按标签分层或挑分区。
5. **已知靶点场景：**验证/测试的实际 `target_sequence` 必须在训练中出现；仅靶点家族已知仍不够。未满足的 586 条保留在审计文件，不重新调整到有利分区。

最初仅用于工程探测的 heuristic 对齐文件和 protocol 已留存为 pilot/previous 文件；最终锁只 pin 全配对产物、真实边、协议和 group audit。

## 跨库隔离实际做了什么

| 排除证据 | 命中行数（非互斥） |
|---|---:|
| PepInter 相似肽家族＋靶点家族分子对 | 1,045 |
| PepInter 来源原论文 PubMed ID（可获得的 PDBbind 来源） | 578 |
| Kliche 实验靶点构建体同源 | 831 |
| Kliche 靶点参考序列同源 | 335 |
| PEPBI 原实验结构模板链同源 | 364 |
| 已暴露旧评估肽/PEPBI 模板肽家族 | 222 |
| Kliche/PEPBI 原 ITC 来源 DOI | 64 |
| **各类并集：外部隔离的唯一记录** | **2,172** |

PepInter 使用官方亲和力 train/test 序列以及可获得的原始源表信息；相似分子对通过真实肽距离边及外部靶点对齐建立，关联到每个外部原记录 ID。相同来源论文优先整篇隔离。

Kliche 从旧实际评估清单保留已暴露肽序列；从官方 UniProt 下载 22 个靶点参考序列，并按已存在的实验构建体坐标生成排除查询。所有下载 URL/hash 保存。PEPBI 经 Europe PMC 官方全文 XML 的 Table 1 得到 32 个 binding groups、32 个 ITC 引用、**22 个独立原论文 DOI**；27 个唯一原实验 PDB 的官方 RCSB FASTA 全部下载成功。它们用于来源和同源对象保守隔离，未把预测突变结构当作实验真值。

`external_exclusion_audit.jsonl` 记录每个隔离行的全部原因和 PepInter 原记录匹配；各原因允许重叠，不能相加充当总排除量。`measurement_provenance.jsonl` 保留全部 5,122 条输入来源映射。

## 重复测量如何处理

**通过合并、平均或删除重复测量减少的记录为 0。** 2,950 条保留测量对应 2,087 种不同“肽序列＋靶点序列＋化学状态”分子对；其中 1,266 行属于存在多个测量的分子对。54 组、126 行有完全相同的来源/测量元数据，作为疑似重复保留。

同肽家族和同来源分组确保这些测量留在同一分区；独立实验是否被多个来源重复引用仍需来源审查。记录级 N、分子对 N、家族/来源组 N 必须分开报告，置信区间以 group_id 或适当的独立来源组重采样。独立实验值不被盲目平均。

## 已验证范围与剩余缺口

实际文件检查确认：两个场景均无跨分区肽家族、来源 DOI 或重复测量组；双新场景还无跨分区靶点家族。已知靶点验证/测试每条 exact target sequence 都在训练出现。最终肽边和实际靶点对齐边逐项复核，未发现跨分区违规。根执行门 `validate_panel_data` 对两个场景均通过。

PEPBI 工作簿的官方 API 返回 401，公开单文件下载返回 403；官方公开 ZIP 表单尝试返回 404。原表完整变体序列、化学状态和构建体还未获取，故 `external_isolation_verified=false`。现有保守排除覆盖原论文和结构模板，完整跨库样本级隔离仍有限制。跨来源母体区段与变体系列谱系也未全部结构化；当前明确定义的序列家族和 DOI 超组提供操作性隔离。

本次基准只支持**参考靶点下、条件缺失明确标记的广覆盖 pKD 预测**。靶点输入仍是数据库参考序列，真实 assay 构建体和条件未核实，严格同条件肽对数仍为 0。专项预测基线可以在当前限定范围训练；统一多任务连接器和完整外部金标准主张继续受原合同约束。

## 复现

```powershell
python scripts/bench_v02/split_affinity.py
python -m unittest discover -s tests -p test_v02_affinity_split.py -v
```

脚本调用已有 WSL MMseqs2，不安装依赖。真实原始对齐、命令、MMseqs 版本、协议、肽边、外部边和 group audit 都被各场景锁文件 pin SHA256。修改已锁证据会使执行门拒绝运行。旧候选数据及旧结果完整保留。
