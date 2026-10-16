# PEPBI 官方实表恢复与 A1 v2 补充隔离

**PEPBI.xlsx 已通过真实浏览器的官方公开下载恢复，文件哈希与 Dryad 完全一致。** 全表核验新增发现 14 条原 A1 保留候选符合保守外部靶点家族排除条件；已生成独立 `a1_locked_v2`。旧锁、旧数据和旧训练结果保持原样，供审计追溯。

恢复方式为 Edge 浏览器打开 [官方 Dryad 页面](https://datadryad.org/dataset/doi:10.5061/dryad.wstqjq2wk)，点击真实 `Download PEPBI.xlsx` 链接。正常浏览器安全验证完成后，文件落入系统 Downloads；同时官方完整 52.6 MB 数据包按钮也成功。未使用凭据或绕过访问控制。下载事件回调超时，实际文件通过本地大小和官方 SHA256 复核确认成功。

- 大小：**217,930 bytes**。
- SHA256：`0f3ae5e9021b5c0e925114c5b53a0cb441de90adc42134e5144d4ac9c4826a24`。
- 独立保存位置：`data/raw/bench_v02/ppikb/pepbi_official/browser_recovery/PEPBI.xlsx`。
- 官方完整 ZIP、下载清单及实际构建体双向对齐也保存在该目录。

## 全部实际记录已进入审计

工作簿包含 **329 行结构记录、84 条不同肽序列、58 条不同实际蛋白构建体序列**。每行完整保存蛋白 tag、双方突变、KD 和误差、温度、实验结构模板及结构 copy 标记。通过 Binding Group 与官方论文 Table 1 的 ITC 引用精确标准化关联，329 行均映射到来源，合计 22 个独立原论文 DOI，无来源关联缺失。

329 个结构记录包含同一实验对象的不同结构副本，不能直接作为 329 个独立实验。工作簿没有明确肽 N/C 末端状态或环化列，样本清单将这些字段保留为 null；外部隔离按序列家族保守执行，不以未知化学差异证明两记录不同。

完整可追踪清单位于 `a1_locked_v2/pepbi_complete_sample_inventory.jsonl`，保留原始 73 列 metadata 及逐行来源 DOI。

## v2 保持原分区，只追加外部事实隔离

先对原始 **5,122 条 broad 候选全池**应用实际 PEPBI 肽/构建体/DOI 排除规则，再从每个已锁场景中过滤命中行。使用原肽 80% global-edit identity 和外部靶点 30% identity、80% 查询覆盖阈值；实际 58×933 靶点对执行全配对比对，未使用模型分数。

全池有 569 行命中实际 PEPBI 原来源或序列家族，其中 555 行已被 v1 排除，新增 **14 条唯一候选**。新增命中均为靶点家族证据；当前保留行中，实际 PEPBI 肽家族、精确肽、同源分子对和原 DOI 新命中均为 0。沿用既定“外部 PEPBI 靶点家族整体隔离”规则，14 行全部进入新版隔离。

| 场景 | v2 训练 | v2 验证 | v2 测试 | 新增隔离 |
|---|---:|---:|---:|---:|
| new_peptide_known_target | 2,062 | 147 | 152 | 原训练 3 行 |
| new_peptide_new_target | 2,063 | 431 | 442 | 原训练 2 行＋验证 12 行 |

已知靶点场景重新核对 exact target sequence 在训练出现，额外依赖剔除为 0。两场景 **test_gold 的 SHA256 均与 v1 完全一致**；没有重新挑选测试分区。保持原划分可防止补充数据检索成为基于旧模型表现调整测试的机会。

## 证据与执行状态

`a1_locked_v2/coverage_supplement.json` 与各场景 manifest/lock 关联父锁、父 samples 和父 test_gold 的哈希；pin 官方工作簿、完整样本清单、解析器、实际对齐、肽边、全池排除审计和新样本。根执行门对两个 v2 场景均通过，新增 4 项恢复验证测试通过。

PEPBI 的实际记录来源和序列家族隔离证据现已补全。整体仍保留 `frozen_restricted_scope`：PepInter 部分原始 DOI/化学信息和跨来源母体变体谱系尚未完全恢复，实际 assay 条件也不是该 broad 预测面板的完整输入。这一状态允许同协议专项基线运行，并保留完整外部零暴露主张的边界。

复现：

```powershell
python -m scripts.bench_v02.supplement_pepbi
python -m unittest discover -s tests -p test_v02_pepbi_recovery.py -v
```
