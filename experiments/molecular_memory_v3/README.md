# 分子记忆连接器 V3：32组已验收消融与完整基线

**36组均已有终态：32组完成10轮训练、完整测试预测并通过独立验收，4组因监控程序竞态中断。现有结果尚未确立完整新架构的稳定优势。** 本轮与旧30项已验收历史结果分别登记，原件均保留。

先看[完整结果分析](results/formal/REPORT.md)、[逐seed图表PDF](results/formal/analysis_figures/report.pdf)及[实际36项执行状态](results/formal/execution_registry.json)。32组每个通道共9,794条TEST记录，严格JSON及数值覆盖均100%；远端原路径canonical校验、逐行重算和取回后的本地重算一致。

新靶点场景M3语言MAE为1.608803±0.086808，相对M2平均改善0.027539±0.049152，只有2/3种子改善；相对去掉L15的A_L平均更差。已知靶点M3只有1/3有效种子，新靶点M0只有2/3，相关完整模型均值或三seed基线配对保持null。数值头另表，不替代语言成绩。

32份[接口及buffer纯tensor权重](checkpoints/index.json)已公开，原值逐tensor校验不变；原PT和未验收partial权重留在私有完整镜像。四项失败的历史、原因和[未来版本修复补丁](results/formal/proc_race_fix/README.md)单独保存，未修改已冻结源码或恢复训练。

## 已完成的结果

固定数据为仓库的 [A1数据](../../data/a1)，两个场景分别为 TRAIN/validation/test=2062/147/152 和2063/431/442。两个场景各自拟合，不共享训练后参数或统计。

| 数值方法，测试 MAE↓ | 新肽／已知靶点，152条 | 新肽／新靶点，442条 |
|---|---:|---:|
| TRAIN均值 | 1.619744 | 1.446283 |
| TRAIN中位数 | 1.633218 | **1.444279** |
| L15/L30 pooled Ridge，α=1000 | **1.385341** | 1.592574 |
| L15/L30 pooled kNN，k=25 | 1.669993 | 1.540300 |

以上均为完整测试人口的独立数值预测。Ridge/kNN使用相同来源的分子与条件信息，但容量、目标和计算预算与联合模型不同。逐样本预测、RMSE、Spearman、靶点宏平均、六份原字节NPZ状态及验收摘要在 [results](results/)。常数预测的Spearman为null。

真实 M3 TRAIN canary 完成4次优化更新、16个样本曝光，scaler在完整TRAIN2062行拟合；另用固定TRAIN前8行验证生成、数值头、C1和保存链路。未来80个答案token变化后，答案前数值输出逐位不变。38项独立核验通过。该早期试跑严格JSON为0/8，完整语言MAE为null；它是工程资格，不能当正式测试成绩。见 [试跑原始输出](results/train_canary/) 和 [独立审查](docs/real_canary_review.md)。

## 架构与固定消融

冻结 ESM2-150M／Qwen2.5-1.5B，训练两层投影、类型与来源标记、条件bank、关系bank、内部Reader、路由和答案前数值头。每个配置1,618,568个训练参数，58×128个float32逻辑memory槽，共29,696字节／样本；这不是实测设备通信量或相同FLOPs声明。

| 配置 | 来源视图 | 路由 | 关系控制 |
|---|---|---|---|
| M0 | L30/L30 | 固定非零查询 | 肽self |
| M1 | L15/L30 | 固定非零查询 | 肽self |
| M2 | L15/L30 | 问题查询 | 肽self |
| M3 | L15/L30 | 问题查询 | 肽→靶点 |
| A_L | L30/L30 | 问题查询 | 肽→靶点 |
| A_Q | L15/L30 | 固定非零查询 | 肽→靶点 |

预登记6配置×2场景×种子17/42/2026，共36项，统一10个完整epoch、训练batch4、生成batch8。按完整validation答案token NLL最小选点，精确相等取最早epoch；原严格JSON／greedy最多24token及完整失败分母保留。语言生成和数值头分别评分，缺seed不补均值。队列于2026-10-17 05:21（北京时间）结束，32成功／4失败，详见[终态及冻结SHA](results/remote_queue_status.json)。

三类评价接口分别为量测预测、同一checkpoint的on/off干预、条件探针。32组正式TEST预测和C1均已取回；条件探针目前完成代码和CPU测试，真实结果尚未取得。C1语言G_off全部为正，但它证明支路影响，不等于泛化精度优势。数据缺少可靠配对、事实问答和解释gold，对应监督保持关闭。

## 现在即可复核

在仓库根目录，用Python3.10或更高版本：

```bash
python experiments/molecular_memory_v3/analyze.py
```

此命令仅用标准库，核对全部基线、32组完整TEST语言／numeric／C1、三seed缺失规则、文件SHA和导出tensor字节；不加载模型、训练或调用GPU。

`src/` 是本轮实现源码存档，16个模块保持原字节；`tests/` 随附CPU测试，归档测试只将规范文件路径改为本目录的`standard/`。训练依赖包括原运行的Torch2.6.0、Transformers4.48.1、NumPy1.26.4等；CPU数值基线实际使用NumPy2.2.6。测试和模型入口的依赖范围不同，不能把默认无模型复核解释成全部训练依赖已经安装。

原科研目录布局与本公共仓库布局不同。完整训练需按清单准备原数据闭包、预训练模型、已校验缓存和新的任务合同；当前配置中的历史截止仍保留，不能直接启动已到期合同。开发配置和正式配置分别保留；本次取回的`formal_10e_b8_v1.json`及`evaluation_10e_b8_v1.json`保持原字节，原含机器路径的队列计划以SHA索引。

## 文献与公开范围

架构依据、具体页码及创新边界见 [文献核验](docs/literature_review.md)、[架构审查](docs/architecture_review.md) 和 [模型实现](docs/model_implementation.md)。冻结LM跨注意力、问题压缩、分层表示已有先例，当前组合的有效性仍需消融结果支持。

PPIKB派生数据遵循 [原仓库数据声明](../../docs/PUBLICATION_SCOPE.md) 的CC BY4.0要求；本目录预测和拟合状态不改变底层数据来源。第三方论文及预训练模型继续适用各自许可；本更新未为原代码新增通用开源许可。

原科研文件保持原样。公开副本省略私有路径、SSH配置、资源身份和原始聊天；初次框架公开清单见[publication_manifest.json](publication_manifest.json)，取回结果清单见[formal/publication_manifest.json](results/formal/publication_manifest.json)。公共验收摘要和审计JSON是原记录的去路径派生说明，数值保持一致；不冒充原合同字节。
