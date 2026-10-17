# v4终态分析工具

`resources/analyze_v4.py` 是CPU标准库工具。它不加载Torch或模型，不训练、不重试、不修改原始产物，也不操作Git。所有根目录由命令行显式提供；冻结计划内原始路径须在执行主机上可访问。

```bash
"$PYTHON" resources/analyze_v4.py \
  --scope-root "$SCOPE" --queue-root "$SCOPE/runs/formal_v1" \
  --output "$SCOPE/results/audit_v1"
```

`--output` 必须是全新目录。若核对已有汇总，可加 `--expected-summary /absolute/previous/formal_summary.json`；不一致会在新报告中明确显示并返回非零退出码，原汇总不覆盖。

工具以冻结 `plan.json` 和哈希链 `queue_events.jsonl` 为权威，保留全部24位置及其最后队列状态。只有队列的 `succeeded` 才进入逐项重验：

1. `accepted_attempt` 校验原监督器清理、数据分母和规范终态归档SHA，且这些SHA须与成功账本事件一致。
2. 在原执行主机重新调用规范validator，`check_files=True`、`valid_for_acceptance=True` 且没有skipped文件。此步会重读实际模型／代码／数据文件，宜在队列结束后运行以减少I/O竞争。
3. 原模型无关scorer重新读取所选权重引用、完整测试预测和验证选点；重算结果与保存的result语义逐项完全一致。
4. 实际接口参数必须与TRAIN canary冻结的底座预算相符，fit/evaluate同卡且显存allocated/reserved均在20GiB上限内。

任何重验失败保留为 `queue_status=succeeded, analysis_status=revalidation_rejected`，写明原因，从科学统计中排除。原失败、运行中、未启动位置不被转换成模型负分。队列尚无终态收据时，报告明确标为进行中的快照。

输出包含：

- `execution_registry.json`：全部24位置、队列／分析状态、重验证据、每组完整科学结果及时间／显存／RSS信息。
- `formal_summary.json`：原scorer汇总及多指标扩展；每底座M0／M3、同种子M0−M3、四格同种子交互 `(M0−M3)4B−(M0−M3)1.5B` 和单独底座主效应。
- `metrics.csv`：逐组语言完整人口MAE／RMSE／Spearman／targetmacro、JSON有效率，数值头独立指标及覆盖率、C1、预算与耗时。CSV空值代表未定义／未测，不是零。
- `contrasts.csv`：每项对照的三个原始种子、完整数量、均值及样本SD。只有3/3完整指标才能给均值／SD；交互要求每个种子四格都完整。
- `REPORT.md`：中文主要误差比较、全部位置明细、交互、计算预算与解释范围。
- `analysis_receipt.json`：分析源码和所有新输出的实际SHA；已有summary比对状态。

严格语言输出任何解析失败时，完整人口MAE保持未定义，也不把有效子集的RMSE／秩相关／靶点宏平均冒充完整测试指标。数值头不会填补语言答案。格式有效率改善不能直接写成预测误差改善。错误值不裁剪、不手改，不挑最佳种子。

报告对MAE只描述三种情况：种子不完整不能判断；三个完整种子均降低误差；方向不一致或均未改善。三个种子同向仍限于当前测试，不自动等于统计显著或新数据上的确认。跨底座总参数／FLOPs不等，底座主效应与连接器收益分开。

负面／中性结果只要完整通过技术核验，仍被接受并保留。当前没有同条件配对或独立生物donor金标准，P/H保持未定义；同权重C1只解释分支依赖。条件缺失只报告覆盖数量，不虚构配对或条件分层。

验证命令：`python -S -m unittest discover -s tests -p test_analyze_v4.py -v`。测试仅在临时目录使用合成账本、假收据和假分数，包含错误重算／错误参数的排除、24分母保留、四格交互及缺种子处理；不构成真实正式分析。
