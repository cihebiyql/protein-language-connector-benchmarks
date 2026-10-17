# 正式冻结、执行与取回

`resources/freeze_formal.py` 是无模型、无GPU的前置冻结工具。它不会启动训练、修改既有数据或恢复旧任务。所有执行根目录、旧输入根目录、Python、规范目录、环境锁和资源证据均须显式传入，适合以原始源码公开。

先等待两个实际 TRAIN canary 退出并清理各自进程组。工具要求每个canary完成M3、种子17、已知靶点场景、4次实际优化／16个训练例，未打开验证／测试；检查完整TRAIN长度扫描、最长有效批量及8行最长生成、8行有限数值输出、答案未来token不影响数值头、真实显存上限和源码／模型／缓存／输入SHA。前期语言JSON无效率可作为学习诊断保留，不要求4次更新就取得预测提升。

工具读取 **`runtime.training_performance.elapsed_seconds`** 计算训练速率；不读取不存在的 `training.elapsed_seconds`。默认冻结10个epoch、micro2×accum2＝有效4、生成批量8。如果实际TRAIN资源测量要求统一减少epoch，必须在正式测试前显式提供新epoch数和 `--budget-change-reason`。训练损失、检查点选择及原测试评分不因分数变化。

已有完整canary若尚未做规范终态归档，可加 `--archive-existing-canaries`：工具只归档已退出的既有产物，以固定TRAIN8行／优化4次／16例／无测试的完成合同调用原规范archiver，并保存显存限制收据的字节一致副本；它不重跑模型。仅监督器返回成功不能代替规范终态接受。

以下POSIX示例中，路径变量由执行负责人填写为真实绝对路径；路径不能指向旧任务的新输出位置。

```bash
"$PYTHON" resources/probe_gate.py --gpu 4 --output "$SCOPE/resources/ready4_after_canaries.json"
"$PYTHON" resources/freeze_formal.py \
  --project-root "$NEW_PROJECT" --input-project-root "$CANONICAL_OLD_PROJECT" \
  --scope-root "$SCOPE" --standard-root "$STANDARD" --python "$PYTHON" --host node1 \
  --environment-lock "$SCOPE/resources/environment_cpu_inspection.json" \
  --canary-config "$SCOPE/configs/canary_m2_a2_g8.json" \
  --evaluation-template "$SCOPE/docs/evaluation_protocol.json" --tag v1 \
  --canary qwen25_1p5b "$CANARY25/artifacts/run_manifest.json" "$CANARY25/archive" \
  --canary qwen3_4b_2507 "$CANARY4/artifacts/run_manifest.json" "$CANARY4/archive" \
  --archive-existing-canaries --ready 4 "$SCOPE/resources/ready4_after_canaries.json"
```

资源证据必须是180秒内的实际ready观察，CPU、RAM、存储、GPU计算进程与利用率均满足原门槛。初始ready只作为冻结时的事实；全部8卡都属于候选，后来资源自然释放时，队列通过原资源门槛后才领取下一任务。冻结计划不修改，观察追加到账本。合同创建与实际启动另行重查；重试不延长2026-10-18 03:26:57北京时间的终止边界。

冻结工具生成全新、不可覆盖的 `configs/formal_v1.json`、`docs/evaluation_v1.json`、`configs/queue_v1.json`、`docs/registry_v1.json` 及 `docs/freeze_v1.json`，其中包含精确启动命令。它对全部 `src/*.py`（含 `mm_data_evaluation.py`）、自身、全部模型文件、旧输入／视图／缓存负载及规范文件重新计算实际SHA。若后半段验证失败，已写出的新文件保留为审计证据；不得把它们当作成功冻结，也不得覆写，可修复根因后使用新的显式版本。

取回和分析按以下固定流程进行：

1. 保存冻结配置、评价协议、24位置登记、源码及SHA，保存完整追加账本和所有失败／未启动记录。
2. 在执行主机上先逐个调用 `mm_queue.accepted_attempt`，独立复核监督器退出清理、所选检查点、科学评分和规范终态接受。未通过者保留，不能因为有预测文件就算成功。
3. 仅对上述接受的任务重新执行模型无关评价，并与保存结果逐项核对。保留原始生成文本、token、数值头输出、C1记录和选择历史；完整测试分母不变。
4. 构建24位置汇总，分别给出每底座的M0／M3和同种子M0−M3；仅在同种子四格完整时计算底座×连接器交互效应。任何指标不足3/3时，均值及样本标准差保持未定义，列出缺失种子与原因。
5. 固定排序，绘制逐种子误差与覆盖率，同时报告数值头和语言答案。跨底座参数／FLOPs不同，底座主效应独立报告。历史v3结果只作参考。
6. 将全部源码、公开所需协议、接受／失败清单、逐行预测、选择历史、接口所选权重和分析同步GitHub；数据与文献继续遵守既有逐文件许可。公开前核查凭证／私人路径，科学数值、原始预测和SHA来源链不得静默改写。推送后核验远端提交及关键文件的匿名取回SHA。

验证可运行 `python -S -m unittest discover -s tests -p test_freeze_formal.py -v`。这些测试只在临时目录构造合成收据和假资产，明确不代表真实canary接受、正式冻结或科学结果。
