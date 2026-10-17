# Qwen 底座与连接器对照：v4

本轮在保留原始 A1 数据与 v3 结果的基础上，比较 Qwen2.5-1.5B-Instruct 和 Qwen3-4B-Instruct-2507。每个底座分别训练 M0 与 M3，两个数据场景、随机种子17/42/2026，共24个全新训练位置。旧32组通过验收的结果只作历史参考。

目前配置为开发草案，正式训练必须先完成 TRAIN canary、冻结共同预算和实际模型文件版本。`status=draft`、未签名模型或 `training_budget.status=UNFROZEN` 均不能进入正式结果。

| 固定项 | 约定 |
|---|---|
| M0 | 双路H30/H30、静态路由、肽自身关系；保留与M3相同活动模块容量 |
| M3 | H15/H30、问题相关路由、肽到靶点的关系 |
| 接入位置 | 1.5B第14块后；4B第18块后，均为预定半深度 |
| 通信 | 58槽×128维×float32＝29,696逻辑字节，两底座相同 |
| 数据 | A1 `a1_locked_v2` 两个场景的原始8个文件SHA、划分与标签不变 |
| 训练 | 暂定10完整epoch、microbatch2×累积2＝有效批量4；canary后统一冻结 |
| 损失／选点 | CE＋1×TRAIN标准化数值MSE；全验证集答案token NLL最低，完全相等取最早epoch |
| 生成 | greedy、最多24token；相同生成批量在资源canary后冻结 |
| 时限 | 2026-10-17 17:26:57至2026-10-18 03:26:57，北京时间；重试不重置 |

主要语言指标沿用严格单键JSON解析及完整测试人口MAE，任何生成失败使完整人口MAE未定义，同时报告有效率及失败ID。数值头的原始pKD MAE、RMSE、Spearman、按靶点宏平均单独报告，不替代语言答案。C1在同一所选权重上执行正常／关闭连接器的对照，保持完整测试分母。

每个底座内先比较同种子的 M0−M3；再计算 `(M0−M3)_4B−(M0−M3)_1.5B`。只有同一项有3/3完整种子才报告均值和样本标准差。跨底座的总接口参数、LM参数和FLOPs不同，因此底座主效应独立报告，不宣称绝对同参数对照。

`src/mm_queue.py` 按种子→场景→底座→连接器的固定顺序领取任务。支持冻结 `candidate_gpu_devices` 与初始 `ready_assignments`：候选卡不是可启动授权，每卡必须先取得独占锁，通过新鲜的GPU利用率／计算进程、CPU、RAM和存储检查，才能领取下一个科学任务。忙卡仅追加资源观察、20秒后重查，不占任务也不记模型失败；卡自然释放后可加入。合同创建和实际启动仍分别重查资源。只有监督器退出清理、模型无关评分及规范终态归档全部接受，任务才成功；中断、失败和未启动位置保留。

```bash
python -S -m unittest discover -s tests -p test_evaluate.py -v
python -S -m unittest discover -s tests -p test_queue.py -v
python -S -m src.mm_evaluate validate-data --data-root /absolute/project/data/processed/bench_v02/a1_locked_v2
python -S -m src.mm_evaluate validate-protocol --protocol docs/evaluation_protocol.json --development
# 真实 TRAIN canary 后由项目负责人写入全新 frozen 配置、评价协议与资源计划，再运行：
python -m src.mm_queue --plan /absolute/scope/configs/frozen_queue.json --output /absolute/scope/runs/formal
```

`tests` 中的假训练、假权重和模拟监督器只验证控制与评分逻辑，均不构成科学结果。Linux进程组／watchdog／动态资源队列测试需在Linux或WSL执行；Windows会显式跳过这些测试。
