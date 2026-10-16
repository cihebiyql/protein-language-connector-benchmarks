# 冻结预测器重放与可靠性基础测试（2026-10-16）

已重新执行 **18 个冻结 checkpoint 的完整测试集推理、6 个确定性控制拟合，以及12个A1 checkpoint的90%/95%残差区间诊断**。18个神经模型均通过CPU重放，B1全部候选排序与原GPU结果一致。此次使用本项目已有权重；模型来源为此前的专项基线训练。

最终进程PID 200687，CPU线程2，耗时54.90秒，峰值RSS 1,446,940,672 bytes（1.348 GiB），退出码0。环境为Python3.10.12、NumPy2.2.6、Torch2.12.1+cu130。全过程使用CPU。

## 重放结果

| 面板 | 实际重放 | 每模型测试量 | 最大原GPU/本次CPU分数差 | 判定 |
|---|---:|---:|---:|---|
| A1，新肽/已见靶点 | 6 | 152 | 4.76837158e-07 | 全部通过rtol=1e-5、atol=1e-4 |
| A1，新肽/新靶点 | 6 | 442 | 9.53674316e-07 | 全部通过rtol=1e-5、atol=1e-4 |
| B1，未见类别 | 6 | 76×31候选 | 7.62939453e-06 | 分数通过rtol/atol=1e-4；所有完整排序一致 |

运行前核验了当前A1 v2及B1物理锁、原始结果与权重指纹、训练脚本、缓存脚本、缓存manifest、实际使用的特征文件、冻结ESM2/Qwen模型权重和词表。A1重新从train重算数值条件统计、标签均值及标准差，并与checkpoint逐项相等。三十轮日志及最早最佳验证选点也通过核对。运行结束再次核对已有输入、报告、权重、结果与脚本指纹，均保持原值。

B1训练发生在A1 v2脚本入口修订前。其原始脚本快照sha与checkpoint登记一致，且13个非main模型/训练定义与当前脚本的AST完全相同。相关版本验证记录保留在每个B1 replay.json中。

## 确定性控制重新拟合

每个场景独立运行现有run_simple_baselines.run；标准化在train拟合，超参按validation MAE选择，test用于最终评价。

| 场景 | 控制 | 验证选择参数 | 测试MAE/pKD | 与原测试预测最大差 |
|---|---|---:|---:|---:|
| 新肽/已见靶点 | train_mean | None | 1.619744 | 0 |
| 新肽/已见靶点 | composition_chemistry_ridge | 1000.0 | 1.592669 | 2.13e-14 |
| 新肽/已见靶点 | composition_chemistry_knn | 25 | 1.711140 | 1.78e-15 |
| 新肽/新靶点 | train_mean | None | 1.446283 | 0 |
| 新肽/新靶点 | composition_chemistry_ridge | 1000.0 | 1.552233 | 1.33e-14 |
| 新肽/新靶点 | composition_chemistry_knn | 25 | 1.592355 | 2.66e-15 |

六组最终选参和测试预测均复现，测试数值差按rtol/atol=1e-10核验。跨Windows/Linux重拟合时，新肽/新靶点kNN的非选中k=3和k=5验证MAE分别相差0.00134825和0.000249975；最终选中的k=25及其验证分数保持一致。完整候选验证分数与差异均记录在summary.json，避免把最终预测复现误报为所有浮点试验逐字节一致。

## 逐checkpoint可靠性诊断

对每个已选中的A1权重重新推理完整validation，计算绝对残差，并取第ceil(n×名义覆盖率)个顺序统计量q。测试区间为prediction±q；下表宽度为2q，单位为pKD。test标签没有进入阈值拟合。每个权重均保留validation_predictions.npz、test_predictions.npz和calibration_thresholds.json，能够逐样本重算。

| 场景 | 模型 | seed | 测试MAE | 90%实测覆盖 | 90%区间宽度 | 95%实测覆盖 | 95%区间宽度 |
|---|---|---:|---:|---:|---:|---:|---:|
| 新肽/已见靶点 | BASE01 | 17 | 1.4619 | 87.50% | 5.2169 | 93.42% | 6.2150 |
| 新肽/已见靶点 | BASE01 | 2026 | 1.4426 | 89.47% | 5.2916 | 92.11% | 5.7466 |
| 新肽/已见靶点 | BASE01 | 42 | 1.4762 | 89.47% | 5.5440 | 92.76% | 5.8590 |
| 新肽/已见靶点 | MEANPOOL | 17 | 1.4687 | 86.84% | 5.4129 | 93.42% | 6.1253 |
| 新肽/已见靶点 | MEANPOOL | 2026 | 1.4563 | 89.47% | 5.2433 | 92.76% | 5.9716 |
| 新肽/已见靶点 | MEANPOOL | 42 | 1.4848 | 86.18% | 5.3600 | 92.11% | 5.9904 |
| 新肽/新靶点 | BASE01 | 17 | 1.4202 | 93.44% | 6.8304 | 97.96% | 7.8793 |
| 新肽/新靶点 | BASE01 | 2026 | 1.4127 | 94.12% | 6.6861 | 97.74% | 7.8091 |
| 新肽/新靶点 | BASE01 | 42 | 1.5395 | 93.89% | 7.4299 | 97.06% | 8.3773 |
| 新肽/新靶点 | MEANPOOL | 17 | 1.4211 | 92.53% | 6.7117 | 97.96% | 7.8789 |
| 新肽/新靶点 | MEANPOOL | 2026 | 1.4216 | 94.57% | 6.7762 | 97.96% | 8.0893 |
| 新肽/新靶点 | MEANPOOL | 42 | 1.5732 | 94.12% | 7.2909 | 96.38% | 8.3238 |

已见靶点场景的90%实测覆盖为86.18%—89.47%，95%实测覆盖为92.11%—93.42%；双新场景的区间更宽，90%实测覆盖为92.53%—94.57%，95%实测覆盖为96.38%—97.96%。这些结果直接显示不同划分下的覆盖与宽度变化。

本轮区间属于描述性全局残差诊断：validation此前已用于checkpoint选择，校准样本并非独立；家族划分存在分布偏移。因此表中保留实测覆盖，不登记有限样本覆盖保证。三种seed分别报告，不将seed间差异解释为单个样本的预测不确定性。该诊断也不计为外部蛋白工程不确定性论文的完整复现。

## 验证与产物

新增6个测试在Windows和既有WSL环境全部通过；内容覆盖validation专属拟合、test标签不影响阈值、空值/非有限值/重复ID拒绝、校准测试ID隔离、浮点容差、非选中试验差异登记及选参变化拒绝。脚本和测试文件通过py_compile。原始退出日志与调试尝试均留存，正式结果以根目录summary.json和exit.json为准。

- [机器可读汇总](../results/legacy/foundation_20261016/replay_predictors/summary.json)
- [实际运行合同及版本](../results/legacy/foundation_20261016/replay_predictors/run_contract.json)
- [最终运行日志](../docs/PUBLICATION_SCOPE.md)
- [WSL六项测试日志](../docs/PUBLICATION_SCOPE.md)
- [执行入口](../historical/scripts/foundation_20261016/replay_predictors.py)
- [测试代码](../historical/tests/test_foundation_replay.py)

完整神经输出包含18个测试预测矩阵、12个验证预测文件、12组90%/95%校准阈值；两个BASE00目录保存6组状态及完整测试预测。重放已经完成，可复查现有产物。脚本拒绝覆盖已完成summary；下一次执行应显式指定新的--output目录。

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc 'cd PROJECT_ROOT/ && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOCAL_HOME/.venvs/peptide-omnipanel-plm5080/bin/python scripts/foundation_20261016/replay_predictors.py --output results/foundation_20261016/replay_predictors_new_run'
```

A1实际实验构建体/条件尚未完全核清，B1未见测试仅含两个类别；本报告沿用这些物理锁的范围。统一多任务连接器、真实生成通道、外部原系统与A2/B2/C面板由对应任务单独验收。
