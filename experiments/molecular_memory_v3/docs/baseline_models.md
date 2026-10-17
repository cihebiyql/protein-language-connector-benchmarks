# 固定分割的数值参考

`configs/baselines_request.json` 固定登记两个场景各自的 TRAIN 均值和中位数。两者只输出数值，不调用模型、tokenizer、CUDA 或生成流程。每个场景有独立状态、输出目录和规范合同；不能复用另一场景的 TRAIN 统计。当前实现与小样本测试完成，本文不报告新的 A1 实测成绩。

## 统计与评价口径

TRAIN 均值为 `fsum(y_train)/N_train`，中位数按排序后的中间值计算，偶数时取中间两值的算术平均。状态保存有序 TRAIN IDs、其 canonical JSON SHA256、TRAIN 标签 SHA256、sum/count、均值/中位数及场景。每个参考实际预测只需一个标量；这与 M0 的联合训练参数、信息和计算预算不同，统一标为 `unmatched_numeric_reference_not_M0_joint_model`。

validation/test 每行必须有一个有限预测，不删除失败行。输出 micro MAE=`Σ|y−ŷ|/N`、RMSE=`sqrt(Σ(y−ŷ)²/N)`、全局 Spearman（并列值平均秩的 Pearson 相关）、靶序列 macro MAE=`Σ_t mean_{i∈t}|y_i−ŷ_i|/T`。靶组由完整 target_sequence 的 SHA256 定义。MAE/RMSE 单位为 pKD，Spearman 无量纲；常数向量的相关系数未定义，保存 JSON null。`numeric_coverage=1` 表示有限数值覆盖，不等于语言生成格式有效率。这些数值定义与本 scope 的 `mm_evaluate.regression` 小样本逐值核对；LM channel 单独评价。

没有“均值和中位数中按 test 选最优”的步骤：两项都完整保留。均值/中位数无超参数，validation 只记分。可选 ridge/kNN 才按预先登记网格的 validation micro MAE 选择参数，并列取声明顺序的第一项。每个已选状态和 `selection_before_test.json` 写完后才解析锁定 test_gold，随后保存每条预测与完整指标。冻结分割、指标和现有结果均不修改。

## 规范入口与放行

Python API：`make_request(...)` 生成单场景精确请求；`execution_binding(request, project_root)` 生成规范合同的 `baseline_run_binding`；`run_baselines(..., execute=False)` 只做 SHA/分割/数据视图检查，不 fit、不解析 test gold、不创建结果目录。只有显式 `execute=True`、精确 binding 和 canonical `validate_reproduction.py` 的实时文件校验通过才可 fit。验证器本身也必须给出 SHA。输出目录 exclusive create，重复执行拒绝覆盖。

固定 suite 是预合同请求配方，不是未校验 SHA 的执行许可。`mm_baselines_cli` 在 prepared manifest 完整后读取其实际 SHA，解析成单场景请求，打印给父任务冻结。合同绑定该 SHA、精确请求、runner/data 源码 SHA 和 WSL 绝对输出路径；CLI/config 应同时列入标准合同 code/data 资产。后续改变 manifest、参数、源码或输出路径会使 binding 不相等，须重新正式冻结，不能绕过标准门禁。

工作目录为 `<PROJECT>/experiments/molecular_memory_v3_20261017`；解释器为 `<PYTHON_ENV>/bin/python`；project root 为 `<PROJECT>`；标准目录为 `<WORKBENCH>/复现/规范`。

先对每个 SCENARIO 分别解析（只读、不 fit）：

```text
<PYTHON_ENV>/bin/python -m src.mm_baselines_cli --project-root <PROJECT> --request configs/baselines_request.json --scenario SCENARIO --print-binding
```

SCENARIO 依次为 `new_peptide_known_target` 和 `new_peptide_new_target`。父任务把返回的 `baseline_run_binding` 放到各自 canonical request，使用 `src/mm_contract.py` 冻结合同。其 `execution.command` 应为如下 argv（CONTRACT 和 VALIDATOR_SHA 为冻结时真实路径/值）：

```text
<PYTHON_ENV>/bin/python -m src.mm_baselines_cli --project-root <PROJECT> --request configs/baselines_request.json --scenario SCENARIO --execute --contract CONTRACT --validator <WORKBENCH>/复现/规范/validate_reproduction.py --validator-sha256 VALIDATOR_SHA
```

随后以现有 `mm_supervise.py --contract CONTRACT --validator .../validate_reproduction.py --output NEW_SUPERVISOR_RECEIPTS` 运行，不提供 `--gpu`。建议每场景 `cpu_threads=2`、`memory_limit_bytes=536870912`、`timeout_seconds=300`；实际当前资源和剩余 deadline 由 canonical contract/supervisor 再检查。mean/median 为 stdlib 串行，300 秒包含重复文件哈希。supervisor 清空 CUDA_VISIBLE_DEVICES、执行进程组/RSS/绝对截止监控；本 runner 也检查合同 timeout/deadline。此资源建议不是实测耗时声明。

## 可选历史特征 API 与下一步

`fit_reference` / `predict_reference` 支持 `ridge52` 和 `knn52`，只用于固定注册请求或 tiny fixture。52 维由 peptide/target 各 20 种氨基酸比例加 log1p 长度（42）、两端化学类别各 3 个指示器（6）、温度/pH/buffer/actual construct 缺失指示器（4）组成，不含 label、ID、DOI、assay 文本或 ESM。TRAIN 均值和总体标准差做标准化，SD 下限 1e−6；验证和测试直接复用。新视图已把空字符串缺失规范为 null，actual construct 使用视图中显式缺失布尔值；并非旧原始输入文件的字节级重放。

ridge 含 52 个系数和一个不惩罚的截距（名义 53，不等于统计有效自由度）；TRAIN Cholesky 解二次正则正规方程。kNN 储存 52N 个标准化特征值和 N 个标签，平方欧氏距离、稳定 TRAIN 顺序破并列、`1/(d²+1e−6)` 权重，实际邻居数 min(k,N)。可登记 ridge alpha `[0.1,1,10,100,1000]` 和 k `[1,3,5,10,25]`；固定 mean/median suite 不启用这两项。stdlib 直接距离和 Cholesky 与旧 NumPy BLAS 路径不保证逐位一致，本文不称旧实现完全复现。复杂度 ridge 每参数约 O(N·53²+53³)，kNN 约 O(N_train·N_eval·52)；均值/中位数为 O(N) / O(N log N)。

旧52不使用新 joint 模型可见的 H15/H30 paired-cache。新同信息参考可另行登记 peptide/target×H15/H30 各640维的 raw-pooled means（2560），再加指定长度4、condition hash128、missing10，做 TRAIN 标准化 ridge(alpha1000) / kNN25；该新特征、cache 行对应和 SHA 绑定尚未在本模块实施，未 fit、无成绩。它仍是不同容量和目标的独立数值参考，不能称严格预算匹配。也不能把 H15/H30 未准备完成时的旧52结果当成它的结果。

## 文献与本课题的对应

均值/中位数首先检验模型是否超过只利用 TRAIN 标签位置统计的任务参照；它们不测量模型是否使用分子表示。DataSAIL（2025，[原论文](https://doi.org/10.1038/s41467-025-58606-8)，物理页1–2、4）强调分割相似性捷径，支持保留两个冻结对象泛化场景，但不提供本课题的分数保证。PepInter（2026，[原论文](https://doi.org/10.1038/s42004-026-02168-3)，物理页8/Table4、14）使用亲和力回归评价，支持单独记录数值误差；任务端点/模型/分割不同，不能直接套用其成绩。Conditional Probing（2021，[原论文](https://aclanthology.org/2021.emnlp-main.122/)，物理页2–3）将基线信息与增量信息区分，支持把简单数值参考与表示使用干预分开解读。逐篇原文版本和本地 SHA 在 `docs/literature_review.json`，历史52结果来源另见 `reports/benchmark_clarity_20261016/numeric_reference_audit.json`；旧报告不代表本 scope 已执行。

验证：`python -m unittest discover -s tests -p test_baselines.py -v` 使用临时 tiny fixtures，包含两个场景的完整落盘流程、标准拒绝/绑定误配、数据篡改、无金标准 test 输入、选点先于 test 读取、TRAIN 隔离、稳定 kNN 并列、ridge 截距与固定评价公式。测试中的假 validator 只属于临时测试目录，生产 CLI 仍要求 canonical validator 的真实 SHA 与标准审核。
