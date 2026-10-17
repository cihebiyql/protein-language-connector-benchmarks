# 同源信息的独立数值参考

新增 ridge2700_alpha1000 与 knn2700_k25，使用 joint 模型对应的同一份完整 H15/H30 缓存、序列长度和允许的条件字段。两者按场景独立 TRAIN 拟合，固定参数，无 validation 调参、无语言生成。它们检验这些输入经简单汇聚后对 pKD 回归的任务效用；参数容量、训练目标和实际计算均与 joint 模型不同，分组为 `same_source_information_unmatched_numeric_reference`，不能称严格预算匹配。

实现、固定配置与 CPU tiny fixtures 已完成。本模块尚未对真实 A1 拟合；真实执行由父任务冻结规范合同并经 supervisor 放行。新基线不替代此前已保存的均值/中位数或旧52维历史参考。

## 2700 维的唯一固定定义

| 位置，0起算 | 维数 | 内容 |
|---|---:|---|
| 0–639 | 640 | peptide H15，8个已保存槽的等权均值 |
| 640–1279 | 640 | peptide H30，同上 |
| 1280–1919 | 640 | target H15，同上 |
| 1920–2559 | 640 | target H30，同上 |
| 2560–2561 | 2 | log1p 完整 peptide 长度、log1p 完整 target 长度 |
| 2562–2571 | 10 | 与 `mm_data.condition_encoding` 完全相同的10槽 observed 布尔位 |
| 2572–2699 | 128 | 同10槽的固定 signed feature hash |

H15 是 block15 输出、最终 encoder norm 之前；H30 是 block30 后且最终 norm 后的 last_hidden_state。`SequenceStore` 核验既有已资格化的具体版本语义和实际文件 SHA。本基线只反序列化 CPU 上的 `weights_only` 缓存张量，不加载 ESM/Qwen、不调用模型 forward、不启动 CUDA。

每个 8×640 张量转换到 float64 后对槽维取算术均值。原缓存使用 adaptive pooling，槽对应的残基区间可能大小不等且重叠，因此此处是已保存槽的等权均值，不能称完整原始残基的精确均值。不会恢复未保存残基或跨 chunk 注意力。

条件哈希复用 `mm_data.context_view` 的完整 allowlist 值：7个 chemical 槽、assay、measurement_context、target_context，并复用相同 observed mask。仅 observed 槽计算 `SHA256(slotname + NUL + canonicalJSON(value))`；前8字节的大端整数模128为桶，字节8偶数取+1、奇数取−1，在桶上累加。哈希可能碰撞；它没有标签、ID、DOI、额外 assay 之外的文本或可拟合词表。这里使用完整 allowlist 值的固定128维哈希，未使用模型的4096类别 ID 哈希16维替代。输入来源相同，表示压缩和函数族不同。

## TRAIN 拟合与预算

对每个场景仅用其 TRAIN 行计算逐维均值 μ 和总体 SD σ（ddof=0），尺度为 max(σ,1e−6)，所有 split 使用同一个 `z=(x−μ)/scale`。不标准化目标 y；y 为原 TRAIN pKD。TRAIN 统计保存有序 IDs、ID SHA、标签 SHA、sum/count。两个场景不得共享任何拟合统计或模型状态。

ridge 固定 α=1000，目标为 `Σ_i(y_i−b−z_i·w)² + 1000||w||²`，截距不惩罚。对 TRAIN 的 z、y 再数值中心化；N≤D 用 dual 系统 `(Zc Zcᵀ+αI)a=yc, w=Zcᵀa`，否则用 primal 系统 `(ZcᵀZc+αI)w=Zcᵀyc`，均调用 float64 `numpy.linalg.solve`，无显式矩阵求逆、无数据驱动 jitter/α 调整。截距为 `mean(y)−mean(z)·w`，2700个系数加1个截距为名义2701个系数，有效自由度不由此断言。另有5400个拟合预处理统计量，不把它们隐去。

kNN 固定 k=25，邻居数 min(25,N_train)，在 TRAIN 标准化空间用平方欧氏距离 d²，权重 `1/(d²+1e−6)`，加权平均原 TRAIN pKD。按锁定 TRAIN 行顺序稳定处理数值路径上的完全相等距离。NumPy 使用分块 BLAS 恒等式计算距离并将负舍入误差截为0；这与逐元素直接距离不承诺逐位一致。每次最多32条 query，不建立 N_eval×N_train×2700 的广播张量。保存2700N个 TRAIN 特征值和N个标签，容量与 ridge、joint 模型均不同。

ridge dual 的主要计算约 O(N²D+N³)，kNN 约 O(N_train·N_eval·D)。全精度特征、预处理、已保存近邻训练集和缓存文件哈希成本单独计入；没有“相同参数/通信/FLOPs预算”声明。固定数值方法各单fit，不把复制相同解三次称多随机种子实验证据。

validation 只记录指标，两个固定方法全部保留。标准化、ridge、kNN状态与 SHA 先写入 `selected_states_before_test.json`，之后才读取原锁定 test_gold。保存每条 validation/test 预测、gold、sample_id、scenario 和 numeric channel。数值评价调用冻结的 `mm_baselines.numeric_metrics`：micro MAE、RMSE、global Spearman、完整靶序列 macro MAE、全部样本覆盖；无效数值不丢行、不补默认值。Spearman 常量/方差为零时为 null。MAE/RMSE 单位 pKD，相关系数无量纲。

这些输出不构成 teacher-forcing C1、固定θ memory干预、conditional probe、pair测试或生物因果证据。pair gold 缺失时不构造替代gold。数值预测覆盖也不冒充语言 JSON 生成有效率。

## API、绑定与执行

```python
request = make_request(config_path, project_root, dataset_root, view_root, cache_root, scenario)
prepared = run(request)  # default execute=False: hashes/checks only
binding = prepared['training_run_binding']
# 父任务把 binding 放入 canonical task contract，并冻结真实资源/代码/资产。
result = run(request, execute=True, contract_path=contract_path,
             standard_root=canonical_standard_root,
             validator_sha256=actual_validator_sha256)
```

`dataset_root` 必须显式指定，直接包含两个 scenario 的原冻结8个文件；不从工作目录推断。`view_root` 是已准备好的独立 train/validation/test_inputs 视图根目录。请求绑定包含：config字节SHA；自身及 mm_data/mm_extract/mm_baselines 源码SHA；view manifest及6个视图SHA；两个场景8个原始源文件SHA；fullcache manifestSHA；每一个已登记 payload 的实际SHA闭包、字节数和总数。缓存包括未被当前场景引用的payload也全部核验。读取 payload 后再次核验；结束时重新计算完整闭包，一旦变化失败并保留partial artifacts。

`prepare` 只读源文件哈希与独立视图，不拟合、不生成特征、不反序列化模型、不解析原 samples 或 test_gold 标签；不创建结果目录。`execute` 使用 canonical `validate_reproduction.py` 的实际SHA，实时 `check_files=True`，要求 `valid_for_execution=true` 且无 skipped 文件，并要求合同 `training_run_binding` 完全相等。合同应把固定 config、prepared manifest、8个源文件和4个代码模块列入实际资产；cache manifest及闭包由子进程再次校验。输出目录 exclusive create，旧失败或完整运行不覆盖。

在新 scope 作为 cwd，用已装环境 `<PYTHON_ENV>/bin/python`：

```text
python -m src.mm_pooled_baselines prepare --config configs/pooled_baselines.json --project-root PROJECT --dataset-root DATASETS --view-root VIEWS --cache-root FULLCACHE --scenario SCENARIO --print-binding
python -m src.mm_pooled_baselines execute --config configs/pooled_baselines.json --project-root PROJECT --dataset-root DATASETS --view-root VIEWS --cache-root FULLCACHE --scenario SCENARIO --contract CONTRACT --standard-root STANDARD --validator-sha256 REAL_SHA
```

SCENARIO 依次为两个已登记场景。先由父任务生成精确 `training_run_binding` 和 canonical contract，再用现有标准 supervisor 执行第二个命令；不传 GPU。建议起始资源 CPU2线程、RAM2GiB、timeout900秒，真实当前资源、剩余deadline与最终timeout由父任务标准合同确定。NumPy BLAS线程由 supervisor 环境统一限制，记录现有 NumPy 版本；资源建议不是实测运行保证。结束结果可用冻结 `mm_archive.archive` 的 generic completion_spec 验收：断言 result.json `/status` 为 `complete`、`/formal_eligible` 为 true、两方法 test `n_total/n_finite` 等于原合同分母，并将 target_counts 指到 `/n_expected_test`；LM生成不存在，不施加生成门槛。

## 已完成验证与文献定位

指定 WSL 的9项 CPU tiny tests 全通过：2700布局、条件哈希泄漏排除、TRAIN标准化、primal/dual一致性、截距不受罚、kNN稳定顺序、无效数值拒绝、两个场景完整落盘、状态先于test读取、合同/请求/配置/视图/未引用缓存文件SHA拒绝。native环境6项NumPy测试通过，3项缓存tensor测试因该环境未安装Torch明确跳过；指定已有WSL环境无跳过。未安装任何新依赖、未执行真实A1fit/forward。

这一比较遵循既有[DataSAIL（2025）](https://doi.org/10.1038/s41467-025-58606-8)对相似性泄漏/对象分割的关注，保留两种冻结泛化场景。使用缓存特征建立单独数值参考，是本课题可检验的工程对照；不是将该论文的作者增益或具体线性方法迁移成已证结论。它与[Conditional Probing（2021）](https://aclanthology.org/2021.emnlp-main.122/)的“相对基线增量信息”问题有关，但此处直接预测 pKD，不是已完成条件探针。原文版本、物理页与SHA继续引用 `docs/literature_review.json`，不新增不受文献支持的机制优胜主张。
