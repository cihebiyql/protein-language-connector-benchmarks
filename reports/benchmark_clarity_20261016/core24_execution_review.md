# 核心24项执行与三随机种子算术独立复核

结论：通过，无未关闭问题。证据固定于21:31快照matched_summary.json（SHA d22da6e9…）及snapshot_manifest；仅24个ORIGINAL V1 batch1完整终态：原来源2项＋新训练22项。gate25及后续活跃目录未读取或操作。

本次独立重算全部7128条严格预测及原MAE、并列平均秩全局Spearman；7128／7128有效、0失败。前16项C1通过既有独立审查与相同result／raw预测／raw C1指纹继承；新增8项另重算2376行C1，连同前证合计覆盖7128行，未声称重新forward或再次全量重算旧C1。

## 新增8项真实执行证据

- 每项三epoch、3×516＝1548条连续optimizer日志；完整validation原full-answer NLL最小值及最早并列checkpoint选择一致，没有test选点或提前停训。
- 八个自产selected reader仅在已有WSL环境CPU加载（CUDA_VISIBLE_DEVICES为空、torch2线程、weights_only=True/map_location=cpu）。每个11个tensor实际共464001个元素，全部float32、CPU、有限；逐名称参数数量、所选epoch／criterion及完整registration均匹配。未加载底座或旧模型、未做forward。
- same16×128 float32接口、8192逻辑字节／例／forward；实际train／validation调用数、payload、attention edges与residual bytes一致。B1自由生成的calls／queries／bytes又由所有raw token长度和prompt长度独立重算。
- strict numerical实际首末flags、BF16底座／FP32 reader／无autocast、冻结梯度声明、2线程、fresh probe、环境freeze、原V1源及固定cache／projection／未训练初态／数据lock指纹均一致。底座15文件实际核验receipt与原V1相同，本次不重复rehash或加载整底座。
- 实际标准合同执行准入／result接受、exit0、自有PGID已退出归档与stdout／stderr／artifact hash均通过；只核历史终态收据，不干预当前其他进程。

| seed2026场景／条件 | 选epoch | monotonic秒 | UTC跨度秒 | 差值秒 |
|---|---:|---:|---:|---:|
| known/early_real | 1 | 788.446 | 875.207 | 86.761 |
| known/early_neutral | 3 | 710.215 | 790.311 | 80.096 |
| known/mid_real | 1 | 556.429 | 616.520 | 60.091 |
| known/mid_neutral | 3 | 559.988 | 623.396 | 63.408 |
| double-new/early_real | 1 | 1002.288 | 1112.404 | 110.116 |
| double-new/early_neutral | 3 | 997.055 | 1107.248 | 110.193 |
| double-new/mid_real | 1 | 887.538 | 987.641 | 100.103 |
| double-new/mid_neutral | 2 | 867.887 | 964.685 | 96.798 |

新增8项的两种实测时长都≤1800秒，结束UTC均早于原23:20绝对截止，未重置时窗。UTC比monotonic长60.091–110.193秒，分别保留，不据较短monotonic宣称wall-time加速。最大采样whole-PGID RSS5.979GiB（限12GiB），Torch allocated6.952GiB（限11.5GiB）；reserved11.207GiB单列。采样RSS不是瞬时峰值证明。

## 三seed与原10个核心contrast

逐seed原完整总体MAE重算，三seed mean及样本SD（ddof1）一致；原10个核心contrast按冻结方向独立重算如下（正值有利于该contrast所命名处理）。另外4个gate／pool contrast在此快照仍0/3，mean／SD保持null，没有读活跃项补值。

| 场景 | 原contrast | 三seed mean | 样本SD |
|---|---|---:|---:|
| new_peptide_known_target | depth_real | -0.109102 | 0.259621 |
| new_peptide_known_target | depth_neutral | -0.368322 | 0.176145 |
| new_peptide_known_target | content_early | -0.411648 | 0.243284 |
| new_peptide_known_target | content_mid | -0.152428 | 0.314803 |
| new_peptide_known_target | depth_content_difference_in_differences | 0.259220 | 0.177757 |
| new_peptide_new_target | depth_real | -0.037186 | 0.367862 |
| new_peptide_new_target | depth_neutral | -0.121169 | 0.257587 |
| new_peptide_new_target | content_early | -0.073888 | 0.064247 |
| new_peptide_new_target | content_mid | 0.010095 | 0.321378 |
| new_peptide_new_target | depth_content_difference_in_differences | 0.083983 | 0.367125 |

另核640个机制scalar汇总的逐seed源值、覆盖、mean／sampleSD或null规则。既有group-bootstrap区间未重新抽样，不将固定test组不确定性与seed SD混为一体。

real深度差三seed均值在已见和双新都为负，且随seed变号；完成的核心24项仍未确立稳定mid优势。content与DiD是分别训练／验证选点系统之间的差，不能识别缺失的同θ_R mean替换；正C1也只是整个分支的teacher-forced作用，不自动证明分子特异或生物因果。

“同预算”是同模块nominal参数及逻辑消息，不等于相同梯度路径、hidden尺度、实际EOS读取或总FLOPs；early是同一种动态reader的早接入，不是旧MLP／经典prefix的替代命名。前2个来源没有重新执行或多计为replicate，24＝原2＋新22。未做新fit、旧CPU/GPU测试、新bootstrap、依赖改动或源码／结果／进程修改。
