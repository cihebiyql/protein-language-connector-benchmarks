# 绘图数据健康检查

模式：validate-only；此文件先于任何图形生成。
来源：PROJECT_ROOT/reports\benchmark_clarity_20261016\interim_20cases_20261016_2020\matched_summary.json
来源SHA256：`4ca600c4b0d8ebf90948d98c60c2bbd2af4c3423acaf34903cafd7022f19ab3d`

预期绘图网格30行；实际30行、23列；DataFrame内存47547字节。
核心完成20/24；有效difference 25/30，原始null 5/30；null保留用于coverage。
没有drop、去重、清洗、插补、outlier筛选或由图形需要改变评价标准。

## 列类型、缺失与唯一值

| column | health dtype | original JSON/Python types | null | unique |
|---|---|---|---|---|
| scenario | object | {"str": 30} | 0 | 2 |
| contrast | object | {"str": 30} | 0 | 5 |
| seed | object | {"int": 30} | 0 | 3 |
| paired_difference | object | {"float": 25, "NoneType": 5} | 5 | 26 |
| paired_difference_exact_json | object | {"str": 30} | 0 | 26 |
| paired_difference_binary64_hex | object | {"str": 25, "NoneType": 5} | 5 | 26 |
| source_k | object | {"int": 30} | 0 | 2 |
| source_coverage | object | {"str": 30} | 0 | 2 |
| source_mean | object | {"float": 15, "NoneType": 15} | 15 | 6 |
| source_mean_exact_json | object | {"str": 30} | 0 | 6 |
| source_sample_SD | object | {"float": 15, "NoneType": 15} | 15 | 6 |
| source_sample_SD_exact_json | object | {"str": 30} | 0 | 6 |
| mean_displayed | object | {"bool": 30} | 0 | 2 |
| formula | object | {"str": 30} | 0 | 5 |
| mae_E_R | object | {"float": 25, "NoneType": 5} | 5 | 6 |
| mae_E_N | object | {"float": 25, "NoneType": 5} | 5 | 6 |
| mae_M_R | object | {"float": 25, "NoneType": 5} | 5 | 6 |
| mae_M_N | object | {"float": 25, "NoneType": 5} | 5 | 6 |
| source_summary_path | object | {"str": 30} | 0 | 1 |
| source_summary_sha256 | object | {"str": 30} | 0 | 1 |
| source_evaluation_version | object | {"str": 30} | 0 | 1 |
| source_protocol_sha256 | object | {"str": 30} | 0 | 1 |
| source_json_pointer | object | {"str": 30} | 0 | 30 |

## 唯一键与重复

唯一键(scenario,contrast,seed)：30；重复行：0。
原始case/contrast重复若存在即阻止绘图，不自动去重。

## 数值描述与coverage

| scenario | contrast | k/3 | finite min | finite max | source mean | source sample_SD |
|---|---|---|---|---|---|---|
| new_peptide_known_target | depth_real | 3/3 | -0.34832101101351526 | 0.16698070616470775 | -0.10910166811246808 | 0.25962112249080277 |
| new_peptide_known_target | depth_neutral | 3/3 | -0.5243968786414849 | -0.17733529539241344 | -0.3683215386917624 | 0.1761448665767846 |
| new_peptide_known_target | content_early | 3/3 | -0.6549044409908352 | -0.16833744000760426 | -0.4116477329550715 | 0.24328350491753636 |
| new_peptide_known_target | content_mid | 3/3 | -0.35678988683890145 | 0.21009473914528387 | -0.1524278623757772 | 0.3148025109426259 |
| new_peptide_known_target | DiD | 3/3 | 0.054911431027873533 | 0.3784321791528881 | 0.2592198705792943 | 0.17775666583109867 |
| new_peptide_new_target | depth_real | 2/3 | -0.06936645930336155 | 0.3457093283445276 | null | null |
| new_peptide_new_target | depth_neutral | 2/3 | -0.4066141553343558 | 0.09395404881922258 | null | null |
| new_peptide_new_target | content_early | 2/3 | -0.14071975260148437 | -0.06836193638990609 | null | null |
| new_peptide_new_target | content_mid | 2/3 | 0.11103552692382067 | 0.2688857596410882 | null | null |
| new_peptide_new_target | DiD | 2/3 | 0.251755279525305 | 0.33724769603099425 | null | null |

仅核对source已有三seed均值与sample_SD的算术；少于3时不计算均值、SD或补值。没有合并不同contrast计算新的总体效应/方差。

## 分类覆盖

| column | counts |
|---|---|
| scenario | {"new_peptide_known_target": 15, "new_peptide_new_target": 15} |
| contrast | {"depth_real": 6, "depth_neutral": 6, "content_early": 6, "content_mid": 6, "DiD": 6} |
| seed | {"17": 10, "42": 10, "2026": 10} |

## 统计与解释边界

N=3配对训练seed；DiD与其余contrast存在精确代数依赖。没有正态、VIF、相关推断、p值、优胜阈值或新CI。黑菱形误差条只可为原sample_SD，不是CI。
content是分别训练的θ_R/θ_N系统总效应，不是固定θ的推理消息利用；正DiD不保证mid绝对更好。
Fixed core figure; gate and pooling are prospectively registered extension stages, not score-filtered.

## 校验结果

无schema/算术错误。

核心未完成24项：当前仅validation-only，禁止当作完整core24图形。
