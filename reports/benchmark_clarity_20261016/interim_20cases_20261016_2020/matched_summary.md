# 同参数与通信预算的固定36项对照

全部种子、缺失和生成失败均保留。仅3/3个完整有效种子才报告三种子均值和样本SD；固定模型的来源／家族组bootstrap另列。

完整终态训练：20/36。canary不计入训练结果。

| 场景 | seed | 位置 | 消息 | 池化 | 门控 | 状态 | 有效/总数 | full MAE | valid-only MAE |
|---|---:|---|---|---|---|---|---:|---:|---:|
| new_peptide_known_target | 17 | early | real | balanced | zero | complete | 152/152 | 1.71721 | 1.71721 |
| new_peptide_known_target | 17 | early | neutral | balanced | zero | complete | 152/152 | 1.54888 | 1.54888 |
| new_peptide_known_target | 17 | mid | real | balanced | zero | complete | 152/152 | 1.86318 | 1.86318 |
| new_peptide_known_target | 17 | mid | neutral | balanced | zero | complete | 152/152 | 2.07327 | 2.07327 |
| new_peptide_new_target | 17 | early | real | balanced | zero | complete | 442/442 | 1.49440 | 1.49440 |
| new_peptide_new_target | 17 | early | neutral | balanced | zero | complete | 442/442 | 1.42604 | 1.42604 |
| new_peptide_new_target | 17 | mid | real | balanced | zero | complete | 442/442 | 1.56376 | 1.56376 |
| new_peptide_new_target | 17 | mid | neutral | balanced | zero | complete | 442/442 | 1.83265 | 1.83265 |
| new_peptide_known_target | 42 | early | real | balanced | zero | complete | 152/152 | 1.93416 | 1.93416 |
| new_peptide_known_target | 42 | early | neutral | balanced | zero | complete | 152/152 | 1.52246 | 1.52246 |
| new_peptide_known_target | 42 | mid | real | balanced | zero | complete | 152/152 | 2.28248 | 2.28248 |
| new_peptide_known_target | 42 | mid | neutral | balanced | zero | complete | 152/152 | 1.92569 | 1.92569 |
| new_peptide_new_target | 42 | early | real | balanced | zero | complete | 442/442 | 1.73214 | 1.73214 |
| new_peptide_new_target | 42 | early | neutral | balanced | zero | complete | 442/442 | 1.59142 | 1.59142 |
| new_peptide_new_target | 42 | mid | real | balanced | zero | complete | 442/442 | 1.38643 | 1.38643 |
| new_peptide_new_target | 42 | mid | neutral | balanced | zero | complete | 442/442 | 1.49747 | 1.49747 |
| new_peptide_known_target | 2026 | early | real | balanced | zero | complete | 152/152 | 2.23333 | 2.23333 |
| new_peptide_known_target | 2026 | early | neutral | balanced | zero | complete | 152/152 | 1.57842 | 1.57842 |
| new_peptide_known_target | 2026 | mid | real | balanced | zero | complete | 152/152 | 2.06635 | 2.06635 |
| new_peptide_known_target | 2026 | mid | neutral | balanced | zero | complete | 152/152 | 1.75576 | 1.75576 |
| new_peptide_new_target | 2026 | early | real | balanced | zero | missing | —/— | null | null |
| new_peptide_new_target | 2026 | early | neutral | balanced | zero | missing | —/— | null | null |
| new_peptide_new_target | 2026 | mid | real | balanced | zero | missing | —/— | null | null |
| new_peptide_new_target | 2026 | mid | neutral | balanced | zero | missing | —/— | null | null |
| new_peptide_known_target | 17 | mid | real | balanced | near_unit | missing | —/— | null | null |
| new_peptide_new_target | 17 | mid | real | balanced | near_unit | missing | —/— | null | null |
| new_peptide_known_target | 42 | mid | real | balanced | near_unit | missing | —/— | null | null |
| new_peptide_new_target | 42 | mid | real | balanced | near_unit | missing | —/— | null | null |
| new_peptide_known_target | 2026 | mid | real | balanced | near_unit | missing | —/— | null | null |
| new_peptide_new_target | 2026 | mid | real | balanced | near_unit | missing | —/— | null | null |
| new_peptide_known_target | 17 | mid | real | global16 | zero | missing | —/— | null | null |
| new_peptide_new_target | 17 | mid | real | global16 | zero | missing | —/— | null | null |
| new_peptide_known_target | 42 | mid | real | global16 | zero | missing | —/— | null | null |
| new_peptide_new_target | 42 | mid | real | global16 | zero | missing | —/— | null | null |
| new_peptide_known_target | 2026 | mid | real | global16 | zero | missing | —/— | null | null |
| new_peptide_new_target | 2026 | mid | real | global16 | zero | missing | —/— | null | null |

| 场景 | 固定配对差值（正值有利） | 完整种子 | 三种子均值 | 种子样本SD |
|---|---|---:|---:|---:|
| new_peptide_known_target | depth_real | 3/3 | -0.10910 | 0.25962 |
| new_peptide_known_target | depth_neutral | 3/3 | -0.36832 | 0.17614 |
| new_peptide_known_target | content_early | 3/3 | -0.41165 | 0.24328 |
| new_peptide_known_target | content_mid | 3/3 | -0.15243 | 0.31480 |
| new_peptide_known_target | depth_content_difference_in_differences | 3/3 | 0.25922 | 0.17776 |
| new_peptide_known_target | gate_zero_initialization | 0/3 | null | null |
| new_peptide_known_target | balanced_pool_allocation | 0/3 | null | null |
| new_peptide_new_target | depth_real | 2/3 | null | null |
| new_peptide_new_target | depth_neutral | 2/3 | null | null |
| new_peptide_new_target | content_early | 2/3 | null | null |
| new_peptide_new_target | content_mid | 2/3 | null | null |
| new_peptide_new_target | depth_content_difference_in_differences | 2/3 | null | null |
| new_peptide_new_target | gate_zero_initialization | 0/3 | null | null |
| new_peptide_new_target | balanced_pool_allocation | 0/3 | null | null |

C1、token分解和门控诊断独立报告；即使full MAE未定义也保留，不能替代主评价。G是逐样本完整答案off−correct NLL的均值；类别NLL使用各类别token分母，贡献项使用完整答案token分母。P/H独立donor仍为0。

| 场景 | seed | 位置/内容/池化/门控 | 原验证最佳NLL | C1 G | 末轮验证tanh门控 | 末轮验证残差/hidden |
|---|---:|---|---:|---:|---:|---:|
| new_peptide_known_target | 17 | early/real/balanced/zero | 0.67215 | 0.52217 | 0.00876 | 0.25589 |
| new_peptide_known_target | 17 | early/neutral/balanced/zero | 0.66522 | 0.53231 | 0.00556 | 0.25275 |
| new_peptide_known_target | 17 | mid/real/balanced/zero | 0.65717 | 0.54387 | -0.02989 | 0.01401 |
| new_peptide_known_target | 17 | mid/neutral/balanced/zero | 0.65565 | 0.54248 | -0.01948 | 0.01367 |
| new_peptide_new_target | 17 | early/real/balanced/zero | 0.67370 | 0.52970 | 0.00546 | 0.29358 |
| new_peptide_new_target | 17 | early/neutral/balanced/zero | 0.67620 | 0.52736 | 0.00522 | 0.21570 |
| new_peptide_new_target | 17 | mid/real/balanced/zero | 0.67152 | 0.53586 | 0.02992 | 0.01425 |
| new_peptide_new_target | 17 | mid/neutral/balanced/zero | 0.66671 | 0.54607 | 0.01911 | 0.01190 |
| new_peptide_known_target | 42 | early/real/balanced/zero | 0.66849 | 0.51834 | 0.00750 | 0.34934 |
| new_peptide_known_target | 42 | early/neutral/balanced/zero | 0.66102 | 0.53688 | -0.00768 | 0.32199 |
| new_peptide_known_target | 42 | mid/real/balanced/zero | 0.66065 | 0.53282 | -0.03258 | 0.01367 |
| new_peptide_known_target | 42 | mid/neutral/balanced/zero | 0.65385 | 0.54386 | -0.01970 | 0.01288 |
| new_peptide_new_target | 42 | early/real/balanced/zero | 0.68352 | 0.52041 | 0.00587 | 0.30566 |
| new_peptide_new_target | 42 | early/neutral/balanced/zero | 0.67312 | 0.53391 | 0.00662 | 0.26153 |
| new_peptide_new_target | 42 | mid/real/balanced/zero | 0.67060 | 0.54216 | -0.02619 | 0.01400 |
| new_peptide_new_target | 42 | mid/neutral/balanced/zero | 0.66698 | 0.54159 | -0.01783 | 0.01017 |
| new_peptide_known_target | 2026 | early/real/balanced/zero | 0.66685 | 0.52784 | 0.00828 | 0.31843 |
| new_peptide_known_target | 2026 | early/neutral/balanced/zero | 0.66463 | 0.53683 | 0.00682 | 0.31066 |
| new_peptide_known_target | 2026 | mid/real/balanced/zero | 0.65819 | 0.54055 | -0.02801 | 0.01392 |
| new_peptide_known_target | 2026 | mid/neutral/balanced/zero | 0.65128 | 0.55097 | -0.01892 | 0.01204 |
| new_peptide_new_target | 2026 | early/real/balanced/zero | null | null | null | null |
| new_peptide_new_target | 2026 | early/neutral/balanced/zero | null | null | null | null |
| new_peptide_new_target | 2026 | mid/real/balanced/zero | null | null | null | null |
| new_peptide_new_target | 2026 | mid/neutral/balanced/zero | null | null | null | null |
| new_peptide_known_target | 17 | mid/real/balanced/near_unit | null | null | null | null |
| new_peptide_new_target | 17 | mid/real/balanced/near_unit | null | null | null | null |
| new_peptide_known_target | 42 | mid/real/balanced/near_unit | null | null | null | null |
| new_peptide_new_target | 42 | mid/real/balanced/near_unit | null | null | null | null |
| new_peptide_known_target | 2026 | mid/real/balanced/near_unit | null | null | null | null |
| new_peptide_new_target | 2026 | mid/real/balanced/near_unit | null | null | null | null |
| new_peptide_known_target | 17 | mid/real/global16/zero | null | null | null | null |
| new_peptide_new_target | 17 | mid/real/global16/zero | null | null | null | null |
| new_peptide_known_target | 42 | mid/real/global16/zero | null | null | null | null |
| new_peptide_new_target | 42 | mid/real/global16/zero | null | null | null | null |
| new_peptide_known_target | 2026 | mid/real/global16/zero | null | null | null | null |
| new_peptide_new_target | 2026 | mid/real/global16/zero | null | null | null | null |

逐epoch原验证NLL、完整format/numeric/mixed分解、同配置逐seed均值和SD及逐步gate轨迹来源，见matched_summary.json。任何机制均值同样要求3/3；缺失或无该类别token时为null。

组bootstrap保持旧group_id定义，2000次、seed2026；不能视为新的独立实验。通信预算是16×128 float32的逻辑载荷，每次8192字节；实际EOS调用数和padding开销从原始诊断读取，不冒称总FLOPs或物理传输相等。
旧数值预测器只作参考。中性消息仍保留原允许提示上下文。旧test已被观察，本轮属于同冻结基准上的预先规定后续对照。

全部36项采用原V1逐行batch1解码。固定前2项读取原始完整训练和原始评估；其余34项来自本次原V1续跑，绝不读取V2/V3答案。
原始已完成来源：2/2；新完成训练：18/34。原PID仅属于原来源，不伪造本次执行。
