# 最终工程决定：恢复原V1逐行推理

**V2与V3均未达到既定TRAIN批量一致性门槛；不再提出batch、backend或数值策略候选。恢复exact原V1 per-row生成，按原冻结36项清单继续剩余34项。** 原两个complete模型和原输出保持同一V1通道，不重训、不重复计数、不用其他版本替换答案。

本次只读取V3资格的accepted／不一致字段、执行收据与suite状态，以及原V1运行状态和科学冻结SHA；未读取新增test预测或分数、未调用GPU、未训练、未修改源码或任何历史审查／失败记录。

## 终止批量候选的依据

V3 `suite_batched_v3_20261016_1530` 共3项TRAIN canary：known early-zero和mid-zero通过；第三项known mid-near_unit失败。失败报告明确reader state未变、test_rows_used=0，`B2_all_four` 的首EOS前token和raw_text与B1不一致。第三项exit1，所有3项进程组均不存在；controller标记failed_or_partial并停止后续队列。前两项通过不能覆盖第三项失败，不能把V3报告为整体合格或进入V3 test。V2和V3原源码、合同、输出、不一致、失败及队列状态全部保留。

## 恢复V1的边界

- 科学冻结SHA仍为 `9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`。不改样本、test、gold、原prompt、标签精度、greedy24、parser、失败分母、dtype／严格数值策略、参数／消息预算、3轮训练、验证选点或原36项顺序。
- 原V1两个early模型已complete、exit0且进程组不存在，保留其完整训练／评价身份；第三项原V1在spawn前停止，不算第三个训练。剩余34项必须使用exact原V1 worker及per-row解码，不能混入V2／V3生成或进行逐case版本选择。
- 原V1六项真实TRAIN资格可以在worker源码、数值flags、底座、cache、环境等绑定逐项SHA一致的条件下复用。复用标为历史资格凭证，不虚称进行了新canary。任何不一致须停止准入核验，不能静默继承旧资格；本文件本身不是这些最终运行pins已重新核验的证明。
- 每个新job仍需fresh资源probe、真实标准合同、独立输出目录、进程组／内存／墙钟监督和结果／退出验收；继续校验旧资产不变。dispatcher可以新增续接账本，但不得改旧worker或旧归档。
- 总窗口保持13:20—23:20，V2／V3资格耗时计入同一窗口，不恢复或延长旧foundation的07:18—17:18合同。到23:20终止自有任务并保留全部partial和队列未完成项。

## 结果报告规则

保留每个seed、每个case的完成／失败／未开始状态及原始输出；不能删除未完成seed、改为“最好seed”，或只平均成功seed。全样本生成失败继续导致full MAE=null；三seed汇总仅在3/3完整有效时成立，否则给k/3和null。若时间不足完成剩余34项，就按原规则归档partial；不得重置时钟、删seed、调整测试集或评价标准来制造“完成”。V2／V3都是工程资格失败，不是新的科学模型重复或可择优的test通道。

此决定依据TRAIN一致性资格失败恢复既有V1执行基准；没有依赖新test分数，也不宣称新训练或剩余34项已完成。
