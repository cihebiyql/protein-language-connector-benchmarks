# V3最后批量候选实现独立审查

实现门禁通过，无未关闭P1/P2。仅审V2→V3窄diff并独立执行32项V3 CPU测试，全部通过（3.487秒、exit0）；未重复旧94项，未执行GPU、下载模型forward或读取新test分数。父任务正在保存最终126项完整测试收据。

主代码SHA：runner98e61cce…、batchgenc98cc308…、controller ec4f4421…、summary a6a6335a…。训练函数按原UTF8源码span验证与V1逐字相同；科学协议9940d77d…保持不变。

发现并关闭一项P2：候选controller复制时误标a1_locked_v3，最终已恢复实际冻结数据名／版本Immutable A1 v2、a1_locked_v2。仅纠正元数据，未改数据、分组、指标或科学协议。最终SHA与文字已独立核验。

V3固定所有36项原序B2与尾批1–2；资格仅比较完整四行TRAIN的B1和B2_all_four（两个batch），逐行首EOS token、raw decode、strict parse精确相同，reader state不变。监督器明确拒绝先前firsttwo仅两行资格，六案例全部在任何fulltrial之前。

原2个完整V1训练状态经checkpoint/history/标准archive核验后仅重新V3推理，本次零optimizer；其余34项新训练。所有V3评价统一版本，旧V1/V2预测独立保留、不补答案、不择优；summary不把复用计为新replicate。原greedy24、parser、完整失败分母、k/3规则和23:20截止不变。

本报告仅关闭实现门禁。父任务仍须最终126物理CPU收据、新source/cache manifest、旧资产保护、fresh resource probe与实际合同，以及全部六个真实TRAIN全四行B1/B2 GPU资格。V3为最后工程候选，任一资格失败即保存并停止，不再搜索batch/backend/dtype或放宽一致性。

TRAIN通过只覆盖实测输入／状态，不能保证所有test输入batch-shape等价。actual EOS读取／padding／FLOPs仍可能不同，必须按账本报告。无显式type/slot embedding及已观察test等既有限制继续保留。
