# 原V1逐条推理续跑独立审查

实现门禁通过，无未关闭P1/P2。最终controller e1a886a2…、summary96dddafd…、tests5d31599c…与owner冻结确认一致。原worker run_trial.py仍为dca9c255…；worker、numerical policy、监督清理、fresh probe及原36矩阵函数与原supervisor逐字一致。科学协议9940d77d…、13:20首时钟及23:20截止保持不变。

独立11项CPU最终全部通过（0.609秒）。第一次11项中一项标准合同fixture在owner并行改源时因SHA漂移拒绝，通过冻结后重跑关闭；该失败已如实保留，未启动GPU。

补强并关闭资格复用门禁P2：当前pip freeze SHA和环境路径须与原六合同完全一致，原numerical_policy_actual flags及pre-model环境须匹配冻结协议；不一致即保存准入失败并停止，不默认继承历史资格。相关环境／数值漂移负向测试通过。

固定原前2个完整V1来源已独立进行真实CPU raw审计，各完整152行：ID顺序、split、gold、旧strict parser及全部旧metric字段重算一致，含失败分母。result、predictions、samples和parser的SHA绑定JSON。机械审计不输出分数，不按分数选择来源。

新控制器仅运行其余34个固定case，全部使用原V1 batch1逐条解码、原训练和标准。原2个实际完成来源保持原PID／receipt，不伪称本次新训练。原6个TRAIN资格仅在完全相同pins/环境/数值设置下复用，不计新canary。每新job仍独立fresh probe、实际合同、PGID监督和终态档案。

summary只将原2与新34建立临时只读链接供旧raw/parser/metric审计；临时路径重写为真实物理job。V2/V3结果不回填、不按版本择优；原失败档案保留。缺失seed仍k/3及null，未改test／group／metric。

本审查不替代运行准入。父任务仍需最终137项完整CPU收据、新manifest与旧资产核验、历史资格环境／数值实际绑定和各job资源／合同门禁。此次只恢复精确原V1，不再任何batch/backend候选；到原23:20截止保存所有未完成项。
