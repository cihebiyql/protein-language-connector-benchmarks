# 三任务初始运行快照

正式计划45组、共同3个完整epoch、三个随机种子。14:29:52UTC快照2accepted／4running／39pending；本机独立复算取回的ELM ESM与M0两个seed17正式模型，保存指标全部一致，均3epochs／285updates。三种子未完整，不能判断稳定提升。

稍后只读队列观察7accepted／4running／34pending，不扩大本次两组取回复算范围。当前状态见COMPLETION_STATUS.json；此页保留初始快照，收尾助手更新JSON和独立结果命名空间。ELM逐查询输入、候选文字／向量、逐查询输出和权重保持私有；所有方法种子的聚合指标、历史、状态按同一规则公开。

原pKD与历史结果不变；各任务独立固定prompt训练，不据此声称丰富问题路由或跨任务迁移。初版causal guard因T与T+2的BF16计算形状产生误报；旧失败和旧source保留。正式同形状未来值检查与原1e-5阈值来源见results/anchor_diagnosis_v1。
