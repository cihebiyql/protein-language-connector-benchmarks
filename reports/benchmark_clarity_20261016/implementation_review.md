# 同参数与通信预算实现独立审查

最终实现审查通过，未发现未关闭的P1/P2。独立WSL CPU实测64项全部通过，无skip；另重跑24项supervision为同一集合子集，不重复计数。本审查没有执行GPU或下载模型forward。

P2已关闭：summary逐条核完整冻结test顺序、ID、split、gold与raw_text，重新调用旧strict parser并重算旧metric全部字段；parse失败也必须保留完整原始记录。删失败行、造假parse/分母/conditional或target-macro MAE、重复或乱序均被负向fixture拒绝。

最终runner的原prompt、answer IDs/EOS、loss与validation选择未改。数值／格式／mixed分解复用原forward并独立计数；C1复用旧逐样本答案NLL定义，完整correct/off评价与原G、贡献分解、fp32 reduction余差分开保存。P/H零donor不改变。

已核最终runner dca9c255…、prepare63e9094a…、summary695b6f87…、supervisor6044a321…。执行规范是新增.frozen.json（9940d77d…）；原.protocol.json（54445ef9…）按字节保留。逐文件完整SHA见JSON。

参数与通信含义：相同464001 nominal参数，sender零训练参数，每次forward每例16×128 float32、8192字节，无新增LM位置；valid、physical padding、EOS调用与attention edges分开记。不同深度的梯度路径/FLOPs并未匹配。

## 需要明确报告的科学边界

新reader没有slot/type embedding，拒绝memory_type_ids；MHA对同时置换KV槽不变。独立CPU probe置换16槽后最大差5.96e-8（浮点约简舍入），因此8+8顺序本身不能让reader识别肽／靶。balanced池化提供固定分配，但pool对照只能解释两种池化系统差异，不能证明显式类型对齐或精确肽关注。

冻结旧prompt仍写“Memory type 0/1”，新接口没有原type IDs或原完整带类型残基memory。保留prompt字节有利于固定比较，但不能声称完整保留旧类型语义。该限制适用于所有新同预算条件，不改变冻结36项、参数或评价。

## 运行门禁

实现门禁已通过，CPU结果不构成GPU放行。父任务仍应核真实cache与source/init pins、旧资产未变、fresh resource probe、实际标准合同及TRAIN-only严格确定性canary；不支持的数值kernel应保存失败并停止排队。23:20总截止、失败归档、PGID终态清理及不盲目重跑保持不变。
