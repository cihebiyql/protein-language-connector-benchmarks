# 有证据改进方案的独立审查

**结论：作为未来研究方案通过；未发现当前36项被修改、数据／端点漂移、保分或首创承诺。P0—P3发现均为0。** 本结论不是未来S1／S2／W实现或执行合同的验收。T、λ和具体交错表尚未选择是方案明示的未来工作，不构成当前执行错误。

只读新方案、已存原文／源码证据和19:29冻结16项快照，未读取seed2026活跃案例，未下载、执行第三方代码、加载模型、forward、fit、校准、使用GPU或操作进程。只新增本文件和同名JSON。科学冻结SHA仍为 `9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`。

## 已核事实与范围

- 16项快照只有seed17／42。双新depth-real从−0.069366变为＋0.345709；已见content-mid从＋0.210095变为−0.356790。方案保留这两条反向事实，三seedmean／SD保持null，没有以两seed汇总替代原统计。
- G／P仍是原矩阵内门控初值与池化配额扩展；S1／S2和W均为当前36之外、未执行的独立未来方案。alpha3没有冒称无门控，8＋8没有冒称类型对齐，已训练系统content总效应没有冒充同θ推理消融。
- S1／S2使用未训练Reader、固定TRAIN ID、branch-off frozen-base的**prompt-only**隐藏状态，不拼gold、不用val／test或pKD计算尺度。当前含teacher-forced答案的validation范数不符合校准输入定义，不能挪用。内容条件分别校准被明示为系统处理的一部分。
- S2的无affine LN在1536维output projection后、gate前，alpha0和有限系数使新增分支为0；没有改为`LN(h+δ)`。无affine LN增加0可训练参数，但校准prepass、缓存、缩放、LN及反向开销单列，没有把参数／消息相同说成同实测计算。
- W不声称固定sender会被损失训练。head读取Reader的未门控u，使aux梯度可达非gate Reader；标签只用于TRAIN监督，不进入prompt/query。aux不依赖alpha，不能称为预热gate；辅助数值预测不能替代原语言生成评分。
- Linear1536→1含1,537参数，联合训练名义元素数为465,538；两臂均有实际接受aux梯度的head。推理丢弃head不抹去训练成本，不能与原464,001参数、无aux曝光的CE-only说成同预算。

## W的更新定义是否足够

方案已经给出A先T_aux再T_CE、B以固定次序交错相同数量的两类更新，并匹配逐loss样本顺序／曝光、梯度累计、λ、优化器组、调度及验证预算。**作为方案定义足够**；其含义应在未来可执行合同落实为下表，不能把B悄悄替换成每一步联合两个loss：

| step类型 | 该次反向目标 | 有任务梯度的路径 | 需计账的曝光 |
|---|---|---|---|
| aux step | 预先固定λ的`λ L_aux` | Reader非gate路径＋真实head；alpha不在aux计算图中 | 对应TRAIN ID、标签／prompt、最后prompt query和消息读取 |
| CE step | 原完整答案`L_CE` | 原门控Reader路径；head不参与CE forward | 原TRAIN ID、答案token、完整文本query及消息读取 |

两臂都应是相同T_aux＋T_CE次optimizer更新，只改变类型次序；累计microbatch／样本及每类目标曝光也各自相同。如果未来改成每步`L_CE+λL_aux`，梯度组合、step数及暴露关系就成为新处理，不能沿用“仅顺序不同”的匹配主张。这里没有选择任何T／λ／交错比率。

未来还须冻结`zero_grad`、未参与该步参数的grad状态与weight-decay／optimizer-step策略，使两臂规则一致；无任务梯度不必然等于无参数更新。head在CE step无任务梯度不构成挂名凑参数，因为两臂都有规定aux step实际使用它。optimizer状态随次序演化则是所研究处理的一部分。

## placement与辅助query的必要解释

未来阶段顺序A／B应首先在**相同placement内**比较。当前early位于第一个decoder block前，其最后prompt token query尚未经过decoder跨token上下文化；mid位于第14块后，query已有上下文。这是现有接入深度的组成部分。方案的“各按原位置提取”应落实为对应hook位置，不能对early也偷用最终decoder隐藏状态。

因此early辅助路径不应被描述成已通过query读取了完整assay／chemistry上下文。若另比较early与mid的预热结果，差异同时包含query语义和传播路径，不能单独归因为预热顺序。此项是未来合同与解释范围，不要求修改当前模型或增加实验。

## 原文和迁移边界

方案明确ProtT3采用不同Galactica底座、原表示目标和LoRA，当前固定ESM／Qwen及新pKD MSE没有等效复现。NormFormer／ReZero的较晚官方commit没有冒充论文v2原实验commit；分支LN与skip缩放、ReZero直接零标量与当前tanh、整网理论与冻结单分支均分开。九篇原文只支持机制动机，不被用作本课题已有效证据或全球首创证明。

具体实现、T／λ、校准ID、选择候选与验证时间仍须在未来正式协议冻结；它们现在未定不等于偷偷已选。现有观察来自已见测试基准，方案也没有包装为未见确认试验。附JSON保留已核输入／9份PDF／13份静态源码指纹、两条反向差值及未来合同落实条件。本审查未复读全部PDF或认证尚不存在的W／S实现。
