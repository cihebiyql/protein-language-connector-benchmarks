# 基准解释独立审查

**结论：通过；未发现新增端点、阈值、失败分母或选择规则漂移，也未发现将现有机制观察升级为无依据因果结论。P0／P1／P2／P3发现均为0。** 本结论针对所提交文档和17:26冻结的首八格快照，不替代后续运行验收或完整三种子结论。

本次只读README、三组报告的Markdown／JSON及科学冻结协议，另外重读其引用的保存预测、snapshot、history和产物指纹。没有模型forward、fit、训练、GPU调用、进程操作、下载或旧文件修改。仅新增本文件与同名JSON。

## 独立核对结果

- 科学冻结SHA仍为 `9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`。2,376条已保存生成重新按原严格JSON规则解析，核对原ID顺序、原精度gold和完整分母；八格均完整有效，重算MAE与报告最大差为1.11×10⁻¹⁵。
- 72项产物SHA重新校验一致。八格各完成3轮／1,548更新，选中epoch均对应原full-validation answer-token NLL的最早最小值，没有换成数值token NLL或按test重选。
- 两场景各7项contrast均与冻结名称、方向和公式一致：10项seed17差值可计算，4项门控／池化扩展差值保持缺失；14项跨seed均值／SD全部null，保留1/3或0/3状态。旧神经预测器已有完整三个seed的均值／样本SD也独立核算一致，未与当前单seed模型混用。
- 共123项只读一致性检查全部通过，详情和来源SHA见[机器记录](benchmark_explanation_review.json)。其中1e−12仅用于核对已保存浮点算术的一致性，不是新增基准分数阈值或运行准入标准。

## 重点解释边界

| 核查点 | 实际判定与来源 |
|---|---|
| neutral保留prompt | 已明确只移除逐样本变化的残基消息，仍有末端化学和assay字符串、监督格式／标签分布适配；没有context-only拟合，未定量归因assay贡献。[数值补核](numeric_reference_audit.md) |
| TRAIN mean通道 | 保存的TRAIN标签均值是每对象同一有限数值的predictor，不是严格JSON语言生成、旧常数prefix或当前neutral reader；六控制为3算法×2场景。[数值控制](numeric_reference_audit.md) |
| 参数与预算 | 旧30轮MSE预测器和本轮3轮答案CE的曝光、参数、选点及通道分列；ridge53系数与kNN记忆不冒称等同reader容量。原family-macro与已有exact-target macro重新汇总明确分开，主MAE未改。[预算与宏平均](numeric_reference_audit.md) |
| early不是prefix | 明确为embedding后的动态residual reader；旧MLP／真正prefix另列。同构reader和8,192字节每forward载荷不被说成同FLOPs、实测搬运或相等有效自由度。无slot／type编码和原prompt语义缺口也已披露。[接口边界](README.md) |
| 单seed与固定差值 | 首八格仅seed17；mid绝对MAE较差与其real−neutral相对差较好同时保留。DiD不是独立复验，组bootstrap不冒充seed波动或物理实验重复。[八格结果](interim_mechanism_8cases.md) |
| gate与残差 | alpha与tanh(alpha)、更新前forward统计与更新后gate分清。gate符号、幅度和δ/h不解释为信息量或贡献；BF16加法前δ不是实际h_after−h_before，未据此声称舍入吞没残差。[gate诊断](interim_mechanism_8cases.md) |
| NLL与C1 | 原逐样本完整答案mean-NLL差保留；格式／numeric／mixed只是诊断贡献，按各行完整答案长度归一。类别均值、token池化均值和原C1行均值不混加；原CE与诊断CE的微小残差原样单列。teacher forcing不冒称自主数值生成或生物因果，donor为0时P/H仍未定义。[C1账本](interim_mechanism_8cases.md) |
| 尺度候选未执行 | S1／S2仍为后续迁移假说，没有标量校准或新训练；当前36项不变。拟议统计是TRAIN prompt-only，现有观察含validation teacher-forced答案，二者不互换。额外校准／LN／缩放计算单列，不称免费同算力。[尺度方案](residual_scale_literature.md) |
| 文献与因果 | NormFormer／ReZero被限定为机制动机，未称冻结Qwen蛋白连接器的已验证改进；activation范数与原文梯度分析、tanh门控与ReZero无界alpha均分开。AdamW零任务梯度仍可能发生weight decay也已说明。[文献边界](residual_scale_literature.md) |

没有因审查而新增拟合、改prompt白名单、增加新主要端点、修答案或调整现有36项。现有局限属于已披露范围，不构成本次新的阻断：三seed尚未齐、场景样本交叉、缺严格同条件／独立donor、固定消息没有显式分子类型、范数关联未隔离因果作用。后续仍应按已冻结计划报告全部seed及未完成项。

本次没有复读全部原论文PDF或重新访问其官方网页，也没有重新证明所有源数据库无偏差／预训练无曝光；审查的是文献转用边界与本地数据、数值和表述一致性。完整已读来源SHA、检查逐项结果、选点及contrast算术均保存在机器记录中。
