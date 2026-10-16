# 固定36次对照的预先解释计划

本计划把已经冻结的七个配对 contrast、C1 token 分解和门控／范数诊断，映射到可回答的科学问题。撰写时只读取本轮 protocol、实现源码和已核对的论文证据，未读取本轮任何测试分数。36次运行矩阵、主指标、解析器、失败规则、种子、统计方法和旧结果均保持原定义。[固定协议](../../historical/configs/matched_budget_20261016.protocol.json) · [七篇文献证据及原文指纹](literature_evidence.json)

## 本轮到底比较什么

主要矩阵是两个独立场景 × 三个配对种子（17、42、2026）× early／mid × real／neutral，共24次训练。另有6次近单位门控初始化、6次全局池化扩展，总计36次。四个主条件使用同一个无参数发送器和同结构的464,001参数接收器；每次前向读取16×128 float32消息，即8,192字节的逻辑载荷。early在第一个解码块前动态添加跨注意力残差，mid在第14个块后添加；early属于动态残差接入，不是旧MLP前缀或经典soft prefix。

固定的是接收器结构、配对初始权重、消息维数、读取次数、训练文本及更新计划。不同接入深度仍具有不同的隐藏分布、query语义和梯度传播路径。early的query来自尚未经过解码器上下文化的token表示，mid来自已有上下文的隐藏状态；这是本轮depth处理本身的一部分。不能把结果进一步拆成“纯层号效应”或“纯上下文化效应”。训练逻辑通信预算匹配；实际运行耗时、FLOPs及因EOS而变化的生成读取总量按原账本报告。

neutral为该场景训练消息的逐槽均值，所有样本使用同一张量；它移除样本变化的分子消息，同时保留允许提示词上下文和可训练分支。correct表示分支开启，对real和neutral都适用；off是另行关闭分支的冻结LM推理参考。

balanced的“8肽＋8靶点”只指发送器独立池化配额。当前接收器没有slot位置编码或分子type embedding，标准KV注意力对同步排列的key/value行保持排列不变；因此不能把8＋8称为类型对齐、分子身份识别或肽—靶点位置对应。global16改变拼接序列的池化分配，比较它与balanced可检验这一具体压缩规则。

## 七个固定contrast的解释

下表完全沿用协议方向：正MAE差值有利于所列处理；差值仅在相同完整样本总体且所有答案均有效时定义。每个场景和种子分别展示，随后执行协议已经规定的汇总，不新增显著性门槛。

| 固定名称／公式 | 若差值偏正，可支持什么 | 若接近零、方向不稳或偏负，如何解释 | 必须联合阅读的证据 |
|---|---|---|---|
| `depth_real`：early-real MAE − mid-real MAE | 在固定消息与接收器预算下，这个mid系统的真实消息任务误差较低 | 该设置没有稳定呈现mid收益；可能存在优化、表示适配或任务异质性，不能仅凭差值确定原因 | `depth_neutral`、DiD、逐种子结果、覆盖率、gate／norm；单独此项不能归因于分子内容 |
| `depth_neutral`：early-neutral MAE − mid-neutral MAE | 即使消息不随样本变化，mid也能带来通用适配／校准收益 | 通用适配的深度收益弱或方向改变；不能据此断定real消息无用 | real深度差和格式／数值NLL；提示词仍含允许上下文 |
| `content_early`：early-neutral MAE − early-real MAE | early系统从变化的真实分子消息中获得任务收益 | 在该预算和训练计划下，real消息相对训练均值没有显示同方向收益 | 同种子的输出覆盖和C1分解；neutral是特定均值控制，比较没有穷尽所有无内容基线 |
| `content_mid`：mid-neutral MAE − mid-real MAE | mid系统获得超出固定均值分支的样本相关消息收益 | 当前mid系统可能只学到通用校准，或不能有效读出压缩消息；两者须由诊断区分 | `content_early`、DiD、数值token诊断与既有表示读出范围 |
| `depth_content_difference_in_differences`：`depth_real − depth_neutral` | mid相对early的收益在real消息下更大，符合深度与样本变化内容存在交互的解释 | 若real和neutral深度收益接近，主要结果更符合通用深度适配；若为负，内容收益更偏向early | 两个主效应、原四个条件及输出覆盖；仅是该四格系统的交互，不能直接建立生物因果机制 |
| `gate_zero_initialization`：mid-real-balanced-alpha3 MAE − 同条件alpha0 MAE | 零初始化门控在相同训练预算内有益，符合身份保持初始化的动机 | 可能是门控打开速度、有限训练预算或本任务不需要该保护；结果不能单独否定原论文 | gate轨迹、初始／训练残差范数比、验证NLL和梯度。alpha3的tanh接近1，仍是可训练门控，不是无门控条件 |
| `balanced_pool_allocation`：mid-real-global16-alpha0 MAE − mid-real-balanced-alpha0 MAE | 独立8＋8池化优于该全局16槽分配，符合避免长靶点挤占配额的假设 | 固定配额可能浪费槽位，或压缩损失来自其他部分；不能由此认定分子长度无关 | 两种发送消息的固定投影／归一化、旧数据长度资料及同种子训练诊断；没有显式类型对齐结论 |

这些contrast共享运行，彼此相关。代数上，DiD同时等于`content_mid − content_early`；不能把七项同方向差值描述成七次独立复验。三个种子提供本次有限的训练波动观察，不能据此声称已经充分估计随机性或得到强泛化结论。遗漏／失败种子的处理继续遵守协议：只有3／3完整有效才给三种子均值与样本标准差，其余保留null及缺项。

### depth结果的几种联合形态

- real和neutral都呈现类似mid收益：优先解释为通用接入深度／校准效应，分子内容的额外贡献由content差与DiD回答。
- real呈现mid收益，neutral没有相同收益，且content-mid也有利：与mid更有效利用变化消息相容；进一步结合数值NLL及覆盖率，不将交互直接升级为蛋白生物机制。
- real在mid较好，但neutral在mid改善更多：real深度差仍成立，DiD反映的额外内容作用较弱或反向。
- content在两个深度均有利，而depth差方向不稳：支持当前消息存在任务价值；尚无稳定的接入深度优劣。
- 各种子出现方向变化：保留逐种子差值和协议汇总，检查训练／验证轨迹；不选择最佳种子，也不改测试阈值或评价总体。

## C1：把格式改善与数值改善分开

本轮对同一已选检查点，使用相同gold答案及token ID，分别计算correct与off的teacher-forced NLL。format包括JSON格式、EOS／special及非数值字符；numeric是完整落入唯一pKD数值字符区间的token；跨格式与数值边界的token归mixed。真实Qwen中负号可能与冒号合并，保留mixed可避免错误归因。类别mask只核对原token序列，不改变分词、标签或原完整答案NLL选点。

沿用原C1行平均定义，对样本i、类别c定义：

`G_c(i) = [loss_sum_off,c(i) − loss_sum_correct,c(i)] / n_full(i)`，

`G_c = mean_i G_c(i)`，单位为nats／完整答案token；三类相加得到诊断G。原fp32完整答案reduction保持为原C1值，诊断与它的微小归约舍入差单列。类别自己的`loss_sum/count_c`可用于描述类别难度，三类类别均值不能相加，也不能代替原G。未包含某类token的样本，其该类full-token贡献为0，类别均值保持未定义。

| 可能形态 | 支持的解释 | 仍需区分的问题 |
|---|---|---|
| G主要来自format，real和neutral表现类似 | 分支明显帮助答案模板、终止或格式先验 | 分子内容的数值收益需看numeric／mixed和原自主生成误差 |
| real相对neutral的numeric贡献更有利，且原MAE同向 | 数值答案敏感性与实际生成的内容收益相容 | teacher forcing依赖真实前序答案token，仍不同于完整自主推理 |
| numeric NLL改善但MAE无相同改善 | 提高给定gold前缀下的局部概率，没有转化为当前贪心生成收益 | 可能涉及token序列、暴露误差或数值校准，不能据此更改解码／解析规则 |
| mixed贡献突出 | 边界合并token对效应占有明显份额 | mixed同时包含格式与数值，不能全部算作纯数值能力 |
| G有利但content contrast弱 | 消息分支有计算作用，样本变化内容的任务价值尚不清楚 | 常数分支、允许上下文重编码和格式改善均可产生这种形态 |

C1是模型内部计算敏感性：gold数值已作为teacher forcing输入的一部分。它既不等同自主生成正确率，也不建立分子实验因果作用。当前严格独立匹配donor仍为0，相应P／H保持未定义。

## gate、范数与优化诊断如何解释

门控轨迹、梯度范数、验证NLL、raw／gated delta与hidden范数比均作为机制诊断；本计划不为它们新增阈值或据其选择测试表现最佳的架构。

- alpha0初始使新增残差为0，接收分支在首步存在预期的零梯度部分，门控可以先更新。首步零分支梯度是该参数化的性质，不能直接报成训练故障。
- 零初始化比alpha3更稳定，且伴随较小的早期gated-delta扰动、较平顺训练／验证轨迹：与Flamingo的身份保持动机相容；优化因果链仍需依据实际轨迹和固定contrast解释。
- gate持续接近初值且content收益弱：提示当前训练未充分利用该通道；不能仅凭gate值断定消息没有信息，因为输出投影与注意力权重也可改变残差。
- raw delta变大而gate变小：先看gated delta／hidden比。alpha单独不标识实际作用强度，存在缩放补偿。
- mid的delta／hidden比与early不同：说明该实现相对对应隐藏状态产生不同幅度的扰动；两个深度hidden分布本身不同，不能用同一绝对范数直接排序优劣。

## 既有C2能说到哪里

旧C2比较的是指定B、指定message摘要／128维投影及线性岭回归族下的额外可读出性。六个非恒定消息摘要增益的既有区间跨0，说明该诊断没有给出稳定的额外线性读出证据。它没有证明完整消息或ESM残基中不存在任务信息，也不是总互信息、信息守恒或生物机制定理。摘要可能有损，有限数据与正则选择也限制结论。[条件探针原论文p2–3](https://aclanthology.org/2021.emnlp-main.122/)

探针可读出性与LM实际使用分别由条件探针和C1／原任务结果回答；旧C2不作为本轮筛掉种子、改变端点或选择新改进的测试打分器。控制任务论文进一步要求把探针记忆能力与真实性能一起看，本轮未新增这类实验。[控制任务原论文p1–3、p5](https://aclanthology.org/D19-1275/)

## 有原文依据的改进方向与本轮执行身份

| 方向 | 已核原文依据 | 本轮身份 | 能检验及不能等效的部分 |
|---|---|---|---|
| 零初始化门控 | Flamingo，2022，PDF p5 §2.2/Fig.4、p8 Table3；SHA `5b96767ada8d76ad160c3087d80975b8af8e60b46786b10248c6b96b0c4699ba`；[原论文](https://arxiv.org/abs/2204.14198) | 已固定36矩阵内的alpha0／alpha3扩展 | 检验当前mid接收器的初始化作用；alpha3不是无门控，当前单插层不等效原Flamingo多层系统 |
| 固定消息瓶颈与池化配额 | Flamingo p5 §2.1的固定64 latent；BLIP-2，2023，PDF p3的32×768 query瓶颈，SHA `d59913d16edf1dc7e661ca3510c6efaa2b001146840d2e464a7fde980d088dd2`；[BLIP-2](https://proceedings.mlr.press/v202/li23q.html) | 本轮所有方法固定16×128；balanced／global16扩展 | 固定瓶颈思想有来源；本轮发送器是无参数池化＋固定随机投影，8＋8是本课题配额假设，不是原学习型resampler或类型对齐复现 |
| 训练分区内的任务表示warmup | BLIP-2 PDF p3、p6–7/Fig.5；ProtT3，2024，PDF p2、p4–6/Table4、p16，SHA `ffa65fc93979bcaf3cff2c60eaa4b832dff9492508f3493112755c3ca97038a1`；[ProtT3](https://aclanthology.org/2024.acl-long.324/) | 候选，未纳入或执行本轮36矩阵 | 原文支持阶段化表示／生成训练动机；pKD辅助数值头及其损失是本课题迁移。未来需先冻结阶段预算与对照；原ProtT3还包含LoRA及不同底座 |

ProtST（2023，PDF p3–4，SHA `db7f7d620f05ed08e0b4b53f4f52a24eed535eed7a825ed4e1c48dbf34610633`）提供mask与序列—文本对齐依据，同时作者更新PLM；本轮固定ESM缓存不开放该途径。[原论文](https://proceedings.mlr.press/v202/xu23t.html) ESM-2本地预印本p3–4（SHA `951003574aafbd1a93b0baf5a6b7ef1fd540af8db3fce6fa42f350c4413c3061`）提供结构读出证据，不能替代条件依赖pKD的验证。[作者正式引文与源码](https://github.com/facebookresearch/esm)

其他两篇依据为条件探针（2021，p2–3，SHA `0d44d3b5c26ab2dbbffbc506e6d2535520bf46af79d061807fbfd7550f0545d1`）和控制任务（2019，p1–3、p5，SHA `557f5d7dc18b17d641c9e54c91c0071b00612f9ec051fc788bcc7117f4d194d0`）。完整题名、版本、URL、本地PDF和适配边界保存在[七论文证据卡](literature_evidence.json)。

后续报告按上述问题逐项解释实际结果，保留反向结果、输出失败、未完成运行及原规则下未定义的统计量。本计划没有新增显著性标准、优胜阈值、测试选择步骤或训练条件。
