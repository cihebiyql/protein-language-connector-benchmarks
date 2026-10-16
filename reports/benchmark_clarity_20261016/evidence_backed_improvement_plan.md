# 有证据的改进方案：先完成原扩展，再检验两类新机制

目前最有依据的行动是完成已预登记的门控与池化扩展，并将尺度校准、任务表示预热作为独立的未来对照。九篇论文和作者源码支持这些机制的研究动机；它们没有证明这些组合能改善当前pKD任务。本计划不承诺分数提升，不调整正在执行的36项矩阵。

本报告依据[19:29冻结16项快照](interim_16cases_20261016_1929/matched_summary.json)、[16项机制核验](interim_mechanism_16cases.md)、[估计对象澄清](causal_estimands_note.md)、[九论文中的原七篇](literature_evidence.json)、[残差尺度两篇](residual_scale_literature.json)及[作者源码补核](residual_code_evidence.md)。没有读取2026活跃case、执行新forward、拟合、校准、GPU或下载；只新增本方案的MD／JSON。

## 当前证据给出的约束

两种子的反向结果必须保留：双新depth-real为seed17 **−0.069366**、seed42 **＋0.345709**；已见靶点content-mid为 **＋0.210095**、**−0.356790**。因此稳定的内部优势和普遍的real消息收益均未确立。三种子mean／SD仍为null，不以两种子均值代替原统计。

real与neutral分别训练得到θ_R、θ_N。已有`L(θ_N,m̄)−L(θ_R,{m_i})`估计训练、选点和评测共同改变的系统总效应。固定θ_R的均值替换没有登记／测量；C1 off关闭整个分支，不能代替该均值干预。未来如单独登记同θ替换，它仍是计算依赖／分布外敏感性诊断，不能充当合法生物donor或补填当前P／H。

当前16项的验证范数包含全部有效prompt和teacher-forced答案位置，不能当作下面所需的**TRAIN prompt-only**尺度校准。gated δ账本是BF16 cast后、相加前的δ，不是实际`h_after−h_before`；范数差与误差的关联尚未识别因果链。输出分布集中、格式NLL收益和有限排序关联可以并存。

## 方案一：零初始化门控与近单位门控

**状态：当前已预登记，扩展在19:29快照中尚未完成。** 继续原`mid / real / balanced`的alpha0与alpha3六次扩展及其复用对照，不增加条件。

**原文事实。** Flamingo PDF p5 §2.2/Fig.4、p8 Table3以零初始化tanh门控保持预训练LM初始行为，并给出门控消融。ReZero v2 p1 Eq.(1)、p2使用直接乘法的零标量，其作者源码`rztx.py:47`同样建立零参数；源码没有tanh和分支LN。[Flamingo](https://arxiv.org/abs/2204.14198) · [ReZero](https://arxiv.org/abs/2003.04887v2)

**本课题组合与假说。** 保留冻结Qwen与单位置Reader，用相同初态张量，仅将已声明的alpha初值设为0或3。假说是身份保持初始化能改善本训练预算内的优化。alpha3的`tanh(3)`接近1，仍是可学习门控，不能称无门控复现。

**可证伪对照。** 使用原七项contrast中的`gate_zero_initialization`及原分数／失败规则，逐种子报告；同时看既有训练／验证NLL、gate和δ／hidden轨迹。若零初始化未在任务端点呈现一致收益，便不支持它在本配置的普遍改进；轨迹更平顺也不单独构成任务收益。零gate下某些任务梯度为0，不等于当前AdamW首步所有非gate参数都不更新；非None零grad仍可能接受weight decay。

**预算。** 两组Reader均464,001个可训练参数，sender为0；每对象每forward读取16×128 float32，即8,192逻辑字节，同一单层动态Reader及原更新计划。实际EOS读取总量、耗时、显存照原账本报告，不将零初始化视为额外算力免费或相同计算影响。

## 方案二：固定瓶颈中的池化配额

**状态：当前已预登记，扩展在19:29快照中尚未完成。** 继续原`mid / real / alpha0`的balanced8＋8与global16六次扩展，不选择新的K或投影种子。

**原文事实。** Flamingo p5 §2.1用固定64 latent的学习型resampler；BLIP-2 p3用32×768查询瓶颈并配合表示学习目标。它们支持将变长编码器特征压成固定接口的动机，未验证当前无参数池化规则。[BLIP-2](https://proceedings.mlr.press/v202/li23q.html)

**本课题组合与假说。** 本轮固定逐残基归一化、池化、640→128随机投影和逐槽归一化。balanced独立给肽与靶点各8槽，global16池化拼接序列；假说是固定分子配额减轻长靶点挤占。当前无slot／type embedding，8＋8只改变压缩配额，不能称类型对齐或分子对应关系学习。

**可证伪对照。** 只使用原`balanced_pool_allocation`，相同Reader初值、数据、位置、种子与更新，保留反向结果。若global16同样或更好，当前固定配额假说不能获得支持；即使balanced较好，也仅支持这两条具体压缩规则的系统差异，不能把作者学习型resampler的结论移植过来。

**预算。** 两发送器均无训练参数，同16×128 float32载荷与Reader464,001参数；独立池化与拼接池化的预处理／缓存构建时间及内存分别记账。相同K和消息字节不意味着每个原残基拥有相同权重，实际计算也不能仅由参数数目推断。

## 方案三：S1尺度校准与S2分支内无affine LN

**状态：未来未执行；另立协议，排除在当前36项之外。** 以下复用已有[残差尺度方案](residual_scale_literature.md)公式，当前不估计任何系数。

**原文与源码事实。** NormFormer v2 p3未编号公式在attention输出分支内增加LN；其作者fairseq源码`transformer_layer.py:451–468`把`attn_ln`放在dropout及残差相加之前。p6–7分析层间梯度失配，p8的可学习skip缩放在1.3B上无益。该论文的skip缩放与本方案的新分支δ缩放不同。ReZero和Flamingo分别支持零初始化分支的动机；当前冻结LM保持原归一化，不能套用ReZero整网identity／isometry结论。[NormFormer](https://arxiv.org/abs/2110.09456v2) · [固定作者源码](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/modules/transformer_layer.py#L451)

**未来校准输入。** 对预定placement、seed和消息条件，用未训练Reader、冻结base和固定TRAIN样本ID，仅输入原prompt，不拼gold答案；不使用val／test或pKD标签计算尺度。定义`r(a)=sqrt(mean_valid_TRAIN_prompt_positions ||a||²)`，h为分支未改变的接收隐藏状态，u为未门控Reader输出。每个校准条件使用各自预先声明的real或neutral消息；相同样本、初始化、padding及统计规则用于S1／S2，全部系数和指纹在训练前冻结。改变内容条件时独立校准本身也进入系统处理，不能解释为纯内容作用。

**S1：固定幅度单位。** `s=r(h)/r(u)`，随后使用`h_new=h+tanh(alpha)·s·u`，alpha初值0。初始未门控相对RMS单位设为1，是预先定义的尺度基准，不是从当前val／test选出的最优值。r(u)为0或非有限等无效统计应拒绝校准；不能转用val或test补齐。

**S2：先改变分支形状，再匹配潜在幅度。** 在1536维output projection之后、gate之前计算`v=LN_affineFalse(u)`，再取`c=r(h)/r(v)`，用`h_new=h+tanh(alpha)·c·v`。LN跨最后1536维，保持每个位置的分支归一化。比较S1与S2，才可在共同初始相对RMS规则下检验输出归一化形状的作用。只加LN不保证early／mid相对注入匹配。

**零点性质与机制边界。** 有限的s／c及alpha0使新增分支为0，保留原冻结base行为。不能替换成`LN(h+δ)`，那会在alpha0时仍改变h。S1／S2同时改变gate梯度和有效优化参数化；比例变近本身不是任务信息或任务改善证据，也不保证训练后两深度相对范数始终相同。

**可证伪匹配对照。** 未来预先固定unscaled／S1／S2、placement、real／neutral和种子，对比相同Reader初值、数据／更新／选点预算；S1与S2使用相同TRAIN prompt校准集与基础前向，可在同一预处理账本同时记录r(h)、r(u)、r(v)，计入实际费用。报告原任务端点、原C1、gate／raw／gated δ／hidden，必要的额外测量另行预登记。若尺度／形状匹配没有带来任务端点改善，便限制该机制解释；不能从当前16项的含gold验证范数构造系数后称为TRAIN校准。

**预算。** S1仅增加冻结标量buffer，无affine的S2增加0训练参数，Reader仍464,001；若未来改用affine LN，则增加3,072，成为467,073，需另列条件。源消息仍8,192字节，但校准prepass、统计与缓存、每位置缩放和LN的O(BLH)运算、激活／反向及存储均计账。共同校准缓存也不是零成本；相同参数与消息预算不意味着相同实测计算预算。

## 方案四：Reader的TRAIN任务表示预热

**状态：未来未执行；不加入当前36项。** 这是BLIP-2／ProtT3阶段化思想的pKD迁移方案，不是当前模型已经具备的训练。

**原文事实与差异。** BLIP-2 p3及p6–7/Fig.5研究表示阶段再接生成阶段；ProtT3 p2、p4–6/Table4及p16研究蛋白文本对比／匹配／captioning再做生成，并在Galactica上使用LoRA。作者LAVIS源码直接联合ITC／ITM／LM损失。当前ESM和Qwen全冻结、sender全固定，无这些原目标或LoRA；原论文收益不能直接转成新辅助回归损失的效力。[ProtT3](https://aclanthology.org/2024.acl-long.324/) · [作者源码](https://github.com/salesforce/LAVIS/blob/05cb670f47296522e24e7727e366b4a997d28582/lavis/models/blip2_models/blip2_qformer.py#L268)

**具体可学习路径。** 给固定sender的输出“加损失”不会让它学习。未来预热应训练Reader及真实辅助头：

1. 对TRAIN样本只输入原prompt，用冻结、分支off的base提取预先固定位置的query隐藏状态，例如最后一个有效prompt token；不拼接gold答案或数值作为输入。early与mid各按其原位置提取，位置语义保持可追溯。
2. 用固定消息z和Reader得到该位置的**未门控**1536维输出`u_i=Reader_ungated(h_prompt,i,z_i)`。辅助head读取u，不读取`tanh(alpha0)·u=0`，从而避免零gate截断辅助任务对Reader的梯度。
3. 增加`ŷ_i=wᵀu_i+b`，以TRAIN标签的原两位小数训练答案标量为目标，例如`L_aux=mean_i(ŷ_i−y_i,2dp)²`。标签只作为监督，不进入query；主测试仍按原未舍入gold评分。辅助损失的具体权重λ和任何训练内标准化需在未来正式协议中固定，当前不估计或挑选。

这条路径可以更新Reader非gate部分和head；aux没有使用alpha，不能靠它预热gate。主CE步骤仍训练原门控路径。辅助MSE单位为pKD²，主CE单位为nats／答案token，不能把两者数值大小当成相同尺度的收益。

**主匹配对照：阶段顺序与交错。** 两臂均包含同一个活跃Linear1536→1 head和同样的Reader／head初值：A先做T_aux个辅助更新，再做T_CE个原CE更新；B按预先固定的交错次序做同样T_aux和T_CE更新。匹配每个损失的样本ID／顺序／曝光、梯度累计、λ、优化器组、调度和验证调用次数，只改变阶段顺序。T_aux、T_CE、λ、交错方案和验证时刻留待未来协议，依据理论及TRAIN／val确定，在任何新测试推理前冻结；不根据本轮测试分数挑值。优化器状态随次序变化属于该处理本身，应记录而不是隐去。

**参数与通信账本。** Linear1536→1有1,537训练参数，联合训练参数为`464001+1537=465538`。head后来丢弃不改变训练账本；推理时Reader才回到464,001。两臂的head都实际接受aux梯度，不能给CE-only挂一个未使用head凑数。若额外列CE-only参考，它仍是464,001参数且无aux曝光的非同预算参考，应单列而非包装成主匹配对照。

aux可以只对最后prompt位置读取一次消息，CE仍按原完整序列查询；每次对象消息载荷维数不变，但总Reader读取、prompt-only base prepass、可选隐藏缓存、head运算、aux反向与标签曝光都增加。阶段与交错两臂需要匹配这些累计费用；不能只说“仍16槽，所以预算相同”。未来若保留既有CE更新并额外添加aux，必须完整报告新增成本。

**可证伪结论范围。** A相对B的原任务差回答阶段顺序的作用；即使训练辅助MSE下降，也不能替代held-out pKD／生成结果。若阶段化未改善或方向不稳，则不支持该顺序在本配置的普遍效力。它不会证明sender变成可学习、完全复现ProtT3，或固定检查点逐例读取正确分子机制。

## 四方案的状态与预算总览

| 方案 | 截止本冻结阶段的身份 | 可训练参数 | 源消息 | 额外计算／监督 |
|---|---|---:|---|---|
| alpha0对alpha3 | 原36内，扩展未在16项快照完成 | 双方464,001 | 双方16×128 float32 | 原更新预算；实际生成读取照实记录 |
| balanced对global16 | 原36内，扩展未在16项快照完成 | sender0，Reader464,001 | 同上 | 两类预处理／缓存成本分别记录 |
| S1／S2 | 未来独立未执行 | 464,001；S2无affine | 同上 | 无标签TRAIN prompt-only校准＋缩放／LN，不使用当前val含gold范数 |
| Reader预热对交错 | 未来独立未执行 | 联合465,538；推理464,001 | 每次对象载荷相同 | 两臂同T_aux／T_CE与样本曝光，另计base prepass、head及aux反向 |

固定θ_R的均值替换、合法供体干预、完整原子事实金标准等属于评价缺口；这里只登记缺口，没有补做或将它们归为改进已有效的证据。原探针、C1与原任务端点继续回答不同问题，不合并为一个“信息量分数”。

## 九篇原文与源码的可追溯性

| 原文 | 相关物理页 | 在本计划中的用途 | PDF SHA256 |
|---|---|---|---|
| Flamingo，2022 | p5、p8 | 门控、固定瓶颈的机制依据 | `5b96767ada8d76ad160c3087d80975b8af8e60b46786b10248c6b96b0c4699ba` |
| BLIP-2，2023 | p3、p6–7 | 查询瓶颈、表示／生成阶段 | `d59913d16edf1dc7e661ca3510c6efaa2b001146840d2e464a7fde980d088dd2` |
| ProtT3，2024 | p2、p4–6、p16 | 蛋白文本两阶段及其适配边界 | `ffa65fc93979bcaf3cff2c60eaa4b832dff9492508f3493112755c3ca97038a1` |
| ProtST，2023 | p3–4 | 作者更新PLM的对齐机制与本轮冻结边界 | `db7f7d620f05ed08e0b4b53f4f52a24eed535eed7a825ed4e1c48dbf34610633` |
| Conditional probing，2021 | p2–3 | 给定基线／函数族的可读出性 | `0d44d3b5c26ab2dbbffbc506e6d2535520bf46af79d061807fbfd7550f0545d1` |
| Control Tasks，2019 | p1–3、p5 | 探针学习与真实表示证据区分 | `557f5d7dc18b17d641c9e54c91c0071b00612f9ec051fc788bcc7117f4d194d0` |
| ESM-2，本地2022预印本／正式2023 | p3–4 | 结构读出不能代替条件pKD能力 | `951003574aafbd1a93b0baf5a6b7ef1fd540af8db3fce6fa42f350c4413c3061` |
| NormFormer，arXiv v2 2021 | p2–3、p6–8 | 分支内LN、梯度动机及负结果 | `c3bd7416cf2e37290a3a0ac807f193c344a905605afb86aad9bef3fd14bfc502` |
| ReZero，arXiv v2 2020 | p1–2、p8、p12 | 零门控初始化及迁移边界 | `54eea1741fad2e7a20e52f55c64b34977f1206c5e3c8e14804ea5385c299c070` |

完整官方URL、版本和本地PDF位于[本方案JSON](evidence_backed_improvement_plan.json)。其中NormFormer源码固定commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`，ReZero为`e2c94a825c5564217e8cf4d75a28d59cab1d7029`；两者是核验过的作者仓库快照，未确认为论文原实验commit。NormFormer实际分支LN与ReZero直接零标量的代码事实保留，原论文收益、私有数据或整网理论均不套成本任务结论。

下一步以完成原已登记扩展为先。未来方案的实现、超参数及比较表须另行冻结；本文件只形成具体、可证伪、可计账的方法组合，不宣称全球首次或保证改分。
