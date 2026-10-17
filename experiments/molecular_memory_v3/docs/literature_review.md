# 分子记忆V0–V2：附件架构的原文核验与评价依据

附件全文已读完。五条重点新引文均能在官方来源核到题名与版本；它们支持部分设计动机，没有哪篇直接验证了“ESM分层typed bank＋独立内部Reader＋肽—靶点关系＋条件＋数值／语言双输出”这一完整组合。V0可以使用成熟算子建立受控接口；V1／V2仍应逐个验证信息来源和新增容量的作用。[逐篇页码、版本与本地SHA](literature_review.json)

本报告只读既有PDF／已存证据及官方网页，不下载模型、执行论文代码、拟合或启动GPU。新目标时间边界为2026-10-17 00:29:40至10:29:40（+08）；不重开旧训练合同，也不将文献事实写成本课题已运行结果。

## 1．五条新引文核验：身份真实，迁移必须限定

| 附件引用 | 核实版本与原文机制 | 对附件的修正／适用边界 |
|---|---|---|
| Draft-KV，2609.34754 | 首投2026-09-28；本地为2026-10-11 v2、41页。p5 §4.1–4.2：发送者自行生成draft，投影**draft位置KV**到独立side memory；Reader复用接收者冻结Q／O和head layout、每KV-head零初始化tanh gate。p6 §4.3：重建→答案对齐→含错配损害guard的任务训练，并回放重建 | 是预印本；官方v2明确更正了模板误留的会议抬头。静态ESM残基不包含问题相关draft计算；独立可训练Reader也不等同复用冻结Q／O。不能把收益直接移植为“加side memory就有效”。[官方v2](https://arxiv.org/abs/2609.34754v2) |
| PLM层分析，2602.20449 | 本地v1，2026-02-24、16页。p4–5：每层接任务MLP，按分类置信度early-exit；p7–9比较GO／EC／定位和二级结构，提出most-confident-layer fallback | 支持任务相关表示不必集中在末层。它没有验证ESM2-150M的第15层最佳、15＋30融合最佳、残基差异被保留或pKD更准；选层／阈值须在训练／验证范围内固定。[官方v1](https://arxiv.org/abs/2602.20449v1) |
| Prot2Chat，btaf396 | 正式Bioinformatics 41(8)，2025-07-28发表、8月typeset。本地PDF是2025-05-22 arXiv v2，首页仍有“Journal Title Here,2022”占位，不能据此写发表年。局部p5 Eq.7–11：ProteinMPNN序列＋结构、问题末token向量加入学习查询，压缩成输入soft prompt；同时训练LLaMA3 LoRA | “问题指导蛋白压缩”有先例；作者不是双冻结ESM/Qwen内部读取。局部版本共约109M训练参数，不能当当前小接口性能证据。[正式文献](https://academic.oup.com/bioinformatics/article/41/8/btaf396/8215464) |
| Gated Attention，NeurIPS2025 | 正式会议记录可核。p2–4、Eq.5：Q相关sigmoid乘法门，比较headwise／elementwise及位置；推荐的是**SDPA输出后、WO前**的head-specific elementwise门 | 作者训练完整dense／MoE语言模型。其门既不同于单个零标量tanh，也不是类型路由器、置信度或分子证据可靠性；给新增Reader用该门是新迁移。[正式会议](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html) |
| C2C，ICLR2026 | 本地为正式会议29页版。p5–6 §3.3：Sharer／Receiver KV拼接投影、特征融合、input-aware head modulation、Gumbel-sigmoid层门，token／末层对齐，替换／融合Receiver原KV；两LM冻结 | 支持跨空间适配，不能证明可直接复制ESM原生KV。残基与文本token没有作者同文本token对齐条件；我们的外部残差bank也不同于改写native KV。[正式会议](https://proceedings.iclr.cc/paper_files/paper/2026/hash/474ada926b331d78f06d95e8913111cc-Abstract-Conference.html) |

本次没有发现这五个URL不存在；仍未把它们的模型、源码执行或最新方法细节设为V0必需依赖。身份可核不等于作者结论已由本项目复现。Prot2Chat正式网页与本地预印本分别记录，物理页码始终指本地文件。

## 2．V0 typed bank与独立读取：哪些成熟，哪些是本课题假说

**成熟依据。** Flamingo p5 §2.1–2.2/Fig.4及p8 Table3支持冻结LM中的独立跨注意力、固定latent压缩与零初始化门控。BLIP-2 p3–4支持学习查询／投影形成可读接口。ProtT3 p2、p4–6支持蛋白表征接到语言模型及表示／生成阶段分工；作者还训练LoRA。[Flamingo](https://arxiv.org/abs/2204.14198) · [BLIP-2](https://proceedings.mlr.press/v202/li23q.html) · [ProtT3](https://aclanthology.org/2024.acl-long.324/)

**V0迁移组合。** 缓存冻结ESM残基，训练640→d投影；明确肽／靶点／条件bank、缺失mask、对象角色及来源层；给不同bank分开读取后组合，并在一个既定LM位置写入门控更新。它是借用成熟模块的新工程／科学组合；已核论文没有证明这套typed-bank划分必胜，也没有证明位置／类型embedding自然对应生物概念。

角色标记应表示A／B／target／condition等输入角色，不能变成逐样本ID、靶点数据库ID或来源DOI捷径。缺失条件应有显式missing mask，不能填0冒充真实pH／温度。ESM位置表示也不是三维结构坐标。当前亲和力资源的实际构建物／条件缺项不能由条件编码器补出；PPIKB本地p8将细化assay条件列为后续资源扩展。

**对照要回答模块来源。** 相同信息的专项预测器、合并bank、无type标记、共享／独立Reader、无变化消息分支应分别定义；参数、消息长度与读取次数分别计账。独立Reader多出来的容量不能归成“类型信息”收益。trainable投影与旧固定随机投影是不同发送器，不能沿用旧发送器0参数或旧固定bank的信息结论。

**工程事实。** 已存config核到ESM2-150M为30层、640维；Qwen2.5-1.5B为28层、1536维。旧cache实际保存layer30；layer15必须建立新缓存，不能复制layer30再改标签。旧长序列规则是非重叠≤1022残基分块后拼接，保留完整残基覆盖，但没有跨chunk上下文化；“全靶点输入”与“全局上下文”应分开。

## 3．V1多层、问题路由和关系模块

| 候选改变 | 原文支持 | 迁移与可检验差异 |
|---|---|---|
| 两层表示、层相关投影 | PLM层文p4–5、p7–9：层与任务的可用性可变 | 15／30只是候选，不是论文最佳层。新增源信息、投影参数、缓存与读取成本必须对照；单末层、单中层、同预算组合分别报告 |
| 问题相关读取／软路由 | Prot2Chat局部p5 Fig.3/Eq.8–10：问题向量进入压缩查询；Gated Attention p3–4：隐藏态相关门 | 前者是问题指导输入soft prompt，后者是完整LM内部自注意力门；V1路由不同bank／层是本课题新算子。路由权重不能作为正确性解释 |
| 肽—靶点关系表示 | PepInter p4–6分析cross-attention，p6–8进行亲和力回归 | 其大规模预训练／ESM-C与我们的冻结ESM、小关系模块不同。`Attn(Q=peptide,K/V=target)`及`MLP[Z,U,Z⊙U]`是具体迁移提议，未在已核论文找到该完整公式的本任务实证 |
| 输出分支归一化／相对尺度 | NormFormer p3、p6–8及已核作者源码：LN在新增attention分支内，研究梯度失配 | 不等于归一化整个`h+δ`。零gate保持base的性质应保留；TRAIN prompt-only定标是新方案，旧含gold验证norm不能用作它的系数 |

PepInter本地p6给出的两组attention—空间距离关联方向并不一致；这进一步限制“attention就是接触图”的解释。专项预测器应得到同样关系特征，才能区分关系建模贡献与语言侧贡献。[PepInter](https://doi.org/10.1038/s42004-026-02168-3) 对RT等没有target的任务，不造假target，关系bank应按输入语义关闭。

## 4．V2配对差异、numeric auxiliary与分阶段目标

**配对支撑与缺口。** Kliche2024 p2、p4对野生型／突变体成对phage-display计数与结合偏好进行分析，说明真实匹配肽对是有文献来源的评价对象。NGS富集并非pKD；不能把全部成对筛选结果换成同条件亲和力差。当前旧A1严格同条件可核配对仍为0，新panel需逐条核对构建体、化学身份、条件及来源后独立冻结。[Kliche2024](https://doi.org/10.1038/s44320-024-00055-4)

`C=(Z_A+Z_B)/2`、`D=Z_B−Z_A`是本课题提出的表示结构；这两条代数关系本身没有证明生物差异保真。对齐位置、长度和构建体不明确的对象不能逐残基相减。共享数值预测器的`f(B,R,c)−f(A,R,c)`提供强对照，并在相同函数／输入角色下天然反对称；普通下游网络读取D并不自动保证反对称，更不保证自由生成正确。必须评价顺序交换、真实差值／方向及未测配对，不能让合成标签替代实验gold。

**Numeric auxiliary。** 生成前隐藏状态的小回归头是单独的数值通道，不能以其结果替换原生成答案。直接Linear1536→1增加1537训练参数；typed bank、独立Reader、条件／关系／路由都要重新统计，旧464001或“约60万”不能原样沿用。回归头如果读gold答案之后的状态会泄漏；应固定到prompt-only、答案生成前位置。头的辅助MSE是迁移目标，BLIP-2／ProtT3没有验证本课题pKD MSE的效力。

**训练拆解。** BLIP-2 p3–4、p6–7/Fig.5和ProtT3 p2、p4–6、p16支持先建立表示再生成的动机；Draft-KV p6进一步区分重建／对齐／任务及回放。它们不证明附件提出的“事实→定量→解释”顺序最优。原Draft-KV重建时有明确payload传输任务，不能把含功能答案的文本放入分子输入，再声称从序列推得功能。

新方案应将可学习对象和梯度路径写清：给固定sender加loss不会使它学习；Reader辅助预热可读未门控u，避免alpha0截断对分支的aux梯度；主CE仍需经过冻结但可微的后半LLM，不能被整体`no_grad()`切断。阶段顺序／交错的主对照应有同一活跃head、同样aux／CE更新与样本曝光；T、λ及选择预算只能在未来合同的TRAIN／val范围固定，不能从本次test成绩倒选。

## 5．至少三条独立有效性评价：原文、含义与gold门槛

以下六类可互补，前三类也不能互相替代。所有物理页与SHA在JSON；当前缺项保持缺项，不把设计写成执行成绩。

| 机制 | 确切原文依据 | 对V0–V2可执行的迁移 | 当前边界／分母 |
|---|---|---|---|
| **实验量测预测＋独立留出** | PepInter p8 Table4、p14 Eq.15–17：MAE／RMSE／R²及排序；DataSAIL p1–2、p4：相似性捷径及双对象留出 | 分子身份／来源／家族隔离的已冻结pKD任务，同信息专项预测器、数值／生成分开报告；相同完整测试对象的配对误差比较 | PepInter的混合Kd／IC50／Ki和五折设置不能直接比较本项目Kd-only分组留出分数。全样本覆盖与失败数保留；微平均／target宏平均名字分开 |
| **同一θ的消息作用／交换** | Draft-KV p2 Eq.1、p4、p6、p8–9：Matched／Deranged／Receiver-only；Geiger2021 p7–9：高层因果变量与内部交换对应 | 未来先登记固定checkpoint的off、合法匹配替换或均值替换；保持问题／gold／解码和分母一致，区分支路作用与变化消息作用 | 重新训练θ_R／θ_N是系统总效应，C1 off不是mean替换。现合法独立donor0，P/H未定义；随机交换至多算计算敏感性，不能叫生物反事实 |
| **条件探针** | Hewitt et al.2021 p2 §2.3 Eq.2、p3 §2.6：`[B;R]`与`[B;0]` | 固定B、原始ESM／bank／receiver表示、相同维数与探针预算，TRAIN拟合、val选正则、test仅评估 | 回归ΔMSE单位pKD²，不是总互信息bits。旧六个摘要probe不能替新typed bank提供结果；投影／摘要读不出不等于完整表示无信息 |
| **探针容量与selectivity** | Hewitt & Liang2019 p1–3、p5 Table1：真实任务与身份随机标签控制 | 线性与小MLP容量档；同一分子身份的随机标签一致，保存种子，报告控制任务和真实任务各自表现 | 家族留出下随机身份覆盖有限；随机任务不是功能负例，分类selectivity不直接变成回归信息量 |
| **证据删除／保留faithfulness** | ERASER2020 p5–6 §4.2 Eq.1–3：删除与保留原预测证据、随机控制 | 对相同数量的slot／bank删除与保留，使用固定原答案／候选的概率或loss，配随机mask与合理空／donor控制 | 可能产生OOD；没有明确slot到残基映射时不能叫残基接触解释，更不能把attention热图当gold |
| **原子事实支持** | FActScore2023 p3 §3.1：原子事实、可靠来源支持精度；不评价recall | 解释拆成分子／条件／主张／来源，支持、矛盾、未确立分别记录，人工盲审校准 | 当前核准B2 QA与C3原子事实gold缺失，不能预报准确率；模型自己补常识不构成支持证据 |

评价原文：[DataSAIL](https://doi.org/10.1038/s41467-025-58606-8)、[Geiger2021](https://arxiv.org/abs/2106.02997v2)、[条件探针](https://aclanthology.org/2021.emnlp-main.122/)、[Control Tasks](https://aclanthology.org/D19-1275/)、[ERASER](https://aclanthology.org/2020.acl-main.408/)、[FActScore](https://aclanthology.org/2023.emnlp-main.741/)。

## 6．对附件可直接采纳的结论

V0的成熟基础是冻结编码器、可训练投影、明确输入／缺失语义、可关闭的独立读取和可审计梯度；typed拆分、独立bank优势及具体接入深度仍是新假说。V1的多层、路由与关系应分别消融，并给强专项预测器相同信息。V2只有在真实配对gold足够且普通共享预测器差分仍不足时，才具备检验差异通道的条件。

“非文本内部通信”“问题条件化压缩”“零门控”“多阶段训练”已有明确先例，不能分别作为首次创新。研究贡献应来自受控组合在真实、适用的任务与机制评价上的新增证据。当前文献足以支持可检验设计，尚不足以保证该架构优于预测器＋LLM、自动修复缺失实验信息，或具有生物因果解释。
