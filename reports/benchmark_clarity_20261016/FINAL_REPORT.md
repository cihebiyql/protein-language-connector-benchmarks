# 蛋白质语言连接器：基准定义、核心结果与改进方案

**本轮终态：30项验收通过，1项池化尝试未通过验收，5项未启动；36项计划全部保留。** 核心24项及门控六项共8,910条严格有效预测。池化两个场景的正式覆盖均为0／3，均值和SD为null。[完整终态表](final_snapshot_20261016_231042/matched_summary.md)保留全部配置；未验收尝试的模型、152条原始输出和分数另存，不计入正式结果。

本轮把问题具体化为：在同一固定蛋白消息、同构Reader和原训练／评分规则下，将动态读取从最早位置移到第14个decoder block后，能否稳定降低pKD生成误差。三随机种子核心结果显示，两场景的real深度差都出现两负一正；原均值分别为−0.109102和−0.037186 pKD。当前证据支持“mid优势不稳定”，下一步应区分优化、表示与相对残差尺度，而不更换测试标准。

## 1．先把基准的对象与标准说清楚

### 数据与可回答的问题

A1评价宽泛数据库条件下、以全长参考靶蛋白为输入的肽—靶标pKD文本预测。两场景分别独立训练、计算TRAIN均值和选择检查点：

| 场景 | TRAIN | VALIDATION | TEST | 主要留出问题 |
|---|---:|---:|---:|---|
| 新肽／已见靶点 | 2062 | 147 | 152 | 新肽对已见参考靶标的预测 |
| 新肽／新靶点 | 2063 | 431 | 442 | 肽与靶标同时外推的预测 |

冻结拆分保留来源／家族／DOI等约束。group_id是这些约束形成的连通操作组，不能当作独立物理实验编号或严格相同条件的重复；组bootstrap据此描述来源分组不确定性。两场景来源总体有重叠，分别训练并不使它们成为两次独立实验复现。

实验构建物与参考蛋白不总能核实一致，assay、缓冲液、温度、pH及修饰条件存在缺项，未知值保留原状。因此当前终点对应参考靶标pKD的预测效用，严格同构建物／同条件的效力、作用机制仍需要相应金标准。既有test已经观察，本轮是固定基准上的受控后续比较。

其他基准各有独立验收条件，本轮A1不能替它们完成验收：原A2跨条件RT因源单位／化学身份问题仍HOLD，现有7,070行目标域数值诊断不是跨条件复现；B1只有旧两种架构的六个检查点，test仅两个功能类，没有本轮新连接器B1结果；B2 reviewed QA及C3裁决事实gold均为0，尚不能正式评分。旧C2的摘要线性探针与本轮新固定消息不同。[逐面板任务、分母与缺口](contract_audit.md)

### 四格系统与参数／消息预算

编码端使用冻结ESM2-150M第30层的640维逐残基缓存；接收底座为冻结Qwen2.5-1.5B-Instruct，28层、hidden1536。长序列按不超过1022残基的非重叠chunk编码，覆盖全部残基，但chunk间的原PLM上下文断开。

冻结缓存经逐残基LN、每分子独立池化8槽、固定640→128 Gaussian投影（seed314159）和逐槽LN，形成16×128 float32消息。发送器无训练参数。8＋8是分子配额，当前没有额外slot/type embedding或学习型分子对齐。

real使用逐样本消息；neutral使用**本场景TRAIN消息的逐slot均值**，不使用标签构造，也不携带逐行ID／原长度。neutral仍保留原允许的chemistry／assay上下文和监督训练适配，所以它不是“完全没有分子或任务信息”的系统。

| 条件 | 含义 | 接收位置 |
|---|---|---|
| E_R／early-real | 变化的个体消息训练系统 | 首decoder块前的动态Reader |
| E_N／early-neutral | 固定TRAIN均值消息训练系统 | 同上 |
| M_R／mid-real | 变化的个体消息训练系统 | 第14块后的同构Reader |
| M_N／mid-neutral | 固定TRAIN均值消息训练系统 | 同上 |

两位置Reader均为**464,001个可训练元素**：query LN、1536→128投影、4头MHA、128→1536投影和scalar tanh门控；同seed加载相同实际初态张量。核心均alpha0。early同样是动态残差读取，**不是classic prefix或旧输入MLP**；两者都不增加LM位置。early的query尚未经过decoder跨token上下文化，mid的query已有前14块上下文，后续传播与反向路径也不同，这些共同构成此次“接入深度”的处理。

每对象每forward的源消息载荷固定为**8192逻辑字节**。它是模块边界张量账本，不是PCI／网络实测流量或信息量。同参数、同载荷也不等于同FLOPs、隐藏尺度或可辨识自由度；EOS还使生成累计读取不同。实际padded query、调用、残差写入、wall time与显存另记，未补空调用强求总量相同。[协议与预算说明](protocol_review.md) · [估计对象澄清](causal_estimands_note.md)

### 训练、选点与失败规则

seeds固定为17、42、2026；每case实际3轮、batch4、1548次更新，冻结底座，仅训练Reader。训练答案是两位小数pKD JSON加EOS；按**完整validation答案token NLL的最早最小值**选检查点，不按数值token NLL或test MAE重选，训练仍完成三轮。

正式科学输出统一原V1：batch1、greedy、最多24新token、无KV cache；原parser要求一个有限数值pKD字段，保留原始文本／token、未舍入gold与完整分母。代码围栏、额外说明、重复键等按原规则失败；不去围栏、不裁剪、不手修。任一生成失败时full-sample MAE为null，valid-only指标单列；少于3个有效seed时，mean和sample_SD保持null，不用成功子集充当完整三seed。

三seed SD描述优化随机性的离散，图中误差条为SD而非CI；来源／家族组bootstrap保留为另一层固定检查点不确定性，二者不混合，也不把重复test测量计为新增生物实验。

## 2．核心24项：全部三seed保留

24＝最初已验收V1来源2项＋同版续跑新训练22项；来源保持真实PID/PGID及原收据，不计为再次训练。24项自主生成共**7128／7128严格有效、0失败**。原完整validation选点分布为epoch1 10项、epoch2 8项、epoch3 6项，训练轮数均为3。

### 八配置MAE

MAE单位pKD，越低越好；±为原三seed sample_SD，完整精度见[固定summary](interim_24cases_20261016_2131/matched_summary.json)。

| 场景 | 配置 | seed17 | seed42 | seed2026 | 原mean ± sample_SD |
|---|---|---:|---:|---:|---|
| 新肽／已见靶点 | early／real | 1.717215 | 1.934158 | 2.233327 | 1.961566 ± 0.259145 |
| 新肽／已见靶点 | early／neutral | 1.548877 | 1.522457 | 1.578422 | 1.549919 ± 0.027997 |
| 新肽／已见靶点 | mid／real | 1.863179 | 2.282479 | 2.066346 | 2.070668 ± 0.209683 |
| 新肽／已见靶点 | mid／neutral | 2.073274 | 1.925689 | 1.755758 | 1.918240 ± 0.158889 |
| 新肽／新靶点 | early／real | 1.494397 | 1.732142 | 1.480016 | 1.568852 ± 0.141596 |
| 新肽／新靶点 | early／neutral | 1.426035 | 1.591422 | 1.467434 | 1.494964 ± 0.086062 |
| 新肽／新靶点 | mid／real | 1.563764 | 1.386433 | 1.867917 | 1.606038 ± 0.243510 |
| 新肽／新靶点 | mid／neutral | 1.832650 | 1.497468 | 1.518282 | 1.616133 ± 0.187797 |

### 五个核心contrast的方向与结果

定义`depth_real=E_R−M_R`、`depth_neutral=E_N−M_N`，正值表示该内容条件下mid误差较低；`content_early=E_N−E_R`、`content_mid=M_N−M_R`，正值表示该位置real训练系统误差较低。`DiD=depth_real−depth_neutral=content_mid−content_early`，是已有四格的代数交互，不是独立复验。

| 场景 | contrast | seed17 | seed42 | seed2026 | 原mean ± sample_SD（pKD） |
|---|---|---:|---:|---:|---|
| 新肽／已见靶点 | depth_real | -0.145965 | -0.348321 | +0.166981 | -0.109102 ± 0.259621 |
| 新肽／已见靶点 | depth_neutral | -0.524397 | -0.403232 | -0.177335 | -0.368322 ± 0.176145 |
| 新肽／已见靶点 | content_early | -0.168337 | -0.411701 | -0.654904 | -0.411648 ± 0.243284 |
| 新肽／已见靶点 | content_mid | +0.210095 | -0.356790 | -0.310588 | -0.152428 ± 0.314803 |
| 新肽／已见靶点 | DiD | +0.378432 | +0.054911 | +0.344316 | +0.259220 ± 0.177757 |
| 新肽／新靶点 | depth_real | -0.069366 | +0.345709 | -0.387901 | -0.037186 ± 0.367862 |
| 新肽／新靶点 | depth_neutral | -0.406614 | +0.093954 | -0.050848 | -0.121169 ± 0.257587 |
| 新肽／新靶点 | content_early | -0.068362 | -0.140720 | -0.012582 | -0.073888 ± 0.064247 |
| 新肽／新靶点 | content_mid | +0.268886 | +0.111036 | -0.349635 | +0.010095 ± 0.321378 |
| 新肽／新靶点 | DiD | +0.337248 | +0.251755 | -0.337053 | +0.083983 ± 0.367125 |

两个场景的depth_real均两负一正。early中neutral在两场景全部三seed的MAE都低于real；mid的content效应在已见靶点一正两负、双新两正一负。双新content_mid的三seed均值为＋0.010095±0.321378，第三seed为−0.349635；DiD第三seed也转为−0.337053。已见靶点DiD全正及两个DiD均值为正，都不能单独推出mid的绝对MAE更低。

[双panel配对图PNG](paired-effects/core24/figures/png/fig_01_core_paired_effects.png) · [独立PDF](paired-effects/core24/figures/pdf/fig_01_core_paired_effects.pdf) · [精确XLSX／JSON指针／binary64 hex](paired-effects/core24/process_data/fig_01_data.xlsx) · [三段caption](paired-effects/core24/fig_01_caption_zh.md) · [数据健康与自检](paired-effects/core24/README.md)

### content比较的估计对象

这里的content差是`L(θ_N,TRAIN均值消息)−L(θ_R,个体消息)`：两套Reader分别训练并各自验证选点，包含训练轨迹、选点、输出适配与推理变化的系统总效应。固定θ_R再将消息替换为TRAIN均值的量尚未登记／测量，不能由θ_N的结果补出。[精确定义](causal_estimands_note.md)

### 已有数值参考：当前分数处在什么位置

以下均是同一冻结测试总体的既有完整数值预测，不是本轮新拟合；每场景每算法只有一个原验证选中fit，因此没有三种子SD。它们没有匹配当前参数、训练或语言输出预算，单列用于判断任务效用。

| 数值参考 | 已见靶点152／152 MAE | 双新442／442 MAE | 训练容量 |
|---|---:|---:|---|
| TRAIN标签均值 | 1.619744 | 1.446283 | 一个TRAIN标签均值标量 |
| 52维ridge，原val选alpha1000 | 1.592669 | 1.552233 | 53个名义系数，非有效自由度 |
| 52维kNN，原val选k25 | 1.711140 | 1.592355 | 保存2062／2063条TRAIN向量及标签 |

当前real四组的三种子平均MAE都高于TRAIN均值参考，说明现有生成系统的数值效用仍需改善；这不是同预算架构优劣或显著性结论。数值参考100%有限输出也不等于LLM完成严格JSON生成。旧30轮神经数值模型另有完整三种子结果，其MSE训练、样本曝光和validation MAE选点均与当前不同。[全部数值参考与容量／预算](baseline_context_note.md)

### 门控扩展：零初始化也没有稳定单向收益

原六个mid／real／balanced的alpha3扩展均已完整生成，额外1,782／1,782条严格有效。它们与核心六个alpha0系统复用配对；alpha3仍是tanh门控。下表为预登记`gate_zero_initialization=MAE(alpha3)−MAE(alpha0)`，正值有利于零初始化，单位pKD。

| 场景 | seed17 | seed42 | seed2026 | 三seed mean ± sample_SD |
|---|---:|---:|---:|---|
| 新肽／已见靶点 | +0.477121 | -0.604664 | -0.273417 | -0.133653 ± 0.554270 |
| 新肽／新靶点 | -0.009364 | +0.245128 | -0.338927 | -0.034388 ± 0.292830 |

两场景都存在种子翻转，未支持零初始化在本配置中的稳定优势，也不能反向宣布近单位初始化普遍更优。该扩展同时改变初始残差幅度和gate梯度参数化，不能单独隔离尺度原因；更不能从中挑最优系统替换核心主比较。[30项固定逐配置表与原始精度](interim_30cases_20261016_2257/matched_summary.md)

门控机制、执行和科学解释分别通过[机制核验](gate_extension_mechanism.md)、[执行复核](gate_extension_execution_review.md)及[科学复核](gate_extension_scientific_review.md)。六对中near-unit所选原validation NLL全部更高，test MAE却有四对更低，说明两个端点不能替代；这些是独立训练系统的比较，不证明更换selector一定改善，当前检查点不重选。[精确值与选点解释](gate_selection_task_note.md)

## 3．有效性评价与“不稳定”的解释

### 三类已执行证据各回答一件事

| 评价 | 当前完成范围 | 可支持的判读 |
|---|---|---|
| 自主任务输出：严格JSON有效率、完整pKD MAE | 核心24项，三seed及全分母 | 在冻结条件下系统能输出什么、误差多大 |
| 受控位置／内容差及跨seed变化 | 8配置、10个原contrast完整3/3 | 固定sender和同构Reader下的系统效应与稳定性 |
| 固定检查点C1：branch on/off的teacher-forced答案NLL及格式／数值分解 | 核心24项原C1及独立核验 | 新增分支对gold答案概率的计算作用 |

三类证据互补，不是三次独立生物效力验证。C1按每行完整答案平均后汇总；格式／数值贡献共享该行完整答案分母。核心24项中，format贡献约0.323548—0.330139、numeric约0.189842—0.221231 nats／答案token，neutral也有正format贡献。故G包含通用格式和输出分布适配，不是分子信息量，也不能与自主MAE差互换。

新增固定消息系统的C2尚未运行。旧架构线性摘要探针的结果只适用于旧摘要／投影／函数族；旧区间跨零不能移写为新系统没有信息。严格独立匹配生物donor仍为0，P/H保持undefined；TRAIN均值不满足合法donor条件。条件探针、控制任务和ESM任务读出为后续评价提供原文依据，见下节。

### 事实、相容解释与缺口分开

**事实。** 深度和内容效应随seed翻转，且完整val NLL选点与生成MAE属于不同目标。在各case所选epoch的validation中，early hidden L2/token RMS约1.005、mid约718—719；seed17两个场景的gated δ/h在early约0.24—0.30、mid约0.011—0.013。这些统计包含有效prompt及teacher-forced答案位置，δ在BF16 cast后、相加前统计，未测实际`h_after−h_before`。保存输出还呈现有限数值档、分布位置／分散变化与部分neutral排序信号。

**相容解释。** query的上下文化程度、剩余层／梯度路径、相对残差尺度与有限精度，以及训练／选点和格式先验之间的交互，都可影响当前系统。相近尺度结构出现不同seed的效应方向，现有记录不能唯一确定“mid被淹没”或BF16舍入是原因。C1关闭整个分支，且固定θ均值替换缺失，当前也未分离内容效应中的训练适配与推理依赖。

**需要补的证据。** 后续应单独登记固定θ干预与新消息的条件可读出性，并在合法donor／同条件金标准可用时检验分子匹配。机制定位应针对明确缺失的量，不把低MAE、正G或单个范数比例替代成生物机制结论。[24项机制报告](core24_mechanism.md) · [独立执行／算术核验](core24_execution_review.md)

## 4．九篇原文支持的四组方案

四组方案分别检验初始化、池化配额、相对残差尺度／形状和表示训练顺序。G/P属于原36项；S/W属于未来独立协议，当前没有实现、校准或执行。原文提供机制动机，任务收益仍由固定评价来检验。

| 方案与状态 | 原文依据／本任务可检验组合 | 参数与必须计账的额外成本 |
|---|---|---|
| **G：门控初值，六项已完成** | Flamingo零tanh残差与ReZero零标量动机；原alpha0／alpha3两场景三seed结果均有方向翻转，未确立稳定初始化优势。alpha3仍为近单位可学习门控 | 双方464001；同8192逻辑字节。沿用原更新预算，EOS累计读取、时间／显存照实记 |
| **P：8＋8／global16，未完成** | 固定瓶颈动机来自Flamingo／BLIP-2／ProtT3。原6项中1项因合同时间窗声明不合规而未验收、5项未启动；两个场景正式0／3，不能判断配额效用。无slot/type embedding，配额不等于类型对齐 | sender0、Reader464001；同16×128；原始未验收产物保留，不补算三seed均值 |
| **S：尺度校准／分支LN，未来未执行** | NormFormer分支内LN与零初始化动机。未训练Reader、冻结base、固定TRAIN prompt-only，计算s＝r(h)/r(u)；或先v＝LN_affineFalse(u)再按同规则匹配潜在幅度，仍alpha0。现有含gold的val范数不能当校准输入 | 无affine时仍464001；affine LN则467073。校准prepass、统计／缓存、O(BLH)缩放／LN和反向计账；不能用LN(h＋δ)破坏gate0的base保持性质 |
| **W：Reader任务预热／交错，未来未执行** | BLIP-2／ProtT3阶段化及ProtST对齐动机。固定sender不会因加损失而学习；让TRAIN prompt-only query与消息经过未门控Reader，再由真实数值head监督。两臂仅改变相同T_aux／T_CE的顺序，T、λ及交错表留待未来协议 | head1537，联合465538，推理Reader464001。两臂head均实际接受aux梯度；base prepass、辅助读取／反向、标签曝光和验证预算均匹配。不能把挂空head的CE-only称同预算 |

S会改变gate梯度和有效优化参数化，范数接近本身不是效用证明；W的aux不依赖alpha，不能称预热gate。两者均需另行冻结实现、选择预算和超参数；当前test不用于调这些量。任何未来外部预训练语料还需对象／家族／来源暴露审计，下载语料没有自动变成本轮训练数据。

### 原文与评价依据

页码为本地PDF从1开始的物理页；版本身份与PDF SHA完整保存在[方案JSON](evidence_backed_improvement_plan.json)。

| 原文／官方入口 | 本地原文与页码 | 在本课题中的用途 |
|---|---|---|
| [Flamingo（2022）](https://arxiv.org/abs/2204.14198) | [本地PDF](https://arxiv.org/abs/2204.14198v2)，物理p5、8 | 固定瓶颈与零tanh门控；G/P动机 |
| [BLIP-2（2023）](https://proceedings.mlr.press/v202/li23q.html) | [本地PDF](../../third_party/papers/blip2_d59913d16edf.pdf)，物理p3、4、6、7 | 查询瓶颈及表示→生成阶段；P/W动机 |
| [ProtT3（2024）](https://aclanthology.org/2024.acl-long.324/) | [本地PDF](../../third_party/papers/ProtT3_ACL2024_ffa65fc93979.pdf)，物理p2、4、5、6、14、16 | 蛋白文本两阶段、数值任务边界；P/W动机 |
| [ProtST（2023）](https://proceedings.mlr.press/v202/xu23t.html) | [本地PDF](https://arxiv.org/pdf/2301.12040)，物理p3、4 | 序列—文本对齐及PLM冻结范围边界 |
| [Conditional Probing（2021）](https://aclanthology.org/2021.emnlp-main.122/) | [本地PDF](../../third_party/papers/conditional-probing_0d44d3b5c26a.pdf)，物理p2、3 | 给定B与函数族的额外可读出性评价 |
| [Control Tasks（2019）](https://aclanthology.org/D19-1275/) | [本地PDF](../../third_party/papers/control-probes_557f5d7dc18b.pdf)，物理p1、2、3、5 | 控制任务与探针容量／身份记忆 |
| [ESM-2（本地2022预印本／正式2023）](https://github.com/facebookresearch/esm) | [本地PDF](../../third_party/papers/ESM2_951003574aaf.pdf)，物理p3、4 | token/结构读出与pKD任务能力的区分 |
| [NormFormer（2021 v2审稿稿）](https://arxiv.org/abs/2110.09456v2) | [本地PDF](https://arxiv.org/abs/2110.09456v2)，物理p2、3、4、5、6、7、8 | attention分支内LN、梯度失配及skip缩放负证据 |
| [ReZero（2020 v2预印本）](https://arxiv.org/abs/2003.04887v2) | [本地PDF](https://arxiv.org/abs/2003.04887v2)，物理p1、2、7、8、12 | 零标量残差初始化与局部保持性质 |

NormFormer原文分析梯度失配，其可选skip缩放在1.3B以上存在负结果；它与本方案δ缩放的对象不同。ReZero源码是共享直接零scalar，当前为单个冻结LM外接Reader的tanh门控，不能套用原整网identity／isometry。两官方仓库固定快照与13份静态证据已核，fairseq保留带omission的regular-file partial view；源码存在不构成本任务改善证明。[源码证据](residual_code_evidence.md) · [四方案细节](evidence_backed_improvement_plan.md) · [独立方案审查](improvement_plan_review.md)

后续C2可依据Conditional Probing，把同维表示与固定B并列比较，报告规定函数族下额外可读出性；Control Tasks要求检查容量和身份记忆，ESM-2则提供token／结构读出的边界。回归ΔMSE的单位是pKD²，不是总互信息或论文分类selectivity的直接替代。当前已有数据尚不足以宣布这些组合是全球首次或已提高分数。

## 5．执行身份、资料保全与未闭合项

正式核心数据全部来自原V1 batch1。V2/B4与V3/B2是另存的工程候选，未进入本核心科学summary；TRAIN批量资格不一致及原始输出／合同保持归档，不按版本择优回填。批量资格失败与接入深度效力属于不同问题。[V1续跑审查](implementation_review_v1_continuation.md) · [批量诊断](batch_numerical_diagnosis.md)

核心24项执行审查通过，覆盖原始严格预测、源分数／三seed／contrast算术、实际更新／选点、参数／逻辑载荷、合同／资源及各已完成job退出收据。另经[独立科学解释复核](core24_scientific_review.md)的95项检查，未发现需关闭的解释阻断项。门控新增六项也通过独立执行与82项科学核验。文稿、算术及行政归档未增加forward、拟合或重新选点。

### 池化停止原因与正式分母

第一项global16尝试实际完成3轮／1,548更新和152条严格有效输出，worker exit0，进程组已退出。但其合同于22:57:09.512冻结，声明timeout1360秒；真正启动为22:57:26.489，距原23:20截止只剩1353.511秒，声明时限多出6.489秒。调度器预留的10秒没有覆盖冻结到启动之间约17秒的间隔，这是本轮执行工程错误。标准终态validator据此拒收，队列于23:07停止；原result-record中的成功字样不能覆盖`valid_for_acceptance=false`。

实际运行早于截止结束，也不能据此倒改合同或放宽验收。原汇总器已按原验收规则将此项标为`invalid_or_incomplete`、metrics为null；其余五项保持missing。原152条输出及数值结果完整保留为隔离证据，未混入正式MAE或seed统计。这个失败不是实际运行超时，也不是解析失败。后续工程修复应在真正启动边界重新验证声明时限并增加相应回归检查，本轮没有改控制器、原合同或validator后重试。

[时窗根因与进程终态核验](pool_admission_failure.md) · [未验收尝试的隔离复核](pool_attempt_execution_review.md)；实际controller PID263028、worker PID312725均已退出，工具会话号与系统PID分开记录。

原窗口仍为13:20—23:20；在剩余窗口内不缩短既定三轮训练或全量评估，也不重置时钟追跑。池化仍缺全部六项正式有效结果，不能把一个未验收seed当成结论。

| 终态事项 | 实际状态 |
|---|---|
| 核心24／G六项 | 30项正式验收，两个场景各配置3／3；原2项＋新28项，不重复计复用对照 |
| P六项 | 1项验收失败＋5项未启动；两个场景0／3，原mean／SD为null |
| 旧1398项锁定资产 | 23:10逐文件最终复核全部unchanged，零变更／缺失；[最终收据](old_assets_final_check.json) |
| 合同与控制器 | 原23:20绝对截止未改；控制器已因验收错误退出，UTC守护正常结束且未发信号，末worker收据确认PGID缺席 |
| 工作台接入 | 本报告在发布前冻结；实际接入结果由随后单独的发布回执和工作台FINAL索引记录，避免回写已捕获文稿 |

最终跨文件科学解释与覆盖核验单独保存于[全轮终态审查](final_scope_scientific_review.md)，不以修改本报告回填审查结果。

新增NormFormer／ReZero两篇已完成原PDF、CPU正文、关键词与远程BGE语义检索登记；它们的文献入库与本报告发布是不同状态。[两篇入库收据](../../docs/PUBLICATION_SCOPE.md)

核心快照SHA为`d22da6e98304eb85301a130480bac413106b60511e9c05d92beaeb0d53156e4b`；[最终原汇总JSON](final_snapshot_20261016_231042/matched_summary.json) SHA为`2e72addb16ad461d017a08fc372a1f3696948048c3be97f516403628ed1773c4`，科学协议SHA仍为`9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`。[机器版报告](FINAL_REPORT.json)保存原值、来源及未完成状态。全部36项并未完成，其他基准的金标准缺口也仍保留。
