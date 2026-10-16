# 基准合同审计：先确定每个分数回答什么问题

本次只读检查冻结 JSON、准备/训练/评分源码和实际逐条生成结果；没有训练、模型推理、下载或改动旧文件。三套物理 panel lock 的104条绑定对应76个不同文件，重新计算91,938,172字节 SHA256，零不一致；8个连接器×correct/off加2个BASE03生成管线共18份逐条test输出的ID全集、严格JSON有效数和MAE均用独立标准库代码重新核算通过。机器结果见同目录 `contract_audit.json`，包含每项事实、源行号及本次审查文件SHA。

当前可执行主比较是 **A1宽泛参考靶标pKD预测的两个独立场景**。它尚不是统一多任务基准，也不是原论文整套复现。新多种子和预算匹配实验可以沿用这些冻结数据；其结论必须限定到实际测试的任务、输入、种子和预算。

## 1. 数据和独立单位

A1来自PPIKB研究来源、标准线性肽5–50残基、明确单个KD及可表达化学状态。标签是 `-log10(KD[mol/L])`；仅关系`=`进入精确回归，删失值不冒充真值。数据库Main.xlsx的原Affinity是标签来源，research.xlsx用于来源身份核验，不用其重写数值替代原标签。负pKD未裁剪，不能看到test极值后修改合法范围。

**观测行是真实研究测量的数据库记录，但不等于已核实独立物理实验。** `repeat_measurement_candidate` 是肽/靶标/化学/DOI元数据hash；可能重复的行保留，没有先求平均。`original_experiment='doi:'+DOI`是整篇来源文章超组；同一DOI可含多个实验，不同DOI也未必证明独立来源。`group_id`是为划分和保守重采样构建的来源/序列家族连通组件，并非实验室批次或已确认生物重复。有关名字里的“physical locked”指物理落盘冻结，不证明生物独立性。来源：`prepare_affinity.py:159–187,236–248`，`split_affinity.py:250–277,328–330`。

全部冻结A1行都有assay字符串；**温度、pH、buffer、实际测定构建物序列、实际构建物核实和可比条件组全部为缺失/false**。输入的target是完整数据库参考序列，不保证等于测定的domain、突变体或实际构建物。严格同条件实验pair文件0行。因此目前能够评价宽泛参考靶标pKD预测，不能计算严格同条件肽变体方向、实测效力排序或推导结合机制。来源：`prepare_affinity.py:171–182`；冻结两个samples全集逐项重算。

| 场景/分区 | 行 | 操作group | DOI | 参考靶标 | 肽序列 | 最大group行数 |
|---|---:|---:|---:|---:|---:|---:|
| known train | 2062 | 305 | 450 | 371 | 1177 | 167 |
| known validation | 147 | 63 | 67 | 47 | 119 | 6 |
| known test | 152 | 67 | 69 | 48 | 115 | 6 |
| double-new train | 2063 | 152 | 536 | 342 | 1210 | 724 |
| double-new validation | 431 | 123 | 164 | 149 | 323 | 14 |
| double-new test | 442 | 125 | 178 | 157 | 325 | 14 |

这些group适合作为声明的相关性阻断和bootstrap单位；独立物理实验数当前未知。double-new train的一个组件占724/2063，不能把2063行看作2063个独立实验。

## 2. 划分含义、外部隔离和两个场景的可比性

肽家族是 `1-Levenshtein/max_length >=0.8` 的连通组件；靶标家族是MMseqs2比对identity≥30%、双向coverage≥80%的连通组件。它们是操作同源阈值，不涵盖所有远程同源。来源DOI整组隔离；double-new还把target家族加入连通图；known的val/test精确参考target必须已在train，否则隔离，不重新分配。按group大小和固定seed贪心分配，不用KD挑选样本。来源：`a1_locked/protocol.json:3–33`；`split_affinity.py:250–299`。

使用 `a1_locked_v2`；旧 `a1_locked`保留历史用途。v2基于完整329行PEPBI workbook、84条肽、58个实际靶标、22个ITC来源DOI，按原来源/实际肽和target家族保守排除；保留原分区分配，不按分数重选。Kliche/PepInter来源及同源匹配也参与隔离。但PepInter原DOI/化学来源仍不完整，两个lock均为`frozen_restricted_scope`、`external_isolation_verified=false`；不能转称所有公共库零曝光。来源：`a1_locked_v2/coverage_protocol.json`与两manifest的limitations。基础模型预训练曝光与项目监督训练划分是另一问题。

**两个scenario不是同一测试人群上的单因素难度实验。** 共同sample共2361条，其中known train→double test有262条，known test→double train有116条；共同test仅22条。完整交叉表在JSON。由此：每个scenario必须独立初始化、独立fit和独立validation选点；不可合并两个scenario训练、跨场景接续或蒸馏，也不能用known训练的checkpoint宣称double零曝光。场景内相同test逐条配对比较有效；不同场景MAE绝对大小不可直接解释为“新靶标使问题更容易/更难”。

冻结samples SHA256：known `d1d9ce44db4aeefcc829f5ce7b50b3e288ad03a28af9d15bb9f1fd550a6b657e`；double `7ff5a67203efaa94db29fad1f7d2774b5f5e5f8a9155fc47a090e0f2a3baee71`。test_gold和lock独立SHA详见JSON。

## 3. 每个面板究竟测什么、已做到哪一步

| 面板 | 要回答的问题与标签 | 实际已跑 | 尚缺与结论边界 |
|---|---|---|---|
| A1 | 真实KD转pKD，两个肽/靶标泛化场景 | 原预测器2架构×2场景×3种子12训练；6确定性拟合；4新生成接口×2场景×seed17共8训练及BASE03两场景推理 | 新接口3epoch单种子；严格同条件pair0；不证明真实构建物效力或统一多任务 |
| A2原跨条件 | PXD005573源→PXD064311目标条件，固定target test和32/128/512/2048总标注预算 | 没有完成原跨条件模型训练/评价 | 原源113215数值已证实iRT，不能当分钟；98192行D/E化学身份有损，2026精确源快照未核实；原HOLD保留 |
| A2目标域诊断 | 单一March基本pH9条件，真实RT分钟 | 全7070行；train4952、val1055、test1063，val再分selection525/calibration530；constant/ridge/kNN实拟合 | exact base sequence阻断，全PTM变体跟随分区；未母体/近同源阻断；不替代原跨条件、不是DeepLC复现 |
| B1 | 母体局部功能区序列→31候选去显式规则描述的已知阳性排名，未见类别 | 原BASE02/MEANALIGN各3种子，共6训练和重放；训练378注释合并348序列查询，val134/test76；22/7/2类 | 两个测试类仅MOD_N-GLC_1和MOD_NMyristoyl；没有新生成连接器B1结果；不能推为游离肽活性或广泛任意功能泛化 |
| B2 | 实验分子事实与明确允许背景的组合QA，joint correctness/必需事实覆盖/矛盾/无据推断 | 4277条证据登记；无正式QA评分 | reviewed实验答案金标准0；证据登记表不是QA gold，synthetic逻辑题不能替代 |
| C1 | correct/off、匹配错配、独立训练常数、swap的计算干预 | 8个checkpoint全test correct/off生成与teacher-forced NLL；独立constant训练 | 严格same-target/长度/观察context/不同group donor两场景均0；P/H与swap未成立；不放宽匹配补数 |
| C2 | raw/message/receiver在基准控制之外能否读出pKD；局部证据是否可靠 | 8job prompt-only表示、128维固定投影、ridge读出、groupbootstrap | 未随机标签control、MLP容量扫描、slot删除曲线、结构接触/功能扰动证据；不是总互信息估计、不是生物因果 |
| C3 | 解释中原子事实支持率、必需事实覆盖、矛盾/无据率 | 未正式执行 | 无双人独立、盲模型、争议裁决的原子事实gold；不能把文本流畅性或语言模型自评当实验真值 |

B1补充核验已完成声明的Benz/Kliche外部来源：1654/1654母体、2813有效context全部解析，8条过宽历史关联拒绝；新增比对无命中，旧冻结数据不变。不能再用旧manifest初始“7母体未解析”描述当前资料状态；也不能擅改旧restricted lock标签。来源：`b1_external_followup/coverage_supplement.json:4–52`。

## 4. 允许输入、输出通道和评分分母

预测器和新连接器都接收完整肽/完整参考target的冻结ESM2-150M第30层残基表示及相同信息allowlist：chemical_state、assay/buffer、温度/pH、缺失标记、actual_construct_verified和target_sequence_role。预测器用train-only数值标准化和固定trigram哈希；LM用可读原字段。ID/DOI/名字/原始元数据/gold不入prompt；LM不直接接收原序列文字。长序列以≤1022非重叠chunk编码，覆盖全部残基，但chunk间原PLM上下文断开。来源：`train_panel_baselines.py:40–81`、`run_connectors.py:55–85,444–460`；不得把“同allowed information”写成“相同表示”。

| 通道/指标 | 定义与分母 | 当前处理 |
|---|---|---|
| 全样本pKD MAE | 所有锁定test行的绝对误差均值，越低越好 | 任一生成失败则full_sample_mae=null，失败数单报；不补值、不clip、不修答案 |
| valid-only MAE | 仅严格JSON且唯一字段pKD、finite数字、无bool/重复键的子集 | 明确条件分母，不能和全样本MAE直接排序 |
| target-macro MAE | 先对每条精确参考target求MAE，再等权平均 | 当前`group_macro_mae`实际按target SHA分组；known48、double157，不是group_id宏平均 |
| within-target-condition Spearman | 相同靶标及已核实同条件的排序 | 当前可比组0，undefined；总体Spearman不是这个指标 |
| pair direction | 经审定同条件实验pair方向 | 当前pair0，undefined；不能用任意两行KD比较替代 |
| B1 class-macro MRR/R@1/R@3 | 每个真实类别内先平均其已知阳性rank指标，再对test类平均 | 31候选整体排名，test2类；sample MRR用首个已知阳性，不能替换classmacro；未注释为未知 |
| C1 G | 每个样本mean answer-token NLL(off)−NLL(correct)，再按行平均 | nats/answer-token；teacher-forced gold只用于C1，不能叫自主生成MAE或bits |
| C2 delta MSE | B-only test MSE减B+表示的test MSE | pKD²；200维B+128投影，TRAIN fit、valMSE选alpha，group_id percentile bootstrap |

当前correct四模型均全152/442可解析；off仅27/152、70/442；BASE03生成91/152、336/442有效。BASE03的源数值预测器全样本MAE分别1.468699、1.421086；生成通道valid-only分别5.922365、4.638924，全样本生成MAE均无定义。渲染器失败必须原样保留；不能用源预测器数值冒充LLM已正确输出，也不能把击败渲染失败当击败强预测器。BASE03是现有预测→同Qwen零样本复制数值，不是B2事实推理，也没有本轮训练/OOF语言训练。

## 5. 当前“内部没有稳定优势”的精确含义

| 已训练接口 | 可训练参数 | 交流形式 | known MAE | double MAE |
|---|---:|---|---:|---:|
| learned constant | 24,576 | 固定16个LM空间prefix token | 1.965609 | 1.767973 |
| input MLP | 282,752 | 拼接肽+靶标残基先整体顺序池化为16slot，再MLP映射prefix | 1.996238 | 1.905624 |
| learned-query cross-attention | 350,848 | 16个学习query读全残基，产生16prefix | 1.442674 | 1.781755 |
| internal cross-attention | 548,608 | 第14decoder block后每个文本位置查询全部残基，残差相加 | 2.161333 | 1.501369 |

这些是同数据、backbone、allowlist、seed17、3epoch、batch4、24最大生成token的具体实现比较，**参数/通信预算/FLOPs未匹配**。internal没有显式gate或零初始化门控；`configs/bench_v02.json`的BASE-06名字“gated”与实际源码不符，未来应正确标识历史实现，不能追溯改名掩盖。source：`connectors.py:43–64,133–177`。input MLP整体池化可能让长target稀释肽信息，但这是待检验假说，当前分数不证明原因。internal读全N残基、其他输出16slot，不是纯插入位置消融。

新接口每个训练1548个batch4步骤，3轮都以validation答案token NLL选点，实际8模型都完成3轮训练，但验证选点不同：known四模型均选第3轮；double的MLP/constant选第3轮，qformer_style/internal_cross_attention选第2轮。原预测器30轮、有效batch64、990步骤，用连续pKD MSE训练，validation MAE选点，3种子；初始化/样本顺序/文本序列化目标不同。更多小batch步骤不代表同总样本或同算力。两类旧结果不可写成同预算试验；多种子seed变化也不等于重新抽样独立实验。

C1全部G>0，但constant也约0.55 nats/answer-token：说明训练过的接口可改变输出分布/格式，不能单靠correct-off证明用了分子特异信息。C2六项非constant message增益的95%区间均跨零，仅说明此投影/摘要/线性probe未提供稳定增量；它不证明所有内部信息不存在。prefix取16输出均值，internal取末prompt位置残差；这种摘要差异也要与插入位置分开。未完成随机标签、容量、局部删除/生物证据控制前，不升级成机制结论。

## 6. 原作者方法与本项目控制方法的区别

当前Qformer_style只有一层小cross-attention与学习query，未执行ProtT3完整两阶段对齐/生成训练，未执行其stage1匹配/语言目标和stage2 mid-LoRA协议；官方README示例8query而本地16。来源：`repos/ProtT3/README.md:33–56`、`model/blip2_stage1.py:20,199–200`、`model/blip2_stage2.py:68,304–315`。只能称学习query输入连接器或ProtT3-inspired控制，不能称ProtT3复现。

B1本地ESM/Qwen描述对齐器不是原ProtNote SwissProt/GO/EC完整系统；ProtNote数据/权重落盘不等于执行原作者测试。OPUS/PepInter/DeepLC同理，当前foundation index对原系统另列未执行/准备状态。没有把别人的原论文分数和当前A1的不同数据/通道强行并排胜负。

## 7. 下一轮运行前应固定的合同，保持旧结果

1. 固定上述两scenario各自samples、gold、lock SHA，禁止跨scenario模型/标签混用；所有新checkpoint独立新建目录，历史失败/恢复和生成失败都保留。
2. 同scenario内预登记方法×种子完整矩阵、固定训练样本暴露/更新与选择规则、24token严格JSON分母、主要配对比较与方向；不能先看test再改NLL/MAE、budget或阈值。多种子报告全部seed和均值/样本SD，不报“最好seed”。
3. 参数预算必须统计实际参与计算和更新的connector参数；通信预算要写清source长度、压缩slot数/维度、读写位置次数、receiver查询位置，以及生成每token是否重算。虚加未使用参数不能算公平；固定slot也不自动等于FLOPs相同。将不能同时匹配的轴分为预先声明的比较，而不是用一个“同等预算”总称。
4. 保留数值预测器强基线、trained constant和OFF；不同输出有效率单报。相同允许输入和样本暴露之外，验证选点准则差异需消除或登记成独立因素；一旦test打开就不重选规则。
5. 源/家族group bootstrap只给固定数据与checkpoint的描述性区间；种子变异另报。多个方法/种子/指标探索不可挑单个显著区间作普遍优势。模型改进由已见test负结果触发，属于同冻结测试集上的稳健性与探索验证；真正未观察确认需要新增独立外部集，不能替换旧test或重分其困难样本。
6. A2/B2/C3缺失应保留为缺失；新A1实跑不能使它们自动“通过”。本轮先测清A1架构/预算的可重复效应，结论仅覆盖A1。

## 8. 问题清单与审查范围

以下P1/P2表示新科学比较的前置阻断项或解释边界，**不表示已验证运行存在代码故障，也不否定已通过的执行与完整性验收**。

P1（影响新比较主张成立）：场景跨分区重合禁止混合训练；真实实验单位/同条件gold未核实；历史参数/通信未匹配且internal无gate；失败分母/通道不可混用；A1组件不等于全部多任务基准完成。

P2（影响统计解释或对照公平）：训练/选点目标与预算不同；C1无合法donor且常数同样改变NLL；C2有限probe和不完全同构摘要；已见test需承认探索属性；B1只有两个heldout类。每条机器记录含精确来源行号，不依赖本段缩写。

实际审查范围：34份关键合同/源码/汇总文件读取与SHA，3套冻结锁全部76个独特pin重新验证，18份逐条test输出重新解析与重算。没有重新forward18个旧checkpoint，没有重新跑全部unit tests，没有对原论文实验身份做人工裁决，没有进行新的外部文献搜索或完整PDF复读；原系统状态基于当前本地源码/登记证据。此报告不改旧评价标准，也不引入新得分。


### 补核：训练完成轮数与验证选点分开记录

前版“8模型best epoch均3”已更正。逐一读取8份实际result、8份完整history，并用zipfile/pickletools静态检查8个真实checkpoint的best_epoch标量（不反序列化、不加载模型、不推理）。8份checkpoint SHA均与实际result一致；各history中最早最低validation答案token NLL与result及checkpoint metadata一致。全部训练完成3轮/1548次更新，不等于最终采用第3轮checkpoint。

| 场景 | 方法 | 完成轮数 | 验证选中轮次 | 实际result源（含行号） | checkpoint metadata源 |
|---|---|---:|---:|---|---|
| known | constant | 3 | 3 | `results/foundation_20261016/gpu_suite/train_constant_seed17/artifacts/result.json:3` | `results/foundation_20261016/gpu_suite/train_constant_seed17/artifacts/best_connector.pt` |
| known | input_mlp | 3 | 3 | `results/foundation_20261016/gpu_suite/train_input_mlp_seed17/artifacts/result.json:3` | `results/foundation_20261016/gpu_suite/train_input_mlp_seed17/artifacts/best_connector.pt` |
| known | internal_cross_attention | 3 | 3 | `results/foundation_20261016/gpu_suite/train_internal_cross_attention_seed17/artifacts/result.json:3` | `results/foundation_20261016/gpu_suite/train_internal_cross_attention_seed17/artifacts/best_connector.pt` |
| known | qformer_style | 3 | 3 | `results/foundation_20261016/gpu_suite/train_qformer_style_seed17/artifacts/result.json:3` | `results/foundation_20261016/gpu_suite/train_qformer_style_seed17/artifacts/best_connector.pt` |
| double-new | constant | 3 | 3 | `results/foundation_20261016/gpu_suite_double_new_recovery1/recovery1_double_new_train_constant_seed17/artifacts/result.json:3` | `results/foundation_20261016/gpu_suite_double_new_recovery1/recovery1_double_new_train_constant_seed17/artifacts/best_connector.pt` |
| double-new | input_mlp | 3 | 3 | `results/foundation_20261016/gpu_suite_double_new_recovery1/recovery1_double_new_train_input_mlp_seed17/artifacts/result.json:3` | `results/foundation_20261016/gpu_suite_double_new_recovery1/recovery1_double_new_train_input_mlp_seed17/artifacts/best_connector.pt` |
| double-new | internal_cross_attention | 3 | 2 | `results/foundation_20261016/gpu_suite_double_new_recovery1/recovery1_double_new_train_internal_cross_attention_seed17/artifacts/result.json:3` | `results/foundation_20261016/gpu_suite_double_new_recovery1/recovery1_double_new_train_internal_cross_attention_seed17/artifacts/best_connector.pt` |
| double-new | qformer_style | 3 | 2 | `results/foundation_20261016/gpu_suite_double_new_recovery1/recovery1_double_new_train_qformer_style_seed17/artifacts/result.json:3` | `results/foundation_20261016/gpu_suite_double_new_recovery1/recovery1_double_new_train_qformer_style_seed17/artifacts/best_connector.pt` |

各result/history/checkpoint完整SHA及选中NLL记录在JSON的`training_checkpoint_selection_audit`，本次仅修审计报告，未改训练产物、历史分数、评价标准或冻结数据。
