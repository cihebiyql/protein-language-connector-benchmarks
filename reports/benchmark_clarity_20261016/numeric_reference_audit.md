# 已有数值基线补核：常数、弱信息与语言接口参考

本次只读原v0.2控制和成功replay产物：校验保存预测的完整ID顺序、原pKD、样本/金标准/锁指纹及保存state；用保存预测重新算误差。**没有新fit、没有模型forward或GPU调用，也未改当前36项、阈值、prompt或评价规则。** 逐文件SHA与源码行号见同名JSON。

## 1. 现成的六个控制究竟是什么

六个是3种算法×2个冻结场景，不是六种算法，也不是六个随机种子。原始文件在 `results/bench_v02/panel_baselines_20261015_v2/A1_known_BASE00` 和 `A1_double_new_BASE00`；既有CPU重算在 `results/foundation_20261016/replay_predictors/BASE00_<scenario>`。本次不再次调用它们的fit函数。

| 数值控制 | known test MAE，152/152 | double-new test MAE，442/442 | 保存模型与选择 |
|---|---:|---:|---|
| TRAIN mean | 1.6197442045 | 1.4462827692 | 1个TRAIN标签均值；无超参数竞争 |
| composition/chemistry ridge | 1.5926691087 | 1.5522326389 | 52维输入+截距，共53个保存系数；val MAE选alpha=1000 |
| composition/chemistry kNN | 1.7111403463 | 1.5923554868 | 同52维标准化空间；val MAE选k=25，保存TRAIN向量与标签 |

全部为完整有限数值输出，覆盖率100%。这是predictor通道的有限值覆盖，不是语言模型100%严格JSON生成。真实单位为pKD的绝对误差。

TRAIN mean保存标量分别为 **5.530977562607410** 和 **5.600067544896424**，对各自所有对象输出同一值。源码以TRAIN均值定义，没有先看test挑常数；本次只读该保存标量并确认原预测全部相等，没有补拟合median。

原控制和已有replay的完整test预测最大差≤2.14e-14，选中alpha/k相同。double kNN在未选中的k=3/5验证试验存在已记录数值差，原记录保持；选中的k=25和完整test预测一致。不能把既有replay重算再算成新增独立模型或新随机种子。

## 2. 这些控制读了多少信息、如何拟合

`run_simple_baselines.py:18–33`的52维依次是：肽20种氨基酸组成+log(1+长度)，完整参考target的20种组成+log(1+长度)，N末端free/acetyl/unknown三项，C末端free/amide/unknown三项，以及温度/pH/buffer/实际构建物的4个缺失指标。没有序列顺序、ESM向量、显式target ID、来源DOI、test标签或assay字符串。

共同center/scale只用TRAIN特征计算并保存（各52数值，std下限1e-6）；当前两场景恰有6个零方差维度：两种末端unknown及4个全缺失条件。TRAIN mean虽然同文件也保存这些公共统计，但其预测不读取任何输入特征。

- Ridge优化平方误差加L2；52个特征系数加不罚截距，**名义53系数不等于统计自由度**。alpha候选固定0.1/1/10/100/1000；每个仅TRAIN拟合，最低validation MAE选最早项，test不refit。
- kNN没有53个线性系数，却不是没有容量：它保留2062/2063个TRAIN标签及对应52维向量（107224/107276个特征数值）。距离为标准化特征的平方欧氏距离，按`1/(distance+1e-6)`加权，距离tie用冻结TRAIN行顺序。候选k固定1/3/5/10/25，按validation MAE选25。它不是编辑距离序列kNN，也没有训练motif库。
- 每场景TRAIN mean只有一个候选。其验证MAE是known1.3847400768、double1.6110667885；ridge选中验证MAE为1.3803197420/1.6890483427；kNN为1.3087771278/1.7204138705。验证上的最好并未被test结果重新选择。

## 3. 已有或缺失，不补算

| 候选参考 | 冻结v0.2六控制与成功replay内的状态 |
|---|---|
| 全局TRAIN mean | 已有，上表可直接核对 |
| TRAIN median | 缺失；本次不从标签补算median或生成新分数 |
| 每个精确target的TRAIN mean/median | 缺失；double遇到未见target还需要预先定义fallback，本次不增加规则 |
| target-sequence-only预测器 | 缺失；现ridge/kNN同时读肽和target组成，不能冒称target-only |
| context-only或assay-only预测器 | 缺失；现BASE00并未使用assay字符串，不能冒称已做context-only消融 |
| peptide-only预测器 | 缺失 |
| 编辑距离sequence-kNN或TRAIN-only motif模型 | 这组实际实现中缺失；配置名字不能替代真实算法 |

缺失范围限定为本次核查的冻结A1控制集合，不声称整个磁盘不存在其他旧探索模型。没有将旧v0.1不同样本或未验收尝试混入该表。

## 4. 同一test，但宏平均名字要分清

两个BASE00原`results.json`的`group_macro_mae`按**target_family**分组（known42组，double140组），并不是当前神经模型用的精确参考target宏平均。原`comparison.json`已另存 `BASE00_recomputed_exact_target_groups`（known48、double157组）；本次从保存预测核回该既有重汇总，数值如下。没有改任何分组或主指标。

| 控制 | 已有exact-target macro MAE：known | double |
|---|---:|---:|
| TRAIN mean | 1.5323092888 | 1.6116596037 |
| ridge | 1.4608874584 | 1.7383988468 |
| kNN | 1.5684932966 | 1.7934876470 |

全部基线的samples SHA分别为known `d1d9ce44db4aeefcc829f5ce7b50b3e288ad03a28af9d15bb9f1fd550a6b657e`、double `7ff5a67203efaa94db29fad1f7d2774b5f5e5f8a9155fc47a090e0f2a3baee71`；原结果receipt、成功replay和当前锁完全一致。当前samples里的test ID顺序及原pKD与test_gold逐项一致。

## 5. 已有神经数值预测器是另一预算与通道

| 架构 | 参数 | known 3seed MAE均值±样本SD | double 3seed MAE均值±样本SD |
|---|---:|---:|---:|
| BASE01残基注意力预测器 | 374081 | 1.460230 ± 0.016829 | 1.457482 ± 0.071107 |
| MEANPOOL预测器 | 324673 | 1.469941 ± 0.014293 | 1.471969 ± 0.087681 |

它们使用完整冻结ESM残基及允许context，**30轮、有效batch64/微批8、990次更新，TRAIN标准化pKD的MSE训练、validation pKD MAE选点**。当前matched生成器为3轮/batch4/1548更新、答案文本CE训练、full-answer token NLL选点；训练样本暴露、损失、选点、参数和输出接口都不同。更新次数更多不等于总样本/训练预算相同。不得写成“同训练预算优于旧数值预测器”。

旧seed17 MEANPOOL数值MAE约known1.468699、double1.421085，确实是同冻test上的已算参考；但double另外两个种子约1.421607和1.573214，不能只挑seed17宣称稳定水平。全部seed原数值和来源保存在JSON。BASE03把这些数值交给LLM渲染又是另一通道，不能拿源数值冒充LLM输出。

## 6. neutral不是无信息常数：实际prompt输入补查

旧`context_view`与`prompt_text`原字节保留。允许输入为化学状态、温度/pH、缺失标记、actual_construct_verified、assay/buffer和target_sequence_role；原序列的可变信息走消息接口，不直接写成序列文字。实际冻结数据中：

- 化学字典固定7个键：topology、n_terminal、c_terminal、residue_modification、chemical_modification、cyclization_method、explicitly_representable。有效变化主要是N末端Free/Acetyl与C末端Free/Amide（含原大小写）。其他化学标志在这些样本恒定。
- 两场景TRAIN各有80种原大小写assay字符串；known validation/test为29/19种，double为41/49种。`not reported`等是原字符串，不能把“非空字段”当已知真实实验条件；逐split sentinel数量见JSON。
- 温度、pH、buffer、实际构建物、verified在两场景全部缺失/false。缺失相同不证明生物条件相同。
- 源码中允许context函数没有读取pkd/KD、sample ID、DOI、target ID或raw_metadata。实际允许值里逐行检查sample ID、来源DOI及≥6字符target UniProt ID，命中0。没有发现直接gold或显式target ID进入输入prompt的证据；未据此声称排除了所有潜在相关性、公共预训练曝光或所有数据偏差。

中性消息固定，只去掉变化的消息内容；**原允许prompt仍有化学末端和assay信息**，接收器仍可随文本query变化、监督训练仍可学习输出格式与标签分布。因此TRAIN mean数值预测器、旧24,576参数常数软token LLM、当前464,001参数neutral-message reader是三种不同参考，不能互换名字。没有context-only拟合，当前不能定量归因assay或化学贡献；不为解释已有分数改prompt白名单。

## 7. 如何使用这些参考

double单一TRAIN常数已经达到MAE1.446283，表明本冻结测试人群存在较强分布基线。ridge/kNN加入这些弱特征没有超过它；这一事实约束“预测准确就证明用了丰富分子信息”的说法。它不是证明任务无价值，也不是内部连接器内容收益的直接检验。

当前核心因果比较仍是既定24项匹配位置×内容对照，门控和池化扩展构成预登记36项；应等待全部规定seed并报告固定同seed差值。这里的旧数值参考只帮助理解任务难度、监督分布和语言接口代价，不替换matched controls、不扩增成功trial数、不调整阈值/评价，也不把单seed相近误差当等效性证明。
