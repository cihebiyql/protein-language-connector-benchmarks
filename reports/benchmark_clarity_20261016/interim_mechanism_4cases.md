# 首四格的只读机制核验：已见靶点、种子17

当前四格呈现“mid内容收益相对较好，但mid绝对误差仍较高”的组合。全部152条答案均有效，四个检查点都由原完整验证答案NLL选择epoch1；三轮、1,548次更新实际完成。三种子均值和标准差继续为null。本报告仅描述固定首四格，未调指标、阈值、检查点或超参数。

[汇总来源](../../docs/PUBLICATION_SCOPE.md)后续会更新；[本次提取JSON](interim_mechanism_4cases.json)固定读取时SHA，并绑定实际result、history、run、选择文件和C1记录。前两格来自原suite_20261016_1426，后两格来自remaining_v1_20261016_1546。已核36项结果／历史产物指纹、70项记录在实际运行中的Python源指纹，608条原始C1记录重算一致。

## 原任务与固定contrast

| 条件 | 有效／总数 | 原pKD MAE | 选中epoch | 原验证full NLL |
|---|---:|---:|---:|---:|
| early / real | 152/152 | 1.717215 | 1 | 0.672149 |
| early / neutral | 152/152 | 1.548877 | 1 | 0.665223 |
| mid / real | 152/152 | 1.863179 | 1 | 0.657166 |
| mid / neutral | 152/152 | 2.073274 | 1 | 0.655648 |

| 已冻结contrast | seed17差值（pKD） | 三种子均值／SD |
|---|---:|---|
| depth_real | -0.145965 | null / null |
| depth_neutral | -0.524397 | null / null |
| content_early | -0.168337 | null / null |
| content_mid | 0.210095 | null / null |
| depth_content_difference_in_differences | 0.378432 | null / null |
| gate_zero_initialization | 未定义 | null / null |
| balanced_pool_allocation | 未定义 | null / null |

正差值沿用协议，有利于所列处理。depth-real为−0.145965，mid-real误差高于early-real；depth-neutral为−0.524397，mid-neutral劣势更大。因此DiD为＋0.378432，表示mid的相对内容收益更强，同时没有mid绝对优势。early内容差为−0.168337，mid内容差为＋0.210095。DiD也等于两项content差之差，这些contrast共享运行，不能当独立复验。原分组区间完整保存在JSON，属于固定检查点的测试组抽样；本报告不据单种子或区间符号宣称稳定优势。

neutral只去掉随样本变化的残基消息，允许提示词中的化学／实验上下文仍在；中性分支可利用这些上下文并提供通用校准。early的query尚未经过decoder上下文化，mid已经上下文化；这是depth处理本身的一部分。balanced8＋8仅改变池化配额，当前没有slot／type embedding，不能称类型对齐。

## 三个epoch的原训练／验证NLL

单位为nats／答案token。训练full值在更新中的批次上累计；验证full值在固定epoch末权重上计算，只有后者用于原选点。类别列给出验证loss_sum／类别count；类别均值为两者相除，各类别均值不能相加。验证总数均为1,471，其中format882、numeric589、mixed0。全epoch训练类别loss/count没有保存，JSON明确列null，未额外forward补造。

| 条件 | epoch | train full NLL | validation full NLL | format loss／count | numeric loss／count | mixed loss／count |
|---|---:|---:|---:|---:|---:|---:|
| early / real | 1＊ | 0.708747 | 0.672149 | 4.974799/882 | 983.756320/589 | 0/0 |
| early / real | 2 | 0.674537 | 0.682870 | 1.975084/882 | 1002.526889/589 | 0/0 |
| early / real | 3 | 0.668120 | 0.672744 | 2.564040/882 | 987.042760/589 | 0/0 |
| early / neutral | 1＊ | 0.707192 | 0.665223 | 3.793250/882 | 974.750331/589 | 0/0 |
| early / neutral | 2 | 0.678081 | 0.683692 | 1.718296/882 | 1003.992128/589 | 0/0 |
| early / neutral | 3 | 0.678466 | 0.672387 | 2.014029/882 | 987.066792/589 | 0/0 |
| mid / real | 1＊ | 0.705530 | 0.657166 | 0.293214/882 | 966.398132/589 | 0/0 |
| mid / real | 2 | 0.656483 | 0.681731 | 0.372386/882 | 1002.453854/589 | 0/0 |
| mid / real | 3 | 0.646115 | 0.667099 | 0.251004/882 | 981.052175/589 | 0/0 |
| mid / neutral | 1＊ | 0.706251 | 0.655648 | 0.255355/882 | 964.202910/589 | 0/0 |
| mid / neutral | 2 | 0.665331 | 0.657888 | 0.212036/882 | 967.541058/589 | 0/0 |
| mid / neutral | 3 | 0.662854 | 0.656137 | 0.302950/882 | 964.874795/589 | 0/0 |

＊为实际选中epoch。后续训练的train NLL整体下降，验证最小值四格都留在epoch1；这揭示训练拟合与有限验证泛化的分离，不构成按测试MAE重选epoch的依据。

## C1：原逐样本平均与token池化分开

原G先计算每个样本完整gold答案的mean-token NLL差，再平均152条；逐条原始C1重算一致。类别贡献同样用该样本完整答案token数作共同分母，再平均样本。150条答案含10个token，2条含11个token，因此token池化与逐样本平均权重不同。

`G_c = mean_i[(loss_off,c(i) − loss_correct,c(i))/n_full(i)]`，单位nats／完整答案token；类别自身NLL为`loss_sum/count_c`，不是相加的G贡献。

| 条件 | 原C1 G | format贡献 | numeric贡献 | mixed贡献 | 原G−类别和 |
|---|---:|---:|---:|---:|---:|
| early / real | 0.52216751 | 0.32760472 | 0.19455005 | 0 | 1.275e-05 |
| early / neutral | 0.53230798 | 0.32749307 | 0.20480017 | 0 | 1.475e-05 |
| mid / real | 0.54386522 | 0.32977817 | 0.21409814 | 0 | -1.110e-05 |
| mid / neutral | 0.54248075 | 0.32979342 | 0.21269795 | 0 | -1.062e-05 |

原C1使用原transpose交叉熵路径，类别诊断使用flatten路径；字段名reduction_rounding_residual保留约±1e−5的数值路径差，未将差额分摊到某一类别，也未覆盖原G。

| 条件 | correct原逐样本NLL | correct诊断全token池化NLL | off原逐样本NLL | off诊断全token池化NLL |
|---|---:|---:|---:|---:|
| early / real | 0.67768219 | 0.67814761 | 1.19984970 | 1.19976235 |
| early / neutral | 0.66754172 | 0.66774933 | 1.19984970 | 1.19976235 |
| mid / real | 0.65598448 | 0.65633844 | 1.19984970 | 1.19976235 |
| mid / neutral | 0.65736895 | 0.65776950 | 1.19984970 | 1.19976235 |

下表是152条的类别loss总和／类别token数，并附该类别自身NLL，属于token池化账本。mixed在这批gold中count0，类别均值未定义，共同完整答案分母贡献为0。

| 条件／模式 | format loss／count；类NLL | numeric loss／count；类NLL | mixed | full loss／count |
|---|---|---|---|---|
| early / real / correct | 3.528045/912；0.003868 | 1028.612612/610；1.686250 | 0/0；未定义 | 1032.140657/1522 |
| early / neutral / correct | 3.693692/912；0.004050 | 1012.620788/610；1.660034 | 0/0；未定义 | 1016.314481/1522 |
| mid / real / correct | 0.216783/912；0.000238 | 998.730319/610；1.637263 | 0/0；未定义 | 998.947102/1522 |
| mid / neutral / correct | 0.193837/912；0.000213 | 1000.931340/610；1.640871 | 0/0；未定义 | 1001.125178/1522 |
| 四格相同off | 502.099885/912；0.550548 | 1323.938407/610；2.170391 | 0/0；未定义 | 1826.038292/1522 |

四格format贡献均约0.3275—0.3298，包含neutral，说明通用格式／终止作用明显；numeric同样有改善。real相对neutral的额外numeric贡献在early为−0.010250，mid为＋0.001400，远小于共同格式改善量级。mid的C1整体比early更好，生成MAE却较差，直接展示teacher forcing与自主生成不等价。C1也没有给出生物实验因果结论；严格donor仍0，P/H未定义。

## gate与相对残差：位置固有隐藏尺度

下表为epoch末固定权重的validation诊断；hidden／raw／gated为全部active提示词与答案位置的每token向量L2范数RMS，ratio为平方和比值开方。完整train与validation诊断均在JSON。epoch末gate不能直接乘时间累计的训练raw RMS来重建训练gated RMS。

| 条件 | epoch | tanh(alpha) | hidden RMS | raw delta RMS | gated delta RMS | raw／hidden | gated／hidden |
|---|---:|---:|---:|---:|---:|---:|---:|
| early / real | 1＊ | 0.006421 | 1.005459 | 37.910121 | 0.243430 | 37.704297 | 0.242108 |
| early / real | 2 | 0.009252 | 1.005459 | 31.858869 | 0.294767 | 31.685898 | 0.293167 |
| early / real | 3 | 0.008764 | 1.005459 | 29.359258 | 0.257290 | 29.199859 | 0.255893 |
| early / neutral | 1＊ | 0.005827 | 1.005459 | 45.626815 | 0.265858 | 45.379094 | 0.264415 |
| early / neutral | 2 | 0.006620 | 1.005459 | 47.284724 | 0.313026 | 47.028002 | 0.311327 |
| early / neutral | 3 | 0.005560 | 1.005459 | 45.706936 | 0.254131 | 45.458781 | 0.252752 |
| mid / real | 1＊ | -0.020791 | 719.011396 | 390.741404 | 8.124043 | 0.543443 | 0.011299 |
| mid / real | 2 | -0.025489 | 719.011396 | 372.783135 | 9.501878 | 0.518466 | 0.013215 |
| mid / real | 3 | -0.029888 | 719.011396 | 336.963484 | 10.071292 | 0.468648 | 0.014007 |
| mid / neutral | 1＊ | -0.020556 | 719.011396 | 446.788193 | 9.184010 | 0.621392 | 0.012773 |
| mid / neutral | 2 | -0.021196 | 719.011396 | 522.225195 | 11.069441 | 0.726310 | 0.015395 |
| mid / neutral | 3 | -0.019476 | 719.011396 | 504.512252 | 9.825706 | 0.701675 | 0.013666 |

选中epoch中early hidden约1.00546，mid约719.01140；early gated／hidden为0.24211／0.26441，mid为0.01130／0.01277。mid门控绝对值和绝对delta更大，相对隐藏扰动却更小。仅凭gate大小推断“使用多少消息”会误导，gate符号也不是功能正负标签。尺度差异提供优化机制线索；四格关联尚不能证明它造成MAE差异，也不能据此临时修改缩放或学习率。

第一步forward的gated残差为0，符合alpha0身份保持；第一步日志中的gate是optimizer更新后值，两个值属于不同时间。JSON保留每epoch首／末门控、梯度范围及原run日志SHA，未做新拟合。

## 原论文对应与固定改进方向

1. **零初始化门控。** Flamingo PDF p5 §2.2/Fig.4、p8 Table3，SHA `5b96767ada8d76ad160c3087d80975b8af8e60b46786b10248c6b96b0c4699ba`，支持身份保持门控动机。本四格全是alpha0，尚无alpha3比较，保留原扩展计划；alpha3仍是门控条件。[原文](https://arxiv.org/abs/2204.14198)
2. **固定瓶颈／池化。** Flamingo p5 §2.1及BLIP-2 p3（SHA `d59913d16edf1dc7e661ca3510c6efaa2b001146840d2e464a7fde980d088dd2`）提供固定latent瓶颈依据。当前无参数16×128发送器与原学习型resampler不同；global16对照仍属固定矩阵，本四格不提供配额优越性证据。[BLIP-2](https://proceedings.mlr.press/v202/li23q.html)
3. **训练内表示warmup。** BLIP-2 p3、p6–7/Fig.5及ProtT3 p2、p4–6/Table4、p16（SHA `ffa65fc93979bcaf3cff2c60eaa4b832dff9492508f3493112755c3ca97038a1`）支持阶段化表示学习动机。pKD辅助数值损失为本课题迁移候选；未执行，也不加入本轮36矩阵。[ProtT3](https://aclanthology.org/2024.acl-long.324/)

条件探针p2–3、控制任务p1–3/p5要求区分给定函数族的可读出性、探针记忆和模型使用；旧C2六个摘要线性增益区间跨0不构成总信息不存在定理。[条件探针](https://aclanthology.org/2021.emnlp-main.122/) · [控制任务](https://aclanthology.org/D19-1275/) ProtST p3–4更新PLM，ESM-2本地预印本p3–4研究结构读出，均不直接保证当前双冻结pKD系统。[ProtST](https://proceedings.mlr.press/v202/xu23t.html) · [ESM作者引文](https://github.com/facebookresearch/esm) 七篇完整版本、URL、页码与SHA在[证据卡](literature_evidence.json)。

本次将内容相对交互、格式／数值概率作用和深度固有尺度分别记录。稳定性继续由原矩阵其余种子与场景检验；三种子统计未齐仍为null，不新增优胜门槛或改变评价规则。
