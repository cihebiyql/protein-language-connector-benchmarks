# Molecular Memory V3：自动评价合同审查

本轮以冻结A1的两个泛化场景开展新架构实验。主评价保持原严格JSON生成、原始精度pKD及全样本MAE；numeric head提供独立的数值读出评价。两条输出通道各自保留覆盖率、失败和选择依据。旧30个完成结果归入historical reference；同优化预算的基线需要在新协议中登记并实际执行。

审查状态为**可据此实现，正式测试前须完成新协议冻结**。本次实际执行了文件SHA、划分/ID/标签一致性核对，以及27项纯函数/定义固定夹具检查；没有模型forward、训练、安装或修改旧数据。机器字段、来源SHA、函数行号及待冻结项见同名JSON。本scope首时刻为2026-10-17 00:29:40+08:00，截止10:29:40，旧执行窗口保持关闭。

## 1. 冻结总体、观测单位与解释范围

| 场景 | train / validation / test | 测试参考靶标数 | 测试操作group数 |
|---|---|---:|---:|
| new_peptide_known_target | 2062 / 147 / 152 | 48 | 67 |
| new_peptide_new_target | 2063 / 431 / 442 | 157 | 125 |

两场景的samples、test_gold、manifest、benchmark_lock均重新计算SHA；test_gold的ID、顺序和未舍入pKD与samples.test完全相等。样本身份及单位保持原定义：exact KD换算pKD=-log10(KD[M])；训练答案按原规则序列化为两位小数，正式评分使用原始精度gold。范围、困难行和异常输出均不按测试表现调整。

group_id是来源/序列家族形成的连通操作组，source DOI与repeat_measurement_candidate也不是已核实独立物理实验编号。known场景允许且要求参考target在train出现；double-new另外隔离target family。两个场景分别报告，来源重叠不形成独立实验复现。已见过的A1测试集用于新架构的冻结比较与稳健性分析；确认未知总体的结论需要未来独立外部测试。

两场景全部2361/2936行的温度、pH、buffer、actual_construct_sequence均缺失，actual_construct_verified均false，assay均存在。预注册的missing-vs-observed分层仍需输出：missing分母为全体，observed分母0、指标null/undefined_no_rows。当前数据不支持这几个字段的缺失与非缺失效应比较。严格同条件pair和独立合格donor覆盖保持0；pair direction、within-target-condition相关、P/H保持undefined。固定TRAIN均值不增加donor覆盖。

来源：旧run_connectors.py:55/75/88/96/117/190；panel_execution.py:36；information_diagnostics.py:100/106；旧contract_audit.md:11–28。完整SHA列在JSON。

## 2. 输入隔离和hash锁

固定骨干为ESM2 t30 150M（revision a695f6045e2e32885fa60af20c13cb35398ce30c）与Qwen2.5-1.5B-Instruct（989aa7980e4cf806f80c7fef2b1adb7bc71aa306）。协议同时锁tokenizer/chat-template、ESM层/分块规则、特征manifest、缓存索引、源码和新checkpoint；骨干冻结情况与实际梯度由runner收据证明。既有来源/家族隔离和外部暴露限制保留。

输入视图由白名单显式构造：peptide/target序列仅送冻结PLM/分子记忆构造；LM保留原匿名prompt和context语义。chemical_state仅允许c_terminal、chemical_modification、cyclization_method、explicitly_representable、n_terminal、residue_modification、topology七键；实验context仅temperature_kelvin、ph、assay、buffer、missing标记、actual_construct_verified和target_sequence_role。ID、DOI、source_record_id、groups、raw_metadata、KD/pKD/label不得进入prompt、memory或probe特征。ID和gold仅存在于评价输出信封。

numeric head读取memory或答案前的prompt-only因果state。若使用Qwen state，明确提取最后真实prompt位置，禁止答案后缀位置或包含答案的pool。TRAIN夹具只改变gold-answer文本而保持输入不变时，numeric预测及C2表示必须不变。teacher forcing仅用于LM监督损失与注册的NLL诊断。numeric head输出不得自动修正、替换或格式化为LM自主答案。

原samples含所有split标签，文件hash存在不等于训练进程隔离。新scope建立train/val有标签视图与test无标签输入视图；视图保留原顺序和映射，原文件不改。训练/选点进程不读取test gold或test诊断；选点记录落盘后才解锁测试生成和独立gold评分。ESM确定性无监督缓存可覆盖全部输入，任何统计拟合、标准化、可训练sender、numeric标签尺度和probe拟合只用TRAIN。协议审核进程核对test身份/分母不参与模型选择。

## 3. 严格自动输入格式

同名JSON给出可直接实现的schema约束。JSON/JSONL拒绝重复键、NaN/Infinity、非JSON尾随内容和bool伪装数字；每条JSONL为一个对象，不跳过损坏行。所有artifact引用必须位于注册只读根或显式锁定数据根，无symlink逃逸，并通过SHA后才评分。

- run_manifest：schema/scope/run/phase/scenario/seed/config/protocol/source/model/data/tokenizer SHA；双通道定义、训练预算、参数/流量统计、选点记录、checkpoint SHA、receipt与artifact pins。formal的scenario/seed/config必须唯一地属于冻结矩阵。
- LM预测：原test顺序每ID一条，split、gold_pkd、checkpoint/protocol身份、raw_text、原始generated IDs、EOS长度与解析记录。scorer从raw_text重新严格解析并核对saved parse；gold从独立锁定文件读取，信封gold必须精确相等。tokenizer仅离线解码核raw_text，不加载模型。
- numeric预测：相同ID顺序与gold，预测为有限原始pKD数值；没有JSON parse这一分母。非有限数或真实推理失败记null及明确原因，全样本numeric指标undefined；有限子集指标仅诊断。
- C1：每ID、同一checkpoint、相同原答案及token数，correct/off/train_mean各自loss_sum/count与每行NLL。缺失模式、标签或token分母不同均拒绝。分解桶固定format/numeric/mixed，三类count重构full，贡献使用该行full-answer token分母。

manifest的训练/选点/输出引用可在not_started、失败或partial状态为null；formal complete必须具备注册双通道及C1全部文件和验收，numeric-only基线另列任务身份。缺行、重复ID、错序、错split、外来ID、gold不符、伪造parse、SHA不符或正式选择记录缺失均为invalid_input，不能通过丢行后重算成为正式结果。真实生成格式失败属于有效运行的parse失败，必须保留；进程只写出partial时，保留已有字节并把未生成行列为missing，禁止补造预测。

## 4. 指标、分母与checkpoint

**主LM评价。** 沿用单键有限数值{"pKD":x}的严格parser，接受正常JSON空白，拒绝解释、code fence、重复pKD、多键、字符串数值、bool、非有限数。不clip、不手工补字、不取文本中的首个数字。n_total固定152/442；parse_valid_rate=n_valid/n_total。任何一行失败，full_sample_mae=null且状态undefined_parse_failures；valid-only MAE明确显示子集分母，不能替代主指标。

**数值输出。** 单独报告numeric full MAE、RMSE=sqrt(mean(error²))、总体Spearman及target-macro MAE；单位分别为pKD、pKD、无量纲、pKD。target-macro按groups.target_sequence_sha256先算每靶标MAE再等权平均，完整覆盖时分母48/157；旧字段group_macro_mae实际就是这一指标，不冒充group_id macro。Spearman使用平均秩处理tie；少于2行或任一侧常数时null。总体Spearman评价总体相关，严格同条件排序另保留undefined。

**选择规则。** 建议正式联合模型沿用全validation答案token NLL最小、最早tie；训练完成后从注册epoch范围选择，先写selection_before_test，再执行测试。同一θ*同时输出LM主结果与numeric head辅助结果，避免按测试分别挑权重。val使用Σanswer_loss_sum/Σanswer_tokens，EOS计入，prompt/pad以-100屏蔽并按causal shift对齐。format/numeric NLL或test MAE不参与该选择。tie按原存储精度作精确比较，算术重构容差不充当tie或优势阈值。

numeric-head-only基线没有可用LM选择目标时，预注册全validation numeric MAE、最早tie，并独立标明selector。若确需额外numeric-endpoint-selected checkpoint，应在首个test前登记为额外输出/预算/独立artifact，不能替换共同θ*结果或追溯择优。新架构、loss权重、epoch/step/batch/LR/seed矩阵和实际主selector须由执行协议冻结；本审查不擅自沿用旧1548step作为新训练预算。

LM生成默认沿用greedy、24新token、batch1、相同EOS/skip_special_tokens解码。新批量或数值策略须先经过固定TRAIN canary并注册，测试后不能搜索批量/重写parser。训练和C1的固定batch、精度、TF32/deterministic策略同样预先登记。

## 5. 固定θ机制诊断

LM C1对每行先求原gold答案token均值，再平均各行：G_off=mean_i(NLL_off,i−NLL_correct,i)，G_mean同理替换为TRAIN-mean。其分母与val token总量加权不同。三类桶的loss先除该行full token数再做行均值，另外记录归约残差；直接相加三类各自条件NLL会改变定义。

branch-off在同一θ*、prompt和答案下把已注册LM memory residual分支关闭。TRAIN-mean在相同message接口逐slot替换；若sender可训练，必须用**所选θ***、eval mode在TRAIN重新计算mean，记录每slot有效计数、missing mask规则、checkpoint/source SHA。mean计算不读取val/test，不更新θ。固定θ干预与独立训练neutral模型各自登记。

numeric head若直接读取memory，关闭LM reader不等于关闭numeric输入，此模式应标not_applicable或明确仅LM-branch干预。numeric的有效memory替换可报告mean_i(|y−pred_mean|−|y−pred_correct|)。两类效应衡量当前模型对消息分布的敏感性，合格生物donor仍为0，不能填P/H或解释为生物因果/互信息。

## 6. 多seed、公平基线与开发路线

正式seed固定17/42/2026，每scenario独立训练；所有注册variant逐seed运行，保留失败和未完成项。方法及配对contrast均报告逐seed值、k/3、missing原因；只有3/3指标定义时才给跨seed均值和样本SD(ddof=1)。run-complete coverage与full-MAE-defined coverage分开。对照缺一个seed，该seed配对差记null。样本/操作组bootstrap如注册，固定θ按group_id整组重采样，并与seed变异分开；67/125组不是独立物理实验数。

旧30结果只列历史参考。新baseline与新模型逐项核训练样本/更新数、batch、optimizer/LR/WD/schedule、选择机会、随机初始化和骨干、输入信息、可训练/有效参与参数、辅助head/损失及通信slot×dim×dtype×调用数。无法匹配的维度明确写出；相同optimizer steps不等于相同FLOPs。branch-off固定θ与retrained ablation分别命名，不能互换。

开发阶段可按预注册train/validation grid持续诊断学习曲线、gradient/gate、head误差、format/numeric NLL、数值读出与LM格式瓶颈；早期无优势不触发按表现丢seed或停止其它对照。技术失败记录原因，修复需新revision与新目录。新架构/hparams只按TRAIN/VAL选择；冻结后完整test及注册C1按固定计划运行，test结果不触发改架构、扩大搜索、改变分母或删困难样本。后续思路进入新协议并承认既有test暴露。

## 7. 无模型实现与扩展端点

公共analysis/analyze_results.py为stdlib实现，SHA38ef732334c3556a786a0c9fc5154c9d72e9599c145306b7068b874ed84d912e。可复用strict_loads/parse_answer、average_ranks/spearman、regression、generation_metrics、seed_statistics；audit_training的3epoch/1548step和旧36矩阵属于旧合同，须由新注册预算驱动，不能照抄硬编码。新增RMSE、numeric覆盖、三模式C1、manifest/schema校验是待实现项。

本次27项无模型固定检查通过，涵盖严格JSON正负例、重复ID/错gold/错split/伪造parse拒绝、parse失败full MAE=null、target-macro与行均值区别、秩tie/常数、seed缺失和NLL分母区别。最后两项是公式夹具，不代表新evaluator已实现。新工具还需加入hash篡改、gold泄露、错checkpoint、selector晚于test、重复run、partial、path escape、numeric非有限、token分母错、mean混入heldout、缺失分层空组和预算不匹配的自动负例。

A2原跨条件任务维持HOLD；现有March分钟RT目标域4952/1055/1063属于独立诊断，selection525/calibration530，不能填回原iRT→分钟迁移端点。B1可单列31候选已知阳性检索任务：378注释训练/134val/76test，训练合并后348查询，test仅2类；未注释候选为unknown，主class-macro MRR/R@1/R@3与sample MRR分开。A1接口、gold和MAE不能直接套到这两个端点。扩展均须先登记独立数据/标签/候选/单位/selector/hash身份，新A1执行不自动完成它们。
