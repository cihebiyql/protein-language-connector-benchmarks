# 16项固定阶段证据独立复核

结论：通过，未发现需更正的数字、方向或估计对象解释。审计对象限定SHA c1b83473…的19:29固定JSON及其中16个ORIGINAL V1终态路径：原来源2项＋新终态14项，seed17与42完整primary四格；未读取seed2026活跃产物或任何V2/V3候选答案。重点补核前次9项后新增的7项。

## 实际核验

- 144项命名物理artifact pins逐文件SHA一致；310项Python登记指纹与各实际result注册表逐条相符。本次未重复rehash／执行Python源码、底座或旧9模型，新增7项只核selected reader文件SHA，不加载权重。
- 16项实际完整三epoch、连续1548个optimizer更新、516／1032／1548日志及最早完整validation NLL最小值checkpoint一致。各case参数登记同为464001，sender零训练参数；每例每次forward16×128 float32逻辑8192字节，真实train／validation payload账本一致。
- 4752条原始自主预测均以标准库严格JSON重放（唯一pKD、有限数值、拒绝布尔／重复键），与保存parse／gold／完整分母一致；MAE与并列平均秩的全局Spearman独立算术重算匹配。Spearman虽处target_macro容器，含义仍是全体实例相关，不是相关系数宏平均。
- 4752条C1逐行记录独立重算：G先取每行完整答案token平均的off−on NLL，再按行平均。format／numeric／mixed贡献共用各行完整token分母，类别和与原G的约简残差单列；未将各类别mean相加或用token池化替代原按行G。strict donor仍0，P/H未定义。
- 原48个epoch的history、类别count/loss_sum组成、原NLL选择与首step记录均一致。原loss／δ来自更新前forward，alpha是更新后字段；零首forward δ／相同标量loss不被夸为完整logits证明。
- 逐case输出mean、总体SD（ddof0）、min／max、精确不同数值数和完整频数重算一致。它们是post-test描述，区别于缺失的三seed样本SD，没有新增endpoint、接受阈值、校准fit、检查点选择或评价标准。

## 方向与缺失统计

独立重算原contrast方向：

| 场景／contrast | seed17 | seed42 | 结论 |
|---|---:|---:|---|
| new_peptide_new_target／depth_real | -0.069366 | 0.345709 | 符号翻转已核 |
| new_peptide_known_target／content_mid | 0.210095 | -0.356790 | 符号翻转已核 |

全部primary只有2/3，gate／pool扩展为0/3；snapshot与机制报告的三种子mean／sample SD均为null。没有用两seed平均冒充三seed结果，也没有用posthoc输出SD替代seed不确定性。四系统DiD为正并不保证mid绝对优势或content-mid为正，16报告保留了这一边界。

## 估计对象与结论边界

θ_R和θ_N分别训练并由原validation选点。已有L(θ_N,m̄)−L(θ_R,{m_i})是训练、选择与评测消息一起改变的系统总效应；不是同一θ_R在推理时逐例使用分子内容的直接测量。实际登记只有一次correct自主生成和correct／off C1，所核正式artifact中未有固定θ_R的TRAIN均值替换；不能将θ_N项或DiD补成缺失的L(θ_R,m̄)。这是被核正式通道的缺失，不是对整个磁盘探索记录的断言。

C1固定同一选后θ关闭整个新增分支，off既不是零memory也不是TRAIN均值替换。neutral仍有变化的允许prompt与动态query，正G不能单独证明确切肽靶内容使用或生物因果。TRAIN均值是16×128消息张量均值，不是pKD标签均值，也不能填入严格donor P/H。

范数与输出分布为相容解释提供描述证据，尚未识别导致优势不稳定的唯一原因；同θ内容干预、实际BF16加法后的hidden更新差、alpha专属梯度等未测量均保持缺失。现有报告对此区分准确。

本次未重采样bootstrap区间或重读论文PDF；只核原点值／区间副本一致性、报告算术及解释范围。独立算术舍入核对容差不构成新的科研接受阈值。未执行旧CPU/GPU测试、第三方／项目模块、模型forward、fit或进程操作；仅新增本复核JSON／MD。
