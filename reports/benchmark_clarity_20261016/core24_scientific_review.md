# 核心24项的独立科学结论复核

**结论：通过，未发现需关闭的科学解释阻断项。** 完成的核心24项支持“在本数据、底座和固定训练预算下，mid优势随seed变化，尚未确立稳定优势”。它不证明内部连接器普遍无效，也没有唯一识别尺度、BF16或其他失败原因。P0—P3新增发现均为0；下列限制属于真实的结论边界。

本次只读21:31固定快照、核心机制／执行审查、估计对象说明、冻结协议、已有文献方案与正式图caption。没有读取活跃gate目录、模型forward、GPU、fit、新bootstrap或进程操作。执行完成与原始预测验收采用已绑定的独立执行审查，本次另外重算快照内的三seed与contrast算术，不冒称再次加载24个检查点或重新验证全部训练。

## 通过的数值与方向核对

快照SHA为 `d22da6e98304eb85301a130480bac413106b60511e9c05d92beaeb0d53156e4b`；科学冻结SHA为 `9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`。

- 核心为两场景×三seed×early／mid×real／neutral，共24个完整V1系统，原来源2项＋新训练22项，不重复计数；7,128／7,128严格有效。八组配置各3/3，全部种子17、42、2026保留。
- 独立重算八组MAE mean／样本SD、30个同seed配对值及十组核心contrast mean／样本SD，均与固定快照、机制报告及执行审查一致。SD分母为3−1，不是CI、SE或新生物重复数；原分组bootstrap只作已存范围阅读，没有重新抽样。
- 正值方向保持原定义：depth是early减mid，content是neutral减real，DiD为depth-real减depth-neutral，也等于content-mid减content-early。另四项gate／pool contrast在本快照仍0/3、mean／SD为null，没有用活跃结果补齐。

| 核心事实 | 可支持的结论 |
|---|---|
| known depth-real均值−0.109102，seed为负、负、正 | 没有一致mid方向；均值为负也不能直接宣布普遍inferiority。 |
| double depth-real均值−0.037186，seed为负、正、负 | 同样存在方向翻转，不能挑seed42报mid稳定胜出。 |
| known DiD三个seed均正，但content-mid为正、负、负 | 正交互可以来自early的内容总效应更差；不等于mid绝对MAE更低或mid内容始终有效。 |
| double seed2026 content-mid＝−0.349635、DiD＝−0.337053 | 不能延续16项阶段的“DiD均正”。content-mid均值＋0.010095、样本SD0.321378只是三次总效应的描述，不是已证正效应或零效应。 |

[正式图caption](paired-effects/core24/fig_01_caption_zh.md)与上述方向、30个原值／10组mean±sampleSD以及θ_R／θ_N解释一致，明确SD不是CI、seed不增加独立生物样本，没有p值或新的优胜标准。本次没有重新渲染或对图像像素作质量验收。

## 估计对象与机制边界通过

[核心机制结论](core24_mechanism.md)正确沿用[估计对象澄清](causal_estimands_note.md)：content是`L(θ_N,m̄)−L(θ_R,{m_i})`，包含独立训练、验证选点与评测的系统总效应。固定θ_R的TRAIN均值替换仍未测；C1 off是移除整个分支，不是均值或零memory替换。neutral仍保留允许prompt，因此有正C1不等于分子特异信息的证据。

P／H继续因严格独立donor为0而undefined。旧C2的有限摘要／投影／线性probe没有被转作新固定消息系统的实测；新C2明确未执行。格式／numeric贡献、输出分布和范数作为诊断分列：teacher-forced NLL不冒充生成MAE，BF16相加前δ不冒充实际h_after−h_before，事后分布统计不替换主要端点。关于上下文化query、尺度、优化和格式的解释均标为相容假说，未声称隔离了单一因果原因。

同预算主张限于同构Reader的464,001名义可训练元素及每对象每forward16×128 float32＝8,192逻辑字节。early是动态reader早接入，不是经典prefix；剩余层数、梯度路径、hidden尺度、实际EOS累计读取、wall time和FLOPs仍不同。核心报告与执行审查均没有用参数／字节匹配冒称全部计算相同。

## 真实风险与不能外推的范围

1. **训练随机性仍只采样三个seed。** 固定测试组区间与seed SD是两种不同不确定性；十项contrast相关且有代数依赖，不能计为十次独立复验或挑单个区间作普遍优势。
2. **两个A1场景分别训练、分别判读。** 冻结协议禁止跨scenario共享trained state；两个场景的数据分配不是一对互斥的外部独立实验。不能合并为独立重复证据，也不能仅凭跨场景MAE差说明难度因果。本次没有重新读取源数据审定所有排除或预训练曝光。
3. **数据库参考靶标与实验条件的范围未被本轮补全。** 快照内的同靶同条件及严格实验pair指标仍undefined。构建物、条件和实际生物独立性的既有不确定性不因24项生成成功消失；这些是宽泛参考靶标pKD预测结果，不能升级成严格同条件效力、实验结合机制或湿实验验证。
4. **比较的是特定固定sender和双冻结底座。** 无slot／type embedding的8＋8只是配额；不同底座、训练规模、真正prefix、完整ProtT3／NormFormer／ReZero及其他任务不在本结论内。尺度、预热等文献方案仍是未来未执行候选。
5. **完成核心24不等于36项或多任务基准全部完成。** 门控／池化在此不可变快照中尚缺；B2／C3和其他gold缺口不能由A1通过自动补齐。这里不审查后续活跃扩展的完成状态。

以上风险不要求重划测试、补值、换指标或改原协议。当前数值和解释可保留并继续原扩展；本审查没有新的停止／阻断要求。完整已读来源SHA、算术核对和范围记录见[机器审查](core24_scientific_review.json)。
