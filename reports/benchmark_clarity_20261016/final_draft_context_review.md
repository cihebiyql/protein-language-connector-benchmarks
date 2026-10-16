# 核心24草稿：参考背景与统计层级复核

**新增数值参考和其他panel状态审核通过；C1两项贡献范围确为24个单case值的极值，原数值无需更正。** 未发现新的数值错误、评价标准漂移或需停止运行的事项。下文有一条非阻断性范围用语建议。此审查仍是core24草稿审查，不是G／P或全轮终态验收。

审核MD SHA为 `d109aaebb49a7b00cd86462228f20c8ee1c3945df3c95d5e78595bbb8b22f173`，与当前JSON自检一致。只读草稿、固定核心24、`baseline_context_note`及已有数值／合同审计；没有读取活跃gate、执行forward、fit、GPU、新bootstrap、下载或进程操作。仅新增本文件和同名JSON。

## C1范围属于哪个层级

每个case是一个scenario×seed×placement×content训练系统。每类C1贡献先对每条测试答案按其**完整答案token数**归一，再在该case的测试行上平均。草稿引用的是这24个case平均值之间的min／max，不是7,128条测试行逐条贡献的极值，也不是8配置各三seed均值的极值。

| 贡献 | 24个case单值min—max：与主稿一致 | 8配置三seed均值的min—max：另一个量 |
|---|---|---|
| format | 0.323547852555874—0.3301392586671855 | 0.32656632581812217—0.33011703962557487 |
| numeric | 0.1898418846747657—0.22123104616957115 | 0.19457877324440093—0.21599918417378053 |

format最小为double／seed42／early-real，最大为double／seed2026／mid-neutral；numeric最小为known／seed42／early-real，最大为known／seed2026／mid-neutral。单位仍是nats／完整答案token，不能改成类别自身均值或概率校准量。`FINAL_REPORT.json.mechanism_descriptive_ranges`与`core24_mechanism.json`及以上独立算术均一致。

## 其他尺度范围的层级

hidden RMS约early1.005、mid718—719，在所选validation快照和全部72个validation epoch的极值范围相同。已改正的seed17 gated δ／hidden则属于**各case所选epoch的validation快照**：

| 范围 | early | mid |
|---|---|---|
| seed17所选epoch：当前主稿约0.24—0.30／0.011—0.013的来源 | 0.242107932349419—0.29528295147630285 | 0.011298907488775673—0.013324953451030436 |
| seed17全部3epoch validation轨迹：不与上行混用 | 0.2156958123322689—0.31375099162245323 | 0.011298907488775673—0.015395362745641135 |

**范围用语建议，非新的数值错误：** 在主稿该句增加“在所选epoch的validation中”，即可直接固定这一层级，避免读者当成seed17所有epoch或全部24个case的轨迹极值。不需重算、换分数或修改任何原history。原“含prompt／teacher-forced答案、BF16 cast后相加前、未测实际状态差”限制继续有效。

## 新增背景与出处一致

- 三种数值参考的两场景micro MAE与`numeric_reference_audit`一致；每算法每场景为一个已选中fit，ridge／kNN仍有此前validation候选搜索，没有新增拟合或捏造三seedSD。53系数与kNN保存容量分别说明，有限数值覆盖不冒充语言严格JSON覆盖。
- 当前real四组三seed平均MAE均高于保存的TRAIN mean：差分别为＋0.341822、＋0.450924、＋0.122569、＋0.159755 pKD。此算术定位准确，不是同预算架构比较、显著性或等效性结论。旧30轮MSE神经参考另列训练／选点／输出预算，neutral不被改名为TRAIN mean数值模型。
- 其他panel段与`contract_audit`相符：A2原跨条件HOLD，7070条目标域诊断不替代跨条件复现；B1为旧两架构六检查点、test两类且无本轮新连接器结果；B2 reviewed QA与C3事实gold缺失；旧C2不替代新系统C2。

新增背景没有改变核心24数值、原contrast、失败规则、受控比较估计对象或未来方案身份。原草稿扩展／资产最终保全／清理／发布待闭合状态继续保留。完整SHA、六个既有数值参考的承接检查、四组参考差及C1两层极值见[JSON审查记录](final_draft_context_review.json)。
