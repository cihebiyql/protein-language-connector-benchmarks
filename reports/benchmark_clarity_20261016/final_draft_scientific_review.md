# FINAL_REPORT核心24草稿的科学审查

**结论：主体解释与已审核心24项一致，有1处P2数值范围需在发布前窄修；没有发现需要更改训练、基准或评分的事项。** 本审查仅针对`review_draft / as_of_core24 / extensions_pending`，不是36项扩展闭合、最终保全、进程清理或工作台发布的终态验收。

## 待修的真实问题

**P2：seed17早接入相对残差范围漏掉双新场景。** [FINAL_REPORT.md第107行](FINAL_REPORT.md)写“seed17的gated δ/h约0.24—0.26与0.011—0.013”，但同段没有限定为已见靶点。

已审`core24_mechanism.json`所选epoch validation统计为：

| seed17 early条件 | gated δ／hidden |
|---|---:|
| known／real | 0.242107932349419 |
| known／neutral | 0.26441474334210835 |
| double／real | 0.2914958213540949 |
| double／neutral | 0.29528295147630285 |

因此两个场景合并描述应约为**0.24—0.30**；也可明确只说known范围，再单列double约0.29—0.30。mid的原近似0.011—0.013可保留。需要同时保留“validation含prompt和teacher-forced答案”“BF16 cast后、相加前δ”“未测实际状态差”的现有限制。JSON没有对应的截短early范围字段；修改MD后须更新`draft_self_check.current_MD_sha256`，不能改原history或机制数值。本审查没有自行修草稿。

## 通过的主体部分

- 顶部可见地声明核心24阶段草稿及扩展／终态待闭合；没有把原36项或其他多任务基准说成全部完成，也没有把两篇论文入库冒充本报告已发布。
- 读者在结果前能看到任务、两scene数据量、来源／家族操作分组、参考构建物和条件缺失、test已观察、跨scenario不是独立复验等边界。
- 自主严格JSON覆盖和完整pKD MAE为主要任务证据；原选点是full-validation answer-token NLL，gold未舍入。失败分母、任何失败full MAE为null、少于3seed不汇总、SD不是CI均明确。C1和描述性机制统计没有替换主分数。
- 八配置三seed值、十个核心contrast、方向与mean／sampleSD均与21:31固定快照一致。known正DiD没有被当成mid绝对优势；double第三seed负content-mid／DiD及相应不稳定结论保留，没有挑seed。
- content被定义为不同θ_R／θ_N训练、选点和评测的系统总效应；固定θ_R均值干预缺失、C1整分支off、donor为0时P／H undefined、新固定消息C2未运行均正确。
- 同预算只指同构Reader的464,001名义元素和每forward 8,192逻辑字节；不是全FLOPs、真实搬运、可辨识自由度或相同EOS累计工作。early非classic prefix、其query与mid上下文化不同、无slot／type embedding均披露。
- G／P保持原36内待终态，S／W明确未来未执行。S用TRAIN prompt-only校准、branch内LN和零gate，不挪用含gold的val范数；W的ungated Reader／真实1537参数head、联合465538和附加暴露／计算成本正确，不假称与原CE-only同预算。具体step／placement等未来门槛仍由所链方案和审查承接，不是已实现合同。
- 九篇论文和现有作者源码作为机制动机使用，没有改分、全球首创、1:1复现或无条件迁移承诺。NormFormer分支LN／skip缩放、ReZero直接零标量与当前tanh及整网理论的差别保留；不同底座与训练目标详见所链原方案。

## 草稿身份与后续验收

当前G／P结果、原1398资产最终保全、全轮完成／partial／失败及自有进程组清理、终态独立审查和报告发布均有明示占位，**本次不关闭这些占位**。待父任务结束并形成新的不可变终态证据后，最终稿应再审；不能把本次主体通过引用为全轮终态已通过。

本次只读草稿、已审核心24快照／报告、估计对象与文献方案，不读活跃gate、不做模型forward、GPU、fit、新bootstrap、下载或进程操作。仅新增本文件与同名JSON，未修改草稿、旧报告、源码、数据或矩阵。完整已读SHA与核查结果见[机器记录](final_draft_scientific_review.json)。
