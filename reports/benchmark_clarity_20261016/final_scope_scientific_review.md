# 全轮终态的只读科学审查

**审核通过：30项正式结果及其解释可发布为本轮部分完成的终态；没有未关闭的科学解释阻断。** 这不是“36项科学测试全部成功”的验收。原池化拒收和5项未启动必须继续保留，不能因本审查通过而改成成功或补入正式统计。

审核绑定：主MD SHA `7eef1f0807a024385de35180eec1adfeaa3e1a4526b11b89e822d9e7d894001f`；修订后主JSON SHA `00a0dab56d66768955341902a186f627f73a2915f44f76ac32183c66e51d3309`；最终原汇总SHA `2e72addb16ad461d017a08fc372a1f3696948048c3be97f516403628ed1773c4`。科学协议仍为 `9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`。

## 终态分母与隔离符合原规则

- 原36项完整保留：30 accepted＝原来源2＋新接受28；1项池化尝试未验收；5项未启动。另有1次实际完成但拒收的新训练，故实际新训练共29、连原来源共31，不能只数成功jobs，也不能把这31个都叫科学accepted。
- 正式30项共8,910条严格有效预测。case31的152条raw输出、模型、history和原分数保留在隔离归档，不进正式指标／seed统计；不能把其exit0或artifact.complete改称正式通过。
- 两场景的pool正式覆盖均0/3，所有seed值及mean／sampleSD仍null。没有用一个未验收case推断balanced／global16优劣，没有删掉未启动seed或改变原contrast。

## 拒收原因没有被改写

case31合同冻结timeout为1360秒，记录actual start到原23:20截止仅1353.511324秒，声明超出6.488676秒。冻结至启动间隔16.976635秒超过调度预留10秒；终态标准validator因此返回false。原result-record自写accepted不覆盖独立validator。

实际worker结束早于截止，UTC跨度617.438976秒、记录monotonic554.066969秒均低于1360秒；所以这不是实际超时、训练未完成或JSON解析失败。提前完成也不能追溯修补声明条件。原合同／start／timeout／deadline／验收标准没有倒改、没有重试，拒收后停止队列及保留partial是与原规则一致的处置。UTC／monotonic差异原因未定，报告没有伪造修正。

进程终态与原1398资产全部unchanged采用已绑定的独立诊断／执行审查和最终收据；本次不再次查询或操作进程，也不重复全盘rehash。其记录支持controller／末worker退出、guard未发信号及5项未启动。验收缺口由报告明示，不伪装成池化方法的科学负结果。

## 已有科学结论保留且没有扩张

核心24的三个seed、8组配置和10个contrast保持原值，real深度效应两场景均跨seed翻向。known正DiD不是mid绝对优势；double第三seed的负content-mid／DiD没有删除。content仍为分别训练／选点的θ_R与θ_N系统总效应，C1是固定θ整分支on／off；固定θ均值替换缺失、严格donor下P／H undefined、新C2未执行等范围没有改变。

门控六项按原near-unit−zero方向，两场景均一正两负，均值负不证明alpha3稳定更优。独立核实：六对near-unit所选原validation NLL全部较高，而test MAE有4/6较低。这涉及不同分区、目标、teacher forcing／生成及不同训练参数，不是selector的受控消融；主报告正确没有按test改选点，也没有从4/6挑winner替换核心主表。

同nominal参数与每forward逻辑消息字节不被说成相同FLOPs、EOS累计读取、隐藏尺度或生物信息量。两个scenario独立训练／选点，不被当成独立外部复验；数据库参考构建物／条件、操作分组和已见test的限制继续保留。九篇论文仅支持机制动机；未来S1／S2、W仍未实施，TRAIN prompt-only／分支LN／辅助头预算与原CE-only的边界保持，不作保分、首创或原论文1:1复现承诺。

## 文稿与发布身份

本次发现的机器元数据旧scope问题已在审核关闭前由父任务修正：JSON顶层现指完整36项终态；终态快照时间与核心21:31时间分开，原核心summary SHA明确为primary表溯源，最终summary另有SHA。主MD科学值未改变。此前selected-epoch validation范围限定也已落实。

主报告全部42个相对本地链接在本审查文件生成后均存在，包括预留的本审查链接；没有要求联网引用在本机成为文件。JSON的41已存在＋1待审查者生成是其pre-review检查时点记录，不是当前缺链。

工作台字段明确是文稿冻结时尚未发布，随后release／实际发布receipt和工作台索引另存；这是有意的不可变文稿策略，不是虚构发布或遗漏当前发布ID。本次批准解释内容及其部分完成身份，不冒称已经执行后续发布。

没有读取或启动新模型、forward、GPU、fit、新bootstrap、第三方代码或重试；没有修改FINAL、原件、数据、评分、时窗或旧审查。仅新增本文件与[同名JSON](final_scope_scientific_review.json)。该JSON保存完整SHA、计数／时间算术、链接检查、已关闭元数据观察和本次审查范围。
