# content contrast与固定检查点干预：估计对象澄清

**已有content contrast是“训练＋验证选点＋评测使用变化消息”的系统总效应；它不是同一训练后检查点在推理时逐样本使用分子内容的直接测量。** C1则固定检查点关闭整个新增分支，二者的处理和损失不同。此澄清不改任何数值、contrast、协议、标准或旧报告，也不补开评测。

## 三个量必须分开

固定一个scenario、seed和placement，令`m_i`为对象i的固定残基消息，`m̄`为该scenario **TRAIN消息张量的逐slot均值**（不是TRAIN标签pKD的标量均值），`c_i`为原允许prompt。`θ_R`、`θ_N`分别是在real／neutral条件下独立训练并由原validation full-answer NLL选中的reader；冻结sender、Qwen、初始化和训练规则按协议匹配，不共享训练后的reader状态。

令`L(θ,{m_i})`表示在相同锁定test、相同`c_i`及原greedy／parser下的全样本pKD MAE。原规则仍是任一生成失败则full MAE未定义，不能为了写公式改用有效子集或补值。

| 量 | 形式与实际状态 | 能回答什么 |
|---|---|---|
| **已有content总效应** | `Δ_total = L(θ_N,m̄) − L(θ_R,{m_i})`；即原`content_early`或`content_mid` | 给训练、选点及评测提供变化消息的整个系统，相对始终提供TRAIN均值消息的系统，任务误差如何改变。包含学习轨迹、选点、格式／输出分布及推理计算的共同变化；不能单独隔离固定θ的推理内容作用。 |
| **同θ_R的均值替换** | `Δ_mean|R = L(θ_R,m̄) − L(θ_R,{m_i})`；**所核冻结matched A1 V1中未登记／未测，保持缺失** | 如未来另行登记执行，可描述固定real检查点对该特定消息替换的计算依赖／分布外敏感性。本次没有补算它，也不能用上面的`θ_N`结果填入`L(θ_R,m̄)`。 |
| **已有固定θ C1** | `G(θ)=mean_i[NLL_i(θ,branch off)−NLL_i(θ,branch on,m_i^*)]`，`m_i^*`取该模型原real或neutral消息；每个NLL先按该行完整答案token平均 | 同一已选检查点中，整个分支对teacher-forced gold答案概率的计算作用；单位nats／答案token，不能与pKD MAE差互换。neutral检查点同样可以有非零G。 |

`Δ_total`的正负沿用原协议。四格DiD仍只是四个分别训练系统的交互，不能通过代数运算得到缺失的`Δ_mean|R`。配对seed和初始化减少训练条件差异，不把两次训练变成同一θ。

## source_fact：实际代码与登记

- `MessageStore`构造时固定`message_input`；neutral分支每行返回同一个16×128 TRAIN均值，real返回该行消息。`collate`对train／validation／test复用这个store，原prompt不变。[run_trial.py:115](../../historical/scripts/matched_budget_20261016/run_trial.py)、[batch:154](../../historical/scripts/matched_budget_20261016/run_trial.py)、[collate:162](../../historical/scripts/matched_budget_20261016/run_trial.py)。均值按TRAIN split构造且不使用标签：[prepare_cache.py:51](../../historical/scripts/matched_budget_20261016/prepare_cache.py)。
- 每个run独立AdamW训练，在其原消息条件下计算validation、保存并恢复所选reader，再将**同一store**交给C1和一次自主生成。[训练:299](../../historical/scripts/matched_budget_20261016/run_trial.py)、[选择及评测:345](../../historical/scripts/matched_budget_20261016/run_trial.py)。登记计划只有一次`correct`自主生成和`correct/off`各一次C1，没有固定real checkpoint再用mean memory的第三推理通道：[登记:595](../../historical/scripts/matched_budget_20261016/run_trial.py)。
- C1在同一model、同一批memory及相同gold token上切换`message_mode=correct/off`，没有改变store内容。[C1:220](../../historical/scripts/matched_budget_20261016/run_trial.py)。`correct`对real／neutral都表示分支开启，不是“正确分子”这一事实判断。
- `off`直接跳过注入hook，运行冻结base；没有把memory置零，也没有替换成TRAIN均值。末尾`alpha×0`只保留零梯度图，不增加分支作用。[forward:230](../../historical/scripts/matched_budget_20261016/connectors.py)。Reader的MHA、query和output投影包含bias：[reader:17](../../historical/scripts/matched_budget_20261016/connectors.py)。

## synthesis：现有表述应落在哪个范围

由这些算子可推得：全零memory仍可能经MHA／output的bias产生非零残差，因此它一般不等于off；这里没有执行额外零memory实验。

[原协议审查](protocol_review.md)已经称real−neutral为“重新训练系统比较”；[解释计划](interpretation_plan.md)及[八格报告](interim_mechanism_8cases.md)中的“内容收益”“相对内容作用”应统一按上述`Δ_total`理解。它们没有报告固定`θ_R`的mean替换结果；不能将系统收益再扩写成“该固定检查点在推理时正确识别了每个分子的生物作用”。这是解释范围的补充，不是更改原contrast或否定已保存分数。

neutral仍有变化的允许chemistry／assay prompt、动态query和受监督训练得到的输出适配，故其分支即使没有逐样本残基消息变化，也能影响NLL并得到G。正G证明的是该分支改变了所测gold概率，不能单独证明分子特异信息；真实消息模型与neutral模型G的差也跨了两套θ，不自动成为固定θ内容消融。

## unknown：未被这些量识别的内容

未测的`Δ_mean|R`保持未测，本次不搜索活跃case或重新推理补齐。即便未来测得非零均值替换效应，也可能包含离开训练消息分布、均值方差／范数变化或格式先验等作用；它不证明正确肽—靶点匹配、结合机制、生物因果或每行都依赖内容。当前严格独立匹配donor仍为0，P／H保持undefined；TRAIN均值替换不满足合法donor条件，不能用来填充P／H。

科学冻结SHA保持 `9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`。只读范围限定为冻结matched A1 V1的协议、源码和所列既有报告，不以此声称整个磁盘不存在其他探索尝试。全部来源SHA、精确源码行号及source_fact／synthesis／unknown分类见同名JSON。未读取活跃case、加载模型、forward、fit、GPU或改动历史文件。
