# 多种子、匹配预算协议独立审查

审查结论：**协议层 R1—R4 已关闭，可进入实现和运行资格验收。** 固定消息、同构 reader、两个位置各自训练的主比较能够回答受限的接入深度问题。协议审查关闭不等于训练放行；下述实现证据齐备后，无须扩展无关 SOTA 或新增测试集即可执行。此次只读审查，没有 GPU、训练、模型推理、数据或旧结果修改。初审快照和关闭复核 SHA 分别保留在同名 JSON。

关闭复核核对了最终 JSON、README 及 `prepare_cache.py` 的窄修：时钟已统一为13:20:00—23:20:00；协议明确有效／padded／整批 EOS 消耗分账；7 项配对差值、方向、固定 case 次序及3/3统计规则均已写入；固定 sender 的36项清单对历史备选建议具有明确优先级。参数自由度、neutral保留prompt和alpha3仍是门控的限制也已注明。下表保留原更正要求，作为关闭依据；运行时落实尚需证据。逐seed差值可以报告，缺seed时禁止发布完整三seed汇总；协议中的“contrast”按这个汇总层含义解释。

随后仅核对命名澄清：`ordered_trial_rule` 的内容顺序为 real、neutral；新增 `content_vs_inference_mode` 明确 CLI `message-input real|neutral` 是训练内容轴，两者均使用模型 `message_mode=correct` 开启分支。`message_mode=off` 是另一个冻结 LM 推理参照，不是第三种训练内容。本次两字段澄清不改变36项矩阵或先前审查结论，更新协议 SHA 并保留上次关闭快照。

运行数值策略追加复核：`numerical_policy` 对全部36项规定相同 bf16 底座、fp32 reader／message、关闭 reader autocast、highest float32 matmul 精度、关闭 TF32／cuDNN benchmark，开启 cuDNN deterministic 和严格 deterministic algorithms（warn_only=false）；CUDA_LAUNCH_BLOCKING=1、CUBLAS_WORKSPACE_CONFIG=:4096:8 等环境固定。该策略在新增正式运行前预定，先以 TRAIN-only canary 验证支持性；不支持的确定性 kernel 应停止资格验收，工程修订须重新绑定源码／协议且不能由测试结果选择。它有助于同环境重复性，不是跨硬件／软件版本逐位相等保证。此处仅完成协议复核，最终源码是否在初始化 CUDA 之前设置环境、是否实际落实 flags，以及真实 canary 的成功，仍属于原执行门槛，尚未由本次审查证明。此前命名澄清保持原值，case 和端点没有因数值策略增加而改变。

## 已成立的设计

- A1 是宽泛数据库参考靶标 pKD 生成回归，不是已核实构建物、严格同条件效力或全部多任务基准。`group_id` 是来源／家族连通组，不是独立物理实验编号；组 bootstrap 和 seed 方差分开报告。
- 两 scenario 跨分区重合明确；逐场景独立训练、归一化和验证选点是必要条件。冻结无标签底座和数据无关初始化可共享，任何已训练状态、TRAIN 均值或验证选择不得跨场景传递。两场景不得合并成独立重复证据或混合 bootstrap。
- Reader 参数算术正确：3,072 + 196,736 + 66,048 + 198,144 + 1 = **464,001**。发送器为零可训练参数；同 seed 的 reader 应从同一个实际张量快照加载，不能仅靠相同 RNG seed 推断一致。
- 主要矩阵 24 + 门控初值扩展 6 + 池化配额扩展 6 = **36 次新训练**。复用的主矩阵对照不是额外独立模型。seeds 17、42、2026；3 轮、batch 4、1,548 更新及原验证全答案 NLL 选点均明确。禁止选最好 seed、按 test 重排指标或改变失败处理。
- real 与 neutral 的接口均固定 16×128 float32、全一 mask、每次消息 8,192 逻辑字节，无额外 LM 位置。neutral 是各场景 TRAIN 的逐 slot 均值；已读 sender/cache 代码不以标签构造消息，也不在 neutral 张量中携带逐行 ID 或原长度。
- 旧 test 已见的事实已披露。本轮属于在既有冻结基准上的预先规定后续比较；不能称全新未见确认试验。旧样本、gold、分组、提示、解析器、原精度计分和历史结果必须保持原字节。

## 关闭 draft 前必须更正的事项

| 项 | 更正和可验收证据 |
|---|---|
| R1：时钟不一致 | `prepare_cache.py` 当前写 13:20:44—23:20:44，protocol／README 为 13:20:00—23:20:00。全链采用最早 **13:20:00—23:20:00**；单 trial 到期时间不得超过总截止。旧 foundation 07:18—17:18 不恢复、不延长。保留新的关联目的与旧合同区别。 |
| R2：有效与实际 query 分开 | protocol 的“every valid text position”只描述有意义输出。实现的 MHA 对 padding 位置也计算，再 mask residual；no-cache 生成中先结束的行仍留在批内。应同时记录有效 token、B×Lmax 的实际 padded queries、整批 reader forward 数、每行首次 EOS／有效生成长度和批次数。不要用每行 EOS 长度反推实际批计算量。 |
| R3：锁定具体配对分析和缺失规则 | 每场景固定报告 early-real 对 mid-real、early-neutral 对 mid-neutral、各位置 real 对 neutral；深度×内容差中差是预定次级机制量。门控为 mid alpha0 对 alpha3；池化为 mid balanced 对 global16。明确差值方向，逐 seed 报告。任一解析失败时该 run 全样本 MAE 为 null；某 seed 缺失时不得只对剩余成功 seed 求均值却写成完整三 seed 优势。给 k/3、失败与未完成表，不做分数驱动补跑。 |
| R4：统一可执行版本 | budget／literature 文档仍有 learned codec、噪声、联训、无门控及 warmup 等建议。其为备选设计，不是这次 36 项任务清单。追加唯一 protocol 的优先级和排除范围；在任何新增 test 推理前绑定 protocol、case 顺序、源码、缓存、初始化与合同 SHA。 |

上述事项是合同或表述的窄范围更正，不要求增加方法、不要求修改原评价标准。

## 需要限制的科学解释

**接入深度。** 早期 reader 与中层 reader 都是动态 query 的 residual 注入，早期接口不是原 input MLP，也不是传统 soft-prefix。比较支持“在这一固定 sender、同构 gated reader、固定预算及冻结底座下，改变读取／注入深度后的系统总效应”。query 所处隐藏表示、剩余传播层数、反向路径和状态尺度是深度干预的组成部分，并未分别隔离。它不能单独证明旧内部系统失败就是插层、参数或池化中的某一个原因，也不能推广为所有内部连接器的优劣。

**参数。** 464,001 是相同结构中参与 forward 的可训练张量元素数量，不是已经证明相等的可辨识有效自由度。neutral 去除了个体差异，可能降低可辨识自由度；MHA 的 K 投影 bias 在 softmax 中存在公共平移抵消；alpha=0 时首步除 gate 外的梯度应为零。这些均不是挂名凑参数的证据，也不应要求每个元素首步梯度非零。验收应确认没有额外挂名模块、全部预期张量参与图，并在 gate 打开后的多步检查中记录有限梯度和实际更新；共同的结构冗余如实声明，不为凑“有效参数”临时改架构。

**neutral。** 固定 TRAIN 均值没有逐样本的残基消息或长度 side channel，但保留数据集输入分布先验；原允许的 chemistry／assay prompt 也仍可携带个体信息。因此 real−neutral 是“固定允许 prompt 之外，个体残基消息可用性”的重新训练系统比较，不是所有分子信息的消除，也不是严格匹配 donor 的因果量。均值消息与真实消息的方差／范数可能不同，差异包含内容与输入分布变化；两者的范数、slot 方差应作 TRAIN／VALIDATION 描述，不能事后按 test 重新缩放。运行层仍须证明 neutral collate 每行完全相同、全一 mask，且不另把 ID、长度、来源或原残基传给 reader。

**门控。** tanh(0)=0；tanh(3)≈0.995055，且导数≈0.009866。alpha3 仍是可学习且近饱和的门控，不是“无门控”。该扩展检验两个门控初始化及相应优化路径，不能写成零门控与无门控的因果对照。当前 36 项无需为了凑文献对照额外扩容。

**通信。** 8,192 字节是模块边界每对象每 forward 的 float32 源消息载荷，不是网络／PCI 流量、互信息或总计算量。MHA 内部两份投影 K/V 另有逻辑张量，query 和 residual 随文本长度变化。训练同序列／顺序时按次预算可匹配；生成只能匹配每次操作和相同 24-token 上限，EOS 造成真实累计次数不同。不得补空调用伪造相同总量；同时报告真实 wall time／显存和 padded 运算账本。

**文献。** Flamingo 支持固定瓶颈和零初始化 tanh 残差的动机；BLIP-2、ProtT3、ProtST 支持表示学习与连接器训练目标需要分开的动机。它们没有证明当前 16×128 固定随机投影在 pKD 上必然有效；类型配额、数值辅助 warmup 属于本项目待验证迁移。当前先只做固定 sender 的深度／内容／初始化／配额，warmup 与外部语料继续排除。条件探针 ΔMSE 不能升格为总互信息；严格 donor 为零时 P/H 保持未定义。

## 训练放行门槛

1. R1—R4 有具体关闭记录；冻结 36 项 case 清单、固定顺序、seed、选点规则和失败／超时处理，观察新 test 后不改。
2. 新 run 只读旧数据、缓存和结果；新独立输出目录。复核旧资产 manifest 与全部输入 pins，禁止跨 scenario 加载任何训练结果或 neutral 均值。
3. CPU 正／负测试覆盖：同消息与初始化逐位一致；neutral 不随 ID／长度变化；原 prompt／label／parser／metrics 重用；gate0 恒等和预期首步梯度、多步更新；hook 异常清理；padding／EOS 分账与全样本失败传播；无额外源通道。
4. fresh 资源检查、标准执行合同、进程组／内存／墙钟监管、总截止清理全部具备后，用 TRAIN-only canary 验证每种结构／门控模式的真实 forward-backward，不能以测试分数选配置。canary 不是正式科学结果。
5. 完成某 run 后，逐条原始输出、n_total/n_valid/n_failed、checkpoint 选择、完整1,548更新、参数／通信计数、SHA和实际进程退出均有验收证据。未完成的 case 明确列出；不把可执行 protocol、下载权重或通过 canary 计为完成基准。

此次没有重新核验全部旧 locks、没有运行 unit tests、没有复读所有论文 PDF；来源／数据事实采用已提交的只读审计，具体参数与接口结合当前实现静态检查。故本结论是协议审查，不是实现或科学结果的通过证明。
