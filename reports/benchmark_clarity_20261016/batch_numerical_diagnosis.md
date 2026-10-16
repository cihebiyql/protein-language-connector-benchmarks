# V2 TRAIN 批量不变性失败诊断

**V2 的 B4 工程资格保持失败，不能进入正式36项评价。根因尚未确定。** 本次只读首个真实TRAIN canary的资格收据、原始B1/B4/B2输出、batch ledger、配置和源码；未读本轮test指标，未调用GPU，未运行任何候选mask/backend改动。唯一写入是本报告及同名JSON。V1/V2所有旧文件保持原样。

证据目录：[首个canary](../../docs/PUBLICATION_SCOPE.md)。JSON保存逐文件SHA、逐行首分歧位置、ledger复算、已安装Transformers源码指纹和判断边界；涉及的五份执行源码均与V2准入SHA一致。

## 已确定的事实

使用同一post-update reader state，资格比较前后所有reader张量一致，四行均属于TRAIN，prompt有效长度分别224、229、229、233。B4中前两行的**第一个生成token**即与B1不同：token515（裸JSON起始）与73594（代码围栏起始）互换。另两行的canonical IDs、文本及parse一致。

| TRAIN sample | B1 | B4 | B2首两行 |
|---|---|---|---|
| PPIRE02255 | 裸JSON，严格parse有效 | code fence，严格parse失败 | 与B1逐token/文本/parse一致 |
| PPIRE09147 | code fence，严格parse失败 | 裸JSON，严格parse有效 | 与B1逐token/文本/parse一致 |
| PPIRE18275 | code fence，严格parse失败 | 与B1一致 | 未运行 |
| PPIRE19086 | code fence，严格parse失败 | 与B1一致 | 未运行 |

两种输出正文中的数字都是10.0，B1/B4各有1行严格parse有效，但这**不能证明等价**：有效样本身份已改变，原定比较字段也真实不一致。不得去除代码围栏、忽略parse、只比数值或以总体有效率相同放行。

首token分歧发生在EOS及任何post-EOS padding之前，因而EOS尾padding/canonicalization不是该首分歧的原因。B2第一行多出的4个尾pad在canonicalization中按约定识别，解码仍使用原始完整IDs和原parser；这一点符合旧decoder语义。

## Ledger 无异常，速度不能抵消资格失败

逐批独立复算均满足：calls=g；padded queries=B[gLmax+g(g−1)/2]；逻辑message bytes=B×g×8192。

| 输出 | rows | reader批forward总数 | padded queries | 逻辑message字节 | 整批时间之和 |
|---|---:|---:|---:|---:|---:|
| B1 | 4 | 68 | 16,124 | 557,056 | 5.5278秒 |
| B4 | 4 | 18 | 17,388 | 589,824 | 2.2419秒 |
| B2 first-two | 2 | 18 | 8,550 | 294,912 | 1.8963秒 |

B4这四行观察到约2.47倍吞吐提升，但资格失败；B2前两行相对其对应B1时间约1.36倍，仅为很小的单次工程观察。没有重复计时/稳态保证，也没有剩余两行或其他配置的B2证据。逻辑字节不是PCI/网络实测字节或Shannon信息量；已结束行在原批量decoder中仍随同未结束行前向，其额外工作已计入，不补dummy调用。

## 错误迹象与数值敏感应如何区分

源码中selected行顺序、collate、store批量索引、zip(selected, raw)和逐行记录相互一致，未发现明显的前两行互换实现。batch ledger中的原始tensor行与逐行记录顺序相符。两个模板恰好互换并不构成行交换证明，因为另外两行也输出相同模板。没有实际collated token/message指纹与逐层trace前，也不能仅凭源码审查彻底排除运行时数据问题。

现有数值收据确认base bfloat16、reader/message float32、关闭autocast和TF32、最高float32 matmul精度、strict deterministic、无cache。**同一shape的确定性不等于不同shape逐位不变。** GEMM/SDPA形状、归约路径及padding分支仍可能改变有限精度结果。第一个token存在裸JSON/围栏竞争，与小margin被数值变化跨过的机制相容；没有保存该步logits，不能证明margin小，也不能证明舍入就是根因。

还有一项更具体的本机实现事实：Transformers4.57.6的`masking_utils.py`中`_ignore_causal_mask_sdpa`（219行）允许全1、q_length=kv_length的mask省略为None；`sdpa_mask_recent_torch`（265行）在有padding时产生4D bool mask。`integrations/sdpa_attention.py`的`use_gqa_in_sdpa`（30行）又根据mask是否None决定enable_gqa或显式repeat_kv，`sdpa_attention_forward`（48行）据此调用SDPA。当前无padding B1和混合长度B4按源码预期会走不同mask/GQA分支；这属于**候选数值路径差异**，没有实际GPU kernel/mask trace，不能把它当作已验证原因。

B2首两行也有left padding但匹配B1，故“只要存在4D mask就必然分歧”已不受这些例子支持；batch大小、padding宽度、GQA路径与logit margin可能交互。此现象既不能证明普通bf16舍入，也不能证明decoder数学错误。

## 能区分原因的TRAIN-only证据（仅候选，未运行）

1. 对四行逐一核对去padding后的input IDs、memory tensor SHA、memory mask、position_ids与最后有效位置；比较运行前后reader state、base参数/有关buffer、eval/dropout状态。
2. 在**相同prompt前缀**上保存首步和首分歧步最后有效位置的logits、top候选ID及top1−top2 margin，比较B1/B2/B4；后续已分叉序列的logits不可当作同条件差值。
3. 重复同shape输入，并反转B1/B4运行顺序。若同shape也不重复，需先追查隐式状态或非预期非确定性；若只有跨shape不同，再定位矩阵/attention路径。
4. 保存实际None/4D mask、is_causal、GQA分支、设备kernel信息；按有效位置对齐逐层hidden及reader delta，定位首次差异在backbone、reader还是最终projection。
5. 固定padding宽度的B1/B2/B4、统一显式4D mask、隔离float32/eager比较可作为后续机制诊断；它们改变执行实现，**没有在本轮运行或获准作为正式评分替代**。即便改善一致性，也不能抹去V2失败或混用各模型的最佳路径。

## 唯一下一个工程候选：统一B2

B2值得做一次有边界的资格验证：现有两行完全匹配，计算形状更接近B1且可能节省时间。但它尚未合格，不能由first-two直接推广到全四行、六配置或36个训练后模型。

保持预先选定的四个TRAIN行及顺序，B2必须覆盖**全部四行**，按连续两行一组，与四次B1比较canonical IDs、原始解码文本和严格parse，三项均精确一致。六项既定canary为两个场景各early-zero、mid-zero、mid-near_unit，均seed17、real balanced message；同一canary的比较始终使用同一个post-update state。还需reader state不变、ledger公式通过、有限值和完整分母。任意一行任意字段失败即停止；不得因为数值pKD相同放宽。

候选只允许统一B2，需新的明确implementation准入/指纹，不能原地改V2 B4源码。若合格，所有36模型按同一固定输入顺序、同一B2分组评价；已有V1/V2字节保留，不按case选择B1/B2，不改变训练、参数量、36项矩阵、greedy24、parser、test、val选择或23:20绝对截止。新评价次数如实登记。

六项TRAIN资格通过只能支持“在这些预定工程探针上未发现不一致”，不证明任意训练后权重、seed或全test都与B1逐位相同。正式结论应围绕统一执行版本及冻结标准，不把本次小样本资格包装为普遍数值不变性。
