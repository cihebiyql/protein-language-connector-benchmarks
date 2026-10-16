# V3统一batch2候选的窄工程审查

结论：**允许将统一generation batch2登记为唯一下一工程候选；状态为 implementation_pending、qualification_pending。** 本次不是V3执行放行，不是资格通过声明。选择依据限于真实TRAIN上的batch数值一致性和吞吐需求，未使用新增test分数，不改变科学评价标准。

本审查只读V2真实TRAIN资格报告、执行收据、队列状态及科学冻结文件；没有读取新增test预测或分数，没有GPU、训练或源码修改。科学冻结SHA仍为 `9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`。

## V2失败证据及V3准入理由

`suite_batched_v2_20261016_1514` 首个TRAIN canary的 `batch_invariance_qualification.json` 为accepted=false、test_rows_used=0、reader_state_unchanged=true。B4在固定4条TRAIN中的前两条产生裸JSON与代码围栏的互换，导致首EOS前token、raw_text及原严格parse不一致。失败清单全部属于B4；B2_first_two没有不一致记录，但它只比较前两条，不证明完整4条batch2或其他配置已通过。

执行收据exit1，进程组已不存在；suite为failed_or_partial，只有一个canary job，队列已停止。V2失败、全部原始输出、源码和合同必须原样保留，不改称运行成功，也不进入V2 test。这个TRAIN-only数值证据允许提出更小batch2的单一候选；它不允许放宽解析、切换精度、修改checkpoint或按test选择版本。

## 唯一候选的接受边界

- V3对全部36项统一使用generation batch2，原锁定行顺序连续分批，尾批1条；禁止按模型、样本或表现切回V1／V2择优。V1原输出继续单独保留；V2失败保持失败。
- greedy24、原EOS／padding和decode规则、无KV cache、严格parse、gold、失败分母、所有参数／消息预算、bf16／fp32及数值flags、36项配置、3轮训练与validation选择均不改变。
- 创建独立V3源码、工程说明、准入manifest和标准合同，绑定既有科学冻结SHA；不可覆盖V1／V2或无记录修改source pins。完整验收的V1训练可以复用同case检查点与history，新增optimizer steps为0，不能计为新训练重复。
- 最早13:20开始、23:20硬截止不重置；因工程准备或资格失败消耗的时间照常计入。

## 必需的TRAIN资格及停止规则

1. 先完成最终源码／cache准入、CPU正负和序列化测试、独立实现审查、fresh资源与标准合同；本次未证明这些已完成。
2. 以既定相同6种TRAIN资格配置、同一批预定4条TRAIN、相同reader state比较B1与B2：**四条全部以连续两批处理**，不得只复用V2的B2_first_two。每行首EOS前token、decoded text及原parse必须严格一致，不设容差、不修文本。
3. 保留短／中／长／最长prompt选择和ID tie-break，固定顺序及全部原输出；CPU继续覆盖混合EOS、pad==EOS／不同、无EOS到24步、raw／规范长度和通信账本。记录reader state前后SHA，不能以重新初始化来“通过”。
4. 全6配置通过前不进入V3正式test或后续完整trial。实际memory／RSS、整批耗时、有效与padded query、先EOS行额外读取均照实计量；不伪分每行耗时或补空调用。
5. 任一V3资格不一致或资源／实现门槛失败立即停队列、保留归档，不继续搜索batch大小、改dtype、改parser或看test选择工程版本。可回到原已接受V1按原规则和剩余时间执行；若不足以完成剩余34项，按23:20截止归档partial与k/3，不能延长窗口。

全部通过只支持V3在这些TRAIN输入／6配置上的资格，不证明所有test输入或所有训练状态跨batchshape逐位相同。若正式V3 test与历史V1出现差异，完整保留并标明工程版本；主比较统一V3，不用V1答案回填。当前没有V3真实资格结果，本文件仅接受这一有界的下一候选。
