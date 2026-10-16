# V2统一batch4推理的窄工程审查

结论：**方案边界可接受，状态为 implementation_pending；TRAIN资格验收尚未完成，不能据此放行GPU或声明V2已通过。** 本次只读实际 `engineering_addendum_v2.json`、科学冻结协议、V1边界handoff及运行状态／退出字段；没有读取新增test预测或分数，没有GPU调用、训练或源码修改。最终实现审查由独立审查者负责。

科学冻结文件 `configs/matched_budget_20261016.frozen.json` 重新计算SHA仍为 `9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`，与已接受版本一致。新增工程说明不修改36项矩阵、seeds、训练3轮／1,548更新、validation选点、sender／reader参数、16×128消息、24-token greedy、无KV cache、严格数值策略、parser、gold和失败分母。

## 可接受边界

- 唯一运行变化是统一采用generation batch4，原锁定行顺序连续分批，末批1—4条；全部36项使用同一V2评价版本。V1每行推理的原输出保留独立通道，禁止逐case择优、混合填补或用V1答案修补V2。
- 完整验收的V1训练可复用同case的best checkpoint及完整history，必须核对数据、初始状态、1,548更新、3轮和原validation选择。复用没有新增optimizer step，只算同一模型／seed，不能计为重训或增加训练重复数。本次尚未独立核验这两份checkpoint/history的全部复用pins。
- 固定总窗口仍为13:20—23:20，不能因V2准备、资格失败、源码新版本或复用重置。到期保留partials，三seed汇总按既定k/3和null规则处理。
- 统一batch4匹配每次预算和24-token上限，实际padded工作、整批forward、早EOS行留在批内的额外读取必须如实计量。保留全部返回token槽、首EOS长度、padding尾、原解码及parse；耗时用整批真实时长，不虚分每行耗时。

## V1边界核验

`V1_boundary_handoff.json` 记录停止目的为统一吞吐版本切换；`suite_state.json` 的退出字段显示：两个early模型的trial均complete、exit0且进程组不存在；第三个mid-real trial为complete=false、`controller_signal_15`、进程组不存在。顶层异常为 `Supervisor stop requested before spawn`。因此该第三项不能计为训练失败或已训练模型；仅有合同／收据空目录，不是第三个完整运行。原V1归档的failed_or_partial身份保持。

## 最低必要资格证据

1. 新建V2源码SHA、缓存准入manifest和真实标准合同，绑定既有科学冻结SHA与工程addendum；最终CPU正负／序列化和独立源码review通过，旧版本不覆盖。
2. fresh资源检查及原进程清理证据齐备，再做TRAIN-only真实资格测试。固定选择短／中／长／最长prompt，ID仅用于固定tie-break；同reader状态逐行比较batch1、batch4及batch2尾批至首EOS的token、decoded text和原严格parse。
3. CPU夹具覆盖混合EOS、pad与EOS相同／不同、无EOS的24步上限、raw／规范长度和通信账本。记录实际GPU／RSS峰值与吞吐，不把纯CPU夹具称为真实CUDA资格。
4. 任意资格不一致均归档并停止，不利用test输出选择版本。严格determinism不保证不同batchshape的logits相同；TRAIN canary一致只证明已测输入，不能声称所有test输出与V1逐位相同。
5. 真实test若产生版本差异，保留两版原输出并明确数值执行版本；新的主比较统一用V2，不回填、不挑有利版本。资格失败或截止前无法完成则诚实归档partial。

本次认可的是减少推理墙钟的统一工程方案。实际source/cache pins、CPU实现验收、标准合同以及TRAIN-only CUDA资格仍pending；没有重新放宽科学协议或评价标准。
