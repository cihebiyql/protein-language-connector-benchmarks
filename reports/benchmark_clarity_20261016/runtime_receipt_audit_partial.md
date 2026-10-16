# 运行收据局部独立审计

固定快照：2026-10-16T09:45:36.628283+00:00（北京时间2026-10-16 17:45:36）。仅纳入此时已完整终态的原来源2项与新续跑7项，共9项；活跃seed42 early-neutral及其后任务不纳入。未读取活跃中间产物作终态证据，未GPU／模型forward或修改源码、结果、控制器、进程。

结论：9项参数、输入、训练、选点、来源、失败分母、资源与终态证据检查全部通过；存在必须保留的UTC与monotonic计时差异，不能混称同一wall time。

- 每项真实checkpoint的reader tensor共464001个float32有限元素，sender零训练参数；16×128消息，每例每次forward逻辑8192字节，无新增语言位置。
- 完整train／validation／test快照与冻结原数据逐字典一致；全部1548步optimizer日志连续，三轮各516步；按完整validation原NLL取最小值、最早并列，checkpoint hash一致，未按test选点。
- 2528条test原始预测逐条重放旧strict parser并重算旧metric全部字段一致；2528条有效、0失败，完整分母保留。此报告不披露MAE等模型分数，也不按分数选择来源。
- 原2项仍位于原suite并保留原PID／receipt，在新suite未重跑；原6个TRAIN资格的pins、环境freeze及strict flags一致，未伪称新资格。
- 每个新job均有实际fresh probe、标准合同、原V1 worker/source/model/cache/environment pins与终态exit0／ownedPGID已退出收据。probe至进程启动17.26–24.18秒。
- 实际采样host group RSS最大5.968GiB（上限12GiB）；Torch peak allocated最大6.952GiB（上限11.5GiB），reserved最大11.203GiB单列。Torch及OMP/MKL/OpenBLAS设置均2线程。

| 来源 | 场景／seed／位置／内容 | test有效／总数 | monotonic秒 | UTC跨度秒 | 差值秒 |
|---|---|---:|---:|---:|---:|
| 原来源 | known/17/early/real | 152/152 | 729.517 | 805.811 | 76.294 |
| 原来源 | known/17/early/neutral | 152/152 | 722.282 | 802.363 | 80.081 |
| 新终态 | known/17/mid/real | 152/152 | 555.799 | 615.864 | 60.065 |
| 新终态 | known/17/mid/neutral | 152/152 | 559.956 | 623.454 | 63.498 |
| 新终态 | double-new/17/early/real | 442/442 | 990.504 | 1103.972 | 113.468 |
| 新终态 | double-new/17/early/neutral | 442/442 | 1107.382 | 1230.849 | 123.467 |
| 新终态 | double-new/17/mid/real | 442/442 | 951.051 | 1054.524 | 103.473 |
| 新终态 | double-new/17/mid/neutral | 442/442 | 979.284 | 1089.400 | 110.116 |
| 新终态 | known/42/early/real | 152/152 | 709.475 | 789.587 | 80.113 |

九项UTC跨度均比receipt monotonic elapsed长60.065–123.467秒，资源时间序列内共246处相邻两时钟增量差超过1秒。具体原因无法仅凭现有收据确定，不能据此声称host暂停或某服务改钟。两种计时分别保留；实测每项均在monotonic与UTC两种1800秒限制之内，结束UTC也早于原23:20绝对截止。不得用较短monotonic数值替代UTC成本或宣称同等wall time加速。

通信账本已独立用全部raw token数量和prompt长度重算：B1生成calls、physical/valid queries、16KV attention edges、逻辑消息及残差字节均一致。实际EOS读取数／FLOPs不要求相等，不把逻辑payload称为物理传输或信息量。

采样RSS不证明瞬时峰值；终态进程退出依据归档收据，未检查或操作其他活跃进程。本局部审计不代表36项或三随机种子完整完成。逐case hash及clock跳变列表见JSON。
