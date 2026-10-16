# 第31计划pool尝试：执行完成但验收拒绝的隔离复核

**本尝试不能计入正式pool结果。** 原result-validation.json明确valid_for_acceptance=false，错误为result.execution.started_at：“contract timeout exceeds the execution window at actual start”。原result-record自写lifecycle_status=succeeded及review.status=accepted，不能覆盖独立标准validator的拒绝；worker exit0或科学artifact.complete也不能替代正式验收。

原合同、result、validator、raw预测、模型和history全部原样保留。本复核没有修合同、改评分／时钟、重试或启动GPU／forward，只新增两份隔离报告。

## 实际留下了什么

本case为known／seed17／mid／real／global16／alpha0，确实完成3epoch、3×516＝1548条连续更新，原完整validation NLL最小值／最早并列选在epoch1，checkpoint SHA一致。原V1 worker及冻结protocol登记、464001名义reader参数、sender零训练参数、16×128 float32／8192逻辑字节和实际train／validation／B1生成通信账本一致。

152条raw自主生成均严格可解析，与gold／原MAE／全局Spearman算术一致；152行C1按行及类别共同full-token分母／残差算术一致。它们仅是**未验收隔离证据**，本报告不披露其模型分数，不填入pool contrast、三seed均值／SD或正式成功计数。模型／raw／history的物理指纹附于JSON。

worker实际PID／PGID312725、exit0、原终态收据记录自有PGID已退出；采样whole-PGID RSS和Torch allocation在已设上限内。真实执行不是训练失败或实际超时，但正式合同资格仍被拒绝。

## 时间条件：提前结束不能修补声明

| 原始事实 | 数值 |
|---|---|
| 记录的actual start UTC | 2026-10-16T14:57:26.488676+00:00 |
| 原绝对截止 | 2026-10-16T23:20:00+08:00 |
| 声明timeout | 1360秒 |
| actual start至截止剩余 | 1353.511324秒 |
| 声明timeout超出剩余 | 6.488676秒 |
| 实际monotonic执行 | 554.066969秒 |
| 实际UTC跨度 | 617.438976秒 |
| 终态UTC | 2026-10-16T15:07:43.927652+00:00 |

1360秒的声明预算在记录actual start时超过剩余窗口约6.489秒；实际617.439秒UTC／554.067秒monotonic并提前结束，不会使该声明条件追溯合法。不得倒改start、缩改原timeout、扩大deadline、重写验收或据此回填正式结果。timeout构造根因与外部进程签名链由独立监督审计负责，本报告只核原始字段和算术。

## 最终冻结汇总一致性

最终原helper快照JSON SHA2e72addb…、MD SHA fc8171d3…核验通过：36个原计划位置完整保留，30 complete、1 invalid_or_incomplete、5 missing；本case为invalid_or_incomplete且metrics=null。正式接受30项的result物理SHA、metric和source origin均与此前30快照完全一致：原来源2＋新接受28，8910／8910严格有效。两个pool contrast均0/3、mean／SD null，未混入本case的152条隔离数据。

5个missing保持原helper状态；parent／controller审计报告它们未启动，本窄复核不单凭missing推定运行历史。原1398资产最终check记录unchanged=true已交叉核验，未重复全盘重hash。

已保存suite终态为failed_or_partial，异常正是terminal archive validation failed；成功jobs列表28项不包含该归档拒绝尝试。UTC guard终态signal_sent=false、termination_reason=null，记录在controller终态后结束。controller OS PID为263028，worker为312725；25342／26213是工具会话ID，不是OS PID。parent报告controller会话exit1、guard正常退出；本报告未重新查询或操作这些进程。

没有新的正式pool成功或稳定性结论；全36仍未完成。原30个已接受科学结果保留，其余尝试和缺失必须在最终汇报中单独列出。
