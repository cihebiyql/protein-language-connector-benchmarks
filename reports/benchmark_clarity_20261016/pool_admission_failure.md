# Pool扩展第31项：准入时间窗失配诊断

第31项实际训练完成并exit0，但标准验收为false，应隔离为未验收产物。原前30项的result-validation仍全部true，另5项未启动。没有重试、改时窗或执行新任务。

| 原始记录 | UTC |
|---|---|
| probe | 2026-10-16T14:57:09.500971+00:00 |
| contract创建／冻结 | 2026-10-16T14:57:09.512041+00:00 |
| 记录的实际启动 | 2026-10-16T14:57:26.488676+00:00 |
| worker结束 | 2026-10-16T15:07:43.927652+00:00 |
| 不变截止 | 2026-10-16T23:20:00+08:00 |

合同timeout为1360秒。启动加timeout为2026-10-16T15:20:06.488676+00:00，超过截止6.488676秒。冻结到启动相隔16.976635秒，超过主控预留的10秒；实际结束仍早于截止736.072348秒。

主控在计算timeout后才构造合同、做模型SHA和check-files；预检通过到真正启动之间没有再次核对冻结预算是否完整容纳。标准validator按“实际start＋冻结timeout≤deadline”检查，故预检pass与终态fail并不矛盾。时间窗blocker不单独否定归档；明确的result.execution.started_at错误才是本次拒绝原因。UTC时长与记录的monotonic时长有差异，原因未知；本报告不修正任何时间。

result-record先写review=accepted，随后validator返回false并抛异常；以validatorfalse为准。该项未被追加到suite_state.jobs，不能仅数该列表识别所有物理尝试。字面artifact为complete、3轮／1548步、152条唯一固定test ID，均不等于已通过标准验收。本报告没有查看分数。

原summarize_remaining_v1调用冻结summarize；后者明确要求valid_for_acceptance is True。因此不会将第31项纳入完整结果。父级冻结快照确认30complete、case31 invalid_or_incomplete且metrics=null、5missing。Pool两场景均保持0/3，均值和SD为null。无需改评价标准或旧summary。

只读/proc检查：实际controller263028、末worker312725均不存在；父级报告句柄25342／26213对应/proc也不存在，但不将句柄当成Linux PID。未发现匹配controller／worker／guard命令签名。Guard终态记录已结束且signal_sent=false；其自身Linux PID未记录，故以终态收据和签名缺席共同核验。没有发送任何signal。

后续仅工程建议：在计算冻结timeout前完成昂贵哈希；科学worker开始前再次fail-closed核对实际启动预算；添加check-files耗时超过10秒的CPU负例；只有验收成功后才发布accepted，否则保存needs_revision及完整尝试索引。当前保持原合同、原分数、原目录和23:20截止不变，不用剩余时间启动预算不足的新任务。

完整逐case验收清单、进程检查、源码行和证据SHA见同名JSON。父快照JSON SHA为2e72addb16ad461d017a08fc372a1f3696948048c3be97f516403628ed1773c4。
