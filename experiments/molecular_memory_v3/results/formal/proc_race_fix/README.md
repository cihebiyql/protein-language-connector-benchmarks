# /proc 枚举竞态：未来版本修复包

修复计划：保留PGID精确匹配、排除zombie、`rsplit(')',1)`解析及已有单PID读失败处理；把可能在iterator内部stat PID的递归glob改为只枚举`/proc`根目录，再在受保护的循环体内读取每个数字PID的stat。根目录不可访问必须明确抛错，不能伪装成空进程组。先运行确定性夹具与Linux真实自有子进程测试，再在未来新合同版本应用补丁。

`safe_proc.py`仅包含建议替换的函数；`mm_supervise_proc_race.patch`为对冻结原源码SHA d02e97147bec77140862491cab10a45051a647728d62bb47324621395f5cb787的unified diff。补丁尚未应用于冻结源码。没有更改四项历史失败状态、原收据、训练窗口或测试评价。

运行：`python -B -m unittest discover -s . -p test_safe_proc.py -v`。4项测试涵盖旧glob iterator确实抛错／新函数跳过已消失PID、仅匹配指定PGID并排除zombie及损坏记录、根目录PermissionError必须传播、Linux真实新session子进程隔离及退出观察。Windows仅执行前三项，Linux执行全部四项。真实测试只通过其自行创建的Popen句柄清理子进程，不调用killpg、不触碰外部进程。无需第三方依赖、模型或GPU。

未来应用会改变监督器代码SHA，需要新修订、新源码锁及新任务合同。补丁解决本次已定位的枚举竞态，不宣称解决所有PID复用、权限隐藏或监控故障。

实际验证：2026-10-17在Windows运行3项通过、1项Linux测试跳过；node1用现有Python3.11通过stdin在内存执行相同helper与测试，4/4通过、0跳过，耗时0.238秒。原监督器运行前后SHA相等。见`execution_receipt.json`的完整日志、helper／test／patch摘要及原源码未变证据。独立helper与unified patch中替换函数的AST已核对相等。
