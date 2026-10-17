# node1用户授权共享版本

本版本保持原始node模型训练源码、模型权重、A1划分、评分、24组／三种子矩阵不变，使用用户明确授权的共享资源策略。与本地streaming版本独立，不计入旧失败pilot或旧结果。

`resources/freeze_formal_shared.py` 需要全部显式根参数、两个规范终态接受的TRAIN canary和新鲜真实资源证据，额外指定 `--selected-gpus 4 5 3`。候选仅这三卡，最多三个自有worker；每workerCPU2、RSS64GiB、GPU allocator20GiB，要求主机至少额外4GiB RAM及所选GPU至少额外512MiB空闲。CPU闲置／I/O／GPU利用率和其他已有计算进程只作观测，不作启动阻塞条件。任何其他进程都不被终止或抢占。

根目录 `USER_RESOURCE_POLICY.json` 的实际SHA同时绑定ready证据、源码／资产pins和冻结配置。模型／缓存／旧输入逐字节核验、独立PGID watchdog、实际开始时间、绝对截止、规范归档和原科学评价均保留。common micro2×accum2＝effective4、gen8／max24、10epoch目前仍为候选；只能在实际TRAIN资源计时后、正式测试前统一冻结，禁止按测试分数调预算。早期语言格式不佳不构成冻结的分数门槛。

绝对截止仍为2026-10-18 03:26:57北京时间。工具自身不启动GPU或执行实际冻结，父任务负责按实际资格释放。

`resources/analyze_v4_shared.py` 保留原模型无关重验、全部24状态、每底座同种子M0−M3、四格交互和独立底座主效应。只有规范验收、文件重验和评分重算一致的任务计入；均值／样本SD仍要求3/3，格式改善不冒充误差收益。

本目录targeted fixture `test_shared_freezer.py` 只测试合成资源证据：忙CPU／高利用率／已有计算允许共享；RAM／VRAM不足、旧证据或错误policy SHA仍拒绝，不代表真实GPU资格或正式结果。
