# 自动执行器与冻结队列合同

实现：`src/mm_case.py`、`src/mm_queue.py`。它们只组合新scope的训练/评价、动态标准合同与既有监督器，不修改旧实验或自动放宽资源阈值。本文描述代码行为；尚未实际启动GPU或正式A1训练。

## CLI

从新scope根目录，以已资格化环境的Python执行：

```text
python -m src.mm_queue --plan /absolute/frozen_queue_plan.json --output /absolute/new_scope/results/queue_001
python -m src.mm_queue --plan /absolute/frozen_queue_plan.json --output /absolute/new_scope/results/queue_001 --resume
```

输出目录首次必须不存在；resume必须是完全相同的冻结plan原字节，且source/assets仍逐文件匹配。queue不会创建正式训练config、冻结budget或给自己分配GPU。CLI全部36成功返回0；保留失败/未启动则返回2。只有父任务明确ready的assignment能参与，且实际主机必须匹配。当前本scope已占GPU为0/1/6，计划应声明`occupied_gpu_devices:[0,1,6]`，只在父任务确认ready的2/3/4/5/7中选择。occupied列表由冻结计划显式登记，不是所有环境永久禁用某个设备。

单案例由queue生成命令并交标准supervisor启动：

```text
python -m src.mm_case --plan PLAN_SNAPSHOT --plan-sha256 SHA --scenario SCENARIO --variant M0 --seed 17 --physical-gpu 2 --contract NEW_ATTEMPT/contract/contract.json --output NEW_ATTEMPT/case
```

case必须继承supervisor绑定的CUDA_VISIBLE_DEVICES，且精确等于physical-gpu；模型内统一cuda:0。fit和evaluate各用普通subprocess，不设shell、setsid、preexec_fn或start_new_session，保留supervisor创建的同一进程组。唯一新session由`mm_supervise.run`创建，所有后代都在它的PGID/RSS/monotonic timeout边界内。

case还有独立POSIX主线程watchdog，避免controller意外SIGKILL后失去唯一timeout监控。CLI必传该attempt实际标准contract路径；路径在freeze前即可确定，运行时读取其SHA，不在argv中放contract SHA形成自引用。启用timer前核`/proc/self/cmdline`与contract.execution.command精确一致、cwd相同、contract已冻结且非模板、实际started.json的PID/PGID/contract SHA绑定一致；自身必须PID=PGID=session leader，且不是parent的进程组。父Popen写started.json有极短竞争时只等最多0.5s，缺失或不一致即拒绝执行，不启动模型。

独立`signal.setitimer(ITIMER_REAL)`的单次时间取`min(contract.timeout - age_since_frozen_at, absolute_deadline-now)`，再扣除计算/安装间的monotonic消耗与10ms安装余量，不晚于contract实际剩余边界。freeze早于原supervisor begin/Popen，因此这是保守边界，会把合同之后的hash/probe开销计入case预算，不能解释为额外训练时间。若已有更早timer则保留更早限制；正常结束/异常finally取消当前timer、恢复原SIGALRM handler及扣除流逝时间后的原timer。

alarm只复核并SIGKILL这个预绑定的自身独立PGID及其继承子进程，不向controller组或任何外部组发信号；SIGKILL使忽略TERM的fit/evaluate后代也不能继续运行。安装证据存case/independent_watchdog.json，含原contract SHA/实际start receipt SHA、计时预算和作用域。此后controller缺失也不取消case内的timer；原supervisor仍正常时，两层限制同时有效。若case被提前杀死则保留已有partial产物，不能伪造完成receipt。

## 固定36项顺序与预算

先core24：seed17→42→2026，每个seed内known→double，每个scenario内M0→M1→M2→M3。再extension12：同seed/scenario顺序，每组A_L→A_Q。不同GPU的完成顺序会不同，dispatch领取顺序仍来自这唯一列表；不读取分数重排、挑seed、改训练epochs或省略不利案例。

每个GPU一个线程、一个独占文件锁，不能同队列/其他同实现队列同时占同一physical GPU。另有controller锁阻止两个controller同时resume同目录。显式登记并记录总并发线程预算=`GPU数×每worker cpu_threads`和总worker RAM上限；不宣称它们是当前实际内存消耗。每case仍须新资源probe：标准create先hash再probe，run临启动再次probe并核实际剩余窗口，CPU/RAM/storage/GPU阈值保持父监督器定义。

统一截止**2026-10-17 10:29:40+08:00**。queue将wall timeout取`min(timeout_seconds_per_case, remaining-120s)`，create在昂贵准备后再按实际剩余窗口缩小，run仍执行实际启动边界检查。剩余≤125s不再提交新case。只有wallclock限制随剩余时间改变；训练config/budget签名始终同一个，无法做完的case保留partial/failed，不缩短epoch冒充完成。

## Plan JSON schema

```json
{
  "schema_version": "mm-v3.queue.v1",
  "status": "FROZEN",
  "deadline": "2026-10-17T10:29:40+08:00",
  "paths": {
    "python": "/absolute/env/bin/python",
    "project_root": "/absolute/project",
    "scope_root": "/absolute/project/experiments/molecular_memory_v3_20261017",
    "config": "/absolute/scope/configs/formal.json",
    "cache_root": "/absolute/scope/cache/full",
    "view_root": "/absolute/scope/data_views",
    "evaluation_protocol": "/absolute/scope/docs/evaluation_protocol.json",
    "data_root": "/absolute/project/data/processed/bench_v02/a1_locked_v2",
    "standard_root": "/absolute/canonical_standard"
  },
  "source_pins": {"/absolute/scope/src/mm_model.py": "64-lowercase-hex"},
  "asset_pins": {"/absolute/scope/configs/formal.json": "64-lowercase-hex"},
  "ready_assignments": [
    {"gpu": 2, "host": "qualified-host", "readiness": "ready",
     "evidence": {"path": "/absolute/scope/resources/ready_gpu2.json", "sha256": "64-lowercase-hex"}}
  ],
  "occupied_gpu_devices": [0, 1, 6],
  "gpu_memory_budget_bytes": 21474836480,
  "timeout_seconds_per_case": 1800,
  "resource_retry": {"max_attempts": 30, "interval_seconds": 10},
  "contract_request": {
    "project_root": "/absolute/project",
    "scope_root": "/absolute/project/experiments/molecular_memory_v3_20261017",
    "environment_lock": "/absolute/scope/resources/environment.json",
    "environment_path": "/absolute/env",
    "code_paths": ["all pinned source paths, plus optional extra canonical components"],
    "objective": "Frozen controlled scalar-affinity benchmark with separate LM and numeric channels",
    "datasets": [{"name": "locked source", "path": "/absolute/locked/file", "sha256": "64-lowercase-hex"}],
    "models": [{"name": "frozen backbone", "path": "/absolute/backbone/weights", "sha256": "64-lowercase-hex"}],
    "resources": {"host": "qualified-host", "cpu_threads": 2,
                  "memory_limit_bytes": 34359738368, "gpu_memory_budget_bytes": 21474836480}
  }
}
```

上述是带占位符的结构示例，不是可执行正式plan。正式plan须包括mm_contract.create所需其他公共字段，例如numerical_policy、metric、metric_definition、checkpoint_selection，沿用父任务最终合法模板。queue不把未知`level`硬改为别的枚举。

`source_pins`最少覆盖scope/src下十一文件：mm_case.py、mm_queue.py、mm_contract.py、mm_supervise.py、mm_train.py、mm_model.py、mm_data.py、mm_extract.py、mm_evaluate.py、mm_worker_limit.py、mm_archive.py；所有这些路径也必须在contract_request.code_paths。可增加其他源。`asset_pins`最少包含实际训练config、evaluation_protocol、cache_root/manifest.json、view_root/manifest.json、environment_lock、standard_root/validate_reproduction.py、standard_root/task-contract.template.json、standard_root/result-record.template.json和每个ready evidence。完整cache payload闭合与底座实际weights由训练器及标准create再按原manifest逐文件检查，不能把只hash一个manifest叫作已检查其全部底层payload。

config须status=frozen、固定完整矩阵、training canonical SHA等于budget_signature。evaluation_protocol.training_budget须status=FROZEN并绑定相同training、training_budget_sha256和真实config SHA；正式评价器还独立核全部更严格schema、数据、训练量和选点。这层不会替父任务发明expected_training或绕过其未冻结状态。

## 单case流程与显存上限

1. 新建独占case目录，核所有pins及精确GPU assignment。
2. `python -m src.mm_worker_limit --gpu-memory-budget-bytes PLAN_BUDGET --receipt case/fit_gpu_limit.json --module src.mm_train -- [公共参数] --phase fit --output case/fit`。
3. 返回0且实际allocator收据匹配后，以同config/scenario/variant/seed、`--checkpoint case/fit/selected.pt`运行phase=evaluate，输出case/evaluate及独立evaluate_gpu_limit.json。
4. 调用`mm_evaluate.evaluate_run(case/evaluate/run_manifest.json, formal_evaluation_protocol, data_root)`。只接受complete、formal_eligible=true及精确三元组，原样保存case/result.json。数值/LM/C1指标不影响启动顺序和成功判定的方向。

每个phase的cap收据须显示同预算、同visible GPU、module=src.mm_train，peak_allocated_bytes及peak_reserved_bytes为非负整数且不超预算。此上限是**PyTorch allocator** cap，不包括CUDA context或其他库开销；实际显卡进程资源仍以父监督器probe/记录为准。case_receipt记录两个cap文件SHA与result SHA，并明确scientific_acceptance=controller_pending。

## 标准合同、状态和恢复

每个attempt单独新目录：`cases/CASE_ID/attempt_0001`。合同request、动态contract/contract.json、资源busy证据、supervision收据、case模型/预测/评分均不覆写；资源重探产生下一个attempt。request逐case填run_id、seed、physical GPU、命令、152/442固定test分母和当前wall timeout，其他科学公共字段保持模板。

queue先检查run返回0、supervisor receipt succeeded/exit0/remaining_group_members=[]/contract SHA一致，以及result complete/formal_eligible/三元组/config SHA/evaluation protocol SHA一致。随后保存独占`attempt/completion_spec.json`，调用`mm_archive.archive(contract_path, supervision_dir, [case_root], standard_root, attempt/archive, completion_spec=spec)`生成独立标准result-record及终态验收。

completion spec只断言原先注册的完整性：result.json的status=complete、formal_eligible严格布尔true、n_expected_test=152或442、LM.n_total等于完整test分母；numeric.n_total/n_finite/full_population.n均等于该分母，n_failed=0、coverage严格float1.0、status=defined。required target `registered_endpoint`的observed denominator只指向`/n_expected_test`，不使用fit/val样本数。LM parse失败仍按原全分母与null规则保留，不新增“parse必须全有效”条件，不以分数正负作为验收标准。

只有archive_receipt status=accepted、契约SHA/record SHA/validation SHA绑定一致，`terminal_validation.json`的valid和valid_for_acceptance均严格true、files_skipped为空，且result-record review=accepted/lifecycle=succeeded并绑定相同completion spec时，queue才追加succeeded。原worker结果自称成功不能覆盖canonical terminal false；拒收、partial archive、预测、模型和所有标准收据原样保留，case记failed。ledger成功事件另外绑定archive_receipt、terminal_validation和result-record三份SHA；resume只读原archive完成验收，不重复生成或覆写归档，也不重训。

queue_events.jsonl保存全36初始not_started，再追加running/succeeded/failed或资源忙not_started；每行绑定plan SHA、连续sequence、前一事件SHA和当前事件SHA，flush+fsync。来源/config/cache/view/标准文件pin不符停止公共启动；合同admission异常也停止公共队列。一般独立训练/评价/验收失败保留为failed并继续其他固定案例，绝不按模型分数自动重试。

资源忙只按同一阈值、已冻结max_attempts（1—100）和interval_seconds（0—60）有限重探；耗尽的case是not_started/exhausted，不能算训练失败或从36分母中删除。本实现不自动重启耗尽资源重探；未耗尽的普通not_started可在相同plan下resume继续。固定截止不会因resume延长。

resume跳过所有succeeded/failed，不重训；running只有在原PGID已无活成员后，依据完整原receipt+result恢复succeeded，否则记interrupted_without_accepted_terminal_receipt为failed。若仍有成员，拒绝开第二个worker。snapshot plan不一致或ledger哈希链破坏会直接停止，保留原物，不自行修补历史。

SIGINT/SIGTERM设置halt并停止领取新任务；cleanup watcher只检查active attempt。必须同时满足started.json contract SHA、PID=PGID、不是本controller组、当前/proc精确argv及当前PGID均与该contract一致，才调用父监督器terminate_group。拒绝碰其他组/复用PID；leader已退出则由仍运行的supervisor原finally清理后代。信号不会让普通fit/eval子进程逃出监督PGID。

## 当前验证范围

21项stdlib CPU测试通过：固定36次序、每GPU独占/总资源账本、busy保留attempt且不重复训练、有限busy耗尽、独立失败分母与终态resume、截止不启动、draft/busy/occupied/重复GPU拒绝、目录不覆写、plan漂移与ledger篡改拒绝、公共pin漂移停止、fit/eval命令与PGID继承、失败fit不评价、已接受收据恢复、活PGID拒绝、真实自有CPU子进程的精确签名清理资格、archive自称accepted不能覆盖terminal false、numeric覆盖不完整拒绝归档、watchdog成功/异常timer清理与handler还原、watchdog拒绝外部组，以及真实独立CPU case组连同自身sleep子进程被独立timer清理。所有训练、评价、合同、archive validator和GPU资源调用均为明确的mock fixtures；真实archiver的只读JSON pointer完整性检查得到复用，但这些mock标准收据不是canonical验收证据。真实子进程仅为自有CPU sleep与该watchdog小脚本；未加载真实模型、调用CUDA、执行A1fit或启动queue正式plan。真实资源资格、正式config/evalprotocol冻结、动态标准合同及完整标准链仍待父任务实际canary验收。
